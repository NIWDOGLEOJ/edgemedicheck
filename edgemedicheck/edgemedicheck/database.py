"""
Local SQLite verification database (Table V of the paper).

This is the offline batch-reference store. It answers one question: does the
batch we just read off the package correspond to a real, currently valid
batch record?

Important scoping note carried over from the paper: this database is seeded
from pharmacy stock records, purchase invoices, distributor-supplied batch
lists, and public regulatory alerts (CDSCO drug alerts / NSQ lists). It is
*not* a complete national batch registry, so "unknown" is a legitimate and
common outcome. Unknown must therefore map to a caution verdict, never to a
counterfeit accusation.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator

from .config import CONFIG
from .dateparse import ParsedDate, dates_agree, parse_db_date

log = logging.getLogger(__name__)


# Status values a product record can carry (Table V, `status` field).
VALID = "valid"
EXPIRED = "expired"
RECALLED = "recalled"
NSQ = "nsq"  # Not of Standard Quality (CDSCO terminology)
SPURIOUS = "spurious"
UNKNOWN = "unknown"

# Result of a lookup, which is a different thing from the stored status.
MATCH_VALID = "valid"
MATCH_EXPIRED = "expired"
MATCH_MISMATCH = "mismatch"
# Weaker than MATCH_MISMATCH: the batch and dates line up, but a free-text
# field disagrees. Manufacturer names sit in stylised logo type that OCR reads
# badly, so this must not by itself condemn a pack.
MATCH_NAME_MISMATCH = "name_mismatch"
MATCH_UNSAFE = "unsafe"
MATCH_UNKNOWN = "unknown"


SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    product_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    product_name  TEXT    NOT NULL,
    manufacturer  TEXT    NOT NULL,
    batch_number  TEXT    NOT NULL,
    mfg_date      TEXT,
    exp_date      TEXT    NOT NULL,
    source_type   TEXT    NOT NULL DEFAULT 'pharmacy',
    status        TEXT    NOT NULL DEFAULT 'valid',
    notes         TEXT,
    last_updated  TEXT    NOT NULL,
    UNIQUE (batch_number, product_name)
);

CREATE INDEX IF NOT EXISTS idx_products_batch
    ON products (batch_number);
CREATE INDEX IF NOT EXISTS idx_products_status
    ON products (status);

CREATE TABLE IF NOT EXISTS scan_log (
    scan_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp      TEXT NOT NULL,
    verdict        TEXT NOT NULL,
    reason_code    TEXT NOT NULL,
    reason_text    TEXT,
    batch_number   TEXT,
    product_name   TEXT,
    manufacturer   TEXT,
    exp_date       TEXT,
    ocr_status     TEXT,
    ocr_confidence REAL,
    db_status      TEXT,
    cnn_score      REAL,
    cnn_backend    TEXT,
    code_found     INTEGER,
    code_batch     TEXT,
    code_exp_date  TEXT,
    code_symbology TEXT,
    crosscheck     TEXT,
    latency_ms     REAL,
    image_path     TEXT,
    details_json   TEXT
);

CREATE INDEX IF NOT EXISTS idx_scan_log_time
    ON scan_log (timestamp DESC);

-- Pharmacist corrections.
--
-- Every time a pharmacist marks a verdict wrong, the scan becomes a labelled
-- training example captured under real counter conditions -- the exact data
-- the CNN needs and the hardest thing to collect deliberately. The scanner
-- therefore builds its own dataset as a by-product of ordinary use.
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id        INTEGER,
    timestamp      TEXT NOT NULL,
    system_verdict TEXT NOT NULL,
    correct_verdict TEXT NOT NULL,
    correct_label  TEXT,          -- "genuine" | "suspicious" | "expired"
    reason_code    TEXT,
    batch_number   TEXT,
    comment        TEXT,
    reported_by    TEXT,
    image_path     TEXT,
    exported       INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (scan_id) REFERENCES scan_log (scan_id)
);

CREATE INDEX IF NOT EXISTS idx_feedback_exported
    ON feedback (exported);
"""

# Columns added after the first release. SQLite has no "ADD COLUMN IF NOT
# EXISTS", so migrations are applied by inspecting the existing schema. This
# keeps a deployed scanner's scan history intact across upgrades rather than
# forcing the audit log to be discarded.
MIGRATIONS: list[tuple[str, str, str]] = [
    ("scan_log", "code_found", "INTEGER"),
    ("scan_log", "code_batch", "TEXT"),
    ("scan_log", "code_exp_date", "TEXT"),
    ("scan_log", "code_symbology", "TEXT"),
    ("scan_log", "crosscheck", "TEXT"),
]


def _apply_migrations(conn: sqlite3.Connection) -> None:
    for table, column, coltype in MIGRATIONS:
        existing = {
            row["name"] for row in conn.execute(f"PRAGMA table_info({table})")
        }
        if not existing:
            continue  # table not created yet; the schema script handles it
        if column not in existing:
            log.info("Migrating %s: adding column %s", table, column)
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


@dataclass
class ProductRecord:
    product_id: int
    product_name: str
    manufacturer: str
    batch_number: str
    mfg_date: str | None
    exp_date: str
    source_type: str
    status: str
    notes: str | None
    last_updated: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LookupResult:
    """Outcome of checking OCR-extracted fields against the local database."""

    status: str  # MATCH_* constant
    record: ProductRecord | None = None
    reason: str = ""
    candidates: int = 0

    @property
    def is_unsafe(self) -> bool:
        return self.status in (MATCH_UNSAFE, MATCH_EXPIRED, MATCH_MISMATCH)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "candidates": self.candidates,
            "record": self.record.to_dict() if self.record else None,
        }


# --------------------------------------------------------------------------
# Connection handling
# --------------------------------------------------------------------------


@contextmanager
def connect(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    path = Path(db_path or CONFIG.db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    # WAL keeps the UI responsive while a scan writes its audit row, but it
    # needs shared-memory support. Network shares, some SD-card mounts, and
    # FUSE filesystems reject it, so fall back to the portable journal mode.
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("SELECT 1").fetchone()
    except sqlite3.OperationalError:
        conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path | str | None = None) -> None:
    """Create the schema if it does not already exist, then migrate it."""
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        _apply_migrations(conn)
    log.info("Database initialised at %s", db_path or CONFIG.db_path)


# --------------------------------------------------------------------------
# Writes
# --------------------------------------------------------------------------


def upsert_product(
    product_name: str,
    manufacturer: str,
    batch_number: str,
    exp_date: str,
    mfg_date: str | None = None,
    source_type: str = "pharmacy",
    status: str = VALID,
    notes: str | None = None,
    db_path: Path | str | None = None,
) -> int:
    """Insert or update one batch record. Returns the product_id."""
    now = datetime.now().isoformat(timespec="seconds")
    with connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO products (product_name, manufacturer, batch_number,
                                  mfg_date, exp_date, source_type, status,
                                  notes, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (batch_number, product_name) DO UPDATE SET
                manufacturer = excluded.manufacturer,
                mfg_date     = excluded.mfg_date,
                exp_date     = excluded.exp_date,
                source_type  = excluded.source_type,
                status       = excluded.status,
                notes        = excluded.notes,
                last_updated = excluded.last_updated
            """,
            (
                product_name.strip(),
                manufacturer.strip(),
                batch_number.strip().upper(),
                mfg_date,
                exp_date,
                source_type,
                status,
                notes,
                now,
            ),
        )
        if cur.lastrowid:
            return int(cur.lastrowid)
        row = conn.execute(
            "SELECT product_id FROM products WHERE batch_number=? AND product_name=?",
            (batch_number.strip().upper(), product_name.strip()),
        ).fetchone()
        return int(row["product_id"]) if row else -1


def bulk_import(records: list[dict], db_path: Path | str | None = None) -> int:
    """Import many records, e.g. from a distributor CSV. Returns count."""
    count = 0
    for r in records:
        upsert_product(db_path=db_path, **r)
        count += 1
    return count


def mark_status(
    batch_number: str, status: str, notes: str | None = None,
    db_path: Path | str | None = None
) -> int:
    """Apply a regulatory alert (recall / NSQ) to an existing batch."""
    now = datetime.now().isoformat(timespec="seconds")
    with connect(db_path) as conn:
        cur = conn.execute(
            "UPDATE products SET status=?, notes=?, last_updated=? "
            "WHERE batch_number=?",
            (status, notes, now, batch_number.strip().upper()),
        )
        return cur.rowcount


# --------------------------------------------------------------------------
# Reads
# --------------------------------------------------------------------------


def _row_to_record(row: sqlite3.Row) -> ProductRecord:
    return ProductRecord(
        product_id=row["product_id"],
        product_name=row["product_name"],
        manufacturer=row["manufacturer"],
        batch_number=row["batch_number"],
        mfg_date=row["mfg_date"],
        exp_date=row["exp_date"],
        source_type=row["source_type"],
        status=row["status"],
        notes=row["notes"],
        last_updated=row["last_updated"],
    )


def find_by_batch(
    batch_number: str, db_path: Path | str | None = None
) -> list[ProductRecord]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM products WHERE batch_number = ?",
            (batch_number.strip().upper(),),
        ).fetchall()
    return [_row_to_record(r) for r in rows]


def all_products(db_path: Path | str | None = None) -> list[ProductRecord]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM products ORDER BY product_name, batch_number"
        ).fetchall()
    return [_row_to_record(r) for r in rows]


def count_products(db_path: Path | str | None = None) -> int:
    with connect(db_path) as conn:
        return int(conn.execute("SELECT COUNT(*) c FROM products").fetchone()["c"])


# --------------------------------------------------------------------------
# Algorithm 2 (second half) -- verification
# --------------------------------------------------------------------------

# Corporate boilerplate carries no identifying signal, so it is stripped before
# comparing two manufacturer strings.
_MFR_STOPWORDS = {
    "ltd", "limited", "pvt", "private", "inc", "llp", "co", "company",
    "pharmaceuticals", "pharmaceutical", "pharma", "laboratories", "labs",
    "lab", "healthcare", "health", "care", "lifesciences", "sciences",
    "industries", "india", "mfd", "by", "the", "and",
}


def _mfr_tokens(name: str) -> set[str]:
    import re as _re

    tokens = _re.findall(r"[a-z]+", name.lower())
    return {t for t in tokens if len(t) > 2 and t not in _MFR_STOPWORDS}


def _manufacturers_agree(printed: str, recorded: str) -> bool:
    """Do two manufacturer strings plausibly name the same company?

    The printed value often arrives with neighbouring label lines merged into
    it, so we ask whether the recorded company's distinctive tokens appear in
    the printed text -- not whether the two strings are equal.
    """
    a, b = _mfr_tokens(printed), _mfr_tokens(recorded)
    if not a or not b:
        return True  # nothing distinctive to compare; do not raise a flag
    return bool(a & b)


def verify(
    batch_number: str | None,
    exp_date: ParsedDate | None = None,
    product_name: str | None = None,
    manufacturer: str | None = None,
    today: date | None = None,
    tolerance_days: int | None = None,
    db_path: Path | str | None = None,
) -> LookupResult:
    """Check extracted label fields against the local batch database.

    Resolution order, most severe first:

      1. No batch read at all              -> unknown
      2. Batch not in database             -> unknown  (NOT counterfeit)
      3. Record carries a regulatory flag  -> unsafe
      4. Record's own date is past         -> expired
      5. Label expiry disagrees with record -> mismatch (tampered label)
      6. Label date is past, record agrees -> expired
      7. Everything lines up               -> valid

    Steps 4-6 are ordered so that the record is consulted before the label.
    A relabelled carton is only detectable as a disagreement between the
    two, so the disagreement has to be tested before the printed date is
    judged in isolation.
    """
    today = today or date.today()
    tolerance = (
        tolerance_days
        if tolerance_days is not None
        else CONFIG.fusion.expiry_mismatch_tolerance_days
    )

    if not batch_number:
        return LookupResult(
            MATCH_UNKNOWN, reason="No batch number could be read from the package."
        )

    candidates = find_by_batch(batch_number, db_path)
    if not candidates:
        return LookupResult(
            MATCH_UNKNOWN,
            reason=(
                f"Batch {batch_number} is not in the local database. "
                "The record may be missing rather than the pack being fake."
            ),
        )

    # Narrow by product name when OCR gave us one we can trust.
    record = candidates[0]
    if product_name and len(candidates) > 1:
        pn = product_name.strip().lower()
        for c in candidates:
            if pn in c.product_name.lower() or c.product_name.lower() in pn:
                record = c
                break

    # 3. Regulatory flags override everything else.
    if record.status in (RECALLED, NSQ, SPURIOUS):
        label = {
            RECALLED: "recalled",
            NSQ: "flagged Not of Standard Quality",
            SPURIOUS: "flagged spurious",
        }[record.status]
        return LookupResult(
            MATCH_UNSAFE,
            record=record,
            candidates=len(candidates),
            reason=f"Batch {record.batch_number} is {label} in the local records.",
        )

    db_exp = parse_db_date(record.exp_date, "exp")

    # 4. Expiry as the *database* records it. The record is the trusted
    # source here: it comes from pharmacy stock, invoices and regulatory
    # alerts, not from a camera pointed at a carton that may be relabelled.
    if record.status == EXPIRED:
        return LookupResult(
            MATCH_EXPIRED,
            record=record,
            candidates=len(candidates),
            reason=f"Batch {record.batch_number} is marked expired in the database.",
        )
    if db_exp and db_exp.effective < today:
        return LookupResult(
            MATCH_EXPIRED,
            record=record,
            candidates=len(candidates),
            reason=(
                f"Database expiry {db_exp.iso} has passed "
                f"(today is {today.isoformat()})."
            ),
        )

    # 5. Label vs record disagreement -- the tampered-expiry case.
    #
    # This has to be tested before the printed date is judged on its own.
    # The record has already been shown to be in date, so any disagreement
    # means the carton does not say what the pharmacy's own records say --
    # which is the finding, whichever direction the printed date points.
    # Checking "printed date is past" first would swallow every relabelling
    # that reads as an earlier date and report it as an ordinary expiry,
    # hiding the mismatch and blaming a database that said the batch was
    # fine.
    if exp_date and db_exp and not dates_agree(exp_date, db_exp, tolerance):
        return LookupResult(
            MATCH_MISMATCH,
            record=record,
            candidates=len(candidates),
            reason=(
                f"Printed expiry {exp_date.iso} does not match the recorded "
                f"expiry {db_exp.iso} for batch {record.batch_number}."
            ),
        )

    # 6. Printed date on its own. Reached when the record carries no expiry
    # to compare against, or when the two agree.
    if exp_date and exp_date.effective < today:
        return LookupResult(
            MATCH_EXPIRED,
            record=record,
            candidates=len(candidates),
            reason=f"Printed expiry {exp_date.iso} has passed.",
        )

    # Manufacturer disagreement is reported but does not by itself condemn the
    # pack: OCR on stylised logo type is unreliable, and neighbouring lines
    # frequently merge into the captured name. Compared on significant tokens
    # rather than raw string equality so that "Cipla Pharmaceuticals Ltd" and
    # "CIPLA PHARMA LTD." are treated as agreeing.
    if manufacturer and record.manufacturer:
        if not _manufacturers_agree(manufacturer, record.manufacturer):
            return LookupResult(
                MATCH_NAME_MISMATCH,
                record=record,
                candidates=len(candidates),
                reason=(
                    f"Printed manufacturer '{manufacturer.strip()}' does not "
                    f"clearly match the recorded manufacturer "
                    f"'{record.manufacturer}'. OCR on logo text is unreliable, "
                    "so confirm this by eye."
                ),
            )

    return LookupResult(
        MATCH_VALID,
        record=record,
        candidates=len(candidates),
        reason=f"Batch {record.batch_number} matches a valid local record.",
    )


# --------------------------------------------------------------------------
# Scan audit log
# --------------------------------------------------------------------------


def log_scan(entry: dict, db_path: Path | str | None = None) -> int:
    columns = (
        "timestamp", "verdict", "reason_code", "reason_text", "batch_number",
        "product_name", "manufacturer", "exp_date", "ocr_status",
        "ocr_confidence", "db_status", "cnn_score", "cnn_backend",
        "code_found", "code_batch", "code_exp_date", "code_symbology",
        "crosscheck", "latency_ms", "image_path", "details_json",
    )
    values = [entry.get(c) for c in columns]
    placeholders = ", ".join("?" * len(columns))
    with connect(db_path) as conn:
        cur = conn.execute(
            f"INSERT INTO scan_log ({', '.join(columns)}) VALUES ({placeholders})",
            values,
        )
        return int(cur.lastrowid or -1)


def recent_scans(limit: int = 25, db_path: Path | str | None = None) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM scan_log ORDER BY scan_id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------
# Pharmacist feedback
# --------------------------------------------------------------------------

# What the pharmacist says the pack actually was. Kept separate from the
# verdict, because "the verdict was wrong" and "this pack is counterfeit" are
# different claims and the training pipeline needs the latter.
LABEL_GENUINE = "genuine"
LABEL_SUSPICIOUS = "suspicious"
LABEL_EXPIRED = "expired"
FEEDBACK_LABELS = (LABEL_GENUINE, LABEL_SUSPICIOUS, LABEL_EXPIRED)


def record_feedback(
    scan_id: int | None,
    system_verdict: str,
    correct_verdict: str,
    correct_label: str | None = None,
    reason_code: str | None = None,
    batch_number: str | None = None,
    comment: str | None = None,
    reported_by: str | None = None,
    image_path: str | None = None,
    db_path: Path | str | None = None,
) -> int:
    """Store one pharmacist correction."""
    with connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO feedback (scan_id, timestamp, system_verdict,
                                  correct_verdict, correct_label, reason_code,
                                  batch_number, comment, reported_by, image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                scan_id,
                datetime.now().isoformat(timespec="seconds"),
                system_verdict,
                correct_verdict,
                correct_label,
                reason_code,
                batch_number,
                comment,
                reported_by,
                image_path,
            ),
        )
        return int(cur.lastrowid or -1)


def latest_scan(db_path: Path | str | None = None) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM scan_log ORDER BY scan_id DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def get_scan(scan_id: int, db_path: Path | str | None = None) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM scan_log WHERE scan_id = ?", (scan_id,)
        ).fetchone()
    return dict(row) if row else None


def pending_feedback(
    db_path: Path | str | None = None, include_exported: bool = False
) -> list[dict]:
    """Corrections not yet exported into a training dataset."""
    query = "SELECT * FROM feedback"
    if not include_exported:
        query += " WHERE exported = 0"
    query += " ORDER BY feedback_id"
    with connect(db_path) as conn:
        return [dict(r) for r in conn.execute(query)]


def mark_feedback_exported(
    ids: list[int], db_path: Path | str | None = None
) -> int:
    if not ids:
        return 0
    with connect(db_path) as conn:
        cur = conn.execute(
            f"UPDATE feedback SET exported = 1 WHERE feedback_id IN "
            f"({','.join('?' * len(ids))})",
            ids,
        )
        return cur.rowcount


def feedback_stats(db_path: Path | str | None = None) -> dict[str, Any]:
    """Correction counts, and the disagreement matrix for error analysis."""
    with connect(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) c FROM feedback").fetchone()["c"]
        pending = conn.execute(
            "SELECT COUNT(*) c FROM feedback WHERE exported = 0"
        ).fetchone()["c"]
        by_label = {
            r["correct_label"]: r["c"]
            for r in conn.execute(
                "SELECT correct_label, COUNT(*) c FROM feedback "
                "GROUP BY correct_label"
            )
            if r["correct_label"]
        }
        matrix = {
            f"{r['system_verdict']}->{r['correct_verdict']}": r["c"]
            for r in conn.execute(
                "SELECT system_verdict, correct_verdict, COUNT(*) c "
                "FROM feedback GROUP BY system_verdict, correct_verdict"
            )
        }
        scans = conn.execute("SELECT COUNT(*) c FROM scan_log").fetchone()["c"]

    return {
        "total": total,
        "pending_export": pending,
        "by_label": by_label,
        "disagreements": matrix,
        # Not an accuracy figure: pharmacists report errors far more often than
        # they confirm correct verdicts, so this is a lower bound on the error
        # rate, never an estimate of it.
        "reported_error_rate": round(total / scans, 4) if scans else None,
    }


def scan_stats(db_path: Path | str | None = None) -> dict[str, Any]:
    """Aggregate counts for the evaluation section and the UI header."""
    with connect(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) c FROM scan_log").fetchone()["c"]
        by_verdict = {
            r["verdict"]: r["c"]
            for r in conn.execute(
                "SELECT verdict, COUNT(*) c FROM scan_log GROUP BY verdict"
            )
        }
        latency = conn.execute(
            "SELECT AVG(latency_ms) a, MIN(latency_ms) mn, MAX(latency_ms) mx "
            "FROM scan_log WHERE latency_ms IS NOT NULL"
        ).fetchone()
    return {
        "total_scans": total,
        "by_verdict": by_verdict,
        "latency_ms": {
            "mean": round(latency["a"], 1) if latency["a"] else None,
            "min": round(latency["mn"], 1) if latency["mn"] else None,
            "max": round(latency["mx"], 1) if latency["mx"] else None,
        },
    }
