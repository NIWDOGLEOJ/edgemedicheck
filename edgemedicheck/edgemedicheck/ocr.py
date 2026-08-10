"""
Algorithm 2 (first half) -- OCR-based field extraction.

Runs Tesseract over the preprocessed image and pulls out the four fields the
paper needs: batch number, manufacturing date, expiry date, and manufacturer.

Design notes
------------
* Two PSM passes. PSM 6 (uniform block) handles flat cartons; PSM 11 (sparse
  text) handles blister strips where fields are scattered around the foil. We
  keep whichever pass yields more required fields, breaking ties on confidence.

* Per-field confidence, not per-character. The paper argues correctly that a
  single wrong digit in an expiry date changes the safety decision, so a field
  is only trusted when *every* word composing it clears the threshold.

* A low-confidence read is never treated as safe. It downgrades the OCR status
  to "uncertain", which Algorithm 3 turns into a yellow "manual verification"
  verdict rather than a green pass.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import CONFIG, OCRConfig
from .dateparse import ParsedDate, parse_date_string

log = logging.getLogger(__name__)

try:
    import pytesseract
    from pytesseract import Output

    TESSERACT_AVAILABLE = True
except ImportError:  # pragma: no cover
    pytesseract = None  # type: ignore
    Output = None  # type: ignore
    TESSERACT_AVAILABLE = False


# --------------------------------------------------------------------------
# Field label vocabulary
# --------------------------------------------------------------------------

BATCH_LABELS = r"(?:B\.?\s*NO|BATCH\s*NO|BATCH|LOT\s*NO|LOT|B/N|BN)"
MFG_LABELS = r"(?:MFG\.?\s*DATE|MFG\.?|MFD\.?|MANUFACTURED|M\.?\s*DATE|MD)"
EXP_LABELS = r"(?:EXP\.?\s*DATE|EXPIRY\s*DATE|EXPIRY|EXP\.?|USE\s*BEFORE|"
EXP_LABELS += r"BEST\s*BEFORE|E\.?\s*DATE|ED)"

# Batch codes are alphanumeric, typically 4-12 chars, and nearly always
# contain at least one digit. Requiring a digit rejects stray words.
BATCH_VALUE = r"([A-Z0-9][A-Z0-9\-/]{2,15})"

MFR_SUFFIXES = (
    "pharmaceuticals", "pharmaceutical", "pharma", "laboratories", "labs",
    "healthcare", "health care", "lifesciences", "life sciences", "biotech",
    "remedies", "drugs", "formulations", "industries", "ltd", "limited",
    "pvt", "private", "inc", "llp",
)


@dataclass
class OCRField:
    """One extracted field with the evidence behind it."""

    name: str
    value: str | None = None
    confidence: float = 0.0
    raw_context: str = ""
    parsed_date: ParsedDate | None = None

    @property
    def found(self) -> bool:
        return self.value is not None

    def is_confident(self, threshold: float) -> bool:
        return self.found and self.confidence >= threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "confidence": round(self.confidence, 1),
            "parsed_date": self.parsed_date.iso if self.parsed_date else None,
            "raw_context": self.raw_context[:120],
        }


@dataclass
class OCRResult:
    """Output of Algorithm 2's extraction half."""

    raw_text: str = ""
    mean_confidence: float = 0.0
    fields: dict[str, OCRField] = field(default_factory=dict)
    psm_used: int = 0
    status: str = "unknown"  # "complete" | "uncertain" | "failed"
    notes: list[str] = field(default_factory=list)

    def get(self, name: str) -> OCRField:
        return self.fields.get(name, OCRField(name))

    @property
    def batch_number(self) -> str | None:
        return self.get("batch_number").value

    @property
    def exp_date(self) -> ParsedDate | None:
        return self.get("exp_date").parsed_date

    @property
    def mfg_date(self) -> ParsedDate | None:
        return self.get("mfg_date").parsed_date

    @property
    def manufacturer(self) -> str | None:
        return self.get("manufacturer").value

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "mean_confidence": round(self.mean_confidence, 1),
            "psm_used": self.psm_used,
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
            "notes": self.notes,
            "raw_text": self.raw_text,
        }


# --------------------------------------------------------------------------
# Tesseract invocation
# --------------------------------------------------------------------------


def _configure_tesseract(cfg: OCRConfig) -> None:
    if cfg.tesseract_cmd and pytesseract is not None:
        pytesseract.pytesseract.tesseract_cmd = cfg.tesseract_cmd


def _run_tesseract(
    image: np.ndarray, psm: int, cfg: OCRConfig
) -> tuple[str, list[dict], float]:
    """Return (text, word records, mean confidence) for one PSM setting."""
    if not TESSERACT_AVAILABLE:
        raise RuntimeError(
            "pytesseract is not installed. Install it and the tesseract-ocr "
            "binary to run OCR extraction."
        )

    config = f"--oem {cfg.oem} --psm {psm}"
    data = pytesseract.image_to_data(
        image, lang=cfg.lang, config=config, output_type=Output.DICT
    )

    words: list[dict] = []
    confidences: list[float] = []
    for i in range(len(data["text"])):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < 0:
            continue
        words.append(
            {
                "text": text,
                "conf": conf,
                "line": data["line_num"][i],
                "block": data["block_num"][i],
                "left": data["left"][i],
                "top": data["top"][i],
                "width": data["width"][i],
                "height": data["height"][i],
            }
        )
        confidences.append(conf)

    lines: dict[tuple[int, int], list[str]] = {}
    for w in words:
        lines.setdefault((w["block"], w["line"]), []).append(w["text"])
    text = "\n".join(" ".join(v) for v in lines.values())

    mean_conf = float(np.mean(confidences)) if confidences else 0.0
    return text, words, mean_conf


def _confidence_for_span(words: list[dict], span_text: str) -> float:
    """Minimum confidence across the words that make up an extracted value.

    Minimum, not mean: one badly-read character invalidates the whole field.
    """
    tokens = [t for t in re.split(r"[\s/\-.:]+", span_text.upper()) if t]
    if not tokens:
        return 0.0

    confs: list[float] = []
    for token in tokens:
        matches = [w["conf"] for w in words if token in w["text"].upper()]
        if matches:
            confs.append(max(matches))
    if not confs:
        return 0.0
    return float(min(confs))


# --------------------------------------------------------------------------
# Field extraction
# --------------------------------------------------------------------------


def _search_labelled(
    text: str, labels: str, value_pattern: str
) -> tuple[str, str] | None:
    """Find `LABEL <separator> VALUE`. Returns (value, surrounding context)."""
    pattern = re.compile(
        rf"{labels}\s*[:.\-]?\s*{value_pattern}", re.IGNORECASE
    )
    m = pattern.search(text)
    if not m:
        return None
    start = max(0, m.start() - 20)
    end = min(len(text), m.end() + 20)
    return m.group(1).strip(), text[start:end].replace("\n", " ")


def extract_batch(text: str, words: list[dict]) -> OCRField:
    f = OCRField("batch_number")
    hit = _search_labelled(text, BATCH_LABELS, BATCH_VALUE)
    if hit:
        value, context = hit
        value = value.upper().strip(" .:-")
        # Reject label words that leaked into the capture group and codes with
        # no digit at all -- real batch numbers always carry digits.
        if any(ch.isdigit() for ch in value) and value not in {
            "NO", "DATE", "MFG", "EXP"
        }:
            f.value = value
            f.raw_context = context
            f.confidence = _confidence_for_span(words, value)
    return f


def extract_date_field(
    text: str, words: list[dict], labels: str, name: str, kind: str
) -> OCRField:
    f = OCRField(name)

    # Look at the text immediately following the label, where the date lives.
    label_re = re.compile(labels, re.IGNORECASE)
    for m in label_re.finditer(text):
        window = text[m.end(): m.end() + 32]
        parsed = parse_date_string(window, kind)
        if parsed:
            f.value = parsed.raw
            f.parsed_date = parsed
            f.raw_context = text[
                max(0, m.start() - 10): m.end() + 32
            ].replace("\n", " ")
            f.confidence = _confidence_for_span(words, parsed.raw)
            return f

    return f


def _fallback_unlabelled_dates(text: str, words: list[dict], result: OCRResult) -> None:
    """Recover dates when the label word itself failed to OCR.

    Many strips print `MFG 03/2025  EXP 08/2027` in tiny type; the label often
    reads worse than the digits. When we find exactly two bare dates and no
    labelled ones, the earlier is manufacturing and the later is expiry.
    """
    have_mfg = result.get("mfg_date").found
    have_exp = result.get("exp_date").found
    if have_mfg and have_exp:
        return

    candidates: list[ParsedDate] = []
    seen: set[str] = set()
    for m in re.finditer(r"\b\d{1,2}\s*[/\-.]\s*\d{2,4}\b", text):
        parsed = parse_date_string(m.group(0), "unknown")
        if parsed and parsed.iso not in seen:
            seen.add(parsed.iso)
            candidates.append(parsed)

    if len(candidates) < 2:
        # A single bare date on a pack is conventionally the expiry.
        if len(candidates) == 1 and not have_exp:
            p = candidates[0]
            exp = ParsedDate(p.year, p.month, p.day, p.raw, "exp")
            result.fields["exp_date"] = OCRField(
                "exp_date",
                value=exp.raw,
                confidence=_confidence_for_span(words, exp.raw),
                raw_context="(unlabelled date, assumed expiry)",
                parsed_date=exp,
            )
            result.notes.append(
                "Expiry label not read; assumed the only date found is expiry."
            )
        return

    candidates.sort(key=lambda p: p.effective)
    earliest, latest = candidates[0], candidates[-1]

    if not have_mfg:
        mfg = ParsedDate(
            earliest.year, earliest.month, earliest.day, earliest.raw, "mfg"
        )
        result.fields["mfg_date"] = OCRField(
            "mfg_date",
            value=mfg.raw,
            confidence=_confidence_for_span(words, mfg.raw),
            raw_context="(unlabelled date, assumed manufacturing)",
            parsed_date=mfg,
        )
    if not have_exp:
        exp = ParsedDate(latest.year, latest.month, latest.day, latest.raw, "exp")
        result.fields["exp_date"] = OCRField(
            "exp_date",
            value=exp.raw,
            confidence=_confidence_for_span(words, exp.raw),
            raw_context="(unlabelled date, assumed expiry)",
            parsed_date=exp,
        )
    result.notes.append(
        "Date labels not read; inferred MFG/EXP from chronological order."
    )


def extract_manufacturer(text: str, words: list[dict]) -> OCRField:
    """Find the manufacturer line by its corporate suffix."""
    f = OCRField("manufacturer")
    best: tuple[str, int] | None = None

    for line in text.split("\n"):
        clean = line.strip()
        if len(clean) < 4:
            continue
        lower = clean.lower()
        for suffix in MFR_SUFFIXES:
            if suffix in lower:
                # Trim anything after the corporate suffix.
                idx = lower.index(suffix) + len(suffix)
                candidate = clean[:idx].strip(" .,:-")
                score = len(suffix)
                if best is None or score > best[1]:
                    best = (candidate, score)
                break

    if best:
        f.value = best[0]
        f.raw_context = best[0]
        f.confidence = _confidence_for_span(words, best[0])
    return f


def extract_product_name(text: str, words: list[dict]) -> OCRField:
    """Best-effort product name: the longest prominent all-caps token run.

    Used only to enrich the database query, never for the safety decision.
    """
    f = OCRField("product_name")
    if not words:
        return f

    heights = [w["height"] for w in words]
    if not heights:
        return f
    cutoff = float(np.percentile(heights, 75))

    large = [
        w for w in words
        if w["height"] >= cutoff
        and re.fullmatch(r"[A-Za-z][A-Za-z\-]{2,}", w["text"])
        and w["text"].lower() not in {"tablets", "capsules", "tablet", "capsule"}
    ]
    if not large:
        return f

    best = max(large, key=lambda w: (w["height"], w["conf"]))
    f.value = best["text"].upper()
    f.confidence = float(best["conf"])
    f.raw_context = best["text"]
    return f


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------


def _extract_all(text: str, words: list[dict]) -> dict[str, OCRField]:
    fields = {
        "batch_number": extract_batch(text, words),
        "mfg_date": extract_date_field(
            text, words, MFG_LABELS, "mfg_date", "mfg"
        ),
        "exp_date": extract_date_field(
            text, words, EXP_LABELS, "exp_date", "exp"
        ),
        "manufacturer": extract_manufacturer(text, words),
        "product_name": extract_product_name(text, words),
    }
    return fields


def _score_pass(fields: dict[str, OCRField], cfg: OCRConfig) -> tuple[int, float]:
    """Rank a PSM pass by required fields found, then by mean confidence."""
    found = sum(1 for name in cfg.required_fields if fields[name].found)
    confs = [f.confidence for f in fields.values() if f.found]
    return found, float(np.mean(confs)) if confs else 0.0


def run_ocr(image: np.ndarray, cfg: OCRConfig | None = None) -> OCRResult:
    """Run Algorithm 2's extraction half on a preprocessed OCR image."""
    cfg = cfg or CONFIG.ocr
    _configure_tesseract(cfg)

    result = OCRResult()

    if not TESSERACT_AVAILABLE:
        result.status = "failed"
        result.notes.append("pytesseract not installed")
        return result

    best: tuple[tuple[int, float], str, list[dict], float, int] | None = None

    for psm in (cfg.psm_primary, cfg.psm_fallback):
        try:
            text, words, mean_conf = _run_tesseract(image, psm, cfg)
        except Exception as exc:
            log.warning("Tesseract PSM %s failed: %s", psm, exc)
            continue

        fields = _extract_all(text, words)
        score = _score_pass(fields, cfg)
        if best is None or score > best[0]:
            best = (score, text, words, mean_conf, psm)

    if best is None:
        result.status = "failed"
        result.notes.append("All Tesseract passes failed")
        return result

    _, text, words, mean_conf, psm = best
    result.raw_text = text
    result.mean_confidence = mean_conf
    result.psm_used = psm
    result.fields = _extract_all(text, words)

    _fallback_unlabelled_dates(text, words, result)

    # Algorithm 2, steps 5-6: missing or low-confidence required fields make
    # the read uncertain. Uncertain is never treated as safe downstream.
    missing = [n for n in cfg.required_fields if not result.fields[n].found]
    low_conf = [
        n
        for n in cfg.required_fields
        if result.fields[n].found
        and result.fields[n].confidence < cfg.min_field_confidence
    ]

    if missing:
        result.status = "uncertain"
        result.notes.append(f"Missing required field(s): {', '.join(missing)}")
    elif low_conf:
        result.status = "uncertain"
        result.notes.append(
            f"Low OCR confidence on: {', '.join(low_conf)} "
            f"(threshold {cfg.min_field_confidence:.0f})"
        )
    else:
        result.status = "complete"

    return result
