"""
Barcode and GS1 decoding -- the fourth evidence stream.

Why this matters more than better OCR
-------------------------------------
Indian pharmaceutical packaging increasingly carries a 2D code (QR or GS1
DataMatrix) that encodes the batch number, expiry date and a serial number as
machine-readable data. Reading it is not just a more reliable way to obtain the
same fields -- it lets the scanner *cross-check two independent renderings of
the same facts*:

    printed text  <->  encoded data

Agreement is strong evidence the label is intact. Disagreement is direct
tamper evidence: a relabelled pack whose printed expiry has been altered will
still carry the original date inside the code, and neither OCR nor the CNN can
see that on its own.

This is a different argument from the "barcode lookup is insufficient" position
in the paper's Table I, and it remains true. A copied code on a counterfeit
pack still verifies, so the code alone proves nothing. Used as a consistency
check against the printed text, it detects a class of tampering the other
streams miss entirely.

Decoder tiers
-------------
1. OpenCV (always available)  -- QR codes and common 1D symbologies. No extra
                                 system libraries, which keeps the Raspberry Pi
                                 install simple.
2. pyzbar (optional)          -- broader 1D coverage.
3. pylibdmtx (optional)       -- GS1 DataMatrix. This is the symbology most
                                 often used on pharmaceutical cartons, so it is
                                 worth installing where DataMatrix is expected.

Each tier degrades silently to the next. With no optional libraries installed
the scanner still reads QR codes.
"""

from __future__ import annotations

import calendar
import logging
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import cv2
import numpy as np

from .dateparse import ParsedDate

log = logging.getLogger(__name__)

# GS1 separator. In encoded data FNC1 is transmitted as ASCII 29 (GS).
GS = "\x1d"


# --------------------------------------------------------------------------
# GS1 Application Identifiers
# --------------------------------------------------------------------------
#
# AIs whose data length is fixed by the GS1 General Specifications. These need
# no separator, which is what makes an unseparated concatenated string
# parseable at all. Everything not listed here is variable length and runs to
# the next FNC1 or to the end of the payload.

PREDEFINED_LENGTHS: dict[str, int] = {
    "00": 18,  # SSCC
    "01": 14,  # GTIN
    "02": 14,  # GTIN of contained trade items
    "03": 14,
    "04": 16,
    "11": 6,   # production date        YYMMDD
    "12": 6,   # due date
    "13": 6,   # packaging date
    "14": 6,
    "15": 6,   # best before
    "16": 6,   # sell by
    "17": 6,   # expiry date            YYMMDD
    "18": 6,
    "19": 6,
    "20": 2,   # variant
    "41": 13,
}

# Human-readable names for the AIs this scanner cares about.
AI_NAMES = {
    "01": "gtin",
    "10": "batch",
    "11": "mfg_date",
    "17": "exp_date",
    "21": "serial",
    "30": "count",
    "710": "nhrn_in",   # national healthcare reimbursement number (India)
}

# Maximum data length for the variable-length AIs we use.
VARIABLE_MAX = {"10": 20, "21": 20, "30": 8, "710": 20}


@dataclass
class DecodedCode:
    """One physical symbol found on the package."""

    raw: str
    symbology: str            # "QR" / "DATAMATRIX" / "EAN13" / ...
    decoder: str              # which tier produced it
    is_gs1: bool = False
    elements: dict[str, str] = field(default_factory=dict)  # AI -> raw value
    gtin: str | None = None
    batch: str | None = None
    serial: str | None = None
    exp_date: ParsedDate | None = None
    mfg_date: ParsedDate | None = None
    parse_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbology": self.symbology,
            "decoder": self.decoder,
            "is_gs1": self.is_gs1,
            "gtin": self.gtin,
            "batch": self.batch,
            "serial": self.serial,
            "exp_date": self.exp_date.iso if self.exp_date else None,
            "mfg_date": self.mfg_date.iso if self.mfg_date else None,
            "elements": self.elements,
            # Truncated: a serialised payload can be long and the audit log
            # keeps the full value separately.
            "raw": self.raw[:120],
            "parse_notes": self.parse_notes,
        }


@dataclass
class BarcodeResult:
    """Everything the barcode stage found on one image."""

    codes: list[DecodedCode] = field(default_factory=list)
    decoders_available: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return bool(self.codes)

    @property
    def primary(self) -> DecodedCode | None:
        """The most informative code on the pack.

        A GS1 code carrying batch and expiry outranks a bare product barcode,
        because only the former can be cross-checked against the printed text.
        """
        if not self.codes:
            return None
        return max(
            self.codes,
            key=lambda c: (
                c.is_gs1,
                c.batch is not None,
                c.exp_date is not None,
                len(c.elements),
            ),
        )

    @property
    def batch(self) -> str | None:
        p = self.primary
        return p.batch if p else None

    @property
    def exp_date(self) -> ParsedDate | None:
        p = self.primary
        return p.exp_date if p else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "found": self.found,
            "count": len(self.codes),
            "decoders_available": self.decoders_available,
            "codes": [c.to_dict() for c in self.codes],
            "notes": self.notes,
        }


# --------------------------------------------------------------------------
# GS1 parsing
# --------------------------------------------------------------------------


def _parse_gs1_date(value: str, kind: str) -> ParsedDate | None:
    """Parse a GS1 YYMMDD date.

    A day of `00` is legal and means "end of the month" -- the same convention
    the printed month-only expiry uses. Getting this wrong would reject stock
    that is still dispensable on, say, the 30th of its expiry month.
    """
    if not re.fullmatch(r"\d{6}", value):
        return None

    yy, mm, dd = int(value[:2]), int(value[2:4]), int(value[4:6])
    # GS1 two-digit year window: 00-49 -> 20xx, 50-99 -> 19xx. Medicine expiry
    # dates are always in the 2000s in practice, but honour the spec.
    year = 2000 + yy if yy <= 49 else 1900 + yy
    if not 1 <= mm <= 12:
        return None

    if dd == 0:
        return ParsedDate(year, mm, None, value, kind)

    last = calendar.monthrange(year, mm)[1]
    if not 1 <= dd <= last:
        return None
    return ParsedDate(year, mm, dd, value, kind)


def _strip_brackets(payload: str) -> tuple[str, bool]:
    """Convert the human-readable bracketed form to separator form.

    Some scanners and test fixtures emit `(01)0345...(17)250910(10)LOT`. That is
    display notation, not what is encoded, but it turns up often enough that
    accepting it avoids confusing failures.
    """
    if "(" not in payload:
        return payload, False
    converted = re.sub(r"\((\d{2,4})\)", lambda m: GS + m.group(1), payload)
    return converted.lstrip(GS), True


def parse_gs1(payload: str) -> tuple[dict[str, str], list[str]]:
    """Parse a GS1 element string into {AI: value}.

    Handles both FNC1-separated and unseparated concatenated payloads. For the
    unseparated case, correctness depends entirely on the predefined-length
    table: a fixed-length AI consumes exactly its defined number of characters,
    and a variable-length AI runs to the next separator or the end.
    """
    notes: list[str] = []
    elements: dict[str, str] = {}

    payload, was_bracketed = _strip_brackets(payload)
    if was_bracketed:
        notes.append("Payload used the bracketed human-readable form.")

    # Some encoders prefix a leading FNC1; some readers surface it as "]d2"
    # or "]Q3" symbology identifiers.
    payload = re.sub(r"^\](?:d2|Q3|C1|e0|D2)", "", payload)
    payload = payload.lstrip(GS)

    i = 0
    n = len(payload)
    guard = 0

    while i < n and guard < 64:
        guard += 1

        if payload[i] == GS:
            i += 1
            continue

        # AI is 2-4 digits. Try the longest known key first so that "710"
        # is not mis-read as "71".
        ai = None
        for length in (4, 3, 2):
            candidate = payload[i:i + length]
            if len(candidate) < length or not candidate.isdigit():
                continue
            if candidate in PREDEFINED_LENGTHS or candidate in VARIABLE_MAX:
                ai = candidate
                break
            # 31xx-36xx measurement AIs are 4 digits with a 6-digit value.
            if length == 4 and re.fullmatch(r"3[1-6]\d\d", candidate):
                ai = candidate
                break

        if ai is None:
            # Unrecognised AI: assume 2 digits and treat the rest as variable,
            # so one unknown field cannot discard the whole payload.
            ai = payload[i:i + 2]
            if not ai.isdigit():
                notes.append(f"Stopped at non-numeric AI at offset {i}.")
                break

        i += len(ai)

        if ai in PREDEFINED_LENGTHS:
            length = PREDEFINED_LENGTHS[ai]
        elif re.fullmatch(r"3[1-6]\d\d", ai):
            length = 6
        else:
            length = None  # variable

        if length is not None:
            value = payload[i:i + length]
            if len(value) < length:
                notes.append(f"AI ({ai}) truncated; expected {length} chars.")
                break
            i += length
        else:
            end = payload.find(GS, i)
            if end == -1:
                end = n
            value = payload[i:end]
            i = end
            cap = VARIABLE_MAX.get(ai)
            if cap and len(value) > cap:
                notes.append(f"AI ({ai}) longer than the GS1 maximum {cap}.")

        if value:
            elements[ai] = value

    if guard >= 64:
        notes.append("Aborted GS1 parse after 64 elements.")

    return elements, notes


def looks_like_gs1(payload: str) -> bool:
    """Cheap check for whether a payload is worth GS1-parsing at all.

    Indian pack QR codes are not always GS1: some encode a verification URL or
    a proprietary token. Those must not be force-parsed into nonsense fields.
    """
    if not payload:
        return False
    if payload.startswith(("http://", "https://", "www.")):
        return False
    if "(" in payload and re.search(r"\(\d{2,4}\)", payload):
        return True
    if GS in payload:
        return True
    # Unseparated form: starts with a known AI and is mostly digits.
    head = payload[:2]
    return head in PREDEFINED_LENGTHS and payload[:4].isdigit()


def interpret(raw: str, symbology: str, decoder: str) -> DecodedCode:
    """Turn a raw decoded payload into a structured DecodedCode."""
    code = DecodedCode(raw=raw, symbology=symbology, decoder=decoder)

    if not looks_like_gs1(raw):
        if raw.startswith(("http://", "https://")):
            code.parse_notes.append(
                "Code contains a URL, not GS1 data. It cannot be cross-checked "
                "against the printed details offline."
            )
        return code

    elements, notes = parse_gs1(raw)
    code.is_gs1 = bool(elements)
    code.elements = elements
    code.parse_notes.extend(notes)

    code.gtin = elements.get("01")
    code.batch = elements.get("10")
    code.serial = elements.get("21")

    if "17" in elements:
        code.exp_date = _parse_gs1_date(elements["17"], "exp")
        if code.exp_date is None:
            code.parse_notes.append(f"Unreadable expiry field: {elements['17']}")
    if "11" in elements:
        code.mfg_date = _parse_gs1_date(elements["11"], "mfg")

    return code


# --------------------------------------------------------------------------
# Decoding tiers
# --------------------------------------------------------------------------


def _decode_opencv(image: np.ndarray) -> list[DecodedCode]:
    """QR and 1D barcodes using only OpenCV. Always available."""
    out: list[DecodedCode] = []

    # QR -- multi so several codes on one carton are all found.
    try:
        detector = cv2.QRCodeDetector()
        ok, decoded, points, _ = detector.detectAndDecodeMulti(image)
        if ok:
            for text in decoded:
                if text:
                    out.append(interpret(text, "QRCODE", "opencv"))
    except Exception as exc:
        log.debug("OpenCV QR multi-detect failed: %s", exc)
        try:
            text, points, _ = cv2.QRCodeDetector().detectAndDecode(image)
            if text:
                out.append(interpret(text, "QRCODE", "opencv"))
        except Exception:
            pass

    # 1D barcodes (EAN/UPC/Code128). Present in OpenCV's objdetect module.
    try:
        if hasattr(cv2, "barcode") and hasattr(cv2.barcode, "BarcodeDetector"):
            res = cv2.barcode.BarcodeDetector().detectAndDecodeMulti(image)
            ok, texts = res[0], res[1]
            # The tail of this tuple was reordered between OpenCV versions:
            #   4.x : retval, decoded_info, decoded_type, points
            #   5.0 : retval, decoded_info, points, straight_code
            # Unpacking positionally puts corner coordinates into the symbology
            # field, which then reaches the interface as the barcode "type".
            # Pick whichever element actually looks like a list of type names.
            types: tuple = ()
            for candidate in res[2:]:
                if isinstance(candidate, (list, tuple)) and all(
                    isinstance(c, str) for c in candidate
                ):
                    types = tuple(candidate)
                    break
            if ok:
                for i, text in enumerate(texts):
                    if not text:
                        continue
                    kind = types[i] if i < len(types) else ""
                    out.append(interpret(text, str(kind) or "BARCODE", "opencv"))
    except Exception as exc:
        log.debug("OpenCV barcode detect failed: %s", exc)

    return out


def _decode_pyzbar(image: np.ndarray) -> list[DecodedCode]:
    """Broader 1D and QR coverage. Optional."""
    try:
        from pyzbar import pyzbar  # type: ignore
    except Exception:
        return []

    out: list[DecodedCode] = []
    try:
        for sym in pyzbar.decode(image):
            # GS1 payloads are binary-ish: FNC1 arrives as \x1d, so decode
            # latin-1 rather than utf-8 to avoid losing it.
            text = sym.data.decode("latin-1", errors="replace")
            if text:
                out.append(interpret(text, sym.type, "pyzbar"))
    except Exception as exc:
        log.debug("pyzbar decode failed: %s", exc)
    return out


def _decode_dmtx(image: np.ndarray, timeout_ms: int = 1500) -> list[DecodedCode]:
    """GS1 DataMatrix. Optional, and the slowest tier.

    A timeout is essential: on a busy image libdmtx will search for a very long
    time, and the paper targets a 3-5 second end-to-end scan.
    """
    try:
        from pylibdmtx import pylibdmtx  # type: ignore
    except Exception:
        return []

    out: list[DecodedCode] = []
    try:
        for sym in pylibdmtx.decode(image, timeout=timeout_ms, max_count=4):
            text = sym.data.decode("latin-1", errors="replace")
            if text:
                out.append(interpret(text, "DATAMATRIX", "pylibdmtx"))
    except Exception as exc:
        log.debug("pylibdmtx decode failed: %s", exc)
    return out


def available_decoders() -> list[str]:
    names = ["opencv"]
    try:
        from pyzbar import pyzbar  # noqa: F401

        names.append("pyzbar")
    except Exception:
        pass
    try:
        from pylibdmtx import pylibdmtx  # noqa: F401

        names.append("pylibdmtx")
    except Exception:
        pass
    return names


def _prepare_variants(image: np.ndarray) -> list[np.ndarray]:
    """Image variants to try, cheapest first.

    Codes printed on foil or glossy carton often fail on the raw frame but
    decode after a contrast stretch or a binarisation.
    """
    variants = [image]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    variants.append(gray)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    variants.append(clahe.apply(gray))

    variants.append(
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    )

    # Upscale: small DataMatrix modules are often below the decoder's limit.
    h, w = gray.shape[:2]
    if max(h, w) < 1200:
        variants.append(
            cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        )
    return variants


def decode(
    image: np.ndarray,
    try_datamatrix: bool = True,
    max_variants: int = 3,
    dmtx_timeout_ms: int = 1500,
) -> BarcodeResult:
    """Find and decode every symbol on a package image.

    Stops as soon as a GS1 code carrying batch or expiry is found, since that
    is the only thing the cross-check needs and further searching only costs
    latency.

    `dmtx_timeout_ms` bounds each DataMatrix search. It is the setting that
    decides what this stage costs on a pack with no 2D code, because the
    search then runs to the full timeout on every variant before giving up.
    """
    result = BarcodeResult(decoders_available=available_decoders())
    seen: set[str] = set()

    def collect(codes: list[DecodedCode]) -> bool:
        useful = False
        for c in codes:
            if c.raw in seen:
                continue
            seen.add(c.raw)
            result.codes.append(c)
            if c.is_gs1 and (c.batch or c.exp_date):
                useful = True
        return useful

    for variant in _prepare_variants(image)[:max_variants]:
        if collect(_decode_opencv(variant)):
            break
        if collect(_decode_pyzbar(variant)):
            break
        if try_datamatrix and collect(_decode_dmtx(variant, dmtx_timeout_ms)):
            break

    if not result.codes:
        if "pylibdmtx" not in result.decoders_available:
            result.notes.append(
                "No code found. GS1 DataMatrix support is not installed; "
                "install pylibdmtx and libdmtx to read DataMatrix symbols, "
                "which are common on pharmaceutical cartons."
            )
        else:
            result.notes.append("No barcode or 2D code found on the package.")

    return result


# --------------------------------------------------------------------------
# Cross-check against the printed text
# --------------------------------------------------------------------------


@dataclass
class CrossCheck:
    """Result of comparing encoded data against printed text."""

    status: str  # "agree" | "conflict" | "partial" | "no_code" | "not_comparable"
    conflicts: list[str] = field(default_factory=list)
    agreements: list[str] = field(default_factory=list)

    @property
    def is_conflict(self) -> bool:
        return self.status == "conflict"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "conflicts": self.conflicts,
            "agreements": self.agreements,
        }


def _normalise_batch(value: str) -> str:
    """Compare batch codes ignoring case, spacing and punctuation.

    Also folds the character pairs OCR and operators most often confuse, so a
    printed `RF0159` read as `RFO159` is not reported as tampering.
    """
    cleaned = re.sub(r"[^A-Za-z0-9]", "", value).upper()
    return cleaned.translate(str.maketrans({"O": "0", "I": "1", "L": "1", "S": "5",
                                            "B": "8", "Z": "2"}))


def cross_check(
    barcode: BarcodeResult,
    ocr_batch: str | None,
    ocr_exp: ParsedDate | None,
) -> CrossCheck:
    """Compare the encoded data against what was printed on the pack.

    A conflict here is the tamper signal: the printed expiry has been altered
    while the encoded original remains. Note the asymmetry in how the two
    fields are treated -- a batch mismatch could still be an OCR error on a
    worn code, but an expiry that differs by month is not something OCR
    produces by accident.
    """
    primary = barcode.primary
    if primary is None:
        return CrossCheck("no_code")

    if not primary.is_gs1 or (primary.batch is None and primary.exp_date is None):
        return CrossCheck(
            "not_comparable",
            conflicts=[],
            agreements=[],
        )

    conflicts: list[str] = []
    agreements: list[str] = []
    compared = 0

    if primary.batch and ocr_batch:
        compared += 1
        if _normalise_batch(primary.batch) == _normalise_batch(ocr_batch):
            agreements.append(f"Batch {ocr_batch} matches the encoded batch.")
        else:
            conflicts.append(
                f"Printed batch '{ocr_batch}' does not match the batch encoded "
                f"in the barcode ('{primary.batch}')."
            )

    if primary.exp_date and ocr_exp:
        compared += 1
        same_month = (primary.exp_date.year, primary.exp_date.month) == (
            ocr_exp.year,
            ocr_exp.month,
        )
        if same_month:
            agreements.append(
                f"Expiry {ocr_exp.iso} matches the encoded expiry."
            )
        else:
            conflicts.append(
                f"Printed expiry {ocr_exp.iso} does not match the expiry "
                f"encoded in the barcode ({primary.exp_date.iso})."
            )

    if compared == 0:
        return CrossCheck("not_comparable")
    if conflicts:
        return CrossCheck("conflict", conflicts, agreements)
    if compared == 1:
        return CrossCheck("partial", [], agreements)
    return CrossCheck("agree", [], agreements)
