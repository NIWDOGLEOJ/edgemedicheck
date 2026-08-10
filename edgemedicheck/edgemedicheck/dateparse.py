"""
Date normalisation for medicine labels.

Indian pharmaceutical packaging is inconsistent about date format. The same
pharmacy counter will see all of these on a single shelf:

    EXP 08/2027        MFG. 03/2025       EXP: AUG 2027
    E 08/27            MFD 03-25          EXPIRY 31/08/2027
    EXP.DATE 08.2027   MFG DATE MAR-2025  USE BEFORE 08/2027

A month-only expiry date legally means "end of that month", so 08/2027 is
valid through 2027-08-31. Getting that wrong makes the scanner reject stock
that is still legally dispensable, which is exactly the false alarm the paper
says must be controlled.

Ambiguity policy: for a two-part date we assume MM/YYYY. For a three-part date
we assume DD/MM/YYYY (Indian convention), falling back to MM/DD/YYYY only when
the first field exceeds 12 -- which is unambiguous.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date

MONTH_NAMES = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

# Two-digit years on medicine packs are always 20xx in practice.
CENTURY = 2000

# OCR routinely confuses these on low-contrast thermal print.
OCR_DIGIT_FIXES = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1",
                                 "|": "1", "S": "5", "B": "8", "Z": "2"})


@dataclass(frozen=True)
class ParsedDate:
    """A normalised label date.

    `day_known` is False for month-only dates. `as_of_date` gives the effective
    date used for comparisons: last day of month for expiry, first day for
    manufacturing.
    """

    year: int
    month: int
    day: int | None
    raw: str
    kind: str  # "exp" | "mfg" | "unknown"

    @property
    def day_known(self) -> bool:
        return self.day is not None

    @property
    def effective(self) -> date:
        if self.day is not None:
            return date(self.year, self.month, self.day)
        if self.kind == "exp":
            # Month-only expiry is valid to the end of that month.
            last = calendar.monthrange(self.year, self.month)[1]
            return date(self.year, self.month, last)
        return date(self.year, self.month, 1)

    @property
    def iso(self) -> str:
        if self.day is not None:
            return f"{self.year:04d}-{self.month:02d}-{self.day:02d}"
        return f"{self.year:04d}-{self.month:02d}"

    def __str__(self) -> str:
        return self.iso


def _normalise_year(value: int) -> int:
    if value < 100:
        return CENTURY + value
    return value


def _valid_ymd(year: int, month: int, day: int | None) -> bool:
    if not (1 <= month <= 12):
        return False
    if not (2000 <= year <= 2099):
        return False
    if day is not None:
        last = calendar.monthrange(year, month)[1]
        if not (1 <= day <= last):
            return False
    return True


# Ordered most-specific first. Whichever matches first wins.
_PATTERNS: list[tuple[re.Pattern, str]] = [
    # 31/08/2027, 31-08-27, 31.08.2027
    (re.compile(r"\b(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{2,4})\b"), "dmy"),
    # AUG 2027, AUGUST-27, AUG/2027
    (re.compile(r"\b([A-Za-z]{3,9})\s*[/\-.\s]\s*(\d{2,4})\b"), "myname"),
    # 2027-08 (ISO-ish, rare but appears on export packs)
    (re.compile(r"\b(20\d{2})\s*[/\-.]\s*(\d{1,2})\b"), "ym"),
    # 08/2027, 08-27, 08.2027
    (re.compile(r"\b(\d{1,2})\s*[/\-.]\s*(\d{2,4})\b"), "my"),
    # 082027 / 0827 with no separator at all
    (re.compile(r"\b(\d{2})(\d{4})\b"), "my_compact4"),
    (re.compile(r"\b(\d{2})(\d{2})\b"), "my_compact2"),
]


def _first_in(text: str, kind: str) -> ParsedDate | None:
    """Return the left-most valid date in `text`.

    Position must win over pattern order. A label reads
    `MFG MAY-26  EXP 05/09/2028`, and the window following "MFG" contains both
    dates. Scanning pattern-by-pattern would match the numeric DD/MM/YYYY form
    first and assign the expiry date as the manufacturing date. Taking the
    left-most match instead keeps each date attached to its own label.

    Ties at the same offset are broken by match length, so `05/09/2028` beats
    the shorter `05/09` reading of the same text.
    """
    best: tuple[int, int, ParsedDate] | None = None

    for pattern, style in _PATTERNS:
        for match in pattern.finditer(text):
            parsed = _build_from_match(match, style, kind)
            if parsed is None:
                continue
            key = (match.start(), -len(match.group(0)))
            if best is None or key < (best[0], best[1]):
                best = (key[0], key[1], parsed)

    return best[2] if best else None


def parse_date_string(text: str, kind: str = "unknown") -> ParsedDate | None:
    """Parse the left-most plausible date out of a fragment of OCR text."""
    if not text:
        return None

    cleaned = text.strip()
    parsed = _first_in(cleaned, kind)
    if parsed is not None:
        return parsed

    # Retry once with common OCR digit confusions repaired. Done second so we
    # never corrupt a date that already parsed cleanly.
    repaired = cleaned.translate(OCR_DIGIT_FIXES)
    if repaired != cleaned:
        return _first_in(repaired, kind)

    return None


def _build_from_match(
    match: re.Match, style: str, kind: str
) -> ParsedDate | None:
    raw = match.group(0)

    if style == "dmy":
        a, b, c = (int(match.group(i)) for i in (1, 2, 3))
        year = _normalise_year(c)
        # Indian convention: DD/MM/YYYY.
        if _valid_ymd(year, b, a):
            return ParsedDate(year, b, a, raw, kind)
        # Unambiguously US order (first field > 12).
        if _valid_ymd(year, a, b):
            return ParsedDate(year, a, b, raw, kind)
        return None

    if style == "myname":
        name = match.group(1).lower()
        if name not in MONTH_NAMES:
            return None
        month = MONTH_NAMES[name]
        year = _normalise_year(int(match.group(2)))
        if _valid_ymd(year, month, None):
            return ParsedDate(year, month, None, raw, kind)
        return None

    if style == "ym":
        year = int(match.group(1))
        month = int(match.group(2))
        if _valid_ymd(year, month, None):
            return ParsedDate(year, month, None, raw, kind)
        return None

    if style in ("my", "my_compact4", "my_compact2"):
        month = int(match.group(1))
        year = _normalise_year(int(match.group(2)))
        if _valid_ymd(year, month, None):
            return ParsedDate(year, month, None, raw, kind)
        # Swapped order, e.g. 2027/08.
        month2 = _normalise_year(int(match.group(2)))
        year2 = _normalise_year(int(match.group(1)))
        if _valid_ymd(year2, month2, None):
            return ParsedDate(year2, month2, None, raw, kind)
        return None

    return None


def is_expired(exp: ParsedDate, today: date | None = None) -> bool:
    """A pack is expired only after the last day of its expiry month."""
    today = today or date.today()
    return exp.effective < today


def days_until_expiry(exp: ParsedDate, today: date | None = None) -> int:
    today = today or date.today()
    return (exp.effective - today).days


def dates_agree(
    a: ParsedDate | None, b: ParsedDate | None, tolerance_days: int = 31
) -> bool:
    """Compare a label date against a database date.

    Month-only dates are compared at month granularity, so a database record of
    2027-08-31 matches a label reading 08/2027.
    """
    if a is None or b is None:
        return False
    if not a.day_known or not b.day_known:
        return (a.year, a.month) == (b.year, b.month)
    return abs((a.effective - b.effective).days) <= tolerance_days


def parse_db_date(value: str | None, kind: str = "unknown") -> ParsedDate | None:
    """Parse an ISO date stored in SQLite back into a ParsedDate."""
    if not value:
        return None
    value = value.strip()
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", value)
    if m:
        y, mo, d = (int(g) for g in m.groups())
        if _valid_ymd(y, mo, d):
            return ParsedDate(y, mo, d, value, kind)
        return None
    m = re.match(r"^(\d{4})-(\d{2})$", value)
    if m:
        y, mo = (int(g) for g in m.groups())
        if _valid_ymd(y, mo, None):
            return ParsedDate(y, mo, None, value, kind)
        return None
    return parse_date_string(value, kind)
