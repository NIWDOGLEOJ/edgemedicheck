"""
Algorithm 3 (second half) -- decision fusion.

Combines the three evidence streams into one pharmacy-facing verdict:

    RED    -- do not dispense. Expired, regulator-flagged, label mismatch, or
              strong visual counterfeit signal.
    YELLOW -- verify manually. Evidence is incomplete or borderline.
    GREEN  -- no issue detected by this device.

Two design rules from the paper are enforced here.

First, precedence. The checks run hard-fail first: expiry, then database
status, then visual suspicion, then uncertainty. An expired pack is red even
if the CNN thinks the packaging looks perfect, because a genuine expired pack
is still unsafe to dispense.

Second, and more important: absence of evidence is never treated as evidence
of safety. An unreadable label or an unknown batch produces YELLOW, not GREEN.
The device is a screening aid, so the failure mode it must avoid is telling a
pharmacist that an unverified pack is fine.

Note also that an unknown batch is never escalated to RED on its own. The
local database is seeded from pharmacy stock and public alerts, not a national
registry, so "not found" usually means the record is missing, not that the
medicine is counterfeit. Accusing a genuine pack destroys trust in the device.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from . import database as db
from .barcode import BarcodeResult, CrossCheck
from .cnn import VisualResult
from .config import CONFIG, CNNConfig, FusionConfig
from .dateparse import days_until_expiry
from .ocr import OCRResult

# Verdicts
GREEN = "green"
YELLOW = "yellow"
RED = "red"

# Not a verdict. A finding marked INFO is shown to the pharmacist for
# transparency but does not by itself change the outcome -- used for context
# such as "this ran without a trained model".
INFO = "info"

# Reason codes. Short, stable, and printable on a small display -- these are
# what the paper calls the "short reason code" shown to the pharmacist.
R_EXPIRED_LABEL = "EXPIRED_LABEL"
R_EXPIRED_DB = "EXPIRED_DB"
R_RECALLED = "RECALLED_OR_NSQ"
R_MISMATCH = "LABEL_DB_MISMATCH"
R_NAME_MISMATCH = "MANUFACTURER_UNCLEAR"
R_CODE_CONFLICT = "CODE_TEXT_CONFLICT"
R_CODE_EXPIRED = "EXPIRED_BARCODE"
R_CODE_AGREE = "CODE_TEXT_AGREE"
R_NO_CODE = "NO_CODE_FOUND"
R_COUNTERFEIT = "VISUAL_COUNTERFEIT"
R_NEAR_EXPIRY = "NEAR_EXPIRY"
R_UNKNOWN_BATCH = "UNKNOWN_BATCH"
R_OCR_UNCERTAIN = "OCR_UNCERTAIN"
R_VISUAL_BORDERLINE = "VISUAL_BORDERLINE"
R_CAPTURE_POOR = "POOR_CAPTURE"
R_NO_MODEL = "NO_TRAINED_MODEL"
R_OK = "OK"

# Plain-language text shown alongside each code.
REASON_TEXT = {
    R_EXPIRED_LABEL: "Printed expiry date has passed.",
    R_EXPIRED_DB: "Database record shows this batch has expired.",
    R_RECALLED: "This batch is flagged as recalled or not of standard quality.",
    R_MISMATCH: "Printed details do not match the verified batch record.",
    R_NAME_MISMATCH: "Printed manufacturer name could not be matched to the record.",
    R_CODE_CONFLICT: "Printed details contradict the data encoded in the barcode.",
    R_CODE_EXPIRED: "The barcode encodes an expiry date that has passed.",
    R_CODE_AGREE: "Printed details match the barcode.",
    R_NO_CODE: "No barcode or 2D code was found on the package.",
    R_COUNTERFEIT: "Package appearance is strongly inconsistent with genuine stock.",
    R_NEAR_EXPIRY: "Medicine is close to its expiry date.",
    R_UNKNOWN_BATCH: "Batch not found in the local database.",
    R_OCR_UNCERTAIN: "Printed details could not be read reliably.",
    R_VISUAL_BORDERLINE: "Package appearance is borderline.",
    R_CAPTURE_POOR: "Image quality was too low for a reliable check.",
    R_NO_MODEL: "Visual check ran without a trained model.",
    R_OK: "No issue detected.",
}

# Severity ordering, used to pick the headline reason when several fire.
SEVERITY = {
    R_RECALLED: 100,
    # Ranked above the plain expiry checks: a contradiction between the printed
    # text and the encoded data is evidence the label itself was altered, which
    # is a more serious finding than stock that has simply aged out.
    R_CODE_CONFLICT: 98,
    R_EXPIRED_DB: 95,
    R_EXPIRED_LABEL: 94,
    R_CODE_EXPIRED: 93,
    R_MISMATCH: 90,
    R_COUNTERFEIT: 85,
    R_CAPTURE_POOR: 60,
    R_OCR_UNCERTAIN: 55,
    R_UNKNOWN_BATCH: 50,
    R_NAME_MISMATCH: 48,
    R_VISUAL_BORDERLINE: 45,
    R_NEAR_EXPIRY: 40,
    R_NO_CODE: 12,
    R_NO_MODEL: 10,
    R_CODE_AGREE: 5,
    R_OK: 0,
}

# Quality gates. Below these the capture itself is the problem.
MIN_SHARPNESS = 25.0
MIN_BRIGHTNESS = 40.0
MAX_BRIGHTNESS = 245.0


@dataclass
class Finding:
    """One triggered check."""

    code: str
    verdict: str
    detail: str = ""

    @property
    def text(self) -> str:
        return self.detail or REASON_TEXT.get(self.code, self.code)

    @property
    def severity(self) -> int:
        return SEVERITY.get(self.code, 0)

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "verdict": self.verdict, "text": self.text}


@dataclass
class Verdict:
    """Final fused decision."""

    verdict: str  # GREEN | YELLOW | RED
    reason_code: str
    reason_text: str
    findings: list[Finding] = field(default_factory=list)
    advice: str = ""

    @property
    def is_safe(self) -> bool:
        return self.verdict == GREEN

    @property
    def display_colour(self) -> str:
        return {GREEN: "#1a7f37", YELLOW: "#bf8700", RED: "#cf222e"}[self.verdict]

    @property
    def headline(self) -> str:
        return {
            GREEN: "No issue detected",
            YELLOW: "Manual verification required",
            RED: "Do not dispense",
        }[self.verdict]

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "headline": self.headline,
            "reason_code": self.reason_code,
            "reason_text": self.reason_text,
            "advice": self.advice,
            "findings": [f.to_dict() for f in self.findings],
        }


ADVICE = {
    RED: "Set this pack aside and do not dispense it. Escalate per pharmacy policy.",
    YELLOW: "Check the printed details against the invoice or supplier record "
            "before dispensing. Rescan if the image was unclear.",
    GREEN: "This device found no issue. It is a screening aid, not a "
           "regulatory or chemical authentication.",
}


def fuse(
    ocr: OCRResult,
    lookup: db.LookupResult,
    visual: VisualResult,
    sharpness: float | None = None,
    brightness: float | None = None,
    region_ok: bool = True,
    today: date | None = None,
    fusion_cfg: FusionConfig | None = None,
    cnn_cfg: CNNConfig | None = None,
    barcode: "BarcodeResult | None" = None,
    crosscheck: "CrossCheck | None" = None,
) -> Verdict:
    """Apply Algorithm 3's decision rules and return the pharmacy-facing verdict."""
    fusion_cfg = fusion_cfg or CONFIG.fusion
    cnn_cfg = cnn_cfg or CONFIG.cnn
    today = today or date.today()

    findings: list[Finding] = []

    # ---- Capture quality gate ------------------------------------------
    # Run first so a bad photo is reported as a bad photo, rather than
    # propagating as a fake OCR failure or a spurious CNN score.
    poor_capture = False
    if sharpness is not None and sharpness < MIN_SHARPNESS:
        poor_capture = True
        findings.append(
            Finding(
                R_CAPTURE_POOR,
                YELLOW,
                f"Image is blurred (sharpness {sharpness:.0f} < {MIN_SHARPNESS:.0f}).",
            )
        )
    if brightness is not None and not (MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS):
        poor_capture = True
        findings.append(
            Finding(
                R_CAPTURE_POOR,
                YELLOW,
                f"Lighting is outside the usable range (brightness {brightness:.0f}).",
            )
        )
    if not region_ok:
        poor_capture = True
        findings.append(
            Finding(
                R_CAPTURE_POOR,
                YELLOW,
                "Package was not fully inside the scan region.",
            )
        )

    # ---- Barcode cross-check -------------------------------------------
    # Run before the expiry checks so that a contradiction between printed and
    # encoded data leads the verdict. A relabelled pack is a more serious
    # finding than one that has merely aged out.
    if crosscheck is not None and crosscheck.is_conflict:
        findings.append(
            Finding(R_CODE_CONFLICT, RED, " ".join(crosscheck.conflicts))
        )
    elif crosscheck is not None and crosscheck.status in ("agree", "partial"):
        findings.append(
            Finding(R_CODE_AGREE, INFO, " ".join(crosscheck.agreements))
        )
    elif barcode is not None and not barcode.found:
        # Informational only. Plenty of legitimate packaging carries no 2D
        # code, so its absence cannot count against the pack.
        findings.append(
            Finding(
                R_NO_CODE,
                INFO,
                "No barcode or 2D code was found, so the printed details could "
                "not be cross-checked against encoded data.",
            )
        )

    # ---- Step 3-4: expiry --------------------------------------------
    # Both renderings of the expiry date are checked. If a label was altered,
    # the printed date may look valid while the encoded original has passed --
    # so treating either as disqualifying is the conservative reading.
    exp = ocr.exp_date
    if exp is not None and exp.effective < today:
        findings.append(
            Finding(
                R_EXPIRED_LABEL,
                RED,
                f"Printed expiry {exp.iso} passed on {exp.effective.isoformat()}.",
            )
        )

    code_exp = barcode.exp_date if barcode is not None else None
    if code_exp is not None and code_exp.effective < today:
        findings.append(
            Finding(
                R_CODE_EXPIRED,
                RED,
                f"Barcode encodes expiry {code_exp.iso}, which passed on "
                f"{code_exp.effective.isoformat()}.",
            )
        )

    # ---- Step 5-6: database status -------------------------------------
    if lookup.status == db.MATCH_UNSAFE:
        findings.append(Finding(R_RECALLED, RED, lookup.reason))
    elif lookup.status == db.MATCH_EXPIRED:
        findings.append(Finding(R_EXPIRED_DB, RED, lookup.reason))
    elif lookup.status == db.MATCH_MISMATCH:
        findings.append(Finding(R_MISMATCH, RED, lookup.reason))
    elif lookup.status == db.MATCH_NAME_MISMATCH:
        # Free-text field disagreement only. Worth showing, not worth blocking.
        findings.append(Finding(R_NAME_MISMATCH, YELLOW, lookup.reason))
    elif lookup.status == db.MATCH_UNKNOWN:
        # Deliberately yellow, not red -- see module docstring.
        findings.append(Finding(R_UNKNOWN_BATCH, YELLOW, lookup.reason))

    # ---- Step 7-8: visual suspicion ------------------------------------
    # An unusable visual stream contributes nothing. This is the "no trained
    # model and no calibration" case: abstaining is correct, because a stream
    # that cannot distinguish genuine from suspicious would only add noise to
    # a safety decision. The OCR and database checks remain fully valid, so a
    # clean pack can still reach GREEN -- with the limitation disclosed.
    if not visual.usable:
        findings.append(
            Finding(
                R_NO_MODEL,
                INFO,
                "Visual authenticity check did not run: no trained CNN and no "
                "reference calibration. Expiry and batch checks were still "
                "applied.",
            )
        )
    else:
        if visual.suspicion_score >= cnn_cfg.suspicion_threshold:
            if visual.is_model_backed:
                findings.append(
                    Finding(
                        R_COUNTERFEIT,
                        RED,
                        f"Visual authenticity score {visual.suspicion_score:.2f} "
                        f"exceeds the counterfeit threshold "
                        f"{cnn_cfg.suspicion_threshold:.2f}.",
                    )
                )
            else:
                # A calibrated heuristic is strong enough to warrant review,
                # but not to accuse a pack of being counterfeit.
                findings.append(
                    Finding(
                        R_VISUAL_BORDERLINE,
                        YELLOW,
                        f"Calibrated visual score {visual.suspicion_score:.2f} is "
                        "high. This is an anomaly measure, not a CNN "
                        "prediction, so treat it as a prompt to inspect.",
                    )
                )
        elif visual.suspicion_score >= cnn_cfg.review_threshold:
            findings.append(
                Finding(
                    R_VISUAL_BORDERLINE,
                    YELLOW,
                    f"Visual authenticity score {visual.suspicion_score:.2f} is in "
                    "the review band.",
                )
            )

        if not visual.is_model_backed:
            findings.append(
                Finding(
                    R_NO_MODEL,
                    INFO,
                    "Visual check used the calibrated heuristic backend; train "
                    "and deploy the CNN for a model-backed result.",
                )
            )

    # ---- Step 9-10: OCR uncertainty ------------------------------------
    if ocr.status in ("uncertain", "failed"):
        detail = "; ".join(ocr.notes) if ocr.notes else REASON_TEXT[R_OCR_UNCERTAIN]
        findings.append(Finding(R_OCR_UNCERTAIN, YELLOW, detail))

    # ---- Near expiry (advisory, not in the paper's minimal rule set) ----
    if exp is not None and exp.effective >= today:
        remaining = days_until_expiry(exp, today)
        if 0 <= remaining <= fusion_cfg.near_expiry_days:
            findings.append(
                Finding(
                    R_NEAR_EXPIRY,
                    YELLOW,
                    f"Expires in {remaining} day(s) on {exp.effective.isoformat()}.",
                )
            )

    # ---- Resolve ---------------------------------------------------------
    # INFO findings are carried through for transparency but never escalate
    # the verdict.
    decisive = [f for f in findings if f.verdict in (RED, YELLOW)]

    if any(f.verdict == RED for f in decisive):
        final = RED
    elif decisive:
        final = YELLOW
    else:
        final = GREEN

    ordered = sorted(findings, key=lambda f: -f.severity)

    if final == GREEN:
        return Verdict(GREEN, R_OK, REASON_TEXT[R_OK], ordered, ADVICE[GREEN])

    # Headline = most severe decisive finding.
    headline = max(
        [f for f in decisive if f.verdict == final] or decisive,
        key=lambda f: f.severity,
    )

    return Verdict(
        verdict=final,
        reason_code=headline.code,
        reason_text=headline.text,
        findings=ordered,
        advice=ADVICE[final],
    )
