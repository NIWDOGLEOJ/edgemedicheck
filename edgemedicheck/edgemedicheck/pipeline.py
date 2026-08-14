"""
End-to-end scan pipeline.

Wires the six functional modules of Table IV into one call:

    capture -> preprocess -> OCR -> database cross-check
            -> CNN authentication -> fusion -> alert and log

Per-stage timings are recorded so the paper's latency claim (3-5 s on
Raspberry Pi 4) can be measured and reported per module rather than only
end to end.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import database as db
from .barcode import BarcodeResult, CrossCheck, cross_check
from .barcode import decode as decode_barcodes
from .capture import CameraSource, LEDController, capture_scan, find_package_region
from .cnn import VisualResult, get_authenticator
from .config import CAPTURE_DIR, CONFIG, Config
from .fusion import Verdict, fuse
from .ocr import OCRResult, run_ocr
from .preprocess import ProcessedImage, preprocess

log = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """Everything produced by one scan, ready for display, logging, or eval."""

    scan_id: str
    timestamp: datetime
    verdict: Verdict
    ocr: OCRResult
    lookup: db.LookupResult
    visual: VisualResult
    barcode: BarcodeResult = field(default_factory=BarcodeResult)
    crosscheck: CrossCheck = field(default_factory=lambda: CrossCheck("no_code"))
    timings_ms: dict[str, float] = field(default_factory=dict)
    image_path: str | None = None
    capture_meta: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, float] = field(default_factory=dict)
    # Row id in scan_log. The UI needs it to attach a pharmacist correction to
    # this exact scan; None when logging was skipped (batch evaluation).
    log_row_id: int | None = None

    @property
    def total_ms(self) -> float:
        return float(sum(self.timings_ms.values()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "scan_id": self.scan_id,
            "timestamp": self.timestamp.isoformat(timespec="seconds"),
            "verdict": self.verdict.to_dict(),
            "ocr": self.ocr.to_dict(),
            "lookup": self.lookup.to_dict(),
            "visual": self.visual.to_dict(),
            "barcode": self.barcode.to_dict(),
            "crosscheck": self.crosscheck.to_dict(),
            "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()},
            "total_ms": round(self.total_ms, 1),
            "log_row_id": self.log_row_id,
            "quality": {k: round(v, 2) for k, v in self.quality.items()},
            "capture_meta": {
                k: v for k, v in self.capture_meta.items() if k != "bbox"
            },
            "image_path": self.image_path,
        }

    @property
    def batch_number(self) -> str | None:
        """Best available batch number: encoded data outranks OCR."""
        return self.barcode.batch or self.ocr.batch_number

    @property
    def exp_date(self):
        """Best available expiry: the printed date, falling back to encoded."""
        return self.ocr.exp_date or self.barcode.exp_date

    def summary_line(self) -> str:
        """One-line console summary, used by the CLI and batch evaluation."""
        batch = self.batch_number or "-"
        exp = self.exp_date.iso if self.exp_date else "-"
        code = self.crosscheck.status[:4] if self.barcode.found else "none"
        return (
            f"[{self.verdict.verdict.upper():<6}] {self.verdict.reason_code:<18} "
            f"batch={batch:<12} exp={exp:<10} "
            f"code={code:<5} visual={self.visual.suspicion_score:.2f} "
            f"({self.total_ms:.0f} ms)"
        )


class _Timer:
    """Accumulates per-stage wall-clock timings."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}
        self._stage: str | None = None
        self._start: float = 0.0

    def stage(self, name: str) -> "_Timer":
        self._stage = name
        return self

    def __enter__(self) -> "_Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        if self._stage:
            elapsed = (time.perf_counter() - self._start) * 1000.0
            self.timings[self._stage] = self.timings.get(self._stage, 0.0) + elapsed


def scan_image(
    frame: np.ndarray,
    cfg: Config | None = None,
    today: date | None = None,
    save_image: bool = True,
    capture_meta: dict[str, Any] | None = None,
    db_path: Path | str | None = None,
    write_log: bool = True,
) -> ScanResult:
    """Run the full pipeline on one already-captured BGR frame.

    This is the function used both by the live UI and by offline batch
    evaluation, so the two paths cannot drift apart.
    """
    cfg = cfg or CONFIG
    today = today or date.today()
    timer = _Timer()
    scan_id = uuid.uuid4().hex[:12]
    timestamp = datetime.now()

    # -- Module 2: preprocessing -----------------------------------------
    with timer.stage("preprocess"):
        processed: ProcessedImage = preprocess(frame, cfg.preprocess)

    # -- Module 3: OCR extraction ----------------------------------------
    with timer.stage("ocr"):
        # Tesseract does better on the contrast-enhanced grayscale than on a
        # hard binarisation for glossy foil; we try it first and fall back.
        ocr = run_ocr(processed.ocr_gray, cfg.ocr)
        if ocr.status != "complete":
            alt = run_ocr(processed.ocr_image, cfg.ocr)
            found_alt = sum(
                1 for n in cfg.ocr.required_fields if alt.fields.get(n, None)
                and alt.fields[n].found
            )
            found_orig = sum(
                1 for n in cfg.ocr.required_fields if ocr.fields.get(n, None)
                and ocr.fields[n].found
            )
            if found_alt > found_orig or (
                found_alt == found_orig and alt.mean_confidence > ocr.mean_confidence
            ):
                ocr = alt

    # -- Module 3b: barcode / GS1 decoding -------------------------------
    barcode = BarcodeResult()
    crosscheck = CrossCheck("no_code")
    if cfg.barcode.enabled:
        with timer.stage("barcode"):
            try:
                barcode = decode_barcodes(
                    processed.deskewed,
                    try_datamatrix=cfg.barcode.try_datamatrix,
                    max_variants=cfg.barcode.max_variants,
                    dmtx_timeout_ms=cfg.barcode.dmtx_timeout_ms,
                )
                crosscheck = cross_check(barcode, ocr.batch_number, ocr.exp_date)
            except Exception as exc:
                log.warning("Barcode stage failed: %s", exc)
                barcode.notes.append(f"Barcode decoding failed: {exc}")

    # -- Module 4: database cross-check ----------------------------------
    with timer.stage("database"):
        # Prefer the encoded batch over the OCR reading. It is machine-read
        # rather than inferred from pixels, so it is not subject to the
        # character confusions that make OCR batch codes unreliable. The OCR
        # value is still used for the cross-check above, which is what would
        # surface a disagreement between the two.
        batch_for_lookup = barcode.batch or ocr.batch_number
        exp_for_lookup = ocr.exp_date or barcode.exp_date

        lookup = db.verify(
            batch_number=batch_for_lookup,
            exp_date=exp_for_lookup,
            product_name=ocr.get("product_name").value,
            manufacturer=ocr.manufacturer,
            today=today,
            tolerance_days=cfg.fusion.expiry_mismatch_tolerance_days,
            db_path=db_path,
        )
        # Fall back to the OCR batch if the encoded one is not on file. A pack
        # can legitimately carry a serialised code whose batch is recorded
        # under a different form.
        if (
            lookup.status == db.MATCH_UNKNOWN
            and barcode.batch
            and ocr.batch_number
            and barcode.batch != ocr.batch_number
        ):
            alt = db.verify(
                batch_number=ocr.batch_number,
                exp_date=exp_for_lookup,
                product_name=ocr.get("product_name").value,
                manufacturer=ocr.manufacturer,
                today=today,
                tolerance_days=cfg.fusion.expiry_mismatch_tolerance_days,
                db_path=db_path,
            )
            if alt.status != db.MATCH_UNKNOWN:
                lookup = alt

    # -- Module 5: CNN visual authentication -----------------------------
    with timer.stage("visual"):
        authenticator = get_authenticator(cfg.cnn)
        # Prefer the database product name over the OCR guess: it is the
        # canonical spelling, so it matches the calibration group key even
        # when OCR clipped the printed brand name.
        product_group = (
            lookup.record.product_name
            if lookup.record
            else ocr.get("product_name").value
        )
        visual = authenticator.predict(
            processed.cnn_image, processed.deskewed, product_group=product_group
        )

    # -- Module 6: fusion -------------------------------------------------
    with timer.stage("fusion"):
        meta = capture_meta or {}
        verdict = fuse(
            ocr=ocr,
            lookup=lookup,
            visual=visual,
            sharpness=processed.sharpness,
            brightness=processed.brightness,
            region_ok=bool(meta.get("region_ok", True)),
            today=today,
            fusion_cfg=cfg.fusion,
            cnn_cfg=cfg.cnn,
            barcode=barcode,
            crosscheck=crosscheck,
        )

    # -- Persist ----------------------------------------------------------
    image_path: str | None = None
    if save_image:
        with timer.stage("save"):
            out = CAPTURE_DIR
            out.mkdir(parents=True, exist_ok=True)
            image_path = str(
                out / f"{timestamp:%Y%m%d_%H%M%S}_{scan_id}_{verdict.verdict}.jpg"
            )
            cv2.imwrite(image_path, processed.deskewed)

    result = ScanResult(
        scan_id=scan_id,
        timestamp=timestamp,
        verdict=verdict,
        ocr=ocr,
        lookup=lookup,
        visual=visual,
        barcode=barcode,
        crosscheck=crosscheck,
        timings_ms=timer.timings,
        image_path=image_path,
        capture_meta=meta,
        quality={
            "sharpness": processed.sharpness,
            "brightness": processed.brightness,
            "skew_angle": processed.skew_angle,
        },
    )

    if write_log:
        with timer.stage("log"):
            result.log_row_id = _persist(result, db_path)

    return result


def _persist(result: ScanResult, db_path: Path | str | None = None) -> int | None:
    """Write the audit row required by Module 6 ('store scan logs')."""
    record = result.lookup.record
    try:
        return db.log_scan(
            {
                "timestamp": result.timestamp.isoformat(timespec="seconds"),
                "verdict": result.verdict.verdict,
                "reason_code": result.verdict.reason_code,
                "reason_text": result.verdict.reason_text,
                "batch_number": result.batch_number,
                "product_name": (
                    record.product_name
                    if record
                    else result.ocr.get("product_name").value
                ),
                "manufacturer": (
                    record.manufacturer if record else result.ocr.manufacturer
                ),
                "exp_date": result.ocr.exp_date.iso if result.ocr.exp_date else None,
                "ocr_status": result.ocr.status,
                "ocr_confidence": result.ocr.mean_confidence,
                "db_status": result.lookup.status,
                "cnn_score": result.visual.suspicion_score,
                "cnn_backend": result.visual.backend,
                "code_found": int(result.barcode.found),
                "code_batch": result.barcode.batch,
                "code_exp_date": (
                    result.barcode.exp_date.iso if result.barcode.exp_date else None
                ),
                "code_symbology": (
                    result.barcode.primary.symbology
                    if result.barcode.primary
                    else None
                ),
                "crosscheck": result.crosscheck.status,
                "latency_ms": result.total_ms,
                "image_path": result.image_path,
                "details_json": json.dumps(result.to_dict(), default=str),
            },
            db_path,
        )
    except Exception as exc:
        log.error("Failed to write scan log: %s", exc)
        return None


def scan_from_file(
    path: str | Path,
    cfg: Config | None = None,
    today: date | None = None,
    db_path: Path | str | None = None,
    write_log: bool = True,
    save_image: bool = False,
) -> ScanResult:
    """Scan a single image file. Used for evaluation over a labelled dataset."""
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Could not read image: {path}")

    region_ok, bbox, ratio = find_package_region(frame)
    return scan_image(
        frame,
        cfg=cfg,
        today=today,
        save_image=save_image,
        capture_meta={
            "region_ok": region_ok,
            "bbox": bbox,
            "area_ratio": ratio,
            "source_file": str(path),
        },
        db_path=db_path,
        write_log=write_log,
    )


def scan_live(
    source: CameraSource,
    cfg: Config | None = None,
    led: LEDController | None = None,
    today: date | None = None,
    db_path: Path | str | None = None,
) -> ScanResult:
    """Capture from the camera and run the full pipeline."""
    cfg = cfg or CONFIG
    frame, meta = capture_scan(source, led)
    return scan_image(
        frame,
        cfg=cfg,
        today=today,
        save_image=True,
        capture_meta=meta,
        db_path=db_path,
    )
