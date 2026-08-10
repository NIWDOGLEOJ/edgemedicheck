"""
EdgeMediCheck -- an offline vision-based medicine authenticity and expiry
verification scanner for pharmacy counters.

Reference implementation of the system described in:
    Devesh R and Srinikesh D, "EdgeMediCheck: An Offline Vision-Based Medicine
    Authenticity and Expiry Verification Scanner for Pharmacy Counters",
    Sathyabama Institute of Science and Technology.

Quick start
-----------
    from edgemedicheck import scan_from_file, init_db
    init_db()
    result = scan_from_file("data/images/sample.jpg")
    print(result.summary_line())

Scope note
----------
This is a decision-support screening tool. It does not perform chemical
testing and is not a regulatory or legal authentication authority.
"""

from .config import CONFIG, Config
from .database import init_db, verify, upsert_product, recent_scans, scan_stats
from .fusion import GREEN, RED, YELLOW, Verdict, fuse
from .ocr import OCRResult, run_ocr
from .pipeline import ScanResult, scan_from_file, scan_image, scan_live
from .preprocess import preprocess

__version__ = "1.0.0"

__all__ = [
    "CONFIG",
    "Config",
    "GREEN",
    "YELLOW",
    "RED",
    "Verdict",
    "OCRResult",
    "ScanResult",
    "fuse",
    "init_db",
    "preprocess",
    "recent_scans",
    "run_ocr",
    "scan_from_file",
    "scan_image",
    "scan_live",
    "scan_stats",
    "upsert_product",
    "verify",
]
