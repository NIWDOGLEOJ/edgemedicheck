"""
Central configuration for EdgeMediCheck.

All tunable thresholds live here so that the evaluation section of the paper
can report exactly one place where operating points were set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

DATA_DIR = Path(os.environ.get("EMC_DATA_DIR", PROJECT_ROOT / "data"))
IMAGE_DIR = DATA_DIR / "images"
CAPTURE_DIR = DATA_DIR / "captures"
MODEL_DIR = DATA_DIR / "models"
DB_PATH = Path(os.environ.get("EMC_DB_PATH", DATA_DIR / "edgemedicheck.sqlite3"))

for _d in (DATA_DIR, IMAGE_DIR, CAPTURE_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------


@dataclass
class CaptureConfig:
    """Camera / capture settings.

    `backend` selects the image source:
      - "auto"    : picamera2 if importable, else opencv webcam, else folder
      - "picamera": Raspberry Pi Camera Module via picamera2
      - "webcam"  : any UVC camera via cv2.VideoCapture
      - "folder"  : replay images from disk (development + reproducible eval)
    """

    backend: str = os.environ.get("EMC_CAPTURE_BACKEND", "auto")
    device_index: int = 0
    width: int = 1280
    height: int = 720
    warmup_frames: int = 5
    # Fraction of the frame that must contain the package for a valid scan.
    # Algorithm 1 step 3: "check whether the package is inside the scan region".
    min_package_area_ratio: float = 0.08
    max_package_area_ratio: float = 0.98
    # LED ring control (Raspberry Pi only). None disables GPIO entirely.
    led_gpio_pin: int | None = None

    # ---- Hands-free presence detector (live screen) --------------------
    # The live screen waits for a pack to be presented instead of scanning on
    # a timer. A full pipeline pass is 0.6-1.3 s on a workstation and several
    # seconds on a Pi 4, so presence is decided by a separate cheap loop and
    # scan_image runs once per pack rather than once per interval.
    #
    # The detector runs on a downscaled copy: find_package_region does no
    # internal resize, and area_ratio is scale-invariant, so shrinking costs
    # nothing but accuracy of the bounding box, which the detector ignores.
    detector_max_edge: int = 480
    # Evaluate every Nth grabbed frame. The grab loop already sleeps 0.04 s,
    # so a stride of 3 gives roughly 8 evaluations/second.
    detector_stride: int = 3
    # Mean absolute frame-to-frame difference, 0-255, below which the scene
    # counts as still. A hand withdrawing from frame is far above this; sensor
    # noise on a static scene is far below.
    motion_threshold: float = 2.5
    # Consecutive detector frames needed before a state change is believed.
    # Debouncing stops a single noisy frame from triggering a scan or dropping
    # a held verdict.
    frames_to_confirm_present: int = 2
    frames_to_confirm_absent: int = 4
    # Consecutive still frames required before scanning, so the pack is
    # scanned once it has been set down rather than while it is moving.
    frames_to_confirm_steady: int = 2


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------


@dataclass
class PreprocessConfig:
    # Longest-edge resize before any processing. Keeps Pi latency predictable.
    working_max_edge: int = 1600
    # Deskew is skipped if the detected angle is below this (degrees) -- avoids
    # resampling blur for images that are already straight.
    deskew_min_angle: float = 0.4
    deskew_max_angle: float = 20.0
    # Non-local-means denoise strength for the OCR stream.
    denoise_h: int = 7
    # CLAHE contrast enhancement.
    clahe_clip_limit: float = 2.5
    clahe_tile_grid: tuple[int, int] = (8, 8)
    # Adaptive threshold window for the binarised OCR image.
    adaptive_block_size: int = 31
    adaptive_c: int = 11
    # Upscale small crops so Tesseract sees ~30px tall glyphs.
    ocr_target_min_height: int = 900
    # CNN input size (MobileNetV2 default).
    cnn_input_size: tuple[int, int] = (224, 224)


# --------------------------------------------------------------------------
# OCR
# --------------------------------------------------------------------------


@dataclass
class OCRConfig:
    tesseract_cmd: str | None = os.environ.get("EMC_TESSERACT_CMD")
    lang: str = "eng"
    # PSM 6 = assume a single uniform block of text. Good for flat cartons.
    # PSM 11 = sparse text. Good for strips where fields are scattered.
    psm_primary: int = 6
    psm_fallback: int = 11
    oem: int = 3
    # Mean per-character confidence below which a field is "low confidence".
    min_field_confidence: float = 55.0
    # Fields that must be present for the OCR stage to be considered complete.
    required_fields: tuple[str, ...] = ("batch_number", "exp_date")


# --------------------------------------------------------------------------
# CNN visual authentication
# --------------------------------------------------------------------------


@dataclass
class CNNConfig:
    tflite_model: Path = MODEL_DIR / "package_authenticity.tflite"
    keras_model: Path = MODEL_DIR / "package_authenticity.keras"
    labels: tuple[str, ...] = ("genuine", "suspicious")
    input_size: tuple[int, int] = (224, 224)
    # Suspicion score above this triggers a red "suspected counterfeit".
    suspicion_threshold: float = 0.65
    # Between review_threshold and suspicion_threshold -> yellow.
    review_threshold: float = 0.40
    # Training hyperparameters (used by training/train_cnn.py).
    batch_size: int = 32
    epochs_head: int = 8
    epochs_finetune: int = 6
    learning_rate_head: float = 1e-3
    learning_rate_finetune: float = 1e-5
    finetune_from_layer: int = 100
    validation_split: float = 0.2
    seed: int = 42


# --------------------------------------------------------------------------
# Decision fusion
# --------------------------------------------------------------------------


@dataclass
class BarcodeConfig:
    """Barcode / GS1 2D code reading.

    DataMatrix decoding is the slowest stage in the whole pipeline, so it is
    bounded by a timeout and can be disabled outright on hardware where the
    3-5 second budget is tight.
    """

    enabled: bool = os.environ.get("EMC_BARCODE", "1") != "0"
    try_datamatrix: bool = True
    dmtx_timeout_ms: int = 1500
    # How many preprocessed image variants to attempt before giving up.
    max_variants: int = 3


@dataclass
class FusionConfig:
    # Grace period in days. A package expiring within this window is flagged
    # yellow ("expiring soon") rather than green.
    near_expiry_days: int = 30
    # If the OCR expiry and the database expiry disagree by more than this many
    # days, treat it as a tampered/mismatched label.
    expiry_mismatch_tolerance_days: int = 31
    # Database statuses that immediately produce a red verdict.
    unsafe_statuses: tuple[str, ...] = ("recalled", "nsq", "spurious", "expired")


# --------------------------------------------------------------------------
# Web UI
# --------------------------------------------------------------------------


@dataclass
class WebConfig:
    host: str = os.environ.get("EMC_HOST", "127.0.0.1")
    port: int = int(os.environ.get("EMC_PORT", "5000"))
    debug: bool = os.environ.get("EMC_DEBUG", "0") == "1"
    history_limit: int = 25


@dataclass
class Config:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    barcode: BarcodeConfig = field(default_factory=BarcodeConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    web: WebConfig = field(default_factory=WebConfig)
    db_path: Path = DB_PATH


CONFIG = Config()
