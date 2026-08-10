"""
Algorithm 1 -- Image Capture and Preprocessing.

Takes one raw BGR frame and produces two derived streams, exactly as described
in the paper:

  * OCR stream  -- deskewed, denoised, contrast-enhanced, binarised grayscale.
                   Optimised for small printed glyphs on foil and carton.
  * CNN stream  -- deskewed, colour-normalised RGB at the network input size.
                   Preserves colour and texture for authenticity checking.

The two streams deliberately diverge after deskew: binarisation destroys the
colour/print-quality cues the CNN needs, while colour normalisation leaves
noise that hurts Tesseract.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .config import CONFIG, PreprocessConfig

log = logging.getLogger(__name__)


@dataclass
class ProcessedImage:
    """Both output streams plus the intermediate artefacts we want to log."""

    ocr_image: np.ndarray  # binarised grayscale, uint8
    ocr_gray: np.ndarray  # enhanced grayscale before threshold (Tesseract often
    # does better on this than on a hard binarisation)
    cnn_image: np.ndarray  # float32 RGB, [0,1], cnn_input_size
    deskewed: np.ndarray  # colour, deskewed, cropped -- for display/audit
    skew_angle: float
    sharpness: float  # variance of Laplacian; low = blurry capture
    brightness: float  # mean V channel
    crop_box: tuple[int, int, int, int] | None


# --------------------------------------------------------------------------
# Individual steps
# --------------------------------------------------------------------------


def resize_to_working(
    image: np.ndarray, cfg: PreprocessConfig | None = None
) -> np.ndarray:
    """Downscale very large frames so Pi latency stays predictable."""
    cfg = cfg or CONFIG.preprocess
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= cfg.working_max_edge:
        return image
    scale = cfg.working_max_edge / longest
    return cv2.resize(
        image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )


def estimate_skew(gray: np.ndarray, cfg: PreprocessConfig | None = None) -> float:
    """Estimate text skew in degrees.

    Uses minAreaRect over thresholded ink pixels. Printed medicine labels have
    dense horizontal text runs, so this is more stable here than a Hough
    line vote over the whole frame.
    """
    cfg = cfg or CONFIG.preprocess

    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    # Join characters into text lines so the rectangle follows the baseline.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    coords = cv2.findNonZero(dilated)
    if coords is None or len(coords) < 50:
        return 0.0

    angle = cv2.minAreaRect(coords)[-1]
    # OpenCV returns angle in (0, 90]. Map to a signed rotation near zero.
    if angle > 45:
        angle -= 90

    if abs(angle) < cfg.deskew_min_angle or abs(angle) > cfg.deskew_max_angle:
        return 0.0
    return float(angle)


def rotate(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate about the centre, expanding the canvas so nothing is clipped."""
    if abs(angle) < 1e-3:
        return image
    h, w = image.shape[:2]
    centre = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(centre, angle, 1.0)

    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w / 2.0) - centre[0]
    M[1, 2] += (new_h / 2.0) - centre[1]

    return cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def crop_to_package(
    image: np.ndarray, pad: int = 12
) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    """Crop away the enclosure background around the package."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 140)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return image, None

    h, w = image.shape[:2]
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 0.05 * h * w:
        return image, None

    x, y, bw, bh = cv2.boundingRect(largest)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    return image[y0:y1, x0:x1], (x0, y0, x1 - x0, y1 - y0)


def enhance_for_ocr(
    bgr: np.ndarray, cfg: PreprocessConfig | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Produce the OCR stream: (enhanced grayscale, binarised).

    Order matters. Denoise before CLAHE, otherwise CLAHE amplifies the sensor
    noise that the Pi Camera shows on dark foil strips.
    """
    cfg = cfg or CONFIG.preprocess
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Upscale small crops -- Tesseract wants roughly 30px glyph height.
    h = gray.shape[0]
    if h < cfg.ocr_target_min_height:
        scale = min(3.0, cfg.ocr_target_min_height / max(h, 1))
        gray = cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )

    denoised = cv2.fastNlMeansDenoising(gray, None, cfg.denoise_h, 7, 21)

    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip_limit, tileGridSize=cfg.clahe_tile_grid
    )
    enhanced = clahe.apply(denoised)

    # Unsharp mask sharpens thermally-printed batch codes, which are the
    # weakest text on most strips.
    blur = cv2.GaussianBlur(enhanced, (0, 0), 3)
    enhanced = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)

    block = cfg.adaptive_block_size | 1  # must be odd
    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        cfg.adaptive_c,
    )
    # Remove single-pixel speckle without eroding thin strokes.
    binary = cv2.medianBlur(binary, 3)

    return enhanced, binary


def normalise_for_cnn(
    bgr: np.ndarray, cfg: PreprocessConfig | None = None
) -> np.ndarray:
    """Produce the CNN stream: colour-normalised float RGB at input size.

    Grey-world white balance removes the residual colour cast from the LED
    ring so that "colour shift" flagged by the CNN reflects the package print,
    not the illumination.
    """
    cfg = cfg or CONFIG.preprocess

    img = bgr.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0)
    grey = float(means.mean())
    if means.min() > 1e-3:
        img *= grey / means
    img = np.clip(img, 0, 255).astype(np.uint8)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, cfg.cnn_input_size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


# --------------------------------------------------------------------------
# Quality metrics
# --------------------------------------------------------------------------


def sharpness_score(gray: np.ndarray) -> float:
    """Variance of Laplacian. Low values indicate a blurred capture."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_score(bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 2].mean())


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------


def preprocess(
    frame: np.ndarray,
    cfg: PreprocessConfig | None = None,
    do_crop: bool = True,
) -> ProcessedImage:
    """Run Algorithm 1 end to end on a single captured frame."""
    cfg = cfg or CONFIG.preprocess

    working = resize_to_working(frame, cfg)

    gray0 = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    angle = estimate_skew(gray0, cfg)
    deskewed = rotate(working, angle)

    crop_box = None
    if do_crop:
        deskewed, crop_box = crop_to_package(deskewed)

    ocr_gray, ocr_binary = enhance_for_ocr(deskewed, cfg)
    cnn_image = normalise_for_cnn(deskewed, cfg)

    return ProcessedImage(
        ocr_image=ocr_binary,
        ocr_gray=ocr_gray,
        cnn_image=cnn_image,
        deskewed=deskewed,
        skew_angle=angle,
        sharpness=sharpness_score(cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)),
        brightness=brightness_score(deskewed),
        crop_box=crop_box,
    )
