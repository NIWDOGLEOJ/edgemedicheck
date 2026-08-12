"""
Module 1 -- Image capture.

Provides a single `CameraSource` interface with three interchangeable
backends so the same pipeline code runs on a laptop during development and on
the Raspberry Pi 4 + Pi Camera Module v2 in the deployed enclosure.

Backends
--------
FolderSource   : replay images from disk. Deterministic, used for evaluation.
WebcamSource   : cv2.VideoCapture. Used for laptop development.
PiCameraSource : picamera2. Used on the Raspberry Pi in the fixed enclosure.

The LED ring described in Table II is driven through `LEDController`, which
degrades to a no-op when RPi.GPIO is unavailable.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np

from .config import CONFIG, CaptureConfig

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class CaptureError(RuntimeError):
    """Raised when no frame could be acquired."""


# --------------------------------------------------------------------------
# LED illumination
# --------------------------------------------------------------------------


class LEDController:
    """Controls the LED ring light (Table II).

    On non-Pi hardware every method is a no-op, so calling code never needs to
    branch on platform.
    """

    def __init__(self, pin: int | None = None) -> None:
        self.pin = pin
        self._gpio = None
        if pin is None:
            return
        try:
            import RPi.GPIO as GPIO  # type: ignore

            GPIO.setmode(GPIO.BCM)
            GPIO.setup(pin, GPIO.OUT)
            self._gpio = GPIO
            log.info("LED ring bound to BCM pin %s", pin)
        except Exception as exc:  # pragma: no cover - hardware only
            log.debug("GPIO unavailable, LED control disabled (%s)", exc)

    def on(self) -> None:
        if self._gpio:
            self._gpio.output(self.pin, self._gpio.HIGH)

    def off(self) -> None:
        if self._gpio:
            self._gpio.output(self.pin, self._gpio.LOW)

    def close(self) -> None:  # pragma: no cover - hardware only
        if self._gpio:
            self._gpio.cleanup(self.pin)

    def __enter__(self) -> "LEDController":
        self.on()
        return self

    def __exit__(self, *exc) -> None:
        self.off()


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------


class CameraSource:
    """Base interface. `read()` returns one BGR frame."""

    name = "base"

    def read(self) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self) -> "CameraSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class FolderSource(CameraSource):
    """Replays images from a directory or an explicit list of files.

    Cycles by default so a demo loop keeps producing frames.
    """

    name = "folder"

    def __init__(
        self,
        path: str | Path | Sequence[str | Path],
        cycle: bool = True,
    ) -> None:
        if isinstance(path, (str, Path)):
            p = Path(path)
            if p.is_dir():
                self.files = sorted(
                    f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
                )
            elif p.is_file():
                self.files = [p]
            else:
                raise CaptureError(f"No such image path: {p}")
        else:
            self.files = [Path(f) for f in path]

        if not self.files:
            raise CaptureError(f"No images found in {path}")

        self.cycle = cycle
        self._index = 0
        self.last_path: Path | None = None

    def read(self) -> np.ndarray:
        if self._index >= len(self.files):
            if not self.cycle:
                raise CaptureError("Folder source exhausted")
            self._index = 0

        path = self.files[self._index]
        self._index += 1
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise CaptureError(f"Could not decode image: {path}")
        self.last_path = path
        return frame

    def __iter__(self) -> Iterator[tuple[Path, np.ndarray]]:
        """Iterate once over every file, yielding (path, frame)."""
        for path in self.files:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is not None:
                yield path, frame


class WebcamSource(CameraSource):
    """UVC / laptop webcam via OpenCV."""

    name = "webcam"

    def __init__(self, cfg: CaptureConfig | None = None) -> None:
        self.cfg = cfg or CONFIG.capture
        self.cap = cv2.VideoCapture(self.cfg.device_index)
        if not self.cap.isOpened():
            raise CaptureError(
                f"Could not open camera index {self.cfg.device_index}"
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        # Let auto-exposure and white balance settle before the real capture.
        for _ in range(self.cfg.warmup_frames):
            self.cap.read()
            time.sleep(0.03)

    def read(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise CaptureError("Webcam returned no frame")
        return frame

    def close(self) -> None:
        if self.cap:
            self.cap.release()


class PiCameraSource(CameraSource):  # pragma: no cover - hardware only
    """Raspberry Pi Camera Module v2 via picamera2."""

    name = "picamera"

    def __init__(self, cfg: CaptureConfig | None = None) -> None:
        self.cfg = cfg or CONFIG.capture
        try:
            from picamera2 import Picamera2  # type: ignore
        except ImportError as exc:
            raise CaptureError("picamera2 is not installed") from exc

        self.cam = Picamera2()
        still = self.cam.create_still_configuration(
            main={"size": (self.cfg.width, self.cfg.height), "format": "RGB888"}
        )
        self.cam.configure(still)
        self.cam.start()
        time.sleep(1.5)  # AWB / AE settle in the fixed enclosure

    def read(self) -> np.ndarray:
        rgb = self.cam.capture_array()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def close(self) -> None:
        try:
            self.cam.stop()
            self.cam.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


def open_source(
    backend: str | None = None,
    folder: str | Path | None = None,
    cfg: CaptureConfig | None = None,
) -> CameraSource:
    """Open the best available capture backend.

    "auto" order: picamera2 -> webcam -> folder. This is what lets the same
    entry point work on the Pi in the enclosure and on a developer laptop.
    """
    cfg = cfg or CONFIG.capture
    backend = (backend or cfg.backend).lower()

    if backend == "folder":
        if folder is None:
            raise CaptureError("folder backend requires a folder path")
        return FolderSource(folder)

    if backend == "picamera":
        return PiCameraSource(cfg)

    if backend == "webcam":
        return WebcamSource(cfg)

    if backend == "auto":
        try:
            return PiCameraSource(cfg)
        except Exception as exc:
            log.debug("picamera unavailable: %s", exc)
        try:
            return WebcamSource(cfg)
        except Exception as exc:
            log.debug("webcam unavailable: %s", exc)
        if folder is not None:
            return FolderSource(folder)
        raise CaptureError(
            "No camera available and no fallback folder supplied. "
            "Pass folder=... or set EMC_CAPTURE_BACKEND=folder."
        )

    raise CaptureError(f"Unknown capture backend: {backend}")


# --------------------------------------------------------------------------
# Scan-region validation (Algorithm 1, step 3)
# --------------------------------------------------------------------------


def find_package_region(
    frame: np.ndarray, cfg: CaptureConfig | None = None
) -> tuple[bool, tuple[int, int, int, int] | None, float]:
    """Locate the medicine package inside the enclosure.

    Returns (is_valid, bounding_box, area_ratio).

    The enclosure gives a controlled background, so a simple saturation +
    edge-density segmentation is enough and is far cheaper on the Pi than a
    detector network. `is_valid` implements the paper's check that the package
    actually sits inside the expected scan region.
    """
    cfg = cfg or CONFIG.capture
    h, w = frame.shape[:2]
    frame_area = float(h * w)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge density picks up the package outline against the flat enclosure mat.
    edges = cv2.Canny(blurred, 40, 140)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return False, None, 0.0

    largest = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(largest) / frame_area
    x, y, bw, bh = cv2.boundingRect(largest)

    is_valid = (
        cfg.min_package_area_ratio <= area_ratio <= cfg.max_package_area_ratio
    )
    return is_valid, (x, y, bw, bh), float(area_ratio)


class PresenceDetector:
    """Decides when a pack has been presented, and when it has been taken away.

    The live screen is hands-free: it waits for a package instead of scanning
    on a timer. It cannot poll with the pipeline to find out whether anything
    is there -- a full `scan_image` is 0.6-1.3 s on a workstation and several
    seconds on a Pi 4 -- so this runs the cheap checks at frame rate and lets
    the expensive pass fire once per pack.

    Two conditions must hold before a scan is worth taking:

    `present`  the largest contour fills enough of the frame, via
               `find_package_region`. This is a fill-fraction test, not
               recognition: it says something is there, not that it is a
               medicine box.
    `steady`   the scene has stopped changing. Scanning while a hand is still
               withdrawing yields motion blur and a wasted pipeline run, so a
               pack is read once it has been set down.

    Both are debounced over consecutive frames, because a single noisy frame
    should neither trigger a scan nor drop a verdict that is still on screen.

    The caller drives this with whatever frames it already has; the detector
    holds only the previous downscaled grey frame and three counters.
    """

    def __init__(self, cfg: CaptureConfig | None = None) -> None:
        self.cfg = cfg or CONFIG.capture
        self._prev_small: np.ndarray | None = None
        self._present_run = 0
        self._absent_run = 0
        self._steady_run = 0
        self.present = False
        self.steady = False
        self.motion = 0.0
        self.area_ratio = 0.0
        self._looks_present = False

    def reset(self) -> None:
        """Forget history, e.g. after the camera has been reopened."""
        self._prev_small = None
        self._present_run = self._absent_run = self._steady_run = 0
        self.present = False
        self.steady = False
        self._looks_present = False

    def _downscale(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        longest = max(h, w)
        edge = self.cfg.detector_max_edge
        if longest <= edge:
            return frame
        scale = edge / longest
        return cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                          interpolation=cv2.INTER_AREA)

    def update(self, frame: np.ndarray) -> dict:
        """Fold one frame in and return the current view of the scene."""
        small = self._downscale(frame)
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # Mean absolute difference against the previous frame. The first frame
        # has nothing to compare against, so treat it as moving: that way a
        # scan is never taken from a single frame with no stability evidence.
        if self._prev_small is not None and self._prev_small.shape == grey.shape:
            self.motion = float(np.abs(
                grey.astype(np.int16) - self._prev_small.astype(np.int16)
            ).mean())
        else:
            self.motion = float("inf")
        self._prev_small = grey

        _, _, self.area_ratio = find_package_region(small, self.cfg)
        looks_present = self.area_ratio >= self.cfg.min_package_area_ratio
        self._looks_present = looks_present

        if looks_present:
            self._present_run += 1
            self._absent_run = 0
        else:
            self._absent_run += 1
            self._present_run = 0

        if self._present_run >= self.cfg.frames_to_confirm_present:
            self.present = True
        if self._absent_run >= self.cfg.frames_to_confirm_absent:
            self.present = False

        if self.motion <= self.cfg.motion_threshold:
            self._steady_run += 1
        else:
            self._steady_run = 0
        self.steady = self._steady_run >= self.cfg.frames_to_confirm_steady

        return self.state()

    def state(self) -> dict:
        return {
            "present": bool(self.present),
            "steady": bool(self.steady),
            # inf is not representable in JSON; the caller reads "very high".
            "motion": (None if self.motion == float("inf")
                       else round(self.motion, 3)),
            "area_ratio": round(float(self.area_ratio), 4),
            # `present` latches through the absent debounce, so on its own it
            # stays true for a few frames after a pack is lifted away. Require
            # the current frame to still look occupied as well, or a scan can
            # fire at an empty counter just as the previous pack leaves.
            "ready": bool(self.present and self.steady and self._looks_present),
        }


def capture_scan(
    source: CameraSource,
    led: LEDController | None = None,
    retries: int = 3,
) -> tuple[np.ndarray, dict]:
    """Capture one validated frame, retrying if the package is out of region.

    Returns (frame, metadata). Metadata always reports whether the region check
    passed so downstream stages can annotate a low-confidence scan rather than
    silently proceeding.
    """
    led = led or LEDController(CONFIG.capture.led_gpio_pin)
    meta: dict = {"attempts": 0, "region_ok": False, "area_ratio": 0.0}

    led.on()
    try:
        frame = None
        for attempt in range(1, retries + 1):
            meta["attempts"] = attempt
            frame = source.read()
            ok, bbox, ratio = find_package_region(frame)
            meta.update(region_ok=ok, bbox=bbox, area_ratio=ratio)
            if ok:
                break
            log.debug(
                "Attempt %s: package area ratio %.3f outside accepted range",
                attempt,
                ratio,
            )
            time.sleep(0.15)
        if frame is None:
            raise CaptureError("No frame captured")
        return frame, meta
    finally:
        led.off()
