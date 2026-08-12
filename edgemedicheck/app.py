#!/usr/bin/env python3
"""
Flask pharmacist interface (Module 6 -- "alert and display").

Deliberately simple. The paper's usability criterion is that a pharmacist can
place a package, start a scan, and read the result without technical
assistance, so the screen shows one large colour-coded verdict, a short reason,
the extracted fields, and nothing that requires interpretation.

Runs entirely offline. No external CSS, fonts, or scripts are fetched.

Start with:
    python run.py serve
    python run.py serve --backend folder --folder data/images
"""

from __future__ import annotations

import base64
import io
import logging
import math
import threading
import time
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from edgemedicheck import database as db
from edgemedicheck.capture import (
    CaptureError, LEDController, PresenceDetector, open_source,
)
from edgemedicheck.cnn import get_authenticator
from edgemedicheck.config import CONFIG
from edgemedicheck.pipeline import ScanResult, scan_image

log = logging.getLogger(__name__)


def _encode_preview(bgr: np.ndarray, max_edge: int = 480) -> str:
    """Return a data URI thumbnail so the page needs no static file serving."""
    h, w = bgr.shape[:2]
    scale = min(1.0, max_edge / max(h, w))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _json_safe(obj):
    """Replace non-finite floats with None, recursively.

    Several visual metrics are legitimately undefined on some frames --
    `text_angle_spread` is the standard deviation of detected text-line angles,
    which is NaN when no text lines were found at all. That is meaningful
    inside the pipeline, but Python's json module serialises it as a bare
    `NaN`, which is not valid JSON: `curl` piped through Python survives it
    because Python's own parser is lenient, while every browser rejects the
    whole response and the page sees a parse error instead of a result.

    So non-finite values are converted here, at the point where they leave
    Python, rather than by changing what the metrics mean.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.floating):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _result_payload(result: ScanResult, preview: str | None = None) -> dict:
    data = result.to_dict()
    record = result.lookup.record
    code = result.barcode.primary
    fields = [
        {"label": "Product",
         "value": (record.product_name if record
                   else result.ocr.get("product_name").value)},
        # Show where the batch came from: an encoded batch is machine-read and
        # much more trustworthy than one inferred from pixels, and the
        # pharmacist should be able to see which they are looking at.
        {"label": "Batch number", "value": result.batch_number,
         "source": "barcode" if result.barcode.batch else "printed text"},
        {"label": "Manufactured",
         "value": result.ocr.mfg_date.iso if result.ocr.mfg_date else None},
        {"label": "Expires",
         "value": result.exp_date.iso if result.exp_date else None,
         "source": None if result.ocr.exp_date else "barcode"},
        {"label": "Manufacturer",
         "value": (record.manufacturer if record else result.ocr.manufacturer)},
    ]
    if code is not None:
        fields.append({
            "label": "Barcode",
            "value": f"{code.symbology}"
                     + (f" · GS1" if code.is_gs1 else "")
                     + (f" · serial {code.serial}" if code.serial else ""),
        })

    data["display"] = {
        "colour": result.verdict.display_colour,
        "preview": preview or "",
        "fields": fields,
        "crosscheck": result.crosscheck.status,
    }
    return _json_safe(data)


class LiveCamera:
    """One reader for the camera, shared by the preview and the scanner.

    A capture device cannot be opened twice, and the live screen needs the
    same frames for two purposes at once: a smooth preview so the pharmacist
    can aim the pack, and a still to scan. So a single background thread owns
    the device and keeps the most recent frame; both consumers read that.

    The device is opened on demand and released once nobody has asked for a
    frame recently, which keeps `POST /api/scan/live` -- which opens the
    camera itself -- working when the live screen is not in use.

    The same thread runs the presence detector, because it is the only place
    that sees consecutive frames. Detection is strided rather than run on
    every grab: the point is to spend almost nothing while the counter is
    empty, since on a Pi that idle cost is paid all day.

    A failed open or a failed read no longer ends the session. The thread
    stops, records why, and backs off; the next reader restarts it. That is
    what lets the live screen survive a webcam being unplugged, a laptop lid
    closing, or another application taking the camera, without anyone
    reloading the page.
    """

    IDLE_TIMEOUT = 10.0  # seconds without a reader before releasing the device
    RETRY_MIN = 1.0      # backoff after a failed open/read
    RETRY_MAX = 10.0

    def __init__(self, backend: str | None, folder: str | Path | None) -> None:
        self._backend = backend
        self._folder = folder
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._source = None
        self._frame: np.ndarray | None = None
        self._last_request = 0.0
        self._running = False
        self._retry_at = 0.0
        self._retry_delay = self.RETRY_MIN
        self._detector = PresenceDetector()
        self._state: dict = self._detector.state()
        self._tick = 0
        self.error: str | None = None

    def _run(self) -> None:
        try:
            self._source = open_source(self._backend, folder=self._folder)
        except CaptureError as exc:
            self._fail(str(exc))
            return

        # A successful open clears the previous failure and the backoff, so a
        # camera that comes and goes is retried promptly each time.
        self.error = None
        self._retry_delay = self.RETRY_MIN
        self._detector.reset()

        try:
            while self._running:
                if time.monotonic() - self._last_request > self.IDLE_TIMEOUT:
                    break
                try:
                    frame = self._source.read()
                except Exception as exc:  # noqa: BLE001
                    self._fail(str(exc))
                    return
                if frame is not None:
                    with self._lock:
                        self._frame = frame
                    self._tick += 1
                    if self._tick % max(1, CONFIG.capture.detector_stride) == 0:
                        state = self._detector.update(frame)
                        with self._lock:
                            self._state = state
                # The pipeline is far slower than the sensor; there is nothing
                # to gain from grabbing faster than the preview can show.
                time.sleep(0.04)
        finally:
            self._running = False
            if self._source is not None:
                try:
                    self._source.close()
                except Exception:  # noqa: BLE001
                    pass
                self._source = None

    def _fail(self, message: str) -> None:
        """Record a failure and hold off before the next attempt."""
        self.error = message
        self._running = False
        self._retry_at = time.monotonic() + self._retry_delay
        self._retry_delay = min(self._retry_delay * 2, self.RETRY_MAX)
        # A stale frame would be presented as though it were live.
        with self._lock:
            self._frame = None
            self._state = self._detector.state()
        self._detector.reset()
        log.warning("Live camera stopped: %s (retrying in %.0fs)",
                    message, self._retry_delay)

    def _ensure_running(self) -> None:
        self._last_request = time.monotonic()
        if self._running:
            return
        if time.monotonic() < self._retry_at:
            return  # still backing off from the last failure
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # Give the device a moment to deliver its first frame, so the caller
        # gets a picture rather than an empty response on the first request.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if self._frame is not None or not self._running:
                return
            time.sleep(0.05)

    def read(self) -> np.ndarray | None:
        self._ensure_running()
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def state(self) -> dict:
        """Current scene reading, without copying a frame out."""
        self._ensure_running()
        with self._lock:
            state = dict(self._state)
        state["camera_ok"] = self._running and self.error is None
        state["error"] = self.error
        return state

    def stop(self) -> None:
        self._running = False


def create_app(
    db_path: str | Path | None = None,
    folder: str | Path | None = None,
    backend: str | None = None,
) -> Flask:
    app = Flask(__name__)
    app.config["EMC_DB_PATH"] = db_path
    app.config["EMC_FOLDER"] = folder
    app.config["EMC_BACKEND"] = backend

    db.init_db(db_path)

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------

    @app.route("/")
    def index():
        auth = get_authenticator()
        calib = auth.calibration
        groups = len(calib.groups) if calib else 0
        visual_ready = bool(
            auth.is_model_backed if hasattr(auth, "is_model_backed") else False
        ) or (calib.has_any if calib else False)

        return render_template(
            "index.html",
            product_count=db.count_products(db_path),
            scan_count=db.scan_stats(db_path)["total_scans"],
            backend=auth.backend,
            visual_ready=visual_ready,
            calibration_groups=groups,
            today=date.today().isoformat(),
        )

    live_camera = LiveCamera(backend, folder)
    app.config["EMC_LIVE_CAMERA"] = live_camera

    @app.route("/live")
    def live():
        return render_template(
            "live.html",
            backend=get_authenticator().backend,
            today=date.today().isoformat(),
        )

    # ------------------------------------------------------------------
    # Live preview
    # ------------------------------------------------------------------

    @app.route("/api/stream.mjpg")
    def stream():
        """Motion-JPEG preview.

        Served from the host camera rather than the browser's, because the
        counter deployment is a fixed enclosure camera attached to the Pi, and
        because `getUserMedia` is blocked on plain HTTP over a LAN address --
        which is exactly how a phone reaches this server.
        """
        def frames():
            boundary = b"--emcframe\r\n"
            while True:
                frame = live_camera.read()
                if frame is None:
                    break
                ok, buf = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ok:
                    yield (boundary
                           + b"Content-Type: image/jpeg\r\n\r\n"
                           + buf.tobytes() + b"\r\n")
                time.sleep(0.08)

        if live_camera.read() is None:
            return jsonify({
                "error": "camera_unavailable",
                "message": live_camera.error or "No camera frames available.",
            }), 503
        return Response(frames(),
                        mimetype="multipart/x-mixed-replace; boundary=emcframe")

    @app.route("/api/live/state")
    def live_state():
        """Is a pack in front of the camera, and has it stopped moving?

        Polled a few times a second by the live screen. Deliberately cheap:
        it reads counters the capture thread has already computed and never
        touches the pipeline, so an empty counter costs almost nothing.
        """
        return jsonify(live_camera.state())

    @app.route("/api/scan/frame", methods=["POST"])
    def scan_frame_endpoint():
        """Scan the current preview frame, without disturbing the preview."""
        frame = live_camera.read()
        if frame is None:
            return jsonify({
                "error": "camera_unavailable",
                "message": live_camera.error or "No camera frames available.",
            }), 503

        # Refuse to spend a pipeline pass on an empty counter. Without this
        # the scan returns amber OCR_UNCERTAIN for a pack that is not there,
        # which on this screen reads as "verify this manually".
        if not request.args.get("force") and not live_camera.state()["present"]:
            return jsonify({"idle": True, "reason": "no_package_in_frame"})

        try:
            from edgemedicheck.capture import find_package_region

            region_ok, bbox, ratio = find_package_region(frame)
            meta = {"region_ok": region_ok, "bbox": bbox, "area_ratio": ratio,
                    "source": "live"}
            meta.update({k: v for k, v in live_camera.state().items()
                         if k in ("motion", "steady")})
            result = scan_image(frame, capture_meta=meta, db_path=db_path)
            return jsonify(_result_payload(result))
        except Exception as exc:  # noqa: BLE001
            log.exception("Live frame scan failed")
            return jsonify({"error": "scan_failed", "message": str(exc)}), 500

    # ------------------------------------------------------------------
    # Scan endpoints
    # ------------------------------------------------------------------

    @app.route("/api/scan/live", methods=["POST"])
    def scan_live_endpoint():
        """Capture from the configured camera and scan."""
        try:
            source = open_source(
                app.config["EMC_BACKEND"], folder=app.config["EMC_FOLDER"]
            )
        except CaptureError as exc:
            return jsonify({
                "error": "camera_unavailable",
                "message": (
                    f"{exc} Start the server with "
                    "`python run.py serve --backend folder --folder data/images` "
                    "to demonstrate the workflow without a camera."
                ),
            }), 503

        try:
            from edgemedicheck.capture import capture_scan

            led = LEDController(CONFIG.capture.led_gpio_pin)
            frame, meta = capture_scan(source, led)
            result = scan_image(frame, capture_meta=meta, db_path=db_path)
            preview = _encode_preview(frame)
            return jsonify(_result_payload(result, preview))
        except Exception as exc:
            log.exception("Live scan failed")
            return jsonify({"error": "scan_failed", "message": str(exc)}), 500
        finally:
            source.close()

    @app.route("/api/scan/upload", methods=["POST"])
    def scan_upload_endpoint():
        """Scan an uploaded image. Used from a tablet or phone at the counter."""
        if "image" not in request.files:
            return jsonify({"error": "no_file", "message": "No image supplied."}), 400

        file = request.files["image"]
        raw = file.read()
        if not raw:
            return jsonify({"error": "empty_file", "message": "Empty upload."}), 400

        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({
                "error": "decode_failed",
                "message": "Could not read that file as an image.",
            }), 400

        try:
            from edgemedicheck.capture import find_package_region

            region_ok, bbox, ratio = find_package_region(frame)
            result = scan_image(
                frame,
                capture_meta={
                    "region_ok": region_ok, "bbox": bbox, "area_ratio": ratio,
                    "source": "upload", "filename": file.filename,
                },
                db_path=db_path,
            )
            return jsonify(_result_payload(result, _encode_preview(frame)))
        except Exception as exc:
            log.exception("Upload scan failed")
            return jsonify({"error": "scan_failed", "message": str(exc)}), 500

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    @app.route("/api/feedback", methods=["POST"])
    def feedback_endpoint():
        """Record a pharmacist correction.

        The scan being corrected is identified by scan_id. When the client
        omits it -- which the UI does, since it never sees the row id -- the
        most recent logged scan is used. That is correct for a single-counter
        device where the correction immediately follows the scan, and it is
        why the payload also carries the batch number: a mismatch means the
        wrong scan would be annotated, so the write is refused.
        """
        data = request.get_json(silent=True) or {}

        correct_verdict = (data.get("correct_verdict") or "").lower()
        if correct_verdict not in ("green", "yellow", "red"):
            return jsonify({
                "error": "bad_verdict",
                "message": "correct_verdict must be green, yellow or red.",
            }), 400

        correct_label = (data.get("correct_label") or "").lower() or None
        if correct_label and correct_label not in db.FEEDBACK_LABELS:
            return jsonify({
                "error": "bad_label",
                "message": f"correct_label must be one of "
                           f"{', '.join(db.FEEDBACK_LABELS)}.",
            }), 400

        scan_id = data.get("scan_id")
        scan = (
            db.get_scan(int(scan_id), db_path)
            if scan_id
            else db.latest_scan(db_path)
        )
        if scan is None:
            return jsonify({
                "error": "no_scan",
                "message": "No scan to attach this correction to.",
            }), 404

        # Guard against annotating the wrong row if scans raced.
        claimed = data.get("batch_number")
        if claimed and scan.get("batch_number") and claimed != scan["batch_number"]:
            return jsonify({
                "error": "scan_mismatch",
                "message": "The most recent scan does not match the one being "
                           "corrected. Rescan the package and try again.",
            }), 409

        fid = db.record_feedback(
            scan_id=scan["scan_id"],
            system_verdict=scan["verdict"],
            correct_verdict=correct_verdict,
            correct_label=correct_label,
            reason_code=scan.get("reason_code"),
            batch_number=scan.get("batch_number"),
            comment=(data.get("comment") or "").strip()[:500] or None,
            reported_by=(data.get("reported_by") or "").strip()[:80] or None,
            image_path=scan.get("image_path"),
            db_path=db_path,
        )

        return jsonify({
            "ok": True,
            "feedback_id": fid,
            "system_verdict": scan["verdict"],
            "correct_verdict": correct_verdict,
            "message": "Thank you. This correction has been saved and will be "
                       "used to improve the scanner.",
        })

    @app.route("/api/feedback/stats")
    def feedback_stats_endpoint():
        return jsonify(db.feedback_stats(db_path))

    # ------------------------------------------------------------------
    # Data endpoints
    # ------------------------------------------------------------------

    @app.route("/api/history")
    def history():
        limit = min(int(request.args.get("limit", CONFIG.web.history_limit)), 200)
        rows = db.recent_scans(limit, db_path)
        for r in rows:
            r.pop("details_json", None)
        return jsonify({"scans": rows, "stats": db.scan_stats(db_path)})

    @app.route("/api/products")
    def products():
        return jsonify({
            "products": [p.to_dict() for p in db.all_products(db_path)],
            "count": db.count_products(db_path),
        })

    @app.route("/api/health")
    def health():
        auth = get_authenticator()
        calib = auth.calibration
        return jsonify({
            "status": "ok",
            "visual_backend": auth.backend,
            "model_backed": auth.backend in ("tflite", "keras"),
            "calibration_groups": sorted(calib.groups) if calib else [],
            "products": db.count_products(db_path),
            "scans": db.scan_stats(db_path)["total_scans"],
        })

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_app().run(
        host=CONFIG.web.host, port=CONFIG.web.port, debug=CONFIG.web.debug
    )
