"""
Algorithm 3 (first half) -- visual authentication.

Produces a scalar "visual suspicion score" in [0, 1] for the package image,
where higher means the package looks less like verified genuine stock.

Backends, selected automatically in this order:

  1. tflite    -- TFLite interpreter over package_authenticity.tflite. This is
                  the deployment path on the Raspberry Pi 4.
  2. keras     -- full TensorFlow. Development on a workstation.
  3. torch     -- a torchvision ResNet-18 checkpoint, as produced by
                  `model training/train_model.py`. Heavier than TFLite, but it
                  runs the trained weights exactly as they were validated,
                  with no conversion step to go wrong.
  4. heuristic -- no trained model available. Falls back to a reference-
                  calibrated anomaly score over classical image cues.

Why the heuristic backend is built the way it is
-----------------------------------------------
An earlier version of this module scored each cue against hand-picked absolute
thresholds. Measured against a real image set, that approach failed badly: the
cues saturated at their limits and two of them ranked genuine packs as *more*
suspicious than defective ones. Absolute thresholds cannot work here, because
what counts as "normal" sharpness, saturation, and texture depends entirely on
the camera, the lens distance, and the LED ring in a given enclosure.

So the heuristic backend measures raw, unnormalised cues and scores a query by
how far it deviates from a reference distribution fitted on *known-genuine*
packs captured in the same enclosure. That is a standard one-class anomaly
formulation, and it is self-calibrating per deployment.

Crucially, before that reference set exists the backend does not guess. It
returns a neutral score with `usable = False`, and the fusion stage then
excludes the visual stream from the decision entirely rather than contributing
noise to a safety judgement.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import cv2
import numpy as np

from .config import CONFIG, MODEL_DIR, CNNConfig

log = logging.getLogger(__name__)

CALIBRATION_PATH = MODEL_DIR / "heuristic_calibration.json"

# Backends that run trained weights, as opposed to the classical-cue
# fallback. Kept in one place: callers used to spell this tuple out inline,
# so adding a backend updated some of them and silently missed others.
MODEL_BACKENDS = ("tflite", "keras", "torch")

# Minimum reference images before a calibration is trustworthy. Below this the
# fitted standard deviations are too noisy to threshold against.
MIN_CALIBRATION_SAMPLES = 12


@dataclass
class VisualResult:
    """Output of the visual authentication stage."""

    suspicion_score: float  # 0 = looks genuine, 1 = looks suspicious
    backend: str  # "tflite" | "keras" | "torch" | "heuristic"
    label: str  # "genuine" | "suspicious" | "unknown"
    usable: bool = True  # False -> fusion must ignore this stream
    cues: dict[str, float] = field(default_factory=dict)
    deviations: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def is_model_backed(self) -> bool:
        return self.backend in MODEL_BACKENDS

    def to_dict(self) -> dict[str, Any]:
        return {
            "suspicion_score": round(self.suspicion_score, 4),
            "backend": self.backend,
            "label": self.label,
            "usable": self.usable,
            "model_backed": self.is_model_backed,
            "cues": {k: round(v, 4) for k, v in self.cues.items()},
            "deviations": {k: round(v, 2) for k, v in self.deviations.items()},
            "notes": self.notes,
        }


# --------------------------------------------------------------------------
# Raw cue measurement
# --------------------------------------------------------------------------
#
# Each function returns a raw, unnormalised measurement. No thresholds, no
# clipping into [0,1] -- calibration happens later against the reference set.
# `higher_is_suspicious` records the expected direction so a one-sided score
# can be built; cues where either extreme is abnormal use two-sided scoring.


def m_edge_energy(bgr: np.ndarray) -> float:
    """Variance of Laplacian. Print sharpness proxy; lower = softer print."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def m_high_freq_ratio(bgr: np.ndarray) -> float:
    """Share of spectral energy above the mid-band.

    Reprinted or photocopied labels lose fine detail, which shows up as a
    smaller high-frequency share independently of overall contrast.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    total = float(spectrum.sum()) + 1e-6
    high = float(spectrum[radius > (min(h, w) * 0.25)].sum())
    return high / total


def m_mean_saturation(bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean() / 255.0)


def m_saturation_spread(bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].std() / 255.0)


def m_colour_balance(bgr: np.ndarray) -> float:
    """Deviation from neutral channel balance -- a colour-cast measure."""
    means = bgr.reshape(-1, 3).mean(axis=0).astype(np.float32) + 1e-6
    return float(means.std() / means.mean())


def m_text_line_angle_spread(bgr: np.ndarray) -> float:
    """Dispersion of near-horizontal text baseline angles, in degrees.

    Restricted to lines within 25 degrees of horizontal so that blister
    pockets, barcodes, and package borders do not swamp the measurement --
    that flaw made the earlier version of this cue useless.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=55, minLineLength=45, maxLineGap=6
    )
    if lines is None:
        return 0.0

    angles: list[float] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1)))
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        if abs(angle) <= 25.0:  # near-horizontal only
            angles.append(angle)

    if len(angles) < 4:
        return 0.0
    return float(np.std(angles))


def m_specular_hue_variance(bgr: np.ndarray) -> float:
    """Hue variance inside bright regions.

    A genuine hologram shifts colour across its surface under the LED ring and
    produces high variance; flat printed foil produces low variance.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, _, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = v > 200
    if mask.sum() < 60:
        return 0.0
    return float(np.var(h[mask].astype(np.float32)))


def m_specular_area(bgr: np.ndarray) -> float:
    """Fraction of the package that is specular highlight."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    return float((v > 200).sum() / v.size)


def m_block_texture_spread(bgr: np.ndarray) -> float:
    """Spread of local standard deviation across a coarse grid.

    Real packaging mixes dense text, flat colour, and logo areas, so local
    contrast varies a lot across the face. A uniformly reproduced label is
    flatter in this respect.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    k = 16
    h, w = gray.shape
    hs, ws = h // k, w // k
    if hs < 2 or ws < 2:
        return 0.0
    blocks = gray[: hs * k, : ws * k].reshape(hs, k, ws, k).transpose(0, 2, 1, 3)
    stds = blocks.reshape(hs * ws, -1).std(axis=1)
    return float(np.std(stds))


# name -> (measure fn, two_sided)
# two_sided=True  : both unusually high and unusually low are suspicious
# two_sided=False : only unusually LOW values are suspicious
CUE_MEASURES: dict[str, tuple[Callable[[np.ndarray], float], bool]] = {
    "edge_energy": (m_edge_energy, False),
    "high_freq_ratio": (m_high_freq_ratio, False),
    "mean_saturation": (m_mean_saturation, True),
    "saturation_spread": (m_saturation_spread, True),
    "colour_balance": (m_colour_balance, True),
    "text_angle_spread": (m_text_line_angle_spread, True),
    "specular_hue_variance": (m_specular_hue_variance, False),
    "specular_area": (m_specular_area, True),
    "block_texture_spread": (m_block_texture_spread, False),
}


def measure_cues(bgr: np.ndarray) -> dict[str, float]:
    """Measure every raw cue on one package image."""
    out: dict[str, float] = {}
    for name, (fn, _) in CUE_MEASURES.items():
        try:
            out[name] = float(fn(bgr))
        except Exception as exc:
            log.debug("Cue %s failed: %s", name, exc)
            out[name] = float("nan")
    return out


# --------------------------------------------------------------------------
# Reference calibration
# --------------------------------------------------------------------------


@dataclass
class Calibration:
    """Per-cue robust location and scale fitted on known-genuine packs.

    Median and MAD are used rather than mean and standard deviation so that a
    few mislabelled or badly-lit reference images do not distort the fit.
    """

    medians: dict[str, float]
    scales: dict[str, float]  # normalised MAD, comparable to a std deviation
    n_samples: int
    source: str = ""

    @property
    def is_trustworthy(self) -> bool:
        return self.n_samples >= MIN_CALIBRATION_SAMPLES

    def to_dict(self) -> dict[str, Any]:
        return {
            "medians": self.medians,
            "scales": self.scales,
            "n_samples": self.n_samples,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        return cls(
            medians=d["medians"],
            scales=d["scales"],
            n_samples=int(d.get("n_samples", 0)),
            source=d.get("source", ""),
        )


def normalise_group(name: str | None) -> str:
    """Canonical form for a calibration group key."""
    if not name:
        return ""
    return "".join(ch for ch in name.upper() if ch.isalnum())


@dataclass
class CalibrationSet:
    """A global calibration plus optional per-product-group calibrations.

    Why grouping matters
    --------------------
    Measured on a held-out set, fitting one distribution across mixed
    packaging (folding cartons and blister strips together) separated genuine
    from defective packs at only AUC 0.66 -- barely useful. Fitting the same
    cues *within* a package form raised that to 0.92 and 0.99 respectively.

    The reason is straightforward: a blister strip and a printed carton differ
    from each other far more than a genuine carton differs from a defective
    one. Pooling them inflates every scale estimate until real defects vanish
    into the noise.

    So calibration is keyed by product group, and a scan is compared against
    reference images of the same product wherever one exists. This mirrors the
    reference-image comparison used in the multimodal-LLM inspection
    literature: judge the pack against what that pack should look like, not
    against packaging in general.
    """

    groups: dict[str, Calibration] = field(default_factory=dict)
    global_calibration: Calibration | None = None
    # True when the global calibration was pooled from the per-product groups
    # rather than fitted on its own ungrouped reference images.
    global_is_pooled: bool = False

    def get(self, group: str | None) -> tuple[Calibration | None, str]:
        """Resolve the best calibration for a product group.

        Returns (calibration, scope) where scope is "group", "global", or
        "none".

        A pooled global calibration is deliberately NOT used to score a product
        that has no group of its own. Pooling several products' reference
        images describes "packaging in general", and judging an unseen product
        against it flags ordinary genuine stock as anomalous -- observed here
        as genuine packs of uncalibrated products scoring above 0.94. Abstaining
        is the honest outcome: we have no reference for this product, so we say
        so rather than raising a false alarm.

        A global calibration fitted from ungrouped loose images is different:
        there the operator explicitly chose a single flat reference set, so it
        is applied as intended.
        """
        key = normalise_group(group)
        if key and key in self.groups:
            calib = self.groups[key]
            if calib.is_trustworthy:
                return calib, "group"

        if (
            self.global_calibration
            and self.global_calibration.is_trustworthy
            and not self.global_is_pooled
        ):
            return self.global_calibration, "global"

        return None, "none"

    @property
    def has_any(self) -> bool:
        calib, scope = self.get(None)
        return scope != "none" or any(
            c.is_trustworthy for c in self.groups.values()
        )

    @property
    def total_samples(self) -> int:
        n = self.global_calibration.n_samples if self.global_calibration else 0
        return n + sum(c.n_samples for c in self.groups.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 2,
            "global": (
                self.global_calibration.to_dict()
                if self.global_calibration
                else None
            ),
            "global_is_pooled": self.global_is_pooled,
            "groups": {k: v.to_dict() for k, v in self.groups.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationSet":
        # Version 1 files stored a single flat calibration.
        if "version" not in d and "medians" in d:
            return cls(groups={}, global_calibration=Calibration.from_dict(d))
        return cls(
            groups={
                k: Calibration.from_dict(v) for k, v in (d.get("groups") or {}).items()
            },
            global_calibration=(
                Calibration.from_dict(d["global"]) if d.get("global") else None
            ),
            global_is_pooled=bool(d.get("global_is_pooled", False)),
        )

    def save(self, path: Path | str = CALIBRATION_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path | str = CALIBRATION_PATH) -> "CalibrationSet | None":
        path = Path(path)
        if not path.exists():
            return None
        try:
            return cls.from_dict(json.loads(path.read_text()))
        except Exception as exc:
            log.warning("Could not read calibration %s: %s", path, exc)
            return None


def fit_calibration(
    images: Iterable[np.ndarray | str | Path],
    source: str = "",
    preprocessed: bool = False,
) -> Calibration:
    """Fit the reference distribution from known-genuine package images.

    Reference images are pushed through the *same* preprocessing the scan path
    applies -- deskew and crop to the package -- before their cues are
    measured. Skipping that step silently compares a cropped package at scan
    time against a full frame including the enclosure mat at fit time, which
    shifts every cue by tens of scale units and makes the whole calibration
    meaningless.

    Set `preprocessed=True` only when the caller has already run preprocess()
    on the supplied arrays.
    """
    from .preprocess import preprocess as _preprocess  # local: avoids a cycle

    samples: list[dict[str, float]] = []
    for item in images:
        if isinstance(item, (str, Path)):
            img = cv2.imread(str(item), cv2.IMREAD_COLOR)
            if img is None:
                log.warning("Skipping unreadable image: %s", item)
                continue
        else:
            img = item

        if not preprocessed:
            try:
                img = _preprocess(img).deskewed
            except Exception as exc:
                log.warning("Preprocessing failed for a reference image: %s", exc)
                continue

        samples.append(measure_cues(img))

    if not samples:
        raise ValueError("No usable reference images supplied")

    medians: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in CUE_MEASURES:
        values = np.array(
            [s[name] for s in samples if not np.isnan(s.get(name, np.nan))],
            dtype=np.float64,
        )
        if values.size == 0:
            medians[name], scales[name] = 0.0, 1.0
            continue
        med = float(np.median(values))
        # 1.4826 scales MAD to be consistent with the standard deviation of a
        # normal distribution.
        mad = float(np.median(np.abs(values - med)) * 1.4826)
        # Floor the scale so a degenerate cue cannot produce huge deviations.
        floor = max(abs(med) * 0.02, 1e-6)
        medians[name] = med
        scales[name] = max(mad, floor)

    return Calibration(medians, scales, len(samples), source)


IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def shrink_towards(
    group: Calibration, pooled: Calibration, weight: float = 0.5
) -> Calibration:
    """Regularise a group's scale estimates towards the pooled estimate.

    A per-product reference set is deliberately homogeneous -- same artwork,
    same package form -- so its MAD can collapse towards zero on cues that
    barely move within the product. Left unregularised, ordinary pose and
    exposure variation then divides by a near-zero scale and every genuine
    pack saturates at maximum suspicion. Measured on a held-out set, that
    failure drove genuine packs to a mean score of 1.000 and AUC below chance.

    Flooring each group scale at a fraction of the pooled scale keeps small or
    tightly-clustered reference sets usable. This is the standard shrinkage
    estimator: trust the group where it has evidence, fall back on the pooled
    spread where it does not.
    """
    scales = {}
    for name, group_scale in group.scales.items():
        pooled_scale = pooled.scales.get(name, group_scale)
        scales[name] = max(group_scale, weight * pooled_scale)
    return Calibration(
        medians=dict(group.medians),
        scales=scales,
        n_samples=group.n_samples,
        source=group.source,
    )


def fit_calibration_set(folder: Path | str) -> CalibrationSet:
    """Fit a grouped calibration from a folder of known-genuine images.

    Layout:

        reference/
            PARACIP/        <- per-product group, compared like-for-like
                001.jpg
            AZITHRAL/
                001.jpg
            loose_image.jpg <- contributes to the global fallback only

    Every image also feeds the global calibration, which is used for products
    that have no group of their own. Group calibrations are strongly preferred
    -- see CalibrationSet's docstring for the measured difference.
    """
    folder = Path(folder)
    groups: dict[str, Calibration] = {}
    all_images: list[Path] = []

    for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
        images = sorted(p for p in sub.rglob("*") if p.suffix.lower() in IMAGE_EXT)
        if not images:
            continue
        all_images.extend(images)
        key = normalise_group(sub.name)
        if not key:
            continue
        try:
            groups[key] = fit_calibration(images, source=str(sub))
        except ValueError:
            log.warning("Skipping empty group: %s", sub)

    loose = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXT)
    all_images.extend(loose)

    global_calib = None
    if all_images:
        global_calib = fit_calibration(all_images, source=str(folder))

    # Regularise every group towards the pooled spread before saving.
    if global_calib is not None:
        groups = {k: shrink_towards(v, global_calib) for k, v in groups.items()}

    # If every reference image belonged to a product group, the global
    # calibration is a pooled artefact used only for shrinkage -- never for
    # scoring an unseen product. See CalibrationSet.get().
    return CalibrationSet(
        groups=groups,
        global_calibration=global_calib,
        global_is_pooled=bool(groups) and not loose,
    )


def score_against(
    cues: dict[str, float], calib: Calibration
) -> tuple[float, dict[str, float]]:
    """Score cues against the reference distribution.

    Returns (suspicion in [0,1], per-cue signed deviation in scale units).

    The aggregate uses the mean of the top-3 deviations rather than the mean of
    all of them. A counterfeit typically fails on a few specific cues while
    looking normal on the rest, so averaging everything would dilute exactly
    the signal we want.
    """
    deviations: dict[str, float] = {}

    for name, (_, two_sided) in CUE_MEASURES.items():
        value = cues.get(name, float("nan"))
        if np.isnan(value):
            continue
        med = calib.medians.get(name, 0.0)
        scale = calib.scales.get(name, 1.0) or 1.0
        z = (value - med) / scale
        deviations[name] = float(z)

    if not deviations:
        return 0.5, {}

    penalties: list[float] = []
    for name, z in deviations.items():
        two_sided = CUE_MEASURES[name][1]
        # One-sided cues are only suspicious when they fall below the norm.
        penalty = abs(z) if two_sided else max(0.0, -z)
        penalties.append(penalty)

    penalties.sort(reverse=True)
    top = penalties[: min(3, len(penalties))]
    aggregate = float(np.mean(top))

    # Map deviation to [0,1]. A mean top-3 deviation of 3 scale units maps to
    # roughly 0.73, and 6 units saturates near 0.95.
    suspicion = 1.0 - float(np.exp(-aggregate / 3.0))
    return float(np.clip(suspicion, 0.0, 1.0)), deviations


# --------------------------------------------------------------------------
# Authenticator
# --------------------------------------------------------------------------


class VisualAuthenticator:
    """Loads the best available backend once and reuses it across scans."""

    @property
    def is_model_backed(self) -> bool:
        """Whether a trained model is driving the visual stream.

        Callers guarded this with `hasattr`, which quietly meant "no" because
        the property only existed on VisualResult. The web index page read
        its "visual ready" flag from that guard, so it never reflected a
        loaded model regardless of backend.
        """
        return self.backend in MODEL_BACKENDS

    def __init__(
        self,
        cfg: CNNConfig | None = None,
        calibration_path: Path | str = CALIBRATION_PATH,
    ) -> None:
        self.cfg = cfg or CONFIG.cnn
        self.backend = "heuristic"
        self._interpreter = None
        self._keras = None
        self._torch = None
        self._torch_classes: list[str] = []
        self._torch_suspicious_index = 0
        self._input_details = None
        self._output_details = None
        self.calibration = CalibrationSet.load(calibration_path)
        self._load()

    # -- loading ---------------------------------------------------------

    def _load(self) -> None:
        if self._try_tflite():
            self.backend = "tflite"
            log.info("Visual authenticator: TFLite backend")
            return
        if self._try_keras():
            self.backend = "keras"
            log.info("Visual authenticator: Keras backend")
            return
        if self._try_torch():
            self.backend = "torch"
            log.info(
                "Visual authenticator: PyTorch backend (classes %s, "
                "suspicious = column %d)",
                self._torch_classes,
                self._torch_suspicious_index,
            )
            return

        self.backend = "heuristic"
        if self.calibration and self.calibration.has_any:
            log.info(
                "Visual authenticator: calibrated heuristic backend "
                "(%d reference images, %d product group(s))",
                self.calibration.total_samples,
                len(self.calibration.groups),
            )
        else:
            log.warning(
                "Visual authenticator: no trained model and no usable "
                "calibration. The visual stream will ABSTAIN. Run "
                "`python run.py calibrate <folder-of-genuine-images>` or train "
                "the CNN to enable visual authentication."
            )

    def _try_tflite(self) -> bool:
        path = Path(self.cfg.tflite_model)
        if not path.exists():
            return False
        interpreter = None
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore

            interpreter = Interpreter(model_path=str(path))
        except ImportError:
            try:
                import tensorflow as tf  # type: ignore

                interpreter = tf.lite.Interpreter(model_path=str(path))
            except ImportError:
                return False
        except Exception as exc:
            log.warning("Failed to load TFLite model: %s", exc)
            return False

        try:
            interpreter.allocate_tensors()
            self._interpreter = interpreter
            self._input_details = interpreter.get_input_details()
            self._output_details = interpreter.get_output_details()
            return True
        except Exception as exc:
            log.warning("Failed to allocate TFLite tensors: %s", exc)
            return False

    def _try_keras(self) -> bool:
        path = Path(self.cfg.keras_model)
        if not path.exists():
            return False
        try:
            import tensorflow as tf  # type: ignore

            self._keras = tf.keras.models.load_model(str(path))
            return True
        except Exception as exc:
            log.warning("Failed to load Keras model: %s", exc)
            return False

    # Class names that mean "this pack is not genuine". The checkpoint
    # records its own labels, and a ResNet trained as Fake/Real orders them
    # the opposite way round to this module's ("genuine", "suspicious"), so
    # the column is resolved by name rather than assumed by position. Get
    # this wrong and every genuine pack scores as counterfeit.
    _SUSPICIOUS_NAMES = {"fake", "counterfeit", "suspicious", "spurious"}

    def _try_torch(self) -> bool:
        path = Path(self.cfg.torch_model)
        if not path.exists():
            return False
        try:
            import torch
            from torchvision import models
        except ImportError:
            log.warning(
                "A torch checkpoint is present at %s but torch/torchvision "
                "are not installed; falling back.", path,
            )
            return False

        try:
            ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            classes = [str(c) for c in ckpt.get("class_names", ["Fake", "Real"])]

            suspicious = [
                i for i, c in enumerate(classes)
                if c.strip().lower() in self._SUSPICIOUS_NAMES
            ]
            if len(suspicious) != 1:
                log.error(
                    "Cannot tell which of %s means 'suspicious'; refusing to "
                    "guess, because guessing wrong inverts the verdict.",
                    classes,
                )
                return False

            model = models.resnet18(weights=None)
            in_features = model.fc.in_features
            model.fc = torch.nn.Sequential(
                torch.nn.Dropout(p=0.3),
                torch.nn.Linear(in_features, len(classes)),
            )
            model.load_state_dict(state)
            model.eval()
            torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

            self._torch = model
            self._torch_classes = classes
            self._torch_suspicious_index = suspicious[0]
            return True
        except Exception as exc:
            log.warning("Failed to load torch model: %s", exc)
            return False

    # -- inference -------------------------------------------------------

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        if image.dtype == np.float32 and image.max() <= 1.0:
            arr = image
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            arr = rgb.astype(np.float32) / 255.0
        if arr.shape[:2] != tuple(self.cfg.input_size):
            arr = cv2.resize(arr, self.cfg.input_size, interpolation=cv2.INTER_AREA)
        return np.expand_dims(arr, axis=0).astype(np.float32)

    def _infer_tflite(self, batch: np.ndarray) -> float:
        inp = self._input_details[0]
        if inp["dtype"] == np.uint8:
            scale, zero = inp["quantization"]
            data = (batch / (scale or 1.0) + zero).astype(np.uint8)
        else:
            data = batch.astype(inp["dtype"])
        self._interpreter.set_tensor(inp["index"], data)
        self._interpreter.invoke()
        out = self._output_details[0]
        raw = self._interpreter.get_tensor(out["index"])
        if out["dtype"] == np.uint8:
            scale, zero = out["quantization"]
            raw = (raw.astype(np.float32) - zero) * (scale or 1.0)
        return self._to_suspicion(np.asarray(raw))

    def _infer_keras(self, batch: np.ndarray) -> float:
        raw = self._keras.predict(batch, verbose=0)
        return self._to_suspicion(np.asarray(raw))

    def _infer_torch(self, batch: np.ndarray) -> float:
        """Score one NHWC [0,1] batch with the ResNet checkpoint.

        `_prepare` produces what the TFLite path wants: NHWC, scaled to
        [0,1], unnormalised. The torch model was trained on ImageNet-
        normalised NCHW input, so both conversions happen here rather than
        changing `_prepare` and disturbing the other backends.
        """
        import torch

        arr = np.ascontiguousarray(batch.transpose(0, 3, 1, 2))  # NHWC -> NCHW
        mean = np.asarray(self.cfg.torch_mean, dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.asarray(self.cfg.torch_std, dtype=np.float32).reshape(1, 3, 1, 1)
        arr = (arr - mean) / std

        with torch.no_grad():
            logits = self._torch(torch.from_numpy(arr.astype(np.float32)))
            probs = torch.softmax(logits, dim=1)[0]
        return float(np.clip(probs[self._torch_suspicious_index].item(), 0.0, 1.0))

    def _to_suspicion(self, raw: np.ndarray) -> float:
        """Normalise model output to P(suspicious).

        Handles both a single sigmoid unit and a 2-class softmax head.
        """
        flat = raw.reshape(-1)
        if flat.size == 1:
            return float(np.clip(flat[0], 0.0, 1.0))
        probs = flat[: len(self.cfg.labels)].astype(np.float64)
        if probs.sum() > 1.01 or probs.min() < 0:  # logits
            e = np.exp(probs - probs.max())
            probs = e / e.sum()
        return float(np.clip(probs[1], 0.0, 1.0))

    # -- public ----------------------------------------------------------

    def predict(
        self,
        cnn_image: np.ndarray,
        colour_image: np.ndarray | None = None,
        product_group: str | None = None,
    ) -> VisualResult:
        """Score one package image.

        `product_group` is the product name read by OCR or resolved from the
        database. When a calibration exists for that product, the pack is
        compared against reference images of the same product, which is far
        more discriminative than comparing it against packaging in general.
        """
        notes: list[str] = []

        if self.backend in ("tflite", "keras", "torch"):
            batch = self._prepare(cnn_image)
            try:
                if self.backend == "tflite":
                    score = self._infer_tflite(batch)
                elif self.backend == "keras":
                    score = self._infer_keras(batch)
                else:
                    score = self._infer_torch(batch)
                label = (
                    "suspicious"
                    if score >= self.cfg.suspicion_threshold
                    else "genuine"
                )
                return VisualResult(score, self.backend, label, True, {}, {}, notes)
            except Exception as exc:
                log.error("Model inference failed: %s", exc)
                notes.append(f"Model inference failed ({exc}); fell back.")

        # Heuristic path.
        source = colour_image
        if source is None:
            source = (np.clip(cnn_image, 0, 1) * 255).astype(np.uint8)
            source = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)

        cues = measure_cues(source)

        calib, scope = (
            self.calibration.get(product_group)
            if self.calibration
            else (None, "none")
        )

        if calib is None:
            # Abstain rather than guess. Fusion excludes this stream.
            have = self.calibration.total_samples if self.calibration else 0
            notes.append(
                "Visual authentication unavailable: no trained CNN and no "
                f"usable reference calibration ({have} reference image(s); "
                f"{MIN_CALIBRATION_SAMPLES} needed). The visual check did not "
                "contribute to this verdict."
            )
            return VisualResult(
                suspicion_score=0.5,
                backend="heuristic",
                label="unknown",
                usable=False,
                cues=cues,
                deviations={},
                notes=notes,
            )

        score, deviations = score_against(cues, calib)
        label = "suspicious" if score >= self.cfg.suspicion_threshold else "genuine"

        worst = sorted(deviations.items(), key=lambda kv: -abs(kv[1]))[:3]
        if scope == "group":
            scope_note = (
                f"compared against {calib.n_samples} reference image(s) of "
                f"{product_group}"
            )
        else:
            scope_note = (
                f"compared against a pooled reference set of "
                f"{calib.n_samples} image(s); no per-product reference exists "
                f"for {product_group or 'this product'}, so this score is "
                "weaker than a like-for-like comparison"
            )

        notes.append(
            f"Calibrated heuristic (not a CNN), {scope_note}. "
            "Largest deviations: "
            + ", ".join(f"{n} {z:+.1f} sigma" for n, z in worst)
        )

        return VisualResult(
            suspicion_score=score,
            backend="heuristic",
            label=label,
            usable=True,
            cues=cues,
            deviations=deviations,
            notes=notes,
        )


# Module-level singleton so the interpreter loads once per process.
_AUTHENTICATOR: VisualAuthenticator | None = None


def get_authenticator(cfg: CNNConfig | None = None) -> VisualAuthenticator:
    global _AUTHENTICATOR
    if _AUTHENTICATOR is None:
        _AUTHENTICATOR = VisualAuthenticator(cfg)
    return _AUTHENTICATOR


def reset_authenticator() -> None:
    """Force a reload, e.g. after training or calibration writes new files."""
    global _AUTHENTICATOR
    _AUTHENTICATOR = None
