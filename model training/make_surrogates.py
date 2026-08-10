#!/usr/bin/env python3
"""
Derive both dataset classes from one pool of genuine packaging photographs.

The idea
--------
Real counterfeit packaging cannot be obtained legally in quantity, so
COLLECTING.md proposes building physical surrogates from genuine cartons:
reprint the artwork on an inkjet, photograph a pack off a monitor, glue on a
fresh expiry panel, swap the substrate, scuff and soak it. This script performs
the digital equivalents, so the surrogate class can be produced at dataset
scale.

What matters is not the realism of any single effect. It is that **every
difference between the two classes is one we deliberately introduced.** The
previous dataset failed because Real images were web thumbnails and Fake images
were screenshots, so the network separated them on encoding and never looked at
a carton.

The shared-capture rule
-----------------------
A naive version of this script would emit raw pool images as Real and heavily
processed images as Fake. That reintroduces exactly the old bug one level up:
the network would learn "has been through an image pipeline", not "is a
counterfeit". Resampling, added noise and JPEG generations are all trivially
detectable.

So both classes run through the SAME simulated capture stage -- perspective,
lighting gradient, defocus, sensor noise, JPEG generation -- drawn from the
same distribution with the same code path. The counterfeit operation is applied
BEFORE that stage, on the Fake branch only:

    genuine pack photo
       |
       +-- Real:  ------------------------> capture_sim() --> encode
       |
       +-- Fake:  counterfeit_op() -------> capture_sim() --> encode

After this, the only systematic difference between the classes is the
counterfeit operation itself. `audit_dataset.py` is the check on that claim,
and its shortcut test should land near chance.

Grouping
--------
Every output filename carries its source pack ID. `split_grouped.py` keeps all
variants of one pack -- both classes -- inside a single fold, so the model
cannot memorise a carton and score on recognising it again.

Usage
-----
    python make_surrogates.py pool/genuine --out dataset_v2
    python make_surrogates.py pool/genuine --out dataset_v2 --per-class 2
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WORK = 512          # working resolution for the effect chain
OUT_SIZE = 256      # final edge, matching the rest of the pipeline
JPEG_Q = 88         # uniform final encode for BOTH classes

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
]


def load_font(size: int):
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return ImageFont.load_default()


# ------------------------------------------------------- counterfeit ops
# Each takes and returns float32 BGR in [0, 255].

def op_reprint(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Artwork rescanned and run off on a low-end inkjet.

    Three things happen on a cheap reprint, and all three are visible:
    the continuous-tone original is screened into halftone dots, the dots
    spread on contact with the paper so midtones darken (dot gain), and
    saturated brand colours fall outside the printer's gamut and get clipped.
    """
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Gamut compression: pull saturation down above a knee, since the
    # vivid brand colours are the ones a 4-ink printer cannot reach.
    hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    knee = 110.0
    s = np.where(s > knee, knee + (s - knee) * 0.55, s)
    hsv[:, :, 1] = np.clip(s, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)

    # Halftone. Screen angles are the traditional ones -- 15/75/0 degrees --
    # which is what produces a rosette rather than a plain grid moire.
    freq = rng.uniform(0.30, 0.46)          # cycles/px at working resolution
    angles = {0: 15.0, 1: 75.0, 2: 0.0}     # B, G, R standing in for C, M, Y
    ink = 1.0 - out / 255.0                  # work in ink coverage
    screened = np.empty_like(ink)
    for c in range(3):
        th = math.radians(angles[c] + rng.uniform(-2, 2))
        u = xx * math.cos(th) + yy * math.sin(th)
        v = -xx * math.sin(th) + yy * math.cos(th)
        screen = (np.sin(2 * math.pi * freq * u) *
                  np.sin(2 * math.pi * freq * v))
        screen = (screen + 1.0) / 2.0        # 0..1 dot profile
        # Dot gain: ink spreads, so coverage is higher than requested.
        gained = np.clip(ink[:, :, c], 0, 1) ** (1.0 / rng.uniform(1.12, 1.30))
        screened[:, :, c] = (gained > screen).astype(np.float32)

    # The press lays down hard dots; the camera that photographs the reprint
    # does not resolve them individually.
    screened = cv2.GaussianBlur(screened, (0, 0), rng.uniform(0.7, 1.15))
    out = (1.0 - screened) * 255.0

    # Registration error: the plates do not line up perfectly.
    for c in (0, 1):
        dx, dy = rng.randint(-1, 1), rng.randint(-1, 1)
        if dx or dy:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            out[:, :, c] = cv2.warpAffine(out[:, :, c], M, (w, h),
                                          borderMode=cv2.BORDER_REPLICATE)

    # Paper is never the same white as the original substrate.
    cast = np.array([rng.uniform(-6, 2), rng.uniform(-3, 3),
                     rng.uniform(-2, 8)], dtype=np.float32)
    return np.clip(out + cast, 0, 255)


def op_recapture(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """A genuine pack photographed off a monitor.

    The giveaways are the display's RGB stripe beating against the sensor
    grid, the backlight's uneven field, and the lifted black point that no
    LCD escapes.
    """
    h, w = img.shape[:2]
    out = img.copy()

    # LCD subpixel stripe, at a frequency close to the sampling limit so it
    # aliases into visible moire rather than staying a clean grid.
    period = rng.uniform(2.6, 3.7)
    theta = math.radians(rng.uniform(-4, 4))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = xx * math.cos(theta) + yy * math.sin(theta)
    for c in range(3):
        phase = 2 * math.pi * (u / period + c / 3.0)
        out[:, :, c] *= (1.0 + rng.uniform(0.05, 0.13) * np.sin(phase))

    # Horizontal scan structure, much fainter than the stripe.
    out *= (1.0 + 0.03 * np.sin(2 * math.pi * yy / rng.uniform(2.4, 3.4)))[..., None]

    # Backlight: brighter in the middle, falling off at the edges.
    cx, cy = w * rng.uniform(0.4, 0.6), h * rng.uniform(0.4, 0.6)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (0.75 * math.hypot(w, h))
    out *= (1.0 + rng.uniform(0.10, 0.22) * (1.0 - r))[..., None]

    # Lifted blacks and slightly compressed contrast.
    lift = rng.uniform(8, 18)
    out = out * rng.uniform(0.86, 0.94) + lift
    return np.clip(out, 0, 255)


def op_relabel(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """A fresh batch/expiry panel printed and stuck over the original.

    This is the cheapest real-world tamper: the carton is genuine, only the
    dates have been renewed. The patch reads as a rectangle of slightly wrong
    white, with its own edge shadow and its own typeface.
    """
    h, w = img.shape[:2]
    pil = Image.fromarray(
        cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB))

    pw, ph = int(w * rng.uniform(0.34, 0.52)), int(h * rng.uniform(0.09, 0.15))
    px = rng.randint(int(w * 0.05), max(int(w * 0.05) + 1, w - pw - int(w * 0.05)))
    py = rng.randint(int(h * 0.55), max(int(h * 0.55) + 1, h - ph - int(h * 0.04)))

    # The replacement label's paper is close to, but not, the carton's white.
    base = rng.randint(232, 252)
    patch = Image.new("RGB", (pw, ph),
                      (base, base - rng.randint(0, 5), base - rng.randint(0, 9)))
    d = ImageDraw.Draw(patch)
    font = load_font(max(9, int(ph * 0.27)))
    mfg = rng.randint(1, 12), rng.randint(2023, 2025)
    exp = rng.randint(1, 12), mfg[1] + rng.randint(2, 4)
    d.text((int(pw * 0.04), int(ph * 0.06)),
           f"B.No. {rng.choice('ABCDEFGHJKLMNPQ')}{rng.randint(100,999)}"
           f"{rng.choice('ABCDEFGHJKLMNPQ')}",
           fill=(30, 30, 34), font=font)
    d.text((int(pw * 0.04), int(ph * 0.38)),
           f"MFG {mfg[0]:02d}/{mfg[1]}", fill=(30, 30, 34), font=font)
    d.text((int(pw * 0.04), int(ph * 0.68)),
           f"EXP {exp[0]:02d}/{exp[1]}", fill=(30, 30, 34), font=font)

    patch = patch.rotate(rng.uniform(-2.5, 2.5), resample=Image.BICUBIC,
                         expand=True, fillcolor=(base, base, base))
    pil.paste(patch, (px, py))

    out = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR).astype(np.float32)

    # A stuck-on label sits proud of the surface and casts a thin shadow.
    sh = np.ones((h, w), np.float32)
    y0, y1 = min(py + patch.height, h - 1), min(py + patch.height + 3, h)
    x0, x1 = px, min(px + patch.width, w)
    if y1 > y0 and x1 > x0:
        sh[y0:y1, x0:x1] = rng.uniform(0.78, 0.9)
    sh = cv2.GaussianBlur(sh, (0, 0), 1.6)
    return np.clip(out * sh[..., None], 0, 255)


def op_substrate(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Correct artwork, wrong board: uncoated stock instead of coated.

    Uncoated paper scatters light, so the fibre shows, the surface loses its
    sheen and the ink sinks slightly duller.
    """
    h, w = img.shape[:2]
    # Correlated noise reads as fibre; white noise reads as sensor grain.
    fib = np.random.default_rng(rng.randrange(1 << 30)).normal(0, 1, (h, w))
    fib = cv2.GaussianBlur(fib.astype(np.float32), (0, 0), rng.uniform(0.6, 1.3))
    fib = fib / (fib.std() + 1e-6) * rng.uniform(5, 11)

    out = img + fib[..., None]
    # Uncoated stock cannot hold density: highlights dull, saturation drops.
    out = out * rng.uniform(0.93, 0.98) + rng.uniform(6, 14)
    hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) *
                           rng.uniform(0.80, 0.92), 0, 255).astype(np.uint8)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)
    # Warm cast of unbleached board.
    return np.clip(out + np.array([-4, 1, 6], np.float32), 0, 255)


def op_damage(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Soaked, scuffed, peeled and re-stuck -- a pack that has been opened."""
    h, w = img.shape[:2]
    out = img.copy()

    # Water staining: a few soft dark blooms.
    stain = np.zeros((h, w), np.float32)
    for _ in range(rng.randint(1, 3)):
        cx, cy = rng.randint(0, w - 1), rng.randint(0, h - 1)
        rad = rng.randint(int(w * 0.10), int(w * 0.28))
        cv2.circle(stain, (cx, cy), rad, rng.uniform(0.10, 0.26), -1)
    stain = cv2.GaussianBlur(stain, (0, 0), rng.uniform(9, 18))
    out *= (1.0 - stain)[..., None]

    # Scuffing: bright abraded streaks where the coating has lifted.
    scuff = np.zeros((h, w), np.float32)
    for _ in range(rng.randint(4, 11)):
        x0, y0 = rng.randint(0, w - 1), rng.randint(0, h - 1)
        ln = rng.randint(int(w * 0.06), int(w * 0.24))
        a = rng.uniform(0, math.pi)
        cv2.line(scuff, (x0, y0),
                 (int(x0 + ln * math.cos(a)), int(y0 + ln * math.sin(a))),
                 rng.uniform(0.15, 0.42), rng.randint(1, 2))
    scuff = cv2.GaussianBlur(scuff, (0, 0), 0.8)
    out += (scuff * 90)[..., None]

    # A corner torn away, showing board underneath.
    if rng.random() < 0.55:
        cs = rng.randint(int(w * 0.08), int(w * 0.18))
        corner = rng.choice([(0, 0), (w, 0), (0, h), (w, h)])
        pts = np.array([[corner,
                         (corner[0] + (cs if corner[0] == 0 else -cs), corner[1]),
                         (corner[0], corner[1] + (cs if corner[1] == 0 else -cs))]],
                       np.int32)
        cv2.fillPoly(out, pts, (208, 212, 214))
    return np.clip(out, 0, 255)


COUNTERFEIT_OPS = {
    "reprint": op_reprint,
    "recapture": op_recapture,
    "relabel": op_relabel,
    "substrate": op_substrate,
    "damage": op_damage,
}


# ------------------------------------------------------- shared capture

def capture_sim(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """The scanner's camera looking at a pack on the mat.

    Applied identically to both classes. Every parameter here is drawn from
    the same distribution regardless of label -- that is the whole point.
    """
    h, w = img.shape[:2]

    # Small off-axis view: the pack is never perfectly square to the lens.
    m = 0.045
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[rng.uniform(-m, m) * w, rng.uniform(-m, m) * h],
                      [w + rng.uniform(-m, m) * w, rng.uniform(-m, m) * h],
                      [w + rng.uniform(-m, m) * w, h + rng.uniform(-m, m) * h],
                      [rng.uniform(-m, m) * w, h + rng.uniform(-m, m) * h]])
    out = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    # Uneven illumination: a tilted plane plus a little vignetting.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    plane = (1.0 + rng.uniform(-0.16, 0.16) * (xx / w - 0.5)
             + rng.uniform(-0.16, 0.16) * (yy / h - 0.5))
    r = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (0.5 * math.hypot(w, h))
    out = out * (plane * (1.0 - rng.uniform(0.05, 0.20) * r ** 2))[..., None]

    # Exposure and white balance drift.
    out = out * rng.uniform(0.90, 1.10) + rng.uniform(-8, 8)
    out *= np.array([rng.uniform(0.96, 1.04), 1.0, rng.uniform(0.96, 1.04)],
                    np.float32)

    # Focus is never perfect at close range.
    if rng.random() < 0.75:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.4, 1.1))

    # Sensor noise, stronger in chroma than luma as in a real CFA pipeline.
    g = np.random.default_rng(rng.randrange(1 << 30))
    out += g.normal(0, rng.uniform(1.5, 4.5), out.shape).astype(np.float32)
    return np.clip(out, 0, 255)


def encode_uniform(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """One JPEG generation at a quality drawn from a shared band.

    Both classes get this. It is here so that neither class is the only one
    carrying compression artefacts.
    """
    q = rng.randint(72, 94)
    ok, buf = cv2.imencode(".jpg", np.clip(img, 0, 255).astype(np.uint8),
                           [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return np.clip(img, 0, 255).astype(np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ----------------------------------------------------------------- main

def build_one(src: np.ndarray, label: str, rng: random.Random):
    """Run one image down one branch. Returns (image, op_name)."""
    work = cv2.resize(src, (WORK, WORK), interpolation=cv2.INTER_AREA)
    op_name = "none"
    if label == "Fake":
        op_name = rng.choice(sorted(COUNTERFEIT_OPS))
        work = COUNTERFEIT_OPS[op_name](work.astype(np.float32), rng)
    work = capture_sim(work.astype(np.float32), rng)
    work = encode_uniform(work, rng)
    out = cv2.resize(work, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)
    return out, op_name


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pool", help="directory of genuine source images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=2,
                    help="variants per source pack per class (default 2)")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--limit", type=int, default=0, help="cap packs, for testing")
    args = ap.parse_args()

    pool = Path(args.pool)
    srcs = sorted(p for p in pool.iterdir()
                  if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if args.limit:
        srcs = srcs[:args.limit]
    if not srcs:
        print(f"No images in {pool}", file=sys.stderr)
        return 1

    out_root = Path(args.out)
    for cls in ("Real", "Fake"):
        (out_root / cls).mkdir(parents=True, exist_ok=True)

    print(f"{len(srcs)} source packs -> {args.per_class} Real + "
          f"{args.per_class} Fake each "
          f"= {len(srcs) * args.per_class * 2} images\n")

    manifest = (out_root / "surrogates.jsonl").open("w")
    stats: Counter = Counter()

    for n, sp in enumerate(srcs, 1):
        img = cv2.imread(str(sp), cv2.IMREAD_COLOR)
        if img is None:
            stats["unreadable"] += 1
            continue
        pack_id = sp.stem                      # the group key for splitting

        for cls in ("Real", "Fake"):
            for k in range(args.per_class):
                # Seeded per (pack, class, variant): reproducible, and the
                # capture parameters are drawn from the same generator for
                # both classes.
                rng = random.Random(f"{args.seed}|{pack_id}|{cls}|{k}")
                out, op = build_one(img, cls, rng)
                name = f"{pack_id}__{cls.lower()}{k}.jpg"
                cv2.imwrite(str(out_root / cls / name), out,
                            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q])
                manifest.write(json.dumps({
                    "file": f"{cls}/{name}", "pack_id": pack_id,
                    "cls": cls, "variant": k, "op": op,
                    "source": sp.name,
                }) + "\n")
                stats[f"{cls}:{op}"] += 1

        if n % 200 == 0:
            print(f"  {n}/{len(srcs)} packs", flush=True)

    manifest.close()
    total = sum(stats.values())
    print(f"\n{'=' * 60}\n{total} images from {len(srcs)} packs -> {out_root}\n"
          f"{'=' * 60}")
    for k, v in sorted(stats.items()):
        print(f"  {k:22s} {v}")
    print(f"\nNext:\n  python split_grouped.py {out_root} --out dataset_v2_split")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
