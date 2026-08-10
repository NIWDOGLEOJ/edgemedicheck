#!/usr/bin/env python3
"""
Dataset audit -- run this BEFORE training, and again after adding images.

A binary image classifier will happily reach 99% accuracy by exploiting
something that has nothing to do with the thing you meant to measure. If every
Real image is a JPEG and every Fake image is a PNG, the network learns the
compression format. If Real images are Paracetamol and Fake images are
Loperamide, it learns which drug is in the photo. Both look like success in
the training log.

This script tries to break your dataset the way a lazy network would, and
reports what it finds. The headline test trains a decision stump on features
that contain no packaging semantics at all -- file size, dimensions, brightness,
JPEG block artefacts. If that alone separates your classes, a CNN certainly
will, and your accuracy number means nothing.

Usage
-----
    python audit_dataset.py dataset
    python audit_dataset.py dataset --classes Real Fake
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import re
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    print("Needs opencv: pip install opencv-python", file=sys.stderr)
    raise SystemExit(1)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
AUG_SUFFIX = re.compile(r"_aug_[a-z]+$", re.IGNORECASE)

RED = "\033[91m"; YEL = "\033[93m"; GRN = "\033[92m"; DIM = "\033[90m"; OFF = "\033[0m"
if not sys.stdout.isatty():
    RED = YEL = GRN = DIM = OFF = ""


def source_id(path: Path) -> str:
    """Identity of the ORIGINAL image, ignoring augmentation suffixes."""
    return AUG_SUFFIX.sub("", path.stem)


def collect(root: Path, classes: list[str]) -> list[dict]:
    """Find every image, recording class and fold."""
    out = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXT:
            continue
        parts = [p for p in path.relative_to(root).parts]
        cls = next((p for p in parts if p in classes), None)
        if cls is None:
            continue
        fold = next((p for p in parts if p.lower() in ("train", "val", "test")),
                    "unsplit")
        out.append({
            "path": path, "cls": cls, "fold": fold,
            "src": source_id(path), "is_aug": bool(AUG_SUFFIX.search(path.stem)),
        })
    return out


def naive_features(path: Path) -> list[float] | None:
    """Features with zero packaging semantics. A model should NOT be able to
    use these to tell genuine from counterfeit."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # JPEG quantises in 8x8 blocks and leaves seams at those boundaries.
    # PNG is lossless and leaves none, so this separates the two formats.
    col = np.abs(np.diff(gray, axis=1))
    row = np.abs(np.diff(gray, axis=0))
    blockiness = (col[:, 7::8].mean() / (col.mean() + 1e-6)
                  + row[7::8, :].mean() / (row.mean() + 1e-6))

    b, g, r = img[:, :, 0].mean(), img[:, :, 1].mean(), img[:, :, 2].mean()
    return [
        h, w, w / max(h, 1),
        path.stat().st_size / (h * w),      # bytes per pixel = format proxy
        gray.mean(), gray.std(), blockiness,
        r - b,                              # colour cast
    ]


FEATURE_NAMES = ["height", "width", "aspect", "bytes/pixel",
                 "brightness", "contrast", "jpeg_blockiness", "colour_cast"]


def best_stump(X: np.ndarray, y: np.ndarray) -> tuple[float, str, float]:
    """Best single-threshold classifier over the naive features."""
    best = (0.0, "", 0.0)
    for i, name in enumerate(FEATURE_NAMES):
        v = X[:, i]
        for t in np.percentile(v, np.arange(2, 99, 2)):
            for sign in (1, -1):
                pred = (v > t) if sign > 0 else (v <= t)
                acc = (pred.astype(int) == y).mean()
                if acc > best[0]:
                    best = (acc, name, float(t))
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--classes", nargs="+", default=["Real", "Fake"])
    ap.add_argument("--sample", type=int, default=500,
                    help="images per class for the shortcut test")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"No such directory: {root}", file=sys.stderr)
        return 1

    items = collect(root, args.classes)
    if not items:
        print(f"No images found under {root} in classes {args.classes}",
              file=sys.stderr)
        return 1

    problems: list[str] = []

    # ---- 1. Inventory -------------------------------------------------
    print(f"\n{'=' * 70}\nDATASET AUDIT: {root}\n{'=' * 70}")
    by = collections.Counter((i["fold"], i["cls"]) for i in items)
    folds = sorted({i["fold"] for i in items})
    print(f"\n{'fold':<10} " + " ".join(f"{c:>10}" for c in args.classes)
          + f"{'sources':>10}")
    print("-" * 52)
    for fold in folds:
        srcs = len({i["src"] for i in items if i["fold"] == fold})
        counts = " ".join(f"{by[(fold, c)]:>10}" for c in args.classes)
        print(f"{fold:<10} {counts} {srcs:>10}")
    total_src = len({i["src"] for i in items})
    print(f"\n  {len(items)} files from {total_src} distinct source images "
          f"({len(items)/max(total_src,1):.1f} per source)")

    # ---- 2. Fold leakage ----------------------------------------------
    print(f"\n{'-' * 70}\n1. FOLD LEAKAGE\n{'-' * 70}")
    where = collections.defaultdict(set)
    for i in items:
        if i["fold"] != "unsplit":
            where[(i["cls"], i["src"])].add(i["fold"])
    leaked = {k: v for k, v in where.items() if len(v) > 1}
    if leaked:
        print(f"{RED}  FAIL: {len(leaked)} source image(s) appear in more than "
              f"one fold.{OFF}")
        for k, v in list(leaked.items())[:5]:
            print(f"    {k[0]} / {k[1][:50]} -> {sorted(v)}")
        problems.append("Source images span multiple folds -- metrics inflated.")
    else:
        print(f"{GRN}  PASS{OFF}  no source image spans folds "
              "(augmented copies stay with their original).")

    # ---- 3. Augmented copies in val/test -------------------------------
    print(f"\n{'-' * 70}\n2. AUGMENTED COPIES IN VAL/TEST\n{'-' * 70}")
    bad = [f for f in ("val", "test")
           if any(i["is_aug"] for i in items if i["fold"] == f)]
    if bad:
        print(f"{YEL}  WARN: augmented copies present in {', '.join(bad)}.{OFF}")
        print("    Evaluation should use real images only, or the score "
              "measures\n    robustness to your own augmentation recipe.")
        problems.append("Augmented images in evaluation folds.")
    else:
        print(f"{GRN}  PASS{OFF}  evaluation folds contain original images only.")

    # ---- 4. Format / filename shortcut ---------------------------------
    print(f"\n{'-' * 70}\n3. FILE FORMAT AND NAMING BY CLASS\n{'-' * 70}")
    fmt = collections.defaultdict(collections.Counter)
    for i in items:
        fmt[i["cls"]][i["path"].suffix.lower()] += 1
    format_split = False
    for cls in args.classes:
        d = dict(fmt[cls])
        print(f"  {cls:<8} {d}")
        if len(d) == 1:
            format_split = True
    sets = [set(fmt[c]) for c in args.classes if fmt[c]]
    if format_split and len(sets) > 1 and not set.intersection(*sets):
        print(f"\n{RED}  FAIL: each class uses a DIFFERENT file format.{OFF}")
        print("    A network can separate the classes on compression artefacts")
        print("    alone, without looking at the packaging.")
        problems.append("Classes are perfectly separated by file format.")
    else:
        print(f"\n{GRN}  PASS{OFF}  formats overlap across classes.")

    # ---- 5. Exact duplicates across classes ----------------------------
    print(f"\n{'-' * 70}\n4. DUPLICATE IMAGES\n{'-' * 70}")
    digest = collections.defaultdict(set)
    for i in items:
        try:
            h = hashlib.md5(i["path"].read_bytes()).hexdigest()
        except OSError:
            continue
        digest[h].add(i["cls"])
    cross = [h for h, c in digest.items() if len(c) > 1]
    dupes = sum(1 for h, c in digest.items() if len(c) == 1)
    print(f"  {len(digest)} unique file hashes across {len(items)} files")
    if cross:
        print(f"{RED}  FAIL: {len(cross)} identical image(s) labelled as BOTH "
              f"classes.{OFF}")
        problems.append("Identical images carry contradictory labels.")
    else:
        print(f"{GRN}  PASS{OFF}  no image appears in two classes.")

    # ---- 6. THE SHORTCUT TEST ------------------------------------------
    print(f"\n{'-' * 70}\n5. SHORTCUT TEST  (the important one)\n{'-' * 70}")
    print(f"{DIM}  Training a one-threshold classifier on features that contain")
    print(f"  no packaging information. It should perform near chance.{OFF}\n")

    X, y, kept = [], [], collections.Counter()
    for ci, cls in enumerate(args.classes[:2]):
        pool = [i for i in items if i["cls"] == cls and not i["is_aug"]]
        if len(pool) < 20:
            pool = [i for i in items if i["cls"] == cls]
        for i in pool[: args.sample]:
            f = naive_features(i["path"])
            if f:
                X.append(f); y.append(ci); kept[cls] += 1
    if len(set(y)) < 2:
        print("  Need two classes for this test.")
        return 1

    X = np.array(X); y = np.array(y)
    majority = max((y == 0).mean(), (y == 1).mean())
    acc, feat, thr = best_stump(X, y)

    print(f"  sampled: {dict(kept)}")
    print(f"  chance (majority class): {majority * 100:.1f}%")
    print(f"  best naive feature     : '{feat}'")
    print(f"  accuracy from it alone : {acc * 100:.1f}%\n")

    if acc > 0.90:
        print(f"{RED}  FAIL: the classes are almost perfectly separable without")
        print(f"  looking at the medicine at all.{OFF}")
        print("  Any CNN accuracy from this dataset is measuring this shortcut,")
        print("  not counterfeit detection.")
        problems.append(
            f"Naive feature '{feat}' alone separates the classes at "
            f"{acc*100:.0f}%."
        )
    elif acc > majority + 0.15:
        print(f"{YEL}  WARN: naive features carry real signal "
              f"(+{(acc-majority)*100:.0f} points over chance).{OFF}")
        problems.append(f"Naive feature '{feat}' gives {acc*100:.0f}%.")
    else:
        print(f"{GRN}  PASS{OFF}  naive features are near chance. Classes differ")
        print("  in content, not in capture or encoding.")

    # ---- 7. Dimension distribution --------------------------------------
    print(f"\n{'-' * 70}\n6. IMAGE SIZE BY CLASS\n{'-' * 70}")
    for ci, cls in enumerate(args.classes[:2]):
        sel = X[y == ci]
        if len(sel):
            print(f"  {cls:<8} {sel[:,0].mean():>6.0f} x {sel[:,1].mean():<6.0f} "
                  f"median bytes/px {np.median(sel[:,3]):.2f}")

    # ---- Verdict --------------------------------------------------------
    print(f"\n{'=' * 70}\nVERDICT\n{'=' * 70}")
    if not problems:
        print(f"{GRN}  No blocking issues found. Numbers from this dataset are "
              f"defensible.{OFF}")
        return 0

    print(f"{RED}  {len(problems)} issue(s) that invalidate reported "
          f"accuracy:{OFF}\n")
    for n, p in enumerate(problems, 1):
        print(f"    {n}. {p}")
    print(f"\n{DIM}  Fix these before reporting any number. A model trained on")
    print(f"  a shortcut scores highly and detects nothing in deployment.{OFF}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
