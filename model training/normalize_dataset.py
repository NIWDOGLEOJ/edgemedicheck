#!/usr/bin/env python3
"""
Normalise a dataset so the classes cannot be told apart by encoding.

Why
---
`audit_dataset.py` found that this dataset's classes were separable at 98.6%
by a single threshold on bytes-per-pixel -- because every Real image was a
small JPEG and every Fake image a larger PNG. A network exploits that in the
first epoch and reports ~99% accuracy while learning nothing about medicine
packaging.

This script removes that particular shortcut by re-encoding every image
identically: same format, same JPEG quality, same longest edge. After running
it, re-run the audit. What you are looking for is the shortcut test dropping
towards chance.

What this does NOT fix
----------------------
Encoding is only the most obvious leak. If the two classes contain different
*products* -- Paracetamol in one and Loperamide in the other -- the model can
still separate them by recognising the drug, and no amount of re-encoding
touches that. Normalisation is a necessary step, not a sufficient one, and the
audit will tell you which of the two you are still facing.

Downsampling everything to a common size also destroys real information: the
small images cannot be un-compressed back to detail they never had. Treat the
output as a diagnostic, not as a better dataset.

Usage
-----
    python normalize_dataset.py dataset --out dataset_norm
    python normalize_dataset.py dataset --out dataset_norm --size 224 --quality 90
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    print("Needs opencv: pip install opencv-python", file=sys.stderr)
    raise SystemExit(1)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def target_size(h: int, w: int, longest: int, mode: str) -> tuple[int, int]:
    """Compute output dimensions."""
    if mode == "square":
        return longest, longest
    scale = longest / max(h, w)
    return max(1, int(round(w * scale))), max(1, int(round(h * scale)))


def normalise_one(src: Path, dst: Path, longest: int, quality: int,
                  mode: str) -> tuple[bool, str]:
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        return False, "unreadable"

    h, w = img.shape[:2]
    new_w, new_h = target_size(h, w, longest, mode)

    # INTER_AREA for shrinking, INTER_CUBIC for enlarging. Using one for both
    # leaves a resampling signature that itself correlates with original size,
    # which would reintroduce a shortcut through the back door.
    interp = cv2.INTER_AREA if (new_w * new_h) < (w * h) else cv2.INTER_CUBIC
    out = cv2.resize(img, (new_w, new_h), interpolation=interp)

    dst.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dst), out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return (ok, "ok" if ok else "write failed")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("root")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--size", type=int, default=256,
                    help="longest edge of the output (default 256)")
    ap.add_argument("--quality", type=int, default=88,
                    help="JPEG quality applied to EVERY image (default 88)")
    ap.add_argument("--mode", default="fit", choices=["fit", "square"],
                    help="'fit' keeps aspect ratio; 'square' forces NxN, which "
                         "also removes the aspect-ratio shortcut")
    ap.add_argument("--force", action="store_true", help="overwrite --out")
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)
    if not root.is_dir():
        print(f"No such directory: {root}", file=sys.stderr)
        return 1
    if out_root.exists():
        if not args.force:
            print(f"{out_root} exists. Pass --force to overwrite.",
                  file=sys.stderr)
            return 1
        shutil.rmtree(out_root)

    files = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in IMAGE_EXT]
    if not files:
        print(f"No images under {root}", file=sys.stderr)
        return 1

    print(f"Normalising {len(files)} image(s)")
    print(f"  format  : JPEG quality {args.quality} (uniform)")
    print(f"  size    : longest edge {args.size}, mode '{args.mode}'")
    print(f"  output  : {out_root}\n")

    before_bytes_per_px: dict[str, list[float]] = {}
    stats = Counter()

    for src in files:
        rel = src.relative_to(root)
        # Every output is .jpg, so the extension carries no class information.
        dst = out_root / rel.with_suffix(".jpg")

        # Record the pre-normalisation leak for the before/after summary.
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is not None:
            cls = next((p for p in rel.parts if p in ("Real", "Fake",
                                                      "genuine", "suspicious")),
                       "?")
            h, w = img.shape[:2]
            before_bytes_per_px.setdefault(cls, []).append(
                src.stat().st_size / (h * w)
            )

        ok, why = normalise_one(src, dst, args.size, args.quality, args.mode)
        stats["ok" if ok else "failed"] += 1
        if not ok:
            print(f"  skipped {rel}: {why}")

    # Copy any manifest alongside so downstream tooling still finds it.
    for extra in root.rglob("manifest.json"):
        target = out_root / extra.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(extra, target)

    print(f"\n  written: {stats['ok']}   failed: {stats['failed']}")

    # ---- Before / after on the leaking feature ------------------------
    after: dict[str, list[float]] = {}
    for p in out_root.rglob("*.jpg"):
        rel = p.relative_to(out_root)
        cls = next((x for x in rel.parts if x in ("Real", "Fake",
                                                  "genuine", "suspicious")), "?")
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None:
            h, w = img.shape[:2]
            after.setdefault(cls, []).append(p.stat().st_size / (h * w))

    classes = [c for c in before_bytes_per_px if c != "?"]
    if len(classes) >= 2:
        print(f"\n  bytes/pixel, the feature that was leaking:")
        print(f"    {'class':<12} {'before':>10} {'after':>10}")
        for cls in classes:
            b = float(np.median(before_bytes_per_px[cls]))
            a = float(np.median(after.get(cls, [0])))
            print(f"    {cls:<12} {b:>10.2f} {a:>10.2f}")

        b_gap = abs(np.median(before_bytes_per_px[classes[0]])
                    - np.median(before_bytes_per_px[classes[1]]))
        a_gap = abs(np.median(after.get(classes[0], [0]))
                    - np.median(after.get(classes[1], [0])))
        print(f"\n    gap between classes: {b_gap:.2f} -> {a_gap:.2f}")

    print(f"\nNow re-run the audit on the normalised copy:")
    print(f"    python audit_dataset.py {out_root}")
    print("\nIf the shortcut test drops near chance, encoding was the whole")
    print("leak. If it stays high, the classes differ in content -- most often")
    print("because they contain different products -- and re-encoding cannot")
    print("fix that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
