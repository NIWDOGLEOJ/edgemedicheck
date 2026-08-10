#!/usr/bin/env python3
"""
Falsification test for the trained counterfeit classifier.

Why this exists
---------------
`output/evaluation_results.json` reports 1.000 accuracy, 1.000 precision,
1.000 recall and 1.000 ROC-AUC, with a perfectly clean confusion matrix.

A perfect score on a hard, real-world problem is not evidence that the model
works. It is evidence that the task, as posed by the data, was easy for the
wrong reason. `audit_dataset.py` already showed why: a single threshold on
bytes-per-pixel separates the two classes at 98.6%, because every Real image
is a small JPEG and every Fake image a larger PNG. The network almost
certainly learned that, not the packaging.

This script tests that hypothesis directly instead of arguing about it. The
model is evaluated on progressively degraded versions of the SAME test images:

    A  original             baseline, expect ~100%
    B  re-encoded           identical content, uniform JPEG + size.
                            If accuracy holds, encoding was not the cue.
                            If it collapses, encoding WAS the cue.
    C  heavy blur           packaging content destroyed, encoding uniform.
                            Anything above chance here cannot come from
                            reading the label.
    D  grey blocks          all image content replaced by its mean colour.
                            Only overall brightness survives.

Interpretation
--------------
A high and B low        -> the model learned file encoding. Result invalid.
A, B, C all high        -> the model learned something even coarser, such as
                           overall brightness or aspect ratio.
A high, B high, C low   -> the model is genuinely using image content. That is
                           the outcome you want, and the only one that
                           supports reporting an accuracy figure.

Usage
-----
    source venv/bin/activate
    python falsify_model.py
    python falsify_model.py --model output/best_counterfeit_model.pth
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from PIL import Image, ImageFilter
    from torchvision import models, transforms
except ImportError as exc:
    print(f"Missing dependency: {exc}\nRun inside your venv.", file=sys.stderr)
    raise SystemExit(1)

HERE = Path(__file__).parent
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Must match train_model.py exactly, or the comparison is meaningless.
EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------
# Degradations
# --------------------------------------------------------------------------


def variant_original(img: Image.Image) -> Image.Image:
    return img


def variant_reencode(img: Image.Image) -> Image.Image:
    """Uniform JPEG quality and size. Content preserved, encoding erased."""
    img = img.convert("RGB").resize((256, 256), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def variant_blur(img: Image.Image) -> Image.Image:
    """Destroy readable packaging content; keep colour and layout mass."""
    img = variant_reencode(img)
    return img.filter(ImageFilter.GaussianBlur(radius=12))


def variant_meancolour(img: Image.Image) -> Image.Image:
    """Replace the entire image with its average colour.

    Nothing survives except overall brightness and colour cast. A model
    scoring above chance here is using neither packaging nor texture.
    """
    img = variant_reencode(img)
    arr = np.asarray(img).astype(np.float32)
    mean = arr.reshape(-1, 3).mean(axis=0)
    flat = np.tile(mean, (256, 256, 1)).astype(np.uint8)
    return Image.fromarray(flat)


VARIANTS = [
    ("A  original", variant_original,
     "baseline -- should reproduce the reported score"),
    ("B  re-encoded", variant_reencode,
     "same content, uniform JPEG + size"),
    ("C  heavy blur", variant_blur,
     "packaging destroyed, encoding uniform"),
    ("D  mean colour", variant_meancolour,
     "only brightness/colour survives"),
]


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def load_model(path: Path, num_classes: int = 2):
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3), nn.Linear(model.fc.in_features, num_classes)
    )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if isinstance(state, dict) and not any(
        k.startswith(("conv1", "fc", "layer")) for k in state
    ):
        for key in ("state_dict", "model", "weights"):
            if key in state:
                state = state[key]
                break
    model.load_state_dict(state)
    return model.to(DEVICE).eval(), ckpt


def gather(test_root: Path, classes: list[str]) -> list[tuple[Path, int]]:
    items = []
    for idx, cls in enumerate(classes):
        d = test_root / cls
        if not d.is_dir():
            print(f"Missing class folder: {d}", file=sys.stderr)
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMAGE_EXT:
                items.append((p, idx))
    return items


@torch.no_grad()
def evaluate(model, items, transform_fn, batch=32) -> dict:
    correct = 0
    per_class = {}
    preds_all, labels_all = [], []

    for start in range(0, len(items), batch):
        chunk = items[start:start + batch]
        tensors, labels = [], []
        for path, label in chunk:
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                continue
            tensors.append(EVAL_TF(transform_fn(img)))
            labels.append(label)
        if not tensors:
            continue

        x = torch.stack(tensors).to(DEVICE)
        out = model(x)
        pred = out.argmax(dim=1).cpu().numpy()
        lab = np.array(labels)

        correct += int((pred == lab).sum())
        preds_all.extend(pred.tolist())
        labels_all.extend(lab.tolist())

    total = len(labels_all)
    for c in set(labels_all):
        mask = np.array(labels_all) == c
        per_class[c] = float(
            (np.array(preds_all)[mask] == c).mean()
        ) if mask.any() else 0.0

    return {
        "accuracy": correct / max(total, 1),
        "n": total,
        "per_class": per_class,
        "pred_distribution": {
            int(c): int((np.array(preds_all) == c).sum())
            for c in set(preds_all)
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", default=str(HERE / "output" / "best_counterfeit_model.pth"))
    ap.add_argument("--test", default=str(HERE / "dataset" / "test"))
    ap.add_argument("--classes", nargs="+", default=["Fake", "Real"])
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"No model at {model_path}", file=sys.stderr)
        return 1

    print(f"\n{'=' * 72}")
    print("FALSIFICATION TEST")
    print(f"{'=' * 72}")
    print(f"  model  : {model_path.name}")
    print(f"  device : {DEVICE}")

    model, ckpt = load_model(model_path, len(args.classes))
    items = gather(Path(args.test), args.classes)
    if not items:
        print("No test images found.", file=sys.stderr)
        return 1

    counts = {c: sum(1 for _, l in items if l == i)
              for i, c in enumerate(args.classes)}
    chance = max(counts.values()) / len(items)
    print(f"  test   : {len(items)} images {counts}")
    print(f"  chance : {chance * 100:.1f}% (always predict the larger class)\n")

    print(f"  {'variant':<16} {'accuracy':>9}   what it isolates")
    print("  " + "-" * 68)

    results = {}
    for name, fn, why in VARIANTS:
        r = evaluate(model, items, fn)
        results[name.strip()] = r
        print(f"  {name:<16} {r['accuracy'] * 100:>8.1f}%   {why}")

    a = results["A  original"]["accuracy"]
    b = results["B  re-encoded"]["accuracy"]
    c = results["C  heavy blur"]["accuracy"]
    d = results["D  mean colour"]["accuracy"]

    print(f"\n{'=' * 72}\nVERDICT\n{'=' * 72}")

    verdict = []
    if a - b > 0.15:
        verdict.append(
            f"Accuracy falls {(a - b) * 100:.0f} points when the images are "
            "merely re-encoded\n  with identical content. The model was keying "
            "on file encoding, not packaging."
        )
    if c > chance + 0.15:
        verdict.append(
            f"Accuracy stays at {c * 100:.0f}% after the packaging is blurred "
            "beyond legibility.\n  Whatever it is using, it is not the printed "
            "label or the print quality."
        )
    if d > chance + 0.15:
        verdict.append(
            f"Accuracy stays at {d * 100:.0f}% when every image is replaced by "
            "a flat block of its\n  average colour. The decision rests on "
            "overall brightness alone."
        )

    if verdict:
        for v in verdict:
            print(f"\n  - {v}")
        print(f"\n  The reported 100% is not a measure of counterfeit detection.")
        print("  Do not report it. Fix the dataset and retrain.")
    else:
        print("\n  The model degrades as content is destroyed and survives "
              "re-encoding.")
        print("  That is consistent with it genuinely using image content.")
        print("  The accuracy figure is defensible, subject to the dataset")
        print("  containing the same products in both classes.")

    out = HERE / "output" / "falsification_results.json"
    out.write_text(json.dumps({
        "model": str(model_path),
        "chance": chance,
        "class_counts": counts,
        "results": {k: v for k, v in results.items()},
    }, indent=2))
    print(f"\n  Written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
