#!/usr/bin/env python3
"""
Which counterfeit cue is the model actually using?

A single test F1 hides the thing worth knowing. The Fake class is built from
five different surrogate operations, and they are not equally hard. Reprinting
rewrites every pixel with halftone texture; relabelling changes one small
rectangle and leaves the rest of the carton untouched. A model can post a
respectable overall score while being blind to the subtle one.

That distinction matters for the deployment claim. Catching reprints is worth
little on its own -- a reprint is the crudest counterfeit there is. Catching
relabels is the harder and more useful capability, because re-dating expired
stock is the common real-world fraud.

This script breaks test-set recall down by operation, so the paper can state
which cues the model learned instead of implying it learned all of them.

Usage
-----
    python analyse_by_op.py --data dataset_v2_split \\
        --manifest dataset_v2/surrogates.jsonl \\
        --model output_v2/best_counterfeit_model.pth
"""

from __future__ import annotations

import argparse
import json
import ssl
from collections import Counter, defaultdict
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")


def build(num_classes: int):
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(m.fc.in_features,
                                                      num_classes))
    return m.to(DEVICE)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="split root (uses test/)")
    ap.add_argument("--manifest", required=True,
                    help="surrogates.jsonl from make_surrogates.py")
    ap.add_argument("--model", required=True, help="checkpoint .pth")
    ap.add_argument("--out", default=None, help="write JSON here")
    args = ap.parse_args()

    op_of = {}
    for line in Path(args.manifest).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        op_of[Path(r["file"]).name] = r["op"]

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_dir = Path(args.data) / "test"
    ds = datasets.ImageFolder(root=test_dir, transform=tf)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    ckpt = torch.load(args.model, map_location=DEVICE)
    class_names = ckpt.get("class_names", ds.classes)
    model = build(len(class_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    fake_i = class_names.index("Fake") if "Fake" in class_names else 0

    preds_all = []
    with torch.no_grad():
        for inputs, _ in loader:
            out = model(inputs.to(DEVICE))
            preds_all.extend(torch.argmax(out, 1).cpu().numpy().tolist())

    hit: Counter = Counter()
    tot: Counter = Counter()
    for (path, label), pred in zip(ds.samples, preds_all):
        name = Path(path).name
        op = op_of.get(name, "unknown")
        key = "genuine" if label != fake_i else op
        tot[key] += 1
        # For genuine the "correct" answer is Real; for a surrogate it is Fake.
        correct = (pred != fake_i) if label != fake_i else (pred == fake_i)
        hit[key] += int(correct)

    print(f"\n{'=' * 62}\nPER-OPERATION TEST RECALL   ({test_dir})\n{'=' * 62}")
    print(f"{'operation':<24}{'n':>6}{'caught':>9}{'recall':>10}")
    print("-" * 62)
    rows = {}
    for key in sorted(tot, key=lambda k: (k == "genuine", k)):
        r = hit[key] / tot[key] if tot[key] else float("nan")
        rows[key] = {"n": tot[key], "correct": hit[key], "recall": r}
        label = "genuine (specificity)" if key == "genuine" else key
        print(f"{label:<24}{tot[key]:>6}{hit[key]:>9}{r * 100:>9.1f}%")

    fakes = {k: v for k, v in rows.items() if k != "genuine"}
    if fakes:
        best = max(fakes, key=lambda k: fakes[k]["recall"])
        worst = min(fakes, key=lambda k: fakes[k]["recall"])
        print(f"\n  strongest cue: {best} "
              f"({fakes[best]['recall'] * 100:.0f}%)")
        print(f"  weakest cue:   {worst} "
              f"({fakes[worst]['recall'] * 100:.0f}%)")
        print("\n  Report these separately. An overall F1 averaged across "
              "operations\n  overstates the model on whichever cue it "
              "handles worst.")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
