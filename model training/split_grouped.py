#!/usr/bin/env python3
"""
Split by pack, not by file.

Every output of `make_surrogates.py` carries the ID of the carton it came
from. A carton contributes several images and appears in BOTH classes -- its
genuine variants and the surrogates derived from it. If those land in
different folds, the network can score on the test set by recognising the
carton it already memorised in training, which is the "carton #3" shortcut
COLLECTING.md warns about.

So the unit of splitting here is the pack. All of a pack's files, in both
classes, go to exactly one fold.

Usage
-----
    python split_grouped.py dataset_v2 --out dataset_v2_split
    python split_grouped.py dataset_v2 --out dataset_v2_split --ratios 0.7 0.15 0.15
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="directory produced by make_surrogates.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.70, 0.15, 0.15],
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    manifest = root / "surrogates.jsonl"
    if not manifest.exists():
        print(f"No surrogates.jsonl in {root}", file=sys.stderr)
        return 1

    recs = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    by_pack: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_pack[r["pack_id"]].append(r)

    packs = sorted(by_pack)
    random.Random(args.seed).shuffle(packs)

    tr, va, te = args.ratios
    if abs(tr + va + te - 1.0) > 1e-6:
        print("Ratios must sum to 1.", file=sys.stderr)
        return 1
    n = len(packs)
    n_tr, n_va = int(n * tr), int(n * va)
    folds = {
        "train": packs[:n_tr],
        "val": packs[n_tr:n_tr + n_va],
        "test": packs[n_tr + n_va:],
    }

    out_root = Path(args.out)
    if out_root.exists():
        if not args.force:
            print(f"{out_root} exists. Pass --force.", file=sys.stderr)
            return 1
        shutil.rmtree(out_root)

    counts: Counter = Counter()
    for fold, plist in folds.items():
        for pack in plist:
            for r in by_pack[pack]:
                dst = out_root / fold / r["cls"] / Path(r["file"]).name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / r["file"], dst)
                counts[(fold, r["cls"])] += 1

    print(f"{n} packs -> {len(recs)} files\n")
    print(f"{'fold':<8}{'packs':>8}{'Real':>8}{'Fake':>8}{'total':>8}")
    print("-" * 40)
    for fold in ("train", "val", "test"):
        rl, fk = counts[(fold, "Real")], counts[(fold, "Fake")]
        print(f"{fold:<8}{len(folds[fold]):>8}{rl:>8}{fk:>8}{rl + fk:>8}")

    # The invariant this script exists to guarantee.
    where = defaultdict(set)
    for fold, plist in folds.items():
        for pack in plist:
            where[pack].add(fold)
    spanning = [p for p, f in where.items() if len(f) > 1]
    print()
    if spanning:
        print(f"  FAIL: {len(spanning)} pack(s) span folds")
        return 2
    print(f"  PASS  no pack spans folds; {n} packs are disjoint across "
          f"train/val/test")
    print(f"\nNext:\n  python audit_dataset.py {out_root} --classes Real Fake")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
