"""
Grouped dataset splitting.

The single most common way a small computer-vision study reports accuracy that
does not survive contact with reality is splitting images at random. Eight
photographs of the same physical carton are not eight independent samples --
they share the same print run, the same creases, the same scuff marks. Put
some in train and some in validation and the network can recognise *that
individual carton* rather than the property you meant to measure. Validation
accuracy climbs, real accuracy does not, and nothing in the training log warns
you.

This module makes the correct split the easy one. Two regimes:

    pack     -- no physical pack appears in more than one fold. This measures
                performance on unseen packs of products the model has seen.

    product  -- entire products are held out. This measures whether the model
                generalises to SKUs it was never trained on, which is the
                honest answer to "how many medicines does it actually know?"

Report both. The gap between them is the most informative number the study
produces, because it quantifies the coverage limitation instead of asserting
it.

Effective sample size
---------------------
Throughout, the unit of statistical independence is the *pack*, not the image.
Thirty cartons photographed ten times each is 300 images and roughly 30
samples. `describe()` prints both so the distinction stays visible.
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

log = logging.getLogger(__name__)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Filenames written by `run.py collect`:
#     PRODUCT__BATCH__pNN__SS.jpg
# The double underscore is the field separator; product and batch codes are
# alphanumeric and never contain it.
FILENAME_RE = re.compile(
    r"^(?P<product>[^_]+(?:_[^_]+)*?)__(?P<batch>[^_]+)__p(?P<pack>\d+)__(?P<shot>\d+)",
    re.IGNORECASE,
)


class LeakageError(AssertionError):
    """Raised when a split would place one physical pack in two folds."""


@dataclass(frozen=True)
class Sample:
    """One image, with the grouping keys needed for an honest split."""

    path: Path
    label: str                  # "genuine" | "suspicious"
    product: str
    pack_id: str                # identifies one physical carton or strip
    batch: str | None = None
    source: str = ""            # how identity was determined, for auditing

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "label": self.label,
            "product": self.product,
            "pack_id": self.pack_id,
            "batch": self.batch,
        }


@dataclass
class Split:
    """A train / validation / test partition."""

    train: list[Sample] = field(default_factory=list)
    val: list[Sample] = field(default_factory=list)
    test: list[Sample] = field(default_factory=list)
    regime: str = "pack"
    seed: int = 42
    notes: list[str] = field(default_factory=list)

    @property
    def folds(self) -> dict[str, list[Sample]]:
        return {"train": self.train, "val": self.val, "test": self.test}

    def paths(self, fold: str) -> list[str]:
        return [str(s.path) for s in self.folds[fold]]

    def labels(self, fold: str, classes: Sequence[str]) -> list[int]:
        index = {c: i for i, c in enumerate(classes)}
        return [index[s.label] for s in self.folds[fold]]

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {"regime": self.regime, "seed": self.seed}
        for name, samples in self.folds.items():
            out[name] = {
                "images": len(samples),
                "packs": len({s.pack_id for s in samples}),
                "products": len({s.product for s in samples}),
                "by_label": dict(Counter(s.label for s in samples)),
            }
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "seed": self.seed,
            "stats": self.stats(),
            "notes": self.notes,
            "files": {
                fold: [s.to_dict() for s in samples]
                for fold, samples in self.folds.items()
            },
        }

    def save(self, path: Path | str) -> None:
        """Persist the split so a reported result can be reproduced exactly."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "Split":
        data = json.loads(Path(path).read_text())
        split = cls(regime=data.get("regime", "pack"), seed=data.get("seed", 42),
                    notes=data.get("notes", []))
        for fold in ("train", "val", "test"):
            for d in data["files"].get(fold, []):
                getattr(split, fold).append(
                    Sample(
                        path=Path(d["path"]), label=d["label"],
                        product=d["product"], pack_id=d["pack_id"],
                        batch=d.get("batch"),
                    )
                )
        return split


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _label_from_path(path: Path, root: Path) -> str | None:
    """Infer the class from a parent directory named after it."""
    for part in path.relative_to(root).parts:
        low = part.lower()
        if low in ("genuine", "suspicious"):
            return low
    return None


def _identity_from_filename(name: str) -> tuple[str, str | None, str | None]:
    """Return (product, batch, pack) parsed from a collect-style filename."""
    m = FILENAME_RE.match(name)
    if m:
        return (
            m.group("product").upper(),
            m.group("batch").upper(),
            m.group("pack"),
        )
    # Older single-underscore form: PRODUCT_BATCH_timestamp_NN.jpg
    parts = Path(name).stem.split("_")
    if len(parts) >= 2:
        return parts[0].upper(), parts[1].upper(), None
    return Path(name).stem.upper(), None, None


def load_samples(
    root: Path | str,
    manifest: Path | str | None = None,
    strict: bool = False,
) -> list[Sample]:
    """Collect labelled samples from a dataset directory.

    Pack identity is taken from the manifest when one exists, because that is
    the only fully reliable record. Otherwise it is parsed from the filename,
    and failing that it falls back to grouping every image of a product+batch
    together.

    That fallback is deliberately *coarser* than reality: it merges several
    physical packs into one group. Merging groups can only make the split more
    conservative, never leakier, which is the correct direction to err. The
    opposite fallback -- treating each image as its own pack -- would silently
    reintroduce exactly the leakage this module exists to prevent.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"No such dataset directory: {root}")

    # `run.py collect` writes each image twice: once under dataset/<class>/ for
    # training and once under by_product/<PRODUCT>/ for calibration. Walking
    # both would double-count every sample, so when the training layout is
    # present it alone is used.
    search_roots = [root]
    if (root / "dataset").is_dir() and (root / "by_product").is_dir():
        search_roots = [root / "dataset"]
        log.info("Found collect layout; reading dataset/ and ignoring by_product/")

    # Manifest lookup. Keyed by relative path first: the same basename can
    # legitimately exist under both genuine/ and suspicious/, and a
    # basename-only key would let one class silently overwrite the other.
    manifest_by_rel: dict[str, dict] = {}
    manifest_by_name: dict[str, dict] = {}
    ambiguous_names: set[str] = set()

    manifest_path = Path(manifest) if manifest else root / "manifest.json"
    if not manifest_path.exists() and (root.parent / "manifest.json").exists():
        manifest_path = root.parent / "manifest.json"
    if manifest_path.exists():
        try:
            for entry in json.loads(manifest_path.read_text()):
                rel = str(Path(entry["file"])).replace("\\", "/")
                manifest_by_rel[rel] = entry
                name = Path(entry["file"]).name
                if name in manifest_by_name:
                    prev = manifest_by_name[name]
                    # Only a problem if the duplicates disagree about anything
                    # that matters for splitting.
                    if (prev.get("label") != entry.get("label")
                            or prev.get("pack_id") != entry.get("pack_id")):
                        ambiguous_names.add(name)
                else:
                    manifest_by_name[name] = entry
        except Exception as exc:
            log.warning("Could not read manifest %s: %s", manifest_path, exc)

    if ambiguous_names:
        log.info(
            "%d manifest filename(s) appear more than once with differing "
            "labels; resolving those from the directory instead.",
            len(ambiguous_names),
        )

    def lookup(path: Path) -> dict:
        for base in (root, *search_roots):
            try:
                rel = str(path.relative_to(base)).replace("\\", "/")
            except ValueError:
                continue
            if rel in manifest_by_rel:
                return manifest_by_rel[rel]
        if path.name in ambiguous_names:
            return {}
        return manifest_by_name.get(path.name, {})

    samples: list[Sample] = []
    unresolved = 0
    seen_paths: set[Path] = set()

    files = sorted(
        p for base in search_roots for p in base.rglob("*")
        if p.suffix.lower() in IMAGE_EXT
    )

    for path in files:
        if path in seen_paths:
            continue
        seen_paths.add(path)

        entry = lookup(path)

        # Directory placement is unambiguous, so it wins. The manifest is
        # authoritative for identity, not for class.
        label = _label_from_path(path, root) or entry.get("label")
        if label not in ("genuine", "suspicious"):
            if strict:
                raise ValueError(f"Cannot determine class for {path}")
            log.debug("Skipping unlabelled file: %s", path)
            continue

        product = (entry.get("product_name") or "").upper()
        batch = (entry.get("batch_number") or "").upper() or None
        pack = entry.get("pack_id")
        source = "manifest"

        if not product or pack is None:
            f_product, f_batch, f_pack = _identity_from_filename(path.name)
            product = product or f_product
            batch = batch or f_batch
            if pack is None and f_pack is not None:
                pack = f_pack
                source = "filename"

        if pack is None:
            # Coarse fallback: one group per product+batch.
            pack = f"{product}|{batch or 'NOBATCH'}|ALL"
            source = "coarse-fallback"
            unresolved += 1
        else:
            pack = f"{product}|{batch or 'NOBATCH'}|p{pack}"

        samples.append(
            Sample(path=path, label=label, product=product,
                   pack_id=str(pack), batch=batch, source=source)
        )

    if unresolved:
        log.warning(
            "%d image(s) had no pack identity; grouped coarsely by "
            "product+batch. Re-collect with `run.py collect` to record pack "
            "IDs and obtain a tighter split.",
            unresolved,
        )

    return samples


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------


def _group(samples: Iterable[Sample], key: str) -> dict[str, list[Sample]]:
    groups: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        groups[getattr(s, key)].append(s)
    return dict(groups)


def _dominant_label(samples: Sequence[Sample]) -> str:
    return Counter(s.label for s in samples).most_common(1)[0][0]


def split_by_pack(
    samples: Sequence[Sample],
    val_frac: float = 0.2,
    test_frac: float = 0.0,
    seed: int = 42,
) -> Split:
    """Partition so that no physical pack spans two folds.

    Packs are stratified by label, so each fold keeps roughly the dataset's
    class balance while remaining pack-disjoint.
    """
    if not samples:
        raise ValueError("No samples to split")
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must leave room for training")

    rng = random.Random(seed)
    packs = _group(samples, "pack_id")

    by_label: dict[str, list[str]] = defaultdict(list)
    for pack_id, items in packs.items():
        by_label[_dominant_label(items)].append(pack_id)

    split = Split(regime="pack", seed=seed)

    for label, pack_ids in by_label.items():
        ids = sorted(pack_ids)
        rng.shuffle(ids)
        n = len(ids)

        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))

        # With few packs, rounding can starve a fold entirely. Guarantee at
        # least one pack wherever a nonzero fraction was requested -- an empty
        # validation fold produces meaningless metrics rather than an error.
        if val_frac > 0 and n_val == 0 and n >= 2:
            n_val = 1
        if test_frac > 0 and n_test == 0 and n - n_val >= 2:
            n_test = 1
        if n_val + n_test >= n:
            n_val = max(1, n - 1) if val_frac > 0 else 0
            n_test = 0
            split.notes.append(
                f"Class '{label}' has only {n} pack(s); the test fold was "
                "dropped for it."
            )

        test_ids = ids[:n_test]
        val_ids = ids[n_test:n_test + n_val]
        train_ids = ids[n_test + n_val:]

        for pid in train_ids:
            split.train.extend(packs[pid])
        for pid in val_ids:
            split.val.extend(packs[pid])
        for pid in test_ids:
            split.test.extend(packs[pid])

    for fold in split.folds.values():
        fold.sort(key=lambda s: str(s.path))

    assert_no_leakage(split)
    return split


def split_by_product(
    samples: Sequence[Sample],
    holdout_products: Sequence[str] | None = None,
    n_holdout: int | None = None,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Split:
    """Hold entire products out of training.

    The held-out products form the test fold. Validation is carved from the
    remaining products at pack level, so the reported validation number stays
    comparable with the pack regime while the test number answers the
    unseen-SKU question.

    Products carrying both classes are preferred for holdout, since a test
    fold containing only one class cannot produce a meaningful precision or
    recall.
    """
    if not samples:
        raise ValueError("No samples to split")

    rng = random.Random(seed)
    by_product = _group(samples, "product")
    all_products = sorted(by_product)

    if holdout_products:
        holdout = [p.upper() for p in holdout_products]
        missing = [p for p in holdout if p not in by_product]
        if missing:
            raise ValueError(f"Products not in the dataset: {', '.join(missing)}")
    else:
        k = n_holdout if n_holdout is not None else max(1, len(all_products) // 5)
        k = min(k, max(1, len(all_products) - 1))

        both = [
            p for p in all_products
            if len({s.label for s in by_product[p]}) > 1
        ]
        single = [p for p in all_products if p not in both]
        rng.shuffle(both)
        rng.shuffle(single)
        holdout = (both + single)[:k]

    split = Split(regime="product", seed=seed)
    split.notes.append(f"Held-out products: {', '.join(sorted(holdout))}")

    remaining: list[Sample] = []
    for product, items in by_product.items():
        if product in holdout:
            split.test.extend(items)
        else:
            remaining.extend(items)

    if not remaining:
        raise ValueError("Holdout consumed every product; nothing left to train on")

    inner = split_by_pack(remaining, val_frac=val_frac, test_frac=0.0, seed=seed)
    split.train = inner.train
    split.val = inner.val
    split.notes.extend(inner.notes)

    test_labels = {s.label for s in split.test}
    if len(test_labels) < 2:
        split.notes.append(
            "The held-out products carry only one class, so precision and "
            "recall on the unseen-product fold are not meaningful. Collect "
            "both genuine and surrogate samples for more products."
        )

    for fold in split.folds.values():
        fold.sort(key=lambda s: str(s.path))

    assert_no_leakage(split)
    return split


def assert_no_leakage(split: Split) -> None:
    """Fail loudly if any pack, or any product in product regime, spans folds."""
    packs = {
        fold: {s.pack_id for s in samples}
        for fold, samples in split.folds.items()
        if samples
    }
    folds = list(packs)
    for i, a in enumerate(folds):
        for b in folds[i + 1:]:
            overlap = packs[a] & packs[b]
            if overlap:
                raise LeakageError(
                    f"{len(overlap)} pack(s) appear in both '{a}' and '{b}': "
                    f"{sorted(overlap)[:3]}"
                )

    if split.regime == "product":
        train_products = {s.product for s in split.train}
        test_products = {s.product for s in split.test}
        overlap = train_products & test_products
        if overlap:
            raise LeakageError(
                f"Product regime, but {sorted(overlap)} appear in both train "
                "and test."
            )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def describe(split: Split, stream=None) -> str:
    """Human-readable summary, printing images and packs side by side."""
    import io

    buf = io.StringIO()
    stats = split.stats()

    print(f"\nSplit regime: {split.regime}  (seed {split.seed})", file=buf)
    print(f"{'fold':<8} {'images':>8} {'packs':>7} {'products':>9}  by class",
          file=buf)
    print("-" * 62, file=buf)

    for fold in ("train", "val", "test"):
        s = stats[fold]
        if not s["images"]:
            continue
        classes = "  ".join(f"{k}={v}" for k, v in sorted(s["by_label"].items()))
        print(
            f"{fold:<8} {s['images']:>8} {s['packs']:>7} {s['products']:>9}  "
            f"{classes}",
            file=buf,
        )

    total_packs = len({s.pack_id for f in split.folds.values() for s in f})
    total_images = sum(len(f) for f in split.folds.values())
    print(
        f"\nEffective sample size: {total_packs} pack(s) "
        f"across {total_images} image(s).",
        file=buf,
    )
    if total_images and total_packs:
        print(
            f"Images per pack: {total_images / total_packs:.1f}. Statistical "
            "power comes from packs, not images.",
            file=buf,
        )

    for note in split.notes:
        print(f"  note: {note}", file=buf)

    warnings = check_adequacy(split)
    for w in warnings:
        print(f"  WARNING: {w}", file=buf)

    text = buf.getvalue()
    if stream is not None:
        stream.write(text)
    return text


def check_adequacy(split: Split, min_packs_per_class: int = 10) -> list[str]:
    """Flag conditions that would make reported numbers unreliable."""
    warnings: list[str] = []

    train_packs_by_label: dict[str, set[str]] = defaultdict(set)
    for s in split.train:
        train_packs_by_label[s.label].add(s.pack_id)

    for label, packs in sorted(train_packs_by_label.items()):
        if len(packs) < min_packs_per_class:
            warnings.append(
                f"Only {len(packs)} training pack(s) for class '{label}'. "
                f"Below about {min_packs_per_class} the model will memorise "
                "individual packs; collect more distinct packs rather than "
                "more photos of the same ones."
            )

    val_packs = {s.pack_id for s in split.val}
    if len(val_packs) < 5:
        warnings.append(
            f"Validation fold holds only {len(val_packs)} pack(s); its metrics "
            "will swing widely between runs."
        )

    val_labels = {s.label for s in split.val}
    if len(val_labels) < 2 and split.val:
        warnings.append("Validation fold contains a single class.")

    if split.regime == "product":
        test_products = {s.product for s in split.test}
        if len(test_products) < 3:
            warnings.append(
                f"Only {len(test_products)} held-out product(s); the "
                "unseen-product result will not generalise as a claim."
            )

    return warnings
