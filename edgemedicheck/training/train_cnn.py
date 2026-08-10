#!/usr/bin/env python3
"""
Train the MobileNetV2 package-authenticity classifier and export TFLite.

Implements the CNN module of the paper: a lightweight, edge-deployable binary
classifier over package images, trained by transfer learning and exported for
TensorFlow Lite inference on the Raspberry Pi 4.

Splitting
---------
This script does NOT split images at random. Several photographs of one
physical carton share its creases, scuffs and print run, so a random split
lets the network recognise individual packs and reports a validation number
that will not hold up in a pharmacy. Two regimes are available:

    --regime pack     no physical pack spans two folds
    --regime product  entire products are held out of training
    --regime both     train once per regime and print the comparison

Report both. The gap between them measures how far the model generalises to
products it has never seen, which is the honest answer to how many medicines
it actually knows.

Expected dataset layout
-----------------------
    dataset/
        genuine/     PRODUCT__BATCH__pNN__SS.jpg
        suspicious/  PRODUCT__BATCH__pNN__SS.jpg
        manifest.json   (written by `run.py collect`)

Filenames or the manifest supply the pack identity. Collect with
`run.py collect` and both are recorded automatically.

Usage
-----
    python training/train_cnn.py --data data/collected/dataset
    python training/train_cnn.py --data data/collected/dataset --regime product
    python training/train_cnn.py --data data/collected/dataset --split-file s.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# --------------------------------------------------------------------------
# Dataset construction
# --------------------------------------------------------------------------


def build_dataset(tf, samples, class_names, image_size, batch_size, shuffle,
                  seed):
    """Build a tf.data pipeline from an explicit sample list.

    Files are listed explicitly rather than through
    `image_dataset_from_directory`, because that helper can only split
    randomly and would undo the grouping this whole module exists to enforce.
    """
    if not samples:
        return None

    index = {c: i for i, c in enumerate(class_names)}
    paths = [str(s.path) for s in samples]
    labels = [index[s.label] for s in samples]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load(path, label):
        raw = tf.io.read_file(path)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, image_size)
        img.set_shape([image_size[0], image_size[1], 3])
        return img, tf.cast(label, tf.float32)

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(min(len(paths), 1000), seed=seed,
                        reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def evaluate_fold(tf, np, model, ds, threshold):
    """Confusion matrix and derived metrics at the deployment threshold."""
    if ds is None:
        return None

    y_true, y_score = [], []
    for images, labels in ds:
        y_true.extend(labels.numpy().reshape(-1).tolist())
        y_score.extend(model.predict(images, verbose=0).reshape(-1).tolist())

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = (y_score >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / max(len(y_true), 1)

    # AUC is threshold-free, so it is the fairer number to compare between
    # regimes when the operating point has not been retuned.
    auc = None
    if len(set(y_true.tolist())) > 1:
        order = np.argsort(y_score)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(y_score) + 1)
        n_pos = int(y_true.sum())
        n_neg = len(y_true) - n_pos
        if n_pos and n_neg:
            auc = float(
                (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2)
                / (n_pos * n_neg)
            )

    return {
        "n": int(len(y_true)), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "auc": auc,
    }


def print_fold(name, m, class_names):
    if m is None:
        print(f"\n  {name}: empty fold, skipped")
        return
    print(f"\n  {name}  (n={m['n']})")
    print(f"    {'':>12} {'pred ' + class_names[0]:>16} "
          f"{'pred ' + class_names[1]:>16}")
    print(f"    {class_names[0]:>12} {m['tn']:>16} {m['fp']:>16}")
    print(f"    {class_names[1]:>12} {m['fn']:>16} {m['tp']:>16}")
    auc = f"{m['auc']:.4f}" if m["auc"] is not None else "n/a"
    print(f"    accuracy {m['accuracy']:.4f}   precision {m['precision']:.4f}   "
          f"recall {m['recall']:.4f}   F1 {m['f1']:.4f}   AUC {auc}")
    if m["fn"]:
        print(f"    {m['fn']} suspicious pack(s) missed -- the costly error.")


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


def build_model(tf, cfg):
    """MobileNetV2 with a fresh binary head.

    Augmentation reflects how a package is presented at a counter: slightly
    rotated, slightly offset, under variable LED brightness. No vertical flips
    -- upside-down packaging is not a real case, and training the model to
    accept it discards a genuine layout cue.
    """
    augment = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.04),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomTranslation(0.06, 0.06),
            tf.keras.layers.RandomBrightness(0.15, value_range=(0, 255)),
            tf.keras.layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )

    base = tf.keras.applications.MobileNetV2(
        input_shape=(*cfg.input_size, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(*cfg.input_size, 3))
    x = augment(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="suspicion")(x)
    return tf.keras.Model(inputs, outputs), base


def metrics_list(tf):
    return [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]


def train_one_regime(tf, np, split, cfg, args, out_dir, tag):
    """Train and evaluate a single split regime."""
    class_names = list(cfg.labels)
    seed = args.seed if args.seed is not None else cfg.seed
    batch_size = args.batch_size or cfg.batch_size

    train_ds = build_dataset(tf, split.train, class_names, cfg.input_size,
                             batch_size, shuffle=True, seed=seed)
    val_ds = build_dataset(tf, split.val, class_names, cfg.input_size,
                           batch_size, shuffle=False, seed=seed)
    test_ds = build_dataset(tf, split.test, class_names, cfg.input_size,
                            batch_size, shuffle=False, seed=seed)

    # Class weights, not resampling: the evaluation cares specifically about
    # recall on the suspicious class, and resampling would distort the
    # validation distribution used to measure it.
    counts = {c: sum(1 for s in split.train if s.label == c) for c in class_names}
    total = sum(counts.values())
    class_weight = None
    if total and min(counts.values()) > 0:
        class_weight = {
            i: total / (len(class_names) * counts[c])
            for i, c in enumerate(class_names)
        }

    print(f"\n{'=' * 68}")
    print(f"REGIME: {split.regime}")
    print(f"{'=' * 68}")
    print(f"  train {len(split.train):>5} images / "
          f"{len({s.pack_id for s in split.train}):>3} packs   {counts}")
    print(f"  val   {len(split.val):>5} images / "
          f"{len({s.pack_id for s in split.val}):>3} packs")
    print(f"  test  {len(split.test):>5} images / "
          f"{len({s.pack_id for s in split.test}):>3} packs")
    if class_weight:
        print(f"  class weights: "
              + ", ".join(f"{class_names[i]}={w:.2f}"
                          for i, w in class_weight.items()))

    model, base = build_model(tf, cfg)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.learning_rate_head),
        loss="binary_crossentropy",
        metrics=metrics_list(tf),
    )

    keras_path = out_dir / f"package_authenticity{tag}.keras"
    monitor = "val_recall" if val_ds is not None else "recall"
    callbacks = [
        # Recall is monitored because missing a suspicious pack is the costly
        # error; a model that maximises accuracy by passing everything is
        # useless here.
        tf.keras.callbacks.ModelCheckpoint(
            str(keras_path), monitor=monitor, mode="max",
            save_best_only=True, verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss" if val_ds is not None else "loss",
            patience=5, restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if val_ds is not None else "loss",
            factor=0.5, patience=3, verbose=0,
        ),
    ]

    epochs_head = (
        args.epochs_head if args.epochs_head is not None else cfg.epochs_head
    )
    epochs_ft = (
        args.epochs_finetune
        if args.epochs_finetune is not None
        else cfg.epochs_finetune
    )

    print("\n  Stage 1: training the classification head")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs_head,
              class_weight=class_weight, callbacks=callbacks, verbose=2)

    if not args.no_finetune and epochs_ft > 0:
        print("\n  Stage 2: fine-tuning the upper backbone")
        base.trainable = True
        for layer in base.layers[: cfg.finetune_from_layer]:
            layer.trainable = False
        # BatchNorm stays frozen: with small batches its running statistics
        # drift and undo the pretrained representation.
        for layer in base.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(cfg.learning_rate_finetune),
            loss="binary_crossentropy",
            metrics=metrics_list(tf),
        )
        model.fit(train_ds, validation_data=val_ds, epochs=epochs_ft,
                  class_weight=class_weight, callbacks=callbacks, verbose=2)

    print(f"\n  Results at threshold {cfg.suspicion_threshold}")
    results = {
        "val": evaluate_fold(tf, np, model, val_ds, cfg.suspicion_threshold),
        "test": evaluate_fold(tf, np, model, test_ds, cfg.suspicion_threshold),
    }
    label = ("unseen packs, seen products" if split.regime == "pack"
             else "unseen packs of seen products")
    print_fold(f"validation ({label})", results["val"], class_names)
    if results["test"]:
        test_label = ("held-out packs" if split.regime == "pack"
                      else "UNSEEN PRODUCTS")
        print_fold(f"test ({test_label})", results["test"], class_names)

    model.save(keras_path)
    return model, results


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data", required=True, help="dataset root with class folders")
    ap.add_argument("--out", default=None, help="output directory for models")
    ap.add_argument("--regime", default="both",
                    choices=["pack", "product", "both"])
    ap.add_argument("--split-file", default=None,
                    help="use a frozen split written by `run.py splits`")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--n-holdout", type=int, default=None,
                    help="products to hold out in the product regime")
    ap.add_argument("--holdout", nargs="*", default=None)
    ap.add_argument("--epochs-head", type=int, default=None)
    ap.add_argument("--epochs-finetune", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--no-finetune", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="train even when the split is too small to be reliable")
    args = ap.parse_args()

    try:
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print(
            "TensorFlow is required for training.\n"
            "  Apple Silicon : pip install tensorflow tensorflow-metal\n"
            "  Other         : pip install tensorflow\n"
            "  Raspberry Pi  : do not train here. Train on a workstation and\n"
            "                  copy the exported .tflite file across.",
            file=sys.stderr,
        )
        return 1

    from edgemedicheck.config import CONFIG, MODEL_DIR
    from edgemedicheck.splits import (
        Split, check_adequacy, describe, load_samples, split_by_pack,
        split_by_product,
    )

    cfg = CONFIG.cnn
    seed = args.seed if args.seed is not None else cfg.seed
    out_dir = Path(args.out) if args.out else MODEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(seed)

    devices = tf.config.list_physical_devices()
    gpus = [d for d in devices if d.device_type in ("GPU", "MPS")]
    print(f"TensorFlow {tf.__version__}  |  "
          f"{'accelerator: ' + gpus[0].device_type if gpus else 'CPU only'}")
    if not gpus:
        print("  (MobileNetV2 on a few thousand images trains fine on CPU; "
              "this is a small job.)")

    # ---- Build splits -------------------------------------------------
    splits: list[Split] = []
    if args.split_file:
        split = Split.load(args.split_file)
        print(f"\nUsing frozen split from {args.split_file}")
        splits = [split]
    else:
        samples = load_samples(args.data)
        if not samples:
            print(f"No labelled images under {args.data}", file=sys.stderr)
            return 1

        regimes = ["pack", "product"] if args.regime == "both" else [args.regime]
        for regime in regimes:
            if regime == "pack":
                splits.append(split_by_pack(
                    samples, val_frac=args.val_frac, test_frac=args.test_frac,
                    seed=seed,
                ))
            else:
                splits.append(split_by_product(
                    samples, holdout_products=args.holdout or None,
                    n_holdout=args.n_holdout, val_frac=args.val_frac, seed=seed,
                ))

    for split in splits:
        print(describe(split))

    # Refuse by default when the split cannot support a believable number.
    blocking = [w for s in splits for w in check_adequacy(s)]
    if blocking and not args.force:
        print("\nRefusing to train: the dataset is too small for these numbers "
              "to mean anything.")
        for w in dict.fromkeys(blocking):
            print(f"  - {w}")
        print("\nCollect more distinct packs, or pass --force to train anyway "
              "(and do not report the result).")
        return 2

    # ---- Train --------------------------------------------------------
    all_results: dict[str, dict] = {}
    last_model = None

    for split in splits:
        tag = "" if len(splits) == 1 else f"_{split.regime}"
        model, results = train_one_regime(tf, np, split, cfg, args, out_dir, tag)
        all_results[split.regime] = results
        last_model = model
        split.save(out_dir / f"split{tag or '_' + split.regime}.json")

    # ---- The comparison that matters ----------------------------------
    if "pack" in all_results and "product" in all_results:
        pack_test = all_results["pack"].get("test")
        prod_test = all_results["product"].get("test")
        if pack_test and prod_test:
            print(f"\n{'=' * 68}")
            print("GENERALISATION GAP")
            print(f"{'=' * 68}")
            print(f"  {'metric':<12} {'unseen packs':>14} "
                  f"{'unseen products':>17} {'drop':>9}")
            for key in ("accuracy", "recall", "f1"):
                a, b = pack_test[key], prod_test[key]
                print(f"  {key:<12} {a:>14.4f} {b:>17.4f} {a - b:>9.4f}")
            drop = pack_test["recall"] - prod_test["recall"]
            print()
            if drop > 0.15:
                print("  Recall falls sharply on products the model never saw.")
                print("  That is the coverage limitation, measured rather than")
                print("  assumed. Report it: it is a finding, not a failure.")
            else:
                print("  Performance holds up on unseen products, which is a")
                print("  strong result. Confirm it is not an artefact of the")
                print("  held-out products being unusually easy.")

    # ---- Export -------------------------------------------------------
    if last_model is not None:
        converter = tf.lite.TFLiteConverter.from_keras_model(last_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        tflite_path = out_dir / "package_authenticity.tflite"
        tflite_path.write_bytes(tflite_model)
        size_mb = len(tflite_model) / (1024 * 1024)
        print(f"\nSaved TFLite model: {tflite_path}  ({size_mb:.2f} MB)")

        metadata = {
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "dataset": str(args.data),
            "class_names": list(cfg.labels),
            "input_size": list(cfg.input_size),
            "suspicion_threshold": cfg.suspicion_threshold,
            "seed": seed,
            "results_by_regime": all_results,
            "split_stats": {s.regime: s.stats() for s in splits},
            "tflite_mb": size_mb,
            "exported_regime": splits[-1].regime,
        }
        (out_dir / "training_metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str)
        )
        print(f"Saved metadata: {out_dir / 'training_metadata.json'}")
        print("\nThe scanner picks up the TFLite model automatically on next "
              "start.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
