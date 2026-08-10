#!/usr/bin/env python3
"""
EdgeMediCheck command-line interface.

Subcommands
-----------
    init      Create the SQLite schema.
    seed      Load batch records into the local database.
    scan      Scan one image file, or capture live from the camera.
    batch     Scan every image in a folder and print a summary table.
    evaluate  Score the pipeline against a ground-truth manifest.
    stats     Show scan-log statistics.
    products  List batch records in the local database.
    serve     Start the Flask pharmacist interface.

Examples
--------
    python run.py init
    python run.py seed --from-manifest data/images/manifest.json
    python run.py scan data/images/001_genuine_azithral_K24039.jpg
    python run.py scan --live
    python run.py batch data/images --limit 20
    python run.py evaluate data/images/manifest.json
    python run.py serve
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from edgemedicheck import database as db  # noqa: E402
from edgemedicheck.config import CONFIG  # noqa: E402

# ANSI colours for the verdict badge. Disabled when not a TTY.
_TTY = sys.stdout.isatty()


def _c(text: str, colour: str) -> str:
    if not _TTY:
        return text
    codes = {"green": "\033[92m", "yellow": "\033[93m", "red": "\033[91m",
             "info": "\033[94m", "grey": "\033[90m", "bold": "\033[1m"}
    if colour not in codes:
        return text
    return f"{codes[colour]}{text}\033[0m"


def _badge(verdict: str) -> str:
    return _c(f" {verdict.upper():^7} ", verdict)


# --------------------------------------------------------------------------
# init
# --------------------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> int:
    db.init_db(args.db)
    print(f"Database initialised: {args.db or CONFIG.db_path}")
    return 0


# --------------------------------------------------------------------------
# seed
# --------------------------------------------------------------------------


DEMO_RECORDS = [
    # (product, manufacturer, batch, mfg, exp, status)
    ("PARACIP", "Cipla Pharmaceuticals Ltd", "PC24101", "2024-01-01", "2027-01-31", db.VALID),
    ("AZITHRAL", "Alkem Laboratories Ltd", "AZ23088", "2023-08-01", "2026-08-31", db.VALID),
    ("CROCIN", "Sun Pharma Laboratories Ltd", "CR22045", "2022-04-01", "2024-04-30", db.EXPIRED),
    ("PANTOCID", "Sun Pharma Laboratories Ltd", "PT24777", "2024-07-01", "2027-07-31", db.RECALLED),
    ("METFOR", "Micro Labs Limited", "MF23012", "2023-01-01", "2026-01-31", db.NSQ),
]


def cmd_seed(args: argparse.Namespace) -> int:
    db.init_db(args.db)
    added = 0

    if args.from_manifest:
        manifest = json.loads(Path(args.from_manifest).read_text())
        for entry in manifest:
            # Only genuine and expired samples get a database record. The
            # "suspicious" samples are deliberately left out so the unknown
            # batch path is exercised too.
            if entry.get("kind") == "suspicious" and not args.include_suspicious:
                continue
            status = db.EXPIRED if entry.get("kind") == "expired" else db.VALID
            db.upsert_product(
                product_name=entry["product_name"],
                manufacturer=entry["manufacturer"],
                batch_number=entry["batch_number"],
                mfg_date=entry.get("mfg_date"),
                exp_date=entry["exp_date"],
                source_type="pharmacy",
                status=status,
                notes="Seeded from synthetic manifest (testing only)",
                db_path=args.db,
            )
            added += 1
        print(f"Seeded {added} records from {args.from_manifest}")

    if args.from_csv:
        import csv

        with open(args.from_csv, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                db.upsert_product(
                    product_name=row["product_name"],
                    manufacturer=row["manufacturer"],
                    batch_number=row["batch_number"],
                    mfg_date=row.get("mfg_date") or None,
                    exp_date=row["exp_date"],
                    source_type=row.get("source_type", "pharmacy"),
                    status=row.get("status", db.VALID),
                    notes=row.get("notes"),
                    db_path=args.db,
                )
                added += 1
        print(f"Seeded from CSV: {args.from_csv}")

    if args.demo:
        for name, mfr, batch, mfg, exp, status in DEMO_RECORDS:
            db.upsert_product(
                product_name=name,
                manufacturer=mfr,
                batch_number=batch,
                mfg_date=mfg,
                exp_date=exp,
                status=status,
                notes="Built-in demo record",
                db_path=args.db,
            )
            added += 1
        print(f"Seeded {len(DEMO_RECORDS)} demo records")

    if not added:
        print("Nothing seeded. Pass --demo, --from-manifest, or --from-csv.")
        return 1

    print(f"Database now holds {db.count_products(args.db)} records.")
    return 0


# --------------------------------------------------------------------------
# collect -- guided dataset capture
# --------------------------------------------------------------------------


def _prompt(label: str, default: str | None = None, required: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        value = input(f"  {label}{suffix}: ").strip()
        if not value and default is not None:
            return default
        if value or not required:
            return value
        print("    This field is required.")


def cmd_collect(args: argparse.Namespace) -> int:
    """Guided capture of a labelled package-image dataset.

    Collecting the dataset is the real blocker between a proposed system and a
    results paper, and doing it ad hoc produces images that cannot be used:
    inconsistent framing, missing labels, no record of which pack is which.

    This walks through one product at a time, captures a burst under the same
    enclosure conditions the scanner will use in service, and writes both the
    class folders that `train_cnn.py` expects and the per-product folders that
    `calibrate` expects -- from a single pass.
    """
    import time

    from edgemedicheck.capture import CaptureError, LEDController, open_source
    from edgemedicheck.capture import find_package_region
    from edgemedicheck.preprocess import preprocess

    out = Path(args.out)
    label = args.label
    if label not in ("genuine", "suspicious"):
        print("--label must be 'genuine' or 'suspicious'", file=sys.stderr)
        return 2

    try:
        source = open_source(args.backend, folder=args.folder)
    except CaptureError as exc:
        print(f"Camera unavailable: {exc}", file=sys.stderr)
        return 1

    led = LEDController(CONFIG.capture.led_gpio_pin)
    manifest_path = out / "manifest.json"
    manifest: list[dict] = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            manifest = []

    print()
    print(_c("  Dataset collection", "bold"))
    print(f"  Writing to {out}  (label: {label})")
    print("  Keep the enclosure, lighting and camera distance fixed for every")
    print("  capture. Press Ctrl-C to stop.")
    print()

    captured_total = 0
    try:
        while True:
            print(_c("  --- New pack ---", "bold"))
            product = _prompt("Product / brand name", required=True).upper()
            product = re.sub(r"[^A-Z0-9]+", "", product) or "UNKNOWN"
            batch = _prompt("Batch number", required=True).upper()
            batch = re.sub(r"[^A-Z0-9]+", "", batch) or "NOBATCH"
            exp = _prompt("Expiry (YYYY-MM or YYYY-MM-DD)", required=True)
            mfr = _prompt("Manufacturer", default="")
            notes = _prompt("Notes (defect type, source)", default="")

            # Each physical carton gets its own ID. This is the unit of
            # statistical independence: several photographs of one pack are
            # not several samples, and the split has to keep them together.
            # Auto-numbered per product+batch so it cannot be forgotten.
            seen = {
                m.get("pack_id")
                for m in manifest
                if m.get("product_name") == product
                and m.get("batch_number") == batch
                and m.get("pack_id") is not None
            }
            pack_no = (max((int(p) for p in seen), default=0) + 1)
            print(f"  Physical pack #{pack_no} of {product} / batch {batch}.")
            print(_c("    If this is the SAME carton you just photographed, "
                     "stop and use more shots instead.", "grey"))

            group_dir = out / "by_product" / product
            class_dir = out / "dataset" / label
            group_dir.mkdir(parents=True, exist_ok=True)
            class_dir.mkdir(parents=True, exist_ok=True)

            print()
            print(f"  Place the {product} pack in the enclosure.")
            print(f"  Capturing {args.shots} shot(s). Reposition slightly "
                  "between each so the set covers real placement variation.")

            taken = 0
            attempts = 0
            while taken < args.shots and attempts < args.shots * 4:
                attempts += 1
                if not args.auto:
                    input(f"    Press Enter for shot {taken + 1}/{args.shots} ")
                else:
                    time.sleep(args.interval)

                led.on()
                try:
                    frame = source.read()
                finally:
                    led.off()

                region_ok, _, ratio = find_package_region(frame)
                p = preprocess(frame)

                # Reject unusable frames at capture time. A blurred image found
                # later is a wasted trip to the pharmacy.
                problems = []
                if not region_ok:
                    problems.append(f"package fills {ratio:.0%} of frame")
                if p.sharpness < 25:
                    problems.append(f"blurred (sharpness {p.sharpness:.0f})")
                if not (40 <= p.brightness <= 245):
                    problems.append(f"lighting off (brightness {p.brightness:.0f})")

                if problems and not args.keep_all:
                    print(_c(f"      rejected: {', '.join(problems)}", "yellow"))
                    continue

                # PRODUCT__BATCH__pNN__SS.jpg -- parseable back into the
                # grouping keys even if the manifest is ever lost.
                name = f"{product}__{batch}__p{pack_no:02d}__{taken:02d}.jpg"
                cv2_write(group_dir / name, frame)
                cv2_write(class_dir / name, frame)

                manifest.append({
                    "file": name,
                    "product_name": product,
                    "batch_number": batch,
                    "pack_id": pack_no,
                    "exp_date": exp,
                    "manufacturer": mfr,
                    "label": label,
                    "notes": notes,
                    "sharpness": round(p.sharpness, 1),
                    "brightness": round(p.brightness, 1),
                    "area_ratio": round(ratio, 3),
                    "captured_at": datetime.now().isoformat(timespec="seconds"),
                })
                taken += 1
                captured_total += 1
                print(_c(f"      saved {name}", "green"))

            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2))
            print(f"\n  {taken} image(s) of {product} pack #{pack_no}. "
                  f"{captured_total} total this session.\n")

            if _prompt("Capture another pack? (y/n)", default="y").lower() != "y":
                break

    except KeyboardInterrupt:
        print("\n  Stopped.")
    finally:
        source.close()
        manifest_path.write_text(json.dumps(manifest, indent=2))

    # Report images AND distinct packs. Only the second drives statistical
    # power, and it is the number that is easy to lose track of while
    # collecting.
    by_product: dict[str, list[int]] = {}
    for m in manifest:
        entry = by_product.setdefault(m["product_name"], [0, set()])
        entry[0] += 1
        entry[1].add((m.get("batch_number"), m.get("pack_id")))

    print()
    print(f"  Dataset now holds {len(manifest)} image(s) in {out}")
    print(f"    {'product':<20} {'images':>7} {'packs':>7}")
    for product, (n_img, packs) in sorted(by_product.items()):
        flag = "" if n_img >= 12 else _c("   (needs 12+ to calibrate)", "yellow")
        print(f"    {product:<20} {n_img:>7} {len(packs):>7}{flag}")

    total_packs = sum(len(p) for _, p in by_product.values())
    print(f"\n  Effective sample size: {total_packs} distinct pack(s).")
    if total_packs and len(manifest) / total_packs > 15:
        print(_c("    Many photographs per pack. Additional shots of the same "
                 "carton add\n    little; collecting more distinct packs adds "
                 "much more.", "yellow"))

    print()
    print("  Next:")
    print(f"    python run.py splits {out / 'dataset'}")
    print(f"    python run.py calibrate {out / 'by_product'}")
    print(f"    python training/train_cnn.py --data {out / 'dataset'}")
    return 0


def cv2_write(path: Path, image) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------


def cmd_splits(args: argparse.Namespace) -> int:
    """Inspect, validate and freeze a dataset split.

    Run this before training. It shows the effective sample size, warns when
    a fold is too small to produce a stable number, and can write the split to
    disk so a reported result is reproducible.
    """
    from edgemedicheck.splits import (
        describe, load_samples, split_by_pack, split_by_product,
    )

    samples = load_samples(args.data, manifest=args.manifest)
    if not samples:
        print(f"No labelled images found under {args.data}", file=sys.stderr)
        print("Expected genuine/ and suspicious/ subfolders, or a manifest.",
              file=sys.stderr)
        return 1

    sources = {}
    for s in samples:
        sources[s.source] = sources.get(s.source, 0) + 1
    print(f"Loaded {len(samples)} image(s) from {args.data}")
    print(f"  pack identity from: "
          + ", ".join(f"{k} ({v})" for k, v in sorted(sources.items())))
    if sources.get("coarse-fallback"):
        print(_c(
            "  Some images carry no pack ID, so every image of a "
            "product+batch was\n  grouped together. That is conservative but "
            "coarse -- recollect with\n  `run.py collect` to record pack IDs.",
            "yellow",
        ))

    regimes = ["pack", "product"] if args.regime == "both" else [args.regime]

    for regime in regimes:
        if regime == "pack":
            split = split_by_pack(
                samples, val_frac=args.val_frac, test_frac=args.test_frac,
                seed=args.seed,
            )
        else:
            split = split_by_product(
                samples,
                holdout_products=args.holdout or None,
                n_holdout=args.n_holdout,
                val_frac=args.val_frac,
                seed=args.seed,
            )
        print(describe(split))

        if args.out:
            out = Path(args.out)
            path = out if len(regimes) == 1 else out.with_name(
                f"{out.stem}_{regime}{out.suffix or '.json'}"
            )
            split.save(path)
            print(f"  Saved split to {path}\n")

    return 0


# --------------------------------------------------------------------------
# feedback
# --------------------------------------------------------------------------


def cmd_feedback(args: argparse.Namespace) -> int:
    """Review pharmacist corrections and export them as training data."""
    import shutil

    db.init_db(args.db)

    if args.export:
        pending = db.pending_feedback(args.db, include_exported=args.all)
        if not pending:
            print("No corrections to export.")
            return 0

        out = Path(args.export)
        exported_ids: list[int] = []
        counts: dict[str, int] = {}
        missing = 0

        for fb in pending:
            label = fb.get("correct_label")
            src = fb.get("image_path")
            if not label or label not in db.FEEDBACK_LABELS:
                continue
            if not src or not Path(src).exists():
                missing += 1
                continue

            # Expired packs are visually genuine, so for the *visual* classifier
            # they belong in the genuine class. Their expiry is handled by OCR
            # and the database, not by appearance.
            visual_class = "genuine" if label == db.LABEL_EXPIRED else label
            dest_dir = out / visual_class
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"fb{fb['feedback_id']:05d}_{Path(src).name}"
            shutil.copy2(src, dest)

            counts[visual_class] = counts.get(visual_class, 0) + 1
            exported_ids.append(fb["feedback_id"])

        if not args.dry_run:
            db.mark_feedback_exported(exported_ids, args.db)

        print(f"Exported {len(exported_ids)} correction(s) to {out}")
        for cls, n in sorted(counts.items()):
            print(f"  {cls:<12} {n}")
        if missing:
            print(_c(f"  {missing} skipped: the captured image is no longer on "
                     "disk.", "yellow"))
        if args.dry_run:
            print(_c("  Dry run: nothing was marked as exported.", "grey"))
        return 0

    # Default: show the report.
    stats = db.feedback_stats(args.db)
    print(f"Corrections recorded: {stats['total']}")
    print(f"  pending export:     {stats['pending_export']}")

    if stats["by_label"]:
        print("\n  Pharmacist labels")
        for label, n in sorted(stats["by_label"].items()):
            print(f"    {label:<12} {n}")

    if stats["disagreements"]:
        print("\n  Where the scanner was wrong (system -> correct)")
        for key, n in sorted(stats["disagreements"].items(), key=lambda kv: -kv[1]):
            print(f"    {key:<22} {n}")

    rate = stats["reported_error_rate"]
    if rate is not None:
        print(f"\n  Reported errors per scan: {rate}")
        print(_c("  This is a lower bound on the error rate, not an estimate "
                 "of it:\n  staff report wrong verdicts far more often than "
                 "they confirm right ones.", "grey"))

    if stats["pending_export"]:
        print(f"\n  Export them as training data:")
        print(f"    python run.py feedback --export data/feedback_dataset")
    return 0


# --------------------------------------------------------------------------
# calibrate
# --------------------------------------------------------------------------


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Fit the heuristic visual backend on known-genuine reference images.

    Run this once per enclosure. The fitted distribution is specific to the
    camera, lens distance, and lighting of the unit it was captured on, so it
    should be refitted if any of those change.
    """
    from edgemedicheck.cnn import (
        MIN_CALIBRATION_SAMPLES,
        CALIBRATION_PATH,
        fit_calibration_set,
        reset_authenticator,
    )

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Not a directory: {folder}", file=sys.stderr)
        return 1

    images = list(_iter_images(folder))
    if not images:
        print(f"No images found in {folder}", file=sys.stderr)
        return 1

    print(f"Fitting visual calibration from {folder} ({len(images)} image(s))")
    calib = fit_calibration_set(folder)
    out = Path(args.out) if args.out else CALIBRATION_PATH
    calib.save(out)
    reset_authenticator()

    print(f"\nWrote calibration to {out}\n")
    print(f"  {'group':<26} {'images':>8}  {'usable':>7}")
    print("  " + "-" * 44)

    usable_groups = 0
    for key, c in sorted(calib.groups.items()):
        flag = "yes" if c.is_trustworthy else "no"
        usable_groups += int(c.is_trustworthy)
        colour = "green" if c.is_trustworthy else "yellow"
        print(f"  {key:<26} {c.n_samples:>8}  {_c(f'{flag:>7}', colour)}")

    g = calib.global_calibration
    if g:
        flag = "yes" if g.is_trustworthy else "no"
        colour = "green" if g.is_trustworthy else "yellow"
        print(f"  {'(global fallback)':<26} {g.n_samples:>8}  "
              f"{_c(f'{flag:>7}', colour)}")

    print()
    if not calib.groups:
        print(
            _c(
                "  No per-product groups found; only the global fallback was "
                "fitted.\n"
                "  Per-product calibration is substantially more accurate. On a "
                "held-out\n"
                "  test set, pooling different package forms separated genuine "
                "from\n"
                "  defective packs at only AUC 0.66, while calibrating within a "
                "single\n"
                "  form reached 0.92-0.99. Organise reference images one "
                "subfolder per\n"
                "  product:\n"
                f"      {folder}/PARACIP/*.jpg\n"
                f"      {folder}/AZITHRAL/*.jpg",
                "yellow",
            )
        )
    else:
        print(f"  {usable_groups} of {len(calib.groups)} product group(s) usable "
              f"(each needs {MIN_CALIBRATION_SAMPLES}+ images).")

    print()
    print(
        _c(
            "  These statistics describe YOUR enclosure and camera. Refit "
            "after any change to lighting, lens distance, or hardware.",
            "grey",
        )
    )
    return 0


# --------------------------------------------------------------------------
# scan
# --------------------------------------------------------------------------


def _print_result(result, verbose: bool = False) -> None:
    v = result.verdict
    print()
    print(f"{_badge(v.verdict)}  {_c(v.headline, 'bold')}")
    print(f"         {v.reason_code}: {v.reason_text}")
    if v.advice:
        print(_c(f"         {v.advice}", "grey"))
    print()

    print("  Extracted from package")
    fields = [
        ("Product", result.ocr.get("product_name").value),
        ("Batch", result.ocr.batch_number),
        ("Manufactured", result.ocr.mfg_date.iso if result.ocr.mfg_date else None),
        ("Expires", result.ocr.exp_date.iso if result.ocr.exp_date else None),
        ("Manufacturer", result.ocr.manufacturer),
    ]
    for label, value in fields:
        shown = value if value else _c("not read", "grey")
        print(f"    {label:<14} {shown}")

    print()
    print("  Checks")
    print(f"    {'OCR':<14} {result.ocr.status} "
          f"(mean confidence {result.ocr.mean_confidence:.0f})")
    print(f"    {'Database':<14} {result.lookup.status} - {result.lookup.reason}")
    backend = result.visual.backend
    tag = "" if result.visual.is_model_backed else _c(" [heuristic, not a CNN]", "grey")
    print(f"    {'Visual':<14} score {result.visual.suspicion_score:.2f} "
          f"via {backend}{tag}")

    if len(v.findings) > 1:
        print()
        print("  All findings")
        for f in v.findings:
            print(f"    {_badge(f.verdict)} {f.code}: {f.text}")

    print()
    stages = " ".join(f"{k}={t:.0f}ms" for k, t in result.timings_ms.items())
    print(_c(f"  {result.total_ms:.0f} ms total  ({stages})", "grey"))

    if verbose:
        print()
        print("  Raw OCR text")
        for line in result.ocr.raw_text.split("\n"):
            print(_c(f"    | {line}", "grey"))
    print()


def cmd_scan(args: argparse.Namespace) -> int:
    from edgemedicheck.capture import LEDController, open_source
    from edgemedicheck.pipeline import scan_from_file, scan_live

    db.init_db(args.db)
    today = date.fromisoformat(args.today) if args.today else None

    if args.live:
        source = open_source(args.backend, folder=args.folder)
        led = LEDController(CONFIG.capture.led_gpio_pin)
        print(f"Capturing from {source.name} ...")
        try:
            result = scan_live(source, led=led, today=today, db_path=args.db)
        finally:
            source.close()
    else:
        if not args.image:
            print("Provide an image path, or use --live.", file=sys.stderr)
            return 2
        result = scan_from_file(
            args.image, today=today, db_path=args.db, save_image=args.save
        )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        _print_result(result, args.verbose)

    return 0 if result.verdict.verdict != "red" else 1


# --------------------------------------------------------------------------
# batch
# --------------------------------------------------------------------------


IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _iter_images(folder: Path):
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in IMAGE_EXT:
            yield p


def cmd_batch(args: argparse.Namespace) -> int:
    from edgemedicheck.pipeline import scan_from_file

    db.init_db(args.db)
    today = date.fromisoformat(args.today) if args.today else None
    folder = Path(args.folder)

    files = list(_iter_images(folder))
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"No images found in {folder}", file=sys.stderr)
        return 1

    print(f"Scanning {len(files)} image(s) from {folder}\n")
    header = f"{'file':<44} {'verdict':<8} {'reason':<19} {'batch':<11} {'exp':<10} {'vis':>5} {'ms':>6}"
    print(header)
    print("-" * len(header))

    counts: dict[str, int] = {}
    latencies: list[float] = []

    for path in files:
        try:
            r = scan_from_file(
                path, today=today, db_path=args.db,
                write_log=not args.no_log, save_image=False,
            )
        except Exception as exc:
            print(f"{path.name[:43]:<44} ERROR: {exc}")
            continue

        counts[r.verdict.verdict] = counts.get(r.verdict.verdict, 0) + 1
        latencies.append(r.total_ms)
        print(
            f"{path.name[:43]:<44} "
            f"{_badge(r.verdict.verdict)} "
            f"{r.verdict.reason_code:<19} "
            f"{(r.ocr.batch_number or '-'):<11} "
            f"{(r.ocr.exp_date.iso if r.ocr.exp_date else '-'):<10} "
            f"{r.visual.suspicion_score:>5.2f} "
            f"{r.total_ms:>6.0f}"
        )

    print()
    print("Summary")
    for verdict in ("green", "yellow", "red"):
        if verdict in counts:
            print(f"  {_badge(verdict)} {counts[verdict]}")
    if latencies:
        latencies.sort()
        mean = sum(latencies) / len(latencies)
        p95 = latencies[int(len(latencies) * 0.95) - 1] if len(latencies) > 1 else latencies[0]
        print(f"  latency: mean {mean:.0f} ms, "
              f"min {latencies[0]:.0f}, max {latencies[-1]:.0f}, p95 {p95:.0f}")
    return 0


# --------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Score the pipeline against a ground-truth manifest.

    Reports the metrics named in Table VI: OCR field accuracy, date parsing
    accuracy, database-match correctness, and end-to-end verdict correctness.
    """
    from edgemedicheck.pipeline import scan_from_file

    db.init_db(args.db)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    today = date.fromisoformat(args.today) if args.today else date.today()

    totals = {
        "n": 0,
        "batch_correct": 0,
        "exp_correct": 0,
        "mfg_correct": 0,
        "batch_read": 0,
        "exp_read": 0,
        "mfg_read": 0,
        "verdict_correct": 0,
    }
    confusion: dict[tuple[str, str], int] = {}
    latencies: list[float] = []
    failures: list[dict] = []

    for entry in manifest:
        path = root / entry["file"]
        if not path.exists():
            continue
        try:
            r = scan_from_file(
                path, today=today, db_path=args.db,
                write_log=False, save_image=False,
            )
        except Exception as exc:
            failures.append({"file": entry["file"], "error": str(exc)})
            continue

        totals["n"] += 1
        latencies.append(r.total_ms)

        # OCR field accuracy.
        if r.ocr.batch_number:
            totals["batch_read"] += 1
            if r.ocr.batch_number.upper() == entry["batch_number"].upper():
                totals["batch_correct"] += 1

        if r.ocr.exp_date:
            totals["exp_read"] += 1
            truth = date.fromisoformat(entry["exp_date"])
            if (r.ocr.exp_date.year, r.ocr.exp_date.month) == (truth.year, truth.month):
                totals["exp_correct"] += 1

        if r.ocr.mfg_date:
            totals["mfg_read"] += 1
            truth = date.fromisoformat(entry["mfg_date"])
            if (r.ocr.mfg_date.year, r.ocr.mfg_date.month) == (truth.year, truth.month):
                totals["mfg_correct"] += 1

        # End-to-end verdict.
        expected = entry["expected_verdict"]
        got = r.verdict.verdict
        ok = (got in ("yellow", "red")) if expected == "yellow_or_red" else (got == expected)
        if ok:
            totals["verdict_correct"] += 1
        else:
            failures.append({
                "file": entry["file"],
                "expected": expected,
                "got": got,
                "reason": r.verdict.reason_code,
                "batch_truth": entry["batch_number"],
                "batch_read": r.ocr.batch_number,
                "exp_truth": entry["exp_date"],
                "exp_read": r.ocr.exp_date.iso if r.ocr.exp_date else None,
            })

        key = (entry["kind"], got)
        confusion[key] = confusion.get(key, 0) + 1

    n = max(totals["n"], 1)

    def pct(a: int, b: int) -> str:
        return f"{(100.0 * a / b):5.1f}%" if b else "    -"

    print()
    print(f"Evaluated {totals['n']} image(s) against {manifest_path}")
    print(f"Reference date: {today.isoformat()}")
    print()
    print("OCR field extraction")
    print(f"  batch  read {totals['batch_read']:>3}/{n:<3} "
          f"correct {totals['batch_correct']:>3}  "
          f"precision {pct(totals['batch_correct'], totals['batch_read'])}  "
          f"recall {pct(totals['batch_correct'], n)}")
    print(f"  expiry read {totals['exp_read']:>3}/{n:<3} "
          f"correct {totals['exp_correct']:>3}  "
          f"precision {pct(totals['exp_correct'], totals['exp_read'])}  "
          f"recall {pct(totals['exp_correct'], n)}")
    print(f"  mfg    read {totals['mfg_read']:>3}/{n:<3} "
          f"correct {totals['mfg_correct']:>3}  "
          f"precision {pct(totals['mfg_correct'], totals['mfg_read'])}  "
          f"recall {pct(totals['mfg_correct'], n)}")
    print()
    print("End-to-end verdict")
    print(f"  correct {totals['verdict_correct']}/{totals['n']}  "
          f"({pct(totals['verdict_correct'], totals['n'])})")
    print()
    print("Confusion (ground truth -> verdict)")
    kinds = sorted({k for k, _ in confusion})
    verdicts = ["green", "yellow", "red"]
    print(f"  {'truth':<12} " + " ".join(f"{v:>8}" for v in verdicts))
    for kind in kinds:
        row = " ".join(f"{confusion.get((kind, v), 0):>8}" for v in verdicts)
        print(f"  {kind:<12} {row}")

    if latencies:
        latencies.sort()
        print()
        print(f"Latency  mean {sum(latencies)/len(latencies):.0f} ms  "
              f"min {latencies[0]:.0f}  max {latencies[-1]:.0f}")

    if failures and args.show_failures:
        print()
        print(f"Mismatches ({len(failures)})")
        for f in failures[:20]:
            print(f"  {f}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "totals": totals,
            "confusion": {f"{k[0]}->{k[1]}": v for k, v in confusion.items()},
            "failures": failures,
            "latency_ms": latencies,
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
            "reference_date": today.isoformat(),
        }, indent=2))
        print(f"\nWrote report to {args.out}")

    print()
    print(_c("Note: if this was run on synthetic images, these numbers measure "
             "the pipeline against a renderer, not against real packaging.",
             "grey"))
    return 0


# --------------------------------------------------------------------------
# stats / products
# --------------------------------------------------------------------------


def cmd_stats(args: argparse.Namespace) -> int:
    db.init_db(args.db)
    s = db.scan_stats(args.db)
    print(f"Total scans: {s['total_scans']}")
    for verdict, count in sorted(s["by_verdict"].items()):
        print(f"  {_badge(verdict)} {count}")
    lat = s["latency_ms"]
    if lat["mean"]:
        print(f"Latency  mean {lat['mean']} ms  min {lat['min']}  max {lat['max']}")
    print(f"Batch records: {db.count_products(args.db)}")
    return 0


def cmd_products(args: argparse.Namespace) -> int:
    db.init_db(args.db)
    records = db.all_products(args.db)
    if not records:
        print("No batch records. Run: python run.py seed --demo")
        return 0
    print(f"{'batch':<12} {'product':<14} {'exp':<12} {'status':<10} manufacturer")
    print("-" * 78)
    for r in records:
        colour = {"valid": "green", "expired": "red", "recalled": "red",
                  "nsq": "red", "spurious": "red"}.get(r.status, "yellow")
        print(f"{r.batch_number:<12} {r.product_name[:13]:<14} {r.exp_date:<12} "
              f"{_c(f'{r.status:<10}', colour)} {r.manufacturer[:30]}")
    print(f"\n{len(records)} record(s)")
    return 0


# --------------------------------------------------------------------------
# serve
# --------------------------------------------------------------------------


def _is_usable_lan_ip(ip: str) -> bool:
    """Reject loopback, link-local autoconfiguration, and unspecified addresses."""
    if not ip or ip.startswith("127.") or ip == "0.0.0.0":
        return False
    if ip.startswith("169.254."):  # APIPA: no DHCP was reached
        return False
    return True


def lan_ip() -> str | None:
    """Best-effort LAN address of this machine.

    Two strategies, because the target deployment is an *offline* pharmacy
    network:

    1. UDP route probe. Opens a datagram socket towards an address and reads
       back which local interface the OS chose. No packet is sent. This is the
       usual method, but it needs a matching route -- on a network with no
       default gateway it raises ENETUNREACH and finds nothing.

    2. Interface enumeration. Falls back to asking the OS directly for the
       addresses bound to this host. This is what covers the isolated-LAN case
       the scanner is actually designed for: switch and access point present,
       no internet uplink.

    `socket.gethostbyname(gethostname())` alone is deliberately not trusted:
    on many Linux systems it returns 127.0.1.1 from /etc/hosts, which would
    print a URL no other device can reach.
    """
    import socket

    # 1. Route probe. Private ranges first so a LAN-only network still matches.
    for probe in ("192.168.1.1", "10.255.255.255", "172.16.0.1", "8.8.8.8"):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.settimeout(0.2)
            s.connect((probe, 1))
            ip = s.getsockname()[0]
            if _is_usable_lan_ip(ip):
                return ip
        except OSError:
            continue
        finally:
            s.close()

    # 2. Interface enumeration.
    candidates: list[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            candidates.append(info[4][0])
    except OSError:
        pass

    # POSIX: read the interface list directly. getaddrinfo only reports what
    # hostname resolution knows about, which on a DHCP-less LAN is often
    # nothing.
    try:
        import subprocess

        out = subprocess.run(
            ["ip", "-4", "-o", "addr", "show", "scope", "global"],
            capture_output=True, text=True, timeout=2,
        ).stdout
        for line in out.splitlines():
            parts = line.split()
            if "inet" in parts:
                candidates.append(parts[parts.index("inet") + 1].split("/")[0])
    except Exception:
        pass

    for ip in candidates:
        if _is_usable_lan_ip(ip):
            return ip
    return None


def print_qr(url: str) -> bool:
    """Print a scannable QR code for `url` in the terminal, if possible."""
    try:
        import qrcode  # type: ignore
    except ImportError:
        return False

    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    qr.make(fit=True)
    m = qr.get_matrix()

    # Two vertical modules per character cell using half-block glyphs, so the
    # code stays square in a terminal with non-square character cells.
    for y in range(0, len(m), 2):
        row = ""
        for x in range(len(m[0])):
            top = m[y][x]
            bottom = m[y + 1][x] if y + 1 < len(m) else False
            row += {(True, True): "█", (True, False): "▀",
                    (False, True): "▄", (False, False): " "}[(top, bottom)]
        print("  " + row)
    return True


def cmd_serve(args: argparse.Namespace) -> int:
    from app import create_app

    db.init_db(args.db)
    app = create_app(db_path=args.db, folder=args.folder, backend=args.backend)

    port = args.port or CONFIG.web.port
    if args.host:
        host = args.host
    elif args.local_only:
        host = "127.0.0.1"
    else:
        # Bind all interfaces so phones and tablets on the pharmacy Wi-Fi can
        # reach the scanner. This is the normal counter deployment.
        host = "0.0.0.0"

    ip = lan_ip()
    scheme = "https" if args.https else "http"
    lan_url = f"{scheme}://{ip}:{port}" if ip else None

    print()
    print(_c("  EdgeMediCheck", "bold"))
    print(f"  This device      {scheme}://127.0.0.1:{port}")
    if host == "0.0.0.0":
        if lan_url:
            print(f"  Other devices    {_c(lan_url, 'green')}")
        else:
            print(_c("  Other devices    could not determine this machine's "
                     "LAN address", "yellow"))
            print("                   run `hostname -I` (Linux) or "
                  "`ipconfig` (Windows) to find it")
    else:
        print(_c(f"  Bound to {host} only -- not reachable from other devices.",
                 "grey"))

    if host == "0.0.0.0" and lan_url:
        print()
        print("  Scan a phone camera at the pharmacy counter by opening the "
              "address above.")
        if args.qr:
            print()
            if not print_qr(lan_url):
                print(_c("  QR output needs the qrcode package: "
                         "pip install qrcode", "grey"))

    if host == "0.0.0.0":
        print()
        if args.https:
            print(_c(
                "  Traffic is encrypted, but the certificate is self-signed, "
                "so each\n"
                "  device shows a warning once and you accept it deliberately. "
                "There is\n"
                "  still no login: run this on a trusted pharmacy network "
                "only.",
                "yellow",
            ))
        else:
            print(_c(
                "  Note: there is no login on this interface and traffic is "
                "plain HTTP.\n"
                "  Run it on a trusted pharmacy network only, never on public "
                "Wi-Fi or a\n"
                "  network reachable from the internet. Use --local-only to "
                "restrict access\n"
                "  to this machine.",
                "yellow",
            ))
            print()
            print(_c(
                "  The live screen will fall back to this machine's camera. "
                "Browsers only\n"
                "  grant a page access to the *viewing device's* camera over "
                "HTTPS, so to\n"
                "  scan with a phone or laptop camera, restart with --https.",
                "grey",
            ))
    print()

    ssl_context = None
    if args.https:
        from edgemedicheck.tls import ensure_certificate

        from edgemedicheck.config import DATA_DIR

        cert, key = ensure_certificate(DATA_DIR / "certs",
                                       hosts=[ip] if ip else [])
        ssl_context = (str(cert), str(key))

    app.run(host=host, port=port, debug=args.debug, threaded=True,
            ssl_context=ssl_context)
    return 0


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="run.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--db", default=None, help="path to the SQLite database")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="create the database schema")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("seed", help="load batch records")
    p.add_argument("--demo", action="store_true", help="insert built-in demo records")
    p.add_argument("--from-manifest", help="seed from a dataset manifest.json")
    p.add_argument("--from-csv", help="seed from a CSV of batch records")
    p.add_argument("--include-suspicious", action="store_true",
                   help="also seed batches from suspicious samples")
    p.set_defaults(func=cmd_seed)

    p = sub.add_parser("collect", help="guided capture of a labelled dataset")
    p.add_argument("--out", default="data/collected", help="output directory")
    p.add_argument("--label", default="genuine",
                   choices=["genuine", "suspicious"],
                   help="class for this collection session")
    p.add_argument("--shots", type=int, default=8,
                   help="images to capture per product")
    p.add_argument("--backend", default=None,
                   choices=["auto", "picamera", "webcam", "folder"])
    p.add_argument("--folder", default=None)
    p.add_argument("--auto", action="store_true",
                   help="capture on a timer instead of waiting for Enter")
    p.add_argument("--interval", type=float, default=1.5,
                   help="seconds between shots in --auto mode")
    p.add_argument("--keep-all", action="store_true",
                   help="keep frames that fail the quality check")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("splits", help="inspect and freeze a dataset split")
    p.add_argument("data", help="dataset root (with genuine/ and suspicious/)")
    p.add_argument("--regime", default="both",
                   choices=["pack", "product", "both"])
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--test-frac", type=float, default=0.0)
    p.add_argument("--holdout", nargs="*", default=None,
                   help="explicit products to hold out (product regime)")
    p.add_argument("--n-holdout", type=int, default=None,
                   help="how many products to hold out (product regime)")
    p.add_argument("--manifest", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="write the split to this JSON file")
    p.set_defaults(func=cmd_splits)

    p = sub.add_parser("feedback", help="review or export pharmacist corrections")
    p.add_argument("--export", default=None,
                   help="write corrections as a labelled dataset here")
    p.add_argument("--all", action="store_true",
                   help="include corrections already exported")
    p.add_argument("--dry-run", action="store_true",
                   help="copy files but do not mark them exported")
    p.set_defaults(func=cmd_feedback)

    p = sub.add_parser(
        "calibrate",
        help="fit the visual heuristic on known-genuine reference images",
    )
    p.add_argument("folder", help="folder of known-genuine package images")
    p.add_argument("--out", default=None, help="where to write the calibration")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("scan", help="scan one image or capture live")
    p.add_argument("image", nargs="?", help="path to an image file")
    p.add_argument("--live", action="store_true", help="capture from the camera")
    p.add_argument("--backend", default=None,
                   choices=["auto", "picamera", "webcam", "folder"])
    p.add_argument("--folder", default=None, help="folder for the folder backend")
    p.add_argument("--today", default=None, help="override today's date (YYYY-MM-DD)")
    p.add_argument("--json", action="store_true", help="emit JSON")
    p.add_argument("--verbose", "-v", action="store_true", help="show raw OCR text")
    p.add_argument("--save", action="store_true", help="save the processed image")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("batch", help="scan every image in a folder")
    p.add_argument("folder")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--today", default=None)
    p.add_argument("--no-log", action="store_true", help="do not write scan-log rows")
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("evaluate", help="score against a ground-truth manifest")
    p.add_argument("manifest")
    p.add_argument("--today", default=None)
    p.add_argument("--out", default=None, help="write a JSON report here")
    p.add_argument("--show-failures", action="store_true")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("stats", help="scan-log statistics")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("products", help="list batch records")
    p.set_defaults(func=cmd_products)

    p = sub.add_parser("serve", help="start the web interface")
    p.add_argument("--host", default=None,
                   help="bind address (default 0.0.0.0, reachable on the LAN)")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--local-only", action="store_true",
                   help="bind 127.0.0.1 only; not reachable from other devices")
    p.add_argument("--qr", action="store_true",
                   help="print a QR code of the LAN URL (needs `pip install qrcode`)")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--backend", default=None,
                   choices=["auto", "picamera", "webcam", "folder"])
    p.add_argument("--folder", default=None,
                   help="image folder used by the folder backend / demo mode")
    p.add_argument("--https", action="store_true",
                   help="serve over TLS with a self-signed certificate. "
                        "Required for the live screen to use the camera of "
                        "the device viewing it: browsers refuse camera access "
                        "to a page served over plain HTTP on a LAN address. "
                        "Each device shows a certificate warning once.")
    p.set_defaults(func=cmd_serve)

    return ap


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
