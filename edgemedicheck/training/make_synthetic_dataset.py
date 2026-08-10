#!/usr/bin/env python3
"""
Synthetic medicine-package image generator.

Purpose and honest scoping
--------------------------
This generator exists to make the pipeline runnable and testable *before* the
real image dataset described in Section V-A of the paper has been collected.
It renders plausible medicine cartons and blister strips with controllable
defects, so every branch of Algorithms 1-3 can be exercised deterministically.

These images are NOT a substitute for real data. A CNN trained only on this
output will learn the renderer, not counterfeit packaging. Use synthetic data
for:

  * unit and integration testing of the pipeline,
  * verifying OCR field extraction and date parsing across label formats,
  * demonstrating the decision-fusion logic,
  * smoke-testing latency.

Do not use it for the accuracy numbers reported in the paper.

Usage
-----
    python training/make_synthetic_dataset.py --out data/images --count 40
    python training/make_synthetic_dataset.py --out data/dataset --split
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --------------------------------------------------------------------------
# Content pools
# --------------------------------------------------------------------------

# A real product SKU has a fixed identity: the same brand colour, the same
# package form, and the same manufacturer on every pack. That consistency is
# precisely what a reference-based visual check exploits, so the generator must
# reproduce it. Each entry therefore pins appearance to the brand rather than
# randomising it per sample.
#
# (brand, strength, generic, brand_colour, form, manufacturer)
PRODUCTS = [
    ("PARACIP",   "500 mg", "Paracetamol Tablets IP",
     (24, 92, 168),   "carton", "Cipla Pharmaceuticals Ltd"),
    ("AZITHRAL",  "250 mg", "Azithromycin Tablets IP",
     (16, 122, 82),   "carton", "Alkem Laboratories Ltd"),
    ("CROCIN",    "650 mg", "Paracetamol Tablets IP",
     (168, 42, 48),   "carton", "Sun Pharma Laboratories Ltd"),
    ("AMOXYCLAV", "625 mg", "Amoxicillin & Clavulanate Tablets",
     (92, 44, 140),   "carton", "Intas Pharmaceuticals Ltd"),
    ("PANTOCID",  "40 mg",  "Pantoprazole Tablets IP",
     (196, 108, 16),  "carton", "Sun Pharma Laboratories Ltd"),
    ("METFOR",    "500 mg", "Metformin HCl Tablets IP",
     (18, 108, 120),  "strip",  "Micro Labs Limited"),
    ("CETZINE",   "10 mg",  "Cetirizine Tablets IP",
     (140, 30, 90),   "strip",  "Torrent Pharmaceuticals Ltd"),
    ("DOLO",      "650 mg", "Paracetamol Tablets IP",
     (200, 60, 30),   "carton", "Micro Labs Limited"),
    ("MONTEK",    "10 mg",  "Montelukast Tablets IP",
     (40, 70, 150),   "strip",  "Zydus Healthcare Ltd"),
    ("ZIFI",      "200 mg", "Cefixime Tablets IP",
     (30, 130, 60),   "carton", "Mankind Pharma Ltd"),
]

MANUFACTURERS = [
    "Cipla Pharmaceuticals Ltd",
    "Sun Pharma Laboratories Ltd",
    "Alkem Laboratories Ltd",
    "Mankind Pharma Ltd",
    "Torrent Pharmaceuticals Ltd",
    "Zydus Healthcare Ltd",
    "Micro Labs Limited",
    "Intas Pharmaceuticals Ltd",
]

# The label format variants the date parser must handle.
DATE_STYLES = [
    ("{m:02d}/{y}", "full"),        # 08/2027
    ("{m:02d}/{yy}", "short"),      # 08/27
    ("{m:02d}-{y}", "full"),        # 08-2027
    ("{M} {y}", "name"),            # AUG 2027
    ("{M}-{yy}", "name_short"),     # AUG-27
    ("{d:02d}/{m:02d}/{y}", "dmy"), # 31/08/2027
    ("{m:02d}.{y}", "full"),        # 08.2027
]

MONTH_ABBR = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

BRAND_COLOURS = [
    (24, 92, 168), (16, 122, 82), (168, 42, 48), (92, 44, 140),
    (196, 108, 16), (18, 108, 120), (140, 30, 90),
]


# --------------------------------------------------------------------------
# Font handling
# --------------------------------------------------------------------------


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Find a usable TrueType font, falling back to PIL's bitmap default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


# --------------------------------------------------------------------------
# Date formatting
# --------------------------------------------------------------------------


def format_date(d: date, style: tuple[str, str]) -> str:
    template, _ = style
    return template.format(
        d=d.day,
        m=d.month,
        y=d.year,
        yy=d.year % 100,
        M=MONTH_ABBR[d.month - 1],
    )


def random_batch(rng: random.Random) -> str:
    letters = "".join(rng.choices("ABCDEFGHJKLMNPRSTUVWXYZ", k=rng.choice([1, 2, 3])))
    digits = "".join(rng.choices("0123456789", k=rng.choice([4, 5, 6])))
    return f"{letters}{digits}"


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render_carton(
    spec: dict, rng: random.Random, width: int = 900, height: int = 560
) -> Image.Image:
    """Render a medicine carton face with the standard Indian label fields."""
    colour = spec["brand_colour"]
    img = Image.new("RGB", (width, height), (252, 251, 248))
    draw = ImageDraw.Draw(img)

    # Brand band across the top.
    band_h = int(height * 0.28)
    draw.rectangle([0, 0, width, band_h], fill=colour)

    f_brand = load_font(int(height * 0.11), bold=True)
    f_strength = load_font(int(height * 0.055), bold=True)
    f_generic = load_font(int(height * 0.042))
    f_field = load_font(int(height * 0.040), bold=True)
    f_value = load_font(int(height * 0.040))
    f_small = load_font(int(height * 0.032))
    f_mfr = load_font(int(height * 0.036), bold=True)

    m = int(width * 0.045)

    draw.text((m, int(band_h * 0.18)), spec["brand"], font=f_brand, fill="white")
    draw.text(
        (width - m - 150, int(band_h * 0.38)),
        spec["strength"],
        font=f_strength,
        fill="white",
    )

    y = band_h + int(height * 0.05)
    draw.text((m, y), spec["generic"], font=f_generic, fill=(40, 40, 40))
    y += int(height * 0.09)

    # Prescription warning rule -- present on real Schedule H packs.
    draw.line([(m, y), (width - m, y)], fill=(180, 30, 30), width=2)
    y += int(height * 0.02)
    draw.text(
        (m, y),
        "Schedule H Prescription Drug - Caution",
        font=f_small,
        fill=(180, 30, 30),
    )
    y += int(height * 0.085)

    # Field block. `field_dx` lets us introduce misalignment for fake packs.
    dx = spec.get("field_dx", 0)
    rows = [
        ("B.No.", spec["batch"]),
        ("Mfg. Date", spec["mfg_text"]),
        ("Exp. Date", spec["exp_text"]),
    ]
    label_w = int(width * 0.22)
    for i, (label, value) in enumerate(rows):
        row_dx = dx if i == spec.get("misaligned_row", -1) else 0
        draw.text((m, y), label, font=f_field, fill=(30, 30, 30))
        draw.text((m + label_w + row_dx, y), value, font=f_value, fill=(20, 20, 20))
        y += int(height * 0.065)

    y += int(height * 0.02)
    draw.text((m, y), "Mfd. by:", font=f_small, fill=(90, 90, 90))
    y += int(height * 0.042)
    draw.text((m, y), spec["manufacturer"], font=f_mfr, fill=(30, 30, 30))

    # Hologram patch. Genuine packs get a multi-hue gradient; fakes get a flat
    # printed rectangle, which is exactly the cue the CNN module targets.
    hx, hy = width - int(width * 0.20), height - int(height * 0.30)
    hw, hh = int(width * 0.14), int(height * 0.20)
    if spec["hologram"] == "genuine":
        for i in range(hh):
            t = i / max(hh - 1, 1)
            r = int(120 + 135 * abs(np.sin(t * 3.1)))
            g = int(120 + 135 * abs(np.sin(t * 3.1 + 2.0)))
            b = int(120 + 135 * abs(np.sin(t * 3.1 + 4.0)))
            draw.line([(hx, hy + i), (hx + hw, hy + i)], fill=(r, g, b))
        draw.rectangle([hx, hy, hx + hw, hy + hh], outline=(200, 200, 200))
    elif spec["hologram"] == "flat":
        draw.rectangle([hx, hy, hx + hw, hy + hh], fill=(178, 178, 182),
                       outline=(150, 150, 150))
    # "none" draws nothing at all.

    # Barcode block for visual realism (never decoded by the pipeline).
    bx, by = m, height - int(height * 0.16)
    x = bx
    while x < bx + int(width * 0.32):
        w = rng.choice([2, 2, 3, 5])
        if rng.random() > 0.35:
            draw.rectangle([x, by, x + w, by + int(height * 0.10)], fill=(20, 20, 20))
        x += w + rng.choice([2, 3])

    return img


def render_strip(
    spec: dict, rng: random.Random, width: int = 900, height: int = 420
) -> Image.Image:
    """Render a blister strip: foil background, blisters, small printed text."""
    img = Image.new("RGB", (width, height), (196, 198, 203))
    draw = ImageDraw.Draw(img)

    # Foil sheen.
    for i in range(height):
        t = i / height
        v = int(178 + 46 * np.sin(t * 6.0))
        draw.line([(0, i), (width, i)], fill=(v, v, min(255, v + 6)))

    # Blister pockets.
    cols, rows = 5, 2
    pad_x, pad_y = int(width * 0.09), int(height * 0.22)
    cw = (width - 2 * pad_x) // cols
    ch = (height - 2 * pad_y) // rows
    for r in range(rows):
        for c in range(cols):
            cx = pad_x + c * cw + cw // 2
            cy = pad_y + r * ch + ch // 2
            rad = int(min(cw, ch) * 0.33)
            draw.ellipse(
                [cx - rad, cy - rad, cx + rad, cy + rad],
                fill=(214, 216, 220),
                outline=(160, 162, 168),
            )

    colour = spec["brand_colour"]
    f_brand = load_font(int(height * 0.085), bold=True)
    f_text = load_font(int(height * 0.052), bold=True)

    draw.text((int(width * 0.04), int(height * 0.04)),
              f"{spec['brand']} {spec['strength']}", font=f_brand, fill=colour)

    # Small print row along the bottom -- the hardest text for OCR, which is
    # the realistic case for batch codes on strips.
    y = height - int(height * 0.14)
    line = (
        f"B.No. {spec['batch']}   "
        f"MFG {spec['mfg_text']}   "
        f"EXP {spec['exp_text']}"
    )
    draw.text((int(width * 0.04), y), line, font=f_text, fill=(30, 30, 30))

    return img


# --------------------------------------------------------------------------
# Degradations
# --------------------------------------------------------------------------


def apply_defects(img: Image.Image, spec: dict, rng: random.Random) -> Image.Image:
    """Apply the visual degradations that distinguish suspicious packs."""
    defects = spec.get("defects", [])

    if "blur" in defects:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(1.4, 2.6)))

    if "colour_shift" in defects:
        arr = np.asarray(img).astype(np.float32)
        shift = np.array([
            rng.uniform(0.80, 1.28),
            rng.uniform(0.80, 1.28),
            rng.uniform(0.80, 1.28),
        ])
        arr = np.clip(arr * shift, 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))

    if "oversaturate" in defects:
        arr = np.asarray(img.convert("HSV")).astype(np.float32)
        arr[:, :, 1] = np.clip(arr[:, :, 1] * 1.55, 0, 255)
        img = Image.fromarray(arr.astype(np.uint8), mode="HSV").convert("RGB")

    if "noise" in defects:
        arr = np.asarray(img).astype(np.float32)
        arr += np.random.normal(0, rng.uniform(6, 14), arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    return img


def place_on_background(
    img: Image.Image, rng: random.Random, spec: dict
) -> Image.Image:
    """Composite onto the enclosure mat, with rotation and lighting variation.

    This models what the Pi Camera actually sees inside the fixed enclosure:
    a package on a neutral mat under an LED ring, never perfectly aligned.
    """
    angle = spec.get("rotation", rng.uniform(-4.0, 4.0))
    rotated = img.rotate(angle, expand=True, resample=Image.BICUBIC,
                         fillcolor=(60, 62, 66))

    pad = int(max(rotated.size) * 0.16)
    canvas = Image.new(
        "RGB",
        (rotated.width + 2 * pad, rotated.height + 2 * pad),
        (58, 60, 64),  # dark neutral enclosure mat
    )
    canvas.paste(rotated, (pad, pad))

    # LED ring falloff: brighter at centre, dimmer at the edges.
    arr = np.asarray(canvas).astype(np.float32)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    dist = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    falloff = np.clip(1.12 - 0.30 * dist, 0.55, 1.15)

    brightness = spec.get("brightness", 1.0)
    arr *= falloff[:, :, None] * brightness
    canvas = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    return canvas


# --------------------------------------------------------------------------
# Spec construction
# --------------------------------------------------------------------------


def build_spec(
    rng: random.Random, kind: str, today: date
) -> dict:
    """Build one sample specification.

    kind: "genuine" | "expired" | "suspicious"
    """
    brand, strength, generic, brand_colour, form, manufacturer = rng.choice(PRODUCTS)
    batch = random_batch(rng)

    mfg_style = rng.choice(DATE_STYLES)
    exp_style = rng.choice(DATE_STYLES)

    if kind == "expired":
        # Expiry 1-30 months in the past.
        months_past = rng.randint(1, 30)
        exp = _shift_months(today, -months_past)
        mfg = _shift_months(exp, -rng.randint(24, 36))
    else:
        months_future = rng.randint(4, 36)
        exp = _shift_months(today, months_future)
        mfg = _shift_months(exp, -rng.randint(24, 36))

    spec = {
        "kind": kind,
        "brand": brand,
        "strength": strength,
        "generic": generic,
        "manufacturer": manufacturer,
        "batch": batch,
        "mfg_date": mfg,
        "exp_date": exp,
        "mfg_text": format_date(mfg, mfg_style),
        "exp_text": format_date(exp, exp_style),
        # Fixed by the SKU, not resampled -- see the PRODUCTS table.
        "brand_colour": brand_colour,
        "form": form,
        "hologram": "genuine",
        "defects": [],
        # Placement and lighting still vary: the pack is put down by hand under
        # a real LED ring, so pose and exposure are never identical.
        "rotation": rng.uniform(-3.5, 3.5),
        "brightness": rng.uniform(0.94, 1.06),
    }

    if kind == "suspicious":
        # Pick 2-3 defects so the visual cues have something to find.
        pool = ["blur", "colour_shift", "oversaturate", "noise",
                "misalign", "flat_hologram", "no_hologram"]
        chosen = rng.sample(pool, k=rng.randint(2, 3))
        spec["defects"] = [d for d in chosen
                           if d not in ("misalign", "flat_hologram", "no_hologram")]
        if "misalign" in chosen:
            spec["misaligned_row"] = rng.randint(0, 2)
            spec["field_dx"] = rng.choice([-24, -16, 16, 24, 32])
            spec["rotation"] = rng.uniform(-9.0, 9.0)
        if "flat_hologram" in chosen:
            spec["hologram"] = "flat"
        if "no_hologram" in chosen:
            spec["hologram"] = "none"
        spec["brightness"] = rng.uniform(0.82, 1.18)

    return spec


def _shift_months(d: date, months: int) -> date:
    """Shift a date by N months, clamping the day to the target month length."""
    import calendar

    total = d.month - 1 + months
    year = d.year + total // 12
    month = total % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def render_sample(spec: dict, rng: random.Random) -> Image.Image:
    if spec["form"] == "strip":
        base = render_strip(spec, rng)
    else:
        base = render_carton(spec, rng)
    base = apply_defects(base, spec, rng)
    return place_on_background(base, rng, spec)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/images", help="output directory")
    ap.add_argument("--count", type=int, default=30, help="total images")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--split",
        action="store_true",
        help="write into genuine/ and suspicious/ subfolders for CNN training",
    )
    ap.add_argument(
        "--manifest",
        default=None,
        help="write ground-truth JSON here (default: <out>/manifest.json)",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    today = date.today()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.split:
        (out / "genuine").mkdir(exist_ok=True)
        (out / "suspicious").mkdir(exist_ok=True)

    # Roughly 50 / 20 / 30 across genuine, expired, suspicious.
    kinds: list[str] = (
        ["genuine"] * int(args.count * 0.5)
        + ["expired"] * int(args.count * 0.2)
        + ["suspicious"] * (args.count - int(args.count * 0.5) - int(args.count * 0.2))
    )
    rng.shuffle(kinds)

    manifest = []
    for i, kind in enumerate(kinds):
        spec = build_spec(rng, kind, today)
        img = render_sample(spec, rng)

        name = f"{i:03d}_{kind}_{spec['brand'].lower()}_{spec['batch']}.jpg"
        if args.split:
            # For CNN training, expired packs are visually genuine.
            folder = "suspicious" if kind == "suspicious" else "genuine"
            path = out / folder / name
        else:
            path = out / name

        img.save(path, quality=93)

        manifest.append({
            "file": str(path.relative_to(out)),
            "kind": kind,
            "expected_verdict": {
                "genuine": "green",
                "expired": "red",
                "suspicious": "yellow_or_red",
            }[kind],
            "product_name": spec["brand"],
            "generic": spec["generic"],
            "manufacturer": spec["manufacturer"],
            "batch_number": spec["batch"],
            "mfg_date": spec["mfg_date"].isoformat(),
            "exp_date": spec["exp_date"].isoformat(),
            "mfg_text": spec["mfg_text"],
            "exp_text": spec["exp_text"],
            "form": spec["form"],
            "defects": spec["defects"],
            "hologram": spec["hologram"],
        })

    manifest_path = Path(args.manifest) if args.manifest else out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    counts: dict[str, int] = {}
    for m in manifest:
        counts[m["kind"]] = counts.get(m["kind"], 0) + 1

    print(f"Wrote {len(manifest)} images to {out}")
    for k, v in sorted(counts.items()):
        print(f"  {k:<12} {v}")
    print(f"Ground truth: {manifest_path}")
    print(
        "\nNOTE: synthetic images are for pipeline testing only. "
        "Do not report accuracy numbers from them."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
