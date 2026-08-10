#!/usr/bin/env python3
"""
Build an out-of-distribution probe from WHO Medical Product Alerts.

Why
---
The training set is built from surrogates: genuine packs put through digital
analogues of reprinting, rephotographing and relabelling. A model can score
well on that and still be useless, because it may have learned the specific
signature of our surrogate code rather than anything about counterfeiting.

The only way to find out is to test on counterfeits we did not make. WHO's
Medical Product Alerts carry field photographs of products confirmed falsified
by the genuine manufacturer -- real counterfeits, photographed by regulators,
in no way derived from our pipeline. This script collects them.

Read the result carefully
-------------------------
WHO publishes photographs of the falsified product only. A probe with one
class in it measures recall and nothing else: a model that answers "fake" to
every input scores 100% on it. The number is only interpretable next to the
model's false-positive rate on genuine packs, and `train_model.py` reports
both together.

The probe is also OOD in ways that have nothing to do with counterfeiting --
different countries, different product types, vials and cartons rather than
Indian OTC blister packs, and photographs taken on whatever was to hand. A
drop in performance here is a lower bound on real-world transfer, not a clean
measurement of it. Say so when reporting it.

Only alerts titled "Falsified" are used. "Substandard" and "contaminated"
alerts concern genuine packaging with out-of-spec contents -- the box is
authentic, so including them would put genuine packs in the counterfeit class.

Usage
-----
    python build_ood_probe.py --out probe/who
"""

from __future__ import annotations

import argparse
import io
import json
import re
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

try:
    import pymupdf
except ImportError:
    raise SystemExit("Needs pymupdf: pip install pymupdf")

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")

INDEX = ("https://www.who.int/teams/regulation-prequalification/"
         "incidents-and-SF/full-list-of-who-medical-product-alerts")

S = requests.Session()
S.headers["User-Agent"] = UA

MIN_PANEL = 90          # px, smallest side of a kept panel
MIN_COLOURS = 200       # same flat-graphic filter as the scraper
MIN_CONTENT = 0.12      # fraction of non-white pixels


def colour_complexity(img: Image.Image) -> int:
    t = img.convert("RGB")
    t.thumbnail((128, 128), Image.LANCZOS)
    a = np.asarray(t).reshape(-1, 3) // 8
    return len(np.unique(a, axis=0))


def split_runs(profile: np.ndarray, thresh: float, min_gap: int) -> list[tuple[int, int]]:
    """Segments of `profile` above `thresh`, separated by gaps of at least
    `min_gap` samples below it."""
    on = profile > thresh
    spans, start = [], None
    gap = 0
    for i, v in enumerate(on):
        if v:
            if start is None:
                start = i
            gap = 0
        else:
            if start is not None:
                gap += 1
                if gap >= min_gap:
                    spans.append((start, i - gap + 1))
                    start = None
    if start is not None:
        spans.append((start, len(on)))
    return spans


def panels(img: Image.Image) -> list[Image.Image]:
    """Cut a WHO comparison table into its individual photographs.

    The alerts lay their photographs out in a table with white gutters and
    white text cells above each photo. Splitting on white gutters gives the
    cells; the colour-complexity filter then drops the text-only ones, since
    black text on white has almost no distinct colours while a photograph of
    a carton has hundreds.
    """
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    content = (np.abs(a - 255).max(axis=2) > 14)
    if content.mean() < 0.02:
        return []

    out = []
    col_prof = content.mean(axis=0)
    for x0, x1 in split_runs(col_prof, 0.02, 4):
        strip = content[:, x0:x1]
        row_prof = strip.mean(axis=1)
        for y0, y1 in split_runs(row_prof, 0.02, 4):
            if (x1 - x0) < MIN_PANEL or (y1 - y0) < MIN_PANEL:
                continue
            crop = img.crop((x0, y0, x1, y1))
            block = content[y0:y1, x0:x1]
            if block.mean() < MIN_CONTENT:
                continue
            if colour_complexity(crop) < MIN_COLOURS:
                continue
            out.append(crop)
    return out


def alert_links() -> list[tuple[str, str]]:
    html = S.get(INDEX, timeout=40).text
    seen, out = set(), []
    for m in re.finditer(
            r'href="([^"]*?/news/item/[^"]*?)"[^>]*>(.*?)</a>', html, re.S):
        href, title = m.group(1), re.sub(r"<[^>]+>", " ", m.group(2))
        title = " ".join(title.split())
        if "falsified" not in title.lower():
            continue
        if href.startswith("/"):
            href = "https://www.who.int" + href
        if href in seen:
            continue
        seen.add(href)
        out.append((href, title))
    return out


def pdf_for(alert_url: str) -> str | None:
    try:
        html = S.get(alert_url, timeout=40).text
    except Exception:  # noqa: BLE001
        return None
    m = re.search(r'https://cdn\.who\.int/media/docs/[^"\'\\ ]+\.pdf[^"\'\\ ]*',
                  html)
    return m.group(0) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "Fake").mkdir(parents=True, exist_ok=True)
    mf = (out / "probe_manifest.jsonl").open("w")

    alerts = alert_links()
    print(f"{len(alerts)} 'Falsified' alerts\n")

    kept = 0
    for i, (url, title) in enumerate(alerts, 1):
        pdf_url = pdf_for(url)
        if not pdf_url:
            print(f"  [{i:2d}] no PDF   {title[:58]}")
            continue
        try:
            raw = S.get(pdf_url, timeout=60).content
            doc = pymupdf.open(stream=raw, filetype="pdf")
        except Exception as e:  # noqa: BLE001
            print(f"  [{i:2d}] pdf fail {type(e).__name__}  {title[:48]}")
            continue

        got = 0
        seen_xrefs: set[int] = set()
        for page in doc:
            for info in page.get_images(full=True):
                xref = info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                try:
                    pix = pymupdf.Pixmap(doc, xref)
                    if pix.n > 4:
                        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                    im = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                except Exception:  # noqa: BLE001
                    continue
                # The WHO banner logo repeats on every page.
                if im.width < 200 or im.height < 120:
                    continue
                for p in panels(im):
                    p2 = p.copy()
                    p2.thumbnail((args.size, args.size), Image.LANCZOS)
                    canvas = Image.new("RGB", (args.size, args.size),
                                       (255, 255, 255))
                    canvas.paste(p2, ((args.size - p2.width) // 2,
                                      (args.size - p2.height) // 2))
                    slug = re.sub(r"[^a-z0-9]+", "-", title.lower())[:60]
                    name = f"{slug}__{kept:04d}.jpg"
                    canvas.save(out / "Fake" / name, quality=88)
                    mf.write(json.dumps({
                        "file": f"Fake/{name}", "cls": "Fake",
                        "alert_title": title, "alert_url": url,
                        "pdf_url": pdf_url,
                        "orig_size": [p.width, p.height],
                    }) + "\n")
                    kept += 1
                    got += 1
        print(f"  [{i:2d}] {got:3d} panels  {title[:58]}", flush=True)
        time.sleep(0.4)

    mf.close()
    print(f"\n{'=' * 60}\n{kept} falsified-product panels -> {out}\n{'=' * 60}")
    print("Single-class probe: reports recall only. Read it beside the "
          "model's\nfalse-positive rate on genuine packs, never on its own.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
