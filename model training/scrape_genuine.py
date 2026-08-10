#!/usr/bin/env python3
"""
Build the genuine-packaging source pool.

Why this exists
---------------
The original dataset separated its classes by acquisition method: Real was
Google-Images thumbnails saved as JPEG, Fake was Windows screenshots saved as
PNG. `audit_dataset.py` showed a single threshold on bytes-per-pixel splitting
them at 98.6%, and the trained model duly reported 100% on everything.

The fix is not "more images". It is a single pool of genuine packaging
photographs, from which BOTH classes are derived, so that no acquisition
artefact correlates with the label. This script builds that pool.
`make_surrogates.py` then derives the counterfeit-surrogate class from it.

Sources
-------
Apollo Pharmacy and PharmEasy OTC catalogues, reached through their published
sitemaps. Both permit crawling of product pages in robots.txt; neither declares
a crawl-delay, so we self-limit. Product photography is studio-shot at
1200-1500px, which leaves room to downscale to 256px without upsampling
artefacts.

Only medicine-like packaging is kept -- tablets, capsules, syrups, injections,
ointments. Soaps and shampoos are dropped: their packaging carries none of the
batch/expiry/hologram structure the scanner is meant to read.

Usage
-----
    python scrape_genuine.py --out pool/genuine --target 2000
    python scrape_genuine.py --out pool/genuine --target 2000 --resume
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from PIL import Image

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")

SITEMAPS = {
    "apollo": "https://www.apollopharmacy.in/sitemap/sitemap-pharma-otc.xml",
    "pharmeasy": "https://pharmeasy.in/sitemaps/sitemap-otc-products.xml",
}

OG_A = re.compile(
    r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', re.I)
OG_B = re.compile(
    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']', re.I)
JSONLD = re.compile(r'"image"\s*:\s*"([^"]+)"')

# Packaging that carries printed batch/expiry panels, not cosmetics.
MEDICINE = re.compile(
    r"tablet|capsule|syrup|injection|\d+\s*mg\b|suspension|ointment|"
    r"drops|sachet|inhaler|granules|powder-for|\btab\b|\bcaps\b|infusion",
    re.I)
# Explicitly not medicine packaging, even when the slug matches above.
EXCLUDE = re.compile(
    r"soap|shampoo|face-wash|body-lotion|perfume|deodorant|diaper|"
    r"sanitary|condom|toothbrush|toothpaste|hand-wash|shower|makeup|"
    r"lipstick|kajal|hair-oil|hair-color|razor|wipes",
    re.I)

MIN_EDGE = 400
MIN_BYTES = 12_000


# ---------------------------------------------------------------- utilities

def dhash(img: Image.Image, size: int = 8) -> str:
    """Perceptual hash. Catches the same stock photo reused across products,
    which exact-hashing misses because the CDN re-encodes per URL."""
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.LANCZOS),
                   dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return "".join("01"[int(b)] for b in bits)


def colour_complexity(img: Image.Image) -> int:
    """Distinct quantised colours in a thumbnail.

    Both retailers serve a flat vector logo when a product has no photograph.
    Those pass the dimension and byte-size filters -- the SVG-derived PNG is
    large and 1200px -- but carry almost no colour. Measured over a sample,
    flat placeholders scored 18-84 and genuine pack photographs 305-1998, so
    the two populations are cleanly separable here.
    """
    t = img.convert("RGB")
    t.thumbnail((128, 128), Image.LANCZOS)
    a = np.asarray(t).reshape(-1, 3) // 8
    return len(np.unique(a, axis=0))


MIN_COLOURS = 200


def slug_of(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1]


class Fetcher:
    """Session pool with a global rate limit, so we stay a polite guest."""

    def __init__(self, rate_per_sec: float = 8.0):
        self._local = threading.local()
        self._lock = threading.Lock()
        self._interval = 1.0 / rate_per_sec
        self._next = 0.0

    @property
    def session(self) -> requests.Session:
        s = getattr(self._local, "s", None)
        if s is None:
            s = requests.Session()
            s.headers["User-Agent"] = UA
            self._local.s = s
        return s

    def _throttle(self):
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next - now)
            self._next = max(now, self._next) + self._interval
        if wait:
            time.sleep(wait)

    def get(self, url: str, timeout: int = 25, tries: int = 3):
        last = None
        for attempt in range(tries):
            self._throttle()
            try:
                r = self.session.get(url, timeout=timeout)
                if r.status_code == 200:
                    return r
                # 404 is a dead product page, not worth retrying.
                if r.status_code in (404, 410):
                    return None
                last = f"HTTP {r.status_code}"
            except Exception as e:  # noqa: BLE001
                last = type(e).__name__
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(last or "failed")


FETCH = Fetcher()


def sitemap_urls(index_url: str) -> list[str]:
    """Expand a sitemap index one level into product URLs."""
    r = FETCH.get(index_url, timeout=40)
    if r is None:
        return []
    children = re.findall(r"<loc>([^<]+)</loc>", r.text)
    if not children or not children[0].endswith(".xml"):
        return children
    out: list[str] = []
    for child in children:
        try:
            rc = FETCH.get(child, timeout=60)
        except RuntimeError:
            continue
        if rc is not None:
            out.extend(re.findall(r"<loc>([^<]+)</loc>", rc.text))
    return out


def first_image_url(html: str) -> str | None:
    for rx in (OG_A, OG_B, JSONLD):
        m = rx.search(html)
        if m:
            u = m.group(1).replace("\\u002F", "/").replace("&amp;", "&")
            if u.startswith("//"):
                u = "https:" + u
            if u.startswith("http") and not u.endswith(".svg"):
                return u
    return None


# ------------------------------------------------------------------ worker

def harvest(job: tuple[str, str]) -> dict | None:
    site, page_url = job
    try:
        r = FETCH.get(page_url)
    except RuntimeError as e:
        return {"status": "page_fail", "why": str(e)}
    if r is None:
        return {"status": "page_404"}

    img_url = first_image_url(r.text)
    if not img_url:
        return {"status": "no_image_field"}

    try:
        ri = FETCH.get(img_url)
    except RuntimeError as e:
        return {"status": "img_fail", "why": str(e)}
    if ri is None:
        return {"status": "img_404"}

    raw = ri.content
    if len(raw) < MIN_BYTES:
        return {"status": "too_small_bytes"}
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
    except Exception:  # noqa: BLE001
        return {"status": "decode_fail"}
    if max(im.size) < MIN_EDGE:
        return {"status": "too_small_dims"}
    ncol = colour_complexity(im)
    if ncol < MIN_COLOURS:
        return {"status": "flat_placeholder"}

    return {
        "colours": ncol,
        "status": "ok",
        "site": site,
        "page_url": page_url,
        "image_url": img_url,
        "slug": slug_of(page_url),
        "width": im.size[0],
        "height": im.size[1],
        "phash": dhash(im),
        "sha1": hashlib.sha1(raw).hexdigest(),
        "_image": im.convert("RGB"),
    }


# -------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output pool directory")
    ap.add_argument("--target", type=int, default=2000,
                    help="how many distinct packs to keep")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rate", type=float, default=8.0,
                    help="max requests/second across all workers")
    ap.add_argument("--resume", action="store_true",
                    help="keep whatever is already in --out and top it up")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    global FETCH
    FETCH = Fetcher(args.rate)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.jsonl"

    seen_phash: set[str] = set()
    seen_slug: set[str] = set()
    kept = 0
    if args.resume and manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            seen_phash.add(rec["phash"])
            seen_slug.add(rec["slug"])
            kept += 1
        print(f"Resuming: {kept} already in pool")
    elif manifest_path.exists():
        print(f"{manifest_path} exists. Pass --resume to top it up.",
              file=sys.stderr)
        return 1

    # ---- build the candidate list ------------------------------------
    print("Reading sitemaps...")
    candidates: list[tuple[str, str]] = []
    for site, idx in SITEMAPS.items():
        urls = sitemap_urls(idx)
        med = [u for u in urls
               if MEDICINE.search(u) and not EXCLUDE.search(u)]
        med = [u for u in med if slug_of(u) not in seen_slug]
        print(f"  {site:10s} {len(urls):6d} urls -> {len(med):6d} "
              f"medicine-like, unseen")
        candidates.extend((site, u) for u in med)

    if not candidates:
        print("No candidate product pages found.", file=sys.stderr)
        return 1

    random.seed(args.seed)
    random.shuffle(candidates)
    print(f"\n{len(candidates)} candidate pages, target {args.target} packs\n")

    stats: Counter = Counter()
    mf = manifest_path.open("a")
    t0 = time.monotonic()

    def flush(rec: dict):
        nonlocal kept
        im = rec.pop("_image")
        name = f"{rec['site']}__{rec['slug'][:70]}"
        # Lossless intermediate; make_surrogates.py does the uniform encode.
        path = out / f"{name}.png"
        n = 1
        while path.exists():
            path = out / f"{name}__{n}.png"
            n += 1
        im.save(path)
        rec["file"] = path.name
        mf.write(json.dumps(rec) + "\n")
        mf.flush()
        kept += 1

    # Submit in chunks rather than mapping the whole candidate list: with
    # ~50k candidates, an eager map would queue every one of them and the
    # executor would refuse to shut down until all had run, long after the
    # target was reached.
    chunk = max(args.workers * 8, 32)
    done = False
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for start in range(0, len(candidates), chunk):
            if done:
                break
            batch = candidates[start:start + chunk]
            for rec in ex.map(harvest, batch):
                if rec is None:
                    stats["none"] += 1
                    continue
                st = rec["status"]
                if st != "ok":
                    stats[st] += 1
                    continue
                if rec["phash"] in seen_phash:
                    stats["dupe_phash"] += 1
                    continue
                if rec["slug"] in seen_slug:
                    stats["dupe_slug"] += 1
                    continue
                seen_phash.add(rec["phash"])
                seen_slug.add(rec["slug"])
                flush(rec)
                stats["ok"] += 1

                if kept % 100 == 0:
                    el = time.monotonic() - t0
                    print(f"  {kept:5d} kept   {el/60:5.1f} min   "
                          f"{kept/max(el,1)*60:.0f}/min", flush=True)
                if kept >= args.target:
                    done = True
                    break

    mf.close()
    print(f"\n{'=' * 60}\nPOOL: {kept} distinct packs in {out}\n{'=' * 60}")
    for k, v in stats.most_common():
        print(f"  {k:18s} {v}")
    print(f"\nNext:\n  python make_surrogates.py {out} --out dataset_v2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
