# Collecting real images on the Mac

You said access to real medicine packs is limited. Read the last section first
— it changes what you should aim for.

---

## Setup (once)

```bash
cd "../edgemedicheck"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install tesseract
```

Grant camera permission the first time: **System Settings → Privacy & Security
→ Camera → Terminal**. Without it `cv2.VideoCapture` opens and returns black
frames rather than raising an error, which looks like a code bug.

Check the camera:

```bash
python run.py scan --live --backend webcam
```

A Mac's built-in FaceTime camera works, but it is fixed to the lid — awkward
for photographing something lying flat. A cheap USB webcam on a desk arm, or an
iPhone via Continuity Camera, is far easier to hold at a fixed distance.

---

## The rig

You do not need the Raspberry Pi enclosure to start. You need **consistency**,
which you can get from cardboard:

- A dark, matte, non-reflective mat. Grey or black card. Not glossy, not white.
- The camera fixed above it — a box, a stack of books, a phone tripod. It must
  not move between packs.
- Even light. A desk lamp bounced off a wall or through paper. Avoid direct
  overhead light: it produces specular hotspots on foil that swamp the
  hologram cue.
- Mark the mat with a pencil rectangle so every pack lands in roughly the same
  place.

**Whatever you build, do not change it mid-collection.** A model trained across
two rig configurations learns the rig.

---

## Capturing

```bash
cd "../edgemedicheck"
python run.py collect --out data/collected --label genuine --shots 8 --backend webcam
```

It prompts for product, batch, expiry and manufacturer, then captures a burst,
rejecting blurred, badly lit, or badly framed frames at the point of capture.

Each **physical carton** gets its own auto-incrementing pack ID:

```
PARACIP__PC24101__p02__05.jpg
   |         |      |    |
product    batch  pack  shot
```

A new pack number means a **genuinely different box**. More angles of the same
box go under `--shots`. Getting this wrong at capture time cannot be repaired
afterwards.

Between shots of one pack, reposition it slightly — rotate a few degrees, nudge
it off centre, flip to the panel that carries the batch code. That variation is
what makes the model robust rather than brittle.

### Framing

- Whole pack in frame, all four edges visible, with margin. Do not crop in.
- Pack filling roughly 40–70% of the frame.
- Include the printed batch/expiry panel, the barcode, and the hologram patch.
- Nothing else in frame — no fingers, no second pack, no clutter.

If brand artwork and the batch panel are on different faces, shoot two frames
and give them the **same pack ID**.

---

## Surrogate ("suspicious") samples

Real counterfeits cannot be obtained legally in quantity. Produce physical
surrogates from packs you already have, and say so plainly in the paper.

```bash
python run.py collect --out data/collected --label suspicious --shots 8 --backend webcam
```

Effective surrogates, roughly in order of realism:

1. **Reprint** — scan or photograph a genuine carton, print it on an inkjet,
   fold it around the original. Reproduces real halftone and colour-gamut
   error, which is how low-end counterfeits actually differ.
2. **Recapture** — photograph a genuine pack displayed on a monitor. Adds
   moiré and backlight signature.
3. **Relabel** — print a new date panel and glue it over the original, or
   over-print the expiry.
4. **Substrate swap** — print the artwork on visibly different paper.
5. **Damage/tamper** — soak, scuff, peel and re-stick the label.

If you tamper a pack you already photographed as genuine, give it the **same
pack ID** in both classes. The splitter then keeps both versions in one fold,
so the model cannot learn "carton #3" as a shortcut.

### The trap

Every incidental difference between your classes becomes the feature. Your
current Kaggle-style dataset failed for exactly this reason: Real was JPEG,
Fake was PNG, and a single threshold on file size separated them at 98.6%.

So:

- **Same rig, same camera, same session** for both classes.
- **Same products in both classes.** If all surrogates are PARACIP and all
  genuine are AZITHRAL, the model learns the brand.
- **Interleave.** Alternate genuine and surrogate rather than doing all of one
  then all of the other.

---

## Check before training

```bash
python run.py splits data/collected/dataset          # leakage + effective size
python ../"model training"/audit_dataset.py data/collected/dataset \
    --classes genuine suspicious
```

The audit's shortcut test should sit near chance (~50%). If it is high, some
capture artefact is separating your classes and the fix is in the rig, not the
model.

---

## If you cannot collect much

With limited access, a CNN is not reachable — it needs a few hundred distinct
packs per class, and no augmentation substitutes for that. Photographing one
box thirty times gives you thirty images and **one** sample.

That is not a dead end. Three of the scanner's four streams need no training
data at all, and can be evaluated on whatever real packs you can find:

| Stream | Needs training data? | Evaluate with |
|---|---|---|
| OCR + date parsing | No | Any real pack photos, ground truth read by eye |
| Barcode / GS1 cross-check | No | Any pack carrying a 2D code |
| Database verification | No | Batch records you enter yourself |
| CNN visual | **Yes** | Not reachable without collection |

A home medicine cabinet — even 10–15 boxes — is enough to produce a real,
reportable evaluation of OCR field accuracy, date-format parsing across Indian
label conventions, and barcode/text agreement. Those are honest numbers on real
Indian packaging, which is worth considerably more than a CNN accuracy figure
derived from a broken dataset.

For the visual stream, report it as implemented but not validated, and explain
why: seized counterfeits are not obtainable, so the class cannot be sampled.
That is a defensible limitation. A 99% accuracy that measures file format is
not.
