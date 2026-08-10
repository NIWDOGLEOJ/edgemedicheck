# The dataset, and why the old one was replaced

## The negative result

The original dataset reported perfect scores. `output/evaluation_results.json`
recorded accuracy, precision, recall and ROC-AUC all at 1.0, with a clean
confusion matrix on 98 test images.

None of it measured counterfeit detection.

The two classes had been collected by different methods, and that difference —
not the packaging — is what separated them:

| | Real | Fake |
|---|---|---|
| distinct sources | 406 | 240 |
| file format | 100% `.jpg` | 100% `.png` |
| filenames | `images01.jpg`, `images03.jpg`, … | `Screenshot 2025-09-17 173830.png`, … |
| mean dimensions | 204 × 209 | 528 × 503 |
| median bytes/pixel | 0.14 | 1.32 |

Real images were saved from image search; Fake images were screen captures.
`audit_dataset.py` separates the classes at **98.6%** using a single threshold
on bytes-per-pixel — a feature containing no packaging information at all. A
ResNet-18 finds that in its first epoch. The reported 100% is the model
reading file encoding.

Re-encoding everything identically (`normalize_dataset.py`, uniform JPEG q88 at
256px) closed the encoding gap from 1.02 to 0.08 bytes/pixel and dropped naive
separability from 98.6% to **77.0%**. The residual was brightness: screenshots
of web pages carry bright UI backgrounds that product thumbnails do not.

So the leak was not one fixable defect. The classes differed in *how they were
acquired*, and no re-encoding repairs that. The old dataset is kept only as
this documented negative result.

## What replaced it

The failure mode is specific: any property that correlates with the label but
is not the packaging becomes the feature. Collecting more images the same way
makes it worse, because genuine packs come from pharmacy e-commerce and
"counterfeit" photos come from news reports — a wider acquisition gap, and a
higher, more misleading score.

The fix is to derive **both classes from one pool**, so no acquisition artefact
can correlate with the label.

```
   Apollo Pharmacy + PharmEasy OTC catalogues
                    |
          scrape_genuine.py
                    |
        pool of genuine pack photos
                    |
        +-----------+-----------+
        |                       |
      Real                    Fake
        |                  counterfeit_op()
        |                       |
        +-----------+-----------+
                    |
             capture_sim()          <-- identical for both classes
                    |
            uniform JPEG q88
```

### The shared-capture rule

The subtle version of the original bug is to emit raw pool images as Real and
processed images as Fake. Resampling, added noise and extra JPEG generations
are all trivially detectable, so the model would learn "has been through an
image pipeline" — the same failure one level up.

Both classes therefore run through the same `capture_sim()`: perspective,
lighting gradient, vignetting, exposure and white-balance drift, defocus,
sensor noise, and a JPEG generation at a quality drawn from a shared band.
Same code path, same distributions, drawn from the same seeded generator. The
counterfeit operation is applied **before** that stage, on the Fake branch
only.

### The five surrogate operations

Digital analogues of the physical surrogates in `COLLECTING.md`:

| op | what it simulates | main artefacts |
|---|---|---|
| `reprint` | artwork rescanned, run off on an inkjet | halftone screening at traditional 15°/75°/0° angles, dot gain, gamut compression, plate misregistration |
| `recapture` | a genuine pack photographed off a monitor | LCD subpixel stripe beating against the sensor grid, backlight falloff, lifted blacks |
| `relabel` | a fresh batch/expiry panel printed and glued on | off-white patch, its own typeface, slight rotation, edge shadow |
| `substrate` | correct artwork on uncoated board | fibre texture, lost gloss, reduced saturation, warm cast |
| `damage` | soaked, scuffed, peeled, re-stuck | water blooms, abrasion streaks, torn corner |

### Grouping

Every output carries its source pack ID. `split_grouped.py` splits by **pack**,
not by file, and puts all of a pack's images — both classes — in one fold. A
carton the model saw in training never reappears in test, so it cannot score by
recognising the carton. This is the "carton #3" shortcut `COLLECTING.md` warns
about.

## Audit

`audit_dataset.py` is the check on the whole claim, and it runs automatically
before training.

| | old dataset | normalised old | new dataset |
|---|---|---|---|
| naive separability | 98.6% | 77.0% | **58.0%** |
| chance | 51.0% | 51.0% | 50.0% |
| formats overlap | FAIL | pass | pass |
| pack spans folds | pass | pass | pass |
| verdict | 2 blocking issues | 1 blocking issue | **no blocking issues** |

58% against 50% chance is near chance, and the audit passes. The residual 8
points come from the counterfeit operations genuinely adding high-frequency
texture — a reprinted carton really does compress differently — not from how
the files were acquired. That is a property of counterfeiting, not of the
collection method.

## Results

2000 packs → 8000 images, split 1400/300/300 packs. ResNet-18 transfer,
15 epochs. Full output in `output_v2/`.

### In-distribution (held-out test set, 1200 images from 300 unseen packs)

| metric | value |
|---|---|
| accuracy | 94.50% |
| ROC-AUC | 0.9820 |
| counterfeits caught (recall on Fake) | **93.00%** (558/600) |
| false alarms on genuine packs | **4.00%** (24/600) |
| macro F1 | 0.9450 |

Both numbers are needed. Recall alone is gamed by answering "Fake" more often,
which would push the false-alarm rate up with it; here it stays at 4%.

### Per operation

| operation | n | recall |
|---|---|---|
| relabel | 113 | 100.0% |
| reprint | 116 | 100.0% |
| substrate | 120 | 96.7% |
| damage | 125 | 89.6% |
| recapture | 126 | 80.2% |
| genuine (specificity) | 600 | 96.0% |

Reprint and relabel are saturated; recapture is the weak cue, which is
consistent with it being the most nearly-invisible of the five at 256px.

### Transfer to real counterfeits — the important result

On the WHO probe of 247 photographs of confirmed falsified products, the model
flagged **18/247 = 7.3%**, mean P(Fake) = 0.102.

A model scoring 93% on held-out surrogates catches roughly one in fourteen real
counterfeits.

The probe is single-class, so that figure means nothing on its own — it has to
be read against the false-alarm rate on genuine packs. Three runs, varying pack
count and epochs:

| run | packs | epochs | test acc | false alarms on genuine | probe recall |
|---|---|---|---|---|---|
| A | 318 | 4 | 76.0% | 19.4% | 13.8% |
| B | 2000 | 4 | 88.9% | 5.5% | 5.7% |
| C | 2000 | 15 | 94.5% | 4.0% | 7.3% |

**Probe recall tracks the false-alarm rate in every run.** The model flags a
real falsified pack at about the same rate it mistakenly flags a genuine one.
That is what no discriminative signal looks like: on real counterfeits the
model is operating at roughly chance, and the apparently higher probe recall in
run A is simply a poorly calibrated model saying "Fake" to everything more
often.

Run B is the control for a tempting misreading. Comparing A to C alone suggests
that training harder *hurts* transfer, since accuracy rose while probe recall
fell. Holding the data fixed and varying only epochs (B → C) moves probe recall
the other way, 5.7% → 7.3%. The A→C difference is the dataset size, not
overfitting, and the honest summary is that transfer is near chance throughout
rather than degrading.

Part of the gap is ordinary domain shift: the probe holds vials, tubs and
bottles photographed in the field, while training is Indian OTC cartons and
blisters shot in studio conditions. So 7.3% is a lower bound on transfer, not a
clean measurement of it. But the tracking between probe recall and false-alarm
rate is not explained by domain shift, and it is the finding worth reporting:
**high surrogate accuracy did not produce a detector that generalises to real
counterfeits.**

## Honest framing for the paper

State plainly:

- The Fake class is **synthetic reproduction surrogates**, not seized
  counterfeits. Say so. Seized counterfeits cannot be obtained legally in
  quantity, which is the reason, and it is a defensible one.
- Report **recall on Fake and the false-alarm rate on genuine together**.
  Either alone is trivially gamed.
- Report **per-operation recall** (`analyse_by_op.py`). The operations are not
  equally hard, and an averaged F1 hides whichever the model handles worst.
- Report the **WHO probe** as a lower bound on transfer, with its limits: it is
  single-class, so it yields recall only, and it is out of distribution in ways
  unrelated to counterfeiting — vials and bottles rather than Indian OTC
  cartons, field photography rather than studio shots.

## Reproducing

```bash
python scrape_genuine.py --out pool/genuine --target 2000
python make_surrogates.py pool/genuine --out dataset_v2 --per-class 2
python split_grouped.py dataset_v2 --out dataset_v2_split
python audit_dataset.py dataset_v2_split --classes Real Fake
python build_ood_probe.py --out probe/who
python train_model.py --data dataset_v2_split --probe probe/who --out output_v2
python analyse_by_op.py --data dataset_v2_split \
    --manifest dataset_v2/surrogates.jsonl \
    --model output_v2/best_counterfeit_model.pth
```

Sources are Apollo Pharmacy and PharmEasy OTC catalogues, reached through their
published sitemaps; both permit crawling of product pages in `robots.txt`. The
probe is built from WHO Medical Product Alerts, restricted to alerts titled
"Falsified" — "substandard" and "contaminated" alerts concern genuine packaging
with out-of-spec contents, and including them would put authentic cartons in
the counterfeit class.
