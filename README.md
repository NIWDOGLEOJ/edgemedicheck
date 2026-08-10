# EdgeMediCheck

An offline, vision-based scanner for checking medicine authenticity and expiry
at a pharmacy counter. Four independent streams — OCR and date parsing,
barcode/GS1 cross-check, database verification, and a CNN visual check — are
fused into one verdict.

This folder is the complete project: application, dataset pipeline, trained
model and results. **It deliberately contains no image data**, and the
manuscripts are kept out of this public repository.
Everything needed to regenerate that data from scratch is here.

---

## Layout

```
EdgeMediCheck/
├── edgemedicheck/       the scanner application
│   ├── run.py           CLI: scan, collect, splits
│   ├── app.py           web interface
│   ├── edgemedicheck/   ocr, barcode, database, cnn, fusion, pipeline
│   └── data/            calibration + verification DB (images excluded)
├── model training/      dataset construction and CNN training
│   ├── DATASET.md       how the dataset is built, and why the old one failed
│   ├── COLLECTING.md    how to photograph real packs yourself
│   └── manifests/       provenance of every image ever used
└── results/             metrics, plots (trained model linked below)
```

## The trained model

The checkpoint is 134 MB, over GitHub's 100 MB per-file limit, so it is hosted
outside the repository.

**[Download from Google Drive](https://drive.google.com/file/d/1eux3KkYOu3W5ArL9ZaMqA0MiDmRMzG_o/view?usp=sharing)** · [GitHub release mirror](https://github.com/NIWDOGLEOJ/edgemedicheck/releases/tag/v1.0)

From the command line, use the `drive.usercontent.google.com` endpoint:

```bash
curl -L -o results/output_v2/best_counterfeit_model.pth \
  "https://drive.usercontent.google.com/download?id=1eux3KkYOu3W5ArL9ZaMqA0MiDmRMzG_o&export=download&confirm=t"
```

The plain `drive.google.com/uc?export=download` form does **not** work for this
file. Drive cannot virus-scan anything over 100 MB, so it serves a 2 KB HTML
warning page instead of the model. Saved under a `.pth` name that fails later
inside `torch.load` with an unrelated-looking error. The `confirm=t` URL above
skips the interstitial and returns the real file.

Verify the download:

```bash
shasum -a 256 results/output_v2/best_counterfeit_model.pth
# 3a3977ada293a74dec902a731e7d00e66c6b33626ff97b1d9fbcc10378678c15
```

Class order is `['Fake', 'Real']`, so `P(fake) = softmax(logits)[0]`.

Place it at `results/output_v2/best_counterfeit_model.pth`. Everything else in
`results/output_v2/` — metrics, per-operation breakdown, confusion matrix,
training curves — is in the repository already.

The model can also be rebuilt from scratch with the commands under
*Rebuilding the dataset and retraining* below. That takes roughly 80 minutes of
scraping plus 15 minutes of training.

---

## What is not here, and how to get it back

Image data is excluded because it is bulky and fully regenerable. The
manifests in `model training/manifests/` record the exact source URL, product
slug, dimensions and hash of every image, so the dataset can be rebuilt
identically.

| excluded | size | regenerate with |
|---|---|---|
| `pool/genuine/` — 2000 genuine pack photos | 1.5 GB | `scrape_genuine.py` |
| `dataset_v2/`, `dataset_v2_split/` — 8000 derived images | 256 MB | `make_surrogates.py`, `split_grouped.py` |
| `probe/who/` — 247 WHO falsified-product photos | 3.6 MB | `build_ood_probe.py` |
| `dataset/`, `dataset_norm/` — the original broken dataset | 200 MB | not regenerable; see below |
| `edgemedicheck/data/*` image folders | 26 MB | `run.py collect` |
| virtualenvs, `__pycache__`, archive zips | ~1 GB | `pip install -r requirements.txt` |

The original dataset is the one exception — it cannot be regenerated, because
it was assembled by hand from image search and screenshots. It is also the one
piece worth *not* keeping: see the negative result below. Its metrics survive
in `results/original_dataset_negative_result/`.

---

## Results

ResNet-18 transfer, 2000 packs → 8000 images, split 1400/300/300 **by pack** so
no carton appears in two folds.

| | value |
|---|---|
| test accuracy / ROC-AUC | 94.50% / 0.982 |
| counterfeits caught | 93.0% (558/600) |
| false alarms on genuine packs | 4.0% (24/600) |
| real WHO falsified products flagged | **7.3%** (18/247) |

Per surrogate operation: relabel 100%, reprint 100%, substrate 96.7%,
damage 89.6%, recapture 80.2%.

**Read the last row before the first.** The model scores 94.5% on held-out
surrogates and catches about one real counterfeit in fourteen. Across three
runs its rate of flagging real counterfeits tracks its false-alarm rate on
genuine packs, which is what no discriminative signal looks like. The visual
stream is implemented and honestly measured; it is not yet a working
counterfeit detector. `model training/DATASET.md` sets out the evidence.

### The negative result this replaced

The original dataset reported 1.0 accuracy, 1.0 precision, 1.0 recall and
1.0 ROC-AUC. It was measuring file encoding. Real images were image-search
saves (JPEG, 204×209); Fake images were screen captures (PNG, 528×503). A
single threshold on bytes-per-pixel separated the classes at **98.6%** without
looking at a medicine box at all.

That is worth reporting in the paper, not hiding. It is a clean demonstration
of shortcut learning, and `audit_dataset.py` now blocks training on any dataset
that shows it.

---

## Running the application

```bash
cd edgemedicheck
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install tesseract
python run.py scan --live --backend webcam
```

Grant camera access under System Settings → Privacy & Security → Camera.
Without it OpenCV returns black frames rather than an error, which looks like
a bug in the code.

### Live scanning

`http://<host>:5000/live` shows a continuous camera preview with the verdict
beside it, rescanning every few seconds. Point the camera at a pack and the
panel turns green, amber or red as the reading settles.

The preview is streamed from the host camera as Motion JPEG rather than opened
in the browser, for two reasons: the counter deployment's camera is the fixed
enclosure unit attached to the Pi, not the tablet's; and browsers block
`getUserMedia` on plain HTTP over a LAN address, which is exactly how a phone
reaches this server.

Scanning is chained rather than run on a timer — a scan takes 0.6–1.0 s on a
laptop and several seconds on a Pi 4, so overlapping requests would queue up
behind a single-frame camera.

To demonstrate the screen without a camera, replay a folder of images:

```bash
python run.py serve --backend folder --folder data/images
```

What the colours mean: they report whether the printed details are internally
consistent and whether the batch is in the local database. They are not a
chemical test, and red most often means expired or unrecognised stock rather
than a detected forgery. The visual stream runs the calibrated heuristic, not
the CNN — see the transfer result above.

---

## Rebuilding the dataset and retraining

```bash
cd "model training"
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision opencv-python pillow numpy requests \
            scikit-learn matplotlib seaborn pymupdf

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

The audit is a gate, not a report: `train_model.py` refuses to train on a
dataset whose classes are trivially separable. `--force` overrides it and
labels the run a smoke test whose numbers must not be published.

Scraping takes roughly 80 minutes and is polite by default (8 requests/second,
6 workers). Both retailers permit crawling of product pages in `robots.txt`.

---

## Legacy scripts

Four scripts predate the dataset rebuild and still default to the old `dataset/`
and `output/` paths, which are not in this folder. They are kept because two of
them are still useful with explicit flags.

| script | state |
|---|---|
| `falsify_model.py` | works, but pass `--model results/output_v2/best_counterfeit_model.pth --test dataset_v2_split/test` |
| `predict.py` | **will not run as-is** — model path is hardcoded to `output/`, with no flag to override |
| `split_dataset.py` | superseded by `split_grouped.py`, which splits by pack instead of by file |
| `augment_lighting.py` | superseded by `capture_sim()` in `make_surrogates.py`, which applies the same jitter to both classes |

`predict.py` needs a one-line change to point at `results/output_v2/` before it
can be used for inference.

---

## Honest framing

- The Fake class is **synthetic reproduction surrogates**, not seized
  counterfeits. Real counterfeits cannot be obtained legally in quantity; that
  is the reason, and it is defensible. Say it plainly.
- Report **recall on Fake together with the false-alarm rate on genuine**.
  Either alone is trivially gamed.
- Report **per-operation recall**. An averaged F1 hides whichever cue the model
  handles worst.
- The **WHO probe is single-class**, so it yields recall and never precision,
  and it is out of distribution in ways unrelated to counterfeiting — vials and
  bottles shot in the field versus Indian OTC cartons shot in studio. 7.3% is a
  lower bound on transfer, not a measurement of it.
- Three of the four streams need no training data and can be evaluated on
  whatever real packs are to hand. Those are the reportable numbers on real
  Indian packaging. `COLLECTING.md` explains how.

---

## Related work

Zakaria, Y., Ishidera, E., Ishiyama, R., Matsui, T., and Yasumoto, K.,
"Counterfeit Medicine Detection by Visual Inspection of Package Design Using
Multimodal LLMs with Text and Image Prompt Engineering." Reports 74.6% binary
accuracy with ChatGPT-4o on medicine packaging collected in Tanzanian retail
outlets, using replica packages built by mimicking authentic designs — the same
surrogate strategy used here, and for the same reason.

That figure is the right reference point for this task. The 94.5% reported
above is on a different and easier distribution and is not a comparable number.

---

## Authors

Devesh R and Srinikesh D, Department of Computer Science and Engineering (AI),
Sathyabama Institute of Science and Technology, Chennai.

