# DeepDefender

**Explainable Hinglish Aggression Detection using Transformer Stacking**

DeepDefender detects aggression in Hindi-English code-mixed ("Hinglish")
social media text, classifying it as `NAG` (Non-Aggressive), `CAG`
(Covertly Aggressive), or `OAG` (Overtly Aggressive), by fine-tuning three
transformer models (XLM-RoBERTa, IndicBERT, mBERT) and combining their
class-probability outputs via a logistic-regression stacking
meta-classifier. See `docs/PAPER_BASELINE.md` for the source material this
reproduces and extends, and `docs/REPRODUCTION_PLAN.md` for exactly what is
reproduced vs. added.

> **Current status:** Phases 1-11 have working code (dataset prep,
> preprocessing, base-model training, stacking, explainability, fairness,
> robustness, the inference API, the browser extension, and the dashboard).
> Base-model training on the full dataset was launched on this machine and
> may still be in progress — check `results/base_models/`,
> `results/stacking/`, and `docs/RESULTS_ANALYSIS.md` for whether real
> numbers exist yet before trusting any accuracy claim. Testing (Phase 12)
> is partial: `tests/` covers preprocessing, dataset integrity, stacking
> feature dimensions, and the API; it does not yet cover every module.

## Architecture (target, once all phases are built)

```
                     ┌─────────────────────┐
   Hinglish text --->│   Preprocessing      │
                     │ (src/preprocessing.py)│
                     └──────────┬───────────┘
                                │ clean_text
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
       ┌────────────┐    ┌────────────┐    ┌────────────┐
       │ XLM-RoBERTa │    │ IndicBERT  │    │   mBERT    │
       │ fine-tuned  │    │ fine-tuned │    │ fine-tuned │
       └──────┬──────┘    └──────┬─────┘    └──────┬─────┘
              │ P(NAG,CAG,OAG)   │ P(NAG,CAG,OAG)   │ P(NAG,CAG,OAG)
              └────────┬─────────┴─────────┬────────┘
                        ▼                   
              9-dim concatenated meta-feature vector
                        │
                        ▼
          Logistic Regression meta-classifier
          (src/stacking.py, out-of-fold trained)
                        │
                        ▼
        {label, probabilities, confidence, explanation}
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
      FastAPI inference API   Browser extension
      (api/main.py)           (extension/)
```

## Project structure

```
DeepDefender/
├── data/{raw,processed,splits,fairness}/   dataset + reproducible splits
├── configs/{config.yaml,models.yaml}        all tunables, no hard-coding
├── src/                                     preprocessing, dataset, training, ...
├── api/                                     FastAPI inference server (Phase 9)
├── extension/                               Chrome MV3 extension (Phase 10)
├── artifacts/{models,stacking}/             trained weights + meta-classifier
├── results/{figures,experiments,base_models,reports}/
├── docs/                                    PAPER_BASELINE, REPRODUCTION_PLAN, ...
└── tests/
```

## Dataset

TRAC-1 Hindi-English code-mixed aggression corpus, already present in this
workspace. See `docs/DATASET_SETUP.md` for exact provenance, file format,
and class distribution (NAG is a minority class at ~18-19%).

## Installation

```powershell
# Install torch first, matching your CUDA driver (see
# https://pytorch.org/get-started/locally/). IMPORTANT: verify it actually
# resolved to a CUDA build afterwards -- on this project's own dev machine
# the cu121 index silently gave a CPU-only wheel (see requirements.txt):
pip install torch --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available())"   # must print True

pip install -r requirements.txt
```

## GPU requirements

Developed against a local **NVIDIA RTX 3050 Laptop GPU (4 GB VRAM)**,
detected via `python -m src.train --check`. Default training config
(`configs/config.yaml`) is conservative for this reason: `batch_size=4`,
`gradient_accumulation_steps=4`, `max_length=128`, `fp16=true`. CPU fallback
works but is slow for transformer fine-tuning. On this machine, torch
2.14.0+cu126 correctly detected the GPU with `nvidia-smi` reporting 4095 MB
VRAM.

## Usage

```bash
# Check environment / GPU
python -m src.train --check

# Inspect raw dataset stats without writing anything
python -m src.dataset --stats

# Build the reproducible subset + stratified 80/10/10 split
# (this run used the full deduplicated pool, size=14472; the task's
# first-dev-run default in configs/config.yaml is 4000)
python -m src.dataset --prepare --size 14472 --seed 42

# Phase 1 smoke test (dataset + preprocessing + tokenizers)
python -m src.train --smoke-test

# Real but tiny train+eval cycle to validate the training pipeline before
# committing to a full run (task section 39)
python -m src.train --model xlm_roberta --tiny

# Full base-model fine-tuning (run sequentially, never simultaneously --
# task section 36 -- a single 4GB GPU can't hold more than one at a time)
python -m src.train --model xlm_roberta
python -m src.train --model indicbert
python -m src.train --model mbert

# Stacking: generate held-out-validation meta-features, then fit + evaluate
# the LogisticRegression meta-classifier (see docs/STACKING.md)
python -m src.stacking --generate
python -m src.stacking --train
python -m src.stacking --predict "tum bahut bakwas ho"
python -m src.stacking --ablation   # optional: all base-model subset combinations (task section 30)

# Rebuild results/model_comparison.csv from whatever results exist
python -m src.evaluation

# Regenerate all figures in results/figures/ from whatever results exist
python -m src.visualize

# Explainability (occlusion is the fast default; shap is slower/opt-in)
python -m src.explainability --text "tum bilkul pagal ho" --method occlusion

# Fairness / robustness probes (task sections 14-15)
python -m src.fairness
python -m src.robustness

# Inference API
uvicorn api.main:app --reload
# then: POST http://localhost:8000/predict {"text": "..."}
#       GET  http://localhost:8000/health

# Research dashboard (reads results/ only, never trains anything)
streamlit run dashboard/app.py

# Tests
python -m pytest tests/ -v                        # Python (32+ tests)
node tests/extension/test_service_worker.js        # browser extension
```

## Browser extension

Load `extension/` as an unpacked extension (`chrome://extensions` ->
Developer mode -> "Load unpacked"). It talks to the local API at
`http://localhost:8000` by default; the popup exposes detection settings
(enable/disable, highlights, explanations, dynamic content, confidence
threshold), while the API URL itself is set in
`extension/background/service-worker.js`'s `DEFAULT_SETTINGS`. See
`docs/PRIVACY.md` for exactly what data it sends and where.

## Preprocessing

Configurable in `configs/config.yaml` (`preprocessing:`). Defaults: lowercase,
strip URLs, strip `@mentions`, strip the `#` marker from hashtags (keeping
the word), normalize whitespace. Punctuation, emojis, and repeated
characters are **kept by default** — they can carry aggression signal — and
only stripped/collapsed if explicitly enabled. See `src/preprocessing.py`
and `tests/test_preprocessing.py`.

## Stacking methodology

See `docs/STACKING.md` (written once Phase 5 is implemented) for the exact
out-of-fold vs. held-out-validation choice made to avoid meta-classifier
data leakage.

## Explainability / Fairness / Robustness / API / Extension

Not yet implemented (Phases 6-10). Directory structure is in place.

## Troubleshooting

- **`torch is NOT installed`** from `--check`: install torch first (see
  Installation above), then `pip install -r requirements.txt`.
- **`torch.cuda.is_available()` is `False` after installing torch**: you
  likely got a CPU-only wheel from an index that no longer publishes CUDA
  builds for your torch version. `pip uninstall torch`, then reinstall from
  a current CUDA index (cu126 confirmed working on this project's RTX 3050
  as of this build).
- **CUDA out-of-memory** during training: lower `training.batch_size` in
  `configs/config.yaml` first, then `max_length`, then enable
  `training.gradient_checkpointing`.
- **`FileNotFoundError` from `src/dataset.py`**: see `docs/DATASET_SETUP.md`.
- **HTTP 401 "gated repo" loading `ai4bharat/indic-bert`**: expected — that
  checkpoint is gated upstream as of this build. `configs/models.yaml`
  already points `indicbert` at the open `ai4bharat/IndicBERTv2-MLM-only`
  instead; see that file's comments and `docs/REPRODUCTION_PLAN.md` §7 if
  you need to swap it again.
- **Before a full multi-hour training run**, always validate with
  `python -m src.train --model <key> --tiny` first (task section 39) — it
  runs the real pipeline on ~40 samples in under a minute and will surface
  API-version breakage (e.g. this build hit a `transformers` 5.x
  `TrainingArguments` argument rename) before it wastes GPU-hours.
- **`node tests/extension/test_service_worker.js`** requires Node.js
  (any recent version); it's the only piece of this project that isn't
  Python.

## Research limitations

- The source material this project reproduces is the team's own prior
  capstone Sprint Review deck, not a published/peer-reviewed paper — see
  `docs/PAPER_BASELINE.md` for exactly what is and isn't specified there.
- The first development run uses a 4,000-row subset of the ~14,500-row
  deduplicated TRAC-1 pool; results on this subset are not directly
  comparable to results on the full pool or to any other paper's full-scale
  experiments.
- No results are reported here until they have actually been measured by
  running this codebase (task research-integrity requirement).

## Future work

MuRIL as an experimental 4th/alternative base model (`configs/models.yaml`
`experimental_models` / `stacks.experimental`), ablation studies over model
subsets, robustness/fairness-driven data augmentation.
