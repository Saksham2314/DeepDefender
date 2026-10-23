# Reproduction Plan

This document explains exactly how DeepDefender's implementation relates to
the source material in `docs/PAPER_BASELINE.md`, and where it deliberately
diverges or adds to it. See that file first for what the source material
does and does not specify.

## What we reproduce

| Source element | Our implementation |
|---|---|
| 3-class labels NAG / CAG / OAG | Same, unchanged. |
| Three base transformers: XLM-R, IndicBERT, mBERT | Same three checkpoints, configurable in `configs/models.yaml`. |
| Hindi-English code-mixed dataset | The TRAC-1 Hindi Facebook subset physically present in `Datasets/` (`agr_hi_train.csv`, `agr_hi_dev.csv`), which is consistent with the corpus the source literature cites. |
| Stacking ensemble as the final model | Implemented explicitly (not left ambiguous) as: 3×3 concatenated class-probability vector → `sklearn.linear_model.LogisticRegression` meta-classifier. |
| SHAP-based explainability | Implemented as documented in `docs/STACKING.md`/`src/explainability.py` (Phase 6), with the exact method used stated plainly rather than assumed. |
| Fairness / bias testing on identity-related terms | Implemented as a configurable, non-exhaustive template test suite (Phase 7), reported descriptively rather than as a pass/fail "fairness" verdict. |

## Where we intentionally go beyond or differ from the source

The source deck leaves almost every experimental detail unspecified (no
seed, split, hyperparameters, or per-class metrics — see
`PAPER_BASELINE.md` §4, §8). Rather than guess and silently present guesses
as "the paper's method," we make every such choice explicit and configurable
here, and label it as *our* choice:

1. **Reproducible subset + fixed splits.** The source reports one aggregate
   number per model with no stated split. We fix `RANDOM_SEED=42`,
   `DATASET_SIZE=4000` (configurable), and a stratified 80/10/10
   train/val/test split, with the exact row IDs saved to
   `data/splits/*.csv` so the experiment is reproducible on demand. TRAC-1
   ships no separate test file in this workspace, so our pool for
   subsetting is `agr_hi_train.csv` + `agr_hi_dev.csv` combined, and our own
   held-out test split is carved from that pool rather than reusing TRAC-1's
   original train/dev boundary.
2. **Out-of-fold stacking to avoid leakage.** The source's "Proposed
   Stacking Ensemble" row does not describe how meta-features were
   generated. We explicitly implement out-of-fold (or, if that is not
   computationally practical on the dev machine, held-out-validation)
   meta-feature generation and document which one was actually used in
   `docs/STACKING.md`, so results cannot be inflated by training the
   meta-classifier on in-sample base-model predictions.
3. **VRAM-conscious training.** The source assumed Google Colab Pro/Pro+
   GPUs (deck slide 12). This project targets a local RTX 3050 laptop
   (4GB VRAM, confirmed via `nvidia-smi`), so batch size, gradient
   accumulation, mixed precision, gradient checkpointing, and max sequence
   length are all conservative defaults and fully configurable — this is a
   hardware-driven adaptation, not a methodological one.
4. **Full metric reporting.** The source reports only accuracy and weighted
   F1 per model. We additionally report macro F1, per-class F1, precision,
   recall, and confusion matrices for every base model and the ensemble,
   on both validation and test sets.
5. **Class imbalance is measured and handled explicitly.** NAG is a
   minority class in the raw data (≈19% of train rows vs. ≈40% each for
   CAG/OAG). The source does not mention this. We surface it in dataset
   statistics and make class-weighting in the loss/meta-classifier
   configurable rather than assumed.
6. **No numbers are copied from the deck into our results.** Every number
   in `results/` and `docs/RESULTS_ANALYSIS.md` will come from an actual
   run of this codebase on this machine. The deck's numbers (§4 of
   `PAPER_BASELINE.md`) are used only as a labeled comparison point, never
   merged into or presented as our own measurements.
7. **IndicBERT checkpoint substitution.** The classic `ai4bharat/indic-bert`
   (ALBERT-based) checkpoint returns HTTP 401 ("gated repo") for
   unauthenticated access as of this build, and its only open mirror found
   (`kornesh/indic-bert`) ships TensorFlow-only weights. We use
   `ai4bharat/IndicBERTv2-MLM-only` instead — AI4Bharat's own official
   successor model, publicly accessible, verified to load cleanly via
   `AutoModelForSequenceClassification`. Still branded "IndicBERT" in all
   reporting, but this is v2, not the original ALBERT-based v1 the source
   deck's results table presumably used. See `configs/models.yaml` for the
   full compatibility note.
8. **Optional MuRIL experimental configuration** (task §31) — kept as an
   explicit, separate, opt-in configuration rather than replacing the
   baseline three-model stack, matching how the source deck itself only
   mentions MuRIL in the results table without including it in the stated
   three-model Methodology slide.

## What is *not* reproduced (because it was never specified)

We cannot reproduce hyperparameters, dataset size, or split ratios "as used"
in the Sprint Review deck, because none are given. Any claim of an exact
reproduction of the 68.00% / 67.00 weighted-F1 ensemble result would be
fabricated. Instead, this project measures its own numbers under a fully
documented configuration and reports them side by side with the source
numbers, explicitly labeled `Reported in prior sprint deck` vs.
`Our experiment`, per the task's research-integrity requirements.

## Phase status

This plan covers the full 12-phase build (task §38). As of this document's
creation, **only Phase 1** (environment check, project structure, dataset
loader, preprocessing, reproducible subset) has been implemented and run.
Phases 2 onward (per-model training, stacking, explainability, fairness,
robustness, API, extension, dashboard, testing) are scaffolded in the
directory structure but not yet implemented or executed — see
`README.md` for current status and next commands.
