# Results Analysis

**Status: measured.** All numbers below come from an actual run of this
codebase on this machine: `python -m src.train --model {xlm_roberta,indicbert,mbert}`
(full ~14,472-row deduplicated TRAC-1 pool, seed=42, 80/10/10 split),
followed by `python -m src.stacking --generate --train`. Raw outputs are in
`results/base_models/*.json`, `results/stacking/*.json`, and
`results/model_comparison.csv`. Nothing here is estimated, copied from the
source deck, or filled in before the corresponding command actually ran.

## Our experiment vs. the prior sprint deck

| Model | Reported in prior sprint deck (accuracy / weighted F1) | Our experiment (accuracy / weighted F1) |
|---|---|---|
| XLM-R (Base) | 54.05 / 65.02 | **64.57 / 64.62** |
| IndicBERT (Base) | 64.27 / 54.31 | **64.50 / 64.50** |
| mBERT (Base) | 63.05 / 62.47 | **61.26 / 61.27** |
| Stacking Ensemble | 68.00 / 67.00 | **65.61 / 65.64** |

Macro F1 (not reported in the deck at all): XLM-R 63.96, IndicBERT 64.18,
mBERT 60.35, **Stacking Ensemble 65.18**.

"Reported in prior sprint deck" numbers are copied verbatim from
`docs/PAPER_BASELINE.md` §4 (`CodeMix-DeepDefender.pptx (5).pdf`, slide 11)
— an internal capstone Sprint Review deck by the same team, not a
peer-reviewed publication, with no disclosed dataset size, split, or
hyperparameters behind those numbers (see that document's §8 for the full
list of caveats). They are shown here only as a labeled comparison point,
never merged into "our" results.

Observed pattern in our own numbers, stated plainly: our XLM-R and IndicBERT
land within half a point of each other (64.57 vs 64.50 accuracy) despite
being architecturally quite different models, and our mBERT is the weakest
base model on every metric. The stacking ensemble beat **every individual
base model on every reported metric** (accuracy, weighted F1, macro F1) —
a real, measured improvement of roughly +1 to +1.3 points over the best
single base model (XLM-R), not assumed in advance.

## Why this comparison is not apples-to-apples

- **Dataset size/version**: our experiment uses the TRAC-1 Hindi Facebook
  subset physically present in this workspace, deduplicated (15,001 raw
  rows -> 14,472 after removing 529 exact-duplicate texts), with our own
  fixed-seed (42) 80/10/10 stratified split: 11,577 train / 1,447
  validation / 1,448 test (`data/splits/`). The deck does not state what
  dataset size or split it used.
- **Checkpoints**: our IndicBERT uses `ai4bharat/IndicBERTv2-MLM-only`, not
  the original ALBERT-based `ai4bharat/indic-bert`, because the latter
  returned HTTP 401 ("gated repo") when this project tried to load it (see
  `configs/models.yaml` and `docs/REPRODUCTION_PLAN.md` §7). The deck does
  not state which IndicBERT variant it used either, so this may or may not
  match.
- **Hyperparameters**: ours are the conservative, VRAM-constrained defaults
  in `configs/config.yaml` (batch_size=4, grad_accum=4, max_length=128,
  4 epochs, learning_rate=2e-5, fp16=true), chosen for and verified on a
  4GB RTX 3050 laptop GPU (`torch.cuda.get_device_properties` reported 4095
  MB). The deck assumed Google Colab Pro/Pro+ GPUs and states no
  hyperparameters at all.
- **Stacking method**: ours is explicit held-out-validation stacking (see
  `docs/STACKING.md`) — base models trained on train only, meta-classifier
  (`sklearn.linear_model.LogisticRegression`) fit on validation-split
  predictions only, final numbers from a single evaluation on the
  never-touched test split. The deck's "Proposed Stacking Ensemble" row
  does not say how its meta-features were generated.

Given all of the above, the accuracy deltas above should **not** be read as
evidence that our pipeline is better or worse than the original — the two
are measuring different, underspecified setups on (very likely) different
data. The value of this comparison is transparency about what we actually
measured, not benchmarking against the deck.

## Per-class performance (test set F1)

| Class | XLM-R | IndicBERT | mBERT | Stacking Ensemble |
|---|---:|---:|---:|---:|
| NAG | 0.611 | 0.628 | 0.563 | **0.632** |
| CAG | 0.633 | 0.623 | 0.593 | **0.638** |
| OAG | 0.675 | 0.674 | 0.654 | **0.685** |

The stacking ensemble improves per-class F1 over every individual base
model on every class, not just in the aggregate metrics — a consistent,
measured result. NAG is the class with the fewest test examples (264 of
1,448, ~18%) and is the hardest class for XLM-R and mBERT, but **not** for
IndicBERT, where CAG (0.623) actually scores slightly lower than NAG
(0.628) — the "minority class is always hardest" intuition does not hold
uniformly across all three models, and is reported here as measured rather
than assumed.

## Data-leakage verification

Before trusting the above numbers, the pipeline was audited end-to-end for
train/validation/test leakage (see `docs/STACKING.md`'s "Limitations,
stated plainly" section for the full writeup):

- Train/validation/test splits have **zero overlapping rows**, verified
  both by content hash and by raw text, on this exact run's split files.
- `src/model_training.py`'s `Trainer` is wired with `eval_dataset=val_dataset`
  only; `test_dataset` is touched exactly once, in a single post-training
  `trainer.predict()` call, never during `.train()` or checkpoint selection.
- `src/stacking.py`'s meta-classifier is fit with `clf.fit(X_val, y_val)`
  only, everywhere in that file including the ablation-study path;
  `X_test`/`y_test` appear only in `.predict()` calls used for reporting.

## Ablation (optional, task section 30)

Not run as part of this pass. `python -m src.stacking --ablation` is
implemented and will produce `results/experiments/ablation_study.csv`
(all 7 non-empty subsets of the 3 base models, reusing the already-generated
meta-features) if/when run.

## Fairness probe findings

`python -m src.fairness` (40 cases, 4 templates × identity-term variants,
`data/fairness/fairness_cases.csv`) found that **3 of 4 templates changed
predicted label depending only on which group term was substituted in**.
Most notable: template T4, *"mujhe {group} logo se koi problem nahi hai"*
("I have no problem with {group} people") — an explicitly non-aggressive,
friendly sentence — was classified **CAG (covertly aggressive)** for
"hindu", "muslim", and "bihari", but **NAG (non-aggressive)** for "sikh",
"christian", "aadmi", "aurat", "north indian", "south indian", and
"punjabi". This is a real, measured behavior difference on an otherwise
identical sentence, not a hypothetical concern, and is reported here as-is
— see `results/fairness_report.json` for every case. Per the task's
research-integrity requirement, this is **not** evidence that the model is
generally biased against any group (40 cases across 4 hand-written
templates is far too small a probe to support that claim either way), only
that this specific, concerning pattern exists and should be investigated
further (more templates, more group terms, statistical testing) before any
real-world deployment decision.

## Robustness probe findings

`python -m src.robustness` (55 cases, 5 base sentences × 10 perturbation
types, `data/robustness/robustness_cases.csv`) found the predicted label
changed vs. the unperturbed base sentence in **10 of 50 (20%) non-base
variants**. Stability by perturbation type (label-match rate vs. base,
higher = more stable):

| Perturbation type | Match rate |
|---|---:|
| misspelling | 100% |
| capitalization | 100% |
| mention | 100% |
| URL | 100% |
| hashtag | 80% |
| emoji | 80% |
| punctuation | 80% |
| repeated characters | 80% |
| spelling variation | 40% |
| English/Hindi code-switch | 40% |

The two least stable perturbation types — Romanized-Hindi spelling
variation (e.g. "bhut bakwaas h") and English/Hindi code-switching (e.g.
"this is bahut bakwas yaar") — are exactly the two phenomena the source
material's problem statement identifies as central to why Hinglish
moderation is hard (`docs/PAPER_BASELINE.md` §1). This is a measured
weak point of the current models, not a claim that the system "handles"
Hinglish variation robustly — see `results/robustness_report.csv` for
every case.

## Limitations

- Results reflect a single train/val/test split with a single random seed
  (42) — no cross-validation variance estimate is reported.
- The stacking meta-classifier is trained on the validation split only
  (1,447 rows, ~10% of the pool), a real but accepted trade-off documented
  in `docs/STACKING.md`.
- This run used the full ~14,472-row deduplicated pool, not the task's
  4,000-row first-development-run default — check
  `results/experiments/experiment_*.json` (`dataset_train_size`) before
  comparing against any other run of this codebase.
- The fairness and robustness probes are small, hand-written test suites
  (40 and 55 cases respectively) intended to surface concrete failure
  patterns, not to certify the system as fair or robust in general.
