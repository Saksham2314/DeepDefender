# Stacking Methodology

Task section 10 requires avoiding data leakage in the stacking
meta-classifier and offers two acceptable approaches: (A) out-of-fold
stacking, or (B) train base models on the training split and generate
meta-features on a completely held-out validation split, with a preference
for (A) "if computationally practical."

## Decision: held-out validation (method B)

**This project uses method (B), `held_out_validation`.**

True out-of-fold (OOF) stacking would require k-fold cross-validation
*retraining* of every base transformer: for `n_folds=5`, that's 5 additional
full fine-tuning runs per base model, i.e. 15 extra transformer fine-tunes
on top of the 3 already being trained once each. On a 4GB-VRAM laptop GPU
where a single full fine-tune already takes on the order of an hour, OOF
stacking would add many additional hours (roughly 5x the base-model training
time) for the first development run. This is judged computationally
impractical for this phase of the project, so method (B) is used instead,
per the task's own explicit allowance for that choice.

## Exact procedure

1. **Base model training** (already covered by `docs/REPRODUCTION_PLAN.md` /
   `src/model_training.py`): each of the three base models (XLM-R, IndicBERT,
   mBERT) is fine-tuned **once**, using only `data/splits/train.csv`. Neither
   the validation nor the test split is seen during base-model training or
   used for early stopping decisions beyond the validation split itself
   (which is legitimate — validation-based early stopping is standard
   practice and does not leak test-set information).
2. **Meta-feature generation**: each trained base model is loaded (one at a
   time — see "VRAM" note below) and run in inference mode over
   `data/splits/validation.csv`, producing a `(N_val, 3)` softmax probability
   matrix per model. The three matrices are concatenated column-wise into a
   `(N_val, 9)` meta-feature matrix, in the fixed order
   `[XLM-R:NAG,CAG,OAG, IndicBERT:NAG,CAG,OAG, mBERT:NAG,CAG,OAG]` (task
   section 10's example ordering).
3. **Meta-classifier training**: `sklearn.linear_model.LogisticRegression`
   (task section 11 configuration: `configs/config.yaml` ->
   `stacking.meta_classifier`) is fit on `(X_val_meta, y_val)` — i.e. the
   validation split's *true* labels, not the base models' predicted labels.
   Because none of the three base models were trained on the validation
   split, their validation-split predictions are a fair, unbiased estimate
   of the predictions they would produce on any other unseen data — this is
   what makes it safe to train the meta-classifier on them.
4. **Final ensemble evaluation**: the same procedure (steps 2-3's feature
   generation, not retraining) is repeated on `data/splits/test.csv` — a
   split neither the base models nor the meta-classifier have ever seen in
   training — producing the final, reported ensemble metrics.

## Why this avoids leakage

- The test split is touched exactly once, at the very end, purely for
  evaluation. It never influences any weight update (base-model fine-tuning
  or meta-classifier fitting) or any hyperparameter/model-selection
  decision in this pipeline.
- The validation split is used twice, for two *different* purposes that
  don't leak into each other: (a) early stopping during base-model
  fine-tuning, and (b) as the meta-classifier's *training* data. This is
  standard, accepted practice for held-out stacking — the meta-classifier
  never sees validation-split labels during base-model training, and the
  base models never receive gradient updates from validation-split data.

## VRAM handling during meta-feature generation (task section 36)

`src/stacking.py` loads one base model onto the GPU at a time, runs
inference for that model over the validation and test splits, then deletes
the model and calls `torch.cuda.empty_cache()` before loading the next one
— never holding more than one fine-tuned transformer in GPU memory
simultaneously, consistent with the base-model training phase.

## Limitations, stated plainly

- **Smaller meta-classifier training set.** Held-out validation stacking
  uses less data to train the meta-classifier (the validation split, ~10%
  of the pool) than true out-of-fold stacking would (which effectively uses
  the full training set's predictions). With a 3-class, 9-feature logistic
  regression this is unlikely to be a severe limitation, but it is a real
  trade-off, not a free substitution.

- **Checkpoint selection is a form of (mild, accepted) validation-set
  fitting, but it does not touch the test set.** Each base model's "best"
  epoch checkpoint is chosen by validation `f1_macro` (`load_best_model_at_end`
  + `metric_for_best_model="f1_macro"` in `src/model_training.py`) — no
  gradient update ever comes from validation data, but validation *labels*
  do influence which checkpoint is kept, which is standard early-stopping
  practice, not something unique to this project. This means the base
  models' predictions on the validation split (used to train the
  meta-classifier) could be a shade more optimistic than predictions on
  genuinely unseen data would be, since that validation split had a role in
  selecting the model. This is exactly why the **test split's reported
  numbers are the ones to trust**: `data/splits/test.csv` is never read by
  `src/model_training.py` for anything except the single final
  `trainer.predict(test_dataset)` call after training is complete (verified:
  `eval_dataset=val_dataset` is the only dataset wired into `Trainer`;
  `test_dataset` is constructed but not touched until that one post-training
  call), and `src/stacking.py`'s meta-classifier is fit with `clf.fit(X_val,
  y_val)` only — `X_test`/`y_test` appear exclusively in `clf.predict(X_test)`
  calls used for reporting, never in a `.fit()` call, anywhere in that file
  (including the ablation-study code path). Train/validation/test splits
  were also verified to have zero overlapping rows, both by content hash and
  by raw text, on the actual dataset used for this project's real training run.
