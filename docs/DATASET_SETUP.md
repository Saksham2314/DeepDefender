# Dataset Setup

## Status: dataset already present

The TRAC-1 Hindi-English code-mixed aggression dataset was found already
present in this workspace at:

```
Datasets/agr_hi_train.csv   (12,000 rows)
Datasets/agr_hi_dev.csv     (3,001 rows)
```

It has been copied (not moved — the originals are untouched) into the
project's canonical raw-data location:

```
data/raw/agr_hi_train.csv
data/raw/agr_hi_dev.csv
```

`configs/config.yaml` (`data.raw_dir`) points at `data/raw`. `data/raw/` is
git-ignored (the CSVs are not committed) — see below for what to do if you
clone this repo without them.

## File format (verified directly, not assumed)

- **No header row.** Each row is positional: `id, text, label`.
- `id` looks like `facebook_corpus_msr_401470` — consistent with the
  TRAC-1 Hindi Facebook corpus (Kumar et al., 2018, "Aggression-annotated
  Corpus of Hindi-English Code-mixed Data").
- `label` is exactly one of `NAG`, `CAG`, `OAG`.
- Encoding: UTF-8 (contains Devanagari script and emoji).

## Measured statistics (train + dev combined, before dedup)

| Label | Count | % |
|---|---:|---:|
| CAG | 6,115 | 40.8% |
| OAG | 6,073 | 40.5% |
| NAG | 2,813 | 18.8% |

(Total 15,001 rows before dedup.)

After exact-duplicate-text removal (529 duplicate rows dropped), the
combined pool used for subsetting is **14,472 rows**: NAG 2,636 (18.2%),
CAG 5,890 (40.7%), OAG 5,946 (41.1%). NAG is a clear minority class —
relevant for both training (class weighting) and evaluation (macro F1,
per-class F1, not just accuracy).

No official TRAC-1 test split file is present in this workspace, so this
project carves its own held-out test set from the combined train+dev pool
(see `docs/REPRODUCTION_PLAN.md` §"Reproducible subset + fixed splits").

## If the dataset is missing (fresh clone / different machine)

1. Obtain the TRAC-1 Hindi-English aggression dataset
   (`agr_hi_train.csv`, `agr_hi_dev.csv`) from a source you have rights to
   use — e.g. the original TRAC-1 / COLING 2018 shared task distribution,
   or wherever your team's copy is archived. **Do not** substitute an
   unofficial or unverified dataset silently.
2. Place both files in `data/raw/` at the project root.
3. Run `python -m src.dataset --stats` to verify they load correctly and
   the label distribution looks as above.
4. Run `python -m src.dataset --prepare --size 4000` to regenerate the
   reproducible subset and splits.

If the files are missing, `src/dataset.py` raises a `FileNotFoundError`
naming the exact missing path(s) rather than downloading anything or
proceeding with fabricated data (task section 37).
