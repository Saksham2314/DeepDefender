# Paper / Source Material Baseline

## 0. Important note on provenance

No standalone IEEE-formatted research paper was found anywhere in the workspace
or on the user's Desktop. Two source documents were located instead, and both
describe the **same prior project** ("CodeMix DeepDefender" / "DeepDefenderCodeMix"),
a Thapar Institute of Engineering & Technology capstone (CPG 201) by the same
team this workspace belongs to (Aditya Tomar, Saksham Tiwari, Yash Kumar Sharma,
Piyajot Jindal, Kashish; mentors Dr. Mahak Gambhir and Dr. Kapil Rana):

1. **`Proposal report.pdf`** (Desktop, 16 pages) — the capstone project proposal.
   It contains a literature survey table (12 external papers) and a general
   methodology description, but proposes only a **single** multilingual
   transformer (mBERT or MuRIL), not a 3-model stack.
2. **`CodeMix-DeepDefender.pptx (5).pdf`** (Desktop, 14 slides, exported from
   PowerPoint) — a **Sprint Review deck** for the same project. This is the
   more specific and more relevant source: it documents an actual 3-model
   stacking pipeline that was apparently already prototyped, together with a
   results table. **This deck is treated as "the old paper" for this project**,
   since it is the only source that matches the architecture described in the
   task brief (XLM-R + IndicBERT + mBERT + stacking ensemble).

There is no evidence in either document of a peer-reviewed / published IEEE
paper. Both are internal capstone project artifacts. Some numbers in the
Sprint Review deck (see §4 below) look preliminary/inconsistent (e.g. a model
with *lower* accuracy than another but *higher* weighted F1) and are recorded
here exactly as found — they are not corrected, smoothed, or reinterpreted.

Two unrelated PDFs on the Desktop (`Ref.pdf`) turned out to be scanned fee
receipts, not related to this project, and were ignored.

## 1. Task / Problem

Detect cyberbullying / aggression in **Hindi-English code-mixed
("Hinglish") social media text**, written in Romanized Hindi mixed with
English, which monolingual/keyword-based moderation systems handle poorly.

## 2. Dataset (as described in the source documents)

The literature survey cites several benchmark datasets, most notably:

- **Kumar et al. [4], 2018** — "Aggression-annotated Corpus of Hindi-English
  Code-mixed Data" (Facebook + Twitter). This is the origin of the **TRAC-1**
  benchmark and its 3-way label scheme **NAG / CAG / OAG**.
- **Sharma & Bhalla [7], 2025** — uses the **TRAC-1 dataset** directly
  (8,626 Hindi-English code-mixed posts), reporting 86% accuracy /
  Precision 89 / Recall 93 / F1 91 using Dual Contrastive Learning +
  classical classifiers (not transformers).
- Several other datasets are surveyed (Bohra et al., HECM, BullyExplain,
  Nafis et al., Sreelakshmi et al., etc.) but are not the dataset actually
  placed in this workspace.

Neither document specifies an exact row count, exact split ratios, or a
license for the dataset actually used in the team's own experiments — the
Sprint Review deck reports results without describing the exact dataset
version behind them.

**The dataset physically present in this workspace**
(`Datasets/agr_hi_train.csv`, `Datasets/agr_hi_dev.csv`) was inspected
directly (not assumed) and confirmed to be the **TRAC-1 Hindi Facebook
subset**:
- No header row; 3 columns: `id` (e.g. `facebook_corpus_msr_401470`),
  `text`, `label`.
- Labels are exactly `NAG` / `CAG` / `OAG`.
- `agr_hi_train.csv`: 12,000 rows — CAG 4,869 / OAG 4,855 / NAG 2,275.
- `agr_hi_dev.csv`: 3,001 rows — CAG 1,246 / OAG 1,216 / NAG 538.
- No official TRAC-1 test file is present in the workspace.

This is consistent with (though not provably identical to) the corpus
referenced by Kumar et al. [4] and Sharma & Bhalla [7].

## 3. Preprocessing (as described)

The proposal and the deck both describe preprocessing only at a high level:
"noise such as URLs, mentions, repeated characters, and inconsistent
transliterations will be handled through suitable preprocessing steps" and,
in the deck, "Clean, normalize, and tokenize noisy multilingual text for
model-ready input." No concrete rules, regexes, or library choices are
specified in either document.

## 4. Model architecture (Sprint Review deck, slide 6 "Methodology" and
   slide 11 "Results")

Slide 6 lists the pipeline stages: Data Collection → Preprocessing →
**Model Training** ("Fine-tune multilingual transformer models such as
mBERT, IndicBERT and XLM-R for harmful-text classification") →
Explainability Analysis (SHAP) → Bias Evaluation → Dashboard Development
(Streamlit).

Slide 11 ("Results") reports the following table **exactly as given**,
with no explanation of dataset size, split, or hyperparameters behind it:

| Model                          | Accuracy (%) | Weighted F1 |
|---------------------------------|:---:|:---:|
| CNN + RoBERTa                   | 55.41 | 54.89 |
| XLM-R + BiLSTM + Attention      | 63.00 | 62.00 |
| XLM-Base                        | 54.05 | 65.02 |
| IndicBERT-Base                  | 64.27 | 54.31 |
| MuRIL-Base                      | 60.81 | 60.90 |
| mBERT-Base                      | 63.05 | 62.47 |
| **Proposed Stacking Ensemble**  | **68.00** | **67.00** |

Observations (recorded, not corrected):
- The "Proposed Stacking Ensemble" row is not broken down further — it is
  not stated whether this is a logistic-regression meta-classifier over
  base-model probabilities, majority voting, or another combination rule.
  There is no confusion matrix, per-class F1, or precision/recall reported.
- The XLM-Base row's weighted F1 (65.02) is *higher* than its own accuracy
  (54.05), which is unusual for standard weighted-F1/accuracy relationships
  on a 3-class task with an imbalanced label distribution; this is reported
  as-is as a probable slide/labeling inconsistency in the source deck, not
  something this project should reproduce or explain away.
- MuRIL-Base is evaluated in the deck's results table even though the deck's
  own Methodology slide (6) does not list MuRIL among the three models to be
  fine-tuned (it lists mBERT, IndicBERT, XLM-R). This project treats MuRIL as
  the explicitly optional/experimental 4th model per the task brief (§31),
  consistent with mentioning it separately from the baseline three.
- No train/val/test sizes, learning rate, epochs, batch size, or max sequence
  length are given anywhere in either source document.

## 5. Explainability (as described)

Slide 7 ("Tools & Hardware") lists **SHAP (SHapley Additive exPlanations)**
explicitly as the explainability framework. No further methodological detail
(e.g. KernelSHAP vs. gradient-based SHAP, sampling budget, which layer/output
it is applied to) is given in either document.

## 6. Fairness / bias

Both documents mention "bias evaluation" / "fairness analysis" testing
"sensitive identity-related expressions and multilingual variations," and
the proposal additionally frames this as testing "consistency" of
predictions. No concrete test cases, protected categories, or metrics are
specified in either document.

## 7. Tools / stack (deck, slide 7)

Python, Hugging Face Transformers (multilingual model fine-tuning), PyTorch,
SHAP, Streamlit (dashboard). The deck's cost-analysis slide (12) assumes
Google Colab Pro/Pro+ for GPU training — i.e., the original project assumed
cloud GPUs, not a local RTX 3050 laptop.

## 8. Limitations of the source material itself

- Neither document is a published/peer-reviewed paper; both are internal
  capstone planning/review artifacts from the same team.
- No reproducible experimental configuration (seed, split, hyperparameters,
  exact checkpoints) is given anywhere.
- No per-class metrics, confusion matrix, or statistical significance
  testing is reported — only a single aggregate accuracy/F1 pair per model.
- The stacking meta-classifier's own methodology is unstated.
- Given all of the above, the numbers in §4 are best treated as a rough,
  informal prior checkpoint from the team's own earlier work — a target to
  compare against, not a validated ground truth to reproduce exactly.
