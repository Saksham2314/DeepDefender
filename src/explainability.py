"""Explainability for DeepDefender predictions (task section 13).

Two methods are implemented, and which one produced a given explanation is
always stated in its output -- never silently assumed:

1. `occlusion` (default): leave-one-word-out perturbation attribution,
   directly on the full ensemble. For each word, the word is removed and the
   ensemble is re-run; the drop in the *predicted class's* probability is
   the word's attributed contribution. This is O(num_words) ensemble calls
   per explanation (each ensemble call = 3 transformer forward passes),
   which is fast enough to run interactively.

2. `shap`: real Shapley-value attribution via the `shap` library's
   Partition text explainer, wrapping the *entire* stacking ensemble
   end-to-end (base models + meta-classifier). This is the SHAP-based
   method the source material (docs/PAPER_BASELINE.md) refers to, applied
   "around the final classifier" per task section 13's explicitly allowed
   fallback -- but each SHAP sample also costs 3 transformer forward
   passes, so it is opt-in and bounded by `max_evals`, not the default,
   because on this hardware it is measurably slower (seconds to low tens of
   seconds per explanation vs. a fraction of a second for occlusion).

Neither method's output is a causal claim. Both explicitly label their
output as feature attribution, per task section 13.
"""

from __future__ import annotations

import argparse
import json
import re
from typing import List, Optional

from src.stacking import StackingPredictor

_WORD_SPLIT_RE = re.compile(r"\s+")

CAVEAT = (
    "This is a feature-attribution estimate (which words most changed the "
    "model's output when perturbed), not a causal explanation of why the "
    "text is or isn't aggressive, and not a guarantee that a human would "
    "cite the same words."
)


def _words(text: str) -> List[str]:
    return [w for w in _WORD_SPLIT_RE.split(text.strip()) if w]


def explain_occlusion(predictor: StackingPredictor, text: str, top_k: int = 5) -> dict:
    baseline = predictor.predict(text)
    predicted_label = baseline["label"]
    base_prob = baseline["probabilities"][predicted_label]

    words = _words(baseline["clean_text"])
    if len(words) <= 1:
        return {
            "method": "occlusion",
            "caveat": CAVEAT,
            "prediction": {
                "label": predicted_label,
                "label_name": baseline["label_name"],
                "confidence": baseline["confidence"],
            },
            "contributions": [],
            "note": "Text has 0-1 words after cleaning; no meaningful word-level attribution possible.",
        }

    contributions = []
    for i in range(len(words)):
        perturbed_text = " ".join(words[:i] + words[i + 1:])
        try:
            perturbed = predictor.predict(perturbed_text)
            perturbed_prob = perturbed["probabilities"][predicted_label]
        except ValueError:
            # Perturbed text became empty after cleaning -- skip this word
            # rather than fabricating a score for it.
            continue
        contributions.append({
            "word": words[i],
            "contribution": round(base_prob - perturbed_prob, 4),
        })

    contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)

    return {
        "method": "occlusion",
        "caveat": CAVEAT,
        "prediction": {
            "label": predicted_label,
            "label_name": baseline["label_name"],
            "confidence": baseline["confidence"],
        },
        "contributions": contributions[:top_k],
        "all_contributions": contributions,
    }


def explain_shap(predictor: StackingPredictor, text: str, max_evals: int = 64, top_k: int = 5) -> dict:
    import numpy as np
    import shap

    baseline = predictor.predict(text)
    predicted_label = baseline["label"]
    predicted_idx = predictor.classes.index(predicted_label)

    def f(texts_batch) -> "np.ndarray":
        out = []
        for t in texts_batch:
            try:
                result = predictor.predict(str(t))
                out.append([result["probabilities"][c] for c in predictor.classes])
            except ValueError:
                out.append([0.0] * len(predictor.classes))
        return np.array(out)

    masker = shap.maskers.Text(tokenizer=r"\W+")
    explainer = shap.Explainer(f, masker, output_names=predictor.classes)
    shap_values = explainer([baseline["clean_text"]], max_evals=max_evals)

    tokens = list(shap_values.data[0])
    values_for_predicted_class = shap_values.values[0][:, predicted_idx]

    contributions = [
        {"word": tok, "contribution": round(float(val), 4)}
        for tok, val in zip(tokens, values_for_predicted_class)
        if tok.strip()
    ]
    contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)

    return {
        "method": "shap",
        "max_evals": max_evals,
        "caveat": CAVEAT,
        "prediction": {
            "label": predicted_label,
            "label_name": baseline["label_name"],
            "confidence": baseline["confidence"],
        },
        "contributions": contributions[:top_k],
        "all_contributions": contributions,
    }


def explain(predictor: StackingPredictor, text: str, method: str = "occlusion",
            top_k: int = 5, max_evals: int = 64) -> dict:
    if method == "occlusion":
        return explain_occlusion(predictor, text, top_k=top_k)
    if method == "shap":
        return explain_shap(predictor, text, max_evals=max_evals, top_k=top_k)
    raise ValueError(f"Unknown explanation method '{method}'. Use 'occlusion' or 'shap'.")


def _cli():
    parser = argparse.ArgumentParser(description="DeepDefender explainability")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--method", type=str, default="occlusion", choices=["occlusion", "shap"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-evals", type=int, default=64, help="SHAP only")
    args = parser.parse_args()

    predictor = StackingPredictor()
    result = explain(predictor, args.text, method=args.method, top_k=args.top_k, max_evals=args.max_evals)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
