"""Fairness / bias probing (task section 14).

Tests whether swapping only an identity/group term within an otherwise
identical sentence changes the model's prediction or confidence. This is a
narrow, template-based probe, not a comprehensive fairness audit -- its
output describes *observed behavior on these specific templates only* and
must never be read as a general "the model is fair/unfair" verdict (task
section 37 explicitly forbids claiming fairness from a few examples).

The test cases in data/fairness/fairness_cases.csv are a small, editable
starter set (religion, gender, and region terms in a handful of Hinglish
sentence templates) -- not a definitive or exhaustive list of protected
categories. Add/remove rows in that CSV to extend coverage; nothing here
hard-codes what counts as a "protected group."

CLI:
    python -m src.fairness
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import pandas as pd

from src.dataset import load_config
from src.stacking import StackingPredictor


def run_fairness_tests(config_path: str = "configs/config.yaml") -> dict:
    config = load_config(config_path)
    cases_path = config["paths"]["fairness_cases"]
    if not os.path.exists(cases_path):
        raise FileNotFoundError(f"[fairness] Missing {cases_path}")

    cases = pd.read_csv(cases_path)
    predictor = StackingPredictor(config_path)

    rows = []
    for _, case in cases.iterrows():
        result = predictor.predict(case["text"])
        rows.append({
            "template_id": case["template_id"],
            "category": case["category"],
            "group_term": case["group_term"],
            "text": case["text"],
            "predicted_label": result["label"],
            "confidence": result["confidence"],
            "prob_NAG": result["probabilities"].get("NAG"),
            "prob_CAG": result["probabilities"].get("CAG"),
            "prob_OAG": result["probabilities"].get("OAG"),
        })

    results_df = pd.DataFrame(rows)

    # Per-template analysis: does the predicted label change across group
    # terms that are otherwise in an identical sentence?
    template_analysis = {}
    for template_id, group in results_df.groupby("template_id"):
        labels = group["predicted_label"].unique().tolist()
        confidences = group["confidence"]
        template_analysis[template_id] = {
            "template_text_example": group["text"].iloc[0],
            "num_group_variants": len(group),
            "distinct_predicted_labels": labels,
            "label_changes_across_group_terms": len(labels) > 1,
            "confidence_min": float(confidences.min()),
            "confidence_max": float(confidences.max()),
            "confidence_range": float(confidences.max() - confidences.min()),
            "per_group_term": group[["group_term", "category", "predicted_label", "confidence"]].to_dict(orient="records"),
        }

    num_templates_with_label_changes = sum(
        1 for t in template_analysis.values() if t["label_changes_across_group_terms"]
    )

    report = {
        "disclaimer": (
            "This report describes observed prediction behavior on a small, "
            "hand-written set of templates and group terms. It is NOT a "
            "comprehensive fairness audit, does not cover all possible "
            "identity groups or phrasing, and passing/failing these cases "
            "does not establish that the model is 'fair' or 'unfair' in "
            "general. See src/fairness.py and data/fairness/fairness_cases.csv."
        ),
        "num_cases": len(results_df),
        "num_templates": len(template_analysis),
        "num_templates_with_label_changes_across_group_terms": num_templates_with_label_changes,
        "templates": template_analysis,
    }

    results_dir = config["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "fairness_report.csv")
    json_path = os.path.join(results_dir, "fairness_report.json")
    results_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[fairness] Ran {len(results_df)} cases across {len(template_analysis)} templates.")
    print(f"[fairness] Templates where the predicted label changed across group terms: "
          f"{num_templates_with_label_changes}/{len(template_analysis)}")
    print(f"[fairness] Saved -> {csv_path}, {json_path}")
    print(f"[fairness] {report['disclaimer']}")

    return report


def _cli():
    parser = argparse.ArgumentParser(description="DeepDefender fairness probe")
    parser.parse_args()
    run_fairness_tests()


if __name__ == "__main__":
    _cli()
