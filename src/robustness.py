"""Robustness testing for Hinglish text variations (task section 15).

Compares the ensemble's prediction on a "base" sentence against variants
with spelling changes, repeated characters, capitalization, punctuation,
emojis, URLs, mentions, hashtags, code-switching, and misspellings, all
applied to the SAME underlying sentence (data/robustness/robustness_cases.csv).
This measures prediction *stability* under surface-level noise -- it does
not measure accuracy (there is no ground-truth label for these synthetic
sentences) and is not a claim that the model "handles" any particular
phenomenon robustly, only a record of what changed and what didn't.

CLI:
    python -m src.robustness
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from src.dataset import load_config
from src.stacking import StackingPredictor


def run_robustness_tests(config_path: str = "configs/config.yaml",
                          cases_path: str = "data/robustness/robustness_cases.csv") -> pd.DataFrame:
    config = load_config(config_path)
    if not os.path.exists(cases_path):
        raise FileNotFoundError(f"[robustness] Missing {cases_path}")

    cases = pd.read_csv(cases_path)
    predictor = StackingPredictor(config_path)

    rows = []
    base_by_group = {}

    for group_id, group in cases.groupby("group_id"):
        base_rows = group[group["perturbation_type"] == "base"]
        if base_rows.empty:
            print(f"[robustness] WARNING: group '{group_id}' has no 'base' row; skipping.")
            continue
        base_text = base_rows.iloc[0]["text"]
        base_result = predictor.predict(base_text)
        base_by_group[group_id] = base_result

        for _, case in group.iterrows():
            result = predictor.predict(case["text"])
            rows.append({
                "group_id": group_id,
                "perturbation_type": case["perturbation_type"],
                "text": case["text"],
                "predicted_label": result["label"],
                "confidence": result["confidence"],
                "base_label": base_result["label"],
                "base_confidence": base_result["confidence"],
                "label_matches_base": result["label"] == base_result["label"],
                "confidence_delta": round(result["confidence"] - base_result["confidence"], 4),
            })

    report_df = pd.DataFrame(rows)

    results_dir = config["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "robustness_report.csv")
    report_df.to_csv(out_path, index=False)

    non_base = report_df[report_df["perturbation_type"] != "base"]
    total = len(non_base)
    changed = int((~non_base["label_matches_base"]).sum())
    print(f"[robustness] Ran {len(report_df)} cases across {len(base_by_group)} groups.")
    print(f"[robustness] Label changed vs. base in {changed}/{total} non-base variants.")
    if total:
        by_type = non_base.groupby("perturbation_type")["label_matches_base"].mean().sort_values()
        print("[robustness] Label-match rate by perturbation type (lower = more label changes):")
        for ptype, rate in by_type.items():
            print(f"    {ptype}: {rate * 100:.0f}% matched base")
    print(f"[robustness] Saved -> {out_path}")

    return report_df


def _cli():
    parser = argparse.ArgumentParser(description="DeepDefender robustness probe")
    parser.parse_args()
    run_robustness_tests()


if __name__ == "__main__":
    _cli()
