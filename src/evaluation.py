"""Shared evaluation utilities (task section 9/28): metric computation,
confusion matrix figures, and the base-model comparison table.

Kept as a standalone module used by src/stacking.py. src/model_training.py
has its own inline copy of similar per-model evaluation logic (written and
validated before this module existed) and is intentionally left as-is here
rather than refactored to depend on this file, to avoid risking the
already-running background base-model training jobs picking up an untested
change mid-run.

CLI:
    python -m src.evaluation   # rebuilds results/model_comparison.csv from
                                # whatever results/base_models/*.json and
                                # results/stacking/*.json already exist.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np


def compute_full_metrics(y_true, y_pred, classes: List[str], split_name: str, num_samples: Optional[int] = None) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    label_ids = list(range(len(classes)))
    acc = accuracy_score(y_true, y_pred)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    per_p, per_r, per_f1, per_support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=label_ids
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)

    return {
        "split": split_name,
        "num_samples": int(num_samples if num_samples is not None else len(y_true)),
        "accuracy": float(acc),
        "precision_weighted": float(precision_w),
        "recall_weighted": float(recall_w),
        "f1_weighted": float(f1_w),
        "precision_macro": float(precision_m),
        "recall_macro": float(recall_m),
        "f1_macro": float(f1_m),
        "per_class": {
            classes[i]: {
                "precision": float(per_p[i]),
                "recall": float(per_r[i]),
                "f1": float(per_f1[i]),
                "support": int(per_support[i]),
            }
            for i in range(len(classes))
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": classes,
        "classification_report": report,
    }


def save_confusion_matrix_figure(cm: List[List[int]], classes: List[str], title: str, out_path: str) -> bool:
    """Best-effort: returns False (and prints a note) instead of raising if
    matplotlib/seaborn aren't installed, since visualization is a later-phase
    nice-to-have, not something that should crash a metrics run."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print(f"[evaluation] matplotlib/seaborn not installed; skipped figure {out_path}")
        return False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def build_model_comparison_table(config_path: str = "configs/config.yaml") -> "object":
    """Reads whatever results/base_models/*.json and
    results/stacking/ensemble_test_metrics.json actually exist on disk and
    writes results/model_comparison.csv. Never fabricates a row for a model
    that hasn't actually been evaluated (task section 29/37)."""
    import pandas as pd

    from src.dataset import load_config

    config = load_config(config_path)
    results_dir = config["paths"]["results_dir"]
    base_models_dir = os.path.join(results_dir, "base_models")

    rows = []
    display_names = {
        "xlm_roberta": "XLM-R",
        "indicbert": "IndicBERT",
        "mbert": "mBERT",
        "muril": "MuRIL",
    }

    if os.path.isdir(base_models_dir):
        for filename in sorted(os.listdir(base_models_dir)):
            if not filename.endswith(".json"):
                continue
            model_key = filename[:-5]
            with open(os.path.join(base_models_dir, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
            test = data["test"]
            rows.append({
                "Model": display_names.get(model_key, model_key),
                "Split": "test",
                "Accuracy": test["accuracy"],
                "Precision (weighted)": test["precision_weighted"],
                "Recall (weighted)": test["recall_weighted"],
                "Macro F1": test["f1_macro"],
                "Weighted F1": test["f1_weighted"],
            })

    stacking_path = os.path.join(results_dir, "stacking", "ensemble_test_metrics.json")
    if os.path.exists(stacking_path):
        with open(stacking_path, "r", encoding="utf-8") as f:
            ensemble = json.load(f)
        rows.append({
            "Model": "Stacking Ensemble",
            "Split": "test",
            "Accuracy": ensemble["accuracy"],
            "Precision (weighted)": ensemble["precision_weighted"],
            "Recall (weighted)": ensemble["recall_weighted"],
            "Macro F1": ensemble["f1_macro"],
            "Weighted F1": ensemble["f1_weighted"],
        })

    df = pd.DataFrame(rows)
    out_path = os.path.join(results_dir, "model_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"[evaluation] Wrote {out_path} with {len(df)} row(s).")
    if df.empty:
        print("[evaluation] No results/base_models/*.json or stacking results found yet -- "
              "nothing to compare. Run training + stacking first.")
    return df


if __name__ == "__main__":
    df = build_model_comparison_table()
    if not df.empty:
        print(df.to_string(index=False))
