"""Visualization generation (task section 28).

Reads only already-computed artifacts (data/splits/, results/base_models/,
results/stacking/, artifacts/models/*/final/training_history.json) and
writes PNGs to results/figures/. Never trains or mutates anything. Each
figure is skipped with a printed note (not a crash) if its source data
doesn't exist yet -- run this after training/stacking have actually
produced results.

CLI:
    python -m src.visualize
"""

from __future__ import annotations

import json
import os

import pandas as pd

from src.dataset import load_config


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_class_distribution(config: dict, figures_dir: str) -> None:
    splits_dir = config["data"]["splits_dir"]
    classes = config["labels"]["classes"]
    paths = {name: os.path.join(splits_dir, f"{name}.csv") for name in ["train", "validation", "test"]}
    if not all(os.path.exists(p) for p in paths.values()):
        print("[visualize] Skipping class_distribution: splits not found.")
        return

    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.25
    x = range(len(classes))
    for i, (name, path) in enumerate(paths.items()):
        df = pd.read_csv(path)
        counts = [int((df["label"] == c).sum()) for c in classes]
        ax.bar([xi + i * width for xi in x], counts, width=width, label=name)
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(classes)
    ax.set_ylabel("Count")
    ax.set_title("Class distribution by split")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(figures_dir, "class_distribution.png")
    fig.savefig(out_path, dpi=150)
    print(f"[visualize] Saved {out_path}")


def plot_training_curves(model_key: str, config: dict, figures_dir: str) -> None:
    history_path = os.path.join(config["paths"]["artifacts_dir"], "models", model_key, "final", "training_history.json")
    if not os.path.exists(history_path):
        print(f"[visualize] Skipping {model_key} training curves: {history_path} not found.")
        return

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    train_loss = [(h["step"], h["loss"]) for h in history if "loss" in h and "step" in h]
    eval_points = [h for h in history if "eval_loss" in h]

    plt = _mpl()

    if train_loss:
        fig, ax = plt.subplots(figsize=(6, 4))
        steps, losses = zip(*train_loss)
        ax.plot(steps, losses, label="train_loss")
        if eval_points:
            ax.plot([h.get("step", h.get("epoch")) for h in eval_points],
                     [h["eval_loss"] for h in eval_points], marker="o", label="eval_loss")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title(f"{model_key} - training/validation loss")
        ax.legend()
        fig.tight_layout()
        out_path = os.path.join(figures_dir, f"{model_key}_loss.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[visualize] Saved {out_path}")

    if eval_points:
        epochs = [h.get("epoch") for h in eval_points]
        fig, ax = plt.subplots(figsize=(6, 4))
        if "eval_accuracy" in eval_points[0]:
            ax.plot(epochs, [h["eval_accuracy"] for h in eval_points], marker="o", label="accuracy")
        if "eval_f1_macro" in eval_points[0]:
            ax.plot(epochs, [h["eval_f1_macro"] for h in eval_points], marker="o", label="macro F1")
        if "eval_f1_weighted" in eval_points[0]:
            ax.plot(epochs, [h["eval_f1_weighted"] for h in eval_points], marker="o", label="weighted F1")
        ax.set_xlabel("epoch")
        ax.set_ylabel("score")
        ax.set_ylim(0, 1)
        ax.set_title(f"{model_key} - validation metrics per epoch")
        ax.legend()
        fig.tight_layout()
        out_path = os.path.join(figures_dir, f"{model_key}_validation_metrics.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[visualize] Saved {out_path}")


def plot_model_comparison(config: dict, figures_dir: str) -> None:
    comparison_path = os.path.join(config["paths"]["results_dir"], "model_comparison.csv")
    if not os.path.exists(comparison_path):
        print(f"[visualize] Skipping model comparison: {comparison_path} not found.")
        return
    try:
        df = pd.read_csv(comparison_path)
    except pd.errors.EmptyDataError:
        print(f"[visualize] Skipping model comparison: {comparison_path} is empty (no results yet).")
        return
    if df.empty:
        print("[visualize] Skipping model comparison: no rows yet.")
        return

    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(df))
    width = 0.35
    ax.bar([xi - width / 2 for xi in x], df["Macro F1"], width=width, label="Macro F1")
    ax.bar([xi + width / 2 for xi in x], df["Weighted F1"], width=width, label="Weighted F1")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["Model"], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison (test set)")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(figures_dir, "model_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[visualize] Saved {out_path}")


def generate_all_figures(config_path: str = "configs/config.yaml") -> None:
    config = load_config(config_path)
    figures_dir = os.path.join(config["paths"]["results_dir"], "figures")
    os.makedirs(figures_dir, exist_ok=True)

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("[visualize] matplotlib not installed; skipping all figures.")
        return

    plot_class_distribution(config, figures_dir)
    for model_key in ["xlm_roberta", "indicbert", "mbert"]:
        plot_training_curves(model_key, config, figures_dir)
    plot_model_comparison(config, figures_dir)


if __name__ == "__main__":
    generate_all_figures()
