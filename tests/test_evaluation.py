import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml

from src.evaluation import build_model_comparison_table, compute_full_metrics


def test_compute_full_metrics_shape():
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 1, 0, 2, 2]
    classes = ["NAG", "CAG", "OAG"]

    metrics = compute_full_metrics(y_true, y_pred, classes, "test")

    assert metrics["split"] == "test"
    assert metrics["num_samples"] == 9
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert set(metrics["per_class"].keys()) == set(classes)
    assert len(metrics["confusion_matrix"]) == 3
    assert len(metrics["confusion_matrix"][0]) == 3


def test_compute_full_metrics_perfect_predictions():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]
    classes = ["NAG", "CAG", "OAG"]

    metrics = compute_full_metrics(y_true, y_pred, classes, "test")
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0
    assert metrics["f1_weighted"] == 1.0


def _make_isolated_config(tmp_path) -> str:
    """Builds a throwaway config.yaml pointing results_dir at a tmp
    directory, so these tests never touch the real (and possibly
    concurrently-being-written-to, by a live training run) results/ tree.
    """
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    results_dir = tmp_path / "results"
    config["paths"]["results_dir"] = str(results_dir)
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    return str(config_path), str(results_dir)


def test_build_model_comparison_table_never_fabricates_missing_models(tmp_path):
    config_path, results_dir = _make_isolated_config(tmp_path)
    os.makedirs(os.path.join(results_dir, "base_models"), exist_ok=True)

    df = build_model_comparison_table(config_path)
    assert df.empty


def test_build_model_comparison_table_reads_real_json(tmp_path):
    config_path, results_dir = _make_isolated_config(tmp_path)
    base_models_dir = os.path.join(results_dir, "base_models")
    os.makedirs(base_models_dir, exist_ok=True)

    fake_result = {
        "model_key": "xlm_roberta",
        "test": {
            "accuracy": 0.5, "precision_weighted": 0.5, "recall_weighted": 0.5,
            "f1_weighted": 0.5, "f1_macro": 0.5,
        },
    }
    with open(os.path.join(base_models_dir, "xlm_roberta.json"), "w", encoding="utf-8") as f:
        json.dump(fake_result, f)

    df = build_model_comparison_table(config_path)
    assert len(df) == 1
    assert df.iloc[0]["Model"] == "XLM-R"
    assert df.iloc[0]["Accuracy"] == 0.5
