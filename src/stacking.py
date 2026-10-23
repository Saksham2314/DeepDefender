"""Stacking ensemble: meta-feature generation + LogisticRegression
meta-classifier (task sections 10-12). See docs/STACKING.md for the exact
methodology (held-out validation, not out-of-fold, and why).

CLI:
    python -m src.stacking --generate   # base models -> meta-features (val+test)
    python -m src.stacking --train      # fit + evaluate the meta-classifier
    python -m src.stacking --predict "tum bahut bakwas ho"
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from src.dataset import load_config
from src.preprocessing import PreprocessConfig, clean_text


def _load_models_config(path: str = "configs/models.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _base_model_final_dir(config: dict, model_key: str) -> str:
    return os.path.join(config["paths"]["artifacts_dir"], "models", model_key, "final")


def _get_probabilities(model, tokenizer, texts: List[str], max_length: int,
                        device: str, batch_size: int = 16) -> np.ndarray:
    import torch

    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = tokenizer(
                batch, truncation=True, max_length=max_length,
                padding=True, return_tensors="pt",
            ).to(device)
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def generate_meta_features_for_split(split_name: str,
                                      config: dict,
                                      models_config: dict,
                                      stack_name: str = "baseline") -> pd.DataFrame:
    """Loads each base model in `stacks.<stack_name>` ONE AT A TIME (task
    section 36: never hold multiple fine-tuned transformers in GPU memory
    simultaneously), runs inference on the given split, and returns a
    dataframe with per-model class-probability columns plus the true label.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    classes = config["labels"]["classes"]
    splits_dir = config["data"]["splits_dir"]
    split_path = os.path.join(splits_dir, f"{split_name}.csv")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"[stacking] Missing split file: {split_path}")

    df = pd.read_csv(split_path)
    texts = df["clean_text"].astype(str).tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stack_keys = models_config["stacks"][stack_name]
    all_models = {**models_config["base_models"], **models_config.get("experimental_models", {})}

    result = df[["id", "text_hash", "label"]].copy()

    for model_key in stack_keys:
        final_dir = _base_model_final_dir(config, model_key)
        if not os.path.exists(final_dir):
            raise FileNotFoundError(
                f"[stacking] No trained model found for '{model_key}' at {final_dir}. "
                f"Run `python -m src.train --model {model_key}` first."
            )

        print(f"[stacking] Loading '{model_key}' from {final_dir} for {split_name} inference...")
        tokenizer = AutoTokenizer.from_pretrained(final_dir)
        model = AutoModelForSequenceClassification.from_pretrained(final_dir).to(device)

        max_length = all_models[model_key].get("max_length", config["training"]["max_length"])
        probs = _get_probabilities(model, tokenizer, texts, max_length, device)

        for i, cls in enumerate(classes):
            result[f"{model_key}_{cls}"] = probs[:, i]

        # Free GPU memory before loading the next base model (task section 36).
        del model
        del tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"[stacking] Done with '{model_key}' for {split_name} "
              f"({len(texts)} samples).")

    return result


def _meta_features_path(config: dict, split_name: str) -> str:
    return os.path.join(config["data"]["processed_dir"], f"meta_features_{split_name}.csv")


def generate_and_save(config_path: str = "configs/config.yaml",
                       models_config_path: str = "configs/models.yaml",
                       stack_name: str = "baseline") -> None:
    config = load_config(config_path)
    models_config = _load_models_config(models_config_path)

    # Only validation + test are needed for held-out-validation stacking
    # (docs/STACKING.md) -- the train split is deliberately NOT used here.
    for split_name in ["validation", "test"]:
        df = generate_meta_features_for_split(split_name, config, models_config, stack_name)
        out_path = _meta_features_path(config, split_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"[stacking] Saved meta-features -> {out_path} ({len(df)} rows)")


def _meta_feature_columns(models_config: dict, classes: List[str], stack_name: str = "baseline") -> List[str]:
    cols = []
    for model_key in models_config["stacks"][stack_name]:
        for cls in classes:
            cols.append(f"{model_key}_{cls}")
    return cols


def train_meta_classifier(config_path: str = "configs/config.yaml",
                           models_config_path: str = "configs/models.yaml",
                           stack_name: str = "baseline") -> dict:
    import joblib
    from sklearn.linear_model import LogisticRegression

    from src.evaluation import compute_full_metrics, save_confusion_matrix_figure

    config = load_config(config_path)
    models_config = _load_models_config(models_config_path)
    classes = config["labels"]["classes"]
    label2id = {c: i for i, c in enumerate(classes)}

    val_path = _meta_features_path(config, "validation")
    test_path = _meta_features_path(config, "test")
    if not os.path.exists(val_path) or not os.path.exists(test_path):
        print("[stacking] Meta-features not found on disk yet; generating now.")
        generate_and_save(config_path, models_config_path, stack_name)

    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    feature_cols = _meta_feature_columns(models_config, classes, stack_name)

    X_val = val_df[feature_cols].values
    y_val = val_df["label"].map(label2id).values
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].map(label2id).values

    mc_cfg = config["stacking"]["meta_classifier"]
    print(f"[stacking] Training LogisticRegression meta-classifier on "
          f"{X_val.shape} validation meta-features (feature order: {feature_cols})")
    clf = LogisticRegression(
        penalty=mc_cfg["penalty"],
        C=mc_cfg["C"],
        max_iter=mc_cfg["max_iter"],
        class_weight=mc_cfg["class_weight"],
        multi_class=mc_cfg.get("multi_class", "auto"),
    )
    clf.fit(X_val, y_val)

    train_preds = clf.predict(X_val)
    train_metrics = compute_full_metrics(y_val, train_preds, classes, "validation (meta-classifier training data)")
    print(f"[stacking] Meta-classifier fit on validation -> "
          f"accuracy={train_metrics['accuracy']:.4f} f1_macro={train_metrics['f1_macro']:.4f} "
          "(this is a training-set score for the meta-classifier itself, NOT the reported ensemble result)")

    test_preds = clf.predict(X_test)
    test_metrics = compute_full_metrics(y_test, test_preds, classes, "test")
    print(f"[stacking] ENSEMBLE test result -> accuracy={test_metrics['accuracy']:.4f} "
          f"f1_weighted={test_metrics['f1_weighted']:.4f} f1_macro={test_metrics['f1_macro']:.4f}")

    stacking_dir = config["paths"]["artifacts_dir"] + "/stacking"
    os.makedirs(stacking_dir, exist_ok=True)
    model_path = os.path.join(stacking_dir, "meta_classifier.joblib")
    joblib.dump({
        "model": clf,
        "feature_columns": feature_cols,
        "classes": classes,
        "label2id": label2id,
        "stack_name": stack_name,
    }, model_path)
    print(f"[stacking] Saved meta-classifier -> {model_path}")

    results_stacking_dir = os.path.join(config["paths"]["results_dir"], "stacking")
    os.makedirs(results_stacking_dir, exist_ok=True)
    with open(os.path.join(results_stacking_dir, "meta_classifier_training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(train_metrics, f, indent=2)
    with open(os.path.join(results_stacking_dir, "ensemble_test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    save_confusion_matrix_figure(
        test_metrics["confusion_matrix"], classes,
        "Stacking Ensemble - Test Confusion Matrix",
        os.path.join(config["paths"]["results_dir"], "figures", "stacking_ensemble_confusion_matrix.png"),
    )

    return {"train_metrics": train_metrics, "test_metrics": test_metrics, "model_path": model_path}


def run_ablation_study(config_path: str = "configs/config.yaml",
                        models_config_path: str = "configs/models.yaml") -> "object":
    """Task section 30: measure whether stacking actually helps, and
    whether each model contributes, WITHOUT claiming it does in advance.

    Reuses the already-generated meta_features_{validation,test}.csv (task
    section 20 / 36 style efficiency: no base model is re-run for this --
    those CSVs already contain every baseline model's 3 columns, so an
    ablation combination is just a column subset + a fresh, cheap
    LogisticRegression fit). Requires `python -m src.stacking --generate`
    (baseline stack) to have been run first.

    A "single model" row is NOT the same number as that model's own
    argmax-based results/base_models/<key>.json accuracy: here the model's
    3 raw probabilities are passed through the same LogisticRegression
    meta-classifier machinery the multi-model rows use, for a fair,
    apples-to-apples comparison of "meta-classifier over feature set X"
    across combinations. This distinction is preserved in the output, not
    collapsed into a single ambiguous number.
    """
    import itertools

    import joblib
    from sklearn.linear_model import LogisticRegression

    from src.evaluation import compute_full_metrics

    config = load_config(config_path)
    models_config = _load_models_config(models_config_path)
    classes = config["labels"]["classes"]
    label2id = {c: i for i, c in enumerate(classes)}
    baseline_models = models_config["stacks"]["baseline"]

    val_path = _meta_features_path(config, "validation")
    test_path = _meta_features_path(config, "test")
    if not os.path.exists(val_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            "[ablation] Meta-features not found. Run `python -m src.stacking --generate` "
            "(baseline stack) first -- ablation reuses those files rather than "
            "re-running inference for every combination."
        )
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    mc_cfg = config["stacking"]["meta_classifier"]

    rows = []
    for r in range(1, len(baseline_models) + 1):
        for combo in itertools.combinations(baseline_models, r):
            feature_cols = [f"{m}_{c}" for m in combo for c in classes]
            X_val = val_df[feature_cols].values
            y_val = val_df["label"].map(label2id).values
            X_test = test_df[feature_cols].values
            y_test = test_df["label"].map(label2id).values

            clf = LogisticRegression(
                penalty=mc_cfg["penalty"], C=mc_cfg["C"], max_iter=mc_cfg["max_iter"],
                class_weight=mc_cfg["class_weight"], multi_class=mc_cfg.get("multi_class", "auto"),
            )
            clf.fit(X_val, y_val)
            test_metrics = compute_full_metrics(y_test, clf.predict(X_test), classes, "test")

            combo_label = "+".join(combo)
            print(f"[ablation] {combo_label} -> test accuracy={test_metrics['accuracy']:.4f} "
                  f"f1_weighted={test_metrics['f1_weighted']:.4f} f1_macro={test_metrics['f1_macro']:.4f}")
            rows.append({
                "models": combo_label,
                "num_models": len(combo),
                "test_accuracy": test_metrics["accuracy"],
                "test_precision_weighted": test_metrics["precision_weighted"],
                "test_recall_weighted": test_metrics["recall_weighted"],
                "test_f1_weighted": test_metrics["f1_weighted"],
                "test_f1_macro": test_metrics["f1_macro"],
            })

    df = pd.DataFrame(rows)
    results_dir = config["paths"]["results_dir"]
    out_path = os.path.join(results_dir, "experiments", "ablation_study.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[ablation] Saved -> {out_path}")
    print("[ablation] NOTE: whether more models or stacking itself improved results "
          "is whatever this table actually shows -- not assumed in advance.")
    return df


class StackingPredictor:
    """Runtime inference wrapper (task section 12). Loads all base models +
    tokenizers + the trained meta-classifier once, and answers predict()
    calls without ever re-reading disk. Intended for use by api/main.py.

    On a 4GB-VRAM machine, holding all 3 base models on GPU simultaneously
    at inference time (batch size 1, short sequences) is the practical
    choice for interactive API latency; unlike training/meta-feature
    generation, this is a small, fixed, low-activation-memory workload.
    If VRAM is insufficient, `device` falls back to CPU (see
    src/environment.py checks) rather than failing to load.
    """

    def __init__(self, config_path: str = "configs/config.yaml",
                 models_config_path: str = "configs/models.yaml"):
        import joblib
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.config = load_config(config_path)
        self.models_config = _load_models_config(models_config_path)
        self.classes = self.config["labels"]["classes"]
        self.label_names = self.config["labels"]["names"]
        self.preprocess_config = PreprocessConfig.from_dict(self.config.get("preprocessing"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        meta_path = os.path.join(self.config["paths"]["artifacts_dir"], "stacking", "meta_classifier.joblib")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"[StackingPredictor] No trained meta-classifier at {meta_path}. "
                "Run `python -m src.stacking --generate` then `--train` first."
            )
        bundle = joblib.load(meta_path)
        self.meta_classifier = bundle["model"]
        self.feature_columns = bundle["feature_columns"]
        self.stack_name = bundle["stack_name"]

        self.models = {}
        self.tokenizers = {}
        self.max_lengths = {}
        all_models = {**self.models_config["base_models"], **self.models_config.get("experimental_models", {})}
        for model_key in self.models_config["stacks"][self.stack_name]:
            final_dir = _base_model_final_dir(self.config, model_key)
            if not os.path.exists(final_dir):
                raise FileNotFoundError(
                    f"[StackingPredictor] No trained model for '{model_key}' at {final_dir}."
                )
            self.tokenizers[model_key] = AutoTokenizer.from_pretrained(final_dir)
            self.models[model_key] = AutoModelForSequenceClassification.from_pretrained(final_dir).to(self.device)
            self.models[model_key].eval()
            self.max_lengths[model_key] = all_models[model_key].get(
                "max_length", self.config["training"]["max_length"]
            )

    def predict(self, text: str) -> dict:
        cleaned = clean_text(text, self.preprocess_config)
        if not cleaned.strip():
            raise ValueError("Input text is empty after preprocessing; cannot classify.")

        feature_row = {}
        for model_key in self.models_config["stacks"][self.stack_name]:
            probs = _get_probabilities(
                self.models[model_key], self.tokenizers[model_key],
                [cleaned], self.max_lengths[model_key], self.device, batch_size=1,
            )[0]
            for i, cls in enumerate(self.classes):
                feature_row[f"{model_key}_{cls}"] = probs[i]

        X = np.array([[feature_row[c] for c in self.feature_columns]])
        pred_idx = int(self.meta_classifier.predict(X)[0])
        pred_probs = self.meta_classifier.predict_proba(X)[0]

        label = self.classes[pred_idx]
        return {
            "label": label,
            "label_name": self.label_names[label],
            "probabilities": {cls: float(pred_probs[i]) for i, cls in enumerate(self.classes)},
            "confidence": float(pred_probs[pred_idx]),
            "clean_text": cleaned,
            "base_model_probabilities": {
                model_key: {
                    cls: float(feature_row[f"{model_key}_{cls}"]) for cls in self.classes
                }
                for model_key in self.models_config["stacks"][self.stack_name]
            },
        }


def _cli():
    parser = argparse.ArgumentParser(description="DeepDefender stacking ensemble")
    parser.add_argument("--generate", action="store_true", help="Generate meta-features from trained base models")
    parser.add_argument("--train", action="store_true", help="Train + evaluate the meta-classifier")
    parser.add_argument("--ablation", action="store_true", help="Run all base-model subset combinations (task section 30)")
    parser.add_argument("--predict", type=str, default=None, help="Run a single prediction through the full stack")
    parser.add_argument("--stack", type=str, default="baseline", choices=["baseline", "experimental"])
    args = parser.parse_args()

    if args.generate:
        generate_and_save(stack_name=args.stack)
        return

    if args.train:
        train_meta_classifier(stack_name=args.stack)
        return

    if args.ablation:
        run_ablation_study()
        return

    if args.predict is not None:
        predictor = StackingPredictor()
        result = predictor.predict(args.predict)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    parser.print_help()


if __name__ == "__main__":
    _cli()
