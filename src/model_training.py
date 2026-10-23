"""Generic fine-tuning + evaluation logic shared by all three base models
(task section 8/9). src/train_xlm_roberta.py, src/train_indicbert.py, and
src/train_mbert.py are thin wrappers around `train_model()` here; so is
`python -m src.train --model <key>`.

Design choices, made explicit rather than left implicit:
  - Uses Hugging Face `Trainer` (not a hand-rolled loop) for fp16, gradient
    accumulation, gradient clipping, checkpointing, and early stopping --
    all individually configurable via configs/config.yaml, none hard-coded.
  - Dynamic padding per batch (DataCollatorWithPadding) instead of fixed
    max_length padding, since VRAM is the binding constraint on this
    hardware (task section 7).
  - Class weighting (if enabled) is applied via a Trainer subclass
    overriding compute_loss with a weighted CrossEntropyLoss, not by
    resampling the dataset -- keeps the measured dataset size accurate.
  - The test set is evaluated exactly once, after training is complete, and
    is never used for early stopping or checkpoint selection (that uses the
    validation set only) -- task section 4's "keep the test set completely
    isolated".
"""

from __future__ import annotations

import glob
import json
import os
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from src.dataset import load_config
from src.environment import check_environment


def _load_models_config(path: str = "configs/models.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _next_experiment_id(experiments_dir: str, model_key: str) -> str:
    os.makedirs(experiments_dir, exist_ok=True)
    existing = glob.glob(os.path.join(experiments_dir, "experiment_*.json"))
    numbers = []
    for path in existing:
        base = os.path.basename(path)
        try:
            numbers.append(int(base.split("_")[1]))
        except (IndexError, ValueError):
            continue
    next_n = (max(numbers) + 1) if numbers else 1
    return f"experiment_{next_n:03d}_{model_key}"


def train_model(model_key: str,
                 config_path: str = "configs/config.yaml",
                 models_config_path: str = "configs/models.yaml",
                 override_epochs: Optional[int] = None,
                 override_batch_size: Optional[int] = None,
                 override_max_length: Optional[int] = None,
                 override_train_path: Optional[str] = None,
                 override_val_path: Optional[str] = None,
                 override_test_path: Optional[str] = None,
                 output_subdir: Optional[str] = None) -> dict:
    import torch
    from torch import nn
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    from src.torch_dataset import AggressionDataset

    config = load_config(config_path)
    models_config = _load_models_config(models_config_path)

    all_models = {**models_config["base_models"], **models_config.get("experimental_models", {})}
    if model_key not in all_models:
        raise ValueError(f"Unknown model_key '{model_key}'. Known: {list(all_models.keys())}")
    model_spec = all_models[model_key]
    checkpoint = model_spec["checkpoint"]

    classes = config["labels"]["classes"]
    label2id = {label: i for i, label in enumerate(classes)}
    id2label = {i: label for label, i in label2id.items()}

    train_cfg = config["training"]
    epochs = override_epochs if override_epochs is not None else train_cfg["epochs"]
    batch_size = override_batch_size if override_batch_size is not None else train_cfg["batch_size"]
    max_length = override_max_length if override_max_length is not None else model_spec.get("max_length", train_cfg["max_length"])

    train_path = override_train_path or os.path.join(config["data"]["splits_dir"], "train.csv")
    val_path = override_val_path or os.path.join(config["data"]["splits_dir"], "validation.csv")
    test_path = override_test_path or os.path.join(config["data"]["splits_dir"], "test.csv")

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"[train:{model_key}] Required split file missing: {p}. "
                "Run `python -m src.dataset --prepare` first."
            )

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    env = check_environment()
    gpu_vram_gb = f"{env.gpu_total_vram_mb / 1024:.1f} GB" if env.gpu_total_vram_mb else "unknown"

    print("=" * 70)
    print(f"DeepDefender - Training '{model_key}' ({model_spec['display_name']})")
    print("=" * 70)
    print(f"Checkpoint             : {checkpoint}")
    print(f"GPU                    : {env.gpu_name or 'none detected'}")
    print(f"GPU VRAM                : {gpu_vram_gb}")
    print(f"CUDA available (torch) : {env.cuda_available}")
    print(f"Batch size              : {batch_size}")
    print(f"Gradient accumulation   : {train_cfg['gradient_accumulation_steps']} "
          f"(effective batch = {batch_size * train_cfg['gradient_accumulation_steps']})")
    print(f"Max sequence length     : {max_length}")
    print(f"Epochs                  : {epochs}")
    print(f"Learning rate            : {train_cfg['learning_rate']}")
    print(f"Dataset size (train)    : {len(train_df)}")
    print(f"Dataset size (val)      : {len(val_df)}")
    print(f"Dataset size (test)     : {len(test_df)}")
    print(f"Class weighting          : {train_cfg['class_weighting']}")
    print("=" * 70)

    device = "cuda" if env.cuda_available else "cpu"
    if device == "cpu":
        print(f"[train:{model_key}] WARNING: CUDA not available, falling back to CPU. "
              "This will be substantially slower.")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=len(classes),
        id2label=id2label,
        label2id=label2id,
    )
    if train_cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()

    train_dataset = AggressionDataset(train_df, tokenizer, max_length, label2id)
    val_dataset = AggressionDataset(val_df, tokenizer, max_length, label2id)
    test_dataset = AggressionDataset(test_df, tokenizer, max_length, label2id)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
        f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)
        return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}

    model_output_dir = os.path.join(
        config["paths"]["artifacts_dir"], "models", output_subdir or model_key
    )
    checkpoints_dir = os.path.join(model_output_dir, "checkpoints")
    os.makedirs(model_output_dir, exist_ok=True)

    fp16 = bool(train_cfg.get("fp16")) and device == "cuda"

    # transformers>=5 dropped `warmup_ratio` from TrainingArguments (only
    # `warmup_steps` remains) -- compute the equivalent step count ourselves
    # so configs/config.yaml's warmup_ratio setting still has an effect.
    steps_per_epoch = max(1, len(train_dataset) // (batch_size * train_cfg["gradient_accumulation_steps"]))
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * train_cfg["warmup_ratio"])

    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=max(batch_size, 8),
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_steps=warmup_steps,
        max_grad_norm=train_cfg["max_grad_norm"],
        fp16=fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=25,
        report_to=[],
        seed=config["project"]["seed"],
        disable_tqdm=False,
    )

    class_weights_tensor = None
    if train_cfg.get("class_weighting"):
        counts = train_df["label"].value_counts()
        weights = [len(train_df) / (len(classes) * counts.get(c, 1)) for c in classes]
        class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
        print(f"[train:{model_key}] Class weights ({classes}): {weights}")

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor)
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    trainer_cls = WeightedTrainer if class_weights_tensor is not None else Trainer

    callbacks = []
    if train_cfg.get("early_stopping_patience"):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=train_cfg["early_stopping_patience"]))

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    resume_from = train_cfg.get("resume_from_checkpoint")
    start_time = time.time()
    trainer.train(resume_from_checkpoint=resume_from)
    training_seconds = time.time() - start_time

    print(f"[train:{model_key}] Training finished in {training_seconds / 60:.1f} minutes.")

    def full_evaluate(dataset, df, split_name: str) -> dict:
        pred_output = trainer.predict(dataset)
        logits = pred_output.predictions
        preds = np.argmax(logits, axis=-1)
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        labels = pred_output.label_ids

        acc = accuracy_score(labels, preds)
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0
        )
        precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        per_class_p, per_class_r, per_class_f1, per_class_support = precision_recall_fscore_support(
            labels, preds, average=None, zero_division=0, labels=list(range(len(classes)))
        )
        cm = confusion_matrix(labels, preds, labels=list(range(len(classes))))
        report = classification_report(
            labels, preds, target_names=classes, output_dict=True, zero_division=0
        )

        result = {
            "split": split_name,
            "num_samples": len(df),
            "accuracy": acc,
            "precision_weighted": precision_w,
            "recall_weighted": recall_w,
            "f1_weighted": f1_w,
            "precision_macro": precision_m,
            "recall_macro": recall_m,
            "f1_macro": f1_m,
            "per_class": {
                classes[i]: {
                    "precision": float(per_class_p[i]),
                    "recall": float(per_class_r[i]),
                    "f1": float(per_class_f1[i]),
                    "support": int(per_class_support[i]),
                }
                for i in range(len(classes))
            },
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_labels": classes,
            "classification_report": report,
        }
        print(f"[train:{model_key}] {split_name} -> accuracy={acc:.4f} "
              f"f1_weighted={f1_w:.4f} f1_macro={f1_m:.4f}")
        return result

    val_metrics = full_evaluate(val_dataset, val_df, "validation")
    test_metrics = full_evaluate(test_dataset, test_df, "test")

    final_dir = os.path.join(model_output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    training_config_record = {
        "model_key": model_key,
        "checkpoint": checkpoint,
        "classes": classes,
        "label2id": label2id,
        "seed": config["project"]["seed"],
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": train_cfg["gradient_accumulation_steps"],
        "effective_batch_size": batch_size * train_cfg["gradient_accumulation_steps"],
        "max_length": max_length,
        "learning_rate": train_cfg["learning_rate"],
        "weight_decay": train_cfg["weight_decay"],
        "warmup_ratio": train_cfg["warmup_ratio"],
        "max_grad_norm": train_cfg["max_grad_norm"],
        "fp16": fp16,
        "gradient_checkpointing": bool(train_cfg.get("gradient_checkpointing")),
        "class_weighting": bool(train_cfg.get("class_weighting")),
        "early_stopping_patience": train_cfg.get("early_stopping_patience"),
        "dataset_train_path": train_path,
        "dataset_val_path": val_path,
        "dataset_test_path": test_path,
        "dataset_train_size": len(train_df),
        "dataset_val_size": len(val_df),
        "dataset_test_size": len(test_df),
        "device": device,
        "gpu_name": env.gpu_name,
        "gpu_vram_mb": env.gpu_total_vram_mb,
        "training_seconds": training_seconds,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(final_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(training_config_record, f, indent=2)
    with open(os.path.join(final_dir, "validation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)
    with open(os.path.join(final_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    with open(os.path.join(final_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    print(f"[train:{model_key}] Saved model+tokenizer -> {final_dir}")

    # Only a real, standard-scale run (output_subdir is None) is allowed to
    # write into the canonical results/ locations. A --tiny or otherwise
    # custom-output_subdir run must never overwrite real base-model results
    # or fabricate an experiment record for a toy-scale run.
    if output_subdir is None:
        base_models_dir = os.path.join(config["paths"]["results_dir"], "base_models")
        os.makedirs(base_models_dir, exist_ok=True)
        with open(os.path.join(base_models_dir, f"{model_key}.json"), "w", encoding="utf-8") as f:
            json.dump({
                "model_key": model_key,
                "checkpoint": checkpoint,
                "training_config": training_config_record,
                "validation": val_metrics,
                "test": test_metrics,
            }, f, indent=2)

        try:
            from src.evaluation import save_confusion_matrix_figure
            save_confusion_matrix_figure(
                test_metrics["confusion_matrix"], classes,
                f"{model_spec['display_name']} - Test Confusion Matrix",
                os.path.join(config["paths"]["results_dir"], "figures", f"{model_key}_confusion_matrix.png"),
            )
        except Exception as e:  # noqa: BLE001
            # Figure generation is a nice-to-have (task section 28); never
            # let it fail a training run that otherwise succeeded.
            print(f"[train:{model_key}] Could not save confusion matrix figure: {e}")

        experiments_dir = config["paths"]["results_dir"] + "/experiments"
        experiment_id = _next_experiment_id(experiments_dir, model_key)
        with open(os.path.join(experiments_dir, f"{experiment_id}.json"), "w", encoding="utf-8") as f:
            json.dump({
                "experiment_id": experiment_id,
                "model_key": model_key,
                "training_config": training_config_record,
                "validation_metrics_summary": {
                    "accuracy": val_metrics["accuracy"],
                    "f1_weighted": val_metrics["f1_weighted"],
                    "f1_macro": val_metrics["f1_macro"],
                },
                "test_metrics_summary": {
                    "accuracy": test_metrics["accuracy"],
                    "f1_weighted": test_metrics["f1_weighted"],
                    "f1_macro": test_metrics["f1_macro"],
                },
            }, f, indent=2)
        print(f"[train:{model_key}] Saved base-model results -> {base_models_dir}/{model_key}.json")
        print(f"[train:{model_key}] Saved experiment record -> {experiments_dir}/{experiment_id}.json")
    else:
        print(f"[train:{model_key}] output_subdir='{output_subdir}' (non-standard run): "
              "skipped writing to results/base_models and results/experiments to avoid "
              "polluting real results with a toy-scale run.")
        experiment_id = None

    return {
        "model_key": model_key,
        "final_dir": final_dir,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "training_seconds": training_seconds,
        "experiment_id": experiment_id,
    }
