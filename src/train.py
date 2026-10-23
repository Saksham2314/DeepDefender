"""Unified training entrypoint for DeepDefender.

CLI:
    python -m src.train --check
    python -m src.train --smoke-test
    python -m src.train --model xlm_roberta [--tiny] [--epochs N] [--batch-size N]
    python -m src.train --model indicbert
    python -m src.train --model mbert

`--tiny` runs a real (not mocked) training+eval cycle on a small sample
(task section 39) so a bug is caught in minutes, not hours, before the full
dataset run is launched.
"""

from __future__ import annotations

import argparse
import os
import sys

from src.environment import check_environment, print_report
from src.dataset import load_config
from src.preprocessing import PreprocessConfig, clean_text


def run_check() -> None:
    print_report(check_environment())


def run_smoke_test(config_path: str = "configs/config.yaml") -> bool:
    """Phase-1-scoped smoke test (task section 39).

    Verifies, end to end, on a tiny sample:
      1. The prepared dataset splits exist and are readable.
      2. Preprocessing runs cleanly on real sample rows.
      3. Label set is exactly {NAG, CAG, OAG}.
      4. If transformers/torch are installed, tokenization with each
         configured base-model checkpoint succeeds on the sample.
    It does NOT attempt a tiny training loop yet (that needs Phase 2's
    model/dataset wiring) -- this is reported explicitly, not silently
    skipped.
    """
    import pandas as pd
    import yaml

    print("=" * 60)
    print("DeepDefender - Smoke Test (Phase 1 scope)")
    print("=" * 60)

    config = load_config(config_path)
    n = config["smoke_test"]["num_samples"]

    train_path = os.path.join(config["data"]["splits_dir"], "train.csv")
    if not os.path.exists(train_path):
        print(f"[smoke-test] FAILED: {train_path} does not exist yet. "
              f"Run `python -m src.dataset --prepare --size 4000` first.")
        return False

    df = pd.read_csv(train_path)
    sample = df.head(n)
    print(f"[smoke-test] Loaded {len(sample)} sample rows from {train_path} "
          f"(requested {n}).")

    labels_seen = set(sample["label"].unique())
    expected = set(config["labels"]["classes"])
    if not labels_seen.issubset(expected):
        print(f"[smoke-test] FAILED: unexpected labels {labels_seen - expected}")
        return False
    print(f"[smoke-test] Label check OK: {sorted(labels_seen)} subset of {sorted(expected)}")

    preprocess_config = PreprocessConfig.from_dict(config.get("preprocessing"))
    cleaned = [clean_text(t, preprocess_config) for t in sample["original_text"]]
    empty_after_clean = sum(1 for c in cleaned if not c.strip())
    print(f"[smoke-test] Preprocessing ran on {len(cleaned)} rows "
          f"({empty_after_clean} became empty after cleaning).")

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        torch_ready = True
    except ImportError:
        torch_ready = False

    if not torch_ready:
        print("[smoke-test] torch/transformers not installed -- skipping "
              "tokenizer + tiny-training checks. Install requirements.txt "
              "and re-run for the full Phase 2 smoke test.")
        print("[smoke-test] Phase 1 checks PASSED (dataset + preprocessing only).")
        return True

    with open("configs/models.yaml", "r", encoding="utf-8") as f:
        models_config = yaml.safe_load(f)

    from transformers import AutoTokenizer

    all_ok = True
    for key in models_config["stacks"]["baseline"]:
        checkpoint = models_config["base_models"][key]["checkpoint"]
        try:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            encoded = tokenizer(
                list(cleaned[:5]),
                padding=True,
                truncation=True,
                max_length=config["smoke_test"]["max_length"],
                return_tensors="pt",
            )
            print(f"[smoke-test] Tokenizer OK for {key} ({checkpoint}): "
                  f"input_ids shape {tuple(encoded['input_ids'].shape)}")
        except Exception as e:  # noqa: BLE001
            print(f"[smoke-test] FAILED to load/tokenize with {key} ({checkpoint}): {e}")
            all_ok = False

    if all_ok:
        print("[smoke-test] All checks PASSED.")
    else:
        print("[smoke-test] Some checks FAILED (see above).")
    return all_ok


def _cli():
    parser = argparse.ArgumentParser(description="DeepDefender training entrypoint")
    parser.add_argument("--check", action="store_true", help="Print environment/GPU report")
    parser.add_argument("--smoke-test", action="store_true", help="Run the Phase 1 smoke test")
    parser.add_argument("--model", type=str, default=None,
                         choices=["xlm_roberta", "indicbert", "mbert", "muril", "hingroberta"],
                         help="Fine-tune a specific base model")
    parser.add_argument("--tiny", action="store_true",
                         help="Real tiny-scale train+eval (smoke_test.num_samples rows, "
                              "smoke_test.epochs epoch) instead of the full dataset")
    parser.add_argument("--epochs", type=int, default=None, help="Override configs/config.yaml training.epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override configs/config.yaml training.batch_size")
    args = parser.parse_args()

    if args.check:
        run_check()
        return

    if args.smoke_test:
        ok = run_smoke_test()
        sys.exit(0 if ok else 1)

    if args.model:
        from src.model_training import train_model

        if args.tiny:
            config = load_config()
            n = config["smoke_test"]["num_samples"]
            splits_dir = config["data"]["splits_dir"]
            tiny_dir = os.path.join(splits_dir, "_tiny")
            os.makedirs(tiny_dir, exist_ok=True)

            import pandas as pd
            for split_name in ["train", "validation", "test"]:
                full_df = pd.read_csv(os.path.join(splits_dir, f"{split_name}.csv"))
                tiny_df = full_df.groupby("label", group_keys=False)[full_df.columns].apply(
                    lambda g: g.head(max(1, n // 3))
                )
                tiny_df.to_csv(os.path.join(tiny_dir, f"{split_name}.csv"), index=False)

            train_model(
                args.model,
                override_epochs=args.epochs or config["smoke_test"]["epochs"],
                override_batch_size=args.batch_size or config["smoke_test"]["batch_size"],
                override_max_length=config["smoke_test"]["max_length"],
                override_train_path=os.path.join(tiny_dir, "train.csv"),
                override_val_path=os.path.join(tiny_dir, "validation.csv"),
                override_test_path=os.path.join(tiny_dir, "test.csv"),
                output_subdir=f"_tiny_{args.model}",
            )
        else:
            train_model(args.model, override_epochs=args.epochs, override_batch_size=args.batch_size)
        return

    parser.print_help()


if __name__ == "__main__":
    _cli()
