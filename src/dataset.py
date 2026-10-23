"""Dataset loading, reproducible subsetting, and stratified splitting for
DeepDefender (task sections 3-4).

The raw TRAC-1 Hindi Facebook CSVs have NO header row and are positionally
id, text, label. This is verified directly against the files in Datasets/,
not assumed.

CLI:
    python -m src.dataset --stats
    python -m src.dataset --prepare --size 4000 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import os
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from src.preprocessing import PreprocessConfig, clean_text

REQUIRED_LABELS = ["NAG", "CAG", "OAG"]


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_raw(config: dict) -> pd.DataFrame:
    """Load and concatenate all configured raw TRAC-1 CSV files.

    Raises FileNotFoundError with a clear message (task section 37: "If the
    dataset is missing: stop dataset-dependent execution and explain exactly
    what is needed") rather than silently downloading or fabricating data.
    """
    raw_dir = config["data"]["raw_dir"]
    columns = config["data"]["columns"]
    frames = []
    missing = []

    for filename in config["data"]["raw_files"]:
        path = os.path.join(raw_dir, filename)
        if not os.path.exists(path):
            missing.append(path)
            continue
        df = pd.read_csv(path, header=None, names=columns, encoding="utf-8")
        df["source_file"] = filename
        frames.append(df)

    if missing:
        raise FileNotFoundError(
            "DeepDefender: required raw dataset file(s) not found:\n  "
            + "\n  ".join(missing)
            + "\n\nSee docs/DATASET_SETUP.md for where to place the TRAC-1 "
            "Hindi-English aggression dataset (agr_hi_train.csv, "
            "agr_hi_dev.csv) before running dataset preparation."
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["label"] = combined["label"].astype(str).str.strip()

    bad_labels = sorted(set(combined["label"].unique()) - set(REQUIRED_LABELS))
    if bad_labels:
        raise ValueError(
            f"DeepDefender: unexpected label values found in raw data: {bad_labels}. "
            f"Expected only {REQUIRED_LABELS}. Refusing to silently drop/relabel rows."
        )

    combined["text_hash"] = combined["text"].astype(str).map(_text_hash)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["text_hash"]).reset_index(drop=True)
    dropped = before - len(combined)
    if dropped:
        print(f"[dataset] Dropped {dropped} exact-duplicate rows (by text hash) "
              f"out of {before} loaded from raw files.")

    return combined


def dataset_stats(df: pd.DataFrame) -> dict:
    counts = df["label"].value_counts().to_dict()
    total = len(df)
    return {
        "total_rows": total,
        "label_counts": counts,
        "label_proportions": {k: round(v / total, 4) for k, v in counts.items()},
    }


def print_stats(title: str, df: pd.DataFrame) -> None:
    stats = dataset_stats(df)
    print(f"--- {title} ---")
    print(f"Total rows: {stats['total_rows']}")
    for label in REQUIRED_LABELS:
        count = stats["label_counts"].get(label, 0)
        prop = stats["label_proportions"].get(label, 0.0)
        print(f"  {label}: {count} ({prop * 100:.1f}%)")


def build_reproducible_subset(df: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    """Stratified sample of `size` rows from `df`, preserving each label's
    proportion in the full pool, with a fixed seed so the exact same subset
    is produced every time (task section 3: "Do NOT randomly select rows
    every time.").
    """
    if size >= len(df):
        print(f"[dataset] Requested subset size {size} >= pool size {len(df)}; "
              "using the full pool instead.")
        return df.reset_index(drop=True)

    rng = np.random.RandomState(seed)
    proportions = df["label"].value_counts(normalize=True)

    parts = []
    allocated = 0
    labels = list(proportions.index)
    for i, label in enumerate(labels):
        if i == len(labels) - 1:
            n = size - allocated  # remainder goes to the last label, avoids rounding loss
        else:
            n = int(round(size * proportions[label]))
            allocated += n
        label_rows = df[df["label"] == label]
        n = min(n, len(label_rows))
        sampled = label_rows.sample(n=n, random_state=rng)
        parts.append(sampled)

    subset = pd.concat(parts).sample(frac=1.0, random_state=rng).reset_index(drop=True)
    return subset


def stratified_split(df: pd.DataFrame, split_config: dict, seed: int):
    from sklearn.model_selection import train_test_split

    train_ratio = split_config["train"]
    val_ratio = split_config["validation"]
    test_ratio = split_config["test"]
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "data.split ratios in configs/config.yaml must sum to 1.0"
    )

    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df["label"],
        random_state=seed,
    )
    relative_test_size = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df["label"],
        random_state=seed,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def prepare_dataset(config_path: str = "configs/config.yaml",
                     size: Optional[int] = None,
                     seed: Optional[int] = None) -> None:
    config = load_config(config_path)
    size = size if size is not None else config["data"]["dataset_size"]
    seed = seed if seed is not None else config["data"]["random_seed"]

    print(f"[dataset] Loading raw TRAC-1 files from '{config['data']['raw_dir']}'...")
    raw_df = load_raw(config)
    print_stats("Full raw pool (train+dev combined)", raw_df)

    print(f"\n[dataset] Building reproducible subset: size={size}, seed={seed}")
    subset = build_reproducible_subset(raw_df, size=size, seed=seed)
    print_stats("Reproducible subset", subset)

    preprocess_config = PreprocessConfig.from_dict(config.get("preprocessing"))
    subset = subset.copy()
    subset["original_text"] = subset["text"]
    subset["clean_text"] = subset["text"].astype(str).map(
        lambda t: clean_text(t, preprocess_config)
    )

    processed_dir = config["data"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)
    subset_path = os.path.join(processed_dir, f"subset_{size}.csv")
    subset_out = subset[["id", "text_hash", "original_text", "clean_text", "label", "source_file"]]
    subset_out.to_csv(subset_path, index=False, encoding="utf-8")
    print(f"\n[dataset] Saved subset ({len(subset_out)} rows) -> {subset_path}")

    print(f"\n[dataset] Splitting: {config['data']['split']} (seed={seed})")
    train_df, val_df, test_df = stratified_split(subset_out, config["data"]["split"], seed)

    splits_dir = config["data"]["splits_dir"]
    os.makedirs(splits_dir, exist_ok=True)
    for name, split_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        path = os.path.join(splits_dir, f"{name}.csv")
        split_df.to_csv(path, index=False, encoding="utf-8")
        print_stats(f"{name} split -> {path}", split_df)

    # Sanity check: no text_hash appears in more than one split (no leakage).
    train_hashes = set(train_df["text_hash"])
    val_hashes = set(val_df["text_hash"])
    test_hashes = set(test_df["text_hash"])
    overlap = (train_hashes & val_hashes) | (train_hashes & test_hashes) | (val_hashes & test_hashes)
    if overlap:
        raise RuntimeError(
            f"[dataset] INTEGRITY ERROR: {len(overlap)} text_hash values appear in "
            "more than one split. This must never happen; aborting rather than "
            "silently shipping a leaky split."
        )
    print("\n[dataset] Integrity check passed: train/validation/test splits are disjoint.")


def _cli():
    parser = argparse.ArgumentParser(description="DeepDefender dataset preparation")
    parser.add_argument("--prepare", action="store_true", help="Build the reproducible subset and splits")
    parser.add_argument("--stats", action="store_true", help="Print stats for the full raw pool only")
    parser.add_argument("--size", type=int, default=None, help="Subset size (default: from config.yaml)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: from config.yaml)")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    if args.stats:
        config = load_config(args.config)
        raw_df = load_raw(config)
        print_stats("Full raw pool (train+dev combined)", raw_df)
        return

    if args.prepare:
        prepare_dataset(args.config, size=args.size, seed=args.seed)
        return

    parser.print_help()


if __name__ == "__main__":
    _cli()
