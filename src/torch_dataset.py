"""PyTorch Dataset wrapper for tokenized Hinglish aggression text.

Kept separate from src/dataset.py, which has no torch dependency and must
stay importable in a Phase-1-only environment (no torch/transformers
installed). Only import this module once torch/transformers are available.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
import torch
from torch.utils.data import Dataset


class AggressionDataset(Dataset):
    """Wraps a prepared split dataframe (must have `clean_text` and `label`
    columns) for use with a Hugging Face tokenizer + Trainer.

    Padding is left to the caller's DataCollator (dynamic padding per batch)
    rather than done here, since that's more VRAM-efficient on a 4GB GPU
    than fixed max_length padding for every example.
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int,
                 label2id: Dict[str, int], text_column: str = "clean_text"):
        texts = df[text_column].astype(str).tolist()
        labels_raw = df["label"].tolist()
        unknown = set(labels_raw) - set(label2id.keys())
        if unknown:
            raise ValueError(f"AggressionDataset: labels {unknown} not in label2id {label2id}")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.labels = [label2id[l] for l in labels_raw]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
