"""Standalone mBERT fine-tuning entrypoint.

    python -m src.train_mbert
"""

from src.model_training import train_model

if __name__ == "__main__":
    train_model("mbert")
