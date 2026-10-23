"""Standalone IndicBERT fine-tuning entrypoint.

    python -m src.train_indicbert
"""

from src.model_training import train_model

if __name__ == "__main__":
    train_model("indicbert")
