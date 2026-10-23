"""Standalone XLM-RoBERTa fine-tuning entrypoint.

    python -m src.train_xlm_roberta
"""

from src.model_training import train_model

if __name__ == "__main__":
    train_model("xlm_roberta")
