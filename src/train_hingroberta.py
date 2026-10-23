"""Standalone HingRoBERTa fine-tuning entrypoint.

HingRoBERTa (l3cube-pune/hing-roberta) is not part of the baseline 3-model
stack (task's XLM-R + IndicBERT + mBERT) -- it's a standalone comparison
model, pretrained specifically on Hindi-English code-mixed text, trained
and evaluated here on the exact same data/splits/{train,validation,test}.csv
as the other models, with no ensembling/stacking involved.

    python -m src.train_hingroberta
"""

from src.model_training import train_model

if __name__ == "__main__":
    train_model("hingroberta")
