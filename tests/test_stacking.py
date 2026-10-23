import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.stacking import _load_models_config, _meta_feature_columns
from src.dataset import load_config


def test_meta_feature_dimension_is_nine_for_baseline_stack():
    # Task section 10/35: 3 base models x 3 class probabilities = 9 features.
    config = load_config("configs/config.yaml")
    models_config = _load_models_config("configs/models.yaml")
    classes = config["labels"]["classes"]

    cols = _meta_feature_columns(models_config, classes, "baseline")
    assert len(cols) == 9


def test_baseline_stack_has_exactly_three_models():
    models_config = _load_models_config("configs/models.yaml")
    assert len(models_config["stacks"]["baseline"]) == 3
    assert models_config["stacks"]["baseline"] == ["xlm_roberta", "indicbert", "mbert"]


def test_meta_feature_column_order_matches_model_then_class_order():
    config = load_config("configs/config.yaml")
    models_config = _load_models_config("configs/models.yaml")
    classes = config["labels"]["classes"]

    cols = _meta_feature_columns(models_config, classes, "baseline")
    expected = [
        "xlm_roberta_NAG", "xlm_roberta_CAG", "xlm_roberta_OAG",
        "indicbert_NAG", "indicbert_CAG", "indicbert_OAG",
        "mbert_NAG", "mbert_CAG", "mbert_OAG",
    ]
    assert cols == expected


def test_experimental_stack_swaps_mbert_for_muril():
    models_config = _load_models_config("configs/models.yaml")
    assert "muril" in models_config["stacks"]["experimental"]
    assert "mbert" not in models_config["stacks"]["experimental"]
    assert len(models_config["stacks"]["experimental"]) == 3
