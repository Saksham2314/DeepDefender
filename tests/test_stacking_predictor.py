"""End-to-end tests against the real StackingPredictor (task section 35:
model loading, prediction output, probability sums). Skipped, not failed,
when the trained models / meta-classifier don't exist yet on this machine
-- these depend on real GPU training + stacking having actually run, which
a plain `pytest` invocation should not require. Once
`python -m src.stacking --generate --train` has been run, these activate
automatically.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.dataset import load_config

config = load_config("configs/config.yaml")
META_CLASSIFIER_PATH = os.path.join(config["paths"]["artifacts_dir"], "stacking", "meta_classifier.joblib")

pytestmark = pytest.mark.skipif(
    not os.path.exists(META_CLASSIFIER_PATH),
    reason=f"No trained meta-classifier at {META_CLASSIFIER_PATH} yet -- "
           "run `python -m src.stacking --generate --train` first.",
)


@pytest.fixture(scope="module")
def predictor():
    from src.stacking import StackingPredictor
    return StackingPredictor()


def test_predict_returns_valid_label(predictor):
    result = predictor.predict("tum bahut bakwas ho")
    assert result["label"] in ["NAG", "CAG", "OAG"]
    assert result["label_name"]


def test_predict_probabilities_sum_to_one(predictor):
    result = predictor.predict("aaj mausam accha hai")
    total = sum(result["probabilities"].values())
    assert abs(total - 1.0) < 1e-4


def test_predict_confidence_matches_max_probability(predictor):
    result = predictor.predict("chup kar bewakoof")
    assert abs(result["confidence"] - max(result["probabilities"].values())) < 1e-6


def test_predict_base_model_probabilities_present_for_all_three_models(predictor):
    result = predictor.predict("ye log kuch nahi samajhte")
    assert set(result["base_model_probabilities"].keys()) == {"xlm_roberta", "indicbert", "mbert"}
    for model_key, probs in result["base_model_probabilities"].items():
        assert abs(sum(probs.values()) - 1.0) < 1e-4


def test_predict_empty_text_raises_value_error(predictor):
    with pytest.raises(ValueError):
        predictor.predict("")


def test_predict_url_only_text_raises_value_error(predictor):
    # After preprocessing (URL stripped), this becomes empty.
    with pytest.raises(ValueError):
        predictor.predict("https://example.com/some/path")
