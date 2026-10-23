import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.explainability import explain_occlusion, CAVEAT

BAD_WORDS = {"bakwas", "pagal", "bewakoof"}


class FakePredictor:
    """Deterministic stand-in for StackingPredictor: OAG probability rises
    with the number of "bad" words present, so occlusion contributions are
    predictable and checkable without loading any real model."""

    classes = ["NAG", "CAG", "OAG"]

    def predict(self, text: str) -> dict:
        text = text.strip()
        if not text:
            raise ValueError("Input text is empty after preprocessing; cannot classify.")
        words = text.lower().split()
        bad_count = sum(1 for w in words if w in BAD_WORDS)
        oag_prob = min(0.95, 0.1 + 0.3 * bad_count)
        remaining = 1.0 - oag_prob
        nag_prob = remaining * 0.5
        cag_prob = remaining * 0.5
        probs = {"NAG": nag_prob, "CAG": cag_prob, "OAG": oag_prob}
        label = max(probs, key=probs.get)
        return {
            "label": label,
            "label_name": label,
            "probabilities": probs,
            "confidence": probs[label],
            "clean_text": text,
        }


def test_occlusion_flags_the_bad_word_as_top_contributor():
    predictor = FakePredictor()
    result = explain_occlusion(predictor, "tum bahut bakwas ho", top_k=3)

    assert result["method"] == "occlusion"
    assert result["caveat"] == CAVEAT
    assert result["prediction"]["label"] == "OAG"

    top_words = [c["word"] for c in result["contributions"]]
    assert "bakwas" in top_words
    # Removing "bakwas" should be the single biggest driver of the OAG
    # probability drop, since it's the only "bad" word in this sentence.
    assert result["contributions"][0]["word"] == "bakwas"
    assert result["contributions"][0]["contribution"] > 0


def test_occlusion_neutral_sentence_has_small_contributions():
    predictor = FakePredictor()
    result = explain_occlusion(predictor, "aaj mausam accha hai", top_k=3)
    assert result["prediction"]["label"] == "NAG"
    for c in result["contributions"]:
        assert abs(c["contribution"]) < 0.2


def test_occlusion_single_word_returns_no_contributions():
    predictor = FakePredictor()
    result = explain_occlusion(predictor, "bakwas", top_k=3)
    assert result["contributions"] == []
    assert "note" in result


def test_occlusion_top_k_limits_returned_contributions():
    predictor = FakePredictor()
    result = explain_occlusion(predictor, "ye bahut bakwas aur pagal jaisi baat hai", top_k=2)
    assert len(result["contributions"]) == 2
    assert len(result["all_contributions"]) > 2
