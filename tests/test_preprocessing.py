import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import PreprocessConfig, clean_text, preprocess_row


def test_default_pipeline_matches_task_example():
    text = "@user TU BAHUT BAKWAS HAI!!! #hate"
    result = clean_text(text, PreprocessConfig())
    # From task spec section 5: should become approximately
    # "tu bahut bakwas hai" (hashtag word kept, punctuation kept by default
    # since strip_punctuation is off, mention/hashtag-marker removed).
    assert "tu bahut bakwas hai" in result
    assert "@user" not in result
    assert "#" not in result


def test_url_removed_by_default():
    text = "check this https://example.com/path?x=1 now"
    result = clean_text(text, PreprocessConfig())
    assert "http" not in result
    assert "check this" in result
    assert "now" in result


def test_mention_removed():
    result = clean_text("@someone tum pagal ho", PreprocessConfig())
    assert "@someone" not in result
    assert "pagal" in result


def test_hashtag_marker_removed_but_word_kept():
    result = clean_text("bahut bakwas #bakchodi", PreprocessConfig())
    assert "#" not in result
    assert "bakchodi" in result


def test_whitespace_normalized():
    result = clean_text("tum    bahut     bakwas   ho", PreprocessConfig())
    assert result == "tum bahut bakwas ho"


def test_punctuation_and_emojis_kept_by_default():
    text = "kya bakwas hai!! 😂"
    result = clean_text(text, PreprocessConfig())
    assert "!!" in result
    assert "😂" in result


def test_strip_punctuation_opt_in():
    # "!" and "?" are deliberately kept even with strip_punctuation=True,
    # since repeated "!!!" can itself be an aggression signal. Other
    # punctuation (e.g. commas) is removed.
    config = PreprocessConfig(strip_punctuation=True)
    result = clean_text("kya, bakwas. hai!!!", config)
    assert "," not in result
    assert "." not in result
    assert "!" in result


def test_strip_emojis_opt_in():
    config = PreprocessConfig(strip_emojis=True)
    result = clean_text("bakwas hai 😂😂", config)
    assert "😂" not in result


def test_collapse_repeated_chars_opt_in():
    config = PreprocessConfig(collapse_repeated_chars=True, max_repeated_chars=2)
    result = clean_text("bahut bakwaaaas hai", config)
    assert "bakwaaaas" not in result
    assert "bakwaas" in result


def test_collapse_repeated_chars_off_by_default():
    result = clean_text("bahut bakwaaaas hai", PreprocessConfig())
    assert "bakwaaaas" in result


def test_lowercase_toggle():
    upper_kept = clean_text("BAHUT BAKWAS HAI", PreprocessConfig(lowercase=False))
    assert upper_kept == "BAHUT BAKWAS HAI"
    lower = clean_text("BAHUT BAKWAS HAI", PreprocessConfig(lowercase=True))
    assert lower == "bahut bakwas hai"


def test_devanagari_text_preserved():
    text = "जय मोदीराज"
    result = clean_text(text, PreprocessConfig())
    assert "जय" in result
    assert "मोदीराज" in result


def test_empty_and_none_input():
    assert clean_text("", PreprocessConfig()) == ""
    assert clean_text(None, PreprocessConfig()) == ""


def test_preprocess_row_keeps_original_and_clean():
    row = preprocess_row("@user TU BAHUT BAKWAS HAI!!! #hate", "OAG", PreprocessConfig())
    assert row["original_text"] == "@user TU BAHUT BAKWAS HAI!!! #hate"
    assert row["label"] == "OAG"
    assert "tu bahut bakwas hai" in row["clean_text"]
