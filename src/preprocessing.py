"""Hinglish text preprocessing for DeepDefender.

Design goal: clean social-media noise (URLs, @mentions, hashtags, extra
whitespace) without destroying linguistic signal that a transformer model
can use. Every step is individually configurable and off-by-default unless
it is clearly safe (see configs/config.yaml -> preprocessing).

The pipeline never mutates the original text in place: callers should keep
both `original_text` and the output `clean_text` (task section 5).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_WHITESPACE_RE = re.compile(r"\s+")
# 3+ repeats of the same character collapsed down to `max_repeated_chars`.
_REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")


@dataclass
class PreprocessConfig:
    lowercase: bool = True
    replace_urls: bool = True
    url_token: str = ""
    handle_mentions: bool = True
    clean_hashtags: bool = True
    normalize_whitespace: bool = True
    strip_punctuation: bool = False
    strip_emojis: bool = False
    collapse_repeated_chars: bool = False
    max_repeated_chars: int = 2

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "PreprocessConfig":
        if not d:
            return cls()
        known = {f: d[f] for f in cls.__dataclass_fields__ if f in d}
        return cls(**known)


# Punctuation kept even when strip_punctuation=True would otherwise remove it,
# because it can carry aggression signal (repeated "!" / "?").
_PUNCTUATION_RE = re.compile(r"[^\w\s!?]")

# Broad emoji / pictograph unicode ranges. Deliberately conservative: this is
# only used when strip_emojis=True (off by default per task section 5).
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)


def clean_text(text: str, config: Optional[PreprocessConfig] = None) -> str:
    """Apply the configured cleaning pipeline to a single string.

    Order matters: hashtags/mentions/URLs are stripped before whitespace
    normalization so the resulting spacing is clean; lowercasing happens
    last so URL/mention detection isn't affected by case changes.
    """
    if config is None:
        config = PreprocessConfig()

    if text is None:
        return ""

    cleaned = str(text)

    if config.replace_urls:
        cleaned = _URL_RE.sub(config.url_token, cleaned)

    if config.handle_mentions:
        cleaned = _MENTION_RE.sub("", cleaned)

    if config.clean_hashtags:
        # "#hate" -> "hate": drop the marker, keep the word (it's often a
        # real content word in Hinglish social text).
        cleaned = _HASHTAG_RE.sub(r"\1", cleaned)

    if config.strip_emojis:
        cleaned = _EMOJI_RE.sub("", cleaned)

    if config.collapse_repeated_chars:
        cleaned = _REPEATED_CHAR_RE.sub(
            lambda m: m.group(1) * config.max_repeated_chars, cleaned
        )

    if config.strip_punctuation:
        cleaned = _PUNCTUATION_RE.sub(" ", cleaned)

    if config.lowercase:
        cleaned = cleaned.lower()

    if config.normalize_whitespace:
        cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()

    return cleaned


def preprocess_row(original_text: str, label: str, config: Optional[PreprocessConfig] = None) -> dict:
    """Return the {original_text, clean_text, label} record shape required
    by task section 5."""
    return {
        "original_text": original_text,
        "clean_text": clean_text(original_text, config),
        "label": label,
    }


def load_preprocess_config(path: str = "configs/config.yaml") -> PreprocessConfig:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        full = yaml.safe_load(f)
    return PreprocessConfig.from_dict(full.get("preprocessing"))
