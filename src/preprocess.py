"""Text preprocessing for news article classification.

Keeps cleaning light and deterministic so the same transformation is applied at
training time and at inference time. Heavy normalization (stemming, stop-word
removal) is delegated to the TF-IDF vectorizer, which is configured in
``train.py``.
"""

from __future__ import annotations

import re

# Matches URLs, email addresses, and standalone numbers — noise that rarely
# helps distinguish reliable from unreliable reporting.
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalize a raw article string into a bag-of-words-friendly form.

    Lowercases, strips URLs/emails/non-alphabetic characters, and collapses
    whitespace. Returns an empty string for ``None`` or non-string input so the
    pipeline never crashes on missing data.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text)
    return text.strip()
