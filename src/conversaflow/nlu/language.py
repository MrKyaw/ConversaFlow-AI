"""Lightweight language identification.

Script detection plus stopword scoring over the four languages the demo corpus
supports. It is not a replacement for fastText, but it is deterministic, costs
microseconds, and is enough to route a session to the right locale or flag a
language mismatch for analytics.
"""

from __future__ import annotations

import re

_SCRIPTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"[\u0400-\u04FF]"), "ru"),
    (re.compile(r"[\u0600-\u06FF]"), "ar"),
    (re.compile(r"[\u4E00-\u9FFF]"), "zh"),
    (re.compile(r"[\u3040-\u30FF]"), "ja"),
    (re.compile(r"[\uAC00-\uD7AF]"), "ko"),
)

_STOPWORDS = {
    "en": frozenset("the is are and to of i you my please want need how what where".split()),
    "es": frozenset("el la los las es y de que mi por favor quiero necesito como donde".split()),
    "fr": frozenset("le la les est et de que mon pour favor je veux besoin comment ou".split()),
    "de": frozenset("der die das ist und von ich mein bitte mochte brauche wie wo".split()),
}
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def detect_language(text: str, default: str = "en") -> str:
    """Return an ISO-639-1 code; unknown scripts fall back to Latin scoring."""

    for pattern, language in _SCRIPTS:
        if pattern.search(text):
            return language
    tokens = _WORD_RE.findall(text.lower())
    if not tokens:
        return default
    scores = {
        language: sum(1 for token in tokens if token in vocabulary)
        for language, vocabulary in _STOPWORDS.items()
    }
    best = max(scores, key=lambda language: (scores[language], language == default, language))
    return best if scores[best] > 0 else default
