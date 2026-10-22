"""Text normalisation and tokenisation shared by NLU, BM25 and evaluation.

The pipeline is deliberately framework-free: unicode NFKC, case folding and a
single token regex. Keeping one canonical tokeniser means training and serving
can never drift apart -- a subtle source of silent accuracy loss in production
NLU systems.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\s'-]", re.UNICODE)

STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have
    he her here him his how i if in into is it its just me my no not of on or our out
    she should so than that the their them then there these they this those to too
    us was we were what when where which who why will with would you your""".split()
)


def normalize(text: str) -> str:
    """Canonical form used for matching, de-duplication and storage."""

    folded = unicodedata.normalize("NFKC", text).casefold()
    folded = folded.replace("\u2019", "'").replace("\u2018", "'")
    folded = PUNCT_RE.sub(" ", folded)
    return WHITESPACE_RE.sub(" ", folded).strip()


def tokenize(text: str) -> list[str]:
    """Split normalised text into unicode word tokens."""

    return TOKEN_RE.findall(normalize(text))


def content_tokens(text: str) -> list[str]:
    """Tokens with English stopwords removed (used by retrieval)."""

    return [token for token in tokenize(text) if token not in STOPWORDS]


def ngrams(tokens: Iterable[str], max_n: int = 2) -> list[str]:
    """Return unigrams and contiguous bigrams, in that order.

    Bigrams are prefixed with ``b:`` so an n-gram can never collide with a
    unigram that happens to contain a space.
    """

    items = list(tokens)
    features = list(items)
    for size in range(2, max_n + 1):
        features.extend(
            f"b:{' '.join(items[index : index + size])}" for index in range(len(items) - size + 1)
        )
    return features


def truncate(text: str, limit: int = 80) -> str:
    """Shorten text for logs without cutting a word in half unnecessarily."""

    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0] + "\u2026"
