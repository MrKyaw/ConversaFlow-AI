"""Lexicon + negation sentiment analysis.

A compact rule-based scorer is the right tool here: it is explainable, runs in
microseconds, needs no model download, and handles the one thing tiny sentiment
classifiers usually get wrong -- negation scope ("not bad" is positive).
"""

from __future__ import annotations

import math
import re
from typing import Literal

from ..models import Sentiment

_POSITIVE = {
    "amazing": 3.0, "awesome": 3.0, "best": 2.5, "brilliant": 2.5, "delighted": 2.5,
    "excellent": 3.0, "fantastic": 3.0, "glad": 1.5, "good": 2.0, "great": 2.5,
    "happy": 2.0, "helpful": 1.5, "impressed": 2.0, "incredible": 3.0, "love": 2.5,
    "lovely": 2.0, "nice": 1.5, "perfect": 3.0, "pleased": 2.0, "quick": 1.0,
    "recommend": 1.5, "resolved": 1.0, "satisfied": 2.0, "smooth": 1.5,
    "thanks": 1.5, "thank": 1.5, "wonderful": 2.5, "works": 1.0,
}  # fmt: skip
_NEGATIVE = {
    "angry": -2.5, "annoyed": -2.0, "awful": -3.0, "bad": -2.0, "broken": -2.5,
    "cancel": -1.0, "confusing": -1.5, "crash": -2.5, "disappointed": -2.5,
    "disgusting": -3.0, "error": -1.5, "fail": -2.5, "failed": -2.5, "frustrated": -2.5,
    "furious": -3.0, "hate": -2.5, "hopeless": -2.5, "horrible": -3.0, "late": -1.5,
    "lost": -1.5, "missing": -1.5, "never": -1.0, "poor": -2.0, "problem": -1.5,
    "refund": -0.5, "ridiculous": -2.5, "slow": -1.5, "stuck": -2.0, "terrible": -3.0,
    "unacceptable": -3.0, "unhappy": -2.0, "useless": -2.5, "worse": -2.5, "worst": -3.0,
    "wrong": -1.5,
}  # fmt: skip
_NEGATORS = frozenset(
    "not no never none nothing nobody nowhere neither nor cannot can't don't doesn't didn't "
    "isn't wasn't aren't weren't won't wouldn't shouldn't couldn't hardly barely".split()
)
_INTENSIFIERS = {
    "very": 1.5, "really": 1.4, "extremely": 1.8, "so": 1.3, "super": 1.5,
    "totally": 1.5, "absolutely": 1.7, "slightly": 0.6, "somewhat": 0.7,
}  # fmt: skip
_EMOJI = {
    "\U0001f600": 2.0, "\U0001f642": 1.5, "\U0001f60a": 2.0, "\U0001f44d": 1.5,
    "\U0001f389": 2.5, "\U0001f622": -2.0, "\U0001f61e": -2.0, "\U0001f621": -3.0,
    "\U0001f44e": -2.0, "\U0001f4a9": -3.0,
}  # fmt: skip
_TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)


def analyse_sentiment(text: str) -> Sentiment:
    tokens = _TOKEN_RE.findall(text.lower().replace("\u2019", "'"))
    total = 0.0
    hits = 0
    for index, token in enumerate(tokens):
        weight = _POSITIVE.get(token, _NEGATIVE.get(token, 0.0))
        if weight == 0.0:
            continue
        window = tokens[max(0, index - 3) : index]
        multiplier = 1.0
        if any(word in _NEGATORS for word in window):
            multiplier *= -1.0
        for word in window:
            multiplier *= _INTENSIFIERS.get(word, 1.0)
        total += weight * multiplier
        hits += 1
    for emoji, weight in _EMOJI.items():
        if emoji in text:
            total += weight
            hits += 1
    if hits == 0:
        return Sentiment(score=0.0, label="neutral", intensity=0.0)
    if "!" in text:
        total *= 1.15
    score = math.tanh(total / 3.0)
    intensity = min(1.0, abs(total) / 6.0)
    if score <= -0.15:
        label: Literal["negative", "neutral", "positive"] = "negative"
    elif score >= 0.15:
        label = "positive"
    else:
        label = "neutral"
    return Sentiment(score=round(score, 4), label=label, intensity=round(intensity, 4))
