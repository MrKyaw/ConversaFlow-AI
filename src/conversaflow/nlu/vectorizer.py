"""A small, dependency-free TF-IDF vectoriser.

Why not scikit-learn? Two reasons that matter for this project:

1. **Deployability.** The runtime container installs in a couple of seconds and
   the serialised model is human-readable JSON.
2. **Determinism.** Vocabulary selection is explicitly ordered, so artefacts are
   reproducible across platforms and Python versions.

The implementation supports unigrams + bigrams, sublinear term frequency,
smoothed inverse document frequency and L2 normalisation, which is the standard
configuration for short-text intent classification.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from .text import ngrams, tokenize


@dataclass
class TfidfVectorizer:
    ngram_max: int = 2
    min_df: int = 1
    max_features: int = 4000
    sublinear_tf: bool = True
    vocabulary: dict[str, int] = field(default_factory=dict)
    idf: list[float] = field(default_factory=list)
    document_count: int = 0

    def fit(self, documents: Iterable[str]) -> TfidfVectorizer:
        corpus = list(documents)
        if not corpus:
            raise ValueError("cannot fit a vectoriser on an empty corpus")
        document_frequency: Counter[str] = Counter()
        for document in corpus:
            document_frequency.update(set(self._features(document)))

        candidates = [
            (feature, count)
            for feature, count in document_frequency.items()
            if count >= self.min_df
        ]
        if not candidates:
            raise ValueError("no features survived min_df; lower min_df or add data")
        # Deterministic selection: strongest features first, then lexical order.
        candidates.sort(key=lambda item: (-item[1], item[0]))
        selected = sorted(feature for feature, _ in candidates[: self.max_features])

        self.vocabulary = {feature: index for index, feature in enumerate(selected)}
        self.document_count = len(corpus)
        self.idf = [
            math.log((1 + len(corpus)) / (1 + document_frequency[feature])) + 1.0
            for feature in selected
        ]
        return self

    def transform(self, text: str) -> dict[int, float]:
        counts = Counter(
            self.vocabulary[feature]
            for feature in self._features(text)
            if feature in self.vocabulary
        )
        if not counts:
            return {}
        vector: dict[int, float] = {}
        for index, count in counts.items():
            term_frequency = 1.0 + math.log(count) if self.sublinear_tf else float(count)
            vector[index] = term_frequency * self.idf[index]
        norm = math.sqrt(sum(value * value for value in vector.values()))
        return {index: value / norm for index, value in vector.items()}

    def transform_batch(self, texts: Sequence[str]) -> list[dict[int, float]]:
        return [self.transform(text) for text in texts]

    def _features(self, text: str) -> list[str]:
        return ngrams(tokenize(text), self.ngram_max)

    @property
    def size(self) -> int:
        return len(self.vocabulary)

    def to_dict(self) -> dict[str, object]:
        return {
            "ngram_max": self.ngram_max,
            "min_df": self.min_df,
            "max_features": self.max_features,
            "sublinear_tf": self.sublinear_tf,
            "document_count": self.document_count,
            "vocabulary": self.vocabulary,
            "idf": self.idf,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> TfidfVectorizer:
        vocabulary = payload["vocabulary"]
        idf = payload["idf"]
        if not isinstance(vocabulary, dict) or not isinstance(idf, list):
            raise ValueError("malformed vectoriser payload")
        return cls(
            ngram_max=_int_field(payload, "ngram_max"),
            min_df=_int_field(payload, "min_df"),
            max_features=_int_field(payload, "max_features"),
            sublinear_tf=_bool_field(payload, "sublinear_tf"),
            document_count=_int_field(payload, "document_count"),
            vocabulary={str(key): int(value) for key, value in vocabulary.items()},
            idf=[float(value) for value in idf],
        )


def _int_field(payload: dict[str, object], key: str) -> int:
    value = payload[key]
    if not isinstance(value, (int, str)):
        raise ValueError(f"expected an integer for {key!r}")
    return int(value)


def _bool_field(payload: dict[str, object], key: str) -> bool:
    value = payload.get(key, True)
    if not isinstance(value, bool):
        raise ValueError(f"expected a boolean for {key!r}")
    return value


def cosine(left: dict[int, float], right: dict[int, float]) -> float:
    """Cosine similarity for already-normalised sparse vectors."""

    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(index, 0.0) for index, value in left.items())
