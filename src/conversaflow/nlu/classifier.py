"""Multinomial logistic regression trained with deterministic gradient descent.

The classifier is intentionally small: a few thousand sparse features and a
handful of intents. Full-batch optimisation with L2 regularisation converges in
well under a second, needs no BLAS, and gives calibrated-enough probabilities
after temperature scaling.

`temperature` is fitted on a held-out split by grid search. Calibration matters
in dialogue: the confidence score drives fallback, disambiguation and handoff
policy, so an over-confident softmax causes bad UX even when accuracy is fine.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from .vectorizer import TfidfVectorizer, cosine

Matrix = list[dict[int, float]]


@dataclass
class TrainingReport:
    epochs: int
    final_loss: float
    train_accuracy: float
    validation_accuracy: float
    temperature: float
    class_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class IntentModel:
    """Trained softmax regression plus its feature extractor."""

    vectorizer: TfidfVectorizer
    classes: list[str]
    weights: list[list[float]]
    biases: list[float]
    temperature: float = 1.0
    metadata: dict[str, object] = field(default_factory=dict)

    # ---------------------------------------------------------------- inference

    def probabilities(self, text: str) -> dict[str, float]:
        vector = self.vectorizer.transform(text)
        if not vector:
            uniform = 1.0 / len(self.classes)
            return dict.fromkeys(self.classes, uniform)
        logits = [
            sum(self.weights[index][feature] * value for feature, value in vector.items())
            + self.biases[index]
            for index in range(len(self.classes))
        ]
        return dict(zip(self.classes, _softmax(logits, self.temperature), strict=True))

    def predict(self, text: str) -> tuple[str, float, dict[str, float]]:
        probabilities = self.probabilities(text)
        ranked = sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))
        name, confidence = ranked[0]
        return name, confidence, probabilities

    def ranked(self, text: str) -> list[tuple[str, float]]:
        probabilities = self.probabilities(text)
        return sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))

    def nearest_class(self, text: str) -> tuple[str | None, float]:
        """Centroid similarity used to explain out-of-vocabulary inputs."""

        vector = self.vectorizer.transform(text)
        if not vector:
            return None, 0.0
        centroids = self.metadata.get("centroids")
        if not isinstance(centroids, dict):
            return None, 0.0
        scores = []
        for name, centroid in centroids.items():
            if not isinstance(centroid, dict):
                continue
            sparse = {int(index): float(value) for index, value in centroid.items()}
            scores.append((name, cosine(vector, sparse)))
        return max(scores, key=lambda item: (item[1], item[0]))

    # ------------------------------------------------------------ persistence

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "classes": self.classes,
            "weights": self.weights,
            "biases": self.biases,
            "temperature": self.temperature,
            "metadata": self.metadata,
            "vectorizer": self.vectorizer.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> IntentModel:
        raw_vectorizer = payload.get("vectorizer")
        raw_weights = payload.get("weights")
        raw_biases = payload.get("biases")
        raw_classes = payload.get("classes")
        raw_metadata = payload.get("metadata") or {}
        raw_temperature = payload.get("temperature", 1.0)
        if (
            not isinstance(raw_vectorizer, dict)
            or not isinstance(raw_weights, list)
            or not isinstance(raw_biases, list)
            or not isinstance(raw_classes, list)
            or not isinstance(raw_metadata, dict)
            or not isinstance(raw_temperature, (int, float))
        ):
            raise ValueError("malformed intent model payload")
        return cls(
            vectorizer=TfidfVectorizer.from_dict(raw_vectorizer),
            classes=[str(name) for name in raw_classes],
            weights=[[float(value) for value in row] for row in raw_weights],
            biases=[float(value) for value in raw_biases],
            temperature=float(raw_temperature),
            metadata={str(key): value for key, value in raw_metadata.items()},
        )


def train_intent_model(
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    epochs: int = 160,
    learning_rate: float = 1.2,
    l2: float = 2e-4,
    validation_fraction: float = 0.2,
    seed: int = 7,
    ngram_max: int = 2,
    min_df: int = 1,
    max_features: int = 4000,
) -> tuple[IntentModel, TrainingReport]:
    if len(texts) != len(labels):
        raise ValueError("texts and labels must be the same length")
    if len(texts) < 4:
        raise ValueError("need at least four training examples")
    if len(set(labels)) < 2:
        raise ValueError("need at least two intent classes")

    shuffled = list(zip(texts, labels, strict=True))
    random.Random(seed).shuffle(shuffled)
    train_size = max(1, round(len(shuffled) * (1 - validation_fraction)))
    train_size = min(train_size, len(shuffled) - 1)
    train_rows, validation_rows = shuffled[:train_size], shuffled[train_size:]

    vectorizer = TfidfVectorizer(ngram_max=ngram_max, min_df=min_df, max_features=max_features).fit(
        [text for text, _ in train_rows]
    )
    classes = sorted({label for _, label in shuffled})
    class_index = {name: index for index, name in enumerate(classes)}

    x_train = [vectorizer.transform(text) for text, _ in train_rows]
    y_train = [class_index[label] for _, label in train_rows]
    x_validation = [vectorizer.transform(text) for text, _ in validation_rows]
    y_validation = [class_index[label] for _, label in validation_rows]

    weights = [[0.0] * vectorizer.size for _ in classes]
    biases = [0.0] * len(classes)
    momentum_w = [[0.0] * vectorizer.size for _ in classes]
    momentum_b = [0.0] * len(classes)
    momentum = 0.9
    loss = 0.0
    for _ in range(epochs):
        grad_w = [[0.0] * vectorizer.size for _ in classes]
        grad_b = [0.0] * len(classes)
        loss = 0.0
        for vector, target in zip(x_train, y_train, strict=True):
            probabilities = _softmax(
                [
                    sum(weights[index][feature] * value for feature, value in vector.items())
                    + biases[index]
                    for index in range(len(classes))
                ]
            )
            loss -= math.log(max(probabilities[target], 1e-12))
            for index in range(len(classes)):
                error = probabilities[index] - (1.0 if index == target else 0.0)
                grad_b[index] += error
                for feature, value in vector.items():
                    grad_w[index][feature] += error * value
        scale = 1.0 / len(x_train)
        for index in range(len(classes)):
            bias_gradient = grad_b[index] * scale
            momentum_b[index] = momentum * momentum_b[index] + bias_gradient
            biases[index] -= learning_rate * momentum_b[index]
            row = weights[index]
            row_momentum = momentum_w[index]
            grad_row = grad_w[index]
            for feature in range(len(row)):
                gradient = grad_row[feature] * scale + l2 * row[feature]
                row_momentum[feature] = momentum * row_momentum[feature] + gradient
                row[feature] -= learning_rate * row_momentum[feature]
        loss = loss * scale + 0.5 * l2 * sum(value * value for row in weights for value in row)

    model = IntentModel(
        vectorizer=vectorizer,
        classes=classes,
        weights=weights,
        biases=biases,
    )
    model.temperature = _fit_temperature(model, x_validation, y_validation)
    model.metadata = {
        "train_size": len(train_rows),
        "validation_size": len(validation_rows),
        "seed": seed,
        "centroids": _centroids(x_train, y_train, vectorizer.size, classes),
    }
    report = TrainingReport(
        epochs=epochs,
        final_loss=round(loss, 6),
        train_accuracy=_accuracy(
            model,
            [text for text, _ in train_rows],
            [label for _, label in train_rows],
        ),
        validation_accuracy=_accuracy(
            model,
            [text for text, _ in validation_rows],
            [label for _, label in validation_rows],
        ),
        temperature=model.temperature,
        class_counts=dict(sorted({name: labels.count(name) for name in classes}.items())),
    )
    return model, report


def _softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    scaled = [value / max(temperature, 1e-6) for value in logits]
    maximum = max(scaled)
    exponents = [math.exp(value - maximum) for value in scaled]
    total = sum(exponents)
    return [value / total for value in exponents]


def _fit_temperature(
    model: IntentModel, vectors: Matrix, targets: list[int], grid: Sequence[float] | None = None
) -> float:
    candidates = grid or [0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.6, 3.0]
    best_temperature = 1.0
    best_loss = math.inf
    for temperature in candidates:
        loss = 0.0
        for vector, target in zip(vectors, targets, strict=True):
            logits = [
                sum(model.weights[index][feature] * value for feature, value in vector.items())
                + model.biases[index]
                for index in range(len(model.classes))
            ]
            probabilities = _softmax(logits, temperature)
            loss -= math.log(max(probabilities[target], 1e-12))
        if loss < best_loss:
            best_loss = loss
            best_temperature = temperature
    return best_temperature


def _accuracy(model: IntentModel, texts: Sequence[str], labels: Sequence[str]) -> float:
    if not texts:
        return 0.0
    correct = sum(
        1 for text, label in zip(texts, labels, strict=True) if model.predict(text)[0] == label
    )
    return round(correct / len(texts), 4)


def _centroids(
    vectors: Matrix, targets: list[int], size: int, classes: list[str]
) -> dict[str, dict[str, float]]:
    accumulators = [[0.0] * size for _ in classes]
    counts = [0] * len(classes)
    for vector, target in zip(vectors, targets, strict=True):
        counts[target] += 1
        for feature, value in vector.items():
            accumulators[target][feature] += value
    centroids: dict[str, dict[str, float]] = {}
    for index, name in enumerate(classes):
        total = counts[index]
        centroids[name] = {
            str(feature): value / total
            for feature, value in enumerate(accumulators[index])
            if value != 0.0 and total
        }
    return centroids
