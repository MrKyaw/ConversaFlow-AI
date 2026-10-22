"""Training and artefact management for the intent classifier."""

from __future__ import annotations

import json
from pathlib import Path

from ..errors import ValidationError
from .classifier import IntentModel, TrainingReport, train_intent_model


def load_training_data(path: Path) -> tuple[list[str], list[str]]:
    """Read a JSONL corpus of ``{"text": ..., "intent": ...}`` rows."""

    if not path.exists():
        raise ValidationError(f"training corpus not found: {path}")
    texts: list[str] = []
    labels: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path}:{line_number}: invalid JSON ({exc.msg})") from exc
            text = row.get("text")
            intent = row.get("intent")
            if not isinstance(text, str) or not text.strip():
                raise ValidationError(f"{path}:{line_number}: missing non-empty 'text'")
            if not isinstance(intent, str) or not intent.strip():
                raise ValidationError(f"{path}:{line_number}: missing non-empty 'intent'")
            texts.append(text.strip())
            labels.append(intent.strip())
    if not texts:
        raise ValidationError(f"training corpus is empty: {path}")
    return texts, labels


def train_from_file(path: Path) -> tuple[IntentModel, TrainingReport]:
    texts, labels = load_training_data(path)
    return train_intent_model(texts, labels)


def save_model(model: IntentModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(model.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def load_model(path: Path) -> IntentModel:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"unable to load model artefact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValidationError(f"model artefact {path} must contain a JSON object")
    try:
        return IntentModel.from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError(f"invalid model artefact {path}: {exc}") from exc


def load_or_train(model_path: Path, train_path: Path) -> tuple[IntentModel, TrainingReport | None]:
    """Load a cached artefact, or train one from the shipped corpus.

    Auto-training keeps a fresh clone runnable with a single command while still
    letting CI and production pin an immutable artefact.
    """

    if model_path.exists():
        return load_model(model_path), None
    model, report = train_from_file(train_path)
    save_model(model, model_path)
    return model, report
