"""Tests for evaluation harness."""

from pathlib import Path

from conversaflow.config import Settings
from conversaflow.evaluation.harness import evaluate_conversations, evaluate_intents, to_dict
from conversaflow.runtime.bootstrap import build_runtime


def test_evaluate_intents():
    settings = Settings.from_env(
        {"CONVERSAFLOW_STORE": "memory", "CONVERSAFLOW_MODEL_PATH": "var/e.json"}
    )
    rt = build_runtime(settings)
    report = evaluate_intents(rt.nlu, Path("data/intents/eval.jsonl"))
    assert report.total == 85
    assert report.accuracy > 0.8
    d = to_dict(report)
    assert "per_intent" in d


def test_evaluate_conversations():
    settings = Settings.from_env(
        {"CONVERSAFLOW_STORE": "memory", "CONVERSAFLOW_MODEL_PATH": "var/e.json"}
    )
    rt = build_runtime(settings)
    report = evaluate_conversations(rt, Path("data/golden"))
    assert report.total == 8
    assert report.success_rate == 1.0


def test_evaluate_empty(tmp_path):
    import pytest

    settings = Settings.from_env(
        {"CONVERSAFLOW_STORE": "memory", "CONVERSAFLOW_MODEL_PATH": "var/e.json"}
    )
    rt = build_runtime(settings)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        evaluate_conversations(rt, empty)
