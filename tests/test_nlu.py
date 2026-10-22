"""Tests for NLU pipeline and training."""

from pathlib import Path

import pytest

from conversaflow.config import Settings
from conversaflow.errors import ValidationError
from conversaflow.nlu.pipeline import NluConfig, NluPipeline
from conversaflow.nlu.train import (
    load_model,
    load_or_train,
    load_training_data,
    save_model,
    train_from_file,
)
from conversaflow.runtime.bootstrap import build_runtime


def test_load_training_data(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text('{"text": "hi", "intent": "greet"}\n')
    texts, labels = load_training_data(p)
    assert texts == ["hi"] and labels == ["greet"]


def test_load_training_data_missing(tmp_path):
    with pytest.raises(ValidationError):
        load_training_data(tmp_path / "nope.jsonl")


def test_load_training_data_empty(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text("")
    with pytest.raises(ValidationError):
        load_training_data(p)


def test_load_training_data_invalid_json(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text("{bad")
    with pytest.raises(ValidationError):
        load_training_data(p)


def test_train_and_save_load(tmp_path):
    model, report = train_from_file(Path("data/intents/train.jsonl"))
    assert report.train_accuracy > 0
    out = tmp_path / "m.json"
    save_model(model, out)
    loaded = load_model(out)
    assert loaded.classes == model.classes
    # pipeline
    pipe = NluPipeline(loaded)
    r = pipe.analyse("hello")
    assert r.intent == "greet"


def test_load_or_train(tmp_path):
    model_path = tmp_path / "m.json"
    train_path = Path("data/intents/train.jsonl")
    _m1, rep1 = load_or_train(model_path, train_path)
    assert rep1 is not None
    _m2, rep2 = load_or_train(model_path, train_path)
    assert rep2 is None


def test_load_model_invalid(tmp_path):
    p = tmp_path / "m.json"
    p.write_text("not json")
    with pytest.raises(ValidationError):
        load_model(p)


def test_pipeline_low_confidence_and_oov():
    settings = Settings.from_env(
        {"CONVERSAFLOW_STORE": "memory", "CONVERSAFLOW_MODEL_PATH": "var/p.json"}
    )
    rt = build_runtime(settings)
    pipe = rt.nlu
    r = pipe.analyse("xyzzy plugh")
    assert r.out_of_vocabulary >= 0
    assert isinstance(r.low_confidence, bool)


def test_pipeline_ambiguous():
    settings = Settings.from_env(
        {"CONVERSAFLOW_STORE": "memory", "CONVERSAFLOW_MODEL_PATH": "var/p.json"}
    )
    rt = build_runtime(settings)
    pipe = rt.nlu
    r = pipe.analyse("yes")
    assert r.alternatives


def test_nlu_config():
    settings = Settings.from_env(
        {"CONVERSAFLOW_STORE": "memory", "CONVERSAFLOW_MODEL_PATH": "var/p.json"}
    )
    rt = build_runtime(settings)
    pipe = NluPipeline(rt.nlu.model, NluConfig(intent_threshold=0.99))
    r = pipe.analyse("hello")
    assert r.low_confidence == (r.confidence < 0.99)


def test_entities_extraction():
    from conversaflow.models import SlotSpec
    from conversaflow.nlu.entities import extract_entities

    specs = {
        "order_id": SlotSpec(name="order_id", type="order_id"),
        "email": SlotSpec(name="email", type="email"),
    }
    found = extract_entities("my order ORD-1234 email a@b.com", specs)
    assert "order_id" in found
    assert "email" in found


def test_language_detection():
    from conversaflow.nlu.language import detect_language

    assert detect_language("hello world") == "en"


def test_sentiment():
    from conversaflow.nlu.sentiment import analyse_sentiment

    s = analyse_sentiment("I am very happy")
    assert s.label in ("positive", "neutral", "negative")
    s2 = analyse_sentiment("this is terrible awful")
    assert s2.score <= 0
