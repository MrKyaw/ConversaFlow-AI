"""Tests for data loaders."""

from pathlib import Path

import pytest

from conversaflow.data_loader import load_flows, load_jsonl, load_knowledge
from conversaflow.errors import ConfigurationError, ValidationError


def test_load_flows_missing_dir(tmp_path):
    with pytest.raises(ConfigurationError):
        load_flows(tmp_path / "nope")


def test_load_flows_empty_dir(tmp_path):
    d = tmp_path / "flows"
    d.mkdir()
    with pytest.raises(ConfigurationError):
        load_flows(d)


def test_load_knowledge_missing():
    assert load_knowledge(Path("/nope/faq.jsonl")) == []


def test_load_jsonl_missing():
    with pytest.raises(ValidationError):
        load_jsonl(Path("/nope/x.jsonl"))


def test_load_jsonl_valid(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text('{"a": 1}\n# comment\n\n{"b": 2}\n')
    assert load_jsonl(p) == [{"a": 1}, {"b": 2}]


def test_load_jsonl_invalid_json(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text("{bad")
    with pytest.raises(ValidationError):
        load_jsonl(p)


def test_load_jsonl_not_object(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text("[1,2]\n")
    with pytest.raises(ValidationError):
        load_jsonl(p)


def test_load_knowledge_invalid(tmp_path):
    p = tmp_path / "kb.jsonl"
    p.write_text("not json\n")
    with pytest.raises(ConfigurationError):
        load_knowledge(p)
