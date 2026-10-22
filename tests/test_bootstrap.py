"""Tests for bootstrap."""

from conversaflow.config import Settings
from conversaflow.runtime.bootstrap import build_runtime
from conversaflow.runtime.stores import InMemorySessionStore


def test_build_runtime_default(tmp_path):
    settings = Settings.from_env(
        {"CONVERSAFLOW_STORE": "memory", "CONVERSAFLOW_MODEL_PATH": str(tmp_path / "m.json")}
    )
    rt = build_runtime(settings)
    assert rt.engine is not None
    assert rt.model is not None


def test_build_runtime_with_store(tmp_path):
    settings = Settings.from_env(
        {"CONVERSAFLOW_STORE": "memory", "CONVERSAFLOW_MODEL_PATH": str(tmp_path / "m.json")}
    )
    store = InMemorySessionStore()
    rt = build_runtime(settings, store=store)
    assert rt.store is store


def test_synthesize():
    settings = Settings.from_env(
        {"CONVERSAFLOW_STORE": "memory", "CONVERSAFLOW_MODEL_PATH": "/tmp/m.json"}
    )
    rt = build_runtime(settings)
    assert rt.synthesize("hello").startswith("<speak")
