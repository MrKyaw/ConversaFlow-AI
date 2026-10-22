"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from conversaflow.config import Settings
from conversaflow.runtime.bootstrap import build_runtime


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings.from_env(
        {
            "CONVERSAFLOW_STORE": "memory",
            "CONVERSAFLOW_MODEL_PATH": str(tmp_path / "model.json"),
            "CONVERSAFLOW_DATA_DIR": "data",
        }
    )


@pytest.fixture
def runtime(settings: Settings, trained_model):
    return build_runtime(settings, model=trained_model)


@pytest.fixture
def engine(runtime):
    return runtime.engine


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
def trained_model(tmp_path_factory):
    """Train once; each runtime still receives isolated stores and knowledge."""
    from conversaflow.nlu.train import load_or_train

    model, _ = load_or_train(
        tmp_path_factory.mktemp("model") / "nlu.json", Path("data/intents/train.jsonl")
    )
    return model
