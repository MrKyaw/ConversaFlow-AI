"""Data loaders with contextual error messages.

Reference data is part of the deployable artefact. A malformed flow should stop
start-up with the file name and a useful reason, never surface as a stack trace
in the middle of a conversation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from .dialogue.flow import Flow
from .errors import ConfigurationError, ValidationError
from .models import KnowledgeDocumentIn


def load_flows(directory: Path) -> dict[str, Flow]:
    if not directory.exists():
        raise ConfigurationError(f"flow directory not found: {directory}")
    flows: dict[str, Flow] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            flow = Flow.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, PydanticValidationError, ValueError) as exc:
            raise ConfigurationError(f"invalid flow {path.name}: {exc}") from exc
        if flow.id in flows:
            raise ConfigurationError(f"duplicate flow id {flow.id!r} in {path.name}")
        flows[flow.id] = flow
    if not flows:
        raise ConfigurationError(f"no flow definitions found in {directory}")
    for flow in flows.values():
        for node in flow.nodes:
            if node.type == "subflow" and node.subflow not in flows:
                raise ConfigurationError(
                    f"flow {flow.id!r} references unknown subflow {node.subflow!r}"
                )
    claimed: dict[str, str] = {}
    for flow in flows.values():
        for intent in flow.triggers.intents:
            if intent in claimed:
                raise ConfigurationError(
                    f"intent {intent!r} is claimed by both {claimed[intent]!r} and {flow.id!r}"
                )
            claimed[intent] = flow.id
    return flows


def load_knowledge(path: Path) -> list[KnowledgeDocumentIn]:
    if not path.exists():
        return []
    documents: list[KnowledgeDocumentIn] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                payload = json.loads(stripped)
                documents.append(KnowledgeDocumentIn.model_validate(payload))
            except (json.JSONDecodeError, PydanticValidationError, TypeError) as exc:
                raise ConfigurationError(f"{path}:{line_number}: invalid document: {exc}") from exc
    return documents


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ValidationError(f"data file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path}:{line_number}: invalid JSON ({exc.msg})") from exc
            if not isinstance(row, dict):
                raise ValidationError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows
