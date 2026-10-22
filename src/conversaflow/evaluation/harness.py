"""Offline evaluation: intent accuracy and multi-turn conversation success."""

from __future__ import annotations

import asyncio
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..models import TurnRequest
from ..nlu.pipeline import NluPipeline


@dataclass
class IntentEvaluation:
    total: int
    correct: int
    accuracy: float
    macro_f1: float
    mean_confidence: float
    mean_margin: float
    low_confidence_rate: float
    per_intent: dict[str, dict[str, float]]
    confusion: list[dict[str, Any]]
    errors: list[dict[str, Any]]


def evaluate_intents(pipeline: NluPipeline, path: Path) -> IntentEvaluation:
    rows = _load_jsonl(path)
    if not rows:
        raise ValueError(f"empty intent corpus: {path}")
    total = len(rows)
    correct = 0
    confidences: list[float] = []
    margins: list[float] = []
    low_confidence = 0
    per_intent: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "count": 0}
    )
    confusion: Counter[tuple[str, str]] = Counter()
    errors: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        text = str(row["text"])
        label = str(row["intent"])
        result = pipeline.analyse(text)
        confidences.append(result.confidence)
        margins.append(result.margin)
        if result.low_confidence:
            low_confidence += 1
        per_intent[label]["count"] += 1
        if result.intent == label:
            correct += 1
            per_intent[label]["tp"] += 1
        else:
            per_intent[label]["fn"] += 1
            per_intent[result.intent]["fp"] += 1
            confusion[(label, result.intent)] += 1
            errors.append(
                {
                    "index": index,
                    "text": text,
                    "expected": label,
                    "predicted": result.intent,
                    "confidence": round(result.confidence, 4),
                    "alternatives": result.alternatives[:3],
                }
            )
    f1_scores: list[float] = []
    per_intent_report: dict[str, dict[str, float]] = {}
    for intent, counts in sorted(per_intent.items()):
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_scores.append(f1)
        per_intent_report[intent] = {
            "support": counts["count"],
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    confusion_rows = [
        {"expected": expected, "predicted": predicted, "count": count}
        for (expected, predicted), count in sorted(confusion.items())
    ]
    return IntentEvaluation(
        total=total,
        correct=correct,
        accuracy=round(correct / total, 4) if total else 0.0,
        macro_f1=round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0,
        mean_confidence=round(sum(confidences) / total, 4) if total else 0.0,
        mean_margin=round(sum(margins) / total, 4) if total else 0.0,
        low_confidence_rate=round(low_confidence / total, 4) if total else 0.0,
        per_intent=per_intent_report,
        confusion=confusion_rows,
        errors=errors,
    )


@dataclass
class TurnExpectation:
    intent: str | None = None
    flow: str | None = None
    contains: list[str] = field(default_factory=list)
    handoff: bool | None = None
    resolved: bool | None = None


@dataclass
class ConversationResult:
    conversation_id: str
    description: str
    success: bool
    turns: int
    handoff: bool
    resolved: bool
    active_flow: str | None
    failures: list[str]
    transcript: list[dict[str, Any]]


@dataclass
class ConversationReport:
    total: int
    successful: int
    success_rate: float
    containment_rate: float
    average_turns: float
    results: list[ConversationResult]


def evaluate_conversations(runtime: Any, directory: Path) -> ConversationReport:
    conversations = _load_conversations(directory)
    if not conversations:
        raise ValueError(f"no golden conversations found in {directory}")
    results: list[ConversationResult] = []
    for spec in conversations:
        result = asyncio.run(_run_conversation(runtime, spec))
        results.append(result)
    successful = sum(1 for item in results if item.success)
    contained = sum(1 for item in results if not item.handoff)
    average_turns = sum(item.turns for item in results) / len(results) if results else 0.0
    return ConversationReport(
        total=len(results),
        successful=successful,
        success_rate=round(successful / len(results), 4) if results else 0.0,
        containment_rate=round(contained / len(results), 4) if results else 0.0,
        average_turns=round(average_turns, 2),
        results=results,
    )


async def _run_conversation(runtime: Any, spec: dict[str, Any]) -> ConversationResult:
    tenant_id = str(spec.get("tenant", "demo"))
    session = await runtime.engine.create_session(
        tenant_id, locale=str(spec.get("locale", "en-US"))
    )
    transcript: list[dict[str, Any]] = []
    failures: list[str] = []
    turns = spec.get("turns", [])
    for index, turn in enumerate(turns, start=1):
        expectation = TurnExpectation(**(turn.get("expect") or {}))
        response = await runtime.engine.handle_turn(
            tenant_id,
            TurnRequest(text=str(turn["text"]), session_id=session.id),
        )
        transcript.append({"user": turn["text"], "assistant": response.text})
        if expectation.intent is not None and response.intent != expectation.intent:
            failures.append(
                f"turn {index}: intent expected {expectation.intent!r}, got {response.intent!r}"
            )
        if expectation.flow is not None and response.dialogue.flow != expectation.flow:
            failures.append(
                f"turn {index}: flow expected {expectation.flow!r}, got {response.dialogue.flow!r}"
            )
        for needle in expectation.contains:
            if needle.lower() not in response.text.lower():
                failures.append(f"turn {index}: expected response to contain {needle!r}")
        if expectation.handoff is not None and (response.handoff.required != expectation.handoff):
            failures.append(
                f"turn {index}: handoff expected {expectation.handoff}, "
                f"got {response.handoff.required}"
            )
        if expectation.resolved is not None and response.resolved != expectation.resolved:
            failures.append(
                f"turn {index}: resolved expected {expectation.resolved}, got {response.resolved}"
            )
        session = await runtime.engine.get_session(session.id, tenant_id)
    final = spec.get("final") or {}
    if (
        "handoff" in final
        and session.handoff is not None
        and bool(session.handoff.get("required", False)) != bool(final["handoff"])
    ):
        failures.append(
            f"final: handoff expected {final['handoff']}, "
            f"got {bool(session.handoff.get('required', False))}"
        )
    if "resolved" in final and session.resolved != final["resolved"]:
        failures.append(f"final: resolved expected {final['resolved']}, got {session.resolved}")
    if "flow" in final and session.active_flow_id != final["flow"]:
        failures.append(f"final: flow expected {final['flow']}, got {session.active_flow_id}")
    if "slots" in final:
        for name, value in final["slots"].items():
            if session.slots.get(name) != value:
                failures.append(
                    f"final: slot {name!r} expected {value!r}, got {session.slots.get(name)!r}"
                )
    success = not failures
    handoff_required = bool(session.handoff.get("required", False)) if session.handoff else False
    return ConversationResult(
        conversation_id=str(spec.get("id", spec.get("description", "conversation"))),
        description=str(spec.get("description", "")),
        success=success,
        turns=len(turns),
        handoff=handoff_required,
        resolved=session.resolved,
        active_flow=session.active_flow_id,
        failures=failures,
        transcript=transcript,
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            rows.append(json.loads(stripped))
    return rows


def _load_conversations(directory: Path) -> list[dict[str, Any]]:
    conversations: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        conversations.append(json.loads(path.read_text(encoding="utf-8")))
    return conversations


def to_dict(report: Any) -> dict[str, Any]:
    return asdict(report)
