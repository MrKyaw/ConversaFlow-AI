"""Conversation state that is safe to serialise and replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from ..dialogue.flow import Flow
from ..models import utcnow


@dataclass
class Frame:
    """One active flow invocation. A stack of frames supports subflows."""

    flow_id: str
    node_id: str
    return_to: str | None = None
    retries: dict[str, int] = field(default_factory=dict)
    confirm_node: str | None = None
    entered_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "flow_id": self.flow_id,
            "node_id": self.node_id,
            "return_to": self.return_to,
            "retries": dict(self.retries),
            "confirm_node": self.confirm_node,
            "entered_at": self.entered_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Frame:
        return cls(
            flow_id=str(payload["flow_id"]),
            node_id=str(payload["node_id"]),
            return_to=payload.get("return_to"),
            retries={str(key): int(value) for key, value in (payload.get("retries") or {}).items()},
            confirm_node=payload.get("confirm_node"),
            entered_at=str(payload.get("entered_at", "")),
        )


@dataclass
class TurnRecord:
    turn_id: str
    role: str
    text: str
    at: str
    intent: str | None = None
    confidence: float = 0.0
    flow: str | None = None
    node: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "role": self.role,
            "text": self.text,
            "at": self.at,
            "intent": self.intent,
            "confidence": self.confidence,
            "flow": self.flow,
            "node": self.node,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TurnRecord:
        return cls(
            turn_id=str(payload["turn_id"]),
            role=str(payload["role"]),
            text=str(payload["text"]),
            at=str(payload["at"]),
            intent=payload.get("intent"),
            confidence=float(payload.get("confidence", 0.0)),
            flow=payload.get("flow"),
            node=payload.get("node"),
        )


@dataclass
class Session:
    id: str
    tenant_id: str
    channel: str
    locale: str
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    frames: list[Frame] = field(default_factory=list)
    slots: dict[str, Any] = field(default_factory=dict)
    slot_entities: dict[str, dict[str, Any]] = field(default_factory=dict)
    fallbacks: int = 0
    handoff: dict[str, Any] | None = None
    resolved: bool = False
    disambiguation: list[dict[str, Any]] = field(default_factory=list)
    barge_ins: int = 0
    last_ssml: bool = False
    tool_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    turns: list[TurnRecord] = field(default_factory=list)
    idempotency: dict[str, dict[str, Any]] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ views

    @property
    def active_frame(self) -> Frame | None:
        return self.frames[-1] if self.frames else None

    @property
    def active_flow_id(self) -> str | None:
        frame = self.active_frame
        return frame.flow_id if frame else None

    @property
    def turn_count(self) -> int:
        return sum(1 for turn in self.turns if turn.role == "user")

    def public_slots(self, flow: Flow | None) -> dict[str, Any]:
        if flow is None:
            return dict(self.slots)
        specs = flow.slot_map
        return {
            name: ("***" if specs.get(name) and specs[name].sensitive else value)
            for name, value in self.slots.items()
            if name in specs or not specs
        }

    # ------------------------------------------------------------ lifecycle

    def touch(self, now: datetime, ttl_seconds: int) -> None:
        self.updated_at = now
        self.expires_at = now + timedelta(seconds=ttl_seconds)

    def is_expired(self, now: datetime) -> bool:
        return now >= self.expires_at

    def add_turn(self, turn: TurnRecord) -> None:
        self.turns.append(turn)
        # Keep the hot state bounded; full transcripts belong in an external sink.
        if len(self.turns) > 200:
            self.turns = self.turns[-200:]

    def remember(self, key: str | None, payload: dict[str, Any]) -> None:
        if not key:
            return
        self.idempotency[key] = payload
        if len(self.idempotency) > 50:
            oldest = next(iter(self.idempotency))
            del self.idempotency[oldest]

    # ---------------------------------------------------------- persistence

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "channel": self.channel,
            "locale": self.locale,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "metadata": self.metadata,
            "frames": [frame.to_dict() for frame in self.frames],
            "slots": self.slots,
            "slot_entities": self.slot_entities,
            "fallbacks": self.fallbacks,
            "handoff": self.handoff,
            "resolved": self.resolved,
            "disambiguation": self.disambiguation,
            "barge_ins": self.barge_ins,
            "last_ssml": self.last_ssml,
            "tool_cache": self.tool_cache,
            "turns": [turn.to_dict() for turn in self.turns],
            "idempotency": self.idempotency,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Session:
        return cls(
            id=str(payload["id"]),
            tenant_id=str(payload["tenant_id"]),
            channel=str(payload.get("channel", "web")),
            locale=str(payload.get("locale", "en-US")),
            created_at=datetime.fromisoformat(payload["created_at"]),
            updated_at=datetime.fromisoformat(payload["updated_at"]),
            expires_at=datetime.fromisoformat(payload["expires_at"]),
            metadata=dict(payload.get("metadata") or {}),
            frames=[Frame.from_dict(item) for item in payload.get("frames") or []],
            slots=dict(payload.get("slots") or {}),
            slot_entities=dict(payload.get("slot_entities") or {}),
            fallbacks=int(payload.get("fallbacks", 0)),
            handoff=payload.get("handoff"),
            resolved=bool(payload.get("resolved", False)),
            disambiguation=list(payload.get("disambiguation") or []),
            barge_ins=int(payload.get("barge_ins", 0)),
            last_ssml=bool(payload.get("last_ssml", False)),
            tool_cache=dict(payload.get("tool_cache") or {}),
            turns=[TurnRecord.from_dict(item) for item in payload.get("turns") or []],
            idempotency=dict(payload.get("idempotency") or {}),
            context=dict(payload.get("context") or {}),
        )


def new_session(
    *,
    session_id: str,
    tenant_id: str,
    channel: str,
    locale: str,
    ttl_seconds: int,
    metadata: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> Session:
    reference = now or utcnow()
    return Session(
        id=session_id,
        tenant_id=tenant_id,
        channel=channel,
        locale=locale,
        created_at=reference,
        updated_at=reference,
        expires_at=reference + timedelta(seconds=ttl_seconds),
        metadata=dict(metadata or {}),
    )
