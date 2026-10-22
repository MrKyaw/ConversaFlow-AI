"""Pydantic models that form the public contract of the service."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .errors import ValidationError


class Channel(StrEnum):
    WEB = "web"
    VOICE = "voice"
    WHATSAPP = "whatsapp"
    SLACK = "slack"
    API = "api"


class Role(StrEnum):
    ADMIN = "admin"
    AGENT = "agent"
    VIEWER = "viewer"


class SlotSpec(BaseModel):
    """Declarative description of a value the dialogue must collect."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")
    type: str = "text"
    prompt: str | None = None
    retry_prompt: str | None = None
    invalid_prompt: str | None = None
    required: bool = True
    enum: list[str] = Field(default_factory=list)
    pattern: str | None = None
    min_length: int | None = Field(default=None, ge=0)
    max_length: int | None = Field(default=None, ge=1)
    min_value: float | None = None
    max_value: float | None = None
    confirm: bool = False
    sensitive: bool = False

    @field_validator("type")
    @classmethod
    def _known_type(cls, value: str) -> str:
        known = {
            "text",
            "order_id",
            "appointment_id",
            "email",
            "phone",
            "date",
            "time",
            "money",
            "tracking_number",
            "city",
            "person_name",
            "enum",
            "number",
        }
        if value not in known:
            raise ValueError(f"unknown slot type {value!r}; expected one of {sorted(known)}")
        return value


class Entity(BaseModel):
    """A span of user text resolved to a canonical slot value."""

    value: Any
    raw: str
    type: str
    start: int = 0
    end: int = 0
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class Sentiment(BaseModel):
    score: float = Field(ge=-1.0, le=1.0)
    label: Literal["negative", "neutral", "positive"]
    intensity: float = 0.0


class IntentPrediction(BaseModel):
    name: str
    confidence: float = Field(ge=0.0, le=1.0)
    runner_up: str | None = None
    margin: float = 0.0
    alternatives: list[tuple[str, float]] = Field(default_factory=list)


class TurnRequest(BaseModel):
    """A single user utterance."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    channel: Channel = Channel.WEB
    locale: str = Field(default="en-US", pattern=r"^[a-z]{2}(-[A-Z]{2})?$")
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=128)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must contain at least one non-whitespace character")
        return value.strip()


class DialogueView(BaseModel):
    flow: str | None = None
    node: str | None = None
    slots: dict[str, Any] = Field(default_factory=dict)
    collected: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)


class Handoff(BaseModel):
    required: bool
    department: str | None = None
    reason: str | None = None
    priority: Literal["low", "normal", "high", "urgent"] = "normal"


class Citation(BaseModel):
    document_id: str
    title: str
    score: float
    source: str | None = None


class TurnResponse(BaseModel):
    session_id: str
    turn_id: str
    text: str
    intent: str | None = None
    confidence: float = 0.0
    entities: dict[str, Entity] = Field(default_factory=dict)
    sentiment: Sentiment = Field(
        default_factory=lambda: Sentiment(score=0.0, label="neutral", intensity=0.0)
    )
    dialogue: DialogueView = Field(default_factory=DialogueView)
    suggestions: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    handoff: Handoff = Field(default_factory=lambda: Handoff(required=False))
    resolved: bool = False
    ssml: str | None = None
    trace_id: str = ""
    latency_ms: float = 0.0
    replayed: bool = False


class SessionView(BaseModel):
    session_id: str
    tenant_id: str
    channel: Channel
    locale: str
    created_at: datetime
    updated_at: datetime
    turns: int
    resolved: bool
    handoff: Handoff
    dialogue: DialogueView


class SessionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channel: Channel = Channel.WEB
    locale: str = Field(default="en-US", pattern=r"^[a-z]{2}(-[A-Z]{2})?$")
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionCreateResponse(BaseModel):
    session_id: str
    expires_at: datetime


class KnowledgeDocumentIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-zA-Z0-9_.-]{1,64}$")
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=200_000)
    tags: list[str] = Field(default_factory=list)
    source: str | None = Field(default=None, max_length=2048)


class KnowledgeIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    documents: list[KnowledgeDocumentIn] = Field(min_length=1, max_length=200)


class KnowledgeIngestResponse(BaseModel):
    ingested: int
    chunks: int
    document_ids: list[str]


class BargeInRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["barge_in"] = "barge_in"
    reason: str = Field(default="user_speech", max_length=100)


class BargeInResponse(BaseModel):
    acknowledged: bool
    session_id: str
    turn_id: str
    cancelled_ssml: bool


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    model_loaded: bool
    flows: int
    documents: int
    uptime_seconds: float


class ErrorEnvelope(BaseModel):
    error: dict[str, Any]


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=4000)
    available_slots: dict[str, Any] = Field(default_factory=dict)


class PlanStep(BaseModel):
    action: Literal["ask", "tool", "handoff", "respond"]
    detail: str
    slot: str | None = None
    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)


class PlanResponse(BaseModel):
    intent: str
    confidence: float
    agent: str | None
    steps: list[PlanStep]
    complete: bool
    reason: str


def utcnow() -> datetime:
    return datetime.now(tz=UTC)


def require(condition: bool, message: str) -> None:
    """Small guard used by validators that need a terse failure path."""

    if not condition:
        raise ValidationError(message)
