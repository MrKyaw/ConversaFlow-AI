"""Request and response models used only by the HTTP surface.

Keeping them here means the domain models stay transport-agnostic while the
OpenAPI schema still documents every endpoint precisely.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..models import Entity, IntentPrediction, Sentiment


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=4000)
    locale: str = Field(default="en-US", pattern=r"^[a-z]{2}(-[A-Z]{2})?$")


class AnalyzeResponse(BaseModel):
    text: str
    normalized: str
    prediction: IntentPrediction
    entities: dict[str, Entity] = Field(default_factory=dict)
    sentiment: Sentiment
    language: str
    out_of_vocabulary: float
    low_confidence: bool
    ambiguous: bool


class RouteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: str = Field(min_length=1, max_length=80)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)


class RouteResponse(BaseModel):
    agent: str
    department: str
    reason: str
    confidence: float
    tools: list[str] = Field(default_factory=list)


class FlowSummary(BaseModel):
    id: str
    name: str
    description: str
    version: int
    intents: list[str]
    slots: list[str]
    nodes: int
    start: str


class KnowledgeListResponse(BaseModel):
    tenant_id: str
    documents: int
    chunks: int
    document_ids: list[str]


class KnowledgeDeleteResponse(BaseModel):
    deleted: bool
    document_id: str


class AgentsResponse(BaseModel):
    agents: list[dict[str, Any]]
