"""Declarative dialogue-flow schema and graph validation.

Flows are data, not code. A flow is a directed graph of typed nodes; validation
happens once at load time, so an operator typo fails fast at start-up instead of
halfway through a customer conversation. ``extra="forbid"`` is deliberate: a
misspelled key such as ``on_erro`` must not be silently ignored.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..errors import FlowDefinitionError
from ..models import SlotSpec

MISSING = object()


class Condition(BaseModel):
    """A small, safe expression evaluated against the dialogue context.

    No ``eval`` and no expression grammar beyond comparisons. Ops are explicit
    so a flow author cannot smuggle arbitrary Python into production data.
    """

    model_config = ConfigDict(extra="forbid")

    var: str = Field(pattern=r"^[a-z_][a-z0-9_.]*$")
    op: Literal[
        "eq", "ne", "gt", "gte", "lt", "lte", "exists", "not_exists", "in", "contains", "matches"
    ] = "exists"
    value: Any = None

    def evaluate(self, context: dict[str, Any]) -> bool:
        current = lookup(context, self.var)
        if self.op == "exists":
            return current is not MISSING and current is not None and current != ""
        if self.op == "not_exists":
            return current is MISSING or current is None or current == ""
        if current is MISSING:
            return False
        try:
            if self.op == "eq":
                return bool(current == self.value)
            if self.op == "ne":
                return bool(current != self.value)
            if self.op == "gt":
                return bool(current > self.value)
            if self.op == "gte":
                return bool(current >= self.value)
            if self.op == "lt":
                return bool(current < self.value)
            if self.op == "lte":
                return bool(current <= self.value)
            if self.op == "in":
                return bool(current in self.value)
            if self.op == "contains":
                return bool(self.value in current)
            return bool(re.search(str(self.value), str(current)))
        except TypeError:
            return False


class Transition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    when: Condition
    goto: str


class Node(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,49}$")
    type: Literal["say", "collect", "confirm", "branch", "tool", "handoff", "end", "subflow"]
    message: str | None = None
    prompt: str | None = None
    next: str | None = None
    slots: list[str] = Field(default_factory=list)
    clear_slots: list[str] = Field(default_factory=list)
    transitions: list[Transition] = Field(default_factory=list)
    default: str | None = None
    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    assign: str | None = None
    on_error: str | None = None
    deny_next: str | None = None
    subflow: str | None = None
    department: str | None = None
    reason: str | None = None
    priority: Literal["low", "normal", "high", "urgent"] | None = None
    resolution: Literal["resolved", "unresolved"] = "resolved"
    quick_replies: list[str] = Field(default_factory=list)


class FlowTriggers(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intents: list[str] = Field(min_length=1)
    min_confidence: float = Field(default=0.35, ge=0.0, le=1.0)


class Flow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,49}$")
    name: str = Field(min_length=1, max_length=100)
    description: str = ""
    version: int = Field(default=1, ge=1)
    triggers: FlowTriggers
    slots: list[SlotSpec] = Field(default_factory=list)
    start: str
    nodes: list[Node] = Field(min_length=1)
    max_retries: int = Field(default=2, ge=0, le=10)
    handoff_department: str = "general"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_graph(self) -> Flow:
        if not self.nodes:
            raise ValueError("flow must define at least one node")
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("node ids must be unique")
        ids = set(node_ids)
        if self.start not in ids:
            raise ValueError(f"start node {self.start!r} is not defined")
        slot_names = [slot.name for slot in self.slots]
        if len(slot_names) != len(set(slot_names)):
            raise ValueError("slot names must be unique")
        slot_map = set(slot_names)

        for slot in self.slots:
            if slot.pattern:
                try:
                    re.compile(slot.pattern)
                except re.error as exc:
                    raise ValueError(f"slot {slot.name!r} has invalid pattern: {exc}") from exc
            if slot.min_length and slot.max_length and slot.min_length > slot.max_length:
                raise ValueError(f"slot {slot.name!r} has min_length > max_length")
            if slot.enum and slot.type not in {"enum", "text"}:
                raise ValueError(f"slot {slot.name!r} defines enum but type is {slot.type!r}")

        for node in self.nodes:
            references = [node.next, node.default, node.on_error, node.deny_next]
            references.extend(transition.goto for transition in node.transitions)
            for reference in references:
                if reference is not None and reference not in ids:
                    raise ValueError(f"node {node.id!r} references unknown node {reference!r}")
            for slot_name in node.slots:
                if slot_name not in slot_map:
                    raise ValueError(f"node {node.id!r} references unknown slot {slot_name!r}")
            for slot_name in node.clear_slots:
                if slot_name not in slot_map:
                    raise ValueError(f"node {node.id!r} clears unknown slot {slot_name!r}")
            if node.type in {"say", "collect", "confirm", "subflow"} and node.next is None:
                raise ValueError(f"{node.type} node {node.id!r} requires 'next'")
            if node.type == "collect" and not node.slots:
                raise ValueError(f"collect node {node.id!r} requires at least one slot")
            if node.type == "confirm" and node.prompt is None and node.message is None:
                raise ValueError(f"confirm node {node.id!r} requires a prompt")
            if node.type == "branch" and not node.transitions and node.default is None:
                raise ValueError(f"branch node {node.id!r} requires transitions or a default")
            if node.type == "tool" and not node.tool:
                raise ValueError(f"tool node {node.id!r} requires 'tool'")
            if node.type == "subflow" and not node.subflow:
                raise ValueError(f"subflow node {node.id!r} requires 'subflow'")
            if node.type == "tool" and node.next is None and node.on_error is None:
                raise ValueError(f"tool node {node.id!r} requires 'next' or 'on_error'")

        reachable = self._reachable(ids)
        if unreachable := ids - reachable:
            raise ValueError(f"unreachable nodes: {', '.join(sorted(unreachable))}")
        if not any(node.type in {"end", "handoff"} for node in self.nodes):
            raise ValueError("flow must contain at least one end or handoff node")
        return self

    def _reachable(self, ids: set[str]) -> set[str]:
        seen = {self.start}
        queue = [self.start]
        node_map = {node.id: node for node in self.nodes}
        while queue:
            node = node_map[queue.pop()]
            targets = [node.next, node.default, node.on_error, node.deny_next]
            targets.extend(transition.goto for transition in node.transitions)
            for target in targets:
                if target in ids and target not in seen:
                    seen.add(target)
                    queue.append(target)
        return seen

    def node(self, node_id: str) -> Node:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise FlowDefinitionError(f"flow {self.id!r} has no node {node_id!r}")

    @property
    def slot_map(self) -> dict[str, SlotSpec]:
        return {slot.name: slot for slot in self.slots}


def lookup(context: dict[str, Any], path: str) -> Any:
    """Resolve a dotted path, returning a sentinel instead of raising."""

    current: Any = context
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return MISSING
        current = current[part]
    return current
