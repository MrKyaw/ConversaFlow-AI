"""The dialogue interpreter.

A flow is executed as a small virtual machine over typed nodes. The runner
never blocks on IO except for tool calls, never trusts a user-supplied variable,
and always terminates: a step budget turns an accidental cycle into a human
handoff instead of a hung request.

One design choice deserves emphasis -- the runner consumes the *triggering*
utterance. If a customer says "refund ORD-1234 because it broke", routing and
slot filling happen in the same turn, which cuts turns-to-resolution and is one
of the clearest signals of a well-integrated NLU rather than a bolted-on one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..agents.tools import ToolRegistry
from ..errors import FlowDefinitionError
from ..models import Entity, SlotSpec, utcnow
from ..nlu.entities import extract_entities
from ..nlu.pipeline import NluResult
from ..nlu.text import normalize
from ..runtime.session import Frame, Session
from .flow import Flow, Node
from .templates import render, render_value

AFFIRMATIVE = frozenset(
    {
        "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "correct", "confirm",
        "confirmed", "proceed", "please do", "go ahead", "sounds good", "that's right",
        "that is right", "absolutely", "definitely", "of course", "affirmative", "true",
    }
)  # fmt: skip
NEGATIVE = frozenset(
    {
        "no", "nope", "nah", "stop", "cancel", "wrong", "incorrect", "negative",
        "don't", "do not", "not really", "no thanks", "no thank you", "never mind",
        "that's wrong", "that is wrong", "not right", "disagree",
    }
)  # fmt: skip


@dataclass
class RunResult:
    messages: list[str] = field(default_factory=list)
    expects: list[str] = field(default_factory=list)
    quick_replies: list[str] = field(default_factory=list)
    handoff: dict[str, Any] | None = None
    resolved: bool = False
    terminal: bool = False
    flow: str | None = None
    node: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    slots: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return " ".join(message for message in self.messages if message).strip()


class FlowRunner:
    def __init__(self, flows: dict[str, Flow], tools: ToolRegistry, max_steps: int = 20) -> None:
        self.flows = flows
        self.tools = tools
        self.max_steps = max_steps

    async def start(
        self,
        session: Session,
        flow_id: str,
        nlu: NluResult,
        now: datetime | None = None,
    ) -> RunResult:
        flow = self.flows.get(flow_id)
        if flow is None:
            raise FlowDefinitionError(f"unknown flow {flow_id!r}")
        session.frames = [Frame(flow_id=flow.id, node_id=flow.start, entered_at=_iso(now))]
        session.slots = {}
        session.slot_entities = {}
        session.handoff = None
        session.resolved = False
        session.tool_cache = {}
        session.context.pop("result", None)
        session.context.pop("last_error", None)
        return await self.resume(session, nlu, now)

    async def resume(
        self, session: Session, nlu: NluResult, now: datetime | None = None
    ) -> RunResult:
        frame = session.active_frame
        if frame is None:
            raise FlowDefinitionError("no active flow to resume")
        flow = self.flows.get(frame.flow_id)
        if flow is None:
            raise FlowDefinitionError(f"active flow {frame.flow_id!r} is no longer loaded")
        reference = now or utcnow()
        result = RunResult(flow=flow.id)
        steps = 0
        while frame is not None and steps < self.max_steps:
            steps += 1
            flow = self.flows.get(frame.flow_id)
            if flow is None:
                raise FlowDefinitionError(f"active flow {frame.flow_id!r} is no longer loaded")
            result.flow = flow.id
            node = flow.node(frame.node_id)
            if node.clear_slots:
                for slot_name in node.clear_slots:
                    session.slots.pop(slot_name, None)
                    session.slot_entities.pop(slot_name, None)
            result.node = node.id
            context = self._context(session, nlu)
            step = await self._dispatch(session, flow, node, nlu, context, reference)
            if step.messages:
                result.messages.extend(step.messages)
            if step.quick_replies:
                result.quick_replies = step.quick_replies
            if step.tool_call is not None:
                result.tool_calls.append(step.tool_call)
            if step.expects:
                result.expects = step.expects
            if step.terminal:
                result.terminal = True
                result.handoff = step.handoff
                result.resolved = session.resolved
                result.slots = dict(session.slots)
                return result
            if step.goto is not None:
                frame.node_id = step.goto
            if len(session.frames) > 1:
                frame = session.active_frame
            elif session.frames:
                frame = session.frames[-1]
            else:
                frame = None
                result.terminal = True
                result.resolved = session.resolved
                break
            if result.expects:
                break
        else:
            if steps >= self.max_steps:
                session.handoff = {
                    "required": True,
                    "department": flow.handoff_department,
                    "reason": "dialogue_loop_guard",
                    "priority": _priority(nlu.sentiment.score),
                }
                session.frames = []
                result.messages.append(
                    "Sorry, I'm having trouble moving this forward. "
                    "Let me connect you with a specialist who can help."
                )
                result.handoff = session.handoff
                result.terminal = True
        result.slots = dict(session.slots)
        return result

    # --------------------------------------------------------------- dispatch

    async def _dispatch(
        self,
        session: Session,
        flow: Flow,
        node: Node,
        nlu: NluResult,
        context: dict[str, Any],
        now: datetime,
    ) -> _Step:
        if node.type == "say":
            return _Step(
                messages=[render(node.message or "", context)],
                goto=node.next,
            )
        if node.type == "collect":
            return self._collect(session, flow, node, nlu, context, now)
        if node.type == "confirm":
            return self._confirm(session, flow, node, nlu, context)
        if node.type == "branch":
            return self._branch(node, context)
        if node.type == "tool":
            return await self._tool(session, flow, node, nlu, context)
        if node.type == "handoff":
            return self._handoff(session, flow, node, nlu, context)
        if node.type == "end":
            return self._end(session, node, context)
        return self._subflow(session, flow, node, nlu, now)

    def _collect(
        self,
        session: Session,
        flow: Flow,
        node: Node,
        nlu: NluResult,
        context: dict[str, Any],
        now: datetime,
    ) -> _Step:
        specs = {name: flow.slot_map[name] for name in node.slots}
        missing = [
            name
            for name in node.slots
            if specs[name].required and session.slots.get(name) in (None, "")
        ]
        if missing:
            extracted = extract_entities(nlu.text, {name: specs[name] for name in missing}, now)
            for name, entity in extracted.items():
                session.slots[name] = entity.value
                session.slot_entities[name] = entity.model_dump()
            if extracted:
                context = self._context(session, nlu)
            missing = [
                name
                for name in node.slots
                if specs[name].required and session.slots.get(name) in (None, "")
            ]

        if missing:
            foreach = frame_retries(session, flow, node)
            slot = missing[0]
            attempts = foreach.get(slot, 0)
            if attempts > flow.max_retries:
                return _Step(
                    messages=[
                        "I'm still missing the information I need. "
                        "Let me connect you with a specialist."
                    ],
                    terminal=True,
                    handoff={
                        "required": True,
                        "department": flow.handoff_department,
                        "reason": "slot_collection_failed",
                        "priority": _priority(nlu.sentiment.score),
                    },
                )
            foreach[slot] = attempts + 1
            prompts: list[str] = []
            if node.message and attempts == 0:
                prompts.append(render(node.message, context))
            for name in missing:
                spec = specs[name]
                template = spec.prompt or f"Could you provide your {name.replace('_', ' ')}?"
                if foreach.get(name, 0) > 0 or (name == slot and attempts > 0):
                    template = spec.retry_prompt or template
                rendered = render(template, context)
                if rendered not in prompts:
                    prompts.append(rendered)
            quick = _quick_replies(missing, specs)
            return _Step(messages=prompts, expects=missing, quick_replies=quick)

        for name in node.slots:
            setattr_retry_clear(session, flow, node, name)
        message = render(node.message, context) if node.message else None
        return _Step(messages=[message] if message else [], goto=node.next)

    def _confirm(
        self,
        session: Session,
        flow: Flow,
        node: Node,
        nlu: NluResult,
        context: dict[str, Any],
    ) -> _Step:
        frame = session.active_frame
        if frame is None:  # pragma: no cover - guarded by resume
            raise FlowDefinitionError("confirm node without a frame")
        if frame.confirm_node != node.id:
            frame.confirm_node = node.id
            prompt = render(node.prompt or node.message or "Does that look right?", context)
            return _Step(messages=[prompt], expects=["affirm"], quick_replies=["Yes", "No"])
        if _is_affirmative(nlu):
            frame.confirm_node = None
            return _Step(goto=node.next)
        if _is_negative(nlu):
            frame.confirm_node = None
            if node.deny_next is not None:
                return _Step(
                    messages=[render(node.message, context)] if node.message else [],
                    goto=node.deny_next,
                )
            return _Step(
                messages=[render(node.message, context)] if node.message else [],
                terminal=True,
            )
        key = f"confirm:{node.id}"
        attempts = frame.retries.get(key, 0) + 1
        frame.retries[key] = attempts
        if attempts > flow.max_retries:
            return _Step(
                messages=[
                    "I need a clear yes or no before I can continue. "
                    "Let me connect you with a specialist."
                ],
                terminal=True,
                handoff={
                    "required": True,
                    "department": flow.handoff_department,
                    "reason": "confirmation_not_understood",
                    "priority": _priority(nlu.sentiment.score),
                },
            )
        return _Step(
            messages=["Please reply yes or no."],
            expects=["affirm"],
            quick_replies=["Yes", "No"],
        )

    def _branch(self, node: Node, context: dict[str, Any]) -> _Step:
        for transition in node.transitions:
            if transition.when.evaluate(context):
                return _Step(goto=transition.goto)
        if node.default is not None:
            return _Step(goto=node.default)
        raise FlowDefinitionError(
            f"branch node {node.id!r} matched no transition"
        )  # pragma: no cover

    async def _tool(
        self,
        session: Session,
        flow: Flow,
        node: Node,
        nlu: NluResult,
        context: dict[str, Any],
    ) -> _Step:
        if node.tool is None:  # pragma: no cover - flow validation
            raise FlowDefinitionError(f"tool node {node.id!r} has no tool")
        key = f"{flow.id}:{node.id}"
        cached = session.tool_cache.get(key)
        if cached is not None:
            data = cached
            latency = 0.0
            ok = True
            error = None
        else:
            arguments = render_value(node.args, context)
            arguments = {key: value for key, value in arguments.items() if value not in ("", None)}
            outcome = await self.tools.call(node.tool, arguments)
            ok = outcome.ok
            data = outcome.data
            latency = outcome.latency_ms
            error = outcome.error
        if not ok:
            session.context["last_error"] = error or "tool failed"
            if node.on_error is not None:
                context = self._context(session, nlu)
                return _Step(
                    goto=node.on_error,
                    tool_call={"tool": node.tool, "ok": False, "error": error},
                )
            return _Step(
                messages=[
                    "I couldn't complete that action right now. "
                    "Let me connect you with a specialist who can finish it."
                ],
                terminal=True,
                tool_call={"tool": node.tool, "ok": False, "error": error},
                handoff={
                    "required": True,
                    "department": flow.handoff_department,
                    "reason": f"tool_failed:{node.tool}",
                    "priority": _priority(nlu.sentiment.score),
                },
            )
        session.tool_cache[key] = data
        session.context["result"] = data
        if node.assign:
            session.slots[node.assign] = data
        context = self._context(session, nlu)
        return _Step(
            messages=[render(node.message, context)] if node.message else [],
            goto=node.next,
            tool_call={
                "tool": node.tool,
                "ok": True,
                "latency_ms": latency,
                "data": data,
            },
        )

    def _handoff(
        self,
        session: Session,
        flow: Flow,
        node: Node,
        nlu: NluResult,
        context: dict[str, Any],
    ) -> _Step:
        department = node.department or flow.handoff_department
        handoff = {
            "required": True,
            "department": department,
            "reason": render(node.reason, context) if node.reason else "flow_requested_handoff",
            "priority": node.priority or _priority(nlu.sentiment.score),
        }
        session.handoff = handoff
        session.frames = []
        message = render(node.message or "I'm connecting you with a specialist now.", context)
        return _Step(messages=[message], terminal=True, handoff=handoff)

    def _end(self, session: Session, node: Node, context: dict[str, Any]) -> _Step:
        message = render(node.message or "", context)
        session.resolved = node.resolution == "resolved"
        if session.resolved:
            session.handoff = None
        popped = session.frames.pop() if session.frames else None
        if session.frames and popped and popped.return_to:
            session.frames[-1].node_id = popped.return_to
            return _Step(messages=[message], goto=popped.return_to)
        return _Step(
            messages=[message],
            terminal=True,
            handoff=session.handoff,
        )

    def _subflow(
        self,
        session: Session,
        flow: Flow,
        node: Node,
        nlu: NluResult,
        now: datetime,
    ) -> _Step:
        if node.subflow is None:  # pragma: no cover - flow validation
            raise FlowDefinitionError(f"subflow node {node.id!r} has no target")
        target = self.flows.get(node.subflow)
        if target is None:
            raise FlowDefinitionError(f"unknown subflow {node.subflow!r}")
        if any(frame.flow_id == node.subflow for frame in session.frames):
            raise FlowDefinitionError(f"recursive subflow detected: {node.subflow!r}")
        session.frames.append(
            Frame(
                flow_id=target.id,
                node_id=target.start,
                return_to=node.next,
                entered_at=_iso(now),
            )
        )
        return _Step(goto=target.start)

    def _context(self, session: Session, nlu: NluResult) -> dict[str, Any]:
        return {
            "slots": dict(session.slots),
            "sentiment": nlu.sentiment.model_dump(),
            "nlu": {
                "intent": nlu.intent,
                "confidence": nlu.confidence,
                "language": nlu.language,
                "text": nlu.text,
            },
            "intent": nlu.intent,
            "confidence": nlu.confidence,
            "language": nlu.language,
            "session": {"id": session.id, "channel": session.channel},
            "channel": session.channel,
            "fallbacks": session.fallbacks,
            "result": session.context.get("result", {}),
            "last_error": session.context.get("last_error", ""),
        }


@dataclass
class _Step:
    messages: list[str] = field(default_factory=list)
    goto: str | None = None
    expects: list[str] = field(default_factory=list)
    quick_replies: list[str] = field(default_factory=list)
    terminal: bool = False
    handoff: dict[str, Any] | None = None
    tool_call: dict[str, Any] | None = None


def frame_retries(session: Session, flow: Flow, node: Node) -> dict[str, int]:
    frame = session.active_frame
    if frame is None:  # pragma: no cover - guarded by resume
        raise FlowDefinitionError("collect node without a frame")
    return frame.retries


def setattr_retry_clear(session: Session, flow: Flow, node: Node, slot: str) -> None:
    frame = session.active_frame
    if frame is not None:
        frame.retries.pop(slot, None)


def _is_affirmative(nlu: NluResult) -> bool:
    return nlu.intent == "affirm" or normalize(nlu.text) in AFFIRMATIVE


def _is_negative(nlu: NluResult) -> bool:
    normalized = normalize(nlu.text)
    return nlu.intent == "deny" or normalized in NEGATIVE


def _priority(score: float) -> str:
    if score <= -0.6:
        return "urgent"
    if score <= -0.25:
        return "high"
    return "normal"


def _quick_replies(missing: list[str], specs: dict[str, SlotSpec]) -> list[str]:
    for name in missing:
        spec = specs[name]
        if spec.enum:
            return list(spec.enum)
    return []


def _iso(now: datetime | None) -> str:
    return (now or utcnow()).isoformat()


def entity_from_slot(name: str, value: Any, raw: str | None = None) -> Entity:
    return Entity(value=value, raw=raw or str(value), type=name, confidence=0.9)
