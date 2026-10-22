"""The conversation runtime that composes NLU, policy, flows and tools.

This is the only place that knows about all the ports. Everything else is a
small component with an explicit contract, which is what makes the system
testable without a browser, a telephony provider or a cloud account.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from datetime import datetime
from typing import Any

from .. import __version__
from ..agents.registry import AgentRouter
from ..agents.tools import ToolRegistry
from ..config import Settings
from ..dialogue.flow import Flow
from ..dialogue.policy import CAPABILITIES, DialoguePolicy
from ..dialogue.runner import FlowRunner, RunResult
from ..errors import NotFoundError
from ..knowledge.bm25 import KnowledgeBase
from ..models import (
    BargeInResponse,
    Channel,
    Citation,
    DialogueView,
    Entity,
    Handoff,
    HealthResponse,
    Sentiment,
    SessionView,
    TurnRequest,
    TurnResponse,
    utcnow,
)
from ..nlu.pipeline import NluPipeline, NluResult
from ..observability.logging import log_event, redact
from ..observability.metrics import MetricsRegistry
from ..observability.tracing import new_trace
from ..voice.ssml import compose as compose_ssml
from .session import Session, TurnRecord, new_session
from .stores import SessionStore

LOGGER = logging.getLogger("conversaflow.engine")

CANCEL_PHRASES = frozenset(
    {
        "cancel",
        "cancel that",
        "never mind",
        "nevermind",
        "stop",
        "stop it",
        "quit",
        "exit",
        "forget it",
    }
)
RESTART_PHRASES = frozenset({"restart", "start over", "start again", "reset"})
HELP_PHRASES = frozenset({"help", "menu", "what can you do", "options", "capabilities"})
HANDOFF_PHRASES = frozenset({"human", "agent", "person", "representative", "support agent"})
ENDING_PHRASES = frozenset(
    {
        "no that's all",
        "no thats all",
        "that's all",
        "thats all",
        "that is all",
        "nothing else",
        "nothing more",
        "i'm done",
        "im done",
        "we're done",
        "were done",
        "all done",
        "all good",
        "no thanks that's all",
        "no thank you that's all",
    }
)


class SessionLocks:
    """Serialize turns per session while allowing different sessions to run in parallel."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}

    def get(self, session_id: str) -> asyncio.Lock:
        return self._locks.setdefault(session_id, asyncio.Lock())

    def discard(self, session_id: str) -> None:
        self._locks.pop(session_id, None)


class ConversationEngine:
    def __init__(
        self,
        *,
        settings: Settings,
        nlu: NluPipeline,
        flows: dict[str, Flow],
        runner: FlowRunner,
        policy: DialoguePolicy,
        knowledge: KnowledgeBase,
        tools: ToolRegistry,
        router: AgentRouter,
        store: SessionStore,
        metrics: MetricsRegistry | None = None,
        clock: Any = None,
    ) -> None:
        self.settings = settings
        self.nlu = nlu
        self.flows = flows
        self.runner = runner
        self.policy = policy
        self.knowledge = knowledge
        self.tools = tools
        self.router = router
        self.store = store
        self.metrics = metrics or MetricsRegistry()
        self.clock = clock or utcnow
        self.locks = SessionLocks()
        self._started = time.monotonic()

    # ------------------------------------------------------------- sessions

    async def create_session(
        self,
        tenant_id: str,
        *,
        channel: str = "web",
        locale: str = "en-US",
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        session = new_session(
            session_id=f"s_{secrets.token_hex(8)}",
            tenant_id=tenant_id,
            channel=channel,
            locale=locale,
            ttl_seconds=self.settings.session_ttl_seconds,
            metadata=metadata,
            now=self.clock(),
        )
        await self.store.save(session)
        self.metrics.increment("conversaflow_sessions_created_total", labels={"channel": channel})
        return session

    async def get_session(self, session_id: str, tenant_id: str) -> Session:
        session = await self.store.get(session_id)
        if session is None or session.tenant_id != tenant_id:
            raise NotFoundError("session not found or access denied")
        if session.is_expired(self.clock()):
            await self.store.delete(session_id)
            self.locks.discard(session_id)
            raise NotFoundError("session has expired")
        return session

    async def delete_session(self, session_id: str, tenant_id: str) -> bool:
        await self.get_session(session_id, tenant_id)
        deleted = await self.store.delete(session_id)
        self.locks.discard(session_id)
        return deleted

    async def purge_expired(self) -> int:
        return await self.store.purge_expired(self.clock())

    def session_view(self, session: Session) -> SessionView:
        flow = self.flows.get(session.active_flow_id) if session.active_flow_id else None
        return SessionView(
            session_id=session.id,
            tenant_id=session.tenant_id,
            channel=Channel(session.channel),
            locale=session.locale,
            created_at=session.created_at,
            updated_at=session.updated_at,
            turns=session.turn_count,
            resolved=session.resolved,
            handoff=self._handoff(session),
            dialogue=self._dialogue_view(session, flow),
        )

    # ------------------------------------------------------------------ turns

    async def handle_turn(
        self,
        tenant_id: str,
        request: TurnRequest,
        *,
        trace_id: str | None = None,
    ) -> TurnResponse:
        started = time.perf_counter()
        trace = new_trace() if trace_id is None else None
        active_trace = trace_id or (trace.trace_id if trace else "")
        now = self.clock()
        session: Session | None = None
        if request.session_id is not None:
            session = await self.get_session(request.session_id, tenant_id)
        else:
            session = await self.create_session(
                tenant_id,
                channel=request.channel.value,
                locale=request.locale,
                metadata=request.metadata,
            )

        if request.idempotency_key and request.idempotency_key in session.idempotency:
            return self._replay(session, request.idempotency_key, active_trace, started)

        lock = self.locks.get(session.id)
        async with lock:
            stored = await self.store.get(session.id)
            if stored is not None:
                session = stored
            if request.idempotency_key and request.idempotency_key in session.idempotency:
                return self._replay(session, request.idempotency_key, active_trace, started)
            if session.is_expired(now):
                await self.store.delete(session.id)
                self.locks.discard(session.id)
                raise NotFoundError("session has expired")
            if session.turn_count >= self.settings.max_turns:
                return await self._close_for_turn_limit(session, active_trace, started, now)

            user_turn = TurnRecord(
                turn_id=f"t_{secrets.token_hex(6)}",
                role="user",
                text=request.text,
                at=now.isoformat(),
            )
            session.add_turn(user_turn)
            nlu = self.nlu.analyse(
                request.text,
                None,
                now,
                session.locale,
            )
            user_turn.intent = nlu.intent
            user_turn.confidence = nlu.confidence

            decision, run, text, citations, suggestions = await self._decide(
                session, nlu, now, request.text
            )

            if run is not None:
                user_turn.flow = run.flow
                user_turn.node = run.node
            session.touch(now, self.settings.session_ttl_seconds)
            response = self._build_response(
                session=session,
                nlu=nlu,
                run=run,
                text=text,
                citations=citations,
                suggestions=suggestions,
                trace_id=active_trace,
                started=started,
                now=now,
                channel=session.channel,
            )
            assistant_turn = TurnRecord(
                turn_id=f"t_{secrets.token_hex(6)}",
                role="assistant",
                text=response.text,
                at=now.isoformat(),
                flow=response.dialogue.flow,
                node=response.dialogue.node,
            )
            session.add_turn(assistant_turn)
            if request.idempotency_key:
                session.remember(
                    request.idempotency_key,
                    response.model_dump(mode="json"),
                )
            await self.store.save(session)
            self._record_metrics(session, nlu, run, response, request.channel.value, decision)
            log_event(
                LOGGER,
                logging.INFO,
                "turn_processed",
                trace_id=active_trace,
                session_id=session.id,
                tenant_id=session.tenant_id,
                intent=nlu.intent,
                confidence=nlu.confidence,
                text=redact(request.text),
                response=redact(response.text),
                handoff=response.handoff.required,
                latency_ms=round(response.latency_ms, 2),
            )
            return response

    async def _decide(
        self,
        session: Session,
        nlu: NluResult,
        now: datetime,
        text: str,
    ) -> tuple[str, RunResult | None, str, list[Citation], list[str]]:
        if not session.frames and (session.handoff is not None or session.resolved):
            session.handoff = None
            session.resolved = False
            session.slots = {}
            session.slot_entities = {}
        normalized = nlu.normalized
        if session.frames:
            if normalized in HANDOFF_PHRASES or nlu.intent == "speak_to_human":
                session.frames = []
                session.slots = {}
                session.handoff = self._handoff_payload(
                    "customer_care", "customer_requested_human", nlu
                )
                return (
                    "handoff",
                    None,
                    "Of course. I'm connecting you with a human specialist now.",
                    [],
                    [],
                )
            if normalized in CANCEL_PHRASES:
                session.frames = []
                session.slots = {}
                session.slot_entities = {}
                session.resolved = False
                session.handoff = None
                return (
                    "cancelled",
                    None,
                    "No problem, I have stopped that. What would you like to do?",
                    [],
                    ["Check my order", "Talk to a person"],
                )
            if normalized in RESTART_PHRASES:
                session.frames = []
                session.slots = {}
                session.slot_entities = {}
                session.resolved = False
                return ("restarted", None, f"Okay, let's start over. {CAPABILITIES}", [], [])
            if normalized in HELP_PHRASES:
                return ("help", None, CAPABILITIES, [], self._default_suggestions())
            run = await self.runner.resume(session, nlu, now)
            return ("flow", run, run.text, [], run.quick_replies)

        if session.disambiguation:
            choice = self.policy.resolve_disambiguation(text, session.disambiguation)
            session.disambiguation = []
            if choice is not None:
                run = await self.runner.start(session, str(choice["flow_id"]), nlu, now)
                session.fallbacks = 0
                return ("flow", run, run.text, [], run.quick_replies)

        if normalized in ENDING_PHRASES:
            session.frames = []
            session.resolved = True
            return ("smalltalk", None, "Thanks for chatting. Take care!", [], [])

        smalltalk = self.policy.smalltalk(nlu)
        if smalltalk is not None and smalltalk.kind == "handoff":
            session.frames = []
            session.handoff = smalltalk.handoff
            if nlu.intent == "speak_to_human":
                session.resolved = False
            return (smalltalk.kind, None, smalltalk.text, [], smalltalk.suggestions)

        flow = self.policy.select_flow(nlu)
        if flow is not None:
            choices = self.policy.disambiguation_choices(nlu)
            if len(choices) >= 2:
                session.disambiguation = choices
                labels = " or ".join(choice["label"] for choice in choices[:2])
                session.resolved = False
                return (
                    "disambiguate",
                    None,
                    f"Just to make sure I understood: do you want {labels}?",
                    [],
                    [choice["label"] for choice in choices[:2]],
                )
            run = await self.runner.start(session, flow.id, nlu, now)
            session.fallbacks = 0
            return ("flow", run, run.text, [], run.quick_replies)

        if smalltalk is not None:
            session.resolved = nlu.intent == "goodbye"
            return (smalltalk.kind, None, smalltalk.text, [], smalltalk.suggestions)

        grounded = self.policy.knowledge_answer(session.tenant_id, nlu)
        if grounded is not None:
            session.resolved = True
            return (
                grounded.kind,
                None,
                grounded.text,
                grounded.citations,
                grounded.suggestions,
            )

        session.fallbacks += 1
        if nlu.sentiment.score <= -0.5:
            session.fallbacks = max(session.fallbacks, 3)
        fallback = self.policy.fallback(session.fallbacks, nlu)
        if fallback.handoff is not None:
            session.frames = []
            session.handoff = fallback.handoff
            session.resolved = False
        return (fallback.kind, None, fallback.text, [], fallback.suggestions)

    async def _close_for_turn_limit(
        self,
        session: Session,
        trace_id: str,
        started: float,
        now: datetime,
    ) -> TurnResponse:
        session.frames = []
        session.handoff = self._handoff_payload("customer_care", "max_turns_reached", None)
        session.touch(now, self.settings.session_ttl_seconds)
        response = TurnResponse(
            session_id=session.id,
            turn_id=f"t_{secrets.token_hex(6)}",
            text=(
                "We have covered a lot. To make sure you get the right help, "
                "I am connecting you with a specialist."
            ),
            handoff=self._handoff(session),
            trace_id=trace_id,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        await self.store.save(session)
        self.metrics.increment(
            "conversaflow_handoffs_total", labels={"reason": "max_turns_reached"}
        )
        return response

    # ---------------------------------------------------------------- helpers

    def _build_response(
        self,
        *,
        session: Session,
        nlu: NluResult,
        run: RunResult | None,
        text: str,
        citations: list[Citation],
        suggestions: list[str],
        trace_id: str,
        started: float,
        now: datetime,
        channel: str,
    ) -> TurnResponse:
        flow = self.flows.get(session.active_flow_id or (run.flow if run else "") or "")
        entities = {
            name: Entity.model_validate(payload) for name, payload in session.slot_entities.items()
        }
        missing = [name for name in (run.expects if run else []) if name != "affirm"]
        response = TurnResponse(
            session_id=session.id,
            turn_id=f"t_{secrets.token_hex(6)}",
            text=text or "I'm here to help. What would you like to do?",
            intent=nlu.intent,
            confidence=nlu.confidence,
            entities=entities,
            sentiment=nlu.sentiment
            if isinstance(nlu.sentiment, Sentiment)
            else Sentiment(score=0.0, label="neutral", intensity=0.0),
            dialogue=DialogueView(
                flow=flow.id if flow else (run.flow if run else None),
                node=run.node if run else None,
                slots=session.public_slots(flow),
                collected=sorted(session.slots),
                missing=missing,
            ),
            suggestions=suggestions,
            citations=citations,
            handoff=self._handoff(session),
            resolved=session.resolved,
            trace_id=trace_id,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        response.ssml = (
            compose_ssml(response.text, locale=session.locale) if channel == "voice" else None
        )
        session.last_ssml = response.ssml is not None
        return response

    def _handoff(self, session: Session) -> Handoff:
        if session.handoff:
            return Handoff(
                required=bool(session.handoff.get("required", True)),
                department=session.handoff.get("department"),
                reason=session.handoff.get("reason"),
                priority=session.handoff.get("priority", "normal"),
            )
        return Handoff(required=False)

    def _dialogue_view(self, session: Session, flow: Flow | None) -> DialogueView:
        missing = []
        if flow is not None:
            missing = [
                name
                for name, spec in flow.slot_map.items()
                if spec.required and name not in session.slots
            ]
        return DialogueView(
            flow=flow.id if flow else None,
            node=session.active_frame.node_id if session.active_frame else None,
            slots=session.public_slots(flow),
            collected=sorted(session.slots),
            missing=missing,
        )

    def _handoff_payload(
        self, department: str, reason: str, nlu: NluResult | None
    ) -> dict[str, Any]:
        return {
            "required": True,
            "department": department,
            "reason": reason,
            "priority": _priority(nlu.sentiment.score if nlu else 0.0),
        }

    def _replay(
        self,
        session: Session,
        key: str,
        trace_id: str,
        started: float,
    ) -> TurnResponse:
        response = TurnResponse.model_validate(session.idempotency[key])
        response.replayed = True
        response.trace_id = trace_id
        response.latency_ms = round((time.perf_counter() - started) * 1000, 3)
        self.metrics.increment("conversaflow_idempotent_replays_total")
        return response

    def _record_metrics(
        self,
        session: Session,
        nlu: NluResult,
        run: RunResult | None,
        response: TurnResponse,
        channel: str,
        decision: str,
    ) -> None:
        self.metrics.increment("conversaflow_turns_total", labels={"channel": channel})
        self.metrics.increment("conversaflow_intent_total", labels={"intent": nlu.intent})
        self.metrics.increment("conversaflow_language_total", labels={"language": nlu.language})
        self.metrics.observe("conversaflow_turn_latency_ms", response.latency_ms)
        if response.handoff.required:
            self.metrics.increment(
                "conversaflow_handoffs_total",
                labels={"reason": response.handoff.reason or "unknown"},
            )
        if decision == "clarify":
            self.metrics.increment("conversaflow_fallbacks_total")
        if run is not None:
            for call in run.tool_calls:
                self.metrics.increment(
                    "conversaflow_tool_calls_total",
                    labels={
                        "tool": str(call.get("tool")),
                        "status": "ok" if call.get("ok") else "error",
                    },
                )

    def _default_suggestions(self) -> list[str]:
        return [
            "Check my order",
            "Request a refund",
            "Book an appointment",
            "Talk to a person",
        ]

    # --------------------------------------------------------------- barge-in

    async def barge_in(
        self, session_id: str, tenant_id: str, reason: str = "user_speech"
    ) -> BargeInResponse:
        session = await self.get_session(session_id, tenant_id)
        cancelled = session.last_ssml
        session.barge_ins += 1
        session.last_ssml = False
        session.context["last_barge_in"] = reason
        session.touch(self.clock(), self.settings.session_ttl_seconds)
        await self.store.save(session)
        self.metrics.increment("conversaflow_barge_ins_total", labels={"reason": reason})
        return BargeInResponse(
            acknowledged=True,
            session_id=session.id,
            turn_id=f"t_{secrets.token_hex(6)}",
            cancelled_ssml=cancelled,
        )

    # ----------------------------------------------------------------- health

    def health(self) -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            model_loaded=bool(self.nlu.model.classes),
            flows=len(self.flows),
            documents=self.knowledge.count_documents(),
            uptime_seconds=round(time.monotonic() - self._started, 3),
        )


def _priority(score: float) -> str:
    if score <= -0.6:
        return "urgent"
    if score <= -0.25:
        return "high"
    return "normal"
