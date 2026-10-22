"""Specialist agent registry and deterministic supervisor routing.

The platform is deliberately not a free-for-all multi-agent conversation. A
supervisor maps an intent to exactly one specialist, which keeps tool
permissions narrow and makes behaviour auditable. Routing decisions include the
reason so they can be surfaced in traces and the `/v1/agents/route` endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentSpec:
    name: str
    description: str
    intents: tuple[str, ...]
    department: str
    tools: tuple[str, ...] = ()
    escalation_threshold: float = -0.5


@dataclass
class RoutingDecision:
    agent: AgentSpec
    intent: str
    confidence: float
    reason: str
    alternatives: list[tuple[str, float]] = field(default_factory=list)


SPECIALISTS: tuple[AgentSpec, ...] = (
    AgentSpec(
        name="orders",
        description="Order status, delivery changes, cancellations and refunds.",
        intents=(
            "check_order_status",
            "change_address",
            "cancel_order",
            "request_refund",
        ),
        department="order_support",
        tools=("order.lookup", "order.update_address", "order.cancel", "refund.create"),
    ),
    AgentSpec(
        name="scheduling",
        description="Booking, moving and cancelling service appointments.",
        intents=(
            "book_appointment",
            "reschedule_appointment",
            "cancel_appointment",
        ),
        department="scheduling",
        tools=(
            "appointment.book",
            "appointment.reschedule",
            "appointment.cancel",
        ),
    ),
    AgentSpec(
        name="billing",
        description="Invoices, charges, payment methods and subscription changes.",
        intents=("billing_question",),
        department="billing",
    ),
    AgentSpec(
        name="technical",
        description="Product faults, login issues and integration failures.",
        intents=("technical_support",),
        department="technical_support",
        tools=("ticket.create",),
    ),
    AgentSpec(
        name="care",
        description="Complaints, compliments, feedback and human escalation.",
        intents=("provide_feedback", "speak_to_human"),
        department="customer_care",
    ),
)


class AgentRouter:
    def __init__(self, agents: tuple[AgentSpec, ...] = SPECIALISTS) -> None:
        self.agents = agents
        self._by_intent = {intent: agent for agent in agents for intent in agent.intents}

    def for_intent(self, intent: str) -> AgentSpec | None:
        return self._by_intent.get(intent)

    def route(
        self,
        intent: str,
        confidence: float,
        *,
        sentiment_score: float = 0.0,
        alternatives: list[tuple[str, float]] | None = None,
    ) -> RoutingDecision | None:
        agent = self.for_intent(intent)
        if agent is None:
            return None
        if sentiment_score <= agent.escalation_threshold:
            reason = f"negative sentiment ({sentiment_score:+.2f}) escalated within {agent.name}"
        else:
            reason = f"intent {intent!r} is owned by {agent.name}"
        return RoutingDecision(
            agent=agent,
            intent=intent,
            confidence=confidence,
            reason=reason,
            alternatives=alternatives or [],
        )

    def escalate(self, decision: RoutingDecision) -> RoutingDecision:
        """Human handoff keeps the specialist context instead of losing it."""

        return RoutingDecision(
            agent=decision.agent,
            intent=decision.intent,
            confidence=decision.confidence,
            reason="human handoff requested; specialist context preserved",
            alternatives=decision.alternatives,
        )

    def catalogue(self) -> list[dict[str, object]]:
        return [
            {
                "name": agent.name,
                "description": agent.description,
                "department": agent.department,
                "intents": list(agent.intents),
                "tools": list(agent.tools),
            }
            for agent in self.agents
        ]
