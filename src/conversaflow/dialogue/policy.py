"""Grounding policy for open-ended questions.

Answers are extractive: the bot returns a retrieved passage, not a generated
one, and always carries a citation. If retrieval is weak the correct behaviour
is to say so and escalate -- never to improvise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..knowledge.bm25 import KnowledgeBase
from ..models import Citation
from ..nlu.pipeline import NluResult
from .flow import Flow

CAPABILITIES = (
    "I can help with order status, refunds, address changes, cancellations, "
    "appointments, billing questions, technical issues, or connect you with a person."
)


@dataclass
class PolicyDecision:
    kind: str
    text: str = ""
    flow_id: str | None = None
    citations: list[Citation] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    handoff: dict[str, Any] | None = None


class DialoguePolicy:
    def __init__(
        self,
        flows: dict[str, Flow],
        knowledge: KnowledgeBase,
        kb_min_score: float = 1.5,
        intent_threshold: float = 0.30,
    ) -> None:
        self.flows = flows
        self.knowledge = knowledge
        self.kb_min_score = kb_min_score
        self.intent_threshold = intent_threshold
        self._by_intent: dict[str, list[Flow]] = {}
        for flow in flows.values():
            for intent in flow.triggers.intents:
                self._by_intent.setdefault(intent, []).append(flow)

    def select_flow(self, nlu: NluResult) -> Flow | None:
        candidates = [
            flow
            for flow in self._by_intent.get(nlu.intent, [])
            if nlu.confidence >= flow.triggers.min_confidence
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda flow: (flow.triggers.min_confidence, flow.id))

    def disambiguation_choices(self, nlu: NluResult) -> list[dict[str, Any]]:
        if not nlu.ambiguous:
            return []
        choices: list[dict[str, Any]] = []
        seen: set[str] = set()
        for name, score in nlu.alternatives[:3]:
            flow = self._by_intent.get(name, [None])[0]
            if flow is None or flow.id in seen:
                continue
            seen.add(flow.id)
            choices.append(
                {
                    "label": flow.name,
                    "flow_id": flow.id,
                    "intent": name,
                    "score": score,
                }
            )
        return choices[:2]

    def resolve_disambiguation(
        self, text: str, choices: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        from ..nlu.text import normalize

        normalized = normalize(text)
        for index, choice in enumerate(choices, start=1):
            if normalized in {str(index), f"option {index}", f"number {index}"}:
                return choice
        for choice in choices:
            label = normalize(choice["label"])
            if label in normalized or any(
                token in normalized for token in label.split() if len(token) > 4
            ):
                return choice
        return None

    def smalltalk(self, nlu: NluResult) -> PolicyDecision | None:
        text = nlu.normalized
        if nlu.intent == "greet":
            return PolicyDecision(
                kind="smalltalk",
                text=f"Hi! I'm ConversaFlow, your virtual assistant. {CAPABILITIES}",
                suggestions=["Check my order", "Request a refund", "Book an appointment"],
            )
        if nlu.intent == "goodbye":
            return PolicyDecision(
                kind="smalltalk",
                text="Thanks for chatting. Take care!",
            )
        if nlu.intent == "thanks":
            return PolicyDecision(
                kind="smalltalk",
                text="You're welcome! Is there anything else I can help with?",
            )
        if nlu.intent == "affirm" and len(text.split()) <= 4:
            return PolicyDecision(
                kind="smalltalk",
                text="Great. What would you like help with?",
                suggestions=["Check my order", "Book an appointment"],
            )
        if nlu.intent == "deny" and len(text.split()) <= 4:
            return PolicyDecision(
                kind="smalltalk",
                text="No problem. What would you like to do instead?",
                suggestions=["Talk to a person", "Check my order"],
            )
        if nlu.intent == "provide_feedback":
            return PolicyDecision(
                kind="handoff",
                text="Thank you for the feedback. I'm passing it to our customer care team.",
                handoff={
                    "required": True,
                    "department": "customer_care",
                    "reason": "customer_feedback",
                    "priority": "high" if nlu.sentiment.score < -0.2 else "normal",
                },
            )
        if nlu.intent == "speak_to_human":
            return PolicyDecision(
                kind="handoff",
                text="Of course. I'm connecting you with a human specialist now.",
                handoff={
                    "required": True,
                    "department": "customer_care",
                    "reason": "customer_requested_human",
                    "priority": "high" if nlu.sentiment.score < -0.2 else "normal",
                },
            )
        return None

    def knowledge_answer(self, tenant_id: str, nlu: NluResult) -> PolicyDecision | None:
        hits = self.knowledge.search(tenant_id, nlu.text, top_k=2)
        if not hits or hits[0].score < self.kb_min_score:
            return None
        best = hits[0]
        citations = [
            Citation(
                document_id=hit.document_id,
                title=hit.title,
                score=hit.score,
                source=hit.source,
            )
            for hit in hits
        ]
        return PolicyDecision(
            kind="knowledge",
            text=best.text,
            citations=citations,
            suggestions=["Talk to a person", "Ask another question"],
        )

    def fallback(self, attempts: int, nlu: NluResult) -> PolicyDecision:
        if attempts <= 1:
            return PolicyDecision(
                kind="clarify",
                text=(f"I'm not sure I understood that. {CAPABILITIES} Which one do you need?"),
                suggestions=[
                    "Check my order",
                    "Request a refund",
                    "Book an appointment",
                    "Talk to a person",
                ],
            )
        if attempts == 2:
            return PolicyDecision(
                kind="clarify",
                text=(
                    "I want to make sure you get the right help. "
                    "I can connect you with a specialist, or you can rephrase your question."
                ),
                suggestions=["Talk to a person", "Ask another question"],
            )
        return PolicyDecision(
            kind="handoff",
            text="Let me connect you with a specialist who can help.",
            handoff={
                "required": True,
                "department": "customer_care",
                "reason": "repeated_fallback",
                "priority": "high",
            },
        )
