"""Deterministic planning over a flow graph.

The planner answers the operational question "what would this agent do next?"
without touching a customer session or executing a side effect. It is useful for
debugging, for human review of a new flow, and as the capability surface an
orchestrator can inspect before handing work to a specialist.

The plan is bounded. A malformed graph or an over-eager loop produces an
explicit budget-exhausted result rather than an unbounded walk.
"""

from __future__ import annotations

from typing import Any

from ..agents.registry import AgentRouter
from ..agents.tools import ToolRegistry
from ..dialogue.flow import Flow
from ..dialogue.templates import render, render_value
from ..models import PlanResponse, PlanStep


class Planner:
    def __init__(
        self,
        flows: dict[str, Flow],
        router: AgentRouter,
        tools: ToolRegistry,
        max_steps: int = 8,
    ) -> None:
        self.flows = flows
        self.router = router
        self.tools = tools
        self.max_steps = max_steps
        self._by_intent = {
            intent: flow for flow in flows.values() for intent in flow.triggers.intents
        }

    def plan(
        self,
        intent: str,
        confidence: float,
        available_slots: dict[str, Any] | None = None,
        *,
        sentiment_score: float = 0.0,
    ) -> PlanResponse:
        slots = dict(available_slots or {})
        agent = self.router.for_intent(intent)
        agent_name = agent.name if agent else None
        flow = self._by_intent.get(intent)
        if flow is None:
            if intent == "speak_to_human":
                return PlanResponse(
                    intent=intent,
                    confidence=confidence,
                    agent=agent_name,
                    steps=[
                        PlanStep(
                            action="handoff",
                            detail="Transfer to customer care with specialist context.",
                        )
                    ],
                    complete=True,
                    reason="human_handoff",
                )
            return PlanResponse(
                intent=intent,
                confidence=confidence,
                agent=agent_name,
                steps=[
                    PlanStep(
                        action="respond",
                        detail=(
                            "Retrieve a grounded answer from the knowledge base; "
                            "escalate if no passage clears the score threshold."
                        ),
                    )
                ],
                complete=False,
                reason="requires_knowledge_retrieval",
            )

        steps: list[PlanStep] = []
        node_id: str | None = flow.start
        reason = "completed"
        complete = False
        while node_id is not None:
            if len(steps) >= self.max_steps:
                reason = "step_budget_exhausted"
                break
            node = flow.node(node_id)
            context = self._context(slots, intent, confidence, sentiment_score)
            if node.type == "say":
                if node.message:
                    steps.append(PlanStep(action="respond", detail=render(node.message, context)))
                node_id = node.next
            elif node.type == "collect":
                missing = [
                    name
                    for name in node.slots
                    if flow.slot_map[name].required and name not in slots
                ]
                if missing:
                    for name in missing:
                        spec = flow.slot_map[name]
                        steps.append(
                            PlanStep(
                                action="ask",
                                slot=name,
                                detail=render(
                                    spec.prompt or f"Ask for {name.replace('_', ' ')}.", context
                                ),
                            )
                        )
                    reason = f"missing_slots:{','.join(missing)}"
                    break
                node_id = node.next
            elif node.type == "confirm":
                steps.append(
                    PlanStep(
                        action="ask",
                        detail=render(
                            node.prompt or node.message or "Confirm the details.", context
                        ),
                    )
                )
                reason = "awaiting_confirmation"
                break
            elif node.type == "branch":
                node_id = self._branch(node, context)
            elif node.type == "tool":
                if node.tool is None:  # pragma: no cover - flow validation
                    reason = "invalid_tool_node"
                    break
                if not self.tools.has(node.tool):
                    reason = f"tool_not_registered:{node.tool}"
                    break
                steps.append(
                    PlanStep(
                        action="tool",
                        tool=node.tool,
                        args=render_value(node.args, context),
                        detail=f"Call {node.tool}.",
                    )
                )
                node_id = node.next
            elif node.type == "handoff":
                steps.append(
                    PlanStep(
                        action="handoff",
                        detail=node.reason
                        or f"Hand off to {node.department or flow.handoff_department}.",
                    )
                )
                complete = True
                reason = "human_handoff"
                break
            elif node.type == "end":
                if node.message:
                    steps.append(PlanStep(action="respond", detail=render(node.message, context)))
                complete = True
                reason = "resolved"
                break
            elif node.type == "subflow":
                steps.append(
                    PlanStep(
                        action="respond",
                        detail=f"Continue in subflow {node.subflow}.",
                    )
                )
                reason = "subflow_boundary"
                break
        return PlanResponse(
            intent=intent,
            confidence=confidence,
            agent=agent_name,
            steps=steps,
            complete=complete,
            reason=reason,
        )

    def _branch(self, node: Any, context: dict[str, Any]) -> str | None:
        for transition in node.transitions:
            if transition.when.evaluate(context):
                return str(transition.goto)
        return node.default  # type: ignore[no-any-return]

    def _context(
        self,
        slots: dict[str, Any],
        intent: str,
        confidence: float,
        sentiment_score: float,
    ) -> dict[str, Any]:
        return {
            "slots": slots,
            "intent": intent,
            "confidence": confidence,
            "sentiment": {"score": sentiment_score},
            "result": {},
            "channel": "api",
            "fallbacks": 0,
        }
