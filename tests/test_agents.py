"""Tests for agents and planner."""

from conversaflow.agents.registry import AgentRouter
from conversaflow.config import Settings
from conversaflow.runtime.bootstrap import build_runtime


def _rt():
    s = Settings.from_env({"CONVERSAFLOW_STORE": "memory", "CONVERSAFLOW_MODEL_PATH": "var/p.json"})
    return build_runtime(s)


def test_router_for_intent():
    r = AgentRouter()
    assert r.for_intent("request_refund") is not None
    assert r.for_intent("nope") is None
    decision = r.route("request_refund", 0.9, sentiment_score=-0.9)
    assert decision is not None and "negative sentiment" in decision.reason
    decision2 = r.route("request_refund", 0.9)
    assert decision2 is not None
    assert r.escalate(decision2).reason.startswith("human handoff")
    assert r.catalogue()


def test_planner_basic():
    rt = _rt()
    plan = rt.planner.plan("request_refund", 0.9, {"order_id": "ORD-1234", "reason": "broken"})
    assert plan.intent == "request_refund"
    assert plan.steps
    plan2 = rt.planner.plan("speak_to_human", 0.9)
    assert plan2.complete is True
    plan3 = rt.planner.plan("ask_question", 0.9)
    assert plan3.complete is False
    plan4 = rt.planner.plan("book_appointment", 0.9, {})
    assert any(s.action == "ask" for s in plan4.steps)
