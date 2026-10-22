"""Tests for dialogue policy."""

from conversaflow.config import Settings
from conversaflow.runtime.bootstrap import build_runtime


def _runtime():
    settings = Settings.from_env(
        {"CONVERSAFLOW_STORE": "memory", "CONVERSAFLOW_MODEL_PATH": "var/p.json"}
    )
    return build_runtime(settings)


def test_smalltalk_greet():
    rt = _runtime()
    nlu = rt.nlu.analyse("hello")
    decision = rt.policy.smalltalk(nlu)
    assert decision is not None and decision.kind == "smalltalk"


def test_smalltalk_goodbye():
    rt = _runtime()
    nlu = rt.nlu.analyse("goodbye")
    decision = rt.policy.smalltalk(nlu)
    assert decision.kind == "smalltalk" and "care" in decision.text.lower()


def test_smalltalk_deny():
    rt = _runtime()
    nlu = rt.nlu.analyse("no")
    decision = rt.policy.smalltalk(nlu)
    assert decision is not None and decision.kind == "smalltalk"


def test_smalltalk_affirm():
    rt = _runtime()
    nlu = rt.nlu.analyse("yes")
    decision = rt.policy.smalltalk(nlu)
    assert decision is not None


def test_smalltalk_thanks():
    rt = _runtime()
    nlu = rt.nlu.analyse("thanks")
    decision = rt.policy.smalltalk(nlu)
    assert decision is not None


def test_smalltalk_feedback():
    rt = _runtime()
    nlu = rt.nlu.analyse("the service was terrible")
    nlu.intent = "provide_feedback"
    decision = rt.policy.smalltalk(nlu)
    assert decision.kind == "handoff"
    assert decision.handoff["department"] == "customer_care"


def test_smalltalk_human():
    rt = _runtime()
    nlu = rt.nlu.analyse("I want to speak to a human")
    decision = rt.policy.smalltalk(nlu)
    assert decision is not None and decision.kind == "handoff"


def test_select_flow():
    rt = _runtime()
    nlu = rt.nlu.analyse("where is my order ORD-1234")
    flow = rt.policy.select_flow(nlu)
    assert flow is not None and flow.id == "order_status"


def test_knowledge_answer():
    rt = _runtime()
    nlu = rt.nlu.analyse("what are your support hours")
    decision = rt.policy.knowledge_answer("demo", nlu)
    assert decision is not None and decision.kind == "knowledge"


def test_fallback():
    rt = _runtime()
    nlu = rt.nlu.analyse("asdfghjkl")
    decision = rt.policy.fallback(1, nlu)
    assert decision.kind == "clarify"
    decision2 = rt.policy.fallback(2, nlu)
    assert decision2.kind == "clarify"
    decision3 = rt.policy.fallback(5, nlu)
    assert decision3.kind == "handoff"


def test_disambiguation_choices():
    rt = _runtime()
    nlu = rt.nlu.analyse("book or cancel")
    choices = rt.policy.disambiguation_choices(nlu)
    assert isinstance(choices, list)


def test_resolve_disambiguation():
    rt = _runtime()
    choices = [
        {
            "label": "Order status",
            "flow_id": "order_status",
            "intent": "check_order_status",
            "score": 0.5,
        }
    ]
    assert rt.policy.resolve_disambiguation("1", choices) is not None
    assert rt.policy.resolve_disambiguation("order status", choices) is not None
    assert rt.policy.resolve_disambiguation("zzzzz", choices) is None
