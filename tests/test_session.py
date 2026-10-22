"""Tests for session serialization."""

from datetime import UTC, datetime, timedelta

from conversaflow.runtime.session import Frame, Session, TurnRecord, new_session


def test_session_roundtrip():
    now = datetime.now(UTC)
    s = new_session(
        session_id="s", tenant_id="t", channel="web", locale="en-US", ttl_seconds=10, now=now
    )
    s.slots["a"] = 1
    s.frames.append(Frame(flow_id="f", node_id="n"))
    s.turns.append(TurnRecord(turn_id="t1", role="user", text="hi", at=now.isoformat()))
    restored = Session.from_dict(s.to_dict())
    assert restored.id == "s"
    assert restored.slots == {"a": 1}
    assert restored.frames[0].flow_id == "f"
    assert restored.turns[0].text == "hi"


def test_session_expired():
    now = datetime.now(UTC)
    s = new_session(
        session_id="s", tenant_id="t", channel="web", locale="en-US", ttl_seconds=1, now=now
    )
    assert s.is_expired(now + timedelta(seconds=2))


def test_session_public_slots_sensitive():
    # build minimal flow not needed; public_slots with None flow returns all
    s = new_session(session_id="s", tenant_id="t", channel="web", locale="en-US", ttl_seconds=10)
    s.slots = {"password": "secret"}
    assert s.public_slots(None) == {"password": "secret"}


def test_turn_count():
    now = datetime.now(UTC)
    s = new_session(
        session_id="s", tenant_id="t", channel="web", locale="en-US", ttl_seconds=10, now=now
    )
    s.turns.append(TurnRecord(turn_id="t1", role="user", text="hi", at=now.isoformat()))
    s.turns.append(TurnRecord(turn_id="t2", role="assistant", text="yo", at=now.isoformat()))
    assert s.turn_count == 1


def test_remember_cap():
    s = new_session(session_id="s", tenant_id="t", channel="web", locale="en-US", ttl_seconds=10)
    for i in range(60):
        s.remember(f"k{i}", {"x": i})
    assert len(s.idempotency) <= 50
    s.remember(None, {"x": 1})
    # no change
    assert len(s.idempotency) <= 50


def test_add_turn_cap():
    s = new_session(session_id="s", tenant_id="t", channel="web", locale="en-US", ttl_seconds=10)
    for i in range(210):
        s.add_turn(
            TurnRecord(turn_id=f"t{i}", role="user", text="hi", at="2026-01-01T00:00:00+00:00")
        )
    assert len(s.turns) <= 200
