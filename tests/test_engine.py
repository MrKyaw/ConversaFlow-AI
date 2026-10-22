"""Tests for the conversation engine."""

import asyncio

import pytest

from conversaflow.config import Settings
from conversaflow.errors import NotFoundError
from conversaflow.models import TurnRequest
from conversaflow.runtime.bootstrap import build_runtime


@pytest.fixture
def rt(tmp_path):
    settings = Settings.from_env(
        {"CONVERSAFLOW_STORE": "memory", "CONVERSAFLOW_MODEL_PATH": str(tmp_path / "m.json")}
    )
    return build_runtime(settings)


def test_create_and_get_session(rt):
    async def run():
        s = await rt.engine.create_session("demo")
        assert s.id.startswith("s_")
        view = rt.engine.session_view(s)
        assert view.session_id == s.id
        fetched = await rt.engine.get_session(s.id, "demo")
        assert fetched.id == s.id

    asyncio.run(run())


def test_delete_session(rt):
    async def run():
        s = await rt.engine.create_session("demo")
        assert await rt.engine.delete_session(s.id, "demo") is True
        with pytest.raises(NotFoundError):
            await rt.engine.get_session(s.id, "demo")

    asyncio.run(run())


def test_handle_turn_new_session(rt):
    async def run():
        r = await rt.engine.handle_turn("demo", TurnRequest(text="hello"))
        assert r.session_id
        assert "ConversaFlow" in r.text

    asyncio.run(run())


def test_idempotency(rt):
    async def run():
        r1 = await rt.engine.handle_turn("demo", TurnRequest(text="hello", idempotency_key="k1"))
        r2 = await rt.engine.handle_turn(
            "demo", TurnRequest(text="hello", session_id=r1.session_id, idempotency_key="k1")
        )
        assert r2.replayed is True

    asyncio.run(run())


def test_barge_in(rt):
    async def run():
        r = await rt.engine.handle_turn("demo", TurnRequest(text="hello", channel="voice"))
        ack = await rt.engine.barge_in(r.session_id, "demo")
        assert ack.acknowledged is True

    asyncio.run(run())


def test_health(rt):
    h = rt.engine.health()
    assert h.status == "ok"
    assert h.flows == 8


def test_purge_expired(rt):
    async def run():
        # nothing expired
        assert await rt.engine.purge_expired() == 0

    asyncio.run(run())


def test_session_not_found(rt):
    async def run():
        with pytest.raises(NotFoundError):
            await rt.engine.get_session("nope", "demo")

    asyncio.run(run())


def test_tenant_isolation(rt):
    async def run():
        s = await rt.engine.create_session("demo")
        with pytest.raises(NotFoundError):
            await rt.engine.get_session(s.id, "other")

    asyncio.run(run())


def test_turn_limit(rt):
    from conversaflow.config import Settings

    async def run():
        s = await rt.engine.create_session("demo")
        # rebuild engine settings with max_turns=1
        new_settings = Settings.from_env(
            {
                "CONVERSAFLOW_STORE": "memory",
                "CONVERSAFLOW_MAX_TURNS": "1",
                "CONVERSAFLOW_MODEL_PATH": "var/p.json",
            }
        )
        rt.engine.settings = new_settings
        await rt.engine.handle_turn("demo", TurnRequest(text="hello", session_id=s.id))
        r2 = await rt.engine.handle_turn("demo", TurnRequest(text="hello", session_id=s.id))
        assert r2.handoff.required or "specialist" in r2.text.lower()

    import asyncio

    asyncio.run(run())


def test_cancel_phrase(rt):
    async def run():
        s = await rt.engine.create_session("demo")
        await rt.engine.handle_turn(
            "demo",
            TurnRequest(text="I want a refund for order ORD-1234 because broken", session_id=s.id),
        )
        r = await rt.engine.handle_turn("demo", TurnRequest(text="cancel", session_id=s.id))
        assert "cancel" in r.text.lower() or r.dialogue.flow is None

    asyncio.run(run())


def test_help_phrase(rt):
    async def run():
        r = await rt.engine.handle_turn("demo", TurnRequest(text="help"))
        assert "order status" in r.text.lower() or "refund" in r.text.lower()

    asyncio.run(run())


def test_restart_phrase(rt):
    async def run():
        r = await rt.engine.handle_turn("demo", TurnRequest(text="restart"))
        assert "start over" in r.text.lower() or "help" in r.text.lower()

    asyncio.run(run())
