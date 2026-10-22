"""Tests for tool registry and demo backend."""

import asyncio

from conversaflow.agents.tools import ToolRegistry, ToolSpec, build_tool_registry


def test_tool_registry_basic():
    reg = ToolRegistry(default_timeout=1.0)
    spec = ToolSpec(name="echo", description="echo", input_schema={}, timeout_seconds=1.0)

    async def handler(args):
        return {"ok": True, "args": args}

    reg.register(spec, handler)
    assert reg.has("echo")
    assert reg.specs()

    async def run():
        result = await reg.call("echo", {})
        assert result.ok is True
        assert result.data["args"] == {}

    asyncio.run(run())


def test_tool_timeout():
    reg = ToolRegistry(default_timeout=0.01)
    spec = ToolSpec(name="slow", description="slow", input_schema={}, timeout_seconds=0.01)

    async def handler(args):
        import asyncio

        await asyncio.sleep(1)
        return {}

    reg.register(spec, handler)

    async def run():
        result = await reg.call("slow", {})
        assert result.ok is False
        assert "timed out" in (result.error or "")

    asyncio.run(run())


def test_tool_exception():
    reg = ToolRegistry()
    spec = ToolSpec(name="boom", description="boom", input_schema={})

    def handler(args):
        raise ValueError("nope")

    reg.register(spec, handler)

    async def run():
        result = await reg.call("boom", {})
        assert result.ok is False
        assert "ValueError" in (result.error or "")

    asyncio.run(run())


def test_tool_unknown():
    reg = ToolRegistry()

    async def run():
        result = await reg.call("missing", {})
        assert result.ok is False

    asyncio.run(run())


def test_build_registry_demo():
    reg = build_tool_registry()
    assert reg.has("order.lookup")
    assert reg.has("refund.create")

    async def run():
        r = await reg.call("order.lookup", {"order_id": "ORD-1234"})
        assert r.ok and r.data["order_id"] == "ORD-1234"
        r2 = await reg.call("order.lookup", {"order_id": "ORD-9999"})
        assert r2.ok is False
        r3 = await reg.call("refund.create", {"order_id": "ORD-1234", "amount": 10})
        assert r3.ok and r3.data["amount_display"] == "$10.00"

    asyncio.run(run())


def test_tool_validation():
    reg = build_tool_registry()

    async def run():
        r = await reg.call("order.lookup", {"order_id": "BAD"})
        assert r.ok is False

    asyncio.run(run())
