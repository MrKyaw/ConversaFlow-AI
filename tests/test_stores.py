"""Tests for session stores."""

import asyncio
from datetime import UTC, datetime, timedelta

from conversaflow.runtime.session import new_session
from conversaflow.runtime.stores import InMemorySessionStore, SqliteSessionStore


def _session():
    now = datetime.now(UTC)
    return new_session(
        session_id="s1", tenant_id="t", channel="web", locale="en-US", ttl_seconds=10, now=now
    )


def test_inmemory_store():
    async def run():
        store = InMemorySessionStore()
        s = _session()
        await store.save(s)
        assert await store.get("s1") is not None
        assert await store.count() == 1
        assert await store.delete("s1") is True
        assert await store.delete("s1") is False
        assert await store.count() == 0

    asyncio.run(run())


def test_inmemory_purge():
    async def run():
        store = InMemorySessionStore()
        s = _session()
        # expire it
        s.expires_at = datetime.now(UTC) - timedelta(seconds=1)
        await store.save(s)
        assert await store.purge_expired(datetime.now(UTC)) == 1

    asyncio.run(run())


def test_sqlite_store(tmp_path):
    async def run():
        db = tmp_path / "test.db"
        store = SqliteSessionStore(db)
        s = _session()
        await store.save(s)
        assert await store.get("s1") is not None
        assert await store.count() == 1
        s2 = await store.get("s1")
        assert s2.tenant_id == "t"
        await store.save(s2)
        assert await store.delete("s1") is True
        assert await store.get("s1") is None
        store.close()

    asyncio.run(run())


def test_sqlite_memory():
    async def run():
        store = SqliteSessionStore(":memory:")
        s = _session()
        await store.save(s)
        assert await store.count() == 1
        assert await store.purge_expired(datetime.now(UTC) + timedelta(days=1)) == 1
        store.close()

    asyncio.run(run())
