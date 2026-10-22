"""Session persistence: in-memory for tests, SQLite for real deployments.

Both implementations are deliberately small. SQLite gives durable sessions and
survives a process restart without asking the operator to run a database; the
repository interface is the seam where Redis or DynamoDB would plug in.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Protocol

from .session import Session

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_sessions_tenant ON sessions(tenant_id);
"""


class SessionStore(Protocol):
    async def get(self, session_id: str) -> Session | None: ...

    async def save(self, session: Session) -> None: ...

    async def delete(self, session_id: str) -> bool: ...

    async def purge_expired(self, now: datetime) -> int: ...

    async def count(self) -> int: ...


class InMemorySessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    async def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    async def save(self, session: Session) -> None:
        self._sessions[session.id] = session

    async def delete(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    async def purge_expired(self, now: datetime) -> int:
        expired = [key for key, value in self._sessions.items() if value.is_expired(now)]
        for key in expired:
            del self._sessions[key]
        return len(expired)

    async def count(self) -> int:
        return len(self._sessions)


class SqliteSessionStore:
    """Durable store with WAL enabled and a process-local write lock.

    A single connection guarded by a lock is the right trade-off for this
    footprint: SQLite serialises writes anyway, and avoiding a connection pool
    removes a whole class of "database is locked" surprises under load.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.executescript(_SCHEMA)
        self._connection.commit()

    async def get(self, session_id: str) -> Session | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return Session.from_dict(json.loads(row[0])) if row else None

    async def save(self, session: Session) -> None:
        payload = json.dumps(session.to_dict(), separators=(",", ":"))
        with self._lock:
            self._connection.execute(
                "INSERT INTO sessions (id, tenant_id, updated_at, expires_at, payload) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET tenant_id=excluded.tenant_id, "
                "updated_at=excluded.updated_at, expires_at=excluded.expires_at, "
                "payload=excluded.payload",
                (
                    session.id,
                    session.tenant_id,
                    session.updated_at.isoformat(),
                    session.expires_at.isoformat(),
                    payload,
                ),
            )
            self._connection.commit()

    async def delete(self, session_id: str) -> bool:
        with self._lock:
            cursor = self._connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            self._connection.commit()
        return cursor.rowcount > 0

    async def purge_expired(self, now: datetime) -> int:
        with self._lock:
            cursor = self._connection.execute(
                "DELETE FROM sessions WHERE expires_at < ?", (now.isoformat(),)
            )
            self._connection.commit()
        return cursor.rowcount

    async def count(self) -> int:
        with self._lock:
            row = self._connection.execute("SELECT COUNT(*) FROM sessions").fetchone()
        return int(row[0])

    def close(self) -> None:
        with self._lock:
            self._connection.close()
