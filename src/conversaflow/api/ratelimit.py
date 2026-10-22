"""Per-principal fixed-window rate limiting.

A fixed window is intentionally modest: one lock, one dictionary, no external
dependency. The runtime is in-memory, which is the right scope for a single
process; a distributed deployment would put the same interface in front of
Redis without changing callers.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from ..errors import RateLimitError


class RateLimiter:
    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.limit = limit
        self.window_seconds = window_seconds
        self._clock = clock or time.monotonic
        self._windows: dict[str, tuple[float, int]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        """Consume one unit for ``key`` or raise :class:`RateLimitError`."""

        now = self._clock()
        with self._lock:
            window_start, used = self._windows.get(key, (now, 0))
            if now - window_start >= self.window_seconds:
                window_start, used = now, 0
            if used >= self.limit:
                retry_after = max(1, int(self.window_seconds - (now - window_start)))
                raise RateLimitError(f"rate limit exceeded; retry in {retry_after} second(s)")
            self._windows[key] = (window_start, used + 1)

    def remaining(self, key: str) -> int:
        now = self._clock()
        with self._lock:
            window_start, used = self._windows.get(key, (now, 0))
            if now - window_start >= self.window_seconds:
                return self.limit
            return max(0, self.limit - used)

    def reset(self) -> None:
        with self._lock:
            self._windows.clear()
