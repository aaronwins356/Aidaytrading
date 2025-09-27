"""Simple in-memory rate limiter with per-key quotas."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Tuple

import anyio


@dataclass
class _RateLimitEntry:
    count: int
    expires_at: dt.datetime


class RateLimiter:
    """Track attempts per key with TTL-based eviction."""

    def __init__(self, *, limit: int, interval: dt.timedelta) -> None:
        self._limit = limit
        self._interval = interval
        self._entries: dict[str, _RateLimitEntry] = {}
        self._lock = anyio.Lock()

    async def check(self, key: str) -> Tuple[bool, float]:
        """Register an attempt and return whether it is permitted."""

        now = dt.datetime.now(dt.timezone.utc)
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.expires_at <= now:
                self._entries[key] = _RateLimitEntry(count=1, expires_at=now + self._interval)
                return True, 0.0

            if entry.count >= self._limit:
                retry_after = (entry.expires_at - now).total_seconds()
                return False, max(retry_after, 0.0)

            entry.count += 1
            return True, 0.0

    async def reset(self, key: str) -> None:
        """Reset the attempt counter for the given key."""

        async with self._lock:
            self._entries.pop(key, None)


login_rate_limiter = RateLimiter(limit=5, interval=dt.timedelta(minutes=15))

__all__ = ["RateLimiter", "login_rate_limiter"]

