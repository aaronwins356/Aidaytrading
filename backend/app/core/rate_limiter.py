"""Concurrency-safe sliding window rate limiter."""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, NamedTuple

from app.core.config import get_settings


class RateLimitResult(NamedTuple):
    allowed: bool
    retry_after: float | None


@dataclass(slots=True)
class SlidingWindowRateLimiter:
    limit: int
    window_seconds: float
    block_seconds: float
    _attempts: Dict[str, Deque[float]] = field(init=False, default_factory=dict)
    _blocked_until: Dict[str, float] = field(init=False, default_factory=dict)
    _lock: asyncio.Lock = field(init=False)

    def __post_init__(self) -> None:
        self._lock = asyncio.Lock()

    async def consume(self, key: str) -> RateLimitResult:
        """Attempt to consume one token for the key."""

        async with self._lock:
            now = time.monotonic()
            blocked_until = self._blocked_until.get(key)
            if blocked_until and blocked_until > now:
                return RateLimitResult(False, blocked_until - now)

            queue = self._attempts.setdefault(key, deque())
            cutoff = now - self.window_seconds
            while queue and queue[0] <= cutoff:
                queue.popleft()

            if len(queue) >= self.limit:
                blocked_until = now + self.block_seconds
                self._blocked_until[key] = blocked_until
                queue.clear()
                return RateLimitResult(False, self.block_seconds)

            queue.append(now)
            self._blocked_until.pop(key, None)
            return RateLimitResult(True, None)

    async def reset(self) -> None:
        """Clear tracked state (used in tests)."""

        async with self._lock:
            self._attempts.clear()
            self._blocked_until.clear()


_settings = get_settings()
login_rate_limiter = SlidingWindowRateLimiter(
    limit=_settings.login_rate_limit_attempts,
    window_seconds=float(_settings.login_rate_limit_window_seconds),
    block_seconds=float(_settings.login_rate_limit_block_seconds),
)


__all__ = ["login_rate_limiter", "SlidingWindowRateLimiter", "RateLimitResult"]
