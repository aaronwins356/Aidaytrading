"""Utility rate limiter helpers for asynchronous trading workflows."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Deque


class SlidingWindowRateLimiter:
    """Asynchronous sliding window rate limiter.

    The limiter ensures no more than ``max_calls`` operations are executed within
    ``window_seconds``.  Callers should await :meth:`acquire` before performing a
    rate-limited action.
    """

    def __init__(self, max_calls: int, window_seconds: float) -> None:
        if max_calls <= 0:
            raise ValueError("max_calls must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.max_calls = int(max_calls)
        self.window = float(window_seconds)
        self._timestamps: Deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            await self._wait_for_slot()
            self._timestamps.append(time.monotonic())

    async def _wait_for_slot(self) -> None:
        while len(self._timestamps) >= self.max_calls:
            now = time.monotonic()
            oldest = self._timestamps[0]
            elapsed = now - oldest
            if elapsed >= self.window:
                self._timestamps.popleft()
                continue
            await asyncio.sleep(self.window - elapsed)

    def reset(self) -> None:
        """Clear tracked timestamps to release waiting callers."""

        self._timestamps.clear()
