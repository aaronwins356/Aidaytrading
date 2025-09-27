"""In-memory pub/sub event bus suitable for WebSocket broadcasts."""
from __future__ import annotations

import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, DefaultDict


class BroadcastChannel:
    """Manage subscribers for a single event channel."""

    def __init__(self, *, max_queue_size: int = 32) -> None:
        self._subscribers: set[asyncio.Queue[Any]] = set()
        self._lock = asyncio.Lock()
        self._max_queue_size = max_queue_size

    @asynccontextmanager
    async def subscribe(self) -> AsyncIterator[asyncio.Queue[Any]]:
        queue: asyncio.Queue[Any] = asyncio.Queue(self._max_queue_size)
        async with self._lock:
            self._subscribers.add(queue)
        try:
            yield queue
        finally:
            async with self._lock:
                self._subscribers.discard(queue)

    async def publish(self, message: Any) -> None:
        async with self._lock:
            subscribers = list(self._subscribers)

        for queue in subscribers:
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(message)
                except asyncio.QueueFull:
                    # If we cannot publish even after dropping the oldest item,
                    # skip this subscriber to avoid blocking the publisher.
                    continue


class EventBus:
    """Simple in-memory event bus with named channels."""

    def __init__(self) -> None:
        self._channels: DefaultDict[str, BroadcastChannel] = defaultdict(BroadcastChannel)
        self._lock = asyncio.Lock()

    async def publish(self, channel: str, message: Any) -> None:
        await self._channels[channel].publish(message)

    @asynccontextmanager
    async def subscribe(self, channel: str) -> AsyncIterator[asyncio.Queue[Any]]:
        async with self._lock:
            broadcast = self._channels[channel]
        async with broadcast.subscribe() as queue:
            yield queue


event_bus = EventBus()

__all__ = ["BroadcastChannel", "EventBus", "event_bus"]

