import asyncio
import json
from collections import deque
from datetime import datetime, timedelta
from typing import Callable

import pytest

from ai_trader.broker.websocket_manager import KrakenWebsocketManager
from ai_trader.services.monitoring import get_monitoring_center


class _DummyConnection:
    def __init__(
        self,
        messages: list[str],
        *,
        drop: bool,
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        self._messages: deque[str] = deque(messages)
        self._drop = drop
        self._on_complete = on_complete
        self.sent: list[str] = []

    async def __aenter__(self) -> "_DummyConnection":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def send(self, message: str) -> None:
        self.sent.append(message)

    def __aiter__(self) -> "_DummyConnection":
        return self

    async def __anext__(self) -> str:
        if self._messages:
            return self._messages.popleft()
        if self._drop:
            self._drop = False
            raise ConnectionError("connection dropped")
        if self._on_complete is not None:
            self._on_complete()
        raise StopAsyncIteration


def _ticker_message(price: float) -> str:
    payload = [
        42,
        {"c": [f"{price:.2f}", "1.0"], "v": ["1.0", "2.0"]},
        "ticker",
        "XBT/USD",
    ]
    return json.dumps(payload)


@pytest.mark.asyncio
async def test_websocket_manager_recovers_after_drop() -> None:
    center = get_monitoring_center()
    center.reset()
    attempts = 0
    stop_trigger: Callable[[], None] = lambda: None

    def _connector(url: str, ping_interval: int = 20) -> _DummyConnection:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return _DummyConnection([_ticker_message(27123.45)], drop=True)
        return _DummyConnection(
            [_ticker_message(27130.10)],
            drop=False,
            on_complete=stop_trigger,
        )

    manager = KrakenWebsocketManager(
        ["BTC/USD"],
        connector=_connector,
        reconnect_base_delay=0.01,
        reconnect_max_delay=0.02,
    )
    stop_trigger = manager._stop_event.set  # type: ignore[attr-defined]

    await manager.start()
    await asyncio.sleep(0.2)
    await manager.stop()

    snapshot = manager.latest_snapshot()
    assert snapshot.prices["BTC/USD"] == pytest.approx(27130.10)
    events = center.recent_events()
    reconnect_events = [event for event in events if event["event_type"] == "websocket_reconnect"]
    assert reconnect_events, "reconnect attempts should be logged"


def test_candle_rollover_handles_malformed_volume() -> None:
    center = get_monitoring_center()
    center.reset()
    manager = KrakenWebsocketManager(["BTC/USD"], candle_interval_seconds=1)
    now = datetime.utcnow()
    manager._update_candle("BTC/USD", 100.0, 1.0, now)
    manager._update_candle("BTC/USD", 101.0, "bad-volume", now + timedelta(seconds=1))
    candles = manager.latest_snapshot().candles["BTC/USD"]
    assert candles, "candle history should be populated"
    first = candles[-1]
    assert first["open"] == pytest.approx(100.0)
    assert first["close"] == pytest.approx(100.0)
