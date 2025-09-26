import asyncio
import asyncio
import json
import logging

import pytest

from ai_trader.broker.websocket_manager import KrakenWebsocketManager


class _StubWebsocket:
    def __init__(self) -> None:
        self.sent_payloads: list[str] = []

    async def send(self, message: str) -> None:
        self.sent_payloads.append(message)


def test_websocket_manager_expands_aliases_and_normalises_ticks(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    manager = KrakenWebsocketManager(["BTC/USD"])
    assert manager.symbols == ["BTC/USD"]

    socket = _StubWebsocket()
    asyncio.run(manager._subscribe(socket))  # type: ignore[attr-defined]
    assert socket.sent_payloads, "subscription payload should be emitted"
    payload = json.loads(socket.sent_payloads[-1])
    assert payload["pair"] == ["XBT/USD"], "Kraken alias should be used when subscribing"
    assert "Subscribed to Kraken tickers: BTC/USD" in caplog.text

    message = json.dumps([42, {"c": ["27123.45", "1.0"], "v": ["1.0", "2.0"]}, "ticker", "XBT/USD"])
    manager._handle_message(message)
    snapshot = manager.latest_snapshot()
    assert snapshot.prices["BTC/USD"] == pytest.approx(27123.45)
