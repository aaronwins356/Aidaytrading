from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from desk.services.feed_updater import CandleStore
from desk.services.kraken_ws import KrakenWebSocketClient, OrderStatus, _kraken_interval_minutes


class DummySocket:
    def __init__(self) -> None:
        self.sent: list[Dict[str, Any]] = []

    async def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))


def test_submit_order_tracks_pending_ack(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def runner() -> None:
        store = CandleStore(tmp_path / "ws.db")
        client = KrakenWebSocketClient(
            pairs={"XBT/USD": "BTC/USD"},
            timeframe="1m",
            store=store,
            token_provider=lambda: "token",
        )
        client._private_socket = DummySocket()  # type: ignore[assignment]
        client._private_ready = asyncio.Event()
        client._private_ready.set()

        async def noop_acquire() -> None:
            return None

        monkeypatch.setattr(client._order_limiter, "acquire", noop_acquire)  # type: ignore[arg-type]

        task = asyncio.create_task(
            client._submit_order_async(
                "XBT/USD",
                side="buy",
                order_type="market",
                volume=1.25,
                client_order_id="abc123",
            )
        )
        await asyncio.sleep(0)

        await client._handle_private_message(
            {
                "event": "addOrderStatus",
                "status": "ok",
                "clientOrderId": "abc123",
                "txid": "TX1",
            }
        )
        result = await task
        assert isinstance(result, OrderStatus)
        assert result.txid == "TX1"
        assert client._private_socket.sent[-1]["event"] == "addOrder"  # type: ignore[index]

    asyncio.run(runner())


def test_interval_rounding_matches_supported_values() -> None:
    assert _kraken_interval_minutes("30s") == 1
    assert _kraken_interval_minutes("1m") == 1
    assert _kraken_interval_minutes("2m") == 5
    assert _kraken_interval_minutes("45m") == 60
    assert _kraken_interval_minutes("1d") == 1_440


def test_ingest_ohlc_appends_to_store(tmp_path: Path) -> None:
    db_path = tmp_path / "ohlc.db"
    store = CandleStore(db_path)
    client = KrakenWebSocketClient(
        pairs={"XBT/USD": "BTC/USD"},
        timeframe="1m",
        store=store,
        token_provider=None,
    )

    payload = [
        1_700_000_000.0,
        1_700_000_060.0,
        "100",
        "110",
        "90",
        "105",
        "107",
        "12.5",
        3,
    ]

    client._ingest_ohlc("XBT/USD", payload)

    candles = store.load("BTC/USD", 5)
    assert candles
    candle = candles[-1]
    assert candle["close"] == 105.0
    assert candle["volume"] == 12.5


def test_balance_handler_updates_snapshot(tmp_path: Path) -> None:
    store = CandleStore(tmp_path / "ws2.db")
    client = KrakenWebSocketClient(
        pairs={"XBT/USD": "BTC/USD"},
        timeframe="1m",
        store=store,
        token_provider=lambda: "token",
    )
    client._handle_balance_update({"balances": {"USD": "1500.5"}})
    snapshot = client.latest_balances()
    assert snapshot["USD"] == pytest.approx(1500.5)


def test_public_loop_reconnects_and_uses_rest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store = CandleStore(tmp_path / "ws3.db")
    client = KrakenWebSocketClient(
        pairs={"XBT/USD": "BTC/USD"},
        timeframe="1m",
        store=store,
        token_provider=None,
    )

    class ExplodingContext:
        async def __aenter__(self):
            raise ConnectionResetError("boom")

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

    def fake_open_socket(*_args, **_kwargs):
        return ExplodingContext()

    monkeypatch.setattr(client, "_open_socket", fake_open_socket)

    fallback_calls = {"count": 0}

    async def fake_rest(symbols=None):
        fallback_calls["count"] += 1

    monkeypatch.setattr(client, "_rest_fallback_refresh", fake_rest)

    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)
        client._stop_event.set()

    monkeypatch.setattr("desk.services.kraken_ws.asyncio.sleep", fake_sleep)

    asyncio.run(client._public_loop())

    assert fallback_calls["count"] == 1
    assert sleeps
