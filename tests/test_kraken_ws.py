"""Tests for the Kraken WebSocket candle ingestor."""

from __future__ import annotations

from pathlib import Path

from desk.services.feed_updater import CandleStore
from desk.services.kraken_ws import KrakenWebSocketFeed, _kraken_interval_minutes


def test_interval_rounding_matches_supported_values() -> None:
    assert _kraken_interval_minutes("30s") == 1
    assert _kraken_interval_minutes("1m") == 1
    assert _kraken_interval_minutes("2m") == 5
    assert _kraken_interval_minutes("45m") == 60
    assert _kraken_interval_minutes("1d") == 1_440


def test_ingest_ohlc_appends_to_store(tmp_path: Path) -> None:
    db_path = tmp_path / "ohlc.db"
    store = CandleStore(db_path)
    feed = KrakenWebSocketFeed(
        pairs={"XBT/USD": "BTC/USD"},
        timeframe="1m",
        store=store,
        logger=None,
    )

    payload = [
        1_700_000_000.0,
        1_700_000_060.0,
        "100",  # open
        "110",  # high
        "90",  # low
        "105",  # close
        "107",  # vwap (ignored)
        "12.5",  # volume
        3,  # trades
    ]

    feed.ingest_ohlc("XBT/USD", payload)

    candles = store.load("BTC/USD", 5)
    assert candles
    candle = candles[-1]
    assert candle["close"] == 105.0
    assert candle["volume"] == 12.5


def test_ingest_ohlc_ignores_unknown_pairs(tmp_path: Path) -> None:
    store = CandleStore(tmp_path / "ohlc.db")
    feed = KrakenWebSocketFeed(
        pairs={"ETH/USD": "ETH/USD"},
        timeframe="1m",
        store=store,
        logger=None,
    )

    feed.ingest_ohlc("BTC/USD", [0, 0, 0, 0, 0, 0])
    assert store.load("ETH/USD", 1) == []

