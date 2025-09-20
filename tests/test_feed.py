from __future__ import annotations

from typing import Any, List

import time

from desk.services.feed import FeedHandler


class FlakyBroker:
    def __init__(self, fail_times: int = 1):
        self.fail_times = fail_times
        self.calls = 0
        ts = time.time() * 1000
        self.payload = [[ts, 100, 101, 99, 100, 10]]

    def fetch_ohlcv(self, symbol, timeframe="1m", limit=50, since=None):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("temporary failure")
        return self.payload


class StaticBroker:
    def __init__(self):
        ts = time.time() * 1000
        self.payload = [[ts, 200, 201, 199, 200, 5]]

    def fetch_ohlcv(self, symbol, timeframe="1m", limit=50, since=None):
        return self.payload


def test_feed_retries_and_caches_success(monkeypatch):
    broker = FlakyBroker(fail_times=1)
    handler = FeedHandler(
        broker,
        timeframe="1m",
        lookback=50,
        max_retries=2,
        backoff_factor=0,
    )
    candles = handler.fetch("BTC/USDT")
    assert candles
    assert candles[-1]["latency"] >= 0
    # second call should succeed without additional failures
    calls_before = broker.calls
    candles_again = handler.fetch("BTC/USDT")
    assert broker.calls == calls_before + 1
    assert candles_again[-1]["close"] == candles[-1]["close"]


def test_feed_uses_fallback_and_circuit_breaker(monkeypatch):
    broker = FlakyBroker(fail_times=10)
    fallback = StaticBroker()
    handler = FeedHandler(
        broker,
        timeframe="1m",
        lookback=50,
        max_retries=1,
        backoff_factor=0,
        circuit_breaker_threshold=1,
        circuit_reset_seconds=30,
        fallback_broker=fallback,
    )
    candles = handler.fetch("BTC/USDT")
    assert candles[0]["close"] == 200
    # circuit breaker active should prevent additional broker calls
    initial_calls = broker.calls
    candles2 = handler.fetch("BTC/USDT")
    assert broker.calls == initial_calls
    assert candles2 == candles
