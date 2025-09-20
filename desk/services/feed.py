"""Feed handler fetching candles from the broker with latency tagging."""

from __future__ import annotations

import time
from typing import Dict, Iterable

from desk.data import normalize_ohlcv


class FeedHandler:
    def __init__(
        self,
        broker,
        timeframe: str,
        lookback: int,
        *,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        circuit_breaker_threshold: int = 5,
        circuit_reset_seconds: float = 30.0,
        fallback_broker=None,
    ):
        self.broker = broker
        self.timeframe = timeframe
        self.lookback = lookback
        self.cache: Dict[str, list[dict[str, float]]] = {}
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.circuit_breaker_threshold = max(1, circuit_breaker_threshold)
        self.circuit_reset_seconds = circuit_reset_seconds
        self.fallback_broker = fallback_broker
        self._failure_counts: Dict[str, int] = {}
        self._circuit_open_until: Dict[str, float] = {}

    def fetch(self, symbol: str) -> list[dict[str, float]]:
        now = time.time()
        circuit_until = self._circuit_open_until.get(symbol)
        if circuit_until and now < circuit_until:
            return self.cache.get(symbol, [])

        attempt = 0
        last_exception: Exception | None = None
        primary_success = False
        while attempt < self.max_retries:
            attempt += 1
            try:
                start = time.time()
                raw = self.broker.fetch_ohlcv(symbol, self.timeframe, limit=self.lookback)
                candles = normalize_ohlcv(raw)
                if candles:
                    latency = time.time() - start
                    candles[-1]["latency"] = latency
                    self.cache[symbol] = candles
                    self._failure_counts.pop(symbol, None)
                    self._circuit_open_until.pop(symbol, None)
                    primary_success = True
                return candles or self.cache.get(symbol, [])
            except Exception as exc:
                last_exception = exc
                if self.backoff_factor > 0 and attempt < self.max_retries:
                    time.sleep(self.backoff_factor * attempt)

        if not primary_success:
            failures = self._failure_counts.get(symbol, 0) + 1
            self._failure_counts[symbol] = failures
            if failures >= self.circuit_breaker_threshold:
                self._circuit_open_until[symbol] = time.time() + self.circuit_reset_seconds

        if self.fallback_broker is not None:
            try:
                raw = self.fallback_broker.fetch_ohlcv(symbol, self.timeframe, limit=self.lookback)
                candles = normalize_ohlcv(raw)
                if candles:
                    self.cache[symbol] = candles
                    if primary_success:
                        self._failure_counts.pop(symbol, None)
                        self._circuit_open_until.pop(symbol, None)
                return candles or self.cache.get(symbol, [])
            except Exception:
                pass

        if last_exception is not None:
            print(f"[FEED] Failed to fetch {symbol} after retries: {last_exception}")
        return self.cache.get(symbol, [])

    def snapshot(self, symbols: Iterable[str]) -> Dict[str, list[dict[str, float]]]:
        snapshot = {}
        for symbol in symbols:
            try:
                candles = self.fetch(symbol)
                snapshot[symbol] = candles
            except Exception as exc:  # pragma: no cover - defensive network guard
                print(f"[FEED] Failed to fetch {symbol}: {exc}")
                if symbol in self.cache:
                    snapshot[symbol] = self.cache[symbol]
        return snapshot

