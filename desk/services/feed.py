"""Feed handler fetching candles from the broker with latency tagging."""

from __future__ import annotations

import time
from typing import Dict, Iterable

from desk.data import normalize_ohlcv


class FeedHandler:
    def __init__(self, broker, timeframe: str, lookback: int):
        self.broker = broker
        self.timeframe = timeframe
        self.lookback = lookback
        self.cache: Dict[str, list[dict[str, float]]] = {}

    def fetch(self, symbol: str) -> list[dict[str, float]]:
        raw = self.broker.fetch_ohlcv(symbol, self.timeframe, limit=self.lookback)
        candles = normalize_ohlcv(raw)
        if candles:
            self.cache[symbol] = candles
        return candles or self.cache.get(symbol, [])

    def snapshot(self, symbols: Iterable[str]) -> Dict[str, list[dict[str, float]]]:
        snapshot = {}
        for symbol in symbols:
            try:
                start = time.time()
                candles = self.fetch(symbol)
                latency = time.time() - start
                if candles:
                    candles[-1]["latency"] = latency
                snapshot[symbol] = candles
            except Exception as exc:  # pragma: no cover - defensive network guard
                print(f"[FEED] Failed to fetch {symbol}: {exc}")
                if symbol in self.cache:
                    snapshot[symbol] = self.cache[symbol]
        return snapshot

