"""Feed handler fetching candles from the broker with latency tagging."""

from __future__ import annotations

import concurrent.futures
import math
import time
from typing import Dict, Iterable

from desk.data import normalize_ohlcv
from desk.services.pretty_logger import pretty_logger


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
        stale_multiplier: float = 2.5,
        max_workers: int | None = None,
        local_store=None,
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
        self._stale_multiplier = max(1.0, stale_multiplier)
        self._timeframe_seconds = self._timeframe_to_seconds(timeframe)
        self._max_workers = max_workers or min(8, (len(getattr(broker, "symbols", [])) or 4))
        self._local_store = local_store

    @staticmethod
    def _timeframe_to_seconds(timeframe: str) -> float:
        units = {
            "s": 1,
            "m": 60,
            "h": 3600,
            "d": 86400,
        }
        try:
            value = float(timeframe[:-1])
            unit = timeframe[-1].lower()
            return value * units.get(unit, 60)
        except Exception:
            return 60.0

    @staticmethod
    def _normalize_timestamp(ts: float) -> float:
        # CCXT timestamps are often in ms.
        if ts > 1e12:
            return ts / 1000.0
        return ts

    def _is_stale(self, candles: list[dict[str, float]]) -> bool:
        if not candles:
            return True
        last_ts = self._normalize_timestamp(float(candles[-1].get("timestamp", 0.0)))
        age = time.time() - last_ts
        expiry = self._timeframe_seconds * self._stale_multiplier
        return age > expiry

    def fetch(self, symbol: str) -> list[dict[str, float]]:
        if self._local_store is not None:
            try:
                local_candles = self._local_store.load(symbol, self.lookback)
            except Exception:
                local_candles = []
            if local_candles:
                # Persist locally cached candles so the trading loop can serve
                # them immediately without touching the REST API again.
                self.cache[symbol] = local_candles
                return local_candles
            # No candles are available yet from the WebSocket feed; fall back to
            # whatever has already been cached in-memory.
            cached = self.cache.get(symbol, [])
            if cached:
                return cached
            return []
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
                since = None
                if self._timeframe_seconds > 0 and self.lookback > 0:
                    window = self.lookback * self._timeframe_seconds
                    since = max(0, int((time.time() - window) * 1000))
                raw = self.broker.fetch_ohlcv(
                    symbol,
                    self.timeframe,
                    limit=self.lookback,
                    since=since,
                )
                candles = normalize_ohlcv(raw)
                if candles:
                    latency = time.time() - start
                    candles[-1]["latency"] = latency
                    if self._is_stale(candles):
                        age = time.time() - self._normalize_timestamp(
                            float(candles[-1].get("timestamp", 0.0))
                        )
                        raise RuntimeError(
                            f"Stale market data for {symbol} (age={age:.2f}s)"
                        )
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
                pretty_logger.rest_fallback_notice()
                since = None
                if self._timeframe_seconds > 0 and self.lookback > 0:
                    window = self.lookback * self._timeframe_seconds
                    since = max(0, int((time.time() - window) * 1000))
                raw = self.fallback_broker.fetch_ohlcv(
                    symbol,
                    self.timeframe,
                    limit=self.lookback,
                    since=since,
                )
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
            pretty_logger.warning(
                f"[Feed] Failed to fetch {symbol} after retries: {last_exception}"
            )
        if self._local_store is not None:
            try:
                cached = self._local_store.load(symbol, self.lookback)
            except Exception:
                cached = []
            if cached:
                self.cache[symbol] = cached
                return cached
        return self.cache.get(symbol, [])

    def snapshot(self, symbols: Iterable[str]) -> Dict[str, list[dict[str, float]]]:
        snapshot: Dict[str, list[dict[str, float]]] = {}
        symbol_list = list(symbols)
        if not symbol_list:
            return snapshot

        workers = min(len(symbol_list), max(1, int(math.ceil(len(symbol_list) / 2))))
        workers = min(workers, self._max_workers)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(self.fetch, symbol): symbol for symbol in symbol_list
            }
            for future in concurrent.futures.as_completed(future_map):
                symbol = future_map[future]
                try:
                    snapshot[symbol] = future.result()
                except Exception as exc:  # pragma: no cover - defensive network guard
                    pretty_logger.warning(f"[Feed] Failed to fetch {symbol}: {exc}")
                    if symbol in self.cache:
                        snapshot[symbol] = self.cache[symbol]
        return snapshot

