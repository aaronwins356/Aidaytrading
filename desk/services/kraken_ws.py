"""Async Kraken WebSocket client that keeps the candle store fresh."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Protocol

try:  # pragma: no cover - optional dependency guard
    import websockets
except ModuleNotFoundError:  # pragma: no cover - exercised in tests
    class _MissingWebSockets:
        def __getattr__(self, name: str) -> Any:
            raise ModuleNotFoundError(
                "websockets is required for the Kraken WebSocket feed but is not installed"
            )

    websockets = _MissingWebSockets()  # type: ignore[assignment]

from desk.services.logger import EventLogger


class CandleSink(Protocol):
    """Protocol describing the subset of the candle store we rely on."""

    def append(self, symbol: str, candles: Iterable[Dict[str, float]]) -> int:
        """Persist the provided candles."""


_SUPPORTED_INTERVALS_MINUTES = [1, 5, 15, 30, 60, 240, 1_440, 10_080, 21_600]


def _parse_timeframe(timeframe: str) -> float:
    """Return the timeframe in seconds, falling back to one minute."""

    units = {"s": 1.0, "m": 60.0, "h": 3_600.0, "d": 86_400.0}
    cleaned = str(timeframe or "1m").strip()
    if not cleaned:
        return 60.0
    try:
        value = float(cleaned[:-1])
        suffix = cleaned[-1].lower()
    except (ValueError, IndexError):
        return 60.0
    return max(1.0, value * units.get(suffix, 60.0))


def _kraken_interval_minutes(timeframe: str) -> int:
    """Map arbitrary timeframe strings to Kraken's discrete buckets."""

    seconds = _parse_timeframe(timeframe)
    minutes = max(1, int(round(seconds / 60.0)))
    for interval in _SUPPORTED_INTERVALS_MINUTES:
        if minutes <= interval:
            return interval
    return _SUPPORTED_INTERVALS_MINUTES[-1]


@dataclass(slots=True)
class _BackoffState:
    delay: float
    base: float
    maximum: float

    def reset(self) -> None:
        self.delay = self.base

    def advance(self) -> float:
        now = self.delay
        self.delay = min(self.delay * 2.0, self.maximum)
        return now


class KrakenWebSocketFeed:
    """Maintain up-to-date OHLCV data via Kraken's public WebSocket API."""

    WS_URL = "wss://ws.kraken.com"

    def __init__(
        self,
        *,
        pairs: Mapping[str, str],
        timeframe: str,
        store: CandleSink,
        logger: Optional[EventLogger] = None,
        reconnect_base: float = 5.0,
        reconnect_max: float = 90.0,
    ) -> None:
        if not pairs:
            raise ValueError("At least one Kraken pair must be provided")
        self._pairs: Dict[str, str] = {str(pair): str(symbol) for pair, symbol in pairs.items()}
        self._store = store
        self._logger = logger
        self._interval = _kraken_interval_minutes(timeframe)
        self._subscription = json.dumps(
            {
                "event": "subscribe",
                "pair": sorted(self._pairs.keys()),
                "subscription": {"name": "ohlc", "interval": self._interval},
            }
        )
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._backoff = _BackoffState(delay=reconnect_base, base=reconnect_base, maximum=reconnect_max)
        self._last_committed: MutableMapping[str, int] = {}

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------
    def start(self) -> None:  # pragma: no cover - threading wrapper
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="KrakenWS", daemon=True)
        self._thread.start()

    def stop(self) -> None:  # pragma: no cover - threading wrapper
        self._stop_event.set()
        loop = self._loop
        if loop and loop.is_running():
            loop.call_soon_threadsafe(lambda: None)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None
        self._loop = None

    # ------------------------------------------------------------------
    # Public helpers for tests and synchronous ingestion
    # ------------------------------------------------------------------
    def ingest_ohlc(self, pair: str, payload: Iterable[Any]) -> None:
        """Update the candle store with the supplied OHLC payload."""

        symbol = self._pairs.get(str(pair))
        if not symbol:
            return
        data = list(payload)
        if len(data) < 6:
            return
        try:
            start = float(data[0])
            end = float(data[1]) if len(data) > 1 else start
            open_price = float(data[2])
            high = float(data[3])
            low = float(data[4])
            close = float(data[5])
            volume = float(data[7]) if len(data) > 7 else 0.0
        except (TypeError, ValueError):
            return

        timestamp_ms = int(start * 1000.0)
        candle = {
            "timestamp": timestamp_ms,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": max(volume, 0.0),
        }
        inserted = self._store.append(symbol, [candle])
        if inserted:
            self._last_committed[str(pair)] = timestamp_ms
            if end > start:
                self._log(
                    "INFO",
                    symbol,
                    "Kraken WebSocket candle closed",
                    detail=f"ts={timestamp_ms}",
                )

    # ------------------------------------------------------------------
    # Internal async machinery
    # ------------------------------------------------------------------
    async def _connect_once(self) -> None:
        self._log("INFO", "ALL", "Connecting to Kraken WebSocket feed")
        async with websockets.connect(self.WS_URL, ping_interval=None) as socket:
            await socket.send(self._subscription)
            self._backoff.reset()
            async for message in socket:
                if self._stop_event.is_set():
                    break
                await self._handle_message(message)

    async def _handle_message(self, message: Any) -> None:
        if isinstance(message, (bytes, bytearray)):
            try:
                payload = json.loads(message.decode("utf-8"))
            except json.JSONDecodeError:
                return
        elif isinstance(message, str):
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                return
        else:
            payload = message

        if isinstance(payload, dict):
            event = payload.get("event")
            if event == "heartbeat":
                return
            if event == "subscriptionStatus" and payload.get("status") == "subscribed":
                pair_list = payload.get("pair") or payload.get("channelID")
                self._log("INFO", "ALL", "Kraken WebSocket subscription active", detail=str(pair_list))
                return
            if event == "error":
                detail = payload.get("errorMessage") or payload
                raise RuntimeError(f"Kraken WebSocket error: {detail}")
            return

        if not isinstance(payload, list) or len(payload) < 2:
            return

        pair = payload[-1]
        data = payload[1]
        if isinstance(pair, str) and isinstance(data, list):
            self.ingest_ohlc(pair, data)

    async def _runner(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self._connect_once()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                self._log("WARNING", "ALL", "Kraken WebSocket disconnected", detail=str(exc))
                delay = self._backoff.advance()
                await asyncio.sleep(delay)
            else:
                await asyncio.sleep(0.1)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._runner())
        finally:
            pending = asyncio.all_tasks(loop=loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
            asyncio.set_event_loop(None)

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------
    def _log(self, level: str, symbol: str, message: str, *, detail: Optional[str] = None) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        meta = f" | {detail}" if detail else ""
        print(f"[{timestamp}][KrakenWS][{level.upper()}] {symbol}: {message}{meta}")
        if self._logger is None:
            return
        try:
            self._logger.log_feed_event(level=level, symbol=symbol, message=message, detail=detail)
        except Exception:
            # Logging must never interrupt market data processing.
            pass


__all__ = ["KrakenWebSocketFeed"]

