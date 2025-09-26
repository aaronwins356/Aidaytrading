"""Lightweight Kraken WebSocket market data feed."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from datetime import datetime, timedelta
from typing import Deque, Dict, List

import websockets

from ai_trader.services.logging import get_logger
from ai_trader.services.types import MarketSnapshot

# Kraken retains legacy asset tickers (e.g. ``XBT`` for ``BTC``) across both
# REST and WebSocket endpoints. Centralising these aliases keeps the
# normalisation logic consistent and simplifies future extensions if Kraken
# introduces additional legacy codes.
_NATIVE_TO_DISPLAY: dict[str, str] = {
    "XBT": "BTC",
}
_DISPLAY_TO_NATIVE: dict[str, str] = {display: native for native, display in _NATIVE_TO_DISPLAY.items()}


class KrakenWebsocketManager:
    """Maintain live price snapshots from Kraken's public WebSocket."""

    def __init__(
        self, symbols: List[str], history: int = 120, candle_interval_seconds: int = 60
    ) -> None:
        normalised: list[str] = []
        seen: set[str] = set()
        for candidate in symbols:
            normalised_symbol = self._normalize_symbol(candidate)
            if not normalised_symbol or normalised_symbol in seen:
                continue
            seen.add(normalised_symbol)
            normalised.append(normalised_symbol)
        if not normalised:
            raise ValueError("At least one valid Kraken symbol must be provided")
        self._symbols = normalised
        self._display_to_kraken = {
            symbol: self._map_to_kraken(symbol) for symbol in self._symbols
        }
        self._kraken_to_display = {
            value: key for key, value in self._display_to_kraken.items()
        }
        self._history = history
        self._url = "wss://ws.kraken.com/"
        self._latest_prices: Dict[str, float] = {}
        self._price_history: Dict[str, Deque[float]] = {
            symbol: deque(maxlen=history) for symbol in self._symbols
        }
        self._ohlcv_history: Dict[str, Deque[dict[str, float]]] = {
            symbol: deque(maxlen=history) for symbol in self._symbols
        }
        self._current_bars: Dict[str, dict[str, float]] = {}
        self._bar_interval = timedelta(seconds=max(1, candle_interval_seconds))
        self._last_volume_totals: Dict[str, float] = {}
        self._logger = get_logger(__name__)
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    @property
    def symbols(self) -> list[str]:
        """Return the Kraken symbols currently subscribed to."""

        return list(self._symbols)

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._stop_event.set()
            await self._task
            self._task = None
            self._stop_event.clear()

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(self._url, ping_interval=20) as websocket:
                    await self._subscribe(websocket)
                    async for message in websocket:
                        if self._stop_event.is_set():
                            break
                        self._handle_message(message)
            except Exception as exc:  # noqa: BLE001
                self._logger.warning("WebSocket reconnect due to %s", exc)
                await asyncio.sleep(3)

    async def _subscribe(self, websocket: websockets.WebSocketClientProtocol) -> None:
        payload = {
            "event": "subscribe",
            "pair": [self._display_to_kraken[symbol] for symbol in self._symbols],
            "subscription": {"name": "ticker"},
        }
        await websocket.send(json.dumps(payload))
        self._logger.info("Subscribed to Kraken tickers: %s", ", ".join(self._symbols))

    def _handle_message(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return
        if isinstance(data, dict):
            return  # heartbeat events ignored
        if not isinstance(data, list) or len(data) < 4:
            return
        payload = data[1]
        if not isinstance(payload, dict):
            return
        raw_pair = str(data[-1]).strip().upper()
        pair = self._kraken_to_display.get(raw_pair)
        if pair is None:
            pair = self._normalize_symbol(raw_pair)
            if pair and pair in self._display_to_kraken:
                self._kraken_to_display[raw_pair] = pair
            else:
                return
        if not pair:
            return
        price_raw = payload.get("c", [None])[0]
        if price_raw is None:
            return
        price = float(price_raw)
        self._latest_prices[pair] = price
        history = self._price_history.setdefault(pair, deque(maxlen=self._history))
        history.append(price)
        now = datetime.utcnow()
        volume_fields = payload.get("v", [None, None])
        volume_total = (
            float(volume_fields[1])
            if isinstance(volume_fields, list) and len(volume_fields) > 1
            else 0.0
        )
        previous_total = self._last_volume_totals.get(pair)
        volume_delta = (
            max(0.0, volume_total - previous_total) if previous_total is not None else 0.0
        )
        self._last_volume_totals[pair] = volume_total
        self._update_candle(pair, price, volume_delta, now)

    def latest_snapshot(self) -> MarketSnapshot:
        self._finalize_stale_bars()
        history = {symbol: list(self._price_history.get(symbol, [])) for symbol in self._symbols}
        candles = {symbol: list(self._ohlcv_history.get(symbol, [])) for symbol in self._symbols}
        return MarketSnapshot(prices=dict(self._latest_prices), history=history, candles=candles)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _update_candle(self, symbol: str, price: float, volume: float, timestamp: datetime) -> None:
        bar = self._current_bars.get(symbol)
        if bar is None:
            bar = {
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": volume,
                "start": timestamp,
            }
            self._current_bars[symbol] = bar
            return

        interval_elapsed = (timestamp - bar["start"]) >= self._bar_interval
        if interval_elapsed:
            finalized = {key: float(bar[key]) for key in ("open", "high", "low", "close", "volume")}
            history = self._ohlcv_history.setdefault(symbol, deque(maxlen=self._history))
            history.append(finalized)
            bar = {
                "open": bar["close"],
                "high": price,
                "low": price,
                "close": price,
                "volume": volume,
                "start": timestamp,
            }
            self._current_bars[symbol] = bar
            return

        bar["high"] = max(bar["high"], price)
        bar["low"] = min(bar["low"], price)
        bar["close"] = price
        bar["volume"] = bar.get("volume", 0.0) + volume

    def _finalize_stale_bars(self) -> None:
        now = datetime.utcnow()
        for symbol, bar in list(self._current_bars.items()):
            if not bar:
                continue
            if (now - bar["start"]) < self._bar_interval:
                continue
            finalized = {key: float(bar[key]) for key in ("open", "high", "low", "close", "volume")}
            history = self._ohlcv_history.setdefault(symbol, deque(maxlen=self._history))
            history.append(finalized)
            bar["start"] = now
            bar["open"] = bar["close"]
            bar["high"] = bar["close"]
            bar["low"] = bar["close"]
            bar["volume"] = 0.0

    @staticmethod
    def _normalize_symbol(symbol: object) -> str | None:
        if symbol is None:
            return None
        text = str(symbol).strip().upper()
        if not text or "/" not in text:
            return None
        base, quote = text.split("/", 1)
        base = _NATIVE_TO_DISPLAY.get(base, base)
        quote = _NATIVE_TO_DISPLAY.get(quote, quote)
        return f"{base}/{quote}"

    @staticmethod
    def _map_to_kraken(symbol: str) -> str:
        base, quote = symbol.split("/", 1)
        base = _DISPLAY_TO_NATIVE.get(base, base)
        quote = _DISPLAY_TO_NATIVE.get(quote, quote)
        return f"{base}/{quote}"
