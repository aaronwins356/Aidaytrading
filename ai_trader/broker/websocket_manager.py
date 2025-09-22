"""Lightweight Kraken WebSocket market data feed."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from typing import Deque, Dict, List

import websockets

from ..services.logging import get_logger
from ..services.types import MarketSnapshot


class KrakenWebsocketManager:
    """Maintain live price snapshots from Kraken's public WebSocket."""

    def __init__(self, symbols: List[str], history: int = 120) -> None:
        self._symbols = symbols
        self._history = history
        self._url = "wss://ws.kraken.com/"
        self._latest_prices: Dict[str, float] = {}
        self._price_history: Dict[str, Deque[float]] = {
            symbol: deque(maxlen=history) for symbol in symbols
        }
        self._logger = get_logger(__name__)
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

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
            "pair": self._symbols,
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
        pair = data[-1]
        if isinstance(pair, str) and pair.startswith("XBT/"):
            pair = pair.replace("XBT", "BTC")
        price = payload.get("c", [None])[0]
        if price is None:
            return
        price = float(price)
        self._latest_prices[pair] = price
        history = self._price_history.setdefault(pair, deque(maxlen=self._history))
        history.append(price)

    def latest_snapshot(self) -> MarketSnapshot:
        history = {symbol: list(self._price_history.get(symbol, [])) for symbol in self._symbols}
        return MarketSnapshot(prices=dict(self._latest_prices), history=history)
