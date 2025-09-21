"""Async Kraken WebSocket client providing market data and execution."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import (Any, Awaitable, Callable, Dict, Iterable, List, Mapping,
                    MutableMapping, Optional, Protocol, Sequence)

try:  # pragma: no cover - optional dependency guard
    import websockets
    from websockets import WebSocketClientProtocol
except ModuleNotFoundError:  # pragma: no cover - exercised in tests
    class _MissingWebSockets:
        def __getattr__(self, name: str) -> Any:
            raise ModuleNotFoundError(
                "websockets is required for the Kraken WebSocket feed but is not installed"
            )

    websockets = _MissingWebSockets()  # type: ignore[assignment]
    WebSocketClientProtocol = object  # type: ignore[misc, assignment]

from desk.services.logger import EventLogger
from desk.utils import SlidingWindowRateLimiter


class CandleSink(Protocol):
    """Protocol describing the subset of the candle store we rely on."""

    def append(self, symbol: str, candles: Iterable[Dict[str, float]]) -> int:
        """Persist the provided candles."""


_SUPPORTED_INTERVALS_MINUTES = [1, 5, 15, 30, 60, 240, 1_440, 10_080, 21_600]

_CONSOLE_COLOURS = {
    "INFO": "\033[36m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "TRADE": "\033[32m",
}
_CONSOLE_RESET = "\033[0m"


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


@dataclass
class OrderStatus:
    """Container for the result of an order submission."""

    client_order_id: str
    txid: Optional[str]
    status: str
    message: str = ""
    descr: Optional[str] = None


class KrakenWebSocketClient:
    """High-level interface around Kraken's public and private WebSockets."""

    PUBLIC_URL = "wss://ws.kraken.com"
    PRIVATE_URL = "wss://ws-auth.kraken.com"

    def __init__(
        self,
        *,
        pairs: Mapping[str, str],
        timeframe: str,
        store: CandleSink,
        token_provider: Optional[Callable[[], Awaitable[str]] | Callable[[], str]] = None,
        logger: Optional[EventLogger] = None,
        reconnect_base: float = 5.0,
        reconnect_max: float = 90.0,
        order_rate_limit: int = 60,
        order_rate_period: float = 10.0,
        order_timeout: float = 10.0,
        public_factory: Optional[Callable[..., Awaitable[WebSocketClientProtocol]]] = None,
        private_factory: Optional[Callable[..., Awaitable[WebSocketClientProtocol]]] = None,
    ) -> None:
        if not pairs:
            raise ValueError("At least one Kraken pair must be provided")
        self._pairs: Dict[str, str] = {str(pair): str(symbol) for pair, symbol in pairs.items()}
        self._store = store
        self._logger = logger
        self._interval = _kraken_interval_minutes(timeframe)
        self._token_provider = token_provider
        self._public_factory = public_factory or (lambda url: websockets.connect(url, ping_interval=None))
        self._private_factory = private_factory or (
            lambda url, **kwargs: websockets.connect(url, ping_interval=None, **kwargs)
        )
        self._order_timeout = float(order_timeout)
        self._public_backoff = _BackoffState(delay=reconnect_base, base=reconnect_base, maximum=reconnect_max)
        self._private_backoff = _BackoffState(delay=reconnect_base, base=reconnect_base, maximum=reconnect_max)
        self._order_limiter = SlidingWindowRateLimiter(order_rate_limit, order_rate_period)
        self._public_handlers: Dict[str, List[Callable[[str, Mapping[str, Any]], None]]] = defaultdict(list)
        self._private_handlers: Dict[str, List[Callable[[Mapping[str, Any]], None]]] = defaultdict(list)
        self._balances: Dict[str, float] = {}
        self._open_orders: Dict[str, Dict[str, Any]] = {}
        self._trades: List[Dict[str, Any]] = []
        self._balance_lock = threading.Lock()
        self._order_lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._shutdown: Optional[asyncio.Event] = None
        self._public_socket: Optional[WebSocketClientProtocol] = None
        self._private_socket: Optional[WebSocketClientProtocol] = None
        self._private_ready: Optional[asyncio.Event] = None
        self._pending_orders: Dict[str, asyncio.Future] = {}
        self._pending_cancels: Dict[str, asyncio.Future] = {}
        self._subscription_payload = json.dumps(
            {
                "event": "subscribe",
                "pair": sorted(self._pairs.keys()),
                "subscription": {"name": "ohlc", "interval": self._interval},
            }
        )
        if token_provider is not None:
            self.register_private_handler("openOrders", self._handle_open_orders)
            self.register_private_handler("ownTrades", self._handle_trade_update)
            self.register_private_handler("accountBalances", self._handle_balance_update)

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
            shutdown = self._shutdown
            if shutdown is not None:
                loop.call_soon_threadsafe(shutdown.set)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None
        self._loop = None

    def _run(self) -> None:  # pragma: no cover - threading wrapper
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        shutdown = asyncio.Event()
        self._shutdown = shutdown
        self._private_ready = asyncio.Event()
        try:
            loop.run_until_complete(self._runner(shutdown))
        finally:
            pending = asyncio.all_tasks(loop=loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
            asyncio.set_event_loop(None)

    async def _runner(self, shutdown: asyncio.Event) -> None:
        tasks = [asyncio.create_task(self._public_loop())]
        if self._token_provider is not None:
            tasks.append(asyncio.create_task(self._private_loop()))
        await shutdown.wait()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # Public data handling
    # ------------------------------------------------------------------
    def register_public_handler(
        self, channel: str, handler: Callable[[str, Mapping[str, Any]], None]
    ) -> None:
        self._public_handlers[channel].append(handler)

    async def _public_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                async with await self._public_factory(self.PUBLIC_URL) as socket:
                    self._public_socket = socket
                    await socket.send(self._subscription_payload)
                    self._public_backoff.reset()
                    async for message in socket:
                        if self._stop_event.is_set():
                            break
                        await self._handle_public_message(message)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                self._log("WARNING", "ALL", "Kraken public WebSocket disconnected", detail=str(exc))
                delay = self._public_backoff.advance()
                await asyncio.sleep(delay)
            finally:
                self._public_socket = None
        self._log("INFO", "ALL", "Kraken public WebSocket loop exiting")

    async def _handle_public_message(self, message: Any) -> None:
        payload: Any
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
                detail = payload.get("pair") or payload.get("channelName")
                self._log("INFO", "ALL", "Kraken public subscription active", detail=str(detail))
                return
            if event == "error":
                detail = payload.get("errorMessage") or payload
                self._log("ERROR", "ALL", "Kraken public error", detail=str(detail))
                return
            return

        if not isinstance(payload, Sequence) or len(payload) < 2:
            return

        channel_name = payload[-2] if len(payload) >= 3 else ""
        pair = payload[-1]
        data = payload[1]

        if isinstance(channel_name, str) and channel_name.startswith("ohlc"):
            if isinstance(pair, str) and isinstance(data, Sequence):
                self._ingest_ohlc(pair, data)
                self._dispatch_public("ohlc", pair, self._normalize_ohlc_payload(pair, data))
            return

        if isinstance(channel_name, str):
            symbol = self._pairs.get(str(pair), str(pair))
            mapping = {"pair": symbol, "raw": payload}
            self._dispatch_public(channel_name, str(pair), mapping)

    def _normalize_ohlc_payload(self, pair: str, payload: Sequence[Any]) -> Dict[str, Any]:
        symbol = self._pairs.get(str(pair), str(pair))
        try:
            start = float(payload[0])
            close = float(payload[4]) if len(payload) > 4 else float(payload[5])
        except (TypeError, ValueError, IndexError):  # pragma: no cover - defensive guard
            start = time.time()
            close = 0.0
        return {"timestamp": start * 1000.0, "close": close, "symbol": symbol}

    def _dispatch_public(self, channel: str, pair: str, payload: Mapping[str, Any]) -> None:
        symbol = self._pairs.get(str(pair), str(pair))
        for handler in list(self._public_handlers.get(channel, [])):
            try:
                handler(symbol, payload)
            except Exception as exc:  # pragma: no cover - defensive handler guard
                self._log("ERROR", symbol, "Public handler error", detail=str(exc))

    def _ingest_ohlc(self, pair: str, payload: Iterable[Any]) -> None:
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
        if inserted and end > start:
            self._log("INFO", symbol, "Kraken candle closed", detail=f"ts={timestamp_ms}")

    # ------------------------------------------------------------------
    # Private data handling
    # ------------------------------------------------------------------
    def register_private_handler(
        self, channel: str, handler: Callable[[Mapping[str, Any]], None]
    ) -> None:
        self._private_handlers[channel].append(handler)

    async def _private_loop(self) -> None:
        if self._token_provider is None:
            return
        ready = self._private_ready
        if ready is None:
            ready = asyncio.Event()
            self._private_ready = ready
        while not self._stop_event.is_set():
            try:
                token = await self._acquire_token()
            except Exception as exc:
                self._log("ERROR", "ALL", "Failed to fetch Kraken WS token", detail=str(exc))
                delay = self._private_backoff.advance()
                await asyncio.sleep(delay)
                continue
            try:
                async with await self._private_factory(self.PRIVATE_URL) as socket:
                    self._private_socket = socket
                    await self._subscribe_private(socket, token)
                    ready.set()
                    self._private_backoff.reset()
                    async for message in socket:
                        if self._stop_event.is_set():
                            break
                        await self._handle_private_message(message)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                self._log("WARNING", "ALL", "Kraken private WebSocket disconnected", detail=str(exc))
                ready.clear()
                delay = self._private_backoff.advance()
                await asyncio.sleep(delay)
            finally:
                self._private_socket = None
        self._log("INFO", "ALL", "Kraken private WebSocket loop exiting")

    async def _acquire_token(self) -> str:
        if self._token_provider is None:
            raise RuntimeError("Private token provider not configured")
        result = self._token_provider()
        if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
            return await result  # type: ignore[return-value]
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, result)

    async def _subscribe_private(self, socket: WebSocketClientProtocol, token: str) -> None:
        subscriptions = [
            {"name": "openOrders", "token": token},
            {"name": "ownTrades", "token": token},
            {"name": "accountBalances", "token": token},
        ]
        for subscription in subscriptions:
            payload = json.dumps({"event": "subscribe", "subscription": subscription})
            await socket.send(payload)

    async def _handle_private_message(self, message: Any) -> None:
        payload: Any
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
            if event == "subscriptionStatus":
                detail = payload.get("subscription", {}).get("name")
                status = payload.get("status")
                self._log("INFO", "ALL", f"Private subscription {status}", detail=str(detail))
                if status != "subscribed":
                    self._log("WARNING", "ALL", "Private subscription issue", detail=str(payload))
                return
            if event == "addOrderStatus":
                await self._resolve_order(payload)
                return
            if event == "cancelOrderStatus":
                await self._resolve_cancel(payload)
                return
            if event == "error":
                self._log("ERROR", "ALL", "Kraken private error", detail=str(payload))
                return
            return

        if isinstance(payload, Sequence) and len(payload) >= 2:
            channel = payload[-2] if len(payload) >= 3 else ""
            data = payload[1]
            if isinstance(channel, str):
                handlers = list(self._private_handlers.get(channel, []))
                event_payload = data if isinstance(data, Mapping) else {"data": data}
                for handler in handlers:
                    try:
                        handler(event_payload)
                    except Exception as exc:  # pragma: no cover - defensive guard
                        self._log("ERROR", "ALL", "Private handler error", detail=str(exc))

    async def _resolve_order(self, payload: Mapping[str, Any]) -> None:
        client_order_id = str(payload.get("clientOrderId") or payload.get("reqid") or "")
        status = str(payload.get("status") or "")
        message = str(payload.get("errorMessage") or payload.get("descr") or "")
        txid = payload.get("txid")
        descr = payload.get("descr")
        future = self._pending_orders.pop(client_order_id, None)
        if future is not None and not future.done():
            future.set_result(
                OrderStatus(
                    client_order_id=client_order_id,
                    txid=str(txid) if txid else None,
                    status=status,
                    message=message,
                    descr=str(descr) if descr else None,
                )
            )
        if status != "ok":
            self._log("WARNING", "ALL", "Order rejected", detail=str(payload))

    async def _resolve_cancel(self, payload: Mapping[str, Any]) -> None:
        client_order_id = str(payload.get("clientOrderId") or payload.get("reqid") or "")
        future = self._pending_cancels.pop(client_order_id, None)
        if future is not None and not future.done():
            future.set_result(payload)

    def _handle_open_orders(self, payload: Mapping[str, Any]) -> None:
        orders = payload.get("open") or payload.get("orders") or {}
        if isinstance(orders, Mapping):
            with self._order_lock:
                for txid, order in orders.items():
                    self._open_orders[str(txid)] = dict(order)

    def _handle_trade_update(self, payload: Mapping[str, Any]) -> None:
        trades = payload.get("trades") or payload.get("data") or {}
        timestamp = time.time()
        if isinstance(trades, Mapping):
            for txid, info in trades.items():
                record = dict(info)
                record["txid"] = str(txid)
                record.setdefault("timestamp", timestamp)
                self._trades.append(record)
                self._log("TRADE", record.get("pair", ""), "Execution update", detail=str(record))

    def _handle_balance_update(self, payload: Mapping[str, Any]) -> None:
        balances = payload.get("balances") or payload.get("account") or payload
        if isinstance(balances, Mapping):
            with self._balance_lock:
                for asset, value in balances.items():
                    try:
                        self._balances[str(asset)] = float(value)
                    except (TypeError, ValueError):
                        continue

    # ------------------------------------------------------------------
    # Order helpers
    # ------------------------------------------------------------------
    async def _submit_order_async(
        self,
        pair: str,
        *,
        side: str,
        order_type: str,
        volume: float,
        price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        validate: bool = False,
    ) -> OrderStatus:
        if self._token_provider is None:
            raise RuntimeError("Private trading is disabled")
        if self._private_ready is None:
            raise RuntimeError("KrakenWebSocketClient not started")
        await self._private_ready.wait()
        client_order_id = client_order_id or uuid.uuid4().hex
        payload: Dict[str, Any] = {
            "event": "addOrder",
            "ordertype": order_type,
            "type": side,
            "volume": str(volume),
            "pair": pair,
            "clientOrderId": client_order_id,
        }
        if price is not None:
            payload["price"] = str(price)
        if validate:
            payload["validate"] = True

        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending_orders[client_order_id] = future
        await self._order_limiter.acquire()
        socket = self._private_socket
        if socket is None:
            raise RuntimeError("Private WebSocket is not available")
        await socket.send(json.dumps(payload))
        try:
            result: OrderStatus = await asyncio.wait_for(future, timeout=self._order_timeout)
        except asyncio.TimeoutError as exc:
            if client_order_id in self._pending_orders:
                self._pending_orders.pop(client_order_id, None)
            raise TimeoutError("Timed out waiting for order acknowledgement") from exc
        return result

    def submit_order(
        self,
        pair: str,
        *,
        side: str,
        order_type: str,
        volume: float,
        price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        validate: bool = False,
    ) -> OrderStatus:
        loop = self._loop
        if loop is None:
            raise RuntimeError("KrakenWebSocketClient not running")
        coro = self._submit_order_async(
            pair,
            side=side,
            order_type=order_type,
            volume=volume,
            price=price,
            client_order_id=client_order_id,
            validate=validate,
        )
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=self._order_timeout + 2.0)

    async def _cancel_order_async(
        self,
        *,
        client_order_id: Optional[str] = None,
        txid: Optional[str] = None,
    ) -> Mapping[str, Any]:
        if not client_order_id and not txid:
            raise ValueError("client_order_id or txid must be provided")
        if self._token_provider is None:
            raise RuntimeError("Private trading is disabled")
        if self._private_ready is None:
            raise RuntimeError("KrakenWebSocketClient not started")
        await self._private_ready.wait()
        request_id = client_order_id or uuid.uuid4().hex
        payload: Dict[str, Any] = {"event": "cancelOrder", "clientOrderId": request_id}
        if txid:
            payload["txid"] = txid
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending_cancels[request_id] = future
        await self._order_limiter.acquire()
        socket = self._private_socket
        if socket is None:
            raise RuntimeError("Private WebSocket is not available")
        await socket.send(json.dumps(payload))
        try:
            result = await asyncio.wait_for(future, timeout=self._order_timeout)
        except asyncio.TimeoutError as exc:
            self._pending_cancels.pop(request_id, None)
            raise TimeoutError("Timed out waiting for cancel acknowledgement") from exc
        return result

    def cancel_order(
        self,
        *,
        client_order_id: Optional[str] = None,
        txid: Optional[str] = None,
    ) -> Mapping[str, Any]:
        loop = self._loop
        if loop is None:
            raise RuntimeError("KrakenWebSocketClient not running")
        coro = self._cancel_order_async(client_order_id=client_order_id, txid=txid)
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=self._order_timeout + 2.0)

    # ------------------------------------------------------------------
    # Snapshot helpers
    # ------------------------------------------------------------------
    def latest_balances(self) -> Dict[str, float]:
        with self._balance_lock:
            return dict(self._balances)

    def open_orders(self) -> Dict[str, Dict[str, Any]]:
        with self._order_lock:
            return dict(self._open_orders)

    def recent_trades(self, limit: int = 25) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        return list(self._trades[-limit:])

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------
    def _log(
        self,
        level: str,
        symbol: str,
        message: str,
        *,
        detail: Optional[str] = None,
    ) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        meta = f" | {detail}" if detail else ""
        text = f"[{timestamp}][KrakenWS][{level.upper()}] {symbol}: {message}{meta}"
        colour = _CONSOLE_COLOURS.get(level.upper())
        if colour:
            text = f"{colour}{text}{_CONSOLE_RESET}"
        print(text)
        if self._logger is None:
            return
        try:
            if level.upper() == "TRADE":
                self._logger.write(
                    {
                        "type": "trade_event",
                        "symbol": symbol,
                        "message": message,
                        "detail": detail,
                    }
                )
            else:
                self._logger.log_feed_event(level=level, symbol=symbol, message=message, detail=detail)
        except Exception:
            pass


__all__ = [
    "KrakenWebSocketClient",
    "OrderStatus",
    "_kraken_interval_minutes",
]
