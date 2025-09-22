"""Async Kraken WebSocket client providing market data and execution."""

from __future__ import annotations

import asyncio
import json
import random
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
)

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
from desk.services.pretty_logger import pretty_logger
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
        jitter = random.uniform(0.0, self.delay * 0.1)
        now = min(self.delay + jitter, self.maximum)
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


class KrakenWSClient:
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
        order_rate_limit: int = 20,
        order_rate_period: float = 60.0,
        order_timeout: float = 10.0,
        public_factory: Optional[Callable[..., Awaitable[WebSocketClientProtocol]]] = None,
        private_factory: Optional[Callable[..., Awaitable[WebSocketClientProtocol]]] = None,
        rest_fetcher: Optional[
            Callable[[str, str, int], Iterable[Mapping[str, Any]]]
        ] = None,
        rest_candle_limit: int = 3,
    ) -> None:
        if not pairs:
            raise ValueError("At least one Kraken pair must be provided")
        self._pairs: Dict[str, str] = {str(pair): str(symbol) for pair, symbol in pairs.items()}
        self._symbols_to_pairs: Dict[str, str] = {
            symbol: pair for pair, symbol in self._pairs.items()
        }
        self._store = store
        self._logger = logger
        self._timeframe_text = str(timeframe)
        self._interval = _kraken_interval_minutes(timeframe)
        self._rest_fetcher = rest_fetcher
        self._rest_limit = max(1, int(rest_candle_limit))
        self._token_provider = token_provider
        self._public_factory = public_factory or (
            lambda url: websockets.connect(url, ping_interval=None)
        )
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
        self._public_state_lock = threading.Lock()
        self._private_state_lock = threading.Lock()
        self._public_subscriptions: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], Set[str]] = defaultdict(set)
        self._private_subscriptions: Set[str] = set()
        self._last_ohlc_timestamp: Dict[str, float] = {}
        self._stale_threshold = max(60.0, float(self._interval) * 60.0 * 3.0)
        self._ws_token: Optional[str] = None
        self._token_lock = threading.Lock()
        # Default subscription for OHLC candles so strategies keep functioning.
        self.subscribe_public(list(self._pairs.values()), ["ohlc"])
        if token_provider is not None:
            self.subscribe_private(["openOrders", "ownTrades", "accountBalances"])
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
        tasks.append(asyncio.create_task(self._watch_public_staleness()))
        await shutdown.wait()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # WebSocket helpers
    # ------------------------------------------------------------------
    class _SocketContextAdapter:
        """Wrap sockets returned by factories that are not context managers."""

        def __init__(self, socket_obj: Any) -> None:
            self._socket_obj = socket_obj

        async def __aenter__(self) -> WebSocketClientProtocol:
            if asyncio.iscoroutine(self._socket_obj) or isinstance(
                self._socket_obj, Awaitable
            ):
                protocol = await self._socket_obj  # type: ignore[assignment]
            else:
                protocol = self._socket_obj
            self._socket_obj = protocol
            return protocol

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            protocol = self._socket_obj
            close = getattr(protocol, "close", None)
            try:
                if asyncio.iscoroutinefunction(close):
                    await close()  # type: ignore[misc]
                elif callable(close):
                    result = close()
                    if asyncio.iscoroutine(result):
                        await result
            finally:
                self._socket_obj = None
            return False

    def _open_socket(
        self,
        factory: Callable[..., Any],
        url: str,
        **kwargs: Any,
    ) -> "KrakenWSClient._SocketContextAdapter | Any":
        socket_obj = factory(url, **kwargs)
        if hasattr(socket_obj, "__aenter__") and hasattr(socket_obj, "__aexit__"):
            return socket_obj
        return KrakenWSClient._SocketContextAdapter(socket_obj)

    def register_public_handler(
        self, channel: str, handler: Callable[[str, Mapping[str, Any]], None]
    ) -> None:
        self._public_handlers[channel].append(handler)

    async def _public_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                async with self._open_socket(self._public_factory, self.PUBLIC_URL) as socket:
                    self._public_socket = socket
                    await self._resubscribe_public(socket)
                    self._public_backoff.reset()
                    async for message in socket:
                        if self._stop_event.is_set():
                            break
                        await self._handle_public_message(message)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                self._log("WARNING", "ALL", "Kraken public WebSocket disconnected", detail=str(exc))
                await self._rest_fallback_refresh()
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
            if event == "subscriptionStatus":
                status = payload.get("status")
                detail = payload.get("pair") or payload.get("channelName")
                self._log(
                    "INFO",
                    "ALL",
                    f"Kraken public subscription {status}",
                    detail=str(detail),
                )
                if status == "error":
                    error_msg = str(payload.get("errorMessage") or "unknown error")
                    if "EGeneral:Invalid arguments" in error_msg:
                        self._handle_invalid_public_subscription(payload)
                    self._log("ERROR", "ALL", "Kraken public error", detail=error_msg)
                return
            if event == "error":
                detail = payload.get("errorMessage") or payload
                if isinstance(detail, str) and "EGeneral:Invalid arguments" in detail:
                    self._handle_invalid_public_subscription(payload)
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
        self._last_ohlc_timestamp[str(pair)] = time.time()

    def _rest_fetch_wrapper(self, symbol: str) -> List[Dict[str, float]]:
        fetcher = self._rest_fetcher
        if fetcher is None:
            return []
        result: Iterable[Mapping[str, Any]]
        result = fetcher(symbol, self._timeframe_text, self._rest_limit)
        candles: List[Dict[str, float]] = []
        for entry in result or []:
            if not isinstance(entry, Mapping):
                continue
            try:
                candles.append(
                    {
                        "timestamp": float(entry.get("timestamp", 0.0)),
                        "open": float(entry.get("open", 0.0)),
                        "high": float(entry.get("high", 0.0)),
                        "low": float(entry.get("low", 0.0)),
                        "close": float(entry.get("close", 0.0)),
                        "volume": float(entry.get("volume", 0.0)),
                    }
                )
            except (TypeError, ValueError):
                continue
        return candles

    async def _rest_fallback_refresh(self, symbols: Optional[Iterable[str]] = None) -> None:
        if self._rest_fetcher is None:
            return
        target_symbols = list(dict.fromkeys(symbols or self._pairs.values()))
        if not target_symbols:
            return
        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(None, self._rest_fetch_wrapper, symbol)
            for symbol in target_symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for symbol, result in zip(target_symbols, results):
            if isinstance(result, Exception):
                self._log("WARNING", symbol, "REST fallback failed", detail=str(result))
                continue
            if not result:
                continue
            pretty_logger.rest_fallback_notice()
            try:
                inserted = self._store.append(symbol, result)
            except Exception as exc:  # pragma: no cover - defensive store guard
                self._log("ERROR", symbol, "REST fallback store failure", detail=str(exc))
                continue
            if inserted:
                pair_name = self._symbols_to_pairs.get(symbol, symbol)
                self._last_ohlc_timestamp[str(pair_name)] = time.time()
                self._log(
                    "INFO",
                    symbol,
                    "REST fallback appended candles",
                    detail=str(inserted),
                )

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
                with self._token_lock:
                    self._ws_token = token
            except Exception as exc:
                self._log("ERROR", "ALL", "Failed to fetch Kraken WS token", detail=str(exc))
                delay = self._private_backoff.advance()
                await asyncio.sleep(delay)
                continue
            try:
                async with self._open_socket(self._private_factory, self.PRIVATE_URL) as socket:
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
                with self._token_lock:
                    self._ws_token = None
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
        if isinstance(result, str):
            return result
        if result is None:
            raise RuntimeError("Token provider returned None")
        if callable(result):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, result)
        return str(result)

    async def _subscribe_private(self, socket: WebSocketClientProtocol, token: str) -> None:
        payloads: List[Dict[str, Any]] = []
        with self._private_state_lock:
            for channel in sorted(self._private_subscriptions):
                payloads.append(
                    {
                        "event": "subscribe",
                        "subscription": {"name": channel, "token": token},
                    }
                )
        if not payloads:
            return
        for entry in payloads:
            await socket.send(json.dumps(entry))
            await asyncio.sleep(0)

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
                detail = str(payload.get("errorMessage") or payload)
                self._log("ERROR", "ALL", "Kraken private error", detail=detail)
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
        detail = {
            "status": status,
            "clientOrderId": client_order_id,
            "txid": txid,
            "message": message,
        }
        try:
            detail_text = json.dumps(detail, default=str)
        except TypeError:
            detail_text = str(detail)
        symbol = self._pairs.get(payload.get("pair", ""), "ALL")
        if status == "ok":
            self._log("TRADE", symbol, "Order acknowledged", detail=detail_text)
        else:
            self._log("WARNING", symbol, "Order rejected", detail=detail_text)

    async def _resolve_cancel(self, payload: Mapping[str, Any]) -> None:
        client_order_id = str(payload.get("clientOrderId") or payload.get("reqid") or "")
        future = self._pending_cancels.pop(client_order_id, None)
        if future is not None and not future.done():
            future.set_result(payload)
        self._log(
            "INFO",
            "ALL",
            "Cancel acknowledgement",
            detail=str({"clientOrderId": client_order_id, "status": payload.get("status")}),
        )

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
    # Subscription helpers
    # ------------------------------------------------------------------

    def subscribe_public(self, symbols: Iterable[str], channels: Iterable[Any]) -> None:
        pairs = self._resolve_pairs(symbols)
        normalized_channels = [
            self._normalize_public_channel(channel) for channel in channels
        ]
        normalized_channels = [entry for entry in normalized_channels if entry is not None]
        if not pairs or not normalized_channels:
            return
        updates: List[Tuple[Tuple[str, Tuple[Tuple[str, Any], ...]], str]] = []
        with self._public_state_lock:
            for channel in normalized_channels:
                for pair in pairs:
                    if pair not in self._public_subscriptions[channel]:
                        self._public_subscriptions[channel].add(pair)
                        updates.append((channel, pair))
        if not updates:
            return
        detail_entries = []
        for channel, pair in updates:
            name, extra = channel
            suffix = ""
            if extra:
                suffix = "," + ",".join(f"{key}={value}" for key, value in extra)
            detail_entries.append(f"{name}{suffix}:{pair}")
        if detail_entries:
            self._log(
                "INFO",
                "ALL",
                "Queued Kraken public subscriptions",
                detail=" | ".join(detail_entries),
            )
        loop = self._loop
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._send_public_subscriptions(updates), loop
            )

    def subscribe_private(self, channels: Iterable[str]) -> None:
        names = [str(channel) for channel in channels if str(channel).strip()]
        if not names:
            return
        changed = False
        with self._private_state_lock:
            for channel in names:
                if channel not in self._private_subscriptions:
                    self._private_subscriptions.add(channel)
                    changed = True
        if not changed:
            return
        self._log(
            "INFO",
            "ALL",
            "Queued Kraken private subscriptions",
            detail=", ".join(sorted(self._private_subscriptions)),
        )
        loop = self._loop
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(self._resubscribe_private(), loop)

    def send_order(self, order_data: Mapping[str, Any]) -> OrderStatus:
        pair = self._normalize_pair(order_data.get("pair") or order_data.get("symbol"))
        if not pair:
            raise ValueError("Order data must include a valid Kraken trading pair")
        side = str(order_data.get("side") or order_data.get("type") or "").lower()
        order_type = str(order_data.get("ordertype") or order_data.get("order_type") or "")
        volume_value = order_data.get("volume") or order_data.get("size")
        if not side or not order_type or volume_value is None:
            raise ValueError("Order data missing side, ordertype, or volume")
        price_value = order_data.get("price")
        client_order_id = order_data.get("clientOrderId") or order_data.get("reqid")
        validate = bool(order_data.get("validate", False))
        return self.submit_order(
            pair,
            side=side,
            order_type=order_type,
            volume=float(volume_value),
            price=float(price_value) if price_value is not None else None,
            client_order_id=str(client_order_id) if client_order_id else None,
            validate=validate,
        )

    async def _resubscribe_private(self) -> None:
        socket = self._private_socket
        if socket is None:
            return
        if self._token_provider is None:
            return
        with self._token_lock:
            token = self._ws_token
        if not token:
            try:
                token = await self._acquire_token()
            except Exception as exc:  # pragma: no cover - network guard
                self._log("ERROR", "ALL", "Failed to refresh private token", detail=str(exc))
                return
            with self._token_lock:
                self._ws_token = token
        await self._subscribe_private(socket, token)

    async def _send_public_subscriptions(
        self, updates: Iterable[Tuple[Tuple[str, Tuple[Tuple[str, Any], ...]], str]]
    ) -> None:
        socket = self._public_socket
        if socket is None:
            return
        batched: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], List[str]] = defaultdict(list)
        for channel, pair in updates:
            batched[channel].append(pair)
        for channel, pairs in batched.items():
            name, extra = channel
            payload = {
                "event": "subscribe",
                "pair": sorted(pairs),
                "subscription": {"name": name},
            }
            if name == "ohlc":
                payload["subscription"]["interval"] = self._interval
            for key, value in extra:
                payload["subscription"][key] = value
            await socket.send(json.dumps(payload))
            await asyncio.sleep(0)

    async def _resubscribe_public(self, socket: WebSocketClientProtocol) -> None:
        updates: List[Tuple[Tuple[str, Tuple[Tuple[str, Any], ...]], List[str]]] = []
        with self._public_state_lock:
            for channel, pairs in self._public_subscriptions.items():
                if not pairs:
                    continue
                updates.append((channel, sorted(pairs)))
        for channel, pairs in updates:
            name, extra = channel
            payload = {
                "event": "subscribe",
                "pair": pairs,
                "subscription": {"name": name},
            }
            if name == "ohlc":
                payload["subscription"]["interval"] = self._interval
            for key, value in extra:
                payload["subscription"][key] = value
            await socket.send(json.dumps(payload))
            await asyncio.sleep(0)

    def _resolve_pairs(self, symbols: Iterable[str]) -> List[str]:
        pairs: List[str] = []
        for symbol in symbols:
            candidate = self._normalize_pair(symbol)
            if candidate:
                pairs.append(candidate)
        return pairs

    def _normalize_pair(self, symbol_or_pair: Any) -> Optional[str]:
        if not symbol_or_pair:
            return None
        text = str(symbol_or_pair)
        if text in self._pairs:
            return text
        if text in self._symbols_to_pairs:
            return self._symbols_to_pairs[text]
        upper = text.upper()
        if upper in self._symbols_to_pairs:
            return self._symbols_to_pairs[upper]
        if upper in self._pairs:
            return upper
        return None

    def _normalize_public_channel(
        self, channel: Any
    ) -> Optional[Tuple[str, Tuple[Tuple[str, Any], ...]]]:
        if isinstance(channel, Mapping):
            name = str(channel.get("name") or "").strip()
            if not name:
                return None
            extras = tuple(
                sorted(
                    ((str(key), channel[key]) for key in channel.keys() if key != "name"),
                    key=lambda item: item[0],
                )
            )
            return (name, extras)
        if channel is None:
            return None
        name = str(channel).strip()
        if not name:
            return None
        return (name, tuple())

    def _handle_invalid_public_subscription(self, payload: Mapping[str, Any]) -> None:
        subscription = payload.get("subscription")
        if not isinstance(subscription, Mapping):
            return
        name = str(subscription.get("name") or "")
        pair = payload.get("pair")
        if isinstance(pair, Sequence) and pair:
            target_pairs = [str(pair[0])]
        elif isinstance(pair, str):
            target_pairs = [pair]
        else:
            target_pairs = []
        with self._public_state_lock:
            if not name or not target_pairs:
                return
            for entry in target_pairs:
                for key in list(self._public_subscriptions.keys()):
                    if key[0] != name:
                        continue
                    self._public_subscriptions[key].discard(entry)
        detail = {
            "channel": name,
            "pair": target_pairs,
            "payload": payload,
        }
        try:
            detail_text = json.dumps(detail, default=str)
        except TypeError:
            detail_text = str(detail)
        self._log("ERROR", "ALL", "Removed invalid public subscription", detail=detail_text)

    async def _watch_public_staleness(self) -> None:
        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(min(60.0, self._stale_threshold / 2.0))
                now = time.time()
                stale_pairs: List[str] = []
                for pair, ts in list(self._last_ohlc_timestamp.items()):
                    if now - ts > self._stale_threshold:
                        stale_pairs.append(pair)
                if not stale_pairs:
                    continue
                self._log(
                    "WARNING",
                    "ALL",
                    "Detected stale Kraken OHLC stream",
                    detail=", ".join(stale_pairs),
                )
                for pair in stale_pairs:
                    self._last_ohlc_timestamp.pop(pair, None)
                await self._send_public_subscriptions(
                    ((("ohlc", tuple()), pair) for pair in stale_pairs)
                )
                await self._rest_fallback_refresh(
                    self._pairs.get(pair, pair) for pair in stale_pairs
                )
        except asyncio.CancelledError:
            return

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
            raise RuntimeError("KrakenWSClient not started")
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
        with self._token_lock:
            token = self._ws_token
        if not token:
            token = await self._acquire_token()
            with self._token_lock:
                self._ws_token = token
        payload["token"] = token
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
        symbol = self._pairs.get(pair, pair)
        self._log(
            "INFO",
            symbol,
            f"Submitting {side.upper()} {volume:.10f} {pair} ({order_type})",
            detail=f"clientOrderId={client_order_id}",
        )
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
            raise RuntimeError("KrakenWSClient not running")
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
            raise RuntimeError("KrakenWSClient not started")
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
            raise RuntimeError("KrakenWSClient not running")
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
        text = f"[KrakenWS] {symbol}: {message}{meta}"
        level_name = level.upper()
        lower_message = message.lower()
        if "rest fallback" in lower_message:
            pretty_logger.rest_fallback_notice()
            if level_name in {"ERROR", "WARNING"}:
                if level_name == "ERROR":
                    pretty_logger.error(text)
                else:
                    pretty_logger.warning(text)
        else:
            dedupe_key = None
            if "subscription" in lower_message:
                dedupe_key = f"ws_subscription:{symbol}:{message}"
            if level_name == "ERROR":
                pretty_logger.error(text, dedupe_key=dedupe_key)
            elif level_name == "WARNING":
                pretty_logger.warning(text, dedupe_key=dedupe_key)
            else:
                pretty_logger.info(text, dedupe_key=dedupe_key)
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


KrakenWebSocketClient = KrakenWSClient

__all__ = [
    "KrakenWSClient",
    "KrakenWebSocketClient",
    "OrderStatus",
    "_kraken_interval_minutes",
]
