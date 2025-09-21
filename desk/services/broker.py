"""Kraken execution adapter backed by the WebSocket API."""

from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

try:  # pragma: no cover - optional dependency guard
    import ccxt  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised in tests
    class _MissingCCXT:
        def __getattr__(self, name: str):
            raise ModuleNotFoundError(
                "ccxt is required for Kraken live trading but is not installed"
            )

    ccxt = _MissingCCXT()  # type: ignore

from desk.services.kraken_ws import KrakenWebSocketClient
from desk.services.logger import EventLogger

try:  # pragma: no cover - typing only
    from typing import TYPE_CHECKING
except ImportError:  # pragma: no cover - narrow Python versions
    TYPE_CHECKING = False

if TYPE_CHECKING:  # pragma: no cover - hints only
    from desk.services.telemetry import TelemetryClient
    from desk.services.feed_updater import CandleStore


@dataclass(frozen=True)
class BalanceSnapshot:
    base: float
    quote: float


class KrakenBroker:
    """Live trading broker that routes orders through Kraken WebSockets."""

    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
        event_logger: Optional[EventLogger] = None,
        telemetry: Optional["TelemetryClient"] = None,
        request_timeout: float = 30.0,
        session_config: Optional[Dict[str, object]] = None,
        symbols: Optional[Iterable[str]] = None,
        timeframe: str = "1m",
        candle_store: Optional["CandleStore"] = None,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.event_logger = event_logger
        self.telemetry = telemetry
        self.request_timeout = max(1.0, float(request_timeout))
        self.session_config = dict(session_config or {})
        self._latency_log: list[Dict[str, float]] = []
        self._last_balance: dict[str, Dict[str, float]] = {}
        self._symbols = [str(symbol) for symbol in (symbols or [])]
        self._candle_store = candle_store
        self._lock = threading.Lock()
        if self.session_config and "timeout" not in self.session_config:
            self.session_config["timeout"] = int(self.request_timeout * 1000)

        self.exchange = self._create_exchange()
        self.websocket = self._create_websocket(timeframe=timeframe)
        self.websocket.start()

    # ------------------------------------------------------------------
    def _create_exchange(self):  # pragma: no cover - network bootstrap
        params = {
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "enableRateLimit": True,
        }
        params.update(self.session_config)
        exchange = ccxt.kraken(params)
        load_markets = getattr(exchange, "load_markets", None)
        if callable(load_markets):
            load_markets()
        return exchange

    def _resolve_ws_pairs(self) -> Dict[str, str]:
        if not self._symbols:
            return {}
        pairs: Dict[str, str] = {}
        for symbol in self._symbols:
            try:
                market = self.exchange.market(symbol)
            except Exception:  # pragma: no cover - defensive guard
                continue
            wsname = None
            info = market.get("info") if isinstance(market, dict) else None
            if isinstance(info, dict):
                wsname = info.get("wsname") or info.get("wsName")
            if not wsname and isinstance(market, dict):
                wsname = market.get("wsname")
            if wsname:
                pairs[str(wsname)] = symbol
                continue
            base = str(market.get("baseId") or market.get("base") or symbol.split("/")[0])
            quote = str(market.get("quoteId") or market.get("quote") or symbol.split("/")[-1])
            pairs[f"{base}/{quote}"] = symbol
        return pairs

    def _ws_token(self) -> str:
        method = getattr(self.exchange, "privatePostGetWebSocketsToken", None)
        if not callable(method):
            raise RuntimeError("ccxt.kraken does not expose privatePostGetWebSocketsToken")
        response = method()
        if not isinstance(response, dict):
            raise RuntimeError("Unexpected Kraken token response")
        errors = response.get("error")
        if isinstance(errors, (list, tuple)) and errors:
            raise RuntimeError(f"Kraken token error: {errors}")
        if isinstance(errors, str) and errors:
            raise RuntimeError(f"Kraken token error: {errors}")
        token = None
        if "token" in response:
            token = response.get("token")
        elif isinstance(response.get("result"), dict):
            token = response["result"].get("token")
        if not token:
            raise RuntimeError("Unexpected Kraken token response")
        return str(token)

    def _create_websocket(self, timeframe: str) -> KrakenWebSocketClient:
        pairs = self._resolve_ws_pairs()
        store = self._candle_store
        if store is None:
            from desk.services.feed_updater import CandleStore  # lazy import to avoid cycles

            store = CandleStore()
            self._candle_store = store
        client = KrakenWebSocketClient(
            pairs=pairs or {"XBT/USD": "BTC/USD"},
            timeframe=timeframe,
            store=store,
            token_provider=self._ws_token,
            logger=self.event_logger,
        )
        symbols = list((pairs or {"XBT/USD": "BTC/USD"}).values()) or list(self._symbols)
        subscribe = getattr(client, "subscribe_public", None)
        if symbols and callable(subscribe):
            try:
                subscribe(symbols, ["ticker", "trade", {"name": "book", "depth": 10}])
            except Exception:
                pass
        return client

    # ------------------------------------------------------------------
    def _log_event(self, level: str, message: str, *, symbol: str = "BROKER", **metadata) -> None:
        payload = {k: v for k, v in metadata.items() if v is not None}
        meta = ""
        if payload:
            meta = " | " + ", ".join(f"{key}={value}" for key, value in payload.items())
        print(f"[KrakenBroker][{level.upper()}] {symbol}: {message}{meta}")
        if self.event_logger is not None:
            self.event_logger.log_feed_event(level=level, symbol=symbol, message=message, **payload)

    @contextlib.contextmanager
    def _measure_latency(self, operation: str):
        started = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - started
            self._latency_log.append(
                {"operation": operation, "duration": elapsed, "timestamp": started}
            )
            if self.telemetry is not None:
                self.telemetry.record_latency(operation, elapsed)

    @property
    def latency_log(self) -> list[Dict[str, float]]:
        return list(self._latency_log)

    # ------------------------------------------------------------------
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        *,
        limit: int = 50,
        since: Optional[int | float] = None,
    ):
        normalized_since = int(since) if since is not None else None
        with self._measure_latency("fetch_ohlcv"):
            return self.exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, limit=limit, since=normalized_since
            )

    def fetch_price(self, symbol: str) -> float:
        with self._measure_latency("fetch_price"):
            ticker = self.exchange.fetch_ticker(symbol)
        price = ticker.get("last") if isinstance(ticker, dict) else None
        return float(price or 0.0)

    # ------------------------------------------------------------------
    def fetch_balance(self) -> Dict[str, Dict[str, float]]:
        payload = self.websocket.latest_balances()
        if payload:
            return {"total": payload}
        with self._measure_latency("fetch_balance"):
            raw = self.exchange.fetch_balance()
        if isinstance(raw, dict):
            self._last_balance = raw
            return raw
        return self._last_balance

    def available_balances(self, symbol: str) -> BalanceSnapshot:
        balances = self.fetch_balance().get("total", {})
        base, quote = _parse_symbol(symbol)
        base_amt = float(balances.get(base, 0.0)) if isinstance(balances, dict) else 0.0
        quote_amt = float(balances.get(quote, 0.0)) if isinstance(balances, dict) else 0.0
        return BalanceSnapshot(base=base_amt, quote=quote_amt)

    def account_equity(self) -> float:
        balances = self.fetch_balance().get("total", {})
        total = 0.0
        if isinstance(balances, dict):
            for asset, amount in balances.items():
                try:
                    total += float(amount)
                except (TypeError, ValueError):
                    continue
        self._latency_log.append({"operation": "account_equity", "duration": 0.0, "timestamp": time.time()})
        return total

    # ------------------------------------------------------------------
    def market_order(self, symbol: str, side: str, qty: float) -> Dict[str, float]:
        pair = self._resolve_order_pair(symbol)
        with self._measure_latency("market_order"):
            result = self.websocket.submit_order(
                pair,
                side=side,
                order_type="market",
                volume=qty,
            )
        payload = {
            "requested_qty": float(qty),
            "price": self.fetch_price(symbol),
            "status": result.status,
            "txid": result.txid,
            "client_order_id": result.client_order_id,
        }
        return payload

    def cancel_order(self, *, client_order_id: Optional[str] = None, txid: Optional[str] = None):
        return self.websocket.cancel_order(client_order_id=client_order_id, txid=txid)

    def can_execute_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        *,
        price: float,
        slippage: float,
    ) -> bool:
        snapshot = self.available_balances(symbol)
        if side.lower() == "buy":
            cost = price * qty * (1 + slippage)
            return snapshot.quote >= cost
        cost = qty * (1 + slippage)
        return snapshot.base >= cost

    # ------------------------------------------------------------------
    def close(self) -> None:
        try:
            self.websocket.stop()
        except Exception:
            pass
        close = getattr(self.exchange, "close", None)
        if callable(close):
            close()

    # ------------------------------------------------------------------
    def _resolve_order_pair(self, symbol: str) -> str:
        pairs = self._resolve_ws_pairs()
        for pair, mapped in pairs.items():
            if mapped == symbol:
                return pair
        return symbol.replace("/", "")


def _parse_symbol(symbol: str) -> tuple[str, str]:
    if "/" in symbol:
        base, quote = symbol.split("/", 1)
    elif "-" in symbol:
        base, quote = symbol.split("-", 1)
    else:
        base, quote = symbol, "USD"
    return base.upper(), quote.upper()


__all__ = ["KrakenBroker", "BalanceSnapshot"]
