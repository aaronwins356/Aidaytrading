"""Broker implementations used by the trading runtime.

``KrakenBroker``
    Thin wrapper around :mod:`ccxt` for routing live orders to Kraken.  The
    implementation focuses on defensive error handling so network hiccups or
    exchange downtime never bubble up as uncaught exceptions inside the trading
    loop.

Paper trading support has been removed so the historical ``PaperBroker``
implementation is no longer shipped with the desk.
"""

from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

try:  # pragma: no cover - import guard for optional dependency
    import ccxt  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised in tests
    class _MissingCCXT:
        def __getattr__(self, name: str):
            raise ModuleNotFoundError(
                "ccxt is required for Kraken live trading but is not installed"
            )

    ccxt = _MissingCCXT()  # type: ignore

try:  # pragma: no cover - type checking only
    from typing import TYPE_CHECKING
except ImportError:  # pragma: no cover
    TYPE_CHECKING = False

if TYPE_CHECKING:  # pragma: no cover - import hints only
    from desk.services.logger import EventLogger
    from desk.services.telemetry import TelemetryClient


def _parse_symbol(symbol: str) -> Tuple[str, str]:
    """Return ``(base, quote)`` currency codes from a CCXT style symbol."""

    if "/" in symbol:
        base, quote = symbol.split("/", 1)
    elif "-" in symbol:
        base, quote = symbol.split("-", 1)
    else:
        # Fallback to assuming the quote is USD when the symbol is a bare asset
        base, quote = symbol, "USD"
    return base.upper(), quote.upper()


@dataclass(frozen=True)
class BalanceSnapshot:
    """Container returned by :meth:`KrakenBroker.available_balances`."""

    base: float
    quote: float


class KrakenBroker:
    """Minimal live-trading wrapper for the Kraken REST API via ccxt."""

    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
        event_logger: Optional["EventLogger"] = None,
        telemetry: Optional["TelemetryClient"] = None,
        request_timeout: float = 30.0,
        session_config: Optional[Dict[str, object]] = None,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.event_logger = event_logger
        self.telemetry = telemetry
        self.request_timeout = max(1.0, float(request_timeout))
        self.session_config = dict(session_config or {})
        self._latency_log: list[Dict[str, float]] = []
        self._last_balance: dict[str, Dict[str, float]] = {}
        self.mode = "live"

        if self.session_config and "timeout" not in self.session_config:
            # ccxt expects timeout in milliseconds
            self.session_config["timeout"] = int(self.request_timeout * 1000)

        self.exchange_name = "kraken"
        self.exchange = self._create_exchange()

    # ------------------------------------------------------------------
    def _log_event(self, level: str, message: str, *, symbol: str = "BROKER", **metadata) -> None:
        payload = {k: v for k, v in metadata.items() if v is not None}
        meta = ""
        if payload:
            meta = " | " + ", ".join(f"{key}={value}" for key, value in payload.items())
        print(f"[KrakenBroker][{level.upper()}] {symbol}: {message}{meta}")
        if self.event_logger is not None:
            self.event_logger.log_feed_event(
                level=level, symbol=symbol, message=message, **payload
            )

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
            try:
                load_markets()
            except Exception as exc:  # pragma: no cover - defensive network guard
                self._log_event(
                    "ERROR",
                    "Failed to load Kraken markets",
                    detail="".join(str(part) for part in getattr(exc, "args", []) if part),
                )
                raise
        return exchange

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
    def _execute(self, operation: str, func):
        try:
            with self._measure_latency(operation):
                return func(self.exchange)
        except Exception as exc:  # pragma: no cover - defensive guard
            message = "".join(str(part) for part in getattr(exc, "args", []) if part) or str(exc)
            self._log_event("ERROR", f"{operation} failed", detail=message)
            raise

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
        return self._execute(
            "fetch_ohlcv",
            lambda exchange: exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, limit=limit, since=normalized_since
            ),
        )

    def fetch_price(self, symbol: str) -> float:
        ticker = self._execute(
            "fetch_price", lambda exchange: exchange.fetch_ticker(symbol)
        )
        price = ticker.get("last") if isinstance(ticker, dict) else None
        return float(price or 0.0)

    # ------------------------------------------------------------------
    def fetch_balance(self) -> Dict[str, Dict[str, float]]:
        """Return the raw balance payload from Kraken.

        The response is cached so subsequent balance validations inside the same
        loop iteration do not repeatedly hit the exchange.
        """

        payload = self._execute("fetch_balance", lambda ex: ex.fetch_balance())
        if isinstance(payload, dict):
            self._last_balance = payload
        return self._last_balance

    # ------------------------------------------------------------------
    def _normalize_order(
        self, order: object, *, symbol: str, requested_qty: float, reference_price: float
    ) -> Dict[str, float]:
        normalized: Dict[str, float] = {
            "symbol": symbol,
            "side": "",
            "qty": float(requested_qty),
            "price": float(reference_price),
        }
        if isinstance(order, dict):
            normalized.update({k: order.get(k) for k in ("id", "orderType", "type")})
            if "side" in order:
                normalized["side"] = str(order["side"]).lower()
            if "amount" in order:
                normalized["qty"] = float(order.get("amount", requested_qty) or requested_qty)
            if "filled" in order and float(order.get("filled", 0.0) or 0.0) > 0:
                normalized["qty"] = float(order.get("filled", normalized["qty"]))
            if "price" in order and float(order.get("price", 0.0) or 0.0) > 0:
                normalized["price"] = float(order.get("price", reference_price))
            if "cost" in order and normalized["qty"]:
                implied_price = float(order.get("cost", 0.0) or 0.0) / normalized["qty"]
                if implied_price > 0:
                    normalized["price"] = implied_price
        normalized.setdefault("side", "")
        normalized.setdefault("timestamp", time.time())
        return normalized

    def market_order(self, symbol: str, side: str, qty: float):
        qty = float(qty)
        if qty <= 0:
            self._log_event(
                "WARNING",
                "Rejected market order with non-positive quantity",
                symbol=symbol,
                requested_qty=qty,
            )
            return None
        side = side.lower().strip()
        price = self.fetch_price(symbol)
        order = self._execute(
            "market_order",
            lambda exchange: exchange.create_order(symbol, "market", side, qty),
        )
        normalized = self._normalize_order(
            order, symbol=symbol, requested_qty=qty, reference_price=price
        )
        normalized.setdefault("requested_qty", qty)
        normalized.setdefault("price", price)
        return normalized

    # ------------------------------------------------------------------
    def account_equity(self) -> float:
        balances = self.fetch_balance()
        return self._extract_equity(balances)

    @staticmethod
    def _extract_equity(balances: object) -> float:
        if isinstance(balances, dict):
            total = balances.get("total")
            if isinstance(total, dict):
                return float(sum(float(value or 0.0) for value in total.values()))
            if total is not None:
                try:
                    return float(total)
                except (TypeError, ValueError):
                    pass
            usd_entry = balances.get("USD")
            if isinstance(usd_entry, dict):
                for key in ("total", "free", "used"):
                    value = usd_entry.get(key)
                    if value is not None:
                        try:
                            return float(value)
                        except (TypeError, ValueError):
                            continue
            if usd_entry is not None:
                try:
                    return float(usd_entry)
                except (TypeError, ValueError):
                    pass
        return 0.0

    def available_balances(self, symbol: str) -> BalanceSnapshot:
        """Return available base/quote balances for ``symbol``.

        Kraken's payload distinguishes between ``free`` and ``used`` balances.
        To remain conservative during live trading we only consider the ``free``
        funds available for routing new orders.
        """

        base, quote = _parse_symbol(symbol)
        payload = self.fetch_balance()
        total = payload.get("total") if isinstance(payload, dict) else None
        free = payload.get("free") if isinstance(payload, dict) else None

        def _resolve(container, currency: str) -> float:
            if isinstance(container, dict) and currency in container:
                value = container[currency]
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return 0.0
            return 0.0

        base_available = _resolve(free, base) or _resolve(total, base)
        quote_available = _resolve(free, quote) or _resolve(total, quote)
        return BalanceSnapshot(base=base_available, quote=quote_available)

    def can_execute_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        *,
        price: float,
        slippage: float,
    ) -> bool:
        """Return ``True`` when sufficient balances exist for a trade."""

        if qty <= 0 or price <= 0:
            return False
        balances = self.available_balances(symbol)
        side = side.lower()
        effective_price = price * (1 + slippage if side == "buy" else 1 - slippage)
        if side == "buy":
            required = effective_price * qty
            return balances.quote >= required
        required = qty
        return balances.base >= required

    # ------------------------------------------------------------------
    def close(self) -> None:
        close = getattr(self.exchange, "close", None)
        if callable(close):  # pragma: no cover - network cleanup
            with contextlib.suppress(Exception):
                close()


