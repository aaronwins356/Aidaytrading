"""Kraken trading client wrapping ccxt."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import math
from decimal import Decimal, ROUND_DOWN
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import ccxt
from ccxt.base.errors import (
    DDoSProtection,
    ExchangeNotAvailable,
    NetworkError,
    RateLimitExceeded,
    RequestTimeout,
)

from ai_trader.services.logging import get_logger
from ai_trader.services.types import OpenPosition


class KrakenClient:
    """Thin wrapper around ccxt's Kraken implementation."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_currency: str,
        rest_rate_limit: float,
        paper_trading: bool = False,
        paper_starting_equity: float = 10000.0,
        allow_shorting: bool = False,
    ) -> None:
        self._logger = get_logger(__name__)
        self._paper_trading = paper_trading
        self._base_currency = base_currency
        self._rest_rate_limit = rest_rate_limit
        self._paper_balances: Dict[str, float] = {base_currency: paper_starting_equity}
        self._allow_shorting = allow_shorting
        self._exchange = ccxt.kraken({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        })
        self._markets: Dict[str, dict] = {}
        self._starting_equity = paper_starting_equity if paper_trading else 0.0
        self._starting_equity_captured = paper_trading
        self._max_retries = 5
        self._backoff_factor = 1.5

    @property
    def starting_equity(self) -> float:
        return self._starting_equity

    @property
    def is_paper_trading(self) -> bool:
        return self._paper_trading

    async def load_markets(self) -> None:
        markets = await self._with_retries(
            self._exchange.load_markets, description="load_markets"
        )
        normalised_markets: Dict[str, dict] = {}
        for symbol, payload in (markets or {}).items():
            canonical = self._normalise_market_symbol(symbol)
            if not canonical:
                continue
            normalised_markets[canonical] = payload
        self._markets = normalised_markets

    async def fetch_price(self, symbol: str) -> Optional[float]:
        ticker = await self._with_retries(
            self._exchange.fetch_ticker,
            symbol,
            description=f"fetch_ticker:{symbol}",
        )
        price = ticker.get("last") or ticker.get("close")
        return float(price) if price else None

    async def fetch_balances(self) -> Dict[str, float]:
        if self._paper_trading:
            return {currency: float(amount) for currency, amount in self._paper_balances.items()}
        balance = await self._with_retries(
            self._exchange.fetch_balance, description="fetch_balance"
        )
        total = balance.get("total", {})
        return {currency: float(amount) for currency, amount in total.items() if amount}

    async def fetch_open_positions(self) -> List[OpenPosition]:
        """Return the broker's open positions as :class:`OpenPosition` records."""

        if self._paper_trading:
            return []

        fetcher = None
        description = "fetch_open_positions"
        for candidate, label in (
            (getattr(self._exchange, "fetch_open_positions", None), "fetch_open_positions"),
            (getattr(self._exchange, "fetch_positions", None), "fetch_positions"),
        ):
            if callable(candidate):
                fetcher = candidate
                description = label
                break

        if fetcher is None:
            self._logger.debug(
                "Exchange implementation lacks open position endpoint; returning empty result set."
            )
            return []

        raw_positions: Iterable[dict[str, Any]] | None = await self._with_retries(
            fetcher, description=description
        )
        normalized: List[OpenPosition] = []
        for payload in raw_positions or []:
            if not isinstance(payload, dict):
                self._logger.debug(
                    "Skipping non-dict position payload from broker: %s", payload
                )
                continue
            position = self._normalise_position_payload(payload)
            if position is not None:
                normalized.append(position)
        return normalized

    async def place_order(
        self,
        symbol: str,
        side: str,
        cash_spent: float,
        *,
        reduce_only: bool | None = None,
    ) -> Tuple[float, float]:
        """Place a market order sized by cash value.

        Returns a tuple of (price, quantity).
        """

        if isinstance(cash_spent, str):
            try:
                cash_spent = float(cash_spent)
            except (TypeError, ValueError) as exc:
                raise ValueError("cash_spent must be numeric") from exc

        assert isinstance(
            cash_spent, (int, float)
        ), f"cash_spent must be float, got {type(cash_spent)}"

        price = await self.fetch_price(symbol)
        if price is None:
            raise RuntimeError(f"Unable to fetch price for {symbol}")
        price = float(price)
        cash_value = float(cash_spent)
        amount = float(cash_value) / float(price)
        amount = self._adjust_amount(symbol, amount)
        amount = float(amount)

        if amount <= 0:
            raise RuntimeError("Calculated trade size is below Kraken minimum")

        base, quote = symbol.split("/")

        if self._paper_trading:
            self._simulate_fill(base, quote, side, amount, price)
            return float(price), float(amount)

        should_reduce_only = False
        if reduce_only is None:
            should_reduce_only = side == "sell" and not self._allow_shorting
        else:
            # Always enforce reduce_only when shorting is disabled even if explicitly
            # overridden. This prevents accidental naked short exposure when the
            # deployment configuration disallows it.
            should_reduce_only = reduce_only or (
                side == "sell" and not self._allow_shorting
            )

        order_params: Dict[str, Any] = {}
        if should_reduce_only:
            order_params["reduce_only"] = True

        try:
            order = await self._with_retries(
                self._exchange.create_order,
                symbol,
                "market",
                side,
                float(amount),
                None,
                order_params,
                description=f"create_order:{symbol}",
            )
        except ccxt.BaseError as exc:  # pragma: no cover - network/broker failures
            self._logger.warning("Kraken rejected order for %s: %s", symbol, exc)
            raise RuntimeError(str(exc)) from exc
        filled = float(order.get("amount", amount))
        avg_price = float(order.get("average") or price)
        return avg_price, filled

    async def close_position(self, symbol: str, side: str, amount: float) -> Tuple[float, float]:
        if isinstance(amount, str):
            try:
                amount = float(amount)
            except (TypeError, ValueError) as exc:
                raise ValueError("amount must be numeric") from exc
        if not isinstance(amount, (int, float)):
            raise TypeError("amount must be numeric")
        assert isinstance(amount, (int, float)), "amount must be numeric"
        amount = float(amount)
        exit_side = "sell" if side == "buy" else "buy"
        if self._paper_trading:
            price = await self.fetch_price(symbol)
            if price is None:
                raise RuntimeError("Missing market price for close")
            price = float(price)
            base, quote = symbol.split("/")
            self._simulate_fill(base, quote, exit_side, amount, price)
            return price, amount

        try:
            order = await self._with_retries(
                self._exchange.create_order,
                symbol,
                "market",
                exit_side,
                float(amount),
                None,
                {"reduce_only": True},
                description=f"close_order:{symbol}",
            )
        except ccxt.BaseError as exc:  # pragma: no cover - network/broker failures
            self._logger.warning("Kraken rejected close order for %s: %s", symbol, exc)
            raise RuntimeError(str(exc)) from exc
        filled = float(order.get("amount", amount))
        avg_price_raw = order.get("average") or order.get("price")
        avg_price = float(avg_price_raw) if avg_price_raw is not None else None
        if avg_price is None:
            market_price = await self.fetch_price(symbol)
            if market_price is None:
                raise RuntimeError("Unable to determine fill price")
            avg_price = float(market_price)
        return avg_price, filled

    async def compute_equity(self, prices: Dict[str, float]) -> tuple[float, Dict[str, float]]:
        """Return the total account equity alongside the raw balances."""

        balances = await self.fetch_balances()
        equity = float(balances.get(self._base_currency, 0.0))
        for asset, amount in balances.items():
            normalized_asset = self._normalise_balance_asset(asset)
            if not normalized_asset or normalized_asset == self._base_currency or amount == 0:
                continue
            symbol = f"{normalized_asset}/{self._base_currency}"
            price = prices.get(symbol)
            if price is None:
                price = await self.fetch_price(symbol)
                if price is None:
                    continue
            equity += float(amount) * float(price)
        if not self._paper_trading and not self._starting_equity_captured and equity > 0.0:
            # Capture the live balance once so downstream performance metrics use the
            # true baseline instead of the paper default.
            self._starting_equity = equity
            self._starting_equity_captured = True
            self._logger.info("Captured live starting equity at %.2f %s", equity, self._base_currency)
        return equity, balances

    def _normalise_balance_asset(self, asset: str) -> str:
        """Translate exchange-specific asset codes into standard symbols."""

        text = str(asset or "").strip()
        if not text:
            return ""
        try:
            # ``safe_currency_code`` normalises ccxt/Kraken specific asset identifiers.
            # It is resilient to missing metadata and simply echoes the input on failure.
            currency = self._exchange.safe_currency_code(text)
        except Exception:  # pragma: no cover - defensive guard
            currency = None
        normalised = str(currency or text).upper()
        if "." in normalised:
            base, suffix = normalised.split(".", 1)
            if suffix in {"F", "S"}:
                normalised = base
        alias_map = {"XBT": "BTC"}
        return alias_map.get(normalised, normalised)

    def _normalise_market_symbol(self, symbol: str) -> str:
        """Return a canonical market symbol for Kraken listings."""

        text = str(symbol or "").strip().upper()
        if "/" not in text:
            return ""
        base, quote = text.split("/", 1)
        base = self._normalise_balance_asset(base)
        quote = self._normalise_balance_asset(quote)
        if not base or not quote:
            return ""
        return f"{base}/{quote}"

    def _adjust_amount(self, symbol: str, amount: float) -> float:
        market = self._markets.get(symbol)
        if not market:
            return amount
        precision = market.get("precision", {}).get("amount", 8)
        min_amount = market.get("limits", {}).get("amount", {}).get("min", 0.0)
        adjusted = self._apply_precision(amount, precision)
        if min_amount and adjusted < min_amount:
            adjusted = 0.0
        return adjusted

    @staticmethod
    def _apply_precision(
        amount: float, precision: int | float | str | None
    ) -> float:
        """Round ``amount`` down to the exchange precision or step size.

        Kraken (via ccxt) occasionally reports the precision as a fractional step
        (e.g. ``0.0001``) instead of the number of decimal places. The original
        implementation treated the value as an integer count of decimals which
        resulted in ``TypeError`` when Python attempted to multiply a string by a
        non-integer float. Handling both representations keeps order sizing
        robust regardless of the market metadata shape.
        """

        amount_dec = Decimal(str(amount))

        if isinstance(precision, str):
            precision = precision.strip()
            if precision:
                try:
                    precision_value: float | int | None = float(precision)
                except ValueError:
                    precision_value = None
            else:
                precision_value = None
        else:
            precision_value = precision

        if isinstance(precision_value, float):
            if math.isnan(precision_value):
                precision_value = None
            elif not precision_value.is_integer():
                step = Decimal(str(precision_value))
                if step <= 0:
                    return float(amount_dec)
                steps = (amount_dec / step).to_integral_value(rounding=ROUND_DOWN)
                return float(steps * step)
            else:
                precision_value = int(precision_value)

        if isinstance(precision_value, int):
            digits = max(0, precision_value)
        else:
            digits = 0

        if digits == 0:
            quantum = Decimal("1")
        else:
            quantum = Decimal("1") / (Decimal("10") ** digits)
        return float(amount_dec.quantize(quantum, rounding=ROUND_DOWN))

    def _simulate_fill(self, base: str, quote: str, side: str, amount: float, price: float) -> None:
        amount = float(amount)
        price = float(price)
        cost = float(amount) * float(price)
        balances = self._paper_balances
        balances.setdefault(base, 0.0)
        balances.setdefault(quote, 0.0)
        if side == "buy":
            if balances[quote] < cost:
                raise RuntimeError("Insufficient paper balance")
            balances[quote] -= cost
            balances[base] += amount
        else:
            if balances[base] < amount and not self._allow_shorting:
                raise RuntimeError("Insufficient asset for paper sell")
            balances[base] -= amount
            balances[quote] += cost

    async def ensure_market(self, symbol: str) -> None:
        if symbol not in self._markets:
            await self.load_markets()

    async def _with_retries(
        self,
        func: Callable[..., Any],
        *args: object,
        description: str,
        **kwargs: object,
    ) -> Any:
        """Execute a blocking ccxt call with retry and exponential backoff."""

        delay = max(self._rest_rate_limit, 0.2)
        for attempt in range(1, self._max_retries + 1):
            try:
                result = await asyncio.to_thread(func, *args, **kwargs)
                await asyncio.sleep(self._rest_rate_limit)
                return result
            except ccxt.BaseError as exc:  # pragma: no cover - network/broker failures
                if not self._should_retry(exc) or attempt == self._max_retries:
                    self._logger.error(
                        "Kraken %s failed after %d attempts: %s",
                        description,
                        attempt,
                        exc,
                    )
                    raise
                sleep_for = self._backoff_delay(delay, attempt)
                self._logger.warning(
                    "Kraken %s retry %d/%d in %.2fs due to %s",
                    description,
                    attempt,
                    self._max_retries,
                    sleep_for,
                    exc,
                )
                await asyncio.sleep(sleep_for)
            except Exception as exc:  # pragma: no cover - defensive fallback
                if attempt == self._max_retries:
                    self._logger.error(
                        "Unexpected error calling %s after %d attempts: %s",
                        description,
                        attempt,
                        exc,
                    )
                    raise
                sleep_for = self._backoff_delay(delay, attempt)
                self._logger.warning(
                    "Unexpected %s failure (%s). Retrying in %.2fs (%d/%d)",
                    description,
                    exc,
                    sleep_for,
                    attempt,
                    self._max_retries,
                )
                await asyncio.sleep(sleep_for)

    def _should_retry(self, exc: Exception) -> bool:
        retryable = (
            NetworkError,
            DDoSProtection,
            RateLimitExceeded,
            RequestTimeout,
            ExchangeNotAvailable,
        )
        return isinstance(exc, retryable)

    def _backoff_delay(self, base_delay: float, attempt: int) -> float:
        exponent = self._backoff_factor ** (attempt - 1)
        jitter = random.uniform(0, base_delay)
        return min(20.0, base_delay * exponent + jitter)

    def _normalise_position_payload(
        self, payload: Dict[str, Any]
    ) -> Optional[OpenPosition]:
        """Convert a broker payload into an :class:`OpenPosition`.

        Kraken exposes multiple position schemas depending on the endpoint used.
        This helper defensively extracts the fields we care about while applying
        sensible fallbacks when the data is missing or malformed. Returning
        ``None`` signals the caller to skip the payload.
        """

        symbol = str(
            payload.get("symbol")
            or payload.get("info", {}).get("symbol")
            or payload.get("info", {}).get("pair")
            or ""
        ).upper()
        symbol = self._normalise_market_symbol(symbol)
        if not symbol:
            return None

        side_raw = payload.get("side") or payload.get("info", {}).get("type")
        side = str(side_raw or "buy").lower()
        if side not in {"buy", "sell"}:
            side = "buy" if side in {"long"} else "sell"

        quantity_candidates = (
            payload.get("amount"),
            payload.get("contracts"),
            payload.get("info", {}).get("vol"),
            payload.get("info", {}).get("vol_exec"),
        )
        quantity: Optional[float] = None
        for candidate in quantity_candidates:
            if candidate is None:
                continue
            try:
                quantity = float(candidate)
            except (TypeError, ValueError):
                continue
            else:
                break
        if quantity is None or quantity <= 0.0:
            return None

        price_candidates = (
            payload.get("entryPrice"),
            payload.get("price"),
            payload.get("info", {}).get("price"),
            payload.get("info", {}).get("cost"),
        )
        price: Optional[float] = None
        for candidate in price_candidates:
            if candidate is None:
                continue
            try:
                price = float(candidate)
            except (TypeError, ValueError):
                continue
            else:
                break
        if price is None or price <= 0.0:
            price = float(payload.get("info", {}).get("mark_price", 0.0) or 0.0)
        if price <= 0.0:
            return None

        cost_value = payload.get("cost") or payload.get("info", {}).get("cost")
        try:
            cash_spent = float(cost_value) if cost_value is not None else float(price * quantity)
        except (TypeError, ValueError):
            cash_spent = float(price * quantity)

        opened_at = self._coerce_open_timestamp(payload)
        worker_hint = payload.get("info", {}).get("userref") or payload.get("clientOrderId")
        worker = str(worker_hint) if worker_hint else "broker::unassigned"
        try:
            position = OpenPosition(
                worker=worker,
                symbol=symbol,
                side="buy" if side in {"buy", "long"} else "sell",
                quantity=quantity,
                entry_price=price,
                cash_spent=cash_spent,
                opened_at=opened_at,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            self._logger.debug("Failed to normalise broker position %s: %s", payload, exc)
            return None
        return position

    @staticmethod
    def _coerce_open_timestamp(payload: Dict[str, Any]) -> datetime:
        """Best-effort conversion of broker timestamps to ``datetime``."""

        candidates = (
            payload.get("timestamp"),
            payload.get("datetime"),
            payload.get("info", {}).get("opentm"),
            payload.get("info", {}).get("time"),
        )
        for candidate in candidates:
            if candidate is None:
                continue
            if isinstance(candidate, datetime):
                return candidate
            if isinstance(candidate, (int, float)):
                try:
                    timestamp = float(candidate)
                except (TypeError, ValueError):
                    continue
                if timestamp > 1e12:
                    timestamp /= 1000.0
                try:
                    return datetime.utcfromtimestamp(timestamp)
                except (OverflowError, OSError, ValueError):
                    continue
            if isinstance(candidate, str):
                normalised = candidate.replace("Z", "+00:00")
                try:
                    parsed = datetime.fromisoformat(normalised)
                except ValueError:
                    continue
                if parsed.tzinfo is not None:
                    return parsed.astimezone(timezone.utc).replace(tzinfo=None)
                return parsed
        return datetime.utcnow()
