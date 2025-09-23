"""Kraken trading client wrapping ccxt."""

from __future__ import annotations

import asyncio
from decimal import Decimal, ROUND_DOWN
import random
from typing import Any, Callable, Dict, Optional, Tuple

import ccxt
from ccxt.base.errors import (
    DDoSProtection,
    ExchangeNotAvailable,
    NetworkError,
    RateLimitExceeded,
    RequestTimeout,
)

from ai_trader.services.logging import get_logger


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
        self._markets = markets or {}

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

    async def place_order(
        self,
        symbol: str,
        side: str,
        cash: float,
        *,
        reduce_only: bool | None = None,
    ) -> Tuple[float, float]:
        """Place a market order sized by cash value.

        Returns a tuple of (price, quantity).
        """

        price = await self.fetch_price(symbol)
        if price is None:
            raise RuntimeError(f"Unable to fetch price for {symbol}")
        price = float(price)
        cash_value = float(cash)
        amount = cash_value / price
        amount = self._adjust_amount(symbol, amount)

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
                amount,
                None,
                order_params,
                description=f"create_order:{symbol}",
            )
        except ccxt.BaseError as exc:  # pragma: no cover - network/broker failures
            self._logger.warning("Kraken rejected order for %s: %s", symbol, exc)
            raise RuntimeError(str(exc)) from exc
        filled = float(order.get("amount", amount))
        avg_price = order.get("average") or price
        return float(avg_price), float(filled)

    async def close_position(self, symbol: str, side: str, amount: float) -> Tuple[float, float]:
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
                amount,
                None,
                {"reduce_only": True},
                description=f"close_order:{symbol}",
            )
        except ccxt.BaseError as exc:  # pragma: no cover - network/broker failures
            self._logger.warning("Kraken rejected close order for %s: %s", symbol, exc)
            raise RuntimeError(str(exc)) from exc
        filled = float(order.get("amount", amount))
        avg_price = order.get("average") or order.get("price")
        if avg_price is None:
            market_price = await self.fetch_price(symbol)
            if market_price is None:
                raise RuntimeError("Unable to determine fill price")
            avg_price = market_price
        return float(avg_price), float(filled)

    async def compute_equity(self, prices: Dict[str, float]) -> tuple[float, Dict[str, float]]:
        """Return the total account equity alongside the raw balances."""

        balances = await self.fetch_balances()
        equity = float(balances.get(self._base_currency, 0.0))
        for asset, amount in balances.items():
            if asset == self._base_currency or amount == 0:
                continue
            symbol = f"{asset}/{self._base_currency}"
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
    def _apply_precision(amount: float, precision: int) -> float:
        quantize_str = "0." + "0" * (precision - 1) + "1" if precision > 0 else "1"
        return float(Decimal(amount).quantize(Decimal(quantize_str), rounding=ROUND_DOWN))

    def _simulate_fill(self, base: str, quote: str, side: str, amount: float, price: float) -> None:
        amount = float(amount)
        price = float(price)
        cost = amount * price
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
