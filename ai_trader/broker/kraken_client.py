"""Kraken trading client wrapping ccxt."""

from __future__ import annotations

import asyncio
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Optional, Tuple

import ccxt

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
        self._starting_equity = paper_starting_equity

    @property
    def starting_equity(self) -> float:
        return self._starting_equity

    async def load_markets(self) -> None:
        self._markets = await asyncio.to_thread(self._exchange.load_markets)

    async def fetch_price(self, symbol: str) -> Optional[float]:
        ticker = await asyncio.to_thread(self._exchange.fetch_ticker, symbol)
        await asyncio.sleep(self._rest_rate_limit)
        price = ticker.get("last") or ticker.get("close")
        return float(price) if price else None

    async def fetch_balances(self) -> Dict[str, float]:
        if self._paper_trading:
            return dict(self._paper_balances)
        balance = await asyncio.to_thread(self._exchange.fetch_balance)
        await asyncio.sleep(self._rest_rate_limit)
        total = balance.get("total", {})
        return {currency: float(amount) for currency, amount in total.items() if amount}

    async def place_order(self, symbol: str, side: str, cash: float) -> Tuple[float, float]:
        """Place a market order sized by cash value.

        Returns a tuple of (price, quantity).
        """

        price = await self.fetch_price(symbol)
        if price is None:
            raise RuntimeError(f"Unable to fetch price for {symbol}")
        amount = cash / price
        amount = self._adjust_amount(symbol, amount)

        if amount <= 0:
            raise RuntimeError("Calculated trade size is below Kraken minimum")

        base, quote = symbol.split("/")

        if self._paper_trading:
            self._simulate_fill(base, quote, side, amount, price)
            return price, amount

        try:
            order = await asyncio.to_thread(
                self._exchange.create_order,
                symbol,
                "market",
                side,
                amount,
                None,
                {"reduce_only": side == "sell"},
            )
        except ccxt.BaseError as exc:  # pragma: no cover - network/broker failures
            self._logger.warning("Kraken rejected order for %s: %s", symbol, exc)
            raise RuntimeError(str(exc)) from exc
        await asyncio.sleep(self._rest_rate_limit)
        filled = order.get("amount", amount)
        avg_price = order.get("average") or price
        return float(avg_price), float(filled)

    async def close_position(self, symbol: str, side: str, amount: float) -> Tuple[float, float]:
        exit_side = "sell" if side == "buy" else "buy"
        if self._paper_trading:
            price = await self.fetch_price(symbol)
            if price is None:
                raise RuntimeError("Missing market price for close")
            base, quote = symbol.split("/")
            self._simulate_fill(base, quote, exit_side, amount, price)
            return price, amount

        try:
            order = await asyncio.to_thread(
                self._exchange.create_order,
                symbol,
                "market",
                exit_side,
                amount,
                None,
                {"reduce_only": True},
            )
        except ccxt.BaseError as exc:  # pragma: no cover - network/broker failures
            self._logger.warning("Kraken rejected close order for %s: %s", symbol, exc)
            raise RuntimeError(str(exc)) from exc
        await asyncio.sleep(self._rest_rate_limit)
        filled = order.get("amount", amount)
        avg_price = order.get("average") or order.get("price")
        if avg_price is None:
            avg_price = await self.fetch_price(symbol)
        return float(avg_price), float(filled)

    async def compute_equity(self, prices: Dict[str, float]) -> tuple[float, Dict[str, float]]:
        """Return the total account equity alongside the raw balances."""

        balances = await self.fetch_balances()
        equity = balances.get(self._base_currency, 0.0)
        for asset, amount in balances.items():
            if asset == self._base_currency or amount == 0:
                continue
            symbol = f"{asset}/{self._base_currency}"
            price = prices.get(symbol)
            if price is None:
                price = await self.fetch_price(symbol)
                if price is None:
                    continue
            equity += amount * price
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
