"""Mean reversion worker."""

from __future__ import annotations

import statistics
from typing import Dict, Optional

from .base import BaseWorker
from ..services.types import MarketSnapshot, OpenPosition, TradeIntent


class MeanReversionWorker(BaseWorker):
    """Trade deviations from a simple moving average."""

    name = "Mean Reverter"
    emoji = "🔄"

    def __init__(self, symbols, window: int = 20, threshold: float = 0.01) -> None:
        super().__init__(symbols=symbols, lookback=window * 3)
        self.window = window
        self.threshold = threshold

    async def evaluate_signal(self, snapshot: MarketSnapshot) -> Dict[str, str]:
        self.update_history(snapshot)
        signals: Dict[str, str] = {}
        for symbol in self.symbols:
            history = list(self.price_history.get(symbol, []))
            if len(history) < self.window:
                continue
            mean_price = statistics.fmean(history[-self.window :])
            current = history[-1]
            deviation = (current - mean_price) / mean_price
            if deviation > self.threshold:
                signals[symbol] = "sell"
            elif deviation < -self.threshold:
                signals[symbol] = "buy"
            elif abs(deviation) < self.threshold / 3:
                signals[symbol] = "flat"
        return signals

    async def generate_trade(
        self,
        symbol: str,
        signal: Optional[str],
        snapshot: MarketSnapshot,
        equity_per_trade: float,
        existing_position: Optional[OpenPosition] = None,
    ) -> Optional[TradeIntent]:
        if signal is None:
            return None
        price = snapshot.prices.get(symbol)
        if price is None or price <= 0:
            return None

        if signal in {"buy", "sell"} and existing_position is None:
            return TradeIntent(
                worker=self.name,
                action="OPEN",
                symbol=symbol,
                side=signal,
                cash_spent=equity_per_trade,
                entry_price=price,
            )

        if signal == "flat" and existing_position is not None:
            pnl = (
                (price - existing_position.entry_price)
                if existing_position.side == "buy"
                else (existing_position.entry_price - price)
            ) * existing_position.quantity
            pnl_percent = pnl / existing_position.cash_spent * 100 if existing_position.cash_spent else 0.0
            return TradeIntent(
                worker=self.name,
                action="CLOSE",
                symbol=symbol,
                side="sell" if existing_position.side == "buy" else "buy",
                cash_spent=existing_position.cash_spent,
                entry_price=existing_position.entry_price,
                exit_price=price,
                pnl_percent=pnl_percent,
                pnl_usd=pnl,
                win_loss="win" if pnl > 0 else "loss",
            )
        return None
