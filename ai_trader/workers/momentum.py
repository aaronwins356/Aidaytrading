"""Simple momentum worker."""

from __future__ import annotations

from statistics import fmean
from typing import Dict, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ai_trader.services.ml import MLService
    from ai_trader.services.trade_log import TradeLog

from ai_trader.services.types import MarketSnapshot, OpenPosition, TradeIntent
from ai_trader.workers.base import BaseWorker


class MomentumWorker(BaseWorker):
    """EMA crossover style momentum worker."""

    name = "Momentum Rider"
    emoji = "⚡"

    def __init__(
        self,
        symbols,
        fast_window: int = 9,
        slow_window: int = 26,
        ml_service: "MLService" | None = None,
        trade_log: "TradeLog" | None = None,
    ) -> None:
        super().__init__(
            symbols=symbols,
            lookback=max(fast_window, slow_window) * 3,
            ml_service=ml_service,
            trade_log=trade_log,
        )
        self.fast_window = fast_window
        self.slow_window = slow_window

    async def evaluate_signal(self, snapshot: MarketSnapshot) -> Dict[str, str]:
        self.update_history(snapshot)
        signals: Dict[str, str] = {}
        for symbol in self.symbols:
            history = list(self.price_history.get(symbol, []))
            if len(history) < self.slow_window:
                continue
            fast_ma = fmean(history[-self.fast_window :])
            slow_ma = fmean(history[-self.slow_window :])
            last_price = history[-1]
            if fast_ma > slow_ma * 1.001:
                signals[symbol] = "buy"
            elif fast_ma < slow_ma * 0.999:
                signals[symbol] = "sell"
            elif abs(last_price - slow_ma) / slow_ma < 0.001:
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
        if signal is None or signal == "flat":
            return None

        price = snapshot.prices.get(symbol)
        if price is None or price <= 0:
            return None

        if signal in {"buy", "sell"} and existing_position is None:
            cash = equity_per_trade
            allowed, confidence = self.ml_confirmation(symbol)
            if not allowed:
                self.update_signal_state(symbol, "ml-block", {"ml_confidence": confidence})
                return None
            return TradeIntent(
                worker=self.name,
                action="OPEN",
                symbol=symbol,
                side=signal,
                cash_spent=cash,
                entry_price=price,
                confidence=confidence,
                metadata={"signal": signal, "price": price},
            )

        if signal == "sell" and existing_position and existing_position.side == "buy":
            pnl = (price - existing_position.entry_price) * existing_position.quantity
            pnl_percent = pnl / existing_position.cash_spent * 100 if existing_position.cash_spent else 0.0
            self.record_trade_event(
                "close_momentum_short",
                symbol,
                {"pnl": pnl, "pnl_percent": pnl_percent},
            )
            return TradeIntent(
                worker=self.name,
                action="CLOSE",
                symbol=symbol,
                side="sell",
                cash_spent=existing_position.cash_spent,
                entry_price=existing_position.entry_price,
                exit_price=price,
                pnl_percent=pnl_percent,
                pnl_usd=pnl,
                win_loss="win" if pnl > 0 else "loss",
                reason="bear-cross",
                metadata={"signal": signal, "pnl": pnl, "pnl_percent": pnl_percent},
            )

        if signal == "buy" and existing_position and existing_position.side == "sell":
            pnl = (existing_position.entry_price - price) * existing_position.quantity
            pnl_percent = pnl / existing_position.cash_spent * 100 if existing_position.cash_spent else 0.0
            self.record_trade_event(
                "close_momentum_cover",
                symbol,
                {"pnl": pnl, "pnl_percent": pnl_percent},
            )
            return TradeIntent(
                worker=self.name,
                action="CLOSE",
                symbol=symbol,
                side="buy",
                cash_spent=existing_position.cash_spent,
                entry_price=existing_position.entry_price,
                exit_price=price,
                pnl_percent=pnl_percent,
                pnl_usd=pnl,
                win_loss="win" if pnl > 0 else "loss",
                reason="bull-cover",
                metadata={"signal": signal, "pnl": pnl, "pnl_percent": pnl_percent},
            )

        return None
