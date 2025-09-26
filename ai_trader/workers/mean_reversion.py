"""Mean reversion worker."""

from __future__ import annotations

import statistics
from typing import Dict, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ai_trader.services.ml import MLService
    from ai_trader.services.trade_log import TradeLog

from ai_trader.services.types import MarketSnapshot, OpenPosition, TradeIntent
from ai_trader.workers.base import BaseWorker


class MeanReversionWorker(BaseWorker):
    """Trade deviations from a simple moving average using long-only entries."""

    name = "Mean Reverter"
    emoji = "🔄"
    long_only = True
    strategy_brief = (
        "Accumulating during pullbacks below the average and exiting as price mean-reverts upward."
    )

    def __init__(
        self,
        symbols,
        window: int = 20,
        threshold: float = 0.01,
        ml_service: "MLService" | None = None,
        trade_log: "TradeLog" | None = None,
    ) -> None:
        super().__init__(
            symbols=symbols,
            lookback=window * 3,
            ml_service=ml_service,
            trade_log=trade_log,
        )
        self.window = window
        self.threshold = threshold

    async def evaluate_signal(self, snapshot: MarketSnapshot) -> Dict[str, str]:
        self.update_history(snapshot)
        signals: Dict[str, str] = {}
        for symbol in self.symbols:
            history = list(self.price_history.get(symbol, []))
            signal: Optional[str] = None
            if len(history) < self.window:
                self.update_signal_state(symbol, signal, {"status": "warmup"})
                continue
            mean_price = statistics.fmean(history[-self.window :])
            current = history[-1]
            deviation = (current - mean_price) / mean_price
            if deviation < -self.threshold:
                signal = "buy"
            elif deviation > self.threshold and deviation < self.threshold * 2:
                signal = "exit"
            elif abs(deviation) < self.threshold / 3:
                signal = "hold"
            indicators = {
                "mean_price": mean_price,
                "last_price": current,
                "deviation": deviation,
                "threshold": self.threshold,
            }
            if self._ml_service is not None:
                indicators["ml_confidence"] = self._ml_service.latest_confidence(symbol, self.name)
            self.update_signal_state(symbol, signal, indicators)
            if signal:
                signals[symbol] = signal
        return signals

    async def generate_trade(
        self,
        symbol: str,
        signal: Optional[str],
        snapshot: MarketSnapshot,
        equity_per_trade: float,
        existing_position: Optional[OpenPosition] = None,
    ) -> Optional[TradeIntent]:
        if signal is None or signal == "hold":
            return None
        price = snapshot.prices.get(symbol)
        if price is None or price <= 0:
            return None

        if existing_position is not None:
            risk_triggered, risk_reason, risk_meta = self.check_risk_exit(symbol, price)
            if risk_triggered:
                self.update_signal_state(symbol, f"close:{risk_reason}")
                self.clear_risk_tracker(symbol)
                payload = {
                    "trigger": risk_reason,
                    **{k: v for k, v in risk_meta.items() if v is not None},
                }
                close_side = "sell" if existing_position.side == "buy" else "buy"
                return TradeIntent(
                    worker=self.name,
                    action="CLOSE",
                    symbol=symbol,
                    side=close_side,
                    cash_spent=existing_position.cash_spent,
                    entry_price=existing_position.entry_price,
                    exit_price=price,
                    confidence=0.0,
                    reason=risk_reason,
                    metadata=payload,
                )

        if signal == "buy" and existing_position is None:
            allowed, confidence = self.ml_confirmation(symbol)
            if not allowed:
                self.update_signal_state(symbol, "ml-block", {"ml_confidence": confidence})
                return None
            risk_meta = self.prepare_entry_risk(symbol, signal, price)
            metadata = {
                "signal": signal,
                "price": price,
                **{k: v for k, v in risk_meta.items() if v is not None},
            }
            cash_value = float(equity_per_trade)
            return TradeIntent(
                worker=self.name,
                action="OPEN",
                symbol=symbol,
                side=signal,
                cash_spent=cash_value,
                entry_price=price,
                confidence=confidence,
                metadata=metadata,
            )

        if (
            signal in {"exit", "sell"}
            and existing_position is not None
            and existing_position.side == "buy"
        ):
            pnl = (price - existing_position.entry_price) * existing_position.quantity
            base_cash = float(existing_position.cash_spent)
            pnl_percent = pnl / base_cash * 100 if base_cash else 0.0
            self.record_trade_event(
                "close_mean_revert_long",
                symbol,
                {
                    "pnl": pnl,
                    "pnl_percent": pnl_percent,
                    "side": existing_position.side,
                    **self.risk_snapshot(symbol),
                },
            )
            self.clear_risk_tracker(symbol)
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
                reason="mean-revert-exit",
                metadata={
                    "signal": signal,
                    "pnl": pnl,
                    "pnl_percent": pnl_percent,
                    **self.risk_snapshot(symbol),
                },
            )
        return None
