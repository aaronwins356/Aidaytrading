"""Momentum-driven short bias worker."""

from __future__ import annotations

import math
from statistics import pstdev
from typing import Dict, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ai_trader.services.ml import MLService

from ai_trader.services.types import MarketSnapshot, OpenPosition, TradeIntent
from ai_trader.workers.base import BaseWorker


class ShortMomentumWorker(BaseWorker):
    """Shorts accelerating breakdowns confirmed by momentum and volatility."""

    name = "Velocity Short"
    emoji = "🩳"

    def __init__(
        self,
        symbols,
        config: Optional[Dict] = None,
        risk_config: Optional[Dict] = None,
        ml_service: "MLService" | None = None,
    ) -> None:
        self.fast_window = int((config or {}).get("fast_window", 12))
        self.slow_window = int((config or {}).get("slow_window", 48))
        lookback = max(self.slow_window * 3, int((config or {}).get("lookback", self.slow_window * 3)))
        super().__init__(symbols=symbols, lookback=lookback, config=config, risk_config=risk_config, ml_service=ml_service)
        self.momentum_threshold = float((config or {}).get("momentum_threshold", 0.004))
        self._position_tracker: Dict[str, Dict[str, float]] = {}

    @staticmethod
    def _ema(prices: list[float], window: int) -> float:
        if not prices:
            return 0.0
        k = 2 / (window + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = price * k + ema * (1 - k)
        return ema

    async def evaluate_signal(self, snapshot: MarketSnapshot) -> Dict[str, str]:
        self.update_history(snapshot)
        signals: Dict[str, str] = {}
        for symbol in self.symbols:
            history = list(self.price_history.get(symbol, []))
            if len(history) < max(self.slow_window, self.warmup_candles):
                self.update_signal_state(symbol, None, {"status": "warmup"})
                continue

            fast = self._ema(history[-self.fast_window * 2 :], self.fast_window)
            slow = self._ema(history[-self.slow_window * 2 :], self.slow_window)
            last_price = history[-1]
            returns = [math.log(history[idx] / history[idx - 1]) for idx in range(1, len(history)) if history[idx - 1] > 0]
            vol = pstdev(returns[-self.slow_window :]) if len(returns) >= self.slow_window else 0.0
            momentum = slow - fast
            threshold = abs(slow) * self.momentum_threshold
            signal: Optional[str] = None

            if not self.is_ready(symbol):
                state = {
                    "status": "warmup",
                    "fast_ema": fast,
                    "slow_ema": slow,
                    "momentum": momentum,
                    "volatility": vol,
                    "price": last_price,
                }
                self.update_signal_state(symbol, signal, state)
                continue

            if fast < slow and abs(momentum) > threshold and last_price < slow:
                signal = "sell"
            elif fast > slow * (1.002):
                signal = "buy"

            indicators = {
                "fast_ema": fast,
                "slow_ema": slow,
                "momentum": momentum,
                "volatility": vol,
                "price": last_price,
                "threshold": threshold,
            }
            if self._ml_service:
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
        price = snapshot.prices.get(symbol)
        if price is None:
            return None

        tracker = self._position_tracker.setdefault(
            symbol,
            {
                "best_price": price,
            },
        )

        if existing_position is None:
            if signal != "sell" or not self.is_ready(symbol):
                return None
            cash = equity_per_trade * (self.position_size_pct / 100)
            cash = max(cash, 0.0)
            if cash == 0:
                return None
            allowed, ml_confidence = self.ml_confirmation(symbol)
            if not allowed:
                self.update_signal_state(symbol, "ml-block", {"ml_confidence": ml_confidence})
                return None
            tracker["best_price"] = price
            tracker["stop_price"] = price * (1 + self.stop_loss_pct / 100) if self.stop_loss_pct else None
            tracker["target_price"] = price * (1 - self.take_profit_pct / 100) if self.take_profit_pct else None
            tracker["trailing"] = price * (1 + self.trailing_stop_pct / 100) if self.trailing_stop_pct else None
            return TradeIntent(
                worker=self.name,
                action="OPEN",
                symbol=symbol,
                side="sell",
                cash_spent=cash * self.leverage,
                entry_price=price,
                confidence=ml_confidence or min(1.0, max(0.0, abs(tracker.get("stop_price", price) - price) / price)),
            )

        # Position management for an open short.
        tracker["best_price"] = min(tracker.get("best_price", price), price)
        if self.trailing_stop_pct:
            tracker["trailing"] = tracker["best_price"] * (1 + self.trailing_stop_pct / 100)

        stop_price = (
            existing_position.entry_price * (1 + self.stop_loss_pct / 100)
            if self.stop_loss_pct
            else tracker.get("stop_price")
        )
        target_price = (
            existing_position.entry_price * (1 - self.take_profit_pct / 100)
            if self.take_profit_pct
            else tracker.get("target_price")
        )
        trailing_price = tracker.get("trailing")

        close_signal = False
        reason = ""
        if signal == "buy":
            close_signal = True
            reason = "reverse"
        if stop_price and price >= stop_price:
            close_signal = True
            reason = "stop"
        if target_price and price <= target_price:
            close_signal = True
            reason = reason or "target"
        if trailing_price and price >= trailing_price:
            close_signal = True
            reason = reason or "trail"

        if close_signal:
            tracker["best_price"] = price
            self.update_signal_state(symbol, f"close:{reason}")
            return TradeIntent(
                worker=self.name,
                action="CLOSE",
                symbol=symbol,
                side="buy",
                cash_spent=existing_position.cash_spent,
                entry_price=existing_position.entry_price,
                exit_price=price,
                confidence=0.7,
            )

        return None
