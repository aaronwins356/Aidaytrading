"""Momentum-driven short bias worker."""

from __future__ import annotations

import math
from statistics import pstdev
from typing import Dict, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ai_trader.services.ml import MLService
    from ai_trader.services.trade_log import TradeLog

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
        trade_log: "TradeLog" | None = None,
    ) -> None:
        self.fast_window = int((config or {}).get("fast_window", 12))
        self.slow_window = int((config or {}).get("slow_window", 48))
        lookback = max(self.slow_window * 3, int((config or {}).get("lookback", self.slow_window * 3)))
        super().__init__(
            symbols=symbols,
            lookback=lookback,
            config=config,
            risk_config=risk_config,
            ml_service=ml_service,
            trade_log=trade_log,
        )
        self.momentum_threshold = float((config or {}).get("momentum_threshold", 0.004))

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

        if existing_position is not None:
            risk_triggered, risk_reason, risk_meta = self.check_risk_exit(symbol, price)
            if risk_triggered:
                self.update_signal_state(symbol, f"close:{risk_reason}")
                self.clear_risk_tracker(symbol)
                payload = {
                    "trigger": risk_reason,
                    **{k: v for k, v in risk_meta.items() if v is not None},
                }
                return TradeIntent(
                    worker=self.name,
                    action="CLOSE",
                    symbol=symbol,
                    side="buy",
                    cash_spent=existing_position.cash_spent,
                    entry_price=existing_position.entry_price,
                    exit_price=price,
                    confidence=0.0,
                    reason=risk_reason,
                    metadata=payload,
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
            risk_meta = self.prepare_entry_risk(symbol, "sell", price)
            metadata = {k: v for k, v in risk_meta.items() if v is not None}
            fallback_confidence = 0.0
            stop_hint = risk_meta.get("stop_price")
            if stop_hint is not None:
                fallback_confidence = min(1.0, abs(stop_hint - price) / max(price, 1e-9))
            confidence = ml_confidence or fallback_confidence
            self.record_trade_event("open_signal", symbol, metadata)
            return TradeIntent(
                worker=self.name,
                action="OPEN",
                symbol=symbol,
                side="sell",
                cash_spent=cash * self.leverage,
                entry_price=price,
                confidence=confidence,
                metadata=metadata,
            )

        close_signal = False
        reason = ""
        if signal == "buy":
            close_signal = True
            reason = "reverse"

        if close_signal:
            self.update_signal_state(symbol, f"close:{reason}")
            metadata = {**self.risk_snapshot(symbol), "trigger": reason}
            if reason in {"stop", "target", "trail"}:
                self.record_trade_event(f"risk_{reason}", symbol, metadata)
            self.clear_risk_tracker(symbol)
            return TradeIntent(
                worker=self.name,
                action="CLOSE",
                symbol=symbol,
                side="buy",
                cash_spent=existing_position.cash_spent,
                entry_price=existing_position.entry_price,
                exit_price=price,
                confidence=0.7,
                reason=reason,
                metadata=metadata,
            )

        return None
