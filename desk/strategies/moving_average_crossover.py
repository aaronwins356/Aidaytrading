"""EMA(9/21) crossover trend-following strategy."""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from .base_strategy import StrategyBase, Trade


class MovingAverageCrossoverStrategy(StrategyBase):
    def __init__(self, symbol: str, params: Optional[Dict[str, float]] = None) -> None:
        super().__init__(symbol, params)
        self._last_cross: Optional[str] = None

    def _ema_values(self, df: pd.DataFrame) -> Dict[str, float]:
        fast = int(self.params.get("fast", 9))
        slow = int(self.params.get("slow", 21))
        ema_fast = self.ema(df["close"], fast)
        ema_slow = self.ema(df["close"], slow)
        return {
            "ema_fast": float(ema_fast.iloc[-1]),
            "ema_slow": float(ema_slow.iloc[-1]),
            "prev_fast": float(ema_fast.iloc[-2]) if len(df) > 1 else float("nan"),
            "prev_slow": float(ema_slow.iloc[-2]) if len(df) > 1 else float("nan"),
        }

    def generate_signals(self, df: pd.DataFrame) -> Optional[str]:
        if len(df) < max(int(self.params.get("fast", 9)), int(self.params.get("slow", 21))) + 2:
            return None
        ema_vals = self._ema_values(df)
        prev_fast = ema_vals["prev_fast"]
        prev_slow = ema_vals["prev_slow"]
        fast_now = ema_vals["ema_fast"]
        slow_now = ema_vals["ema_slow"]
        signal = None
        if prev_fast <= prev_slow and fast_now > slow_now:
            signal = "buy"
        elif prev_fast >= prev_slow and fast_now < slow_now:
            signal = "sell"
        self._last_cross = signal
        self._ema_context = ema_vals
        return signal

    def plan_trade(self, side: str, df: pd.DataFrame) -> Dict[str, float]:
        price = float(df["close"].iloc[-1])
        stop_pct = float(self.params.get("stop_pct", 0.01))
        target_pct = float(self.params.get("target_pct", 0.015))
        hold_minutes = float(self.params.get("max_hold_minutes", 240.0))
        if side.upper() == "BUY":
            stop_loss = price * (1 - stop_pct)
            take_profit = price * (1 + target_pct)
        else:
            stop_loss = price * (1 + stop_pct)
            take_profit = price * (1 - target_pct)
        context = getattr(self, "_ema_context", {})
        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "max_hold_minutes": hold_minutes,
            "metadata": context,
        }

    def check_exit(self, trade: Trade, df: pd.DataFrame):
        ema_vals = self._ema_values(df)
        price = float(df["close"].iloc[-1])
        stop_pct = float(self.params.get("stop_pct", 0.01))
        side = self.trade_side(trade)

        if side == "BUY":
            if price <= trade.stop_loss:
                return True, "1% stop"
            if ema_vals["ema_fast"] < ema_vals["ema_slow"]:
                return True, "Bearish cross"
        elif side == "SELL":
            if price >= trade.stop_loss:
                return True, "1% stop"
            if ema_vals["ema_fast"] > ema_vals["ema_slow"]:
                return True, "Bullish cross"
        return False, None

    def extract_features(self, df: pd.DataFrame):
        ema_vals = getattr(self, "_ema_context", self._ema_values(df))
        return {
            "ema_fast": float(ema_vals.get("ema_fast", 0.0)),
            "ema_slow": float(ema_vals.get("ema_slow", 0.0)),
        }
