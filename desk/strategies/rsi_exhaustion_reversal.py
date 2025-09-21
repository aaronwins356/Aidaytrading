"""RSI exhaustion reversal strategy for intraday extremes."""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from .base_strategy import StrategyBase, Trade


class RSIExhaustionReversalStrategy(StrategyBase):
    def __init__(self, symbol: str, params: Optional[Dict[str, float]] = None) -> None:
        super().__init__(symbol, params)
        self._context: Dict[str, float] = {}

    def _streak_length(self, df: pd.DataFrame, direction: str) -> int:
        streak = 0
        closes = df["close"].values
        opens = df["open"].values
        for close, open_ in zip(closes[::-1], opens[::-1]):
            if direction == "up" and close >= open_:
                streak += 1
            elif direction == "down" and close <= open_:
                streak += 1
            else:
                break
        return streak

    def generate_signals(self, df: pd.DataFrame) -> Optional[str]:
        if len(df) < 6:
            self._context = {}
            return None
        rsi_period = int(self.params.get("rsi_period", 14))
        rsi_series = self.rsi(df["close"], rsi_period)
        rsi_val = float(rsi_series.iloc[-1])
        up_streak = self._streak_length(df.iloc[-10:], "up")
        down_streak = self._streak_length(df.iloc[-10:], "down")
        signal: Optional[str] = None
        if rsi_val >= float(self.params.get("rsi_short", 85.0)) and up_streak >= int(self.params.get("min_streak", 5)):
            signal = "sell"
        elif rsi_val <= float(self.params.get("rsi_long", 15.0)) and down_streak >= int(self.params.get("min_streak", 5)):
            signal = "buy"
        if signal:
            self._context = {
                "rsi": rsi_val,
                "up_streak": up_streak,
                "down_streak": down_streak,
            }
        else:
            self._context = {}
        return signal

    def plan_trade(self, side: str, df: pd.DataFrame) -> Dict[str, float]:
        price = float(df["close"].iloc[-1])
        stop_pct = float(self.params.get("stop_pct", 0.015))
        hold_minutes = float(self.params.get("max_hold_minutes", 90.0))
        if side.upper() == "BUY":
            stop_loss = price * (1 - stop_pct)
            take_profit = price * (1 + stop_pct)
        else:
            stop_loss = price * (1 + stop_pct)
            take_profit = price * (1 - stop_pct)
        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "max_hold_minutes": hold_minutes,
            "metadata": self._context,
        }

    def check_exit(self, trade: Trade, df: pd.DataFrame):
        rsi_series = self.rsi(df["close"], int(self.params.get("rsi_period", 14)))
        rsi_val = float(rsi_series.iloc[-1])
        price = float(df["close"].iloc[-1])
        side = self.trade_side(trade)

        if side == "BUY":
            if price <= trade.stop_loss:
                return True, "Stop"
            if rsi_val >= 50:
                return True, "RSI normalized"
        elif side == "SELL":
            if price >= trade.stop_loss:
                return True, "Stop"
            if rsi_val <= 50:
                return True, "RSI normalized"
        return False, None

    def extract_features(self, df: pd.DataFrame):
        context = getattr(self, "_context", {})
        return {
            "rsi": float(context.get("rsi", 50.0)),
            "up_streak": float(context.get("up_streak", 0.0)),
            "down_streak": float(context.get("down_streak", 0.0)),
        }
