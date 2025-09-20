"""Breakout strategy based on consolidation range + volume confirmation."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base_strategy import StrategyBase, Trade


class BreakoutStrategy(StrategyBase):
    def __init__(self, symbol: str, params: Optional[Dict[str, float]] = None) -> None:
        super().__init__(symbol, params)
        self._context: Dict[str, float] = {}

    def _range_context(self, df: pd.DataFrame, lookback: int) -> Optional[Dict[str, float]]:
        if len(df) <= lookback:
            return None
        window = df.iloc[-(lookback + 1) : -1]
        if window.empty:
            return None
        range_high = float(window["high"].max())
        range_low = float(window["low"].min())
        if not np.isfinite(range_high) or not np.isfinite(range_low):
            return None
        mid = (range_high + range_low) / 2.0
        atr_period = int(self.params.get("atr_period", 14))
        atr_series = self.atr(df, atr_period)
        atr_value = float(atr_series.iloc[-1]) if len(atr_series.dropna()) else float("nan")
        return {
            "range_high": range_high,
            "range_low": range_low,
            "mid": mid,
            "atr": atr_value,
        }

    def generate_signals(self, df: pd.DataFrame) -> Optional[str]:
        lookback = int(self.params.get("lookback", 30))
        volume_mult = float(self.params.get("volume_multiplier", 1.2))
        context = self._range_context(df, lookback)
        if context is None:
            self._context = {}
            return None

        latest = df.iloc[-1]
        window = df.iloc[-(lookback + 1) : -1]
        avg_volume = float(window["volume"].mean()) if len(window) else 0.0
        if avg_volume <= 0:
            self._context = {}
            return None

        volume_ratio = float(latest["volume"]) / avg_volume if avg_volume else 0.0
        price = float(latest["close"])

        signal: Optional[str] = None
        if price > context["range_high"] and volume_ratio >= volume_mult:
            signal = "buy"
        elif price < context["range_low"] and volume_ratio >= volume_mult:
            signal = "sell"

        if signal:
            context.update({
                "volume_ratio": volume_ratio,
                "signal_price": price,
            })
            self._context = context
        else:
            self._context = {}
        return signal

    def plan_trade(self, side: str, df: pd.DataFrame) -> Dict[str, float]:
        context = getattr(self, "_context", {}) or {}
        price = float(df["close"].iloc[-1])
        atr = float(context.get("atr", np.nan))
        if not np.isfinite(atr) or atr <= 0:
            atr = float(self.atr(df, int(self.params.get("atr_period", 14))).iloc[-1])
            if not np.isfinite(atr) or atr <= 0:
                atr = price * 0.01

        target_multiple = float(self.params.get("target_multiple", 1.75))
        range_mid = float(context.get("mid", price * 0.99))

        if side.upper() == "BUY":
            stop_loss = range_mid
            take_profit = price + atr * target_multiple
        else:
            stop_loss = range_mid
            take_profit = price - atr * target_multiple

        hold_minutes = float(self.params.get("max_hold_minutes", 240.0))
        return {
            "stop_loss": float(stop_loss),
            "take_profit": float(take_profit),
            "max_hold_minutes": hold_minutes,
            "metadata": {
                "range_high": context.get("range_high", 0.0),
                "range_low": context.get("range_low", 0.0),
                "range_mid": range_mid,
                "atr": atr,
            },
        }

    def check_exit(self, trade: Trade, df: pd.DataFrame):
        price = float(df["close"].iloc[-1])
        atr = self.atr(df, int(self.params.get("atr_period", 14)))
        atr_val = float(atr.iloc[-1]) if len(atr.dropna()) else price * 0.01
        target_multiple = float(self.params.get("target_multiple", 1.75))

        if trade.side == "buy":
            if price <= trade.stop_loss:
                return True, "Range mid stop"
            if price >= trade.entry_price + atr_val * target_multiple:
                return True, "ATR target"
            if price < trade.entry_price and price < trade.meta.get("range_high", trade.entry_price):
                return True, "Failed breakout"
        else:
            if price >= trade.stop_loss:
                return True, "Range mid stop"
            if price <= trade.entry_price - atr_val * target_multiple:
                return True, "ATR target"
            if price > trade.entry_price and price > trade.meta.get("range_low", trade.entry_price):
                return True, "Failed breakdown"
        return False, None

    def extract_features(self, df: pd.DataFrame):
        context = getattr(self, "_context", {}) or {}
        lookback = int(self.params.get("lookback", 30))
        window = df.iloc[-(lookback + 1) : -1]
        volume_ratio = context.get("volume_ratio", 0.0)
        range_width = context.get("range_high", 0.0) - context.get("range_low", 0.0)
        return {
            "range_width": float(range_width),
            "volume_ratio": float(volume_ratio),
            "lookback": float(lookback),
        }
