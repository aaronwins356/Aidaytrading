"""Range bound oscillation strategy for sideways markets."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base_strategy import StrategyBase, Trade


class RangeBoundOscillationStrategy(StrategyBase):
    def __init__(self, symbol: str, params: Optional[Dict[str, float]] = None) -> None:
        super().__init__(symbol, params)
        self._range: Dict[str, float] = {}

    def _establish_range(self, df: pd.DataFrame) -> Optional[Dict[str, float]]:
        setup_bars = int(self.params.get("range_setup_bars", 120))
        if len(df) < setup_bars:
            return None
        initial = df.iloc[:setup_bars]
        high = float(initial["high"].max())
        low = float(initial["low"].min())
        if not np.isfinite(high) or not np.isfinite(low) or high <= low:
            return None
        return {"high": high, "low": low, "mid": (high + low) / 2.0}

    def generate_signals(self, df: pd.DataFrame) -> Optional[str]:
        tolerance = float(self.params.get("tolerance", 0.002))
        range_context = self._establish_range(df)
        if range_context is None:
            self._range = {}
            return None
        price = float(df["close"].iloc[-1])
        latest = df.iloc[-1]
        open_price = float(latest["open"])
        high = range_context["high"]
        low = range_context["low"]
        signal: Optional[str] = None
        if price <= low * (1 + tolerance) and price >= low:
            if price > open_price:
                signal = "buy"
        elif price >= high * (1 - tolerance) and price <= high:
            if price < open_price:
                signal = "sell"
        if signal:
            range_context["tolerance"] = tolerance
            self._range = range_context
        else:
            self._range = {}
        return signal

    def plan_trade(self, side: str, df: pd.DataFrame) -> Dict[str, float]:
        price = float(df["close"].iloc[-1])
        context = getattr(self, "_range", {})
        high = float(context.get("high", price * 1.01))
        low = float(context.get("low", price * 0.99))
        mid = float(context.get("mid", price))
        hold_minutes = float(self.params.get("max_hold_minutes", 180.0))
        buffer = float(self.params.get("buffer_pct", 0.0015))
        if side.upper() == "BUY":
            stop_loss = low * (1 - buffer)
            take_profit = mid
        else:
            stop_loss = high * (1 + buffer)
            take_profit = mid
        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "max_hold_minutes": hold_minutes,
            "metadata": context,
        }

    def check_exit(self, trade: Trade, df: pd.DataFrame):
        price = float(df["close"].iloc[-1])
        context = getattr(self, "_range", {})
        mid = float(context.get("mid", (trade.take_profit + trade.stop_loss) / 2))
        high = float(context.get("high", trade.take_profit))
        low = float(context.get("low", trade.stop_loss))
        side = self.trade_side(trade)

        if side == "BUY":
            if price <= trade.stop_loss:
                return True, "Range floor broken"
            if price >= mid:
                return True, "Mid-range exit"
            if price >= high:
                return True, "Breakout abort"
        elif side == "SELL":
            if price >= trade.stop_loss:
                return True, "Range ceiling broken"
            if price <= mid:
                return True, "Mid-range exit"
            if price <= low:
                return True, "Breakdown abort"
        return False, None

    def extract_features(self, df: pd.DataFrame):
        context = getattr(self, "_range", {})
        return {
            "range_high": float(context.get("high", 0.0)),
            "range_low": float(context.get("low", 0.0)),
            "range_mid": float(context.get("mid", 0.0)),
        }
