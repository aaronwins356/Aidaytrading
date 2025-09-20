"""Liquidity grab / stop hunt fade strategy."""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from .base_strategy import StrategyBase, Trade


class LiquidityGrabStrategy(StrategyBase):
    def __init__(self, symbol: str, params: Optional[Dict[str, float]] = None) -> None:
        super().__init__(symbol, params)
        self._context: Dict[str, float] = {}

    def _swing_levels(self, df: pd.DataFrame, lookback: int) -> Optional[Dict[str, float]]:
        if len(df) <= lookback:
            return None
        window = df.iloc[-(lookback + 1) : -1]
        if window.empty:
            return None
        return {
            "recent_high": float(window["high"].max()),
            "recent_low": float(window["low"].min()),
        }

    def generate_signals(self, df: pd.DataFrame) -> Optional[str]:
        lookback = int(self.params.get("lookback", 40))
        volume_mult = float(self.params.get("volume_multiplier", 1.1))
        context = self._swing_levels(df, lookback)
        if context is None:
            self._context = {}
            return None

        window = df.iloc[-(lookback + 1) : -1]
        avg_volume = float(window["volume"].mean()) if len(window) else 0.0
        latest = df.iloc[-1]
        price = float(latest["close"])
        high = float(latest["high"])
        low = float(latest["low"])
        volume = float(latest["volume"])
        volume_ratio = volume / avg_volume if avg_volume else 0.0

        signal: Optional[str] = None
        if high > context["recent_high"] and price < context["recent_high"] and volume_ratio <= volume_mult:
            signal = "sell"
        elif low < context["recent_low"] and price > context["recent_low"] and volume_ratio <= volume_mult:
            signal = "buy"

        if signal:
            prior_low = float(df.iloc[-lookback:]["low"].min())
            prior_high = float(df.iloc[-lookback:]["high"].max())
            prior_mid = (prior_high + prior_low) / 2.0
            self._context = {
                **context,
                "prior_mid": prior_mid,
                "volume_ratio": volume_ratio,
                "high": high,
                "low": low,
            }
        else:
            self._context = {}
        return signal

    def plan_trade(self, side: str, df: pd.DataFrame) -> Dict[str, float]:
        context = getattr(self, "_context", {})
        price = float(df["close"].iloc[-1])
        buffer_pct = float(self.params.get("buffer_pct", 0.001))
        prior_mid = float(context.get("prior_mid", price))
        if side.upper() == "BUY":
            stop_loss = float(context.get("low", price)) * (1 - buffer_pct)
            take_profit = prior_mid
        else:
            stop_loss = float(context.get("high", price)) * (1 + buffer_pct)
            take_profit = prior_mid
        hold_minutes = float(self.params.get("max_hold_minutes", 120.0))
        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "max_hold_minutes": hold_minutes,
            "metadata": context,
        }

    def check_exit(self, trade: Trade, df: pd.DataFrame):
        price = float(df["close"].iloc[-1])
        context = getattr(self, "_context", {})
        prior_mid = float(context.get("prior_mid", trade.entry_price))
        if trade.side == "buy":
            if price <= trade.stop_loss:
                return True, "Liquidity sweep extended"
            if price >= prior_mid:
                return True, "Range mean achieved"
        else:
            if price >= trade.stop_loss:
                return True, "Liquidity sweep extended"
            if price <= prior_mid:
                return True, "Range mean achieved"
        return False, None

    def extract_features(self, df: pd.DataFrame):
        context = getattr(self, "_context", {})
        return {
            "recent_high": float(context.get("recent_high", 0.0)),
            "recent_low": float(context.get("recent_low", 0.0)),
            "volume_ratio": float(context.get("volume_ratio", 0.0)),
        }
