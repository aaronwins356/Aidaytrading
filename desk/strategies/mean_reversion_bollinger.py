"""Mean reversion strategy using Bollinger Bands and RSI extremes."""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from .base_strategy import StrategyBase, Trade


class MeanReversionBollingerStrategy(StrategyBase):
    def __init__(self, symbol: str, params: Optional[Dict[str, float]] = None) -> None:
        super().__init__(symbol, params)
        self._context: Dict[str, float] = {}

    def generate_signals(self, df: pd.DataFrame) -> Optional[str]:
        period = int(self.params.get("period", 20))
        mult = float(self.params.get("mult", 2.0))
        if len(df) < period + 5:
            self._context = {}
            return None

        lower, mid, upper = self.bbands(df["close"], period, mult)
        rsi_series = self.rsi(df["close"], int(self.params.get("rsi_period", 14)))
        price = float(df["close"].iloc[-1])
        lower_val = float(lower.iloc[-1])
        upper_val = float(upper.iloc[-1])
        mid_val = float(mid.iloc[-1])
        rsi_val = float(rsi_series.iloc[-1])

        signal: Optional[str] = None
        if price <= lower_val and rsi_val <= float(self.params.get("rsi_long", 30.0)):
            signal = "buy"
        elif price >= upper_val and rsi_val >= float(self.params.get("rsi_short", 70.0)):
            signal = "sell"

        if signal:
            self._context = {
                "lower": lower_val,
                "upper": upper_val,
                "mid": mid_val,
                "rsi": rsi_val,
            }
        else:
            self._context = {}
        return signal

    def plan_trade(self, side: str, df: pd.DataFrame) -> Dict[str, float]:
        price = float(df["close"].iloc[-1])
        context = getattr(self, "_context", {})
        mid_val = float(context.get("mid", price))
        if side.upper() == "BUY":
            stop_loss = float(context.get("lower", price * 0.99))
            take_profit = mid_val
        else:
            stop_loss = float(context.get("upper", price * 1.01))
            take_profit = mid_val
        hold_minutes = float(self.params.get("max_hold_minutes", 180.0))
        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "max_hold_minutes": hold_minutes,
            "metadata": context,
        }

    def check_exit(self, trade: Trade, df: pd.DataFrame):
        lower, mid, upper = self.bbands(df["close"], int(self.params.get("period", 20)), float(self.params.get("mult", 2.0)))
        rsi_series = self.rsi(df["close"], int(self.params.get("rsi_period", 14)))
        price = float(df["close"].iloc[-1])
        mid_val = float(mid.iloc[-1])
        rsi_val = float(rsi_series.iloc[-1])

        side = self.trade_side(trade)

        if side == "BUY":
            if price <= trade.stop_loss:
                return True, "Bollinger stop"
            if price >= mid_val or rsi_val >= 50:
                return True, "Re-entered band"
        elif side == "SELL":
            if price >= trade.stop_loss:
                return True, "Bollinger stop"
            if price <= mid_val or rsi_val <= 50:
                return True, "Re-entered band"
        return False, None

    def extract_features(self, df: pd.DataFrame):
        context = getattr(self, "_context", {})
        return {
            "band_mid": float(context.get("mid", 0.0)),
            "band_upper": float(context.get("upper", 0.0)),
            "band_lower": float(context.get("lower", 0.0)),
            "rsi": float(context.get("rsi", 50.0)),
        }
