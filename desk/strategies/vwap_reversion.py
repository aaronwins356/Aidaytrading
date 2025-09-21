"""VWAP reversion strategy for intraday mean-reversion trades."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base_strategy import StrategyBase, Trade


def _compute_vwap(df: pd.DataFrame) -> pd.Series:
    price_volume = df["close"] * df["volume"]
    cumulative_pv = price_volume.cumsum()
    cumulative_volume = df["volume"].replace(0, np.nan).cumsum()
    vwap = cumulative_pv / cumulative_volume
    return vwap.fillna(method="ffill").fillna(df["close"])


class VWAPReversionStrategy(StrategyBase):
    def __init__(self, symbol: str, params: Optional[Dict[str, float]] = None) -> None:
        super().__init__(symbol, params)
        self._last_vwap: float = float("nan")
        self._last_diff: float = 0.0

    def generate_signals(self, df: pd.DataFrame) -> Optional[str]:
        if df.empty:
            return None
        threshold = float(self.params.get("threshold_pct", 0.02))
        self._vwap_series = _compute_vwap(df)
        vwap_value = float(self._vwap_series.iloc[-1])
        price = float(df["close"].iloc[-1])
        if not np.isfinite(vwap_value):
            return None
        diff_pct = (price - vwap_value) / vwap_value
        self._last_vwap = vwap_value
        self._last_diff = diff_pct
        if diff_pct >= threshold:
            return "sell"
        if diff_pct <= -threshold:
            return "buy"
        return None

    def plan_trade(self, side: str, df: pd.DataFrame) -> Dict[str, float]:
        price = float(df["close"].iloc[-1])
        vwap_value = float(getattr(self, "_vwap_series", _compute_vwap(df)).iloc[-1])
        stop_pct = float(self.params.get("stop_pct", 0.025))
        hold_minutes = float(self.params.get("max_hold_minutes", 90.0))
        if side.upper() == "BUY":
            stop_loss = price * (1 - stop_pct)
            take_profit = vwap_value
        else:
            stop_loss = price * (1 + stop_pct)
            take_profit = vwap_value
        return {
            "stop_loss": float(stop_loss),
            "take_profit": float(take_profit),
            "max_hold_minutes": hold_minutes,
            "metadata": {
                "vwap": vwap_value,
                "diff_pct": self._last_diff,
            },
        }

    def check_exit(self, trade: Trade, df: pd.DataFrame):
        price = float(df["close"].iloc[-1])
        vwap_value = float(_compute_vwap(df).iloc[-1])
        stop_pct = float(self.params.get("stop_pct", 0.025))
        side = self.trade_side(trade)

        if side == "BUY":
            if price <= trade.stop_loss:
                return True, "Stop beyond 2.5 pct"
            if price >= vwap_value:
                return True, "VWAP touched"
            if (price - trade.entry_price) / trade.entry_price <= -stop_pct:
                return True, "Adverse drift"
        elif side == "SELL":
            if price >= trade.stop_loss:
                return True, "Stop beyond 2.5 pct"
            if price <= vwap_value:
                return True, "VWAP touched"
            if (trade.entry_price - price) / trade.entry_price <= -stop_pct:
                return True, "Adverse drift"
        return False, None

    def extract_features(self, df: pd.DataFrame):
        vwap_value = float(getattr(self, "_vwap_series", _compute_vwap(df)).iloc[-1])
        price = float(df["close"].iloc[-1])
        diff_pct = (price - vwap_value) / vwap_value if vwap_value else 0.0
        return {
            "vwap": vwap_value,
            "price": price,
            "diff_pct": diff_pct,
        }
