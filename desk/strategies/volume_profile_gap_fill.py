"""Volume profile gap fill strategy."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .base_strategy import StrategyBase, Trade


class VolumeProfileGapFillStrategy(StrategyBase):
    def __init__(self, symbol: str, params: Optional[Dict[str, float]] = None) -> None:
        super().__init__(symbol, params)
        self._context: Dict[str, float] = {}

    def _volume_profile(self, df: pd.DataFrame) -> Tuple[pd.IntervalIndex, np.ndarray]:
        bins = int(self.params.get("bins", 30))
        prices = df["close"].values
        volumes = df["volume"].values
        if len(prices) < 2 or np.all(volumes == 0):
            raise ValueError("Insufficient data for volume profile")
        hist, edges = np.histogram(prices, bins=bins, weights=volumes)
        intervals = pd.IntervalIndex.from_breaks(edges)
        return intervals, hist

    def generate_signals(self, df: pd.DataFrame) -> Optional[str]:
        if len(df) < int(self.params.get("min_bars", 60)):
            self._context = {}
            return None
        try:
            intervals, hist = self._volume_profile(df)
        except ValueError:
            self._context = {}
            return None
        hist_series = pd.Series(hist, index=intervals)
        positive_hist = hist_series[hist_series > 0]
        if positive_hist.empty:
            self._context = {}
            return None
        quantile = float(self.params.get("volume_quantile", 0.25))
        threshold = float(positive_hist.quantile(quantile))
        price = float(df["close"].iloc[-1])
        previous_price = float(df["close"].iloc[-2])
        current_interval = intervals[intervals.contains(price)]
        if len(current_interval) == 0:
            self._context = {}
            return None
        idx = intervals.get_loc(current_interval[0])
        volume_here = float(hist[idx])
        if volume_here > threshold:
            self._context = {}
            return None
        zone_low = float(current_interval[0].left)
        zone_high = float(current_interval[0].right)
        signal: Optional[str] = None
        if previous_price < price:
            signal = "buy"
        elif previous_price > price:
            signal = "sell"
        else:
            self._context = {}
            return None
        self._context = {
            "zone_low": zone_low,
            "zone_high": zone_high,
            "volume_score": volume_here,
            "threshold": threshold,
        }
        return signal

    def plan_trade(self, side: str, df: pd.DataFrame) -> Dict[str, float]:
        price = float(df["close"].iloc[-1])
        context = getattr(self, "_context", {})
        zone_low = float(context.get("zone_low", price * 0.99))
        zone_high = float(context.get("zone_high", price * 1.01))
        atr_period = int(self.params.get("atr_period", 14))
        atr_series = self.atr(df, atr_period)
        atr_value = float(atr_series.iloc[-1]) if len(atr_series.dropna()) else price * 0.01
        buffer = max(atr_value * float(self.params.get("buffer_mult", 0.5)), price * 0.001)
        hold_minutes = float(self.params.get("max_hold_minutes", 180.0))
        if side.upper() == "BUY":
            stop_loss = zone_low - buffer
            take_profit = zone_high
        else:
            stop_loss = zone_high + buffer
            take_profit = zone_low
        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "max_hold_minutes": hold_minutes,
            "metadata": context,
        }

    def check_exit(self, trade: Trade, df: pd.DataFrame):
        price = float(df["close"].iloc[-1])
        context = getattr(self, "_context", {})
        zone_low = float(context.get("zone_low", trade.stop_loss))
        zone_high = float(context.get("zone_high", trade.take_profit))
        side = self.trade_side(trade)

        if side == "BUY":
            if price <= trade.stop_loss:
                return True, "Gap stop"
            if price >= zone_high:
                return True, "Gap filled"
        elif side == "SELL":
            if price >= trade.stop_loss:
                return True, "Gap stop"
            if price <= zone_low:
                return True, "Gap filled"
        return False, None

    def extract_features(self, df: pd.DataFrame):
        context = getattr(self, "_context", {})
        return {
            "zone_low": float(context.get("zone_low", 0.0)),
            "zone_high": float(context.get("zone_high", 0.0)),
            "volume_score": float(context.get("volume_score", 0.0)),
            "threshold": float(context.get("threshold", 0.0)),
        }
