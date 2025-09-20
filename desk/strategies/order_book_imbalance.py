"""Scalping strategy using order book bid/ask imbalance signals."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base_strategy import StrategyBase, Trade


class OrderBookImbalanceStrategy(StrategyBase):
    def __init__(self, symbol: str, params: Optional[Dict[str, float]] = None) -> None:
        super().__init__(symbol, params)
        self._context: Dict[str, float] = {}

    def _imbalance_series(self, df: pd.DataFrame) -> Optional[pd.Series]:
        bid_col = self.params.get("bid_col", "bid_imbalance")
        ask_col = self.params.get("ask_col", "ask_imbalance")
        if bid_col in df.columns and ask_col in df.columns:
            return df[bid_col] - df[ask_col]
        if "buy_volume" in df.columns and "sell_volume" in df.columns:
            total = df["buy_volume"] + df["sell_volume"]
            total = total.replace(0, np.nan)
            return (df["buy_volume"] - df["sell_volume"]) / total
        return None

    def generate_signals(self, df: pd.DataFrame) -> Optional[str]:
        if len(df) < 5:
            self._context = {}
            return None
        imbalance_series = self._imbalance_series(df)
        if imbalance_series is None:
            self._context = {}
            return None
        window = int(self.params.get("lookback", 50))
        threshold_mult = float(self.params.get("threshold_mult", 3.0))
        recent = imbalance_series.iloc[-window:]
        baseline = float(recent.abs().median()) if len(recent) else 0.0
        latest = float(imbalance_series.iloc[-1])
        if baseline <= 0:
            self._context = {}
            return None
        signal: Optional[str] = None
        if latest >= baseline * threshold_mult:
            signal = "buy"
        elif latest <= -baseline * threshold_mult:
            signal = "sell"
        if signal:
            self._context = {
                "imbalance": latest,
                "baseline": baseline,
            }
        else:
            self._context = {}
        return signal

    def plan_trade(self, side: str, df: pd.DataFrame) -> Dict[str, float]:
        price = float(df["close"].iloc[-1])
        target_pct = float(self.params.get("target_pct", 0.003))
        stop_pct = float(self.params.get("stop_pct", 0.002))
        hold_minutes = float(self.params.get("max_hold_minutes", 10.0))
        if side.upper() == "BUY":
            stop_loss = price * (1 - stop_pct)
            take_profit = price * (1 + target_pct)
        else:
            stop_loss = price * (1 + stop_pct)
            take_profit = price * (1 - target_pct)
        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "max_hold_minutes": hold_minutes,
            "metadata": self._context,
        }

    def check_exit(self, trade: Trade, df: pd.DataFrame):
        price = float(df["close"].iloc[-1])
        target_pct = float(self.params.get("target_pct", 0.003))
        if trade.side == "buy":
            if price <= trade.stop_loss:
                return True, "Scalp stop"
            if price >= trade.entry_price * (1 + target_pct):
                return True, "Hit scalp target"
        else:
            if price >= trade.stop_loss:
                return True, "Scalp stop"
            if price <= trade.entry_price * (1 - target_pct):
                return True, "Hit scalp target"
        return False, None

    def extract_features(self, df: pd.DataFrame):
        context = getattr(self, "_context", {})
        return {
            "imbalance": float(context.get("imbalance", 0.0)),
            "imbalance_baseline": float(context.get("baseline", 0.0)),
        }
