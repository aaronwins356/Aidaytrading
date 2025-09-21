"""Momentum ignition strategy to capture liquidation-driven spikes."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base_strategy import StrategyBase, Trade


class MomentumIgnitionStrategy(StrategyBase):
    def __init__(self, symbol: str, params: Optional[Dict[str, float]] = None) -> None:
        super().__init__(symbol, params)
        self._context: Dict[str, float] = {}

    def generate_signals(self, df: pd.DataFrame) -> Optional[str]:
        if len(df) < 3:
            self._context = {}
            return None

        volume_lookback = int(self.params.get("volume_lookback", 30))
        volume_mult = float(self.params.get("volume_multiplier", 1.5))
        min_move = float(self.params.get("min_move_pct", 0.01))

        window = df.iloc[-volume_lookback:]
        avg_volume = float(window["volume"].mean()) if len(window) else 0.0
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        move = (float(latest["close"]) - float(latest["open"])) / max(float(latest["open"]), 1e-9)
        volume_ratio = float(latest["volume"]) / avg_volume if avg_volume else 0.0

        if abs(move) < min_move or volume_ratio < volume_mult:
            self._context = {}
            return None

        direction = "buy" if move > 0 else "sell"
        self._context = {
            "move": move,
            "volume_ratio": volume_ratio,
            "prev_high": float(prev["high"]),
            "prev_low": float(prev["low"]),
        }
        return direction

    def plan_trade(self, side: str, df: pd.DataFrame) -> Dict[str, float]:
        price = float(df["close"].iloc[-1])
        target_pct = float(self.params.get("target_pct", 0.0075))
        context = getattr(self, "_context", {})
        if side.upper() == "BUY":
            stop_loss = float(context.get("prev_low", price * (1 - 0.01)))
            take_profit = price * (1 + target_pct)
        else:
            stop_loss = float(context.get("prev_high", price * (1 + 0.01)))
            take_profit = price * (1 - target_pct)
        hold_minutes = float(self.params.get("max_hold_minutes", 15.0))
        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "max_hold_minutes": hold_minutes,
            "metadata": {
                "move": context.get("move", 0.0),
                "volume_ratio": context.get("volume_ratio", 0.0),
            },
        }

    def check_exit(self, trade: Trade, df: pd.DataFrame):
        price = float(df["close"].iloc[-1])
        target_pct = float(self.params.get("target_pct", 0.0075))
        side = self.trade_side(trade)

        if side == "BUY":
            if price <= trade.stop_loss:
                return True, "Broke prior low"
            if price >= trade.entry_price * (1 + target_pct):
                return True, "Momentum target"
        elif side == "SELL":
            if price >= trade.stop_loss:
                return True, "Broke prior high"
            if price <= trade.entry_price * (1 - target_pct):
                return True, "Momentum target"
        return False, None

    def extract_features(self, df: pd.DataFrame):
        context = getattr(self, "_context", {})
        return {
            "momentum_move": float(context.get("move", 0.0)),
            "volume_ratio": float(context.get("volume_ratio", 0.0)),
        }
