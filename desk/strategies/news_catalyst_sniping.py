"""News catalyst momentum scalping strategy."""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from .base_strategy import StrategyBase, Trade


class NewsCatalystSnipingStrategy(StrategyBase):
    def __init__(self, symbol: str, params: Optional[Dict[str, float]] = None) -> None:
        super().__init__(symbol, params)
        self._context: Dict[str, float] = {}

    def _sentiment_value(self, df: pd.DataFrame) -> float:
        sentiment_columns = [
            "sentiment",
            "sentiment_score",
            "news_sentiment",
            "liquidation_pressure",
        ]
        for column in sentiment_columns:
            if column in df.columns:
                return float(df[column].iloc[-1])
        return 0.0

    def generate_signals(self, df: pd.DataFrame) -> Optional[str]:
        if len(df) < 10:
            self._context = {}
            return None
        volume_mult = float(self.params.get("volume_multiplier", 2.0))
        sentiment_threshold = float(self.params.get("sentiment_threshold", 0.2))
        window = df.iloc[-20:]
        avg_volume = float(window["volume"].mean()) if len(window) else 0.0
        latest = df.iloc[-1]
        price_change = (float(latest["close"]) - float(latest["open"])) / max(float(latest["open"]), 1e-9)
        volume_ratio = float(latest["volume"]) / avg_volume if avg_volume else 0.0
        sentiment = self._sentiment_value(df)
        signal: Optional[str] = None
        if volume_ratio >= volume_mult and abs(sentiment) >= sentiment_threshold:
            signal = "buy" if sentiment > 0 or price_change > 0 else "sell"
        elif abs(price_change) >= float(self.params.get("price_change_threshold", 0.01)) and volume_ratio >= volume_mult:
            signal = "buy" if price_change > 0 else "sell"
        if signal:
            self._context = {
                "sentiment": sentiment,
                "volume_ratio": volume_ratio,
                "price_change": price_change,
            }
        else:
            self._context = {}
        return signal

    def plan_trade(self, side: str, df: pd.DataFrame) -> Dict[str, float]:
        price = float(df["close"].iloc[-1])
        target_pct = float(self.params.get("target_pct", 0.007))
        stop_pct = float(self.params.get("stop_pct", 0.005))
        hold_minutes = float(self.params.get("max_hold_minutes", 5.0))
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
        target_pct = float(self.params.get("target_pct", 0.007))
        side = self.trade_side(trade)

        if side == "BUY":
            if price <= trade.stop_loss:
                return True, "Catalyst stop"
            if price >= trade.entry_price * (1 + target_pct):
                return True, "News scalp target"
        elif side == "SELL":
            if price >= trade.stop_loss:
                return True, "Catalyst stop"
            if price <= trade.entry_price * (1 - target_pct):
                return True, "News scalp target"
        return False, None

    def extract_features(self, df: pd.DataFrame):
        context = getattr(self, "_context", {})
        return {
            "sentiment": float(context.get("sentiment", 0.0)),
            "volume_ratio": float(context.get("volume_ratio", 0.0)),
            "price_change": float(context.get("price_change", 0.0)),
        }
