"""Correlation divergence strategy leveraging leader/laggard dynamics."""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from .base_strategy import StrategyBase, Trade


class CorrelationDivergenceStrategy(StrategyBase):
    def __init__(self, symbol: str, params: Optional[Dict[str, float]] = None) -> None:
        super().__init__(symbol, params)
        self._context: Dict[str, float] = {}

    def _reference_series(self, df: pd.DataFrame) -> Optional[pd.Series]:
        ref_col = self.params.get("reference_col")
        if ref_col and ref_col in df.columns:
            return df[ref_col]
        default_cols = ["reference_close", "leader_close", "btc_close", "eth_close"]
        for col in default_cols:
            if col in df.columns:
                return df[col]
        return None

    def generate_signals(self, df: pd.DataFrame) -> Optional[str]:
        ref_series = self._reference_series(df)
        if ref_series is None or len(df) < 5:
            self._context = {}
            return None
        lookback = int(self.params.get("lookback", 5))
        threshold = float(self.params.get("threshold", 0.01))
        ref_returns = ref_series.pct_change(lookback)
        asset_returns = df["close"].pct_change(lookback)
        ref_ret = float(ref_returns.iloc[-1])
        asset_ret = float(asset_returns.iloc[-1])
        spread = ref_ret - asset_ret
        signal: Optional[str] = None
        if spread >= threshold:
            signal = "buy"  # our asset lagging leader's rally
        elif spread <= -threshold:
            signal = "sell"  # our asset outperforming vs leader
        if signal:
            self._context = {
                "spread": spread,
                "ref_return": ref_ret,
                "asset_return": asset_ret,
            }
        else:
            self._context = {}
        return signal

    def plan_trade(self, side: str, df: pd.DataFrame) -> Dict[str, float]:
        price = float(df["close"].iloc[-1])
        threshold = float(self.params.get("threshold", 0.01))
        stop_multiplier = float(self.params.get("stop_multiplier", 1.25))
        take_multiplier = float(self.params.get("take_multiplier", 0.75))
        hold_minutes = float(self.params.get("max_hold_minutes", 120.0))
        stop_pct = threshold * stop_multiplier
        take_pct = threshold * take_multiplier
        if side.upper() == "BUY":
            stop_loss = price * (1 - stop_pct)
            take_profit = price * (1 + take_pct)
        else:
            stop_loss = price * (1 + stop_pct)
            take_profit = price * (1 - take_pct)
        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "max_hold_minutes": hold_minutes,
            "metadata": self._context,
        }

    def check_exit(self, trade: Trade, df: pd.DataFrame):
        ref_series = self._reference_series(df)
        if ref_series is None:
            return False, None
        lookback = int(self.params.get("lookback", 5))
        ref_returns = ref_series.pct_change(lookback)
        asset_returns = df["close"].pct_change(lookback)
        spread = float(ref_returns.iloc[-1] - asset_returns.iloc[-1])
        threshold = float(self.params.get("threshold", 0.01))
        side = self.trade_side(trade)

        if side == "BUY":
            if spread <= threshold * 0.2:
                return True, "Spread normalized"
            if df["close"].iloc[-1] <= trade.stop_loss:
                return True, "Stop"
        elif side == "SELL":
            if spread >= -threshold * 0.2:
                return True, "Spread normalized"
            if df["close"].iloc[-1] >= trade.stop_loss:
                return True, "Stop"
        return False, None

    def extract_features(self, df: pd.DataFrame):
        context = getattr(self, "_context", {})
        return {
            "spread": float(context.get("spread", 0.0)),
            "ref_return": float(context.get("ref_return", 0.0)),
            "asset_return": float(context.get("asset_return", 0.0)),
        }
