
from .base_strategy import StrategyBase, Trade
import pandas as pd

class ATRTrailingStopStrategy(StrategyBase):
    def generate_signals(self, df: pd.DataFrame):
        n = int(self.params.get("atr_length", 14))
        mult = float(self.params.get("atr_mult", 3.0))
        if len(df) < n + 2:
            return None
        atr = self.atr(df, n)
        # Simple breakout: close breaks highest high/lowest low of lookback
        look = int(self.params.get("lookback", 20))
        if len(df) < max(n, look) + 2:
            return None
        hh = df["high"].rolling(look, min_periods=look).max()
        ll = df["low"].rolling(look, min_periods=look).min()
        if df["close"].iloc[-1] > hh.iloc[-1]:
            return "buy"
        if df["close"].iloc[-1] < ll.iloc[-1]:
            return "sell"
        return None

    def check_exit(self, trade: Trade, df: pd.DataFrame):
        n = int(self.params.get("atr_length", 14))
        mult = float(self.params.get("atr_mult", 3.0))
        atr = self.atr(df, n)
        c = df["close"].iloc[-1]
        # Dynamic trailing stop based on trade direction
        if trade.side == "buy":
            trail = c - mult * atr.iloc[-1]
            if c <= trail or c <= trade.stop_loss or c >= trade.take_profit:
                return True, "ATR trail or SL/TP"
        else:
            trail = c + mult * atr.iloc[-1]
            if c >= trail or c <= trade.take_profit or c >= trade.stop_loss:
                return True, "ATR trail or SL/TP"
        return False, None

    def extract_features(self, df: pd.DataFrame):
        a = self.atr(df, int(self.params.get("atr_length", 14)))
        return {"atr": float(a.iloc[-1])}
