
from .base_strategy import StrategyBase, Trade
import pandas as pd

class MomentumStrategy(StrategyBase):
    def generate_signals(self, df: pd.DataFrame):
        n = int(self.params.get("length", 10))
        thr = float(self.params.get("threshold", 0.01))
        if len(df) < n + 2:
            return None
        m = self.momentum(df["close"], n)
        if m.iloc[-1] > thr and m.iloc[-2] <= thr:
            return "buy"
        if m.iloc[-1] < -thr and m.iloc[-2] >= -thr:
            return "sell"
        return None

    def check_exit(self, trade: Trade, df: pd.DataFrame):
        price = float(df["close"].iloc[-1])
        if price <= trade.stop_loss or price >= trade.take_profit:
            return True, "SL/TP"
        m = self.momentum(df["close"], int(self.params.get("length", 10)))
        side = self.trade_side(trade)

        if side == "BUY" and m.iloc[-1] < 0:
            return True, "Momentum faded"
        if side == "SELL" and m.iloc[-1] > 0:
            return True, "Momentum faded"
        return False, None

    def extract_features(self, df: pd.DataFrame):
        m = self.momentum(df["close"], int(self.params.get("length", 10)))
        return {"mom": float(m.iloc[-1])}
