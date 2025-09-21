
from .base_strategy import StrategyBase, Trade
import pandas as pd

class BollingerBandStrategy(StrategyBase):
    def generate_signals(self, df: pd.DataFrame):
        n = int(self.params.get("length", 20))
        mult = float(self.params.get("mult", 2.0))
        if len(df) < n + 2:
            return None
        lower, mid, upper = self.bbands(df["close"], n, mult)
        c = df["close"]
        if c.iloc[-1] < lower.iloc[-1] and c.iloc[-2] >= lower.iloc[-2]:
            return "buy"  # mean reversion long
        if c.iloc[-1] > upper.iloc[-1] and c.iloc[-2] <= upper.iloc[-2]:
            return "sell"  # mean reversion short
        return None

    def check_exit(self, trade: Trade, df: pd.DataFrame):
        n = int(self.params.get("length", 20))
        mult = float(self.params.get("mult", 2.0))
        lower, mid, upper = self.bbands(df["close"], n, mult)
        price = float(df["close"].iloc[-1])
        if price <= trade.stop_loss or price >= trade.take_profit:
            return True, "SL/TP"
        side = self.trade_side(trade)

        if side == "BUY" and price >= mid.iloc[-1]:
            return True, "Mean reversion hit"
        if side == "SELL" and price <= mid.iloc[-1]:
            return True, "Mean reversion hit"
        return False, None

    def extract_features(self, df: pd.DataFrame):
        lower, mid, upper = self.bbands(df["close"],
                                        int(self.params.get("length", 20)),
                                        float(self.params.get("mult", 2.0)))
        c = df["close"].iloc[-1]
        return {"bb_pos": float((c - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1] + 1e-9))}
