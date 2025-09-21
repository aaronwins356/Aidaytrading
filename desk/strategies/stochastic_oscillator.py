
from .base_strategy import StrategyBase, Trade
import pandas as pd

class StochasticOscillatorStrategy(StrategyBase):
    def generate_signals(self, df: pd.DataFrame):
        k_len = int(self.params.get("k", 14))
        d_len = int(self.params.get("d", 3))
        low_th = float(self.params.get("low", 20))
        high_th = float(self.params.get("high", 80))
        if len(df) < k_len + d_len + 2:
            return None
        k, d = self.stoch(df, k_len, d_len)
        if k.iloc[-2] < low_th and k.iloc[-1] >= low_th and k.iloc[-1] > d.iloc[-1]:
            return "buy"
        if k.iloc[-2] > high_th and k.iloc[-1] <= high_th and k.iloc[-1] < d.iloc[-1]:
            return "sell"
        return None

    def check_exit(self, trade: Trade, df: pd.DataFrame):
        k, d = self.stoch(df, int(self.params.get("k", 14)), int(self.params.get("d", 3)))
        price = float(df["close"].iloc[-1])
        if price <= trade.stop_loss or price >= trade.take_profit:
            return True, "SL/TP"
        side = self.trade_side(trade)

        if side == "BUY" and k.iloc[-1] > 80 and k.iloc[-1] < d.iloc[-1]:
            return True, "Stoch cross down"
        if side == "SELL" and k.iloc[-1] < 20 and k.iloc[-1] > d.iloc[-1]:
            return True, "Stoch cross up"
        return False, None

    def extract_features(self, df: pd.DataFrame):
        k, d = self.stoch(df, int(self.params.get("k", 14)), int(self.params.get("d", 3)))
        return {"stoch_k": float(k.iloc[-1]), "stoch_d": float(d.iloc[-1])}
