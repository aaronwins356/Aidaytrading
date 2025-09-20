
from .base_strategy import StrategyBase, Trade
import pandas as pd

class MACDTrendStrategy(StrategyBase):
    def generate_signals(self, df: pd.DataFrame):
        line, sig, hist = self.macd(df["close"],
                                    int(self.params.get("fast", 12)),
                                    int(self.params.get("slow", 26)),
                                    int(self.params.get("signal", 9)))
        if len(df) < int(self.params.get("slow", 26)) + int(self.params.get("signal", 9)) + 2:
            return None
        crossed_up = line.iloc[-1] > sig.iloc[-1] and line.iloc[-2] <= sig.iloc[-2]
        crossed_dn = line.iloc[-1] < sig.iloc[-1] and line.iloc[-2] >= sig.iloc[-2]
        if crossed_up:
            return "buy"
        if crossed_dn:
            return "sell"
        return None

    def check_exit(self, trade: Trade, df: pd.DataFrame):
        price = float(df["close"].iloc[-1])
        if price <= trade.stop_loss or price >= trade.take_profit:
            return True, "SL/TP"
        line, sig, _ = self.macd(df["close"],
                                 int(self.params.get("fast", 12)),
                                 int(self.params.get("slow", 26)),
                                 int(self.params.get("signal", 9)))
        if trade.side == "buy" and line.iloc[-1] < sig.iloc[-1]:
            return True, "MACD flip"
        if trade.side == "sell" and line.iloc[-1] > sig.iloc[-1]:
            return True, "MACD flip"
        return False, None

    def extract_features(self, df: pd.DataFrame):
        line, sig, hist = self.macd(df["close"])
        return {"macd_hist": float(hist.iloc[-1]), "macd_diff": float((line - sig).iloc[-1])}
