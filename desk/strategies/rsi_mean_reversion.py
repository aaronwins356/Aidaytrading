
from .base_strategy import StrategyBase, Trade
import pandas as pd

class RSIMeanReversionStrategy(StrategyBase):
    def generate_signals(self, df: pd.DataFrame):
        n = int(self.params.get("length", 14))
        low_th = float(self.params.get("low", 30))
        high_th = float(self.params.get("high", 70))
        if len(df) < n + 2:
            return None
        r = self.rsi(df["close"], n)
        if r.iloc[-2] < low_th and r.iloc[-1] >= low_th:
            return "buy"
        if r.iloc[-2] > high_th and r.iloc[-1] <= high_th:
            return "sell"
        return None

    def check_exit(self, trade: Trade, df: pd.DataFrame):
        price = float(df["close"].iloc[-1])
        if price <= trade.stop_loss or price >= trade.take_profit:
            return True, "SL/TP"
        r = self.rsi(df["close"], int(self.params.get("length", 14)))
        mid = float(self.params.get("mid", 50))
        side = self.trade_side(trade)

        if side == "BUY" and r.iloc[-1] > mid:
            return True, "RSI normalized"
        if side == "SELL" and r.iloc[-1] < mid:
            return True, "RSI normalized"
        return False, None

    def extract_features(self, df: pd.DataFrame):
        r = self.rsi(df["close"], int(self.params.get("length", 14)))
        return {"rsi": float(r.iloc[-1])}
