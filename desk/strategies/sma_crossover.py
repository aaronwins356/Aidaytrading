
from .base_strategy import StrategyBase, Trade
import pandas as pd

class SMACrossoverStrategy(StrategyBase):
    def generate_signals(self, df: pd.DataFrame):
        fast_len = int(self.params.get("fast_length", 20))
        slow_len = int(self.params.get("slow_length", 50))
        if len(df) < slow_len + 2:
            return None
        fast = self.sma(df["close"], fast_len)
        slow = self.sma(df["close"], slow_len)
        crossed_up = fast.iloc[-1] > slow.iloc[-1] and fast.iloc[-2] <= slow.iloc[-2]
        crossed_dn = fast.iloc[-1] < slow.iloc[-1] and fast.iloc[-2] >= slow.iloc[-2]
        if crossed_up:
            return "buy"
        if crossed_dn:
            return "sell"
        return None

    def check_exit(self, trade: Trade, df: pd.DataFrame):
        price = float(df["close"].iloc[-1])
        if price <= trade.stop_loss or price >= trade.take_profit:
            return True, "SL/TP"
        fast = self.sma(df["close"], int(self.params.get("fast_length", 20)))
        slow = self.sma(df["close"], int(self.params.get("slow_length", 50)))
        side = self.trade_side(trade)

        if side == "BUY" and fast.iloc[-1] < slow.iloc[-1]:
            return True, "Reverse cross"
        if side == "SELL" and fast.iloc[-1] > slow.iloc[-1]:
            return True, "Reverse cross"
        return False, None

    def extract_features(self, df: pd.DataFrame):
        fast = self.sma(df["close"], int(self.params.get("fast_length", 20)))
        slow = self.sma(df["close"], int(self.params.get("slow_length", 50)))
        spread = (fast / slow - 1).iloc[-1]
        slope = (fast.diff().iloc[-5:].mean())
        return {"sma_spread": float(spread), "sma_slope": float(slope)}
