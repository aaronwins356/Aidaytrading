
from .base_strategy import StrategyBase, Trade
import pandas as pd

class EMABreakoutStrategy(StrategyBase):
    def generate_signals(self, df: pd.DataFrame):
        n = int(self.params.get("length", 34))
        if len(df) < n + 2:
            return None
        ema = self.ema(df["close"], n)
        c = df["close"]
        # Break above/below after a consolidation (simple: previous close below/above)
        if c.iloc[-1] > ema.iloc[-1] and c.iloc[-2] <= ema.iloc[-2]:
            return "buy"
        if c.iloc[-1] < ema.iloc[-1] and c.iloc[-2] >= ema.iloc[-2]:
            return "sell"
        return None

    def check_exit(self, trade: Trade, df: pd.DataFrame):
        price = float(df["close"].iloc[-1])
        if price <= trade.stop_loss or price >= trade.take_profit:
            return True, "SL/TP"
        ema = self.ema(df["close"], int(self.params.get("length", 34)))
        side = self.trade_side(trade)

        if side == "BUY" and price < ema.iloc[-1]:
            return True, "Close < EMA"
        if side == "SELL" and price > ema.iloc[-1]:
            return True, "Close > EMA"
        return False, None

    def extract_features(self, df: pd.DataFrame):
        ema = self.ema(df["close"], int(self.params.get("length", 34)))
        dist = (df["close"].iloc[-1] / ema.iloc[-1]) - 1
        vol = df["close"].pct_change().rolling(20).std().iloc[-1]
        return {"ema_dist": float(dist), "vol_20": float(vol)}
