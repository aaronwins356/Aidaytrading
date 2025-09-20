import time

class FeedHandler:
    """
    Pulls latest candles from broker and standardizes them.
    """

    def __init__(self, broker, lookback=50):
        self.broker = broker
        self.lookback = lookback
        self.cache = {}

    def get_latest_candles(self, workers):
        candles = {}
        for w in workers:
            try:
                sym = w.symbol
                ohlcv = self.broker.fetch_ohlcv(sym, "1m", self.lookback)
                if not ohlcv:
                    continue
                ts, o, h, l, c, v = ohlcv[-1]
                bar = Bar(ts, o, h, l, c, v)
                candles[sym] = bar
                self.cache[sym] = bar
            except Exception as e:
                print(f"[FEED] Failed to fetch {w.symbol}: {e}")
        return candles


class Bar:
    """Simple wrapper for OHLCV data"""
    def __init__(self, ts, o, h, l, c, v):
        self.ts = ts
        self.open = o
        self.high = h
        self.low = l
        self.close = c
        self.volume = v
