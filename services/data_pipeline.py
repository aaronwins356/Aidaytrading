import numpy as np

class DataPipeline:
    """
    Extracts features from bar data for use in ML models and strategies.
    """

    def __init__(self, lookback=20):
        self.lookback = lookback
        self.cache = {}

    def features_from_bar(self, bar):
        """
        Base OHLCV feature vector.
        """
        return [
            bar.Open,
            bar.High,
            bar.Low,
            bar.Close,
            bar.Volume,
            bar.Close - bar.Open,
            bar.High - bar.Low
        ]

    def rolling_features(self, symbol, bar):
        """
        Maintain rolling window for a symbol and return rolling stats.
        """
        if symbol not in self.cache:
            self.cache[symbol] = []
        self.cache[symbol].append(bar.Close)
        if len(self.cache[symbol]) > self.lookback:
            self.cache[symbol].pop(0)

        prices = self.cache[symbol]
        mean = np.mean(prices)
        std = np.std(prices)

        return [mean, std]
