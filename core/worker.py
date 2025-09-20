import importlib
import pandas as pd
from core.learner import Learner
from core.logger import EventLogger

ACRONYMS = {"sma", "ema", "macd", "rsi", "atr"}

class Worker:
    def __init__(self, name, symbol, strategy, params, logger=None, config=None):
        self.name = name
        self.symbol = symbol
        self.strategy_name = strategy
        self.params = params
        if not self.strategy_name:
            raise ValueError(f"Worker {name} missing 'strategy' field")

        # Import strategy module dynamically
        module = importlib.import_module(f"strategies.{self.strategy_name}")

        # Build class name (with acronym handling)
        parts = self.strategy_name.split("_")
        class_name = "".join([p.upper() if p in ACRONYMS else p.capitalize() for p in parts]) + "Strategy"
        alt_class_name = "".join([p.title() for p in parts]) + "Strategy"

        if hasattr(module, class_name):
            strat_cls = getattr(module, class_name)
        elif hasattr(module, alt_class_name):
            strat_cls = getattr(module, alt_class_name)
        else:
            raise ImportError(f"Strategy class not found for {self.strategy_name}")

        # Instantiate strategy
        self.strategy = strat_cls(self.symbol, self.params.get("params", {}))
        self.logger = logger or EventLogger()
        self.learner = Learner()
        self.config = config or {}

        self.state = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0}

    def passes_rules(self, data):
        df = pd.DataFrame(data[self.symbol])
        signal = self.strategy.generate_signals(df)
        return signal in ("buy", "sell")

    def score_edge(self, data):
        df = pd.DataFrame(data[self.symbol])
        signal = self.strategy.generate_signals(df)
        if signal not in ("buy", "sell"):
            return 0.0  # no trade setup

        base_score = 0.6  # rule-based confidence
        candles = list(data.values())[0] if data else []
        ml_score = 0.5
        if candles:
            latest_candle = candles[-1]
            ml_score = self.learner.predict_edge(self, latest_candle)

        ml_weight = self.config.get("risk", {}).get("ml_weight", 0.5)
        return (1 - ml_weight) * base_score + ml_weight * ml_score

    def check_exit(self, data, trade):
        df = pd.DataFrame(data[self.symbol])
        exit_now, reason = self.strategy.check_exit(trade, df)
        if exit_now:
            price = df["close"].iloc[-1]
            if trade["side"] == "BUY":
                pnl = (price - trade["entry_price"]) * trade["qty"]
            else:
                pnl = (trade["entry_price"] - price) * trade["qty"]
            return reason, price, pnl
        return None, None, 0.0

    def extract_features(self, data):
        df = pd.DataFrame(data[self.symbol])
        return self.strategy.extract_features(df)


