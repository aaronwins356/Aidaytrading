
# strategies/base_strategy.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np

@dataclass
class Trade:
    side: str  # "buy" or "sell"
    entry_price: float
    stop_loss: float
    take_profit: float
    meta: Dict[str, Any]

class StrategyBase(ABC):
    """Abstract interface for rule-based workers.
    Each worker must implement:
      - generate_signals(df): -> "buy" | "sell" | None
      - check_exit(trade, df): -> (exit_now: bool, reason: Optional[str])
      - extract_features(df): -> Dict[str, float] (for ML in main.py)
    No ML lives here; just classic TA logic + hard exits.
    """

    def __init__(self, symbol: str, params: Optional[Dict[str, Any]] = None):
        self.symbol = symbol
        self.params = params or {}

    # ---------- required ----------
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> Optional[str]:
        pass

    @abstractmethod
    def check_exit(self, trade: Trade, df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        pass

    @abstractmethod
    def extract_features(self, df: pd.DataFrame) -> Dict[str, float]:
        pass

    # ---------- helpers (pure TA) ----------
    @staticmethod
    def sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(n, min_periods=n).mean()

    @staticmethod
    def ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=n, adjust=False, min_periods=n).mean()

    @staticmethod
    def rsi(s: pd.Series, n: int = 14) -> pd.Series:
        delta = s.diff()
        up = delta.clip(lower=0.0)
        down = -delta.clip(upper=0.0)
        gain = up.ewm(alpha=1/n, adjust=False).mean()
        loss = down.ewm(alpha=1/n, adjust=False).mean()
        rs = gain / (loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    @staticmethod
    def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = StrategyBase.ema(s, fast)
        ema_slow = StrategyBase.ema(s, slow)
        line = ema_fast - ema_slow
        signal_line = StrategyBase.ema(line, signal)
        hist = line - signal_line
        return line, signal_line, hist

    @staticmethod
    def bbands(s: pd.Series, n: int = 20, mult: float = 2.0):
        ma = StrategyBase.sma(s, n)
        std = s.rolling(n, min_periods=n).std()
        upper = ma + mult * std
        lower = ma - mult * std
        return lower, ma, upper

    @staticmethod
    def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(n, min_periods=n).mean()

    @staticmethod
    def stoch(df: pd.DataFrame, k: int = 14, d: int = 3):
        low_k = df["low"].rolling(k, min_periods=k).min()
        high_k = df["high"].rolling(k, min_periods=k).max()
        percent_k = 100 * (df["close"] - low_k) / (high_k - low_k).replace(0, np.nan)
        percent_d = percent_k.rolling(d, min_periods=d).mean()
        return percent_k.fillna(50), percent_d.fillna(50)

    @staticmethod
    def momentum(s: pd.Series, n: int = 10) -> pd.Series:
        return s / s.shift(n) - 1.0

    # ---------- risk helpers ----------
    def default_sl_tp(self, price: float) -> Tuple[float, float]:
        sl_pct = float(self.params.get("stop_loss_pct", 0.01))
        tp_pct = float(self.params.get("take_profit_pct", 0.02))
        return price * (1 - sl_pct), price * (1 + tp_pct)
