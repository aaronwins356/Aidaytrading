from __future__ import annotations

import importlib
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from core.learner import Learner
from core.logger import EventLogger
from strategies.base_strategy import StrategyBase, Trade as StrategyTrade
from utils.data_utils import candles_to_dataframe

ACRONYMS = {"sma", "ema", "macd", "rsi", "atr"}


class Worker:
    def __init__(
        self,
        name: str,
        symbol: str,
        strategy: str,
        params: Dict[str, Any],
        logger: Optional[EventLogger] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.symbol = symbol
        self.strategy_name = strategy
        self.params = params
        if not self.strategy_name:
            raise ValueError(f"Worker {name} missing 'strategy' field")

        module = importlib.import_module(f"strategies.{self.strategy_name}")
        parts = self.strategy_name.split("_")
        class_name = "".join(
            [p.upper() if p in ACRONYMS else p.capitalize() for p in parts]
        ) + "Strategy"
        alt_class_name = "".join([p.title() for p in parts]) + "Strategy"

        if hasattr(module, class_name):
            strat_cls = getattr(module, class_name)
        elif hasattr(module, alt_class_name):
            strat_cls = getattr(module, alt_class_name)
        else:
            raise ImportError(f"Strategy class not found for {self.strategy_name}")

        self.strategy: StrategyBase = strat_cls(
            self.symbol, self.params.get("params", {})
        )
        self.logger = logger or EventLogger()
        self.learner = Learner()
        self.config = config or {}

        self.allocation = float(self.params.get("allocation", 0.0) or 0.0)
        if self.allocation <= 0:
            self.allocation = float(
                self.params.get("params", {}).get("allocation", 0.1) or 0.1
            )
        self.state: Dict[str, Any] = {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "pnl": 0.0,
            "candles": [],
        }

    def _risk_budget_usd(self, risk_cfg: Optional[Dict[str, Any]], price: float) -> float:
        risk_cfg = risk_cfg or {}
        fixed = float(risk_cfg.get("fixed_risk_usd", 0.0) or 0.0)
        usd_budget = fixed * self.allocation if fixed and self.allocation else fixed
        if usd_budget <= 0:
            params = self.params.get("params", {})
            order_size = float(params.get("order_size", 0.0) or 0.0)
            if order_size > 0:
                return order_size
            risk_fraction = float(params.get("risk_per_trade", 0.0) or 0.0)
            balance = float(self.config.get("settings", {}).get("balance", 0.0) or 0.0)
            if risk_fraction > 0 and balance > 0:
                usd_budget = balance * risk_fraction
        if price <= 0:
            return 0.0
        return usd_budget

    def _determine_quantity(self, risk_cfg: Optional[Dict[str, Any]], price: float) -> float:
        usd_budget = self._risk_budget_usd(risk_cfg, price)
        if usd_budget <= 0 or price <= 0:
            return 0.0
        qty = usd_budget / price
        params = self.params.get("params", {})
        min_qty = float(params.get("min_qty", 0.0) or 0.0)
        if min_qty > 0:
            qty = max(qty, min_qty)
        precision = int(params.get("qty_precision", 6) or 6)
        return round(qty, precision)

    @staticmethod
    def _build_trade_adapter(trade: Dict[str, Any]) -> StrategyTrade:
        return StrategyTrade(
            side=str(trade.get("side", "buy")).lower(),
            entry_price=float(trade.get("entry_price", 0.0) or 0.0),
            stop_loss=float(trade.get("stop_loss", 0.0) or 0.0),
            take_profit=float(trade.get("take_profit", 0.0) or 0.0),
            meta=trade,
        )

    def _candles_df(self, candles: Iterable[Dict[str, Any]]) -> pd.DataFrame:
        return candles_to_dataframe(candles)

    def generate_signal(
        self, candles: List[Dict[str, Any]], risk_cfg: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[str, float, float]]:
        df = self._candles_df(candles)
        if df.empty:
            return None
        signal = self.strategy.generate_signals(df)
        if signal not in {"buy", "sell"}:
            return None
        price = float(df["close"].iloc[-1])
        qty = self._determine_quantity(risk_cfg, price)
        if qty <= 0:
            return None
        return signal.upper(), qty, price

    def passes_rules(self, candles: Iterable[Dict[str, Any]]) -> bool:
        df = self._candles_df(candles)
        signal = self.strategy.generate_signals(df)
        return signal in {"buy", "sell"}

    def score_edge(self, candles: Iterable[Dict[str, Any]]) -> float:
        df = self._candles_df(candles)
        signal = self.strategy.generate_signals(df)
        if signal not in {"buy", "sell"}:
            return 0.0
        base_score = 0.6
        ml_score = 0.5
        if not df.empty:
            latest = df.iloc[-1].to_dict()
            ml_score = self.learner.predict_edge(self, latest)
        ml_weight = float(self.config.get("risk", {}).get("ml_weight", 0.5) or 0.5)
        return (1 - ml_weight) * base_score + ml_weight * ml_score

    def check_exit(
        self, candles: List[Dict[str, Any]], trade: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[float], float]:
        df = self._candles_df(candles)
        if df.empty:
            return None, None, 0.0
        strat_trade = self._build_trade_adapter(trade)
        exit_now, reason = self.strategy.check_exit(strat_trade, df)
        if not exit_now:
            return None, None, 0.0
        price = float(df["close"].iloc[-1])
        entry = float(trade.get("entry_price", price) or price)
        qty = float(trade.get("qty", 0.0) or 0.0)
        side = str(trade.get("side", "BUY")).upper()
        pnl = (price - entry) * qty if side == "BUY" else (entry - price) * qty
        return reason or "exit", price, pnl

    def extract_features(self, candles: Iterable[Dict[str, Any]]):
        df = self._candles_df(candles)
        if df.empty:
            return {}
        return self.strategy.extract_features(df)


