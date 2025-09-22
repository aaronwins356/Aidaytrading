"""Equity tracking utilities."""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Deque, Dict, Iterable, Tuple

from .trade_log import TradeLog


class EquityEngine:
    """Track account equity and performance metrics."""

    def __init__(self, trade_log: TradeLog, starting_equity: float) -> None:
        self._trade_log = trade_log
        self._starting_equity = starting_equity
        self._equity_history: Deque[Tuple[datetime, float, float, float]] = deque(maxlen=2000)
        self._latest_equity: float = starting_equity
        self._latest_pnl_percent: float = 0.0
        self._latest_pnl_usd: float = 0.0

    @property
    def latest_equity(self) -> float:
        return self._latest_equity

    @property
    def history(self) -> Iterable[Tuple[datetime, float, float, float]]:
        return list(self._equity_history)

    def update(self, equity: float) -> None:
        """Update metrics when a new equity value is available."""

        self._latest_equity = equity
        pnl_usd = equity - self._starting_equity
        pnl_percent = (pnl_usd / self._starting_equity) * 100 if self._starting_equity else 0.0
        snapshot = (datetime.utcnow(), equity, pnl_percent, pnl_usd)
        self._equity_history.append(snapshot)
        self._latest_pnl_percent = pnl_percent
        self._latest_pnl_usd = pnl_usd
        self._trade_log.record_equity(equity, pnl_percent, pnl_usd)

    def get_latest_metrics(self) -> Dict[str, float]:
        """Return a dictionary summarising key performance metrics."""

        return {
            "equity": self._latest_equity,
            "pnl_percent": self._latest_pnl_percent,
            "pnl_usd": self._latest_pnl_usd,
        }
