"""Equity tracking utilities."""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime
from typing import Deque, Dict, Iterable, Tuple

from ai_trader.services.trade_log import TradeLog


class EquityEngine:
    """Track account equity and performance metrics."""

    def __init__(self, trade_log: TradeLog, starting_equity: float | None) -> None:
        self._trade_log = trade_log
        self._starting_equity = (
            float(starting_equity)
            if starting_equity is not None and starting_equity > 0.0
            else None
        )
        self._equity_history: Deque[Tuple[datetime, float, float, float]] = deque(maxlen=2000)
        self._latest_equity: float = float(starting_equity or 0.0)
        self._latest_pnl_percent: float = 0.0
        self._latest_pnl_usd: float = 0.0

    @property
    def latest_equity(self) -> float:
        return self._latest_equity

    @property
    def history(self) -> Iterable[Tuple[datetime, float, float, float]]:
        return list(self._equity_history)

    def update(self, equity: float, starting_equity: float | None = None) -> None:
        """Update metrics when a new equity value is available."""

        if starting_equity is not None and starting_equity > 0.0:
            if self._starting_equity is None or not math.isclose(
                self._starting_equity,
                starting_equity,
                rel_tol=1e-9,
                abs_tol=1e-6,
            ):
                self._starting_equity = float(starting_equity)

        baseline = self._starting_equity
        self._latest_equity = equity
        if baseline:
            pnl_usd = equity - baseline
            pnl_percent = (pnl_usd / baseline) * 100
        else:
            # Without a baseline we cannot compute meaningful performance metrics.
            pnl_usd = 0.0
            pnl_percent = 0.0
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
