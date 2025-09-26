"""Risk management utilities with tuned default risk limits for moderate returns."""

from __future__ import annotations

from datetime import datetime
from typing import Dict

from ai_trader.services.types import TradeIntent


class RiskManager:
    """Evaluate trades against configurable risk rules."""

    def __init__(self, config: Dict[str, float]) -> None:
           self._max_drawdown = float(config.get("max_drawdown_percent", 20.0))
        self._daily_loss_limit = float(config.get("daily_loss_limit_percent", 5.0))
        self._max_duration_minutes = float(config.get("max_position_duration_minutes", 240))
        self._daily_anchor = datetime.utcnow().date()
        self._daily_peak_equity: float | None = None

    def reset_daily_limits(self) -> None:
        today = datetime.utcnow().date()
        if today != self._daily_anchor:
            self._daily_anchor = today
            self._daily_peak_equity = None

    def check_trade(
        self,
        intent: TradeIntent,
        equity_metrics: Dict[str, float],
        open_positions: int,
        max_open_positions: int,
    ) -> bool:
        """Return True if the trade is allowed."""

        self.reset_daily_limits()
        if open_positions >= max_open_positions and intent.action == "OPEN":
            return False

        equity = float(equity_metrics.get("equity", 0.0))
        pnl_percent = float(equity_metrics.get("pnl_percent", 0.0))

        if pnl_percent <= -self._max_drawdown:
            return False

        if self._daily_peak_equity is None or equity > self._daily_peak_equity:
            self._daily_peak_equity = equity

        if self._daily_peak_equity:
            drawdown = (
                (equity - self._daily_peak_equity) / self._daily_peak_equity * 100
            )
            if drawdown <= -self._daily_loss_limit:
                return False

        return True

    @property
    def max_duration_minutes(self) -> float:
        return self._max_duration_minutes
