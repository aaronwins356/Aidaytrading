"""Risk engine implementing trap doors and circuit breakers."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable


@dataclass
class EquityTrapdoor:
    floor: float
    locked_at: float


class RiskEngine:
    """Evaluates desk-level and bot-level risk constraints."""

    def __init__(
        self,
        daily_dd: float | None,
        weekly_dd: float | None,
        default_stop_pct: float,
        max_concurrent: int,
        halt_on_dd: bool,
        trapdoor_pct: float,
        *,
        max_position_value: float | None = None,
    ) -> None:
        self.daily_dd = daily_dd or 0.0
        self.weekly_dd = weekly_dd or 0.0
        self.default_stop_pct = max(0.0001, float(default_stop_pct))
        self.max_concurrent = max_concurrent
        self.halt_on_dd = halt_on_dd
        self.trapdoor_pct = trapdoor_pct
        self.max_position_value = max_position_value if max_position_value and max_position_value > 0 else None

        self.start_equity: float | None = None
        self.equity_high: float | None = None
        self.trapdoor: EquityTrapdoor | None = None
        self.halted = False

    # ------------------------------------------------------------------
    def initialise(self, equity: float) -> None:
        self.start_equity = equity
        self.equity_high = equity
        self.trapdoor = EquityTrapdoor(floor=equity * (1 - self.trapdoor_pct), locked_at=time.time())

    def check_account(self, equity: float) -> None:
        if self.start_equity is None:
            self.initialise(equity)
        assert self.start_equity is not None

        previous_high = self.equity_high if self.equity_high is not None else equity
        self.equity_high = max(previous_high, equity)

        # Daily/weekly drawdown checks.
        if self.daily_dd and equity < self.start_equity * (1 - self.daily_dd):
            print("[RISK] Daily drawdown threshold breached – monitoring closely.")

        if self.weekly_dd and equity < self.start_equity * (1 - self.weekly_dd):
            print("[RISK] Weekly drawdown threshold breached – monitoring closely.")

        # Trap door: once equity makes a new high, raise the floor.
        if self.trapdoor:
            if equity > previous_high:
                new_floor = equity * (1 - self.trapdoor_pct)
                self.trapdoor = EquityTrapdoor(floor=new_floor, locked_at=time.time())
            elif equity < self.trapdoor.floor:
                print("[RISK] Trapdoor activated – locking in gains and halting trading.")
                self.halted = True
        if equity <= 0:
            print("[RISK] Equity at or below zero – halting trading to protect capital.")
            self.halted = True

    def enforce_position_limits(self, open_positions: Iterable) -> bool:
        count = sum(1 for _ in open_positions)
        if count >= self.max_concurrent:
            print("[RISK] Max concurrent positions reached.")
            return False
        return True

    def position_size(
        self,
        price: float,
        risk_budget: float,
        *,
        stop_loss: float | None,
        side: str,
    ) -> float:
        """Return quantity that respects the configured risk budget."""

        if price <= 0 or risk_budget <= 0:
            return 0.0

        stop_distance = None
        if stop_loss is not None:
            if side.upper() == "BUY":
                stop_distance = price - stop_loss
            else:
                stop_distance = stop_loss - price
            if stop_distance is not None:
                stop_distance = abs(stop_distance)

        if not stop_distance or stop_distance <= 0:
            stop_distance = price * self.default_stop_pct

        if stop_distance <= 0:
            return 0.0

        qty = risk_budget / stop_distance
        if self.max_position_value and price > 0:
            max_qty = self.max_position_value / price
            qty = min(qty, max_qty)
        return max(qty, 0.0)

