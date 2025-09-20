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
        trade_stop_loss: float,
        max_concurrent: int,
        halt_on_dd: bool,
        trapdoor_pct: float,
    ) -> None:
        self.daily_dd = daily_dd or 0.0
        self.weekly_dd = weekly_dd or 0.0
        self.trade_stop_loss = trade_stop_loss
        self.max_concurrent = max_concurrent
        self.halt_on_dd = halt_on_dd
        self.trapdoor_pct = trapdoor_pct

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

        self.equity_high = max(self.equity_high or equity, equity)

        # Daily/weekly drawdown checks.
        if self.daily_dd and equity < self.start_equity * (1 - self.daily_dd):
            print("[RISK] Daily drawdown threshold breached.")
            if self.halt_on_dd:
                self.halted = True

        if self.weekly_dd and equity < self.start_equity * (1 - self.weekly_dd):
            print("[RISK] Weekly drawdown threshold breached.")
            if self.halt_on_dd:
                self.halted = True

        # Trap door: once equity makes a new high, raise the floor.
        if self.trapdoor:
            if equity > (self.equity_high or equity):
                new_floor = equity * (1 - self.trapdoor_pct)
                self.trapdoor = EquityTrapdoor(floor=new_floor, locked_at=time.time())
            elif equity < self.trapdoor.floor:
                print("[RISK] Trapdoor activated – locking in gains and halting trading.")
                self.halted = True

    def enforce_position_limits(self, open_positions: Iterable) -> bool:
        count = sum(1 for _ in open_positions)
        if count >= self.max_concurrent:
            print("[RISK] Max concurrent positions reached.")
            return False
        return True

    def per_trade_notional(self, price: float, allocation_usd: float) -> float:
        if price <= 0:
            return 0.0
        max_loss = allocation_usd * self.trade_stop_loss
        qty = max_loss / max(price * self.trade_stop_loss, 1e-9)
        return qty

