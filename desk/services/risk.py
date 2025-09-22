"""Risk engine implementing trap doors and circuit breakers."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass
class EquityTrapdoor:
    floor: float
    locked_at: float


@dataclass(frozen=True)
class PositionSizingResult:
    """Summary of the sizing decision used for logging and downstream checks."""

    quantity: float
    notional: float
    requested_risk: float
    risk_amount: float
    risk_pct: float
    equity: float
    allocation: float
    min_notional_applied: bool
    min_qty_applied: bool
    minimum_qty: float
    min_notional: float

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "quantity": float(self.quantity),
            "notional": float(self.notional),
            "requested_risk": float(self.requested_risk),
            "risk_amount": float(self.risk_amount),
            "risk_pct": float(self.risk_pct),
            "equity": float(self.equity),
            "allocation": float(self.allocation),
            "min_notional_applied": bool(self.min_notional_applied),
            "min_qty_applied": bool(self.min_qty_applied),
            "minimum_qty": float(self.minimum_qty),
            "min_notional": float(self.min_notional),
        }


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
        equity_floor: float | None = None,
        risk_per_trade_pct: float = 0.03,
        max_risk_per_trade_pct: Optional[float] = 0.04,
        min_notional: float = 10.0,
    ) -> None:
        self.daily_dd = daily_dd or 0.0
        self.weekly_dd = weekly_dd or 0.0
        self.default_stop_pct = max(0.0001, float(default_stop_pct))
        self.max_concurrent = max(1, int(max_concurrent))
        self.halt_on_dd = halt_on_dd
        self.trapdoor_pct = max(0.0, float(trapdoor_pct))
        self.max_position_value = (
            max_position_value if max_position_value and max_position_value > 0 else None
        )
        self.equity_floor = equity_floor if equity_floor and equity_floor > 0 else None
        self.risk_per_trade_pct = max(0.0, float(risk_per_trade_pct))
        self.max_risk_per_trade_pct = (
            float(max_risk_per_trade_pct) if max_risk_per_trade_pct else None
        )
        if self.max_risk_per_trade_pct is not None:
            self.max_risk_per_trade_pct = max(self.max_risk_per_trade_pct, self.risk_per_trade_pct)
        self.min_notional = max(0.0, float(min_notional))

        self.start_equity: float | None = None
        self.equity_high: float | None = None
        self.trapdoor: EquityTrapdoor | None = None
        self.current_equity: float | None = None
        self.halted = False

    # ------------------------------------------------------------------
    def initialise(self, equity: float) -> None:
        self.start_equity = equity
        self.equity_high = equity
        if self.trapdoor_pct > 0:
            self.trapdoor = EquityTrapdoor(
                floor=equity * (1 - self.trapdoor_pct), locked_at=time.time()
            )
        else:
            self.trapdoor = None
        self.current_equity = equity

    def check_account(self, equity: float) -> None:
        if self.start_equity is None:
            self.initialise(equity)
        assert self.start_equity is not None
        self.current_equity = equity

        previous_high = self.equity_high if self.equity_high is not None else equity
        self.equity_high = max(previous_high, equity)

        # Daily/weekly drawdown checks.
        dd_breached = False
        if self.daily_dd and equity < self.start_equity * (1 - self.daily_dd):
            print("[RISK] Daily drawdown threshold breached – monitoring closely.")
            dd_breached = True

        if self.weekly_dd and equity < self.start_equity * (1 - self.weekly_dd):
            print("[RISK] Weekly drawdown threshold breached – monitoring closely.")
            dd_breached = True

        if dd_breached and self.halt_on_dd:
            print("[RISK] Trapdoor activated – drawdown limit reached.")
            self.halted = True
            return

        # Trap door: once equity makes a new high, raise the floor.
        if self.trapdoor:
            if equity > previous_high:
                new_floor = equity * (1 - self.trapdoor_pct)
                self.trapdoor = EquityTrapdoor(floor=new_floor, locked_at=time.time())
            elif equity < self.trapdoor.floor:
                print("[RISK] Trapdoor activated – trailing floor breached.")
                self.halted = True
                return
        if self.equity_floor and equity < self.equity_floor:
            print("[RISK] Trapdoor activated – equity floor reached.")
            self.halted = True
            return
        if equity <= 0:
            print("[RISK] Trapdoor activated – equity at or below zero.")
            self.halted = True

    def enforce_position_limits(self, open_positions: Iterable) -> bool:
        count = sum(1 for _ in open_positions)
        if count >= self.max_concurrent:
            print("[RISK] Max concurrent positions reached.")
            return False
        return True

    def risk_budget(self, allocation: float = 1.0) -> float:
        """Return the USD risk allocation for a single trade."""

        equity = self.current_equity if self.current_equity is not None else self.start_equity
        if equity is None or equity <= 0:
            return 0.0
        allocation = max(0.0, float(allocation))
        base_pct = self.risk_per_trade_pct * allocation
        if self.max_risk_per_trade_pct is not None:
            base_pct = min(base_pct, self.max_risk_per_trade_pct)
        return max(0.0, equity * base_pct)

    def size_position(
        self,
        price: float,
        *,
        stop_loss: float | None,
        side: str,
        allocation: float = 1.0,
        risk_budget: float | None = None,
        minimum_qty: float = 0.0,
        precision: Optional[int] = None,
    ) -> PositionSizingResult:
        """Determine the executable quantity based on equity-driven risk rules."""

        equity = self.current_equity if self.current_equity is not None else self.start_equity
        if price <= 0 or (risk_budget is not None and risk_budget <= 0) or equity is None:
            return PositionSizingResult(
                quantity=0.0,
                notional=0.0,
                requested_risk=0.0 if risk_budget is None else max(risk_budget, 0.0),
                risk_amount=0.0,
                risk_pct=0.0,
                equity=float(equity or 0.0),
                allocation=max(0.0, allocation),
                min_notional_applied=False,
                min_qty_applied=False,
                minimum_qty=max(0.0, minimum_qty),
                min_notional=self.min_notional,
            )

        # If the caller does not provide a risk budget, derive it from equity percentage.
        requested_risk = self.risk_budget(allocation=allocation) if risk_budget is None else max(risk_budget, 0.0)
        if requested_risk <= 0:
            return PositionSizingResult(
                quantity=0.0,
                notional=0.0,
                requested_risk=requested_risk,
                risk_amount=0.0,
                risk_pct=0.0,
                equity=float(equity or 0.0),
                allocation=max(0.0, allocation),
                min_notional_applied=False,
                min_qty_applied=False,
                minimum_qty=max(0.0, minimum_qty),
                min_notional=self.min_notional,
            )

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
            return PositionSizingResult(
                quantity=0.0,
                notional=0.0,
                requested_risk=requested_risk,
                risk_amount=0.0,
                risk_pct=0.0,
                equity=float(equity or 0.0),
                allocation=max(0.0, allocation),
                min_notional_applied=False,
                min_qty_applied=False,
                minimum_qty=max(0.0, minimum_qty),
                min_notional=self.min_notional,
            )

        qty = requested_risk / stop_distance
        min_qty_applied = False
        minimum_qty = max(0.0, minimum_qty)
        if minimum_qty > 0 and qty < minimum_qty:
            qty = minimum_qty
            min_qty_applied = True

        if self.max_position_value and price > 0:
            max_qty = self.max_position_value / price
            qty = min(qty, max_qty)

        min_notional_applied = False
        if self.min_notional > 0 and price > 0:
            min_qty_for_notional = self.min_notional / price
            if qty < min_qty_for_notional:
                qty = min_qty_for_notional
                min_notional_applied = True

        qty = max(qty, 0.0)
        if precision is not None:
            try:
                precision = max(int(precision), 0)
            except (TypeError, ValueError):
                precision = None
        if precision is not None:
            qty = round(qty, precision)
        notional = qty * price
        actual_risk = qty * stop_distance
        risk_pct = actual_risk / equity if equity > 0 else 0.0

        return PositionSizingResult(
            quantity=qty,
            notional=notional,
            requested_risk=requested_risk,
            risk_amount=actual_risk,
            risk_pct=risk_pct,
            equity=equity,
            allocation=max(0.0, allocation),
            min_notional_applied=min_notional_applied,
            min_qty_applied=min_qty_applied,
            minimum_qty=minimum_qty,
            min_notional=self.min_notional,
        )

    def position_size(
        self,
        price: float,
        risk_budget: float | None,
        *,
        stop_loss: float | None,
        side: str,
        allocation: float = 1.0,
        minimum_qty: float = 0.0,
    ) -> float:
        """Return quantity that respects the configured risk budget."""

        sizing = self.size_position(
            price,
            stop_loss=stop_loss,
            side=side,
            allocation=allocation,
            risk_budget=risk_budget,
            minimum_qty=minimum_qty,
        )
        return sizing.quantity

