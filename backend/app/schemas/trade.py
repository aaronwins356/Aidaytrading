"""Trading related schemas."""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, List

from pydantic import BaseModel, ConfigDict, model_validator


class TradeOut(BaseModel):
    """Serialized trade record matching the mobile contract."""

    id: int
    symbol: str
    side: str
    size: Decimal
    pnl: Decimal
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)

    @model_validator(mode="before")
    @classmethod
    def from_trade(cls, value: Any) -> Any:
        """Normalize ORM trades into the public schema."""

        # Deferred import to avoid circular dependencies at module import time.
        from ..models.trade import Trade

        if isinstance(value, Trade):
            return {
                "id": value.id,
                "symbol": value.symbol,
                "side": value.side,
                "size": value.quantity,
                "pnl": value.pnl,
                "timestamp": value.executed_at,
            }
        return value


class TradePage(BaseModel):
    """Paginated trade response."""

    items: List[TradeOut]
    page: int
    page_size: int
    total: int

    model_config = ConfigDict(from_attributes=True)


class ProfitSummary(BaseModel):
    current_balance: Decimal
    total_pl_amount: Decimal
    total_pl_percent: Decimal
    win_rate: float

    model_config = ConfigDict(from_attributes=True)


class CalendarEntry(BaseModel):
    trading_day: str
    pnl: Decimal
    pnl_percent: Decimal
    win_rate: Decimal

    model_config = ConfigDict(from_attributes=True)


class BalanceSnapshotOut(BaseModel):
    timestamp: datetime
    balance: Decimal
    equity: Decimal

    model_config = ConfigDict(from_attributes=True)


class StatusResponse(BaseModel):
    running: bool
    mode: str
    uptime_seconds: float | None
