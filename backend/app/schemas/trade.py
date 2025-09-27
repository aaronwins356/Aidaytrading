"""Trading related schemas."""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import List

from pydantic import BaseModel


class TradeOut(BaseModel):
    id: int
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    pnl: Decimal
    balance: Decimal
    executed_at: datetime
    strategy: str

    class Config:
        from_attributes = True


class TradePage(BaseModel):
    items: List[TradeOut]
    total: int


class EquityPointOut(BaseModel):
    timestamp: datetime
    value: Decimal

    class Config:
        from_attributes = True


class ProfitSummary(BaseModel):
    equity: Decimal
    pnl_absolute: Decimal
    pnl_percent: Decimal
    win_rate: float


class CalendarEntry(BaseModel):
    trading_day: str
    pnl: Decimal
    pnl_percent: Decimal
    win_rate: Decimal

    class Config:
        from_attributes = True


class BalanceSnapshotOut(BaseModel):
    timestamp: datetime
    balance: Decimal
    equity: Decimal

    class Config:
        from_attributes = True


class StatusResponse(BaseModel):
    running: bool
    mode: str
    uptime_seconds: float | None
