"""Pydantic schemas for reporting APIs."""
from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Dict, List

from pydantic import BaseModel, ConfigDict, Field

from app.models.device_token import DevicePlatform
from app.models.trade import TradeSide


decimal_encoders = {Decimal: lambda value: format(value, "f")}


class ProfitResponse(BaseModel):
    current_balance: Decimal
    total_pl_amount: Decimal
    total_pl_percent: Decimal
    win_rate: float

    model_config = ConfigDict(json_encoders=decimal_encoders)


class CalendarDay(BaseModel):
    pnl: Decimal
    color: str

    model_config = ConfigDict(json_encoders=decimal_encoders)


class StatusResponse(BaseModel):
    running: bool
    uptime_seconds: int


class TradeRecord(BaseModel):
    id: int
    symbol: str
    side: TradeSide
    size: Decimal
    pnl: Decimal
    timestamp: dt.datetime

    model_config = ConfigDict(json_encoders=decimal_encoders)


class TradesResponse(BaseModel):
    items: List[TradeRecord]
    page: int
    page_size: int
    total: int


class DeviceRegistrationRequest(BaseModel):
    token: str = Field(min_length=10, max_length=255)
    platform: DevicePlatform = Field(default=DevicePlatform.IOS)


class DeviceRegistrationResponse(BaseModel):
    message: str


class EquityCurvePoint(BaseModel):
    timestamp: dt.datetime
    equity: Decimal

    model_config = ConfigDict(json_encoders=decimal_encoders)


EquityCurveResponse = List[List[str]]
CalendarResponse = Dict[str, CalendarDay]

__all__ = [
    "ProfitResponse",
    "CalendarDay",
    "CalendarResponse",
    "StatusResponse",
    "TradeRecord",
    "TradesResponse",
    "DeviceRegistrationRequest",
    "DeviceRegistrationResponse",
    "EquityCurvePoint",
    "EquityCurveResponse",
]

