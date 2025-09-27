"""Risk schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field


class RiskSettings(BaseModel):
    max_drawdown_percent: float = Field(ge=5.0, le=90.0)
    daily_loss_limit_percent: float = Field(ge=1.0, le=50.0)
    risk_per_trade: float = Field(ge=0.005, le=0.1)
    max_open_positions: int = Field(ge=1, le=10)
    atr_stop_loss_multiplier: float = Field(ge=0.5, le=3.0)
    atr_take_profit_multiplier: float = Field(ge=1.0, le=5.0)

    class Config:
        from_attributes = True
