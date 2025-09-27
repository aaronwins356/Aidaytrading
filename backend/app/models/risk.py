"""Risk setting model."""
from __future__ import annotations

from sqlalchemy import Float, Integer
from sqlalchemy.orm import Mapped, mapped_column

from ..database import Base


class RiskSetting(Base):
    """Singleton table storing global risk configuration."""

    __tablename__ = "risk_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    max_drawdown_percent: Mapped[float] = mapped_column(Float, default=20.0)
    daily_loss_limit_percent: Mapped[float] = mapped_column(Float, default=5.0)
    risk_per_trade: Mapped[float] = mapped_column(Float, default=0.01)
    max_open_positions: Mapped[int] = mapped_column(Integer, default=3)
    atr_stop_loss_multiplier: Mapped[float] = mapped_column(Float, default=1.5)
    atr_take_profit_multiplier: Mapped[float] = mapped_column(Float, default=3.0)
