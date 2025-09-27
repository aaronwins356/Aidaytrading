"""Reporting and analytics data models."""
from __future__ import annotations

import datetime as dt
from decimal import Decimal
from enum import Enum

from sqlalchemy import Date, DateTime, Enum as PgEnum, Index, Numeric, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class EquitySnapshot(Base):
    """Point-in-time equity measurement."""

    __tablename__ = "equity_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    timestamp: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    equity: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="bot")


class DailyPnLColor(str, Enum):
    """Colour coding for calendar heatmap."""

    GREEN = "green"
    RED = "red"
    NEUTRAL = "neutral"


class DailyPnL(Base):
    """Aggregated profit and loss per calendar day."""

    __tablename__ = "daily_pnl"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    date: Mapped[dt.date] = mapped_column(Date(), nullable=False, unique=True, index=True)
    pnl_amount: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    color: Mapped[DailyPnLColor] = mapped_column(
        PgEnum(DailyPnLColor, name="daily_pnl_color"), nullable=False
    )

    __table_args__ = (UniqueConstraint("date", name="uq_daily_pnl_date"),)


class SystemStatus(Base):
    """Singleton row describing trading system lifecycle."""

    __tablename__ = "system_status"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    running: Mapped[bool] = mapped_column(nullable=False, default=False)
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    stopped_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (Index("ix_system_status_singleton", "id"),)

