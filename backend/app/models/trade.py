"""Models related to trading activity and telemetry."""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base


class Trade(Base):
    """Executed trade record."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(4), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    price: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    pnl: Mapped[Decimal] = mapped_column(Numeric(18, 8), default=Decimal("0"))
    balance: Mapped[Decimal] = mapped_column(Numeric(18, 2), default=Decimal("0"))
    executed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    strategy: Mapped[str] = mapped_column(String(64), nullable=False, default="global")


class EquityPoint(Base):
    """Equity curve data sampled at regular intervals."""

    __tablename__ = "equity_points"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    value: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, index=True)


class BalanceSnapshot(Base):
    """Account balance snapshots for streaming."""

    __tablename__ = "balance_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    balance: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    equity: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, index=True)


class DailyPnL(Base):
    """Daily profit and loss aggregated for calendar view."""

    __tablename__ = "daily_pnl"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trading_day: Mapped[str] = mapped_column(String(10), unique=True, nullable=False)
    pnl: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    pnl_percent: Mapped[Decimal] = mapped_column(Numeric(8, 4), nullable=False)
    win_rate: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False)


class BotState(Base):
    """Singleton table representing bot runtime state."""

    __tablename__ = "bot_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    running: Mapped[bool] = mapped_column(default=False)
    mode: Mapped[str] = mapped_column(String(16), default="paper")
    uptime_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SchedulerRun(Base):
    """Track scheduler executions for observability."""

    __tablename__ = "scheduler_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_name: Mapped[str] = mapped_column(String(64), nullable=False)
    ran_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    status: Mapped[str] = mapped_column(String(32), default="success")
    detail: Mapped[str | None] = mapped_column(String(256))
