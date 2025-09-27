"""Trade execution models."""
from __future__ import annotations

import datetime as dt
from decimal import Decimal
from enum import Enum

from sqlalchemy import DateTime, Enum as PgEnum, Index, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class TradeSide(str, Enum):
    """Enumerated trade sides."""

    BUY = "buy"
    SELL = "sell"
    SHORT = "short"
    COVER = "cover"


class Trade(Base):
    """Recorded trade including realised P&L."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[TradeSide] = mapped_column(PgEnum(TradeSide, name="trade_side"), nullable=False)
    size: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    pnl: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    timestamp: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)

    __table_args__ = (
        Index("ix_trades_symbol_timestamp", "symbol", "timestamp"),
    )

