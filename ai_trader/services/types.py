"""Common data structures shared across trading services."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional

TradeAction = Literal["OPEN", "CLOSE"]
TradeSide = Literal["buy", "sell"]


@dataclass(slots=True)
class TradeIntent:
    """Structured intent returned by a worker."""

    worker: str
    action: TradeAction
    symbol: str
    side: TradeSide
    cash_spent: float
    entry_price: float
    exit_price: Optional[float] = None
    pnl_percent: Optional[float] = None
    pnl_usd: Optional[float] = None
    win_loss: Optional[str] = None
    confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class OpenPosition:
    """Simple representation of an open position managed by the engine."""

    worker: str
    symbol: str
    side: TradeSide
    quantity: float
    entry_price: float
    cash_spent: float
    opened_at: datetime = field(default_factory=datetime.utcnow)

    def unrealized_pnl(self, last_price: float) -> float:
        """Return the unrealized profit/loss for the position."""

        if self.side == "buy":
            return (last_price - self.entry_price) * self.quantity
        return (self.entry_price - last_price) * self.quantity


@dataclass(slots=True)
class MarketSnapshot:
    """Represents the latest market data available to workers."""

    prices: dict[str, float]
    history: dict[str, list[float]]
    timestamp: datetime = field(default_factory=datetime.utcnow)
