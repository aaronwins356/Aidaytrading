"""Common data structures shared across trading services."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Literal, Optional

TradeAction = Literal["OPEN", "CLOSE"]
TradeSide = Literal["buy", "sell"]


@dataclass(slots=True)
class TradeIntent:
    """Structured intent returned by a worker.

    The broker integrations occasionally return numeric fields as strings, and
    configuration files may also express numbers using quoted scalars. To avoid
    subtle bugs when these values participate in arithmetic, we normalise the
    core numeric attributes to floats immediately after initialisation.
    """

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
    reason: Optional[str] = None
    metadata: Optional[Dict[str, object]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        """Coerce numeric fields to floats for safe downstream arithmetic."""

        self.cash_spent = float(self.cash_spent)
        self.entry_price = float(self.entry_price)
        if self.exit_price is not None:
            self.exit_price = float(self.exit_price)
        if self.pnl_percent is not None:
            self.pnl_percent = float(self.pnl_percent)
        if self.pnl_usd is not None:
            self.pnl_usd = float(self.pnl_usd)
        self.confidence = float(self.confidence)


@dataclass(slots=True)
class OpenPosition:
    """Simple representation of an open position managed by the engine."""

    worker: str
    symbol: str
    side: TradeSide
    quantity: float
    entry_price: float
    cash_spent: float
    fees_paid: float = 0.0
    opened_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        """Normalise numeric fields to floats for consistent arithmetic."""

        self.quantity = float(self.quantity)
        self.entry_price = float(self.entry_price)
        self.cash_spent = float(self.cash_spent)
        self.fees_paid = float(self.fees_paid)

    def unrealized_pnl(self, last_price: float) -> float:
        """Return the unrealized profit/loss for the position."""

        if self.side == "buy":
            gross = (last_price - self.entry_price) * self.quantity
        else:
            gross = (self.entry_price - last_price) * self.quantity
        return gross - self.fees_paid


@dataclass(slots=True)
class MarketSnapshot:
    """Represents the latest market data available to workers."""

    prices: dict[str, float]
    history: dict[str, list[float]]
    candles: Dict[str, List[dict[str, float]]] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
