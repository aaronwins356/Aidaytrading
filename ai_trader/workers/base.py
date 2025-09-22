"""Base worker definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional

from ..services.types import MarketSnapshot, OpenPosition, TradeIntent


class BaseWorker(ABC):
    """Abstract interface that all workers must implement."""

    name: str = "BaseWorker"
    emoji: str = "🤖"

    def __init__(self, symbols: Iterable[str], lookback: int = 50) -> None:
        self.symbols: List[str] = list(symbols)
        self.lookback = lookback
        self.price_history: Dict[str, Deque[float]] = {
            symbol: deque(maxlen=lookback) for symbol in self.symbols
        }
        self.active: bool = True

    def update_history(self, snapshot: MarketSnapshot) -> None:
        for symbol, price in snapshot.prices.items():
            if symbol not in self.price_history:
                self.price_history[symbol] = deque(maxlen=self.lookback)
            self.price_history[symbol].append(price)

    @abstractmethod
    async def evaluate_signal(self, snapshot: MarketSnapshot) -> Dict[str, str]:
        """Analyse market data and return per-symbol signals."""

    @abstractmethod
    async def generate_trade(
        self,
        symbol: str,
        signal: Optional[str],
        snapshot: MarketSnapshot,
        equity_per_trade: float,
        existing_position: Optional[OpenPosition] = None,
    ) -> Optional[TradeIntent]:
        """Translate a signal into a trade intent."""

    def deactivate(self) -> None:
        self.active = False

    def activate(self) -> None:
        self.active = True
