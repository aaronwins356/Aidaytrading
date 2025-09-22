"""Base worker definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Optional

from ..services.types import MarketSnapshot, OpenPosition, TradeIntent


class BaseWorker(ABC):
    """Abstract interface that all workers must implement."""

    name: str = "BaseWorker"
    emoji: str = "🤖"
    is_researcher: bool = False

    def __init__(
        self,
        symbols: Iterable[str],
        lookback: int = 50,
        config: Optional[Dict[str, Any]] = None,
        risk_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.symbols: List[str] = list(symbols)
        self.lookback = lookback
        self.price_history: Dict[str, Deque[float]] = {
            symbol: deque(maxlen=lookback) for symbol in self.symbols
        }
        self.active: bool = True
        self.config: Dict[str, Any] = config or {}
        self.risk_config: Dict[str, Any] = risk_config or {}
        self.warmup_candles: int = max(1, int(self.config.get("warmup_candles", 10)))
        self.position_size_pct: float = float(self.risk_config.get("position_size_pct", 100.0))
        self.leverage: float = float(self.risk_config.get("leverage", 1.0))
        self.stop_loss_pct: float = float(self.risk_config.get("stop_loss_pct", 0.0))
        self.take_profit_pct: float = float(self.risk_config.get("take_profit_pct", 0.0))
        self.trailing_stop_pct: float = float(self.risk_config.get("trailing_stop_pct", 0.0))
        self._latest_signals: Dict[str, Optional[str]] = {symbol: None for symbol in self.symbols}
        self._state: Dict[str, Dict[str, Any]] = {}

    def update_history(self, snapshot: MarketSnapshot) -> None:
        for symbol, price in snapshot.prices.items():
            if symbol not in self.price_history:
                self.price_history[symbol] = deque(maxlen=self.lookback)
            self.price_history[symbol].append(price)

    def is_ready(self, symbol: str) -> bool:
        """Return True when the worker has enough candles to act."""

        history = self.price_history.get(symbol, [])
        return len(history) >= self.warmup_candles

    def update_signal_state(
        self, symbol: str, signal: Optional[str], indicators: Optional[Dict[str, Any]] = None
    ) -> None:
        """Persist the latest signal and indicator snapshot for dashboards."""

        self._latest_signals[symbol] = signal
        state = self._state.setdefault(symbol, {})
        state["last_signal"] = signal
        if indicators:
            state.setdefault("indicators", {}).update(indicators)
        state["leverage"] = self.leverage
        state["position_size_pct"] = self.position_size_pct
        state["stop_loss_pct"] = self.stop_loss_pct
        state["take_profit_pct"] = self.take_profit_pct
        state["trailing_stop_pct"] = self.trailing_stop_pct

    def get_state_snapshot(self, symbol: str) -> Dict[str, Any]:
        """Return the last-known state for a symbol."""

        snapshot = dict(self._state.get(symbol, {}))
        snapshot.setdefault("status", "ready" if self.is_ready(symbol) else "warmup")
        snapshot.setdefault("last_signal", self._latest_signals.get(symbol))
        return snapshot

    def get_all_state_snapshots(self) -> Dict[str, Dict[str, Any]]:
        """Expose the entire state dictionary for persistence."""

        return {symbol: self.get_state_snapshot(symbol) for symbol in self.symbols}

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

    async def observe(
        self,
        snapshot: MarketSnapshot,
        equity_metrics: Optional[Dict[str, Any]] = None,
        open_positions: Optional[List[OpenPosition]] = None,
    ) -> None:
        """Optional hook for researcher-style workers."""

        # No-op by default. Sub-classes may override.
        _ = (snapshot, equity_metrics, open_positions)
