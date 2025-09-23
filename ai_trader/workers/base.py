"""Base worker definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ai_trader.services.ml import MLService
    from ai_trader.services.trade_log import TradeLog

from ai_trader.services.logging import get_logger
from ai_trader.services.types import MarketSnapshot, OpenPosition, TradeIntent


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
        ml_service: "MLService" | None = None,
        trade_log: "TradeLog" | None = None,
    ) -> None:
        self.symbols: List[str] = list(symbols)
        self.lookback = lookback
        self.price_history: Dict[str, Deque[float]] = {
            symbol: deque(maxlen=lookback) for symbol in self.symbols
        }
        self.active: bool = True
        self.config: Dict[str, Any] = config or {}
        self.risk_config: Dict[str, Any] = risk_config or {}
        self.warmup_candles: int = max(1, int(self.config.get("warmup_candles", 2)))
        self.position_size_pct: float = float(self.risk_config.get("position_size_pct", 100.0))
        self.leverage: float = float(self.risk_config.get("leverage", 1.0))
        self.stop_loss_pct: float = float(self.risk_config.get("stop_loss_pct", 0.0))
        self.take_profit_pct: float = float(self.risk_config.get("take_profit_pct", 0.0))
        self.trailing_stop_pct: float = float(self.risk_config.get("trailing_stop_pct", 0.0))
        self._latest_signals: Dict[str, Optional[str]] = {symbol: None for symbol in self.symbols}
        self._state: Dict[str, Dict[str, Any]] = {}
        self._ml_service = ml_service
        self._trade_log = trade_log
        self._ml_gate_enabled: bool = True
        self._logger = get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._ml_threshold_override: Optional[float] = None
        self._config_ml_threshold: Optional[float] = None
        threshold_override = self.config.get("ml_threshold")
        if threshold_override is not None:
            try:
                value = float(threshold_override)
                self._ml_threshold_override = value
                self._config_ml_threshold = value
            except (TypeError, ValueError):
                self._logger.warning(
                    "Invalid ml_threshold for %s; using service default.", self.name
                )
                self._ml_threshold_override = None
        self._warmup_notified: Dict[str, bool] = {symbol: False for symbol in self.symbols}
        self._ml_warmup_state: Dict[str, bool] = {symbol: True for symbol in self.symbols}

    def update_history(self, snapshot: MarketSnapshot) -> None:
        for symbol, price in snapshot.prices.items():
            if symbol not in self.price_history:
                self.price_history[symbol] = deque(maxlen=self.lookback)
            self.price_history[symbol].append(price)

    def is_ready(self, symbol: str) -> bool:
        """Return True when the worker has enough candles to act."""

        history = self.price_history.get(symbol, [])
        ready = len(history) >= self.warmup_candles
        if not ready and not self._warmup_notified.get(symbol, False):
            self._logger.info(
                "Warmup active for %s on %s – need %d candles, have %d",
                symbol,
                self.name,
                self.warmup_candles,
                len(history),
            )
            self._warmup_notified[symbol] = True
        elif ready:
            self._warmup_notified[symbol] = False
        return ready

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
        if self._ml_service is not None:
            snapshot.setdefault("ml_warming_up", self._ml_warmup_state.get(symbol, True))
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

    def apply_control_flags(self, flags: Dict[str, str]) -> None:
        gate_flag = flags.get(f"ml::{self.name}")
        if gate_flag:
            self._ml_gate_enabled = gate_flag.lower() not in {"off", "false", "0", "disabled"}
        threshold_flag = flags.get(f"ml::{self.name}::threshold")
        if threshold_flag:
            try:
                self._ml_threshold_override = float(threshold_flag)
            except (TypeError, ValueError):
                self._ml_threshold_override = None
        else:
            self._ml_threshold_override = self._config_ml_threshold

    async def observe(
        self,
        snapshot: MarketSnapshot,
        equity_metrics: Optional[Dict[str, Any]] = None,
        open_positions: Optional[List[OpenPosition]] = None,
    ) -> None:
        """Optional hook for researcher-style workers."""

        # No-op by default. Sub-classes may override.
        _ = (snapshot, equity_metrics, open_positions)

    # ------------------------------------------------------------------
    # Machine learning gating helpers
    # ------------------------------------------------------------------
    def ml_confirmation(
        self,
        symbol: str,
        *,
        features: Optional[Mapping[str, float]] = None,
        threshold: Optional[float] = None,
    ) -> tuple[bool, float]:
        """Return whether ML gating approves the trade and the associated confidence."""

        if self._ml_service is None or not self._ml_gate_enabled:
            return True, 0.0
        feature_payload = features or self._ml_service.latest_features(symbol)
        if feature_payload is None:
            self._ml_warmup_state[symbol] = True
            self._logger.info(
                "ML gating warming up for %s on %s – awaiting feature pipeline",
                self.name,
                symbol,
            )
            return False, 0.0
        self._ml_warmup_state[symbol] = False
        gate = (
            threshold
            if threshold is not None
            else self._ml_threshold_override or self._ml_service.default_threshold
        )
        decision, confidence = self._ml_service.predict(
            symbol, feature_payload, worker=self.name, threshold=gate
        )
        if not decision:
            self._logger.info(
                "ML gate blocked %s signal on %s (confidence=%.3f threshold=%.3f)",
                self.name,
                symbol,
                confidence,
                gate,
        )
        return decision, confidence

    # ------------------------------------------------------------------
    # Trade log helper
    # ------------------------------------------------------------------
    def record_trade_event(self, event: str, symbol: str, details: Optional[Dict[str, object]] = None) -> None:
        """Persist worker-specific events when a trade log is available."""

        if self._trade_log is None:
            return
        try:
            self._trade_log.record_trade_event(
                worker=self.name,
                symbol=symbol,
                event=event,
                details=details or {},
            )
        except Exception as exc:  # noqa: BLE001 - keep workers resilient
            self._logger.debug("Failed to record trade event %s for %s: %s", event, symbol, exc)
