"""Machine-learning assisted shorting worker."""

from __future__ import annotations

from typing import Dict, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..services.ml import MLService

from .base import BaseWorker
from ..services.types import MarketSnapshot, OpenPosition, TradeIntent


class MLShortWorker(BaseWorker):
    """Uses an online logistic model to size short entries."""

    name = "ML Short Alpha"
    emoji = "🧠"

    def __init__(
        self,
        symbols,
        ml_service: "MLService",
        config: Optional[Dict] = None,
        risk_config: Optional[Dict] = None,
    ) -> None:
        self._ml_service = ml_service
        self.feature_lookback = int((config or {}).get("feature_lookback", 30))
        lookback = max(self.feature_lookback * 2, int((config or {}).get("lookback", self.feature_lookback * 2)))
        super().__init__(symbols=symbols, lookback=lookback, config=config, risk_config=risk_config, ml_service=ml_service)
        self.threshold = float((config or {}).get("probability_threshold", 0.6))
        self.cover_threshold = float((config or {}).get("cover_threshold", 0.48))
        self._last_probabilities: Dict[str, float] = {}

    async def evaluate_signal(self, snapshot: MarketSnapshot) -> Dict[str, str]:
        self.update_history(snapshot)
        signals: Dict[str, str] = {}
        for symbol in self.symbols:
            feature_payload = self._ml_service.latest_features(symbol)
            if feature_payload is None:
                self.update_signal_state(symbol, None, {"status": "awaiting-features"})
                continue
            decision, probability = self._ml_service.predict(
                symbol,
                feature_payload,
                worker=self.name,
                threshold=self.threshold,
            )
            self._last_probabilities[symbol] = probability
            signal: Optional[str] = None
            if decision and probability >= self.threshold and self.is_ready(symbol):
                signal = "sell"
            elif probability <= self.cover_threshold:
                signal = "buy"
            indicator_state = {
                "probability": probability,
                "features": feature_payload,
                "ml_decision": decision,
            }
            self.update_signal_state(symbol, signal, indicator_state)
            if signal:
                signals[symbol] = signal
        return signals

    async def generate_trade(
        self,
        symbol: str,
        signal: Optional[str],
        snapshot: MarketSnapshot,
        equity_per_trade: float,
        existing_position: Optional[OpenPosition] = None,
    ) -> Optional[TradeIntent]:
        price = snapshot.prices.get(symbol)
        if price is None:
            return None

        probability = self._last_probabilities.get(symbol, 0.5)
        features = self._ml_service.latest_features(symbol)
        if existing_position is None:
            if signal != "sell" or not self.is_ready(symbol):
                return None
            cash = equity_per_trade * (self.position_size_pct / 100)
            if cash <= 0:
                return None
            allowed, ml_confidence = self.ml_confirmation(symbol, features=features, threshold=self.threshold)
            if not allowed:
                self.update_signal_state(symbol, "ml-block", {"ml_confidence": ml_confidence})
                return None
            confidence = ml_confidence or min(1.0, max(0.0, probability - 0.5))
            return TradeIntent(
                worker=self.name,
                action="OPEN",
                symbol=symbol,
                side="sell",
                cash_spent=cash * self.leverage,
                entry_price=price,
                confidence=confidence,
            )

        close_signal = False
        reason = ""
        if signal == "buy":
            close_signal = True
            reason = "prob-revert"
        elif probability <= self.cover_threshold:
            close_signal = True
            reason = "prob-floor"

        if self.stop_loss_pct:
            stop_price = existing_position.entry_price * (1 + self.stop_loss_pct / 100)
            if price >= stop_price:
                close_signal = True
                reason = "stop"
        if self.take_profit_pct:
            target_price = existing_position.entry_price * (1 - self.take_profit_pct / 100)
            if price <= target_price:
                close_signal = True
                reason = "target"

        if close_signal:
            self.update_signal_state(symbol, f"close:{reason}")
            return TradeIntent(
                worker=self.name,
                action="CLOSE",
                symbol=symbol,
                side="buy",
                cash_spent=existing_position.cash_spent,
                entry_price=existing_position.entry_price,
                exit_price=price,
                confidence=probability,
            )

        return None
