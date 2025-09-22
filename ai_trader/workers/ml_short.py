"""Machine-learning assisted shorting worker."""

from __future__ import annotations

from typing import Dict, Optional

from .base import BaseWorker
from ..services.ml_pipeline import MLPipeline
from ..services.types import MarketSnapshot, OpenPosition, TradeIntent


class MLShortWorker(BaseWorker):
    """Uses an online logistic model to size short entries."""

    name = "ML Short Alpha"
    emoji = "🧠"

    def __init__(
        self,
        symbols,
        pipeline: MLPipeline,
        config: Optional[Dict] = None,
        risk_config: Optional[Dict] = None,
    ) -> None:
        self._pipeline = pipeline
        self.feature_lookback = int((config or {}).get("feature_lookback", 30))
        lookback = max(self.feature_lookback * 2, int((config or {}).get("lookback", self.feature_lookback * 2)))
        super().__init__(symbols=symbols, lookback=lookback, config=config, risk_config=risk_config)
        self.threshold = float((config or {}).get("probability_threshold", 0.6))
        self.cover_threshold = float((config or {}).get("cover_threshold", 0.48))
        self._last_probabilities: Dict[str, float] = {}

    async def evaluate_signal(self, snapshot: MarketSnapshot) -> Dict[str, str]:
        self.update_history(snapshot)
        signals: Dict[str, str] = {}
        for symbol in self.symbols:
            feature_vector = self._pipeline.latest_feature(symbol)
            if feature_vector is None:
                self.update_signal_state(symbol, None, {"status": "awaiting-features"})
                continue
            self._pipeline.train(symbol)
            probability = self._pipeline.predict(symbol, feature_vector.features)
            self._last_probabilities[symbol] = probability
            signal: Optional[str] = None
            if probability >= self.threshold and self.is_ready(symbol):
                signal = "sell"
            elif probability <= self.cover_threshold:
                signal = "buy"
            indicator_state = {
                "probability": probability,
                "label": feature_vector.label,
                "features": dict(zip(self._pipeline.feature_keys, feature_vector.features)),
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
        if existing_position is None:
            if signal != "sell" or not self.is_ready(symbol):
                return None
            cash = equity_per_trade * (self.position_size_pct / 100)
            if cash <= 0:
                return None
            confidence = min(1.0, max(0.0, probability - 0.5))
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
