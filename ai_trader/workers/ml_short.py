"""Machine-learning assisted shorting worker."""

from __future__ import annotations

from typing import Dict, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ai_trader.services.ml import MLService
    from ai_trader.services.trade_log import TradeLog

from ai_trader.services.types import MarketSnapshot, OpenPosition, TradeIntent
from ai_trader.workers.base import BaseWorker


class MLShortWorker(BaseWorker):
    """Uses an online logistic model to size short entries."""

    name = "ML Short Alpha"
    emoji = "🧠"

    def __init__(
        self,
        symbols,
        ml_service: "MLService",
        trade_log: "TradeLog" | None = None,
        config: Optional[Dict] = None,
        risk_config: Optional[Dict] = None,
    ) -> None:
        self._ml_service = ml_service
        self.feature_lookback = int((config or {}).get("feature_lookback", 30))
        lookback = max(
            self.feature_lookback * 2,
            int((config or {}).get("lookback", self.feature_lookback * 2)),
        )
        super().__init__(
            symbols=symbols,
            lookback=lookback,
            config=config,
            risk_config=risk_config,
            ml_service=ml_service,
            trade_log=trade_log,
        )
        self.threshold = float((config or {}).get("probability_threshold", 0.6))
        self.cover_threshold = float((config or {}).get("cover_threshold", 0.48))
        self._last_probabilities: Dict[str, float] = {}

    async def evaluate_signal(self, snapshot: MarketSnapshot) -> Dict[str, str]:
        self.update_history(snapshot)
        signals: Dict[str, str] = {}
        for symbol in self.symbols:
            feature_payload = self._ml_service.latest_features(symbol)
            if feature_payload is None:
                self._logger.warning(
                    "ML Short Alpha waiting for features from researcher for %s", symbol
                )
                self.update_signal_state(symbol, None, {"status": "awaiting-features"})
                continue
            decision, probability = self._ml_service.predict(
                symbol,
                feature_payload,
                worker=self.name,
                threshold=self.threshold,
            )
            self._last_probabilities[symbol] = probability
            self._logger.debug(
                "Prediction for %s -> probability=%.4f decision=%s (threshold=%.4f)",
                symbol,
                probability,
                decision,
                self.threshold,
            )
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
            cash = float(equity_per_trade) * (self.position_size_pct / 100)
            if cash <= 0:
                return None
            allowed, ml_confidence = self.ml_confirmation(
                symbol, features=features, threshold=self.threshold
            )
            if not allowed:
                self.update_signal_state(symbol, "ml-block", {"ml_confidence": ml_confidence})
                return None
            confidence = ml_confidence or min(1.0, max(0.0, probability - 0.5))
            metadata = {
                "probability": probability,
                "features_used": list((features or {}).keys()),
            }
            risk_meta = self.prepare_entry_risk(symbol, "sell", price)
            metadata.update({k: v for k, v in risk_meta.items() if v is not None})
            self.record_trade_event(
                "open_signal",
                symbol,
                {
                    "probability": probability,
                    "threshold": self.threshold,
                    "cover_threshold": self.cover_threshold,
                },
            )
            cash_value = float(cash) * float(self.leverage)
            return TradeIntent(
                worker=self.name,
                action="OPEN",
                symbol=symbol,
                side="sell",
                cash_spent=cash_value,
                entry_price=price,
                confidence=confidence,
                metadata=metadata,
            )

        risk_triggered, risk_reason, risk_meta = self.check_risk_exit(symbol, price)
        if risk_triggered:
            self.update_signal_state(symbol, f"close:{risk_reason}")
            self.clear_risk_tracker(symbol)
            payload = {
                "probability": probability,
                "trigger": risk_reason,
                **{k: v for k, v in risk_meta.items() if v is not None},
            }
            return TradeIntent(
                worker=self.name,
                action="CLOSE",
                symbol=symbol,
                side="buy",
                cash_spent=existing_position.cash_spent,
                entry_price=existing_position.entry_price,
                exit_price=price,
                confidence=probability,
                reason=risk_reason,
                metadata=payload,
            )

        close_signal = False
        reason = ""
        if signal == "buy":
            close_signal = True
            reason = "prob-revert"
        elif probability <= self.cover_threshold:
            close_signal = True
            reason = "prob-floor"

        if close_signal:
            self.update_signal_state(symbol, f"close:{reason}")
            if reason in {"stop", "target", "prob-revert", "prob-floor"}:
                self.record_trade_event(
                    f"close_{reason}",
                    symbol,
                    {
                        "probability": probability,
                        "confidence": self._last_probabilities.get(symbol, probability),
                    },
                )
            self.clear_risk_tracker(symbol)
            return TradeIntent(
                worker=self.name,
                action="CLOSE",
                symbol=symbol,
                side="buy",
                cash_spent=existing_position.cash_spent,
                entry_price=existing_position.entry_price,
                exit_price=price,
                confidence=probability,
                reason=reason,
                metadata={
                    "probability": probability,
                    "trigger": reason,
                    **{k: v for k, v in risk_meta.items() if v is not None},
                },
            )

        return None
