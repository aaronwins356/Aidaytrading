"""Mean reversion short specialist."""

from __future__ import annotations

from statistics import fmean, pstdev
from typing import Dict, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ai_trader.services.ml import MLService
    from ai_trader.services.trade_log import TradeLog

from ai_trader.services.types import MarketSnapshot, OpenPosition, TradeIntent
from ai_trader.workers.base import BaseWorker


class ShortMeanReversionWorker(BaseWorker):
    """Fades euphoric spikes expecting price to mean revert."""

    name = "Reversion Raider"
    emoji = "🔄"

    def __init__(
        self,
        symbols,
        config: Optional[Dict] = None,
        risk_config: Optional[Dict] = None,
        ml_service: "MLService" | None = None,
        trade_log: "TradeLog" | None = None,
    ) -> None:
        band_window = int((config or {}).get("band_window", 20))
        lookback = max(band_window * 3, int((config or {}).get("lookback", band_window * 3)))
        super().__init__(
            symbols=symbols,
            lookback=lookback,
            config=config,
            risk_config=risk_config,
            ml_service=ml_service,
            trade_log=trade_log,
        )
        self.band_window = band_window
        self.band_std_dev = float((config or {}).get("band_std_dev", 2.2))
        self.reversion_threshold = float((config or {}).get("reversion_threshold", 0.003))

    async def evaluate_signal(self, snapshot: MarketSnapshot) -> Dict[str, str]:
        self.update_history(snapshot)
        signals: Dict[str, str] = {}
        for symbol in self.symbols:
            history = list(self.price_history.get(symbol, []))
            if len(history) < max(self.band_window, self.warmup_candles):
                self.update_signal_state(symbol, None, {"status": "warmup"})
                continue
            window = history[-self.band_window :]
            mid = fmean(window)
            std_dev = pstdev(window) if len(window) > 1 else 0.0
            upper_band = mid + std_dev * self.band_std_dev
            lower_band = mid - std_dev * self.band_std_dev
            last_price = history[-1]
            distance = (last_price - mid) / mid if mid else 0.0
            signal: Optional[str] = None

            if not self.is_ready(symbol):
                self.update_signal_state(
                    symbol,
                    signal,
                    {
                        "status": "warmup",
                        "mid": mid,
                        "upper_band": upper_band,
                        "lower_band": lower_band,
                        "distance": distance,
                    },
                )
                continue

            if last_price > upper_band * (1 + self.reversion_threshold):
                signal = "sell"
            elif last_price <= mid or last_price < upper_band * (1 - self.reversion_threshold):
                signal = "buy"

            indicators = {
                "mid": mid,
                "upper_band": upper_band,
                "lower_band": lower_band,
                "price": last_price,
                "distance": distance,
            }
            if self._ml_service:
                indicators["ml_confidence"] = self._ml_service.latest_confidence(symbol, self.name)
            self.update_signal_state(symbol, signal, indicators)
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

        indicators = self._state.get(symbol, {}).get("indicators", {})

        if existing_position is not None:
            target_override: float | None = None
            if self.take_profit_pct:
                target_override = existing_position.entry_price * (1 - self.take_profit_pct / 100)
            else:
                target_override = indicators.get("mid")
            if target_override is not None:
                self.update_risk_levels(symbol, target=target_override)
            risk_triggered, risk_reason, risk_meta = self.check_risk_exit(symbol, price)
            if risk_triggered:
                self.update_signal_state(symbol, f"close:{risk_reason}")
                self.clear_risk_tracker(symbol)
                payload = {
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
                    confidence=0.7,
                    reason=risk_reason,
                    metadata=payload,
                )

        if existing_position is None:
            if signal != "sell" or not self.is_ready(symbol):
                return None
            cash = float(equity_per_trade) * (self.position_size_pct / 100)
            if cash <= 0:
                return None
            allowed, ml_confidence = self.ml_confirmation(symbol)
            if not allowed:
                self.update_signal_state(symbol, "ml-block", {"ml_confidence": ml_confidence})
                return None
            mean_price = indicators.get("mid", price)
            risk_meta = self.prepare_entry_risk(symbol, "sell", price, target=mean_price)
            metadata = {k: v for k, v in risk_meta.items() if v is not None}
            self.record_trade_event("open_signal", symbol, metadata)
            cash_value = float(cash) * float(self.leverage)
            return TradeIntent(
                worker=self.name,
                action="OPEN",
                symbol=symbol,
                side="sell",
                cash_spent=cash_value,
                entry_price=price,
                confidence=ml_confidence or 0.65,
                metadata=metadata,
            )

        should_close = False
        reason = ""
        if signal == "buy":
            should_close = True
            reason = "mean-hit"

        if should_close:
            self.update_signal_state(symbol, f"close:{reason}")
            metadata = {**self.risk_snapshot(symbol), "trigger": reason}
            if reason in {"stop", "target", "trail"}:
                self.record_trade_event(f"risk_{reason}", symbol, metadata)
            self.clear_risk_tracker(symbol)
            return TradeIntent(
                worker=self.name,
                action="CLOSE",
                symbol=symbol,
                side="buy",
                cash_spent=existing_position.cash_spent,
                entry_price=existing_position.entry_price,
                exit_price=price,
                confidence=0.7,
                reason=reason,
                metadata=metadata,
            )

        return None
