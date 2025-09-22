"""Mean reversion short specialist."""

from __future__ import annotations

from statistics import fmean, pstdev
from typing import Dict, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ai_trader.services.ml import MLService

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
    ) -> None:
        band_window = int((config or {}).get("band_window", 20))
        lookback = max(band_window * 3, int((config or {}).get("lookback", band_window * 3)))
        super().__init__(symbols=symbols, lookback=lookback, config=config, risk_config=risk_config, ml_service=ml_service)
        self.band_window = band_window
        self.band_std_dev = float((config or {}).get("band_std_dev", 2.2))
        self.reversion_threshold = float((config or {}).get("reversion_threshold", 0.003))
        self._position_tracker: Dict[str, Dict[str, float]] = {}

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

        tracker = self._position_tracker.setdefault(symbol, {"best_price": price})

        if existing_position is None:
            if signal != "sell" or not self.is_ready(symbol):
                return None
            cash = equity_per_trade * (self.position_size_pct / 100)
            if cash <= 0:
                return None
            allowed, ml_confidence = self.ml_confirmation(symbol)
            if not allowed:
                self.update_signal_state(symbol, "ml-block", {"ml_confidence": ml_confidence})
                return None
            tracker["best_price"] = price
            tracker["stop_price"] = price * (1 + self.stop_loss_pct / 100) if self.stop_loss_pct else None
            tracker["mean_price"] = self._state.get(symbol, {}).get("indicators", {}).get("mid", price)
            return TradeIntent(
                worker=self.name,
                action="OPEN",
                symbol=symbol,
                side="sell",
                cash_spent=cash * self.leverage,
                entry_price=price,
                confidence=ml_confidence or 0.65,
            )

        tracker["best_price"] = min(tracker.get("best_price", price), price)
        if self.take_profit_pct:
            tracker["target_price"] = existing_position.entry_price * (1 - self.take_profit_pct / 100)
        else:
            mean_price = tracker.get("mean_price") or existing_position.entry_price
            tracker["target_price"] = mean_price

        stop_price = tracker.get("stop_price") or existing_position.entry_price * (1 + self.stop_loss_pct / 100)
        target_price = tracker.get("target_price")
        trailing_price = None
        if self.trailing_stop_pct:
            trailing_price = tracker["best_price"] * (1 + self.trailing_stop_pct / 100)

        should_close = False
        reason = ""
        if signal == "buy":
            should_close = True
            reason = "mean-hit"
        if target_price and price <= target_price:
            should_close = True
            reason = reason or "target"
        if stop_price and price >= stop_price:
            should_close = True
            reason = "stop"
        if trailing_price and price >= trailing_price:
            should_close = True
            reason = reason or "trail"

        if should_close:
            self.update_signal_state(symbol, f"close:{reason}")
            return TradeIntent(
                worker=self.name,
                action="CLOSE",
                symbol=symbol,
                side="buy",
                cash_spent=existing_position.cash_spent,
                entry_price=existing_position.entry_price,
                exit_price=price,
                confidence=0.7,
            )

        return None
