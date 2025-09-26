"""Enhanced risk management utilities with configurable controls."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import Dict, Mapping, MutableMapping, Sequence

from ai_trader.services.logging import get_logger
from ai_trader.services.types import TradeIntent


@dataclass(slots=True)
class RiskConfig:
    """Runtime configuration for :class:`RiskManager`."""

    max_drawdown_percent: float = 20.0
    daily_loss_limit_percent: float = 5.0
    max_position_duration_minutes: float = 240.0
    risk_per_trade: float = 0.02
    max_open_positions: int = 3
    atr_stop_loss_multiplier: float = 1.5
    atr_take_profit_multiplier: float = 2.5
    min_trades_per_day: int = 30
    confidence_relax_percent: float = 0.1
    atr_period: int = 14
    min_stop_buffer: float = 0.005

    @classmethod
    def from_mapping(cls, config: Mapping[str, float | int | None]) -> "RiskConfig":
        payload: Dict[str, float | int | None] = dict(config or {})
        # Support legacy keys while exposing the newer schema.
        if "risk_per_trade" not in payload and "risk_percent" in payload:
            payload["risk_per_trade"] = payload["risk_percent"]
        if "confidence_relax_percent" not in payload and "throttle_relax_percent" in payload:
            payload["confidence_relax_percent"] = payload["throttle_relax_percent"]
        return cls(
            max_drawdown_percent=float(payload.get("max_drawdown_percent", 20.0)),
            daily_loss_limit_percent=float(payload.get("daily_loss_limit_percent", 5.0)),
            max_position_duration_minutes=float(
                payload.get("max_position_duration_minutes", 240)
            ),
            risk_per_trade=float(payload.get("risk_per_trade", 0.02)),
            max_open_positions=int(payload.get("max_open_positions", 3)),
            atr_stop_loss_multiplier=float(payload.get("atr_stop_loss_multiplier", 1.5)),
            atr_take_profit_multiplier=float(payload.get("atr_take_profit_multiplier", 2.5)),
            min_trades_per_day=int(payload.get("min_trades_per_day", 30)),
            confidence_relax_percent=max(
                0.0, float(payload.get("confidence_relax_percent", 0.1))
            ),
            atr_period=int(payload.get("atr_period", 14)),
            min_stop_buffer=max(0.0005, float(payload.get("min_stop_buffer", 0.005))),
        )


@dataclass(slots=True)
class RiskState:
    """Mutable runtime state that the risk manager maintains per session."""

    trading_day: date = field(default_factory=lambda: datetime.utcnow().date())
    daily_start_equity: float | None = None
    daily_peak_equity: float | None = None
    trades_today: int = 0
    halted: bool = False
    halt_reason: str | None = None

    def copy(self) -> "RiskState":
        return replace(self)


@dataclass(slots=True)
class RiskAssessment:
    """Return object describing the outcome of a trade risk evaluation."""

    allowed: bool
    intent: TradeIntent
    reason: str | None = None
    state: RiskState | None = None
    adjustments: Dict[str, float | str | bool] = field(default_factory=dict)


class RiskManager:
    """Evaluate trade intents against global risk constraints."""

    def __init__(self, config: Mapping[str, float | int | None] | None = None) -> None:
        self._config = RiskConfig.from_mapping(config or {})
        self._state = RiskState()
        self._logger = get_logger(__name__)

    @property
    def max_duration_minutes(self) -> float:
        return self._config.max_position_duration_minutes

    @property
    def state(self) -> RiskState:
        return self._state.copy()

    def set_state(self, state: RiskState) -> None:
        self._state = state.copy()

    def reset_daily_limits(self, *, now: datetime | None = None) -> None:
        self._state = self._refresh_state(self._state, None, now, force_reset=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate_trade(
        self,
        intent: TradeIntent,
        *,
        equity: float,
        equity_metrics: Mapping[str, float],
        open_positions: int,
        max_open_positions: int | None = None,
        price: float | None = None,
        candles: Sequence[Mapping[str, float]] | None = None,
        state: RiskState | None = None,
        update_state: bool = True,
    ) -> RiskAssessment:
        """Return a :class:`RiskAssessment` detailing whether a trade may proceed."""

        working_state = self._prepare_state(state, equity)
        if working_state.halted:
            reason = working_state.halt_reason or "halted"
            return RiskAssessment(False, intent, reason=reason, state=working_state)

        assessment = RiskAssessment(True, intent, state=working_state)

        # Limit concurrent exposure
        max_positions = max_open_positions or self._config.max_open_positions
        if intent.action == "OPEN" and open_positions >= max_positions:
            assessment.allowed = False
            assessment.reason = "max_open_positions"
            if update_state:
                self._state = working_state
            return assessment

        # Global drawdown guard based on lifetime equity curve
        overall_drawdown = float(equity_metrics.get("pnl_percent", 0.0))
        if overall_drawdown <= -abs(self._config.max_drawdown_percent):
            working_state.halted = True
            working_state.halt_reason = "max_drawdown"
            assessment.allowed = False
            assessment.reason = "max_drawdown"
            if update_state:
                self._state = working_state
            return assessment

        # Daily drawdown guard using session peak equity
        if (
            working_state.daily_peak_equity
            and working_state.daily_peak_equity > 0.0
        ):
            drawdown_pct = (equity - working_state.daily_peak_equity) / working_state.daily_peak_equity * 100
            if drawdown_pct <= -abs(self._config.daily_loss_limit_percent):
                working_state.halted = True
                working_state.halt_reason = "daily_loss_limit"
                assessment.allowed = False
                assessment.reason = "daily_loss_limit"
                if update_state:
                    self._state = working_state
                return assessment

        if intent.action == "OPEN":
            adjustments = self._apply_position_sizing(
                intent, equity, price=price, candles=candles
            )
            if adjustments:
                assessment.adjustments.update(adjustments)

        if update_state:
            self._state = working_state
        assessment.state = working_state
        return assessment

    def check_trade(
        self,
        intent: TradeIntent,
        equity_metrics: Mapping[str, float],
        open_positions: int,
        max_open_positions: int,
        *,
        equity: float | None = None,
        price: float | None = None,
        candles: Sequence[Mapping[str, float]] | None = None,
    ) -> bool:
        equity_value = float(equity) if equity is not None else float(
            equity_metrics.get("equity", 0.0)
        )
        assessment = self.evaluate_trade(
            intent,
            equity=equity_value,
            equity_metrics=equity_metrics,
            open_positions=open_positions,
            max_open_positions=max_open_positions,
            price=price,
            candles=candles,
            state=None,
            update_state=True,
        )
        if not assessment.allowed and assessment.reason:
            self._logger.info(
                "[RISK] Blocked %s trade for %s (%s)",
                intent.worker,
                intent.symbol,
                assessment.reason,
            )
        elif assessment.allowed and assessment.adjustments:
            self._logger.info(
                "[RISK] Adjusted %s trade for %s: %s",
                intent.worker,
                intent.symbol,
                assessment.adjustments,
            )
        return assessment.allowed

    def effective_confidence_threshold(
        self,
        base_threshold: float,
        *,
        state: RiskState | None = None,
        now: datetime | None = None,
    ) -> tuple[float, bool]:
        """Return the threshold adjusted for dynamic throttling."""

        if base_threshold <= 0.0:
            return base_threshold, False
        working_state = self._prepare_state(state, equity=None, now=now)
        if working_state.trades_today == 0 or working_state.trades_today >= self._config.min_trades_per_day:
            return base_threshold, False
        relax_pct = min(0.9, max(0.0, self._config.confidence_relax_percent))
        adjusted = max(0.0, base_threshold * (1.0 - relax_pct))
        return adjusted, adjusted != base_threshold

    def on_trade_executed(
        self, intent: TradeIntent, *, state: RiskState | None = None, now: datetime | None = None
    ) -> RiskState:
        """Update state counters when a trade is successfully executed."""

        working_state = self._prepare_state(state, equity=None, now=now)
        if intent.action == "OPEN":
            working_state.trades_today += 1
        if state is None:
            self._state = working_state
        return working_state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_state(
        self,
        state: RiskState | None,
        equity: float | None,
        now: datetime | None = None,
        *,
        force_reset: bool = False,
    ) -> RiskState:
        base_state = state.copy() if state is not None else self._state.copy()
        refreshed = self._refresh_state(base_state, equity, now, force_reset=force_reset)
        if equity is not None:
            if refreshed.daily_start_equity is None:
                refreshed.daily_start_equity = equity
            if refreshed.daily_peak_equity is None or equity > refreshed.daily_peak_equity:
                refreshed.daily_peak_equity = equity
        return refreshed

    def _refresh_state(
        self,
        state: RiskState,
        equity: float | None,
        now: datetime | None,
        *,
        force_reset: bool = False,
    ) -> RiskState:
        today = (now or datetime.utcnow()).date()
        if force_reset or state.trading_day != today:
            new_state = RiskState(trading_day=today)
            if equity is not None:
                new_state.daily_start_equity = equity
                new_state.daily_peak_equity = equity
            return new_state
        return state

    def _apply_position_sizing(
        self,
        intent: TradeIntent,
        equity: float,
        *,
        price: float | None,
        candles: Sequence[Mapping[str, float]] | None,
    ) -> Dict[str, float | str | bool]:
        if intent.action != "OPEN":
            return {}

        entry_price = float(intent.entry_price)
        market_price = float(price) if price is not None else entry_price
        metadata: MutableMapping[str, float | str | bool] = dict(intent.metadata or {})
        original_cash = float(intent.cash_spent) if float(intent.cash_spent or 0.0) > 0 else None

        atr_value = self._extract_atr(metadata, candles)
        stop_price = self._resolve_stop_price(intent.side, entry_price, atr_value)
        target_price = self._resolve_target_price(intent.side, entry_price, atr_value)

        stop_distance = max(1e-8, abs(entry_price - stop_price))
        risk_amount = max(0.0, equity * self._config.risk_per_trade)
        quantity = risk_amount / stop_distance
        if quantity <= 0.0:
            quantity = risk_amount / max(entry_price, 1e-6)
        cash_spent = quantity * entry_price
        if original_cash is not None:
            cash_spent = min(cash_spent, original_cash)

        reward_distance = abs(target_price - entry_price)
        risk_reward = reward_distance / stop_distance if stop_distance else math.inf

        metadata.update(
            {
                "atr": atr_value,
                "stop_price": stop_price,
                "target_price": target_price,
                "risk_amount": risk_amount,
                "risk_reward": risk_reward,
                "market_price": market_price,
                "position_size": quantity,
                "original_cash_spent": original_cash,
            }
        )
        intent.cash_spent = cash_spent
        intent.metadata = dict(metadata)

        return {
            "cash_spent": cash_spent,
            "stop_price": stop_price,
            "target_price": target_price,
            "atr": atr_value if atr_value is not None else 0.0,
            "risk_reward": risk_reward,
        }

    def _extract_atr(
        self, metadata: Mapping[str, float | str | bool], candles: Sequence[Mapping[str, float]] | None
    ) -> float | None:
        if "atr" in metadata:
            try:
                value = float(metadata["atr"])  # type: ignore[arg-type]
                if value > 0:
                    return value
            except (TypeError, ValueError):
                pass
        if not candles:
            return None
        return self._compute_atr(candles, period=self._config.atr_period)

    def _resolve_stop_price(
        self, side: str, entry_price: float, atr: float | None
    ) -> float:
        buffer = entry_price * self._config.min_stop_buffer
        if atr is not None and atr > 0:
            offset = atr * self._config.atr_stop_loss_multiplier
        else:
            offset = max(buffer, entry_price * 0.01)
        if side.lower() == "buy":
            return max(0.0, entry_price - offset)
        return entry_price + offset

    def _resolve_target_price(
        self, side: str, entry_price: float, atr: float | None
    ) -> float:
        if atr is not None and atr > 0:
            offset = atr * self._config.atr_take_profit_multiplier
        else:
            offset = entry_price * max(self._config.min_stop_buffer * 2, 0.02)
        if side.lower() == "buy":
            return entry_price + offset
        return max(0.0, entry_price - offset)

    def _compute_atr(
        self, candles: Sequence[Mapping[str, float]], *, period: int
    ) -> float | None:
        if len(candles) < 2:
            return None
        true_ranges: list[float] = []
        prev_close = float(candles[0].get("close", 0.0))
        for candle in candles[1:]:
            high = float(candle.get("high", 0.0))
            low = float(candle.get("low", 0.0))
            close = float(candle.get("close", 0.0))
            tr_components = [high - low, abs(high - prev_close), abs(low - prev_close)]
            true_range = max(comp for comp in tr_components if not math.isnan(comp))
            true_ranges.append(true_range)
            prev_close = close
        if not true_ranges:
            return None
        sample = true_ranges[-period:]
        return sum(sample) / len(sample)


__all__ = [
    "RiskManager",
    "RiskConfig",
    "RiskState",
    "RiskAssessment",
]
