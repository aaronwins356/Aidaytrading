"""QA coverage for RiskManager behaviour."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import pytest

from ai_trader.services.risk_manager import RiskManager
from ai_trader.services.types import TradeIntent


def _build_candles(base: float = 100.0, count: int = 20) -> List[dict[str, float]]:
    candles: List[dict[str, float]] = []
    price = base
    for idx in range(count):
        move = (-1) ** idx * 0.8
        high = price + 1.5 + idx * 0.05
        low = price - 1.5 - idx * 0.05
        close = price + move
        candles.append(
            {
                "open": price,
                "high": high,
                "low": low,
                "close": close,
                "volume": 10.0 + idx,
            }
        )
        price = close
    return candles


def test_position_sizing_and_atr_metadata() -> None:
    manager = RiskManager(
        {
            "risk_per_trade": 0.04,
            "atr_stop_loss_multiplier": 1.6,
            "atr_take_profit_multiplier": 3.2,
            "atr_period": 14,
            "min_stop_buffer": 0.002,
        }
    )
    intent = TradeIntent(
        worker="qa",
        action="OPEN",
        symbol="BTC/USDT",
        side="buy",
        cash_spent=0.0,
        entry_price=102.5,
        confidence=0.7,
        metadata={},
    )
    candles = _build_candles()
    assessment = manager.evaluate_trade(
        intent,
        equity=5000.0,
        equity_metrics={"equity": 5000.0, "pnl_percent": 0.0},
        open_positions=0,
        price=103.0,
        candles=candles,
    )
    assert assessment.allowed
    assert intent.metadata is not None
    atr_value = intent.metadata.get("atr")
    stop_price = intent.metadata.get("stop_price")
    target_price = intent.metadata.get("target_price")
    assert isinstance(atr_value, float) and atr_value > 0
    assert isinstance(stop_price, float) and stop_price < intent.entry_price
    assert isinstance(target_price, float) and target_price > intent.entry_price
    risk_amount = intent.metadata.get("risk_amount")
    assert pytest.approx(5000.0 * 0.04, rel=1e-3) == risk_amount
    assert intent.cash_spent > 0.0


def test_drawdown_guard_halts_trading() -> None:
    manager = RiskManager({"max_drawdown_percent": 5.0})
    intent = TradeIntent(
        worker="qa",
        action="OPEN",
        symbol="BTC/USDT",
        side="buy",
        cash_spent=100.0,
        entry_price=100.0,
    )
    assessment = manager.evaluate_trade(
        intent,
        equity=4000.0,
        equity_metrics={"equity": 4000.0, "pnl_percent": -6.0},
        open_positions=0,
        price=100.0,
    )
    assert not assessment.allowed
    assert assessment.reason == "max_drawdown"
    assert assessment.state is not None and assessment.state.halted


def test_daily_loss_limit_resets_with_new_day() -> None:
    manager = RiskManager({"daily_loss_limit_percent": 3.0})
    intent = TradeIntent(
        worker="qa",
        action="OPEN",
        symbol="BTC/USDT",
        side="buy",
        cash_spent=100.0,
        entry_price=100.0,
    )
    now = datetime(2024, 5, 1, tzinfo=timezone.utc)
    state = manager.state
    state.daily_start_equity = 5000.0
    state.daily_peak_equity = 5000.0
    state.trades_today = 5
    manager.set_state(state)
    assessment = manager.evaluate_trade(
        intent,
        equity=4800.0,
        equity_metrics={"equity": 4800.0, "pnl_percent": -3.5},
        open_positions=0,
        price=100.0,
        update_state=True,
    )
    assert not assessment.allowed
    assert assessment.reason == "daily_loss_limit"
    assert assessment.state is not None and assessment.state.halted

    manager.reset_daily_limits(now=now.replace(day=2))
    second_assessment = manager.evaluate_trade(
        intent,
        equity=5000.0,
        equity_metrics={"equity": 5000.0, "pnl_percent": 0.0},
        open_positions=0,
        price=100.0,
        update_state=True,
    )
    assert second_assessment.allowed
