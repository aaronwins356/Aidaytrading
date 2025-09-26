import pytest

from ai_trader.services.risk_manager import RiskManager
from ai_trader.services.types import TradeIntent


def _build_intent(action: str = "OPEN", side: str = "buy") -> TradeIntent:
    return TradeIntent(
        worker="unit-test",
        action=action,
        symbol="BTC/USD",
        side=side,
        cash_spent=0.0,
        entry_price=100.0,
        confidence=0.6,
    )


def test_position_sizing_small_balance() -> None:
    manager = RiskManager({"risk_per_trade": 0.05, "atr_stop_loss_multiplier": 1.5})
    intent = _build_intent()
    intent.metadata = {"atr": 5.0}
    assessment = manager.evaluate_trade(
        intent,
        equity=100.0,
        equity_metrics={"equity": 100.0, "pnl_percent": 0.0},
        open_positions=0,
        max_open_positions=5,
        price=100.0,
    )
    assert assessment.allowed
    assert intent.cash_spent == pytest.approx(66.666, rel=1e-3)
    assert intent.metadata is not None
    assert intent.metadata["stop_price"] < intent.entry_price
    assert intent.metadata["target_price"] > intent.entry_price


def test_stop_loss_take_profit_from_atr() -> None:
    manager = RiskManager({"risk_per_trade": 0.02})
    intent = _build_intent()
    candles = [
        {"open": 100 + i, "high": 102 + i, "low": 98 + i, "close": 101 + i, "volume": 10.0}
        for i in range(20)
    ]
    assessment = manager.evaluate_trade(
        intent,
        equity=1000.0,
        equity_metrics={"equity": 1000.0, "pnl_percent": 0.0},
        open_positions=0,
        max_open_positions=5,
        price=105.0,
        candles=candles,
    )
    assert assessment.allowed
    assert intent.metadata is not None
    assert intent.metadata["stop_price"] < intent.entry_price
    assert intent.metadata["target_price"] > intent.entry_price
    assert intent.metadata["atr"] is not None


def test_drawdown_guard_halts_trading() -> None:
    manager = RiskManager({"max_drawdown_percent": 5})
    intent = _build_intent()
    assessment = manager.evaluate_trade(
        intent,
        equity=1000.0,
        equity_metrics={"equity": 1000.0, "pnl_percent": -6.0},
        open_positions=0,
        max_open_positions=5,
        price=100.0,
    )
    assert not assessment.allowed
    assert assessment.reason == "max_drawdown"
    assert assessment.state is not None and assessment.state.halted
