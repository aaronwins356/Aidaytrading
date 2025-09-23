"""Unit tests for the risk manager logic."""

from __future__ import annotations

from ai_trader.services.risk import RiskManager
from ai_trader.services.types import TradeIntent


def _make_open_intent() -> TradeIntent:
    return TradeIntent(
        worker="tester",
        action="OPEN",
        symbol="BTC/USD",
        side="buy",
        cash_spent=100.0,
        entry_price=10_000.0,
        confidence=0.5,
    )


def test_check_trade_allows_at_starting_equity() -> None:
    """Trades should be allowed when the account has not drawn down."""

    risk = RiskManager(
        {
            "max_drawdown_percent": 25,
            "daily_loss_limit_percent": 10,
            "max_position_duration_minutes": 120,
        }
    )
    equity_metrics = {"equity": 5_000.0, "pnl_percent": 0.0}

    allowed = risk.check_trade(_make_open_intent(), equity_metrics, open_positions=0, max_open_positions=5)

    assert allowed, "Risk manager should not block trades at the baseline equity"
