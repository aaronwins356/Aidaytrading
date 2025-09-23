"""Tests for numeric coercion safeguards."""

import pytest

from ai_trader.services.types import TradeIntent


def test_trade_intent_casts_numeric_fields_to_float() -> None:
    """Ensure string inputs are coerced to floats for downstream math."""

    intent = TradeIntent(
        worker="tester",
        action="OPEN",
        symbol="BTC/USD",
        side="buy",
        cash_spent="5.25",
        entry_price="100.5",
        exit_price="101.1",
        pnl_percent="0.5",
        pnl_usd="0.6",
        confidence="0.44",
    )

    assert isinstance(intent.cash_spent, float)
    assert isinstance(intent.confidence, float)
    assert intent.cash_spent == pytest.approx(5.25)
    # Downstream arithmetic should not raise type errors when mixing values.
    assert intent.cash_spent * intent.confidence == pytest.approx(5.25 * 0.44)
    assert intent.entry_price * 2 == pytest.approx(201.0)
    assert intent.exit_price is not None
    assert intent.exit_price - intent.entry_price == pytest.approx(0.6)
    assert intent.pnl_usd is not None
    assert intent.pnl_usd / intent.cash_spent == pytest.approx(0.6 / 5.25)
