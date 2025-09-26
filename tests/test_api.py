"""Integration tests for the FastAPI monitoring service."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ai_trader.api_service import app, attach_services, reset_services
from ai_trader.services.risk import RiskManager
from ai_trader.services.runtime_state import RuntimeStateStore
from ai_trader.services.trade_log import MemoryTradeLog
from ai_trader.services.types import OpenPosition, TradeIntent


@pytest.fixture
def api_context() -> tuple[TestClient, RuntimeStateStore, RiskManager, MemoryTradeLog]:
    reset_services(state_file=None)
    trade_log = MemoryTradeLog()
    runtime_state = RuntimeStateStore(state_file=None)
    risk_manager = RiskManager(
        {
            "risk_per_trade": 0.02,
            "max_drawdown_percent": 20.0,
            "max_open_positions": 3,
        }
    )
    runtime_state.set_base_currency("USD")
    runtime_state.set_starting_equity(1000.0)
    runtime_state.update_risk_settings(risk_manager.config_dict())
    attach_services(trade_log=trade_log, runtime_state=runtime_state, risk_manager=risk_manager)

    # Seed trade history for the API responses.
    open_intent = TradeIntent(
        worker="alpha",
        action="OPEN",
        symbol="ETH/USD",
        side="buy",
        cash_spent=150.0,
        entry_price=1500.0,
        confidence=0.75,
        metadata={"fill_quantity": 0.1},
    )
    trade_log.record_trade(open_intent)
    runtime_state.record_trade(open_intent)

    close_intent = TradeIntent(
        worker="alpha",
        action="CLOSE",
        symbol="ETH/USD",
        side="sell",
        cash_spent=150.0,
        entry_price=1500.0,
        exit_price=1575.0,
        pnl_usd=7.5,
        pnl_percent=5.0,
        reason="target",
        metadata={"fill_quantity": 0.1, "fill_price": 1575.0},
    )
    trade_log.record_trade(close_intent)
    runtime_state.record_trade(close_intent)

    # Active open position used for status/unrealised PnL metrics.
    open_position = OpenPosition(
        worker="beta",
        symbol="BTC/USD",
        side="buy",
        quantity=0.01,
        entry_price=20000.0,
        cash_spent=200.0,
    )
    runtime_state.update_account(
        equity=1017.5,
        balances={"USD": 800.0, "BTC": 0.01},
        pnl_percent=1.75,
        pnl_usd=17.5,
        open_positions=[open_position],
        prices={"BTC/USD": 21000.0},
        starting_equity=1000.0,
    )

    client = TestClient(app)
    yield client, runtime_state, risk_manager, trade_log
    reset_services(state_file=None)


def test_status_endpoint(
    api_context: tuple[TestClient, RuntimeStateStore, RiskManager, MemoryTradeLog]
) -> None:
    client, _, _, _ = api_context
    response = client.get("/status")
    assert response.status_code == 200
    payload = response.json()
    assert pytest.approx(payload["equity"], rel=1e-6) == 1017.5
    assert pytest.approx(payload["balance"], rel=1e-6) == 800.0
    assert len(payload["open_positions"]) == 1
    assert pytest.approx(payload["open_positions"][0]["unrealized_pnl"], rel=1e-6) == 10.0
    assert payload["last_trade_timestamp"] is not None


def test_profit_endpoint(
    api_context: tuple[TestClient, RuntimeStateStore, RiskManager, MemoryTradeLog]
) -> None:
    client, _, _, _ = api_context
    response = client.get("/profit")
    assert response.status_code == 200
    payload = response.json()
    assert pytest.approx(payload["realized"]["usd"], rel=1e-6) == 7.5
    assert pytest.approx(payload["realized"]["percent"], rel=1e-6) == 0.75
    assert pytest.approx(payload["unrealized"]["usd"], rel=1e-6) == 10.0
    assert pytest.approx(payload["total"]["usd"], rel=1e-6) == 17.5


def test_trades_endpoint_limit(
    api_context: tuple[TestClient, RuntimeStateStore, RiskManager, MemoryTradeLog]
) -> None:
    client, _, _, _ = api_context
    response = client.get("/trades", params={"limit": 1})
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert len(payload["trades"]) == 1
    trade = payload["trades"][0]
    assert trade["pair"] == "ETH/USD"
    assert trade["side"] == "sell"
    assert pytest.approx(trade["pnl"]["usd"], rel=1e-6) == 7.5


def test_config_updates_risk_settings(
    api_context: tuple[TestClient, RuntimeStateStore, RiskManager, MemoryTradeLog]
) -> None:
    client, runtime_state, risk_manager, _ = api_context
    response = client.post(
        "/config",
        json={"risk_per_trade": 0.05, "max_drawdown_percent": 15.0},
    )
    assert response.status_code == 200
    payload = response.json()
    assert pytest.approx(payload["config"]["risk_per_trade"], rel=1e-6) == 0.05
    assert pytest.approx(risk_manager.config_dict()["risk_per_trade"], rel=1e-6) == 0.05
    risk_snapshot = runtime_state.risk_snapshot()
    assert pytest.approx(risk_snapshot["risk_per_trade"], rel=1e-6) == 0.05
    assert pytest.approx(risk_snapshot["max_drawdown_percent"], rel=1e-6) == 15.0
