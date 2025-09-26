"""QA regression tests for FastAPI endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest
from fastapi.testclient import TestClient

from ai_trader.api_service import app, attach_services, reset_services
from ai_trader.services.risk_manager import RiskManager
from ai_trader.services.runtime_state import RuntimeStateStore
from ai_trader.services.trade_log import TradeLog
from ai_trader.services.types import TradeIntent


@pytest.fixture()
def api_client(tmp_path: Path) -> TestClient:
    state_path = tmp_path / "runtime_state.json"
    db_path = tmp_path / "trades.db"
    runtime_state = RuntimeStateStore(state_path)
    risk_manager = RiskManager({"risk_per_trade": 0.02})
    trade_log = TradeLog(db_path)
    reset_services(state_file=state_path)
    attach_services(trade_log=trade_log, runtime_state=runtime_state, risk_manager=risk_manager)
    client = TestClient(app)
    yield client
    reset_services(state_file=state_path)


def _seed_runtime_state(runtime_state: RuntimeStateStore) -> None:
    runtime_state.set_base_currency("USD")
    runtime_state.set_starting_equity(10000.0)
    runtime_state.update_account(
        equity=10100.0,
        balances={"USD": 10100.0},
        pnl_percent=1.0,
        pnl_usd=100.0,
        open_positions=[],
        prices={},
        starting_equity=10000.0,
    )


def _record_trade(trade_log: TradeLog, runtime_state: RuntimeStateStore) -> Dict[str, float]:
    trade = TradeIntent(
        worker="qa-bot",
        action="CLOSE",
        symbol="BTC/USDT",
        side="buy",
        cash_spent=500.0,
        entry_price=20000.0,
        exit_price=20500.0,
        pnl_usd=25.0,
        pnl_percent=5.0,
        confidence=0.8,
        metadata={"fill_quantity": 0.025},
    )
    trade_log.record_trade(trade)
    runtime_state.record_trade(trade)
    return {"pnl_usd": trade.pnl_usd or 0.0}


def test_status_and_profit_endpoints(api_client: TestClient, tmp_path: Path) -> None:
    runtime_state = RuntimeStateStore(tmp_path / "runtime_state.json")
    trade_log = TradeLog(tmp_path / "trades.db")
    risk_manager = RiskManager({})
    reset_services(state_file=tmp_path / "runtime_state.json")
    attach_services(trade_log=trade_log, runtime_state=runtime_state, risk_manager=risk_manager)
    _seed_runtime_state(runtime_state)
    pnl = _record_trade(trade_log, runtime_state)

    status_response = api_client.get("/status")
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["equity"] == pytest.approx(10100.0, rel=1e-6)
    assert status_payload["last_trade_timestamp"] is not None

    profit_response = api_client.get("/profit")
    assert profit_response.status_code == 200
    profit_payload = profit_response.json()
    assert profit_payload["realized"]["usd"] >= pnl["pnl_usd"]
    assert profit_payload["total"]["usd"] >= pnl["pnl_usd"]


def test_trades_and_risk_endpoints(api_client: TestClient, tmp_path: Path) -> None:
    runtime_state = RuntimeStateStore(tmp_path / "runtime_state.json")
    trade_log = TradeLog(tmp_path / "trades.db")
    risk_manager = RiskManager({"risk_per_trade": 0.02})
    reset_services(state_file=tmp_path / "runtime_state.json")
    attach_services(trade_log=trade_log, runtime_state=runtime_state, risk_manager=risk_manager)
    _seed_runtime_state(runtime_state)
    _record_trade(trade_log, runtime_state)

    trades_response = api_client.get("/trades", params={"limit": 5})
    assert trades_response.status_code == 200
    payload = trades_response.json()
    assert payload["count"] == len(payload["trades"])
    assert payload["count"] >= 1
    first_trade = payload["trades"][0]
    assert first_trade["pair"] == "BTC/USDT"
    assert first_trade["pnl"]["usd"] is not None

    risk_response = api_client.get("/risk")
    assert risk_response.status_code == 200
    risk_payload = risk_response.json()
    assert "risk_per_trade" in risk_payload


def test_update_config_endpoint(api_client: TestClient, tmp_path: Path) -> None:
    runtime_state = RuntimeStateStore(tmp_path / "runtime_state.json")
    trade_log = TradeLog(tmp_path / "trades.db")
    risk_manager = RiskManager({"risk_per_trade": 0.02})
    reset_services(state_file=tmp_path / "runtime_state.json")
    attach_services(trade_log=trade_log, runtime_state=runtime_state, risk_manager=risk_manager)
    _seed_runtime_state(runtime_state)

    response = api_client.post("/config", json={"risk_per_trade": 0.05, "max_open_positions": 4})
    assert response.status_code == 200
    payload = response.json()
    assert payload["config"]["risk_per_trade"] == pytest.approx(0.05)
    assert payload["config"]["max_open_positions"] == 4

    # ensure runtime state persisted the update
    latest = runtime_state.risk_snapshot()
    assert latest["risk_per_trade"] == pytest.approx(0.05)
