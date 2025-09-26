"""Tests for the Prometheus metrics endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from ai_trader.api_service import app, attach_services, reset_services
from ai_trader.services.monitoring import get_monitoring_center
from ai_trader.services.risk import RiskManager
from ai_trader.services.runtime_state import RuntimeStateStore
from ai_trader.services.trade_log import MemoryTradeLog
from ai_trader.services.types import OpenPosition


class DummyMLService:
    """Provide deterministic validation metrics for testing."""

    def latest_validation_metrics(self) -> dict[str, dict[str, float]]:
        return {
            "BTC/USDT": {"accuracy": 0.82},
            "ETH/USDT": {"accuracy": 0.76},
        }


def test_metrics_endpoint_exports_prometheus_payload() -> None:
    reset_services(state_file=None)
    runtime_state = RuntimeStateStore(state_file=None)
    trade_log = MemoryTradeLog()
    risk_manager = RiskManager()
    ml_service = DummyMLService()
    attach_services(
        trade_log=trade_log,
        runtime_state=runtime_state,
        risk_manager=risk_manager,
        ml_service=ml_service,
    )

    position = OpenPosition(
        worker="momentum",
        symbol="BTC/USDT",
        side="buy",
        quantity=0.5,
        entry_price=20000.0,
        cash_spent=10000.0,
    )
    runtime_state.update_account(
        equity=10150.0,
        balances={"USD": 5100.0},
        pnl_percent=1.5,
        pnl_usd=150.0,
        open_positions=[position],
        prices={"BTC/USDT": 20500.0},
        starting_equity=10000.0,
    )

    trade_log.record_equity(10000.0, 0.0, 0.0)
    trade_log.record_equity(9900.0, -1.0, -100.0)
    trade_log.record_equity(10150.0, 1.5, 150.0)

    center = get_monitoring_center()
    center.reset()
    center.record_event("websocket_reconnect", "WARNING", "Reconnect", metadata={"attempt": 1})
    center.record_event("heartbeat", "INFO", "Tick")

    runtime_state.mark_runtime_update()

    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    assert "trader_equity_total" in body
    assert "trader_open_positions 1" in body
    assert "trader_max_drawdown_percent" in body
    assert "trader_ml_validation_accuracy 0.820000" in body
    assert "trader_websocket_reconnect_total 1" in body
    assert "trader_watchdog_last_update_age_seconds" in body
