from fastapi.testclient import TestClient

from ai_trader.api_service import app, attach_services
from ai_trader.services.monitoring import get_monitoring_center
from ai_trader.services.risk import RiskManager
from ai_trader.services.runtime_state import RuntimeStateStore
from ai_trader.services.trade_log import MemoryTradeLog


def test_monitoring_endpoint_surfaces_recent_events() -> None:
    center = get_monitoring_center()
    center.reset()
    runtime_state = RuntimeStateStore(state_file=None)
    trade_log = MemoryTradeLog()
    risk_manager = RiskManager()
    attach_services(trade_log=trade_log, runtime_state=runtime_state, risk_manager=risk_manager)

    center.record_event(
        "watchdog_stall",
        "WARNING",
        "Runtime stalled",
        metadata={"timeout_seconds": 60},
    )
    center.set_runtime_degraded(True, "Runtime stalled")

    client = TestClient(app)
    response = client.get("/monitoring")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["events"][0]["event_type"] == "watchdog_stall"

    status_response = client.get("/status")
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["runtime_degraded"] is True
    assert status_payload["runtime_degraded_reason"] == "Runtime stalled"
