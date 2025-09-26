from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from ai_trader.api_service import app, attach_services, reset_services
from ai_trader.services.ml import MLService
from ai_trader.services.risk import RiskManager
from ai_trader.services.runtime_state import RuntimeStateStore
from ai_trader.services.trade_log import MemoryTradeLog


def _train_service(db_path: Path) -> MLService:
    service = MLService(
        db_path=db_path,
        feature_keys=["f1"],
        ensemble=False,
        threshold=0.5,
        warmup_samples=5,
    )
    for index in range(60):
        feature_value = float(index % 2)
        label = index % 2
        service.update("BTC/USD", {"f1": feature_value}, label=label)
    return service


def test_validation_metrics_exposed_via_api(tmp_path: Path) -> None:
    db_path = tmp_path / "ml.db"
    service = _train_service(db_path)

    metrics = service.latest_validation_metrics("BTC/USD")
    assert metrics
    assert metrics["accuracy"] >= 0.5
    assert "reward" in metrics

    state_path = tmp_path / "state.json"
    reset_services(state_file=state_path)
    runtime_state = RuntimeStateStore(state_path)
    attach_services(
        trade_log=MemoryTradeLog(),
        runtime_state=runtime_state,
        risk_manager=RiskManager(),
        ml_service=service,
    )
    client = TestClient(app)
    try:
        response = client.get("/ml-metrics")
        assert response.status_code == 200
        payload = response.json()
        assert "metrics" in payload
        btc_metrics = payload["metrics"].get("BTC/USD")
        assert btc_metrics is not None
        assert btc_metrics["accuracy"] >= 0.5
        assert "reward" in btc_metrics
    finally:
        client.close()
        reset_services(state_file=state_path)
