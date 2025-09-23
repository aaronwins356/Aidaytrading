"""Smoke tests for dashboard initialisation helpers."""

from __future__ import annotations

from ai_trader.dashboard import app
from ai_trader.services.ml import MLService


def test_init_ml_service_loads(tmp_path, monkeypatch) -> None:
    """The dashboard ML initialiser should construct an MLService."""

    db_path = tmp_path / "ml.db"
    monkeypatch.setattr(app, "DB_PATH", db_path, raising=False)
    config = {
        "ml": {
            "feature_keys": ["alpha", "beta"],
            "learning_rate": 0.02,
            "regularization": 0.001,
            "threshold": 0.3,
            "ensemble": False,
            "forest_size": 2,
            "random_state": 11,
            "warmup_target": 3,
            "warmup_samples": 2,
            "confidence_stall_limit": 4,
        }
    }

    service = app.init_ml_service.__wrapped__(config)  # type: ignore[attr-defined]
    assert isinstance(service, MLService)
    assert tuple(service.feature_keys) == ("alpha", "beta")
