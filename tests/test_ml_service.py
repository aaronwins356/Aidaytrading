"""Behavioural tests for the streaming ML service."""

from __future__ import annotations

from ai_trader.services.ml import MLService
from ai_trader.services.trade_log import TradeLog

import pytest


def test_ml_service_prediction_cycle(tmp_path) -> None:
    """The ML service should learn, score, and expose importances."""

    db_path = tmp_path / "ml.db"
    TradeLog(db_path)
    service = MLService(
        db_path=db_path,
        feature_keys=["f1", "f2"],
        learning_rate=0.1,
        regularization=0.01,
        threshold=0.4,
        ensemble=True,
        forest_size=5,
        random_state=42,
    )
    service.probe()

    confidence = service.update("BTC/USD", {"f1": 1.0, "f2": 0.5}, label=1.0)
    assert 0.0 <= confidence <= 1.0

    # Repeat once more to ensure the logistic weights receive multiple updates.
    service.update("BTC/USD", {"f1": 0.8, "f2": 0.2}, label=1.0)

    decision, probability = service.predict(
        "BTC/USD", {"f1": 0.9, "f2": 0.4}, worker="unit-test"
    )
    assert isinstance(decision, bool)
    assert probability > 0.0

    importances = service.feature_importance("BTC/USD")
    assert importances
    assert set(importances).issubset({"f1", "f2"})

    assert service.ensemble_requested is True
    assert isinstance(service.ensemble_available, bool)
    assert service.ensemble_backend


def test_ml_service_build_pipeline(tmp_path) -> None:
    """Building the pipeline should succeed without raising exceptions."""

    service = MLService(db_path=tmp_path / "ml.db", feature_keys=["x"], learning_rate=0.05)
    pipeline = service._build_pipeline()
    assert pipeline is not None


def test_ml_service_build_forest(tmp_path) -> None:
    """The ensemble builder should return an ARF classifier when available."""

    service = MLService(
        db_path=tmp_path / "ml.db",
        feature_keys=["x"],
        learning_rate=0.05,
        ensemble=True,
        forest_size=3,
    )
    if not service.ensemble_available:
        pytest.skip("river.forest.ARFClassifier unavailable in this environment")
    forest_model = service._build_forest()
    assert forest_model is not None
    assert forest_model.n_models == 3
