"""Behavioural tests for the streaming ML service."""

from __future__ import annotations

from ai_trader.services.ml import MLService


def test_ml_service_prediction_cycle(tmp_path) -> None:
    """The ML service should learn, score, and expose importances."""

    db_path = tmp_path / "ml.db"
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
