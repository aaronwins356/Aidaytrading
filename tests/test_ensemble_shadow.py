from __future__ import annotations

import asyncio

import pytest

from ai_trader.services.types import MarketSnapshot
from ai_trader.workers.ml_ensemble_worker import EnsembleMLWorker


class FakeMLService:
    def __init__(self, metrics: dict[str, dict[str, float]] | None = None) -> None:
        self._metrics = metrics or {}
        self.default_threshold = 0.5

    def latest_validation_metrics(self, symbol: str) -> dict[str, float]:
        return dict(self._metrics.get(symbol, {}))

    def set_metrics(self, symbol: str, metrics: dict[str, float]) -> None:
        self._metrics[symbol] = metrics


@pytest.fixture
def snapshot() -> MarketSnapshot:
    return MarketSnapshot(prices={"BTC/USD": 30000.0}, history={}, candles={})


def test_shadow_mode_blocks_trades(snapshot: MarketSnapshot) -> None:
    ml_service = FakeMLService({"BTC/USD": {"accuracy": 0.7, "reward": 0.5, "support": 40}})
    worker = EnsembleMLWorker(
        symbols=["BTC/USD"],
        window_size=40,
        retrain_interval=10,
        ml_service=ml_service,
        shadow_mode=True,
        validation_min_accuracy=0.6,
        validation_min_reward=0.1,
        validation_min_support=20,
    )
    worker._latest_prob["BTC/USD"] = 0.85
    trade = asyncio.run(
        worker.generate_trade(
            "BTC/USD", "buy", snapshot, equity_per_trade=1000.0, existing_position=None
        )
    )
    assert trade is None


def test_validation_gating(snapshot: MarketSnapshot) -> None:
    ml_service = FakeMLService({"BTC/USD": {"accuracy": 0.5, "reward": -0.1, "support": 40}})
    worker = EnsembleMLWorker(
        symbols=["BTC/USD"],
        window_size=40,
        retrain_interval=10,
        ml_service=ml_service,
        shadow_mode=False,
        validation_min_accuracy=0.55,
        validation_min_reward=0.0,
        validation_min_support=20,
    )
    worker._latest_prob["BTC/USD"] = 0.75
    blocked_trade = asyncio.run(
        worker.generate_trade(
            "BTC/USD", "buy", snapshot, equity_per_trade=1500.0, existing_position=None
        )
    )
    assert blocked_trade is None

    ml_service.set_metrics(
        "BTC/USD",
        {"accuracy": 0.72, "reward": 0.35, "support": 45, "avg_confidence": 0.66},
    )
    allowed_trade = asyncio.run(
        worker.generate_trade(
            "BTC/USD", "buy", snapshot, equity_per_trade=1500.0, existing_position=None
        )
    )
    assert allowed_trade is not None
    assert allowed_trade.validation_score == pytest.approx(0.35, rel=1e-3)
    assert allowed_trade.metadata is not None
    assert "validation_metrics" in allowed_trade.metadata
