import asyncio
import math
from typing import List

import pytest

from ai_trader.services.types import MarketSnapshot, OpenPosition, TradeIntent
from ai_trader.workers.ml_ensemble_worker import EnsembleMLWorker


@pytest.fixture
def sample_candles() -> List[dict[str, float]]:
    candles: List[dict[str, float]] = []
    price = 100.0
    for idx in range(200):
        change = math.sin(idx / 4.0) * 0.8 + math.cos(idx / 7.0) * 0.4
        close = price + change
        high = max(price, close) + 0.5
        low = min(price, close) - 0.5
        candles.append(
            {
                "open": price,
                "high": high,
                "low": low,
                "close": close,
                "volume": 5.0,
            }
        )
        price = close
    return candles


def test_feature_generation(sample_candles: List[dict[str, float]]) -> None:
    worker = EnsembleMLWorker(
        ["BTC/USD"], window_size=80, retrain_interval=15, sequence_length=8, min_history=60
    )
    dataset = worker._build_dataset(sample_candles)  # pylint: disable=protected-access
    assert dataset is not None
    df, _ = dataset
    for column in ("ema_fast", "ema_slow", "bollinger_z", "volatility"):
        assert column in df.columns


def test_models_train_and_predict(sample_candles: List[dict[str, float]]) -> None:
    worker = EnsembleMLWorker(
        ["BTC/USD"], window_size=80, retrain_interval=10, sequence_length=6, min_history=60
    )
    closes = [candle["close"] for candle in sample_candles]
    snapshot = MarketSnapshot(
        prices={"BTC/USD": closes[-1]},
        history={"BTC/USD": closes},
        candles={"BTC/USD": sample_candles},
    )
    asyncio.run(worker.evaluate_signal(snapshot))
    assert "BTC/USD" in worker._bundles  # pylint: disable=protected-access
    probability = worker._latest_prob.get("BTC/USD")  # pylint: disable=protected-access
    assert probability is not None and 0.0 <= probability <= 1.0
    state = worker.get_state_snapshot("BTC/USD")
    last_signal = state.get("last_signal")
    assert last_signal in {"buy", "sell", "hold"}


def test_generate_trade_outputs_intent(sample_candles: List[dict[str, float]]) -> None:
    worker = EnsembleMLWorker(
        ["BTC/USD"], window_size=80, retrain_interval=10, sequence_length=6, min_history=60
    )
    closes = [candle["close"] for candle in sample_candles]
    snapshot = MarketSnapshot(
        prices={"BTC/USD": closes[-1]},
        history={"BTC/USD": closes},
        candles={"BTC/USD": sample_candles},
    )
    asyncio.run(worker.evaluate_signal(snapshot))

    async def _open_trade() -> TradeIntent | None:
        return await worker.generate_trade(
            "BTC/USD", "buy", snapshot, equity_per_trade=50.0, existing_position=None
        )

    open_intent = asyncio.run(_open_trade())
    assert open_intent is not None
    assert open_intent.action == "OPEN"
    position = OpenPosition(
        worker=worker.name,
        symbol="BTC/USD",
        side="buy",
        quantity=0.1,
        entry_price=closes[-2],
        cash_spent=10.0,
    )

    async def _close_trade() -> TradeIntent | None:
        return await worker.generate_trade(
            "BTC/USD",
            "sell",
            snapshot,
            equity_per_trade=50.0,
            existing_position=position,
        )

    close_intent = asyncio.run(_close_trade())
    assert close_intent is not None
    assert close_intent.action == "CLOSE"
