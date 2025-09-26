"""QA-focused unit tests covering ML and rule-based workers."""

from __future__ import annotations

import asyncio
import math
from typing import List

import pytest

from ai_trader.services.types import MarketSnapshot, OpenPosition, TradeIntent
from ai_trader.workers.mean_reversion import MeanReversionWorker
from ai_trader.workers.momentum import MomentumWorker
from ai_trader.workers.ml_ensemble_worker import EnsembleMLWorker


@pytest.fixture()
def ensemble_candles() -> List[dict[str, float]]:
    """Return a deterministic candle series for the ensemble worker."""

    candles: List[dict[str, float]] = []
    price = 100.0
    for idx in range(180):
        drift = math.sin(idx / 8.0) * 0.8 + math.cos(idx / 5.0) * 0.4
        close = price + drift + (idx % 5) * 0.2
        high = max(price, close) + 0.6
        low = min(price, close) - 0.6
        candles.append(
            {
                "open": price,
                "high": high,
                "low": low,
                "close": close,
                "volume": 12.0 + idx * 0.05,
            }
        )
        price = close
    return candles


@pytest.fixture()
def ensemble_snapshot(ensemble_candles: List[dict[str, float]]) -> MarketSnapshot:
    closes = [candle["close"] for candle in ensemble_candles]
    return MarketSnapshot(
        prices={"BTC/USDT": closes[-1]},
        history={"BTC/USDT": closes},
        candles={"BTC/USDT": ensemble_candles},
    )


def test_ml_ensemble_generates_probabilities(ensemble_snapshot: MarketSnapshot) -> None:
    worker = EnsembleMLWorker(
        ["BTC/USDT"],
        window_size=90,
        retrain_interval=15,
        sequence_length=8,
        min_history=80,
    )
    worker.price_history["BTC/USDT"].extend(ensemble_snapshot.history["BTC/USDT"][-100:])
    asyncio.run(worker.evaluate_signal(ensemble_snapshot))
    state = worker.get_state_snapshot("BTC/USDT")
    assert state["last_signal"] in {"buy", "sell", "hold", None}
    probability = worker._latest_prob.get("BTC/USDT")  # pylint: disable=protected-access
    assert probability is not None and 0.0 <= probability <= 1.0


def test_ml_ensemble_trade_generation(ensemble_snapshot: MarketSnapshot) -> None:
    worker = EnsembleMLWorker(
        ["BTC/USDT"],
        window_size=90,
        retrain_interval=10,
        sequence_length=6,
        min_history=80,
    )
    worker.price_history["BTC/USDT"].extend(ensemble_snapshot.history["BTC/USDT"][-100:])
    asyncio.run(worker.evaluate_signal(ensemble_snapshot))

    async def _open() -> TradeIntent | None:
        return await worker.generate_trade(
            "BTC/USDT",
            "buy",
            ensemble_snapshot,
            equity_per_trade=250.0,
            existing_position=None,
        )

    open_intent = asyncio.run(_open())
    assert open_intent is not None
    assert open_intent.action == "OPEN"
    assert open_intent.metadata and "probability" in open_intent.metadata

    position = OpenPosition(
        worker=worker.name,
        symbol="BTC/USDT",
        side="buy",
        quantity=0.01,
        entry_price=open_intent.entry_price,
        cash_spent=open_intent.cash_spent,
    )

    async def _close() -> TradeIntent | None:
        return await worker.generate_trade(
            "BTC/USDT",
            "sell",
            ensemble_snapshot,
            equity_per_trade=250.0,
            existing_position=position,
        )

    close_intent = asyncio.run(_close())
    assert close_intent is not None
    assert close_intent.action == "CLOSE"
    assert close_intent.metadata and "probability" in close_intent.metadata


def _build_snapshot(prices: List[float]) -> MarketSnapshot:
    candles = []
    for price in prices:
        candles.append(
            {
                "open": price,
                "high": price + 0.5,
                "low": price - 0.5,
                "close": price,
                "volume": 10.0,
            }
        )
    return MarketSnapshot(
        prices={"BTC/USDT": prices[-1]},
        history={"BTC/USDT": prices},
        candles={"BTC/USDT": candles},
    )


def test_momentum_worker_emits_buy_and_exit() -> None:
    worker = MomentumWorker(["BTC/USDT"], fast_window=5, slow_window=15)
    rising_prices = [100.0 + idx for idx in range(40)]
    worker.price_history["BTC/USDT"].extend(rising_prices[-worker.lookback :])
    snapshot = _build_snapshot(rising_prices)
    signals = asyncio.run(worker.evaluate_signal(snapshot))
    assert signals.get("BTC/USDT") == "buy"
    open_intent = asyncio.run(
        worker.generate_trade(
            "BTC/USDT", "buy", snapshot, equity_per_trade=200.0, existing_position=None
        )
    )
    assert open_intent is not None and open_intent.action == "OPEN"

    falling_prices = [200.0 - idx for idx in range(40)]
    worker.price_history["BTC/USDT"].clear()
    worker.price_history["BTC/USDT"].extend(falling_prices[-worker.lookback :])
    exit_snapshot = _build_snapshot(falling_prices)
    signals = asyncio.run(worker.evaluate_signal(exit_snapshot))
    assert signals.get("BTC/USDT") in {"exit", "sell"}
    position = OpenPosition(
        worker=worker.name,
        symbol="BTC/USDT",
        side="buy",
        quantity=0.1,
        entry_price=open_intent.entry_price,
        cash_spent=open_intent.cash_spent,
    )
    close_intent = asyncio.run(
        worker.generate_trade(
            "BTC/USDT",
            signals.get("BTC/USDT"),
            exit_snapshot,
            equity_per_trade=200.0,
            existing_position=position,
        )
    )
    assert close_intent is not None
    assert close_intent.action == "CLOSE"


def test_mean_reversion_worker_switches_modes() -> None:
    worker = MeanReversionWorker(["BTC/USDT"], window=10, threshold=0.05)
    baseline = [100.0 for _ in range(9)] + [90.0]
    worker.price_history["BTC/USDT"].extend(baseline[-worker.lookback :])
    buy_snapshot = _build_snapshot(baseline)
    signals = asyncio.run(worker.evaluate_signal(buy_snapshot))
    assert signals.get("BTC/USDT") == "buy"
    buy_intent = asyncio.run(
        worker.generate_trade(
            "BTC/USDT", "buy", buy_snapshot, equity_per_trade=150.0, existing_position=None
        )
    )
    assert buy_intent is not None and buy_intent.action == "OPEN"

    exit_prices = [100.0 for _ in range(9)] + [107.0]
    worker.price_history["BTC/USDT"].clear()
    worker.price_history["BTC/USDT"].extend(exit_prices[-worker.lookback :])
    exit_snapshot = _build_snapshot(exit_prices)
    signals = asyncio.run(worker.evaluate_signal(exit_snapshot))
    assert signals.get("BTC/USDT") in {"exit", "sell"}
    position = OpenPosition(
        worker=worker.name,
        symbol="BTC/USDT",
        side="buy",
        quantity=0.05,
        entry_price=buy_intent.entry_price,
        cash_spent=buy_intent.cash_spent,
    )
    close_intent = asyncio.run(
        worker.generate_trade(
            "BTC/USDT",
            signals.get("BTC/USDT"),
            exit_snapshot,
            equity_per_trade=150.0,
            existing_position=position,
        )
    )
    assert close_intent is not None
    assert close_intent.action == "CLOSE"
