from __future__ import annotations

from typing import List

from desk.research import run_strategy_stub


def _sample_candles() -> List[dict[str, float]]:
    candles = []
    timestamp = 1_600_000_000
    price = 10_000.0
    for _ in range(120):
        candle = {
            "timestamp": float(timestamp),
            "open": price,
            "high": price * 1.002,
            "low": price * 0.998,
            "close": price * 1.001,
            "volume": 12.0,
        }
        candles.append(candle)
        timestamp += 60
        price *= 1.001
    return candles


def test_strategy_stub_runs_without_error() -> None:
    candles = _sample_candles()
    result = run_strategy_stub("breakout_strategy", candles, risk_budget=50.0)
    assert isinstance(result.entries, list)
    assert isinstance(result.exits, list)
    if result.exits:
        exit_event = result.exits[-1]
        assert "pnl" in exit_event
        assert "reason" in exit_event
