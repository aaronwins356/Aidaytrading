from __future__ import annotations

import math
import types

import pytest

from desk.runtime import TradingRuntime
from desk.services import Intent, VetoResult


class DummyPortfolio:
    def __init__(self) -> None:
        self.eligibility: dict[str, bool] = {}
        self.recorded_workers: list[str] = []

    def eligible(self, worker_name: str) -> bool:
        return self.eligibility.get(worker_name, True)

    def allocate(self, workers):
        workers = list(workers)
        self.recorded_workers = [worker.name for worker in workers]
        if not workers:
            return {}
        weight = 1.0 / len(workers)
        return {worker.name: weight for worker in workers}


def _intent(name: str) -> Intent:
    worker = types.SimpleNamespace(name=name, allocation=1.0)
    return Intent(
        worker=worker,
        symbol="BTC/USD",
        side="BUY",
        qty=1.0,
        price=100.0,
        score=1.0,
        vetoes=[VetoResult(name="ok", passed=True)],
        features={},
        ml_score=0.5,
    )


def test_allocate_eligible_intents_filters_ineligible_workers():
    runtime = TradingRuntime.__new__(TradingRuntime)
    runtime.portfolio = DummyPortfolio()

    intent_a = _intent("alpha")
    intent_b = _intent("bravo")

    runtime.portfolio.eligibility = {"alpha": True, "bravo": False}

    eligible, allocations = runtime._allocate_eligible_intents([intent_a, intent_b])

    assert [intent.worker.name for intent in eligible] == ["alpha"]
    assert allocations == {"alpha": 1.0}
    # Ensure the allocator only saw eligible workers.
    assert runtime.portfolio.recorded_workers == ["alpha"]


def test_allocate_eligible_intents_returns_empty_when_none_eligible():
    runtime = TradingRuntime.__new__(TradingRuntime)
    runtime.portfolio = DummyPortfolio()

    intent_a = _intent("alpha")
    intent_b = _intent("bravo")

    runtime.portfolio.eligibility = {"alpha": False, "bravo": False}

    eligible, allocations = runtime._allocate_eligible_intents([intent_a, intent_b])

    assert eligible == []
    assert allocations == {}


def test_allocate_eligible_intents_uniform_when_allocator_zero():
    runtime = TradingRuntime.__new__(TradingRuntime)
    runtime.portfolio = DummyPortfolio()

    intent_a = _intent("alpha")
    intent_b = _intent("bravo")

    runtime.portfolio.eligibility = {"alpha": True, "bravo": True}

    def allocate(workers):
        runtime.portfolio.recorded_workers = [worker.name for worker in workers]
        return {worker.name: 0.0 for worker in workers}

    runtime.portfolio.allocate = allocate

    eligible, allocations = runtime._allocate_eligible_intents([intent_a, intent_b])

    assert [intent.worker.name for intent in eligible] == ["alpha", "bravo"]
    assert allocations == {"alpha": 0.5, "bravo": 0.5}


def test_compute_base_risk_targets_weekly_growth():
    runtime = TradingRuntime.__new__(TradingRuntime)
    runtime.workers = [types.SimpleNamespace(name="alpha"), types.SimpleNamespace(name="bravo")]
    runtime._fixed_risk_usd = 25.0
    runtime._weekly_return_target = 1.0
    runtime._trading_days_per_week = 5.0
    runtime._expected_trades_per_day = 4.0

    base_risk = runtime._compute_base_risk(1_000.0)

    daily_target = (1.0 + 1.0) ** (1.0 / 5.0) - 1.0
    expected = 1_000.0 * (daily_target / 4.0)
    assert math.isclose(base_risk, expected, rel_tol=1e-9)


@pytest.mark.parametrize("equity, target, expected", [
    (1_000.0, 0.0, 25.0),
    (-100.0, 1.0, 25.0),
])
def test_compute_base_risk_falls_back_to_fixed(equity: float, target: float, expected: float) -> None:
    runtime = TradingRuntime.__new__(TradingRuntime)
    runtime.workers = [types.SimpleNamespace(name="alpha")]
    runtime._fixed_risk_usd = 25.0
    runtime._weekly_return_target = target
    runtime._trading_days_per_week = 5.0
    runtime._expected_trades_per_day = None

    assert runtime._compute_base_risk(equity) == expected


def test_compute_base_risk_defaults_to_worker_count():
    runtime = TradingRuntime.__new__(TradingRuntime)
    runtime.workers = [
        types.SimpleNamespace(name="alpha"),
        types.SimpleNamespace(name="bravo"),
        types.SimpleNamespace(name="charlie"),
    ]
    runtime._fixed_risk_usd = 15.0
    runtime._weekly_return_target = 1.0
    runtime._trading_days_per_week = 5.0
    runtime._expected_trades_per_day = None

    base_risk = runtime._compute_base_risk(900.0)

    daily_target = (1.0 + 1.0) ** (1.0 / 5.0) - 1.0
    expected = 900.0 * (daily_target / 3.0)
    assert math.isclose(base_risk, expected, rel_tol=1e-9)
