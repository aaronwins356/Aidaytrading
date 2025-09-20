from __future__ import annotations

import types

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
        symbol="BTC/USDT",
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
