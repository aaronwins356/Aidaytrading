from __future__ import annotations

from typing import Dict, List
import types

from desk.services.worker import Intent, Worker
from desk.services.risk import RiskEngine


class StubLearner:
    def __init__(self, score: float = 0.75):
        self.score = score
        self.observations: List[Dict[str, float]] = []

    def predict_edge(self, worker, features):
        self.observations.append(features)
        return self.score


def _candle(ts: int, price: float, volume: float) -> Dict[str, float]:
    return {
        "timestamp": float(ts),
        "open": price,
        "high": price + 1,
        "low": price - 1,
        "close": price,
        "volume": volume,
    }

class MockSeries:
    def __init__(self, data):
        self.data = list(data)
        self.iloc = _Indexer(self.data)

    def rolling(self, window):
        return RollingWindow(self.data, window)

    def pct_change(self):
        result = [0.0]
        for prev, current in zip(self.data[:-1], self.data[1:]):
            if prev == 0:
                result.append(0.0)
            else:
                result.append((current - prev) / prev)
        return MockSeries(result)

    def tail(self, n):
        if n <= 0:
            return MockSeries([])
        return MockSeries(self.data[-n:])

    def abs(self):
        return MockSeries([abs(x) for x in self.data])

    def mean(self):
        return sum(self.data) / len(self.data) if self.data else 0.0

    def std(self):
        if not self.data:
            return 0.0
        mean = self.mean()
        variance = sum((x - mean) ** 2 for x in self.data) / len(self.data)
        return variance ** 0.5

    def fillna(self, value):
        return MockSeries([value if _is_nan(x) else x for x in self.data])

    def isna(self):
        return MockSeries([_is_nan(x) for x in self.data])

    def __sub__(self, other):
        return MockSeries([a - b for a, b in zip(self.data, other.data)])

    def __len__(self):
        return len(self.data)


class RollingWindow:
    def __init__(self, data, window):
        self.data = data
        self.window = window

    def max(self):
        return MockSeries(_rolling_reduce(self.data, self.window, max))

    def min(self):
        return MockSeries(_rolling_reduce(self.data, self.window, min))


class _Indexer:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]


def _rolling_reduce(data, window, reducer):
    output = []
    for idx in range(len(data)):
        start = max(0, idx - window + 1)
        window_slice = data[start : idx + 1]
        if len(window_slice) < window:
            output.append(float("nan"))
        else:
            output.append(reducer(window_slice))
    return output


def _is_nan(value):
    return isinstance(value, float) and value != value


class MockDataFrame:
    def __init__(self, rows: List[Dict[str, float]]):
        self.rows = rows
        self.columns = list(rows[0].keys()) if rows else []
        self.iloc = _Indexer(rows)

    def __getitem__(self, key):
        return MockSeries([row[key] for row in self.rows])

    @property
    def empty(self):
        return not self.rows

    def __len__(self):
        return len(self.rows)


class StubStrategy:
    def __init__(self, *_args, **_kwargs):
        pass

    def generate_signals(self, _df):
        return "buy"

    def extract_features(self, _df):
        return {"alpha": 1.0}

    def plan_trade(self, _side, df):
        price = df.iloc[-1]["close"]
        return {
            "stop_loss": price * 0.99,
            "take_profit": price * 1.02,
            "max_hold_minutes": 60,
        }


def build_worker(monkeypatch) -> Worker:
    params = {"params": {"fast_length": 3, "slow_length": 5, "min_qty": 0.01}}
    learner = StubLearner()
    module = types.SimpleNamespace(StubStrategy=StubStrategy)
    monkeypatch.setattr("desk.services.worker.importlib.import_module", lambda name: module)

    worker = Worker(
        name="stub",
        symbol="BTC/USD",
        strategy="stub",
        params=params,
        learner=learner,
    )

    def _frame():
        return MockDataFrame(worker.state.get("candles", []))

    monkeypatch.setattr(worker, "_candles_df", _frame)
    return worker


def test_worker_builds_intent_with_features(monkeypatch):
    worker = build_worker(monkeypatch)
    price = 100.0
    for idx in range(1, 61):
        price += 0.3
        worker.push_candle(_candle(idx * 60 + 600, price, 100 + idx), max_history=100)
    intent = worker.build_intent(risk_budget=50)
    assert intent is not None
    assert isinstance(intent, Intent)
    assert intent.qty > 0
    assert intent.features["ml_edge"] == 0.75
    assert "signal_return_volatility_short" in intent.features
    assert "signal_volume_delta" in intent.features
    assert intent.features["learning_risk_multiplier"] == 1.0


def test_compute_quantity_respects_minimum(monkeypatch):
    worker = build_worker(monkeypatch)
    qty = worker.compute_quantity(price=100, risk_budget=10)
    assert qty >= 0.01


def test_worker_uses_risk_engine_for_sizing(monkeypatch):
    worker = build_worker(monkeypatch)
    worker.risk_engine = RiskEngine(0.05, 0.1, 0.02, 5, False, 0.05)
    qty = worker.compute_quantity(
        price=100,
        risk_budget=50,
        stop_loss=95,
        side="BUY",
    )
    assert qty == 10


def test_learning_risk_profile_scales_with_experience(monkeypatch):
    worker = build_worker(monkeypatch)
    worker.risk_profile = {
        "initial_multiplier": 1.6,
        "floor_multiplier": 0.8,
        "tighten_trades": 50,
        "target_win_rate": 0.6,
    }

    price = 100.0
    for idx in range(1, 61):
        price += 0.2
        worker.push_candle(_candle(idx * 60 + 600, price, 200 + idx), max_history=120)

    early_intent = worker.build_intent(risk_budget=50)
    assert early_intent.features["learning_risk_multiplier"] > 1.0

    worker.state["trades"] = 80
    worker.state["wins"] = 50
    worker.state["losses"] = 30

    mature_intent = worker.build_intent(risk_budget=50)
    assert mature_intent.features["learning_risk_multiplier"] <= early_intent.features["learning_risk_multiplier"]
    assert mature_intent.qty < early_intent.qty
    assert mature_intent.stop_loss > early_intent.stop_loss
