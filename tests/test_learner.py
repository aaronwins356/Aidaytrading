from __future__ import annotations

from pathlib import Path

import pytest

from desk.services.learner import Learner, pd


@pytest.mark.skipif(pd is None, reason="pandas is required for learner persistence tests")
def test_observe_appends_rows(tmp_path):
    learner = Learner(model_dir=tmp_path)
    trade_base = {
        "timestamp": 1.0,
        "symbol": "BTC/USD",
        "side": "BUY",
        "qty": 0.5,
        "entry_price": 100.0,
        "worker": "alice",
    }

    learner.observe({**trade_base, "features": {"alpha": 1.0}})
    learner.observe({**trade_base, "timestamp": 2.0, "features": {"alpha": 2.0}})

    observations_path = Path(tmp_path) / "observations.csv"
    assert observations_path.exists()

    data = pd.read_csv(observations_path)
    assert len(data) == 2
    assert list(data["timestamp"]) == [1.0, 2.0]
    assert list(data["alpha"]) == [1.0, 2.0]


def test_history_methods_without_pandas(monkeypatch, tmp_path):
    learner = Learner(model_dir=tmp_path)
    monkeypatch.setattr("desk.services.learner.pd", None)

    history = learner.load_trade_history("alice")
    assert history is None

    learner.record_result("alice", {"pnl": 1.0})
    history_path = learner._history_path("alice")
    assert not history_path.exists()
