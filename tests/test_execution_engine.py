import types

import pytest

from desk.services.execution import ExecutionEngine, PositionStore
from desk.services.risk import RiskEngine


class DummyLogger:
    def __init__(self) -> None:
        self.events = []

    def log_feed_event(self, *_args, **_kwargs) -> None:
        pass

    def write(self, payload) -> None:
        self.events.append(payload)

    def log_trade(self, *_args, **_kwargs) -> None:
        pass

    def log_trade_end(self, *_args, **_kwargs) -> None:
        pass


class DummyBroker:
    mode = "paper"

    def __init__(self) -> None:
        self.last_qty = 0.0
        self.attempts = 0

    def minimum_order_config(self, _symbol):
        return 0.1, 3

    def can_execute_market_order(self, *_args, **_kwargs):
        return True

    def market_order(self, symbol, side, qty, **_kwargs):
        self.attempts += 1
        self.last_qty = float(qty)
        return {
            "status": "ok",
            "qty": qty,
            "price": 100.0,
            "client_order_id": "abc",
            "txid": "tx",
        }


class FlakyBroker(DummyBroker):
    def market_order(self, symbol, side, qty, **kwargs):
        if self.attempts == 0:
            self.attempts += 1
            return {"status": "error", "error": "EOrder:Invalid precision"}
        return super().market_order(symbol, side, qty, **kwargs)


def _make_engine(tmp_path, broker=None):
    broker = broker or DummyBroker()
    logger = DummyLogger()
    risk_engine = RiskEngine(0.05, 0.1, 0.02, 5, False, 0.05, base_risk_pct=0.02, max_concurrent_risk_pct=0.05)
    risk_engine.check_account(10_000)
    store = PositionStore(db_path=tmp_path / "positions.db")
    engine = ExecutionEngine(
        broker,
        logger,
        {"min_trade_notional": 10.0},
        risk_engine=risk_engine,
        telemetry=None,
        dashboard_recorder=None,
        position_store=store,
    )
    return engine, broker, logger


def test_open_position_honors_exchange_minimum(tmp_path, capsys):
    engine, broker, _logger = _make_engine(tmp_path)
    worker = types.SimpleNamespace(name="stub")
    trade = engine.open_position(
        worker,
        "BTC/USD",
        "BUY",
        100.0,
        stop_loss=95.0,
        metadata={},
        allocation=1.0,
        risk_budget=50.0,
        minimum_qty=0.2,
        precision=3,
    )
    assert trade is not None
    assert broker.last_qty >= 0.2 - 1e-6
    out = capsys.readouterr().out
    assert "TRADE OPENED" in out
    engine._finalize_trade(trade, 105.0, "take_profit")
    out = capsys.readouterr().out
    assert "TRADE CLOSED" in out
    engine.close()


def test_open_position_retries_on_size_error(tmp_path):
    engine, broker, _logger = _make_engine(tmp_path, broker=FlakyBroker())
    worker = types.SimpleNamespace(name="stub")
    trade = engine.open_position(
        worker,
        "BTC/USD",
        "BUY",
        100.0,
        stop_loss=95.0,
        metadata={},
        allocation=1.0,
        risk_budget=50.0,
        minimum_qty=0.1,
        precision=3,
    )
    assert trade is not None
    assert broker.attempts == 2
    engine.close()
