import sqlite3

import pytest

from desk.services.execution import ExecutionEngine, PositionStore
from desk.services.logger import EventLogger


class DummyLogger:
    def log_trade(self, *args, **kwargs):
        pass

    def write(self, *args, **kwargs):
        pass

    def log_trade_end(self, *args, **kwargs):
        pass


@pytest.fixture()
def position_store(tmp_path):
    store = PositionStore(db_path=tmp_path / "positions.db")
    yield store
    store.close()


def test_event_logger_close(tmp_path):
    logger = EventLogger(logdir=tmp_path / "logs")
    logger.close()
    with pytest.raises(sqlite3.ProgrammingError):
        logger.conn.execute("SELECT 1")


def test_position_store_close(tmp_path):
    store = PositionStore(db_path=tmp_path / "positions.db")
    store.close()
    with pytest.raises(sqlite3.ProgrammingError):
        store.conn.execute("SELECT 1")


def test_execution_engine_close_closes_store(position_store):
    engine = ExecutionEngine(
        broker=None,
        logger=DummyLogger(),
        risk_config={},
        position_store=position_store,
    )
    engine.close()
    with pytest.raises(sqlite3.ProgrammingError):
        position_store.conn.execute("SELECT 1")
