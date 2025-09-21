from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from desk.services.logger import EventLogger


class DummyWorker:
    def __init__(self, name: str) -> None:
        self.name = name


def test_logger_trade_lifecycle(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    logger = EventLogger(logdir=tmp_path, db_path=tmp_path / "trades.db")
    worker = DummyWorker("alpha")

    logger.log_trade(worker, "BTC/USD", "BUY", 0.5, 25_000.0)
    logger.log_trade_end(worker, "BTC/USD", 26_000.0, "take_profit", 500.0)

    output = capsys.readouterr().out
    assert "TRADE OPENED" in output
    assert "TRADE CLOSED" in output

    conn = sqlite3.connect(tmp_path / "trades.db")
    row = conn.execute(
        "SELECT status, exit_price, exit_reason, pnl FROM trades ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    assert row is not None
    status, exit_price, exit_reason, pnl = row
    assert status == "CLOSED"
    assert exit_price == pytest.approx(26_000.0)
    assert exit_reason == "take_profit"
    assert pnl == pytest.approx(500.0)


def test_logger_handles_sql_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    logger = EventLogger(logdir=tmp_path, db_path=tmp_path / "trades.db")

    class BrokenConnection:
        def cursor(self):
            raise sqlite3.OperationalError("boom")

        def rollback(self):
            pass

        def commit(self):
            pass

    logger.conn = BrokenConnection()  # type: ignore[assignment]

    logger.log_trade(None, "BTC/USD", "BUY", 0.1, 20_000.0)

    output = capsys.readouterr().out
    assert "SQLite error" in output

    # Ensure subsequent calls still do not raise even though the cursor is broken
    logger.write_equity(10_000.0)
