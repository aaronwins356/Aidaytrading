"""Regression tests for SQLite durability improvements."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

from ai_trader.services.trade_log import TradeLog
from ai_trader.services.types import TradeIntent


def test_trade_log_enables_wal_mode(tmp_path) -> None:
    """WAL mode should be activated on initialisation for safer concurrency."""

    db_path = tmp_path / "trades.db"
    TradeLog(db_path)

    with sqlite3.connect(db_path) as connection:
        mode = connection.execute("PRAGMA journal_mode").fetchone()[0]

    assert str(mode).lower() == "wal"


def test_trade_log_retries_on_locked_writes(monkeypatch, tmp_path) -> None:
    """Simulate a transient lock and ensure the trade insert is retried."""

    db_path = tmp_path / "trades.db"
    trade_log = TradeLog(db_path)

    original_connect = TradeLog._connect
    proxy_state: dict[str, "ProxyConnection"] = {}

    class ProxyConnection:
        def __init__(self, real_conn: sqlite3.Connection) -> None:
            self._inner = real_conn
            self.insert_calls = 0

        def execute(self, statement: str, parameters=()):
            if (
                statement.strip().upper().startswith("INSERT INTO TRADES")
                and self.insert_calls == 0
            ):
                self.insert_calls += 1
                raise sqlite3.OperationalError("database is locked")
            self.insert_calls += 1
            return self._inner.execute(statement, parameters)

        def rollback(self) -> None:
            self._inner.rollback()

        def commit(self) -> None:
            self._inner.commit()

        def __getattr__(self, item):
            return getattr(self._inner, item)

    @contextmanager
    def flaky_connect(self) -> Iterator[sqlite3.Connection]:
        with original_connect(self) as real_conn:
            proxy = ProxyConnection(real_conn)
            proxy_state["proxy"] = proxy
            yield proxy

    monkeypatch.setattr(TradeLog, "_connect", flaky_connect)

    trade = TradeIntent(
        worker="test-worker",
        action="OPEN",
        symbol="BTC/USD",
        side="buy",
        cash_spent=100.0,
        entry_price=25000.0,
    )

    trade_log.record_trade(trade)

    proxy = proxy_state["proxy"]
    assert isinstance(proxy, ProxyConnection)
    assert proxy.insert_calls >= 2

    with sqlite3.connect(db_path) as connection:
        count = connection.execute("SELECT COUNT(1) FROM trades").fetchone()[0]
    assert count == 1
