"""SQLite-backed trade logging utilities."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

from .types import TradeIntent


class TradeLog:
    """Persist trades and equity metrics in SQLite."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    worker TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    cash_spent REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    pnl_percent REAL,
                    pnl_usd REAL,
                    win_loss TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS equity_curve (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    equity REAL NOT NULL,
                    pnl_percent REAL NOT NULL,
                    pnl_usd REAL NOT NULL
                )
                """
            )
            conn.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        try:
            yield conn
        finally:
            conn.close()

    def record_trade(self, trade: TradeIntent) -> None:
        """Persist a trade intent to the database."""

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trades (
                    timestamp, worker, symbol, side, cash_spent,
                    entry_price, exit_price, pnl_percent, pnl_usd, win_loss
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade.created_at.isoformat(),
                    trade.worker,
                    trade.symbol,
                    trade.side,
                    trade.cash_spent,
                    trade.entry_price,
                    trade.exit_price,
                    trade.pnl_percent,
                    trade.pnl_usd,
                    trade.win_loss,
                ),
            )
            conn.commit()

    def record_equity(self, equity: float, pnl_percent: float, pnl_usd: float) -> None:
        """Store an equity snapshot."""

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO equity_curve (timestamp, equity, pnl_percent, pnl_usd) VALUES (?, ?, ?, ?)",
                (datetime.utcnow().isoformat(), equity, pnl_percent, pnl_usd),
            )
            conn.commit()

    def fetch_trades(self) -> Iterable[tuple]:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT timestamp, worker, symbol, side, cash_spent, entry_price, exit_price, pnl_percent, pnl_usd, win_loss FROM trades ORDER BY timestamp DESC"
            )
            return cursor.fetchall()

    def fetch_equity_curve(self) -> Iterable[tuple]:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT timestamp, equity, pnl_percent, pnl_usd FROM equity_curve ORDER BY timestamp ASC"
            )
            return cursor.fetchall()
