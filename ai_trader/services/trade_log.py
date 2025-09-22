"""SQLite-backed trade logging utilities."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS market_features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    features_json TEXT NOT NULL,
                    label REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS account_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    equity REAL NOT NULL,
                    balances_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bot_state (
                    worker TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    status TEXT,
                    last_signal TEXT,
                    indicators_json TEXT,
                    risk_json TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(worker, symbol)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS control_flags (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._ensure_market_feature_columns(conn)
            conn.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
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

    def record_market_features(self, payload: Dict[str, object]) -> None:
        """Persist engineered market features for research and ML."""

        symbol = str(payload.get("symbol"))
        timeframe = str(payload.get("timeframe", "1m"))
        features = payload.get("features", {})
        label = payload.get("label")
        open_price = payload.get("open")
        high = payload.get("high")
        low = payload.get("low")
        close = payload.get("close")
        volume = payload.get("volume")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO market_features(
                    timestamp, symbol, timeframe, open, high, low, close, volume, features_json, label
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.utcnow().isoformat(),
                    symbol,
                    timeframe,
                    float(open_price) if open_price is not None else None,
                    float(high) if high is not None else None,
                    float(low) if low is not None else None,
                    float(close) if close is not None else None,
                    float(volume) if volume is not None else None,
                    json.dumps(features),
                    float(label) if label is not None else None,
                ),
            )
            conn.commit()

    def fetch_market_features(self, symbol: str, limit: int = 500) -> Iterable[sqlite3.Row]:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT timestamp, symbol, timeframe, open, high, low, close, volume, features_json, label
                FROM market_features
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (symbol, limit),
            )
            return cursor.fetchall()

    def record_account_snapshot(self, balances: Dict[str, float], equity: float) -> None:
        """Persist the broker balances for dashboard consumption."""

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO account_snapshots(timestamp, equity, balances_json)
                VALUES(?, ?, ?)
                """,
                (datetime.utcnow().isoformat(), equity, json.dumps(balances)),
            )
            conn.commit()

    def fetch_latest_account_snapshot(self) -> Dict[str, object] | None:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT timestamp, equity, balances_json
                FROM account_snapshots
                ORDER BY timestamp DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return None
        return {
            "timestamp": row["timestamp"],
            "equity": float(row["equity"]),
            "balances": json.loads(row["balances_json"] or "{}"),
        }

    def _ensure_market_feature_columns(self, conn: sqlite3.Connection) -> None:
        """Add legacy columns if the database pre-dates the OHLCV schema."""

        cursor = conn.execute("PRAGMA table_info(market_features)")
        existing: List[str] = [row[1] for row in cursor.fetchall()]
        for column in ["open", "high", "low", "close", "volume"]:
            if column not in existing:
                conn.execute(f"ALTER TABLE market_features ADD COLUMN {column} REAL")

    def record_bot_state(self, worker: str, symbol: str, state: Dict[str, object]) -> None:
        """Store a bot's runtime state for dashboard consumption."""

        indicators = state.get("indicators", {})
        risk_profile = {
            key: state.get(key)
            for key in ("position_size_pct", "leverage", "stop_loss_pct", "take_profit_pct", "trailing_stop_pct")
            if state.get(key) is not None
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO bot_state(worker, symbol, status, last_signal, indicators_json, risk_json, updated_at)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(worker, symbol) DO UPDATE SET
                    status = excluded.status,
                    last_signal = excluded.last_signal,
                    indicators_json = excluded.indicators_json,
                    risk_json = excluded.risk_json,
                    updated_at = excluded.updated_at
                """,
                (
                    worker,
                    symbol,
                    state.get("status"),
                    state.get("last_signal"),
                    json.dumps(indicators),
                    json.dumps(risk_profile),
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()

    def fetch_bot_states(self) -> Iterable[sqlite3.Row]:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT worker, symbol, status, last_signal, indicators_json, risk_json, updated_at FROM bot_state"
            )
            return cursor.fetchall()

    def set_control_flag(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO control_flags(key, value, updated_at)
                VALUES(?, ?, datetime('now'))
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (key, value),
            )
            conn.commit()

    def fetch_control_flags(self) -> Dict[str, str]:
        with self._connect() as conn:
            cursor = conn.execute("SELECT key, value FROM control_flags")
            return {row["key"]: row["value"] for row in cursor.fetchall()}
