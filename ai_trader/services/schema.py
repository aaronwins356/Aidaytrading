"""Central definitions for the SQLite schema used by bot services."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict

# NOTE: Ordered dictionaries are used to keep migrations reproducible when
# iterating over the tables. This avoids oscillating log output in tests.

TRADE_LOG_TABLES: Dict[str, str] = OrderedDict(
    {
        "trades": """
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
                win_loss TEXT,
                reason TEXT,
                metadata_json TEXT
            )
        """,
        "equity_curve": """
            CREATE TABLE IF NOT EXISTS equity_curve (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                equity REAL NOT NULL,
                pnl_percent REAL NOT NULL,
                pnl_usd REAL NOT NULL
            )
        """,
        "market_features": """
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
        """,
        "account_snapshots": """
            CREATE TABLE IF NOT EXISTS account_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                equity REAL NOT NULL,
                balances_json TEXT NOT NULL
            )
        """,
        "bot_state": """
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
        """,
        "control_flags": """
            CREATE TABLE IF NOT EXISTS control_flags (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """,
        "trade_events": """
            CREATE TABLE IF NOT EXISTS trade_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                worker TEXT NOT NULL,
                symbol TEXT NOT NULL,
                event TEXT NOT NULL,
                details_json TEXT NOT NULL
            )
        """,
    }
)

ML_TABLES: Dict[str, str] = OrderedDict(
    {
        "ml_models_state": """
            CREATE TABLE IF NOT EXISTS ml_models_state (
                symbol TEXT PRIMARY KEY,
                logistic BLOB,
                forest BLOB,
                updated_at TEXT NOT NULL,
                metadata_json TEXT
            )
        """,
        "ml_predictions": """
            CREATE TABLE IF NOT EXISTS ml_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                worker TEXT,
                confidence REAL NOT NULL,
                decision INTEGER NOT NULL,
                threshold REAL NOT NULL
            )
        """,
        "ml_metrics": """
            CREATE TABLE IF NOT EXISTS ml_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                mode TEXT NOT NULL,
                precision REAL,
                recall REAL,
                win_rate REAL,
                support INTEGER
            )
        """,
    }
)

ALL_TABLES: Dict[str, str] = OrderedDict({**TRADE_LOG_TABLES, **ML_TABLES})


__all__ = ["TRADE_LOG_TABLES", "ML_TABLES", "ALL_TABLES"]
