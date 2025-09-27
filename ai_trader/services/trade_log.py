"""SQLite-backed trade logging utilities."""

from __future__ import annotations

import json
import math
import sqlite3
import logging
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

from ai_trader.services.schema import TRADE_LOG_TABLES
from ai_trader.services.types import TradeIntent


class TradeLog:
    """Persist trades and equity metrics in SQLite."""

    _WRITE_RETRY_ATTEMPTS: int = 3
    _WRITE_RETRY_DELAY_SECONDS: float = 0.2

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(__name__)
        self._wal_initialized = False
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            for statement in TRADE_LOG_TABLES.values():
                self._execute_with_retry(conn, statement)
            self._ensure_market_feature_columns(conn)
            self._ensure_trade_columns(conn)
            conn.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        if not self._wal_initialized:
            # Enable WAL once to improve concurrent writer safety.
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.commit()
            self._wal_initialized = True
        conn.execute("PRAGMA busy_timeout = 5000;")
        try:
            yield conn
        finally:
            conn.close()

    def _execute_with_retry(
        self,
        conn: sqlite3.Connection,
        statement: str,
        parameters: Iterable[object] | None = None,
    ) -> sqlite3.Cursor:
        """Execute a write statement with retry/backoff on transient locks."""

        params = tuple(parameters or ())
        last_error: Exception | None = None
        for attempt in range(1, self._WRITE_RETRY_ATTEMPTS + 1):
            try:
                return conn.execute(statement, params)
            except sqlite3.OperationalError as exc:
                message = str(exc).lower()
                if "locked" not in message and "busy" not in message:
                    raise
                last_error = exc
                conn.rollback()
                self._logger.warning(
                    "SQLite write contention on %s (attempt %d/%d)",
                    self._db_path,
                    attempt,
                    self._WRITE_RETRY_ATTEMPTS,
                )
                time.sleep(self._WRITE_RETRY_DELAY_SECONDS)
        assert last_error is not None
        raise last_error

    def record_trade(self, trade: TradeIntent) -> None:
        """Persist a trade intent to the database."""

        metadata = dict(trade.metadata or {})
        stop_price = self._coerce_float(metadata.get("stop_price"), trade.atr_stop)
        target_price = self._coerce_float(metadata.get("target_price"), trade.atr_target)
        atr_value = self._coerce_float(metadata.get("atr"), trade.atr_value)
        validation_snapshot = metadata.get("validation_metrics") or metadata.get("validation")
        validation_value = self._coerce_float(
            metadata.get("validation_score"),
            trade.validation_score,
        )
        if validation_value is None and isinstance(validation_snapshot, Mapping):
            validation_value = self._coerce_float(validation_snapshot.get("reward"))

        with self._connect() as conn:
            self._execute_with_retry(
                conn,
                """
                INSERT INTO trades (
                    timestamp, worker, symbol, side, cash_spent,
                    entry_price, exit_price, pnl_percent, pnl_usd, win_loss,
                    reason, metadata_json, confidence, atr_stop, atr_target,
                    atr_value, validation_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    trade.reason,
                    json.dumps(metadata),
                    trade.confidence,
                    stop_price,
                    target_price,
                    atr_value,
                    validation_value,
                ),
            )
            conn.commit()

    def has_trade_entry(
        self, worker: str, symbol: str, entry_price: float, cash_spent: float
    ) -> bool:
        """Return ``True`` if a matching trade entry already exists."""

        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT entry_price, cash_spent FROM trades WHERE worker = ? AND symbol = ?",
                (worker, symbol),
            )
            for row in cursor.fetchall():
                logged_entry = float(row["entry_price"])
                logged_cash = float(row["cash_spent"])
                if math.isclose(
                    logged_entry, float(entry_price), rel_tol=1e-6, abs_tol=1e-6
                ) and math.isclose(logged_cash, float(cash_spent), rel_tol=1e-6, abs_tol=1e-4):
                    return True
        return False

    def record_equity(self, equity: float, pnl_percent: float, pnl_usd: float) -> None:
        """Store an equity snapshot."""

        with self._connect() as conn:
            self._execute_with_retry(
                conn,
                "INSERT INTO equity_curve (timestamp, equity, pnl_percent, pnl_usd) VALUES (?, ?, ?, ?)",
                (datetime.utcnow().isoformat(), equity, pnl_percent, pnl_usd),
            )
            conn.commit()

    def fetch_trades(self) -> Iterable[tuple]:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT timestamp, worker, symbol, side, cash_spent, entry_price, exit_price,
                       pnl_percent, pnl_usd, win_loss, reason, metadata_json
                FROM trades
                ORDER BY timestamp DESC
                """
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
            self._execute_with_retry(
                conn,
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

    def backfill_market_feature_label(self, symbol: str, timeframe: str, label: float) -> None:
        """Update the oldest unlabeled feature row once ground truth arrives."""

        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT id FROM market_features
                WHERE symbol = ? AND timeframe = ? AND label IS NULL
                ORDER BY timestamp ASC
                LIMIT 1
                """,
                (symbol, timeframe),
            )
            row = cursor.fetchone()
            if row is None:
                return
            self._execute_with_retry(
                conn,
                "UPDATE market_features SET label = ? WHERE id = ?",
                (float(label), int(row["id"])),
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
            self._execute_with_retry(
                conn,
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
                self._execute_with_retry(
                    conn, f"ALTER TABLE market_features ADD COLUMN {column} REAL"
                )

    def _ensure_trade_columns(self, conn: sqlite3.Connection) -> None:
        """Ensure optional trade auditing columns exist for legacy installs."""

        cursor = conn.execute("PRAGMA table_info(trades)")
        existing = {row[1] for row in cursor.fetchall()}
        if "reason" not in existing:
            self._execute_with_retry(conn, "ALTER TABLE trades ADD COLUMN reason TEXT")
        if "metadata_json" not in existing:
            self._execute_with_retry(conn, "ALTER TABLE trades ADD COLUMN metadata_json TEXT")
        if "confidence" not in existing:
            self._execute_with_retry(conn, "ALTER TABLE trades ADD COLUMN confidence REAL")
        if "atr_stop" not in existing:
            self._execute_with_retry(conn, "ALTER TABLE trades ADD COLUMN atr_stop REAL")
        if "atr_target" not in existing:
            self._execute_with_retry(conn, "ALTER TABLE trades ADD COLUMN atr_target REAL")
        if "atr_value" not in existing:
            self._execute_with_retry(conn, "ALTER TABLE trades ADD COLUMN atr_value REAL")
        if "validation_score" not in existing:
            self._execute_with_retry(conn, "ALTER TABLE trades ADD COLUMN validation_score REAL")

    @staticmethod
    def _coerce_float(*values: object | None) -> float | None:
        for value in values:
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def record_risk_settings(
        self, settings: Mapping[str, object], *, revision: int | None = None
    ) -> int:
        """Persist the provided risk settings and return the applied revision."""

        payload = json.dumps(dict(settings))
        with self._connect() as conn:
            if revision is None:
                cursor = conn.execute("SELECT COALESCE(MAX(revision), 0) FROM risk_settings")
                row = cursor.fetchone()
                latest_revision = int(row[0]) if row and row[0] is not None else 0
                revision = latest_revision + 1
            self._execute_with_retry(
                conn,
                """
                INSERT INTO risk_settings(revision, settings_json, updated_at)
                VALUES(?, ?, ?)
                ON CONFLICT(revision) DO UPDATE SET
                    settings_json = excluded.settings_json,
                    updated_at = excluded.updated_at
                """,
                (revision, payload, datetime.utcnow().isoformat()),
            )
            conn.commit()
        return revision

    def fetch_latest_risk_settings(self) -> Optional[Tuple[int, Dict[str, object]]]:
        """Return the most recent risk settings revision, if any."""

        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT revision, settings_json FROM risk_settings ORDER BY revision DESC LIMIT 1"
            )
            row = cursor.fetchone()
        if row is None:
            return None
        try:
            settings = json.loads(row["settings_json"])
        except (TypeError, ValueError):
            settings = {}
        return int(row["revision"]), dict(settings)

    def record_bot_state(self, worker: str, symbol: str, state: Dict[str, object]) -> None:
        """Store a bot's runtime state for dashboard consumption."""

        indicators = state.get("indicators", {})
        if state.get("ml_warming_up") is not None:
            indicators = {**indicators, "ml_warming_up": state.get("ml_warming_up")}
        risk_profile = {
            key: state.get(key)
            for key in (
                "position_size_pct",
                "leverage",
                "stop_loss_pct",
                "take_profit_pct",
                "trailing_stop_pct",
            )
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

    def record_trade_event(
        self,
        *,
        worker: str,
        symbol: str,
        event: str,
        details: Optional[Dict[str, object]] = None,
    ) -> None:
        """Record supplemental trade lifecycle events for auditing."""

        payload = json.dumps(details or {})
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trade_events (timestamp, worker, symbol, event, details_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (datetime.utcnow().isoformat(), worker, symbol, event, payload),
            )
            conn.commit()

    def fetch_trade_events(self, limit: int = 200) -> Iterable[sqlite3.Row]:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT timestamp, worker, symbol, event, details_json
                FROM trade_events
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (int(limit),),
            )
            return cursor.fetchall()


class MemoryTradeLog:
    """In-memory drop-in replacement used for dry-run sessions."""

    def __init__(self) -> None:
        self._trades: list[dict[str, object]] = []
        self._trade_events: list[dict[str, object]] = []
        self._bot_states: list[tuple[str, str, dict[str, object]]] = []
        self._equity_curve: list[tuple[datetime, float, float, float]] = []
        self._account_snapshots: list[dict[str, object]] = []
        self._market_features: list[dict[str, object]] = []
        self._risk_settings: list[tuple[int, dict[str, object]]] = []

    def record_trade(self, trade: TradeIntent) -> None:
        payload = {
            "timestamp": trade.created_at.isoformat(),
            "worker": trade.worker,
            "symbol": trade.symbol,
            "side": trade.side,
            "cash_spent": trade.cash_spent,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "pnl_percent": trade.pnl_percent,
            "pnl_usd": trade.pnl_usd,
            "win_loss": trade.win_loss,
            "reason": trade.reason,
            "metadata_json": json.dumps(trade.metadata or {}),
            "confidence": trade.confidence,
            "atr_stop": trade.atr_stop,
            "atr_target": trade.atr_target,
            "atr_value": trade.atr_value,
            "validation_score": trade.validation_score,
        }
        self._trades.append(payload)

    def has_trade_entry(
        self, worker: str, symbol: str, entry_price: float, cash_spent: float
    ) -> bool:
        for row in self._trades:
            if row["worker"] != worker or row["symbol"] != symbol:
                continue
            logged_entry = float(row["entry_price"])
            logged_cash = float(row["cash_spent"])
            if math.isclose(
                logged_entry, float(entry_price), rel_tol=1e-6, abs_tol=1e-6
            ) and math.isclose(logged_cash, float(cash_spent), rel_tol=1e-6, abs_tol=1e-4):
                return True
        return False

    def record_equity(self, equity: float, pnl_percent: float, pnl_usd: float) -> None:
        self._equity_curve.append(
            (datetime.utcnow(), float(equity), float(pnl_percent), float(pnl_usd))
        )

    def fetch_trades(self) -> Iterable[dict[str, object]]:
        return list(reversed(self._trades))

    def fetch_equity_curve(self) -> Iterable[tuple]:
        return list(self._equity_curve)

    def record_market_features(self, payload: Dict[str, object]) -> None:
        self._market_features.append(dict(payload))

    def backfill_market_feature_label(self, symbol: str, timeframe: str, label: float) -> None:
        for row in self._market_features:
            if (
                row.get("symbol") == symbol
                and row.get("timeframe") == timeframe
                and row.get("label") is None
            ):
                row["label"] = float(label)
                return

    def fetch_market_features(self, symbol: str, limit: int = 500) -> Iterable[dict[str, object]]:
        rows = [row for row in self._market_features if row.get("symbol") == symbol]
        return list(reversed(rows[-limit:]))

    def record_account_snapshot(self, balances: Dict[str, float], equity: float) -> None:
        snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "equity": float(equity),
            "balances": dict(balances),
        }
        self._account_snapshots.append(snapshot)

    def record_risk_settings(
        self, settings: Mapping[str, object], *, revision: int | None = None
    ) -> int:
        payload = dict(settings)
        if revision is None:
            revision = self._risk_settings[-1][0] + 1 if self._risk_settings else 1
        else:
            revision = int(revision)
        self._risk_settings.append((revision, payload))
        return revision

    def fetch_latest_risk_settings(self) -> Optional[Tuple[int, Dict[str, object]]]:
        if not self._risk_settings:
            return None
        revision, payload = self._risk_settings[-1]
        return revision, dict(payload)

    def record_trade_event(
        self,
        worker: str,
        symbol: str,
        event: str,
        details: Dict[str, object],
    ) -> None:
        self._trade_events.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "worker": worker,
                "symbol": symbol,
                "event": event,
                "details": dict(details),
            }
        )

    def record_bot_state(self, worker: str, symbol: str, state: Dict[str, object]) -> None:
        self._bot_states.append((worker, symbol, dict(state)))
