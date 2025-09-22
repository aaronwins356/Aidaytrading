"""Central event logging for the trading desk."""

from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Iterable, Optional, Sequence, Tuple

from desk.config import DESK_ROOT
from desk.services.pretty_logger import pretty_logger

class EventLogger:
    """Write-ahead logging to both JSONL and SQLite stores."""

    def __init__(self, logdir: str | Path | None = None, db_path: str | Path | None = None):
        logdir = Path(logdir or DESK_ROOT / "logs")
        logdir.mkdir(parents=True, exist_ok=True)
        self.logfile = logdir / "events.jsonl"

        db_path = Path(db_path or logdir / "trades.db")
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.lock = RLock()
        self._rotating_logger = logging.getLogger("desk.rotating")
        if not self._rotating_logger.handlers:
            handler = RotatingFileHandler(
                logdir / "desk.log",
                maxBytes=5_000_000,
                backupCount=5,
                encoding="utf-8",
            )
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            )
            self._rotating_logger.addHandler(handler)
            self._rotating_logger.propagate = False
        self._rotating_logger.setLevel(logging.INFO)
        self._init_tables()

    # ---------- low-level event file ----------
    def write(self, event: dict) -> None:
        event["timestamp"] = datetime.utcnow().isoformat()
        with self.logfile.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")

    # ---------- schema ----------
    def _init_tables(self):
        create_trades = """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                created_at REAL,
                worker TEXT,
                symbol TEXT,
                side TEXT,
                qty REAL,
                entry_price REAL,
                exit_price REAL,
                exit_reason TEXT,
                pnl REAL,
                status TEXT
            )
        """
        create_equity = """
            CREATE TABLE IF NOT EXISTS equity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                equity REAL
            )
        """
        create_feed = """
            CREATE TABLE IF NOT EXISTS feed_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                level TEXT,
                symbol TEXT,
                message TEXT,
                metadata TEXT
            )
        """
        self.safe_execute(create_trades, commit=True)
        self.safe_execute(create_equity, commit=True)
        self.safe_execute(create_feed, commit=True)
        self._ensure_trade_columns()

    # ---------- helpers ----------
    def safe_execute(
        self,
        sql: str,
        params: Sequence[object] | None = None,
        *,
        commit: bool = False,
    ) -> Optional[sqlite3.Cursor]:
        params_tuple: Tuple[object, ...] = tuple(params or ())
        with self.lock:
            try:
                cursor = self.conn.cursor()
                cursor.execute(sql, params_tuple)
                if commit:
                    self.conn.commit()
                return cursor
            except sqlite3.Error as exc:
                self._rollback_silently()
                self._handle_sql_error("executing SQL", exc, sql=sql, params=params_tuple)
                return None

    def _ensure_trade_columns(self) -> None:
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("PRAGMA table_info(trades)")
            columns = {str(row[1]) for row in cursor.fetchall()}
        if "status" not in columns:
            self.safe_execute(
                "ALTER TABLE trades ADD COLUMN status TEXT DEFAULT 'OPEN'",
                commit=True,
            )
            self.safe_execute(
                """
                UPDATE trades
                SET status = CASE WHEN exit_price IS NULL THEN 'OPEN' ELSE 'CLOSED' END
                """,
                commit=True,
            )
        if "created_at" not in columns:
            self.safe_execute(
                "ALTER TABLE trades ADD COLUMN created_at REAL",
                commit=True,
            )
            now_ts = float(time.time())
            self.safe_execute(
                """
                UPDATE trades
                SET created_at = COALESCE(created_at, timestamp, ?)
                """,
                (now_ts,),
                commit=True,
            )

    # ---------- trade lifecycle ----------
    def log_trade(
        self,
        worker,
        symbol: str,
        side: str,
        qty,
        price,
        pnl: float = 0.0,
        *,
        equity_pct: float | None = None,
    ) -> None:
        worker_name = getattr(worker, "name", str(worker)) if worker is not None else "Unknown"
        timestamp = float(time.time())
        cursor = self.safe_execute(
            """
            INSERT INTO trades (
                timestamp,
                created_at,
                worker,
                symbol,
                side,
                qty,
                entry_price,
                exit_price,
                exit_reason,
                pnl,
                status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                timestamp,
                str(worker_name),
                str(symbol),
                str(side).upper(),
                float(qty),
                float(price),
                None,
                None,
                float(pnl),
                "OPEN",
            ),
            commit=True,
        )
        if cursor and cursor.rowcount > 0:
            usd_value = float(qty) * float(price)
            pretty_logger.trade_opened(
                symbol,
                side,
                float(qty),
                usd_value,
                float(equity_pct or 0.0),
                float(price),
            )

    def log_trade_end(self, worker, symbol: str, exit_price, exit_reason: str, pnl: float) -> None:
        worker_name = getattr(worker, "name", str(worker)) if worker is not None else "Unknown"
        selector = self.safe_execute(
            """
            SELECT id, side, qty, entry_price FROM trades
            WHERE worker=? AND symbol=? AND status='OPEN'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (
                str(worker_name),
                str(symbol),
            ),
        )
        if selector is None:
            return
        row = selector.fetchone()
        if row is None:
            message = (
                "No open trade found to close for worker="
                f"{worker_name} symbol={symbol}."
            )
            self._emit_console("WARNING", message)
            return
        trade_id = int(row[0])
        side = str(row[1])
        qty = float(row[2] or 0.0)
        entry_price = float(row[3] or 0.0)
        updater = self.safe_execute(
            """
            UPDATE trades
            SET exit_price=?, exit_reason=?, pnl=?, status='CLOSED'
            WHERE id=?
            """,
            (
                float(exit_price),
                str(exit_reason),
                float(pnl),
                trade_id,
            ),
            commit=True,
        )
        if updater and updater.rowcount > 0:
            price_delta = 0.0
            if entry_price > 0:
                price_delta = (float(exit_price) - entry_price) / entry_price
            direction = 1.0 if side.upper() in {"BUY", "LONG"} else -1.0
            pnl_pct = price_delta * direction
            pretty_logger.trade_closed(
                symbol,
                side,
                float(exit_price),
                pnl_pct,
                float(pnl),
            )

    # ---------- equity snapshots ----------
    def write_equity(self, equity: float) -> None:
        self.safe_execute(
            "INSERT INTO equity (timestamp, equity) VALUES (?, ?)",
            (float(time.time()), float(equity)),
            commit=True,
        )

    def log_feed_event(self, level: str, symbol: str, message: str, **metadata) -> None:
        payload = {
            "type": "feed",
            "level": str(level).upper(),
            "symbol": str(symbol),
            "message": str(message),
        }
        if metadata:
            payload["metadata"] = metadata
        self.write(payload)
        meta_json = json.dumps(metadata) if metadata else None
        cursor = self.safe_execute(
            """
            INSERT INTO feed_events (timestamp, level, symbol, message, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                float(time.time()),
                str(level).upper(),
                str(symbol),
                str(message),
                meta_json,
            ),
            commit=True,
        )
        if cursor:
            text = f"[Feed] {symbol}: {message}"
            dedupe_key = f"feed:{symbol}:{message}"
            level_name = str(level).upper()
            if level_name == "ERROR":
                pretty_logger.error(text, dedupe_key=dedupe_key)
            elif level_name == "WARNING":
                pretty_logger.warning(text, dedupe_key=dedupe_key)
            else:
                pretty_logger.info(text, dedupe_key=dedupe_key)

    def close(self) -> None:
        """Close the underlying SQLite connection."""

        with self.lock:
            try:
                self.conn.close()
            except Exception:
                pass

    def _emit_console(self, level: str, message: str) -> None:
        level_name = level.upper()
        if level_name == "ERROR":
            pretty_logger.error(message)
        elif level_name == "WARNING":
            pretty_logger.warning(message)
        elif level_name == "TRADE":
            pretty_logger.trade_message(message)
        else:
            pretty_logger.info(message)
        try:
            log_level = getattr(logging, level.upper(), logging.INFO)
            self._rotating_logger.log(log_level, message)
        except Exception:
            pass

    def _handle_sql_error(
        self,
        action: str,
        exc: Exception,
        *,
        sql: str | None = None,
        params: Sequence[object] | None = None,
    ) -> None:
        detail = action
        if sql:
            preview = " ".join(sql.strip().split())
            detail += f" (SQL: {preview}"
            if params:
                detail += f" | params={tuple(params)!r}"
            detail += ")"
        message = f"SQLite error while {detail}: {exc}"
        self._emit_console("ERROR", message)
        try:
            self._rotating_logger.exception(message)
        except Exception:
            pass

    def _rollback_silently(self) -> None:
        try:
            self.conn.rollback()
        except Exception:
            pass
