"""Central event logging for the trading desk."""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from threading import Lock

from desk.config import DESK_ROOT

class EventLogger:
    """Write-ahead logging to both JSONL and SQLite stores."""

    def __init__(self, logdir: str | Path | None = None, db_path: str | Path | None = None):
        logdir = Path(logdir or DESK_ROOT / "logs")
        logdir.mkdir(parents=True, exist_ok=True)
        self.logfile = logdir / "events.jsonl"

        db_path = Path(db_path or logdir / "trades.db")
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.lock = Lock()
        self._init_tables()

    # ---------- low-level event file ----------
    def write(self, event: dict) -> None:
        event["timestamp"] = datetime.utcnow().isoformat()
        with self.logfile.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")

    # ---------- schema ----------
    def _init_tables(self):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    worker TEXT,
                    symbol TEXT,
                    side TEXT,
                    qty REAL,
                    entry_price REAL,
                    exit_price REAL,
                    exit_reason TEXT,
                    pnl REAL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS equity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    equity REAL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS feed_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    level TEXT,
                    symbol TEXT,
                    message TEXT,
                    metadata TEXT
                )
            """)
            self.conn.commit()

    # ---------- trade lifecycle ----------
    def log_trade(self, worker, symbol: str, side: str, qty, price, pnl: float = 0.0) -> None:
        worker_name = getattr(worker, "name", str(worker)) if worker is not None else "Unknown"
        with self.lock:
            self.conn.execute(
                """
                INSERT INTO trades
                (timestamp, worker, symbol, side, qty, entry_price, exit_price, exit_reason, pnl)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    float(time.time()),
                    str(worker_name),
                    str(symbol),
                    str(side),
                    float(qty),
                    float(price),
                    None,  # exit_price
                    None,  # exit_reason
                    float(pnl),
                ),
            )
            self.conn.commit()
        print(
            f"[LOGGER] Trade logged: {worker_name} {side} {float(qty):.4f} "
            f"{symbol} @ {float(price):.2f} | PnL {float(pnl):.2f}"
        )

    def log_trade_end(self, worker, symbol: str, exit_price, exit_reason: str, pnl: float) -> None:
        worker_name = getattr(worker, "name", str(worker)) if worker is not None else "Unknown"
        with self.lock:
            self.conn.execute(
                """
                UPDATE trades
                SET exit_price=?, exit_reason=?, pnl=?
                WHERE worker=? AND symbol=? AND exit_price IS NULL
                ORDER BY id DESC LIMIT 1
                """,
                (
                    float(exit_price),
                    str(exit_reason),
                    float(pnl),
                    str(worker_name),
                    str(symbol),
                ),
            )
            self.conn.commit()
        print(
            f"[LOGGER] Trade closed: {worker_name} {exit_reason} @ "
            f"{float(exit_price):.2f} | Realized PnL {float(pnl):.2f}"
        )

    # ---------- equity snapshots ----------
    def write_equity(self, equity: float) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT INTO equity (timestamp, equity) VALUES (?, ?)",
                (float(time.time()), float(equity)),
            )
            self.conn.commit()

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
        with self.lock:
            self.conn.execute(
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
            )
            self.conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""

        with self.lock:
            try:
                self.conn.close()
            except Exception:
                pass
