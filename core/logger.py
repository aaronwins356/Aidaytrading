# core/logger.py
import sqlite3
import os
import time
from threading import Lock
import json
from datetime import datetime

class EventLogger:
    def __init__(self, logdir="logs", db_path="trades.db"):
        os.makedirs(logdir, exist_ok=True)
        self.logfile = os.path.join(logdir, "events.jsonl")

        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.lock = Lock()
        self._init_tables()

    # ---------- low-level event file ----------
    def write(self, event: dict):
        event["timestamp"] = datetime.utcnow().isoformat()
        with open(self.logfile, "a") as f:
            f.write(json.dumps(event) + "\n")

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
            self.conn.commit()

    # ---------- trade lifecycle ----------
    def log_trade(self, worker, symbol: str, side: str, qty, price, pnl: float = 0.0):
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
        print(f"[LOGGER] Trade logged: {worker_name} {side} {float(qty):.4f} {symbol} @ {float(price):.2f} | PnL {float(pnl):.2f}")

    def log_trade_end(self, worker, symbol: str, exit_price, exit_reason: str, pnl: float):
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
        print(f"[LOGGER] Trade closed: {worker_name} {exit_reason} @ {float(exit_price):.2f} | Realized PnL {float(pnl):.2f}")

    # ---------- equity snapshots ----------
    def write_equity(self, equity):
        with self.lock:
            self.conn.execute(
                "INSERT INTO equity (timestamp, equity) VALUES (?, ?)",
                (float(time.time()), float(equity)),
            )
            self.conn.commit()
