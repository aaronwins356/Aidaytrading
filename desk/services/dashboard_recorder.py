"""Persistence layer that feeds the Streamlit control room database."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Mapping, Optional

from desk.config import DESK_ROOT


class DashboardRecorder:
    """Mirror trading activity into the Streamlit dashboard SQLite store."""

    def __init__(
        self,
        mode: str,
        *,
        db_dir: str | Path | None = None,
    ) -> None:
        self.mode = "Live" if str(mode).lower() == "live" else "Paper"
        base = Path(db_dir or DESK_ROOT / "db")
        base.mkdir(parents=True, exist_ok=True)
        name = "live_trading.sqlite" if self.mode == "Live" else "paper_trading.sqlite"
        self.db_path = base / name
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = Lock()
        self._ensure_schema()

    # ------------------------------------------------------------------
    def _ensure_schema(self) -> None:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    opened_at REAL,
                    closed_at REAL,
                    symbol TEXT,
                    side TEXT,
                    qty REAL,
                    entry REAL,
                    exit REAL,
                    pnl REAL,
                    fees REAL,
                    worker TEXT,
                    status TEXT,
                    stop REAL,
                    target REAL,
                    note TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS equity (
                    ts REAL PRIMARY KEY,
                    balance REAL,
                    mode TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS positions (
                    position_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    side TEXT,
                    qty REAL,
                    avg_entry REAL,
                    stop REAL,
                    target REAL,
                    unrealized REAL,
                    opened_at REAL,
                    worker TEXT,
                    mode TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS ml_scores (
                    ts REAL,
                    worker TEXT,
                    symbol TEXT,
                    proba_win REAL,
                    features_json TEXT,
                    label INTEGER
                )
                """
            )
            self.conn.commit()

    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize_features(features: Mapping[str, Any] | None) -> Dict[str, Any]:
        cleaned: Dict[str, Any] = {}
        if not features:
            return cleaned
        for key, value in features.items():
            try:
                cleaned[str(key)] = float(value)  # normalise to numeric for plotting
            except (TypeError, ValueError):
                continue
        return cleaned

    def _append_note(self, metadata: Mapping[str, Any] | None) -> str:
        if not metadata:
            return ""
        try:
            return json.dumps(metadata)
        except (TypeError, ValueError):
            safe_meta = {str(k): v for k, v in metadata.items()}
            return json.dumps(safe_meta)

    # ------------------------------------------------------------------
    def record_trade_open(
        self,
        trade,
        *,
        fee: float = 0.0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        note = self._append_note(metadata)
        status = f"OPEN|{self.mode}"
        with self.lock:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO trades
                (trade_id, opened_at, closed_at, symbol, side, qty, entry, exit,
                 pnl, fees, worker, status, stop, target, note)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade.trade_id,
                    float(trade.opened_at),
                    None,
                    trade.symbol,
                    trade.side,
                    float(trade.qty),
                    float(trade.entry_price),
                    None,
                    None,
                    float(fee),
                    trade.worker,
                    status,
                    float(trade.stop_loss),
                    float(trade.take_profit),
                    note,
                ),
            )
            self.conn.execute(
                """
                INSERT OR REPLACE INTO positions
                (position_id, symbol, side, qty, avg_entry, stop, target, unrealized,
                 opened_at, worker, mode)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade.trade_id,
                    trade.symbol,
                    trade.side,
                    float(trade.qty),
                    float(trade.entry_price),
                    float(trade.stop_loss),
                    float(trade.take_profit),
                    0.0,
                    float(trade.opened_at),
                    trade.worker,
                    self.mode,
                ),
            )
            self.conn.commit()

    def record_trade_close(
        self,
        trade,
        *,
        exit_price: float,
        exit_reason: str,
        pnl: float,
    ) -> None:
        status = f"CLOSED|{self.mode}|{exit_reason.upper()}"
        closed_at = time.time()
        with self.lock:
            self.conn.execute(
                """
                UPDATE trades
                SET closed_at=?, exit=?, pnl=?, status=?, note=COALESCE(note, '')
                WHERE trade_id=?
                """,
                (
                    closed_at,
                    float(exit_price),
                    float(pnl),
                    status,
                    trade.trade_id,
                ),
            )
            self.conn.execute(
                "DELETE FROM positions WHERE position_id = ?",
                (trade.trade_id,),
            )
            self.conn.commit()

    def record_equity(self, equity: float) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO equity (ts, balance, mode) VALUES (?, ?, ?)",
                (time.time(), float(equity), self.mode),
            )
            self.conn.commit()

    def record_ml_score(
        self,
        worker: str,
        symbol: str,
        *,
        probability: Optional[float] = None,
        features: Optional[Mapping[str, Any]] = None,
        trade_id: Optional[str] = None,
        label: Optional[int] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "features": self._sanitize_features(features),
        }
        if trade_id:
            payload["trade_id"] = trade_id
        timestamp = time.time()
        with self.lock:
            self.conn.execute(
                """
                INSERT INTO ml_scores (ts, worker, symbol, proba_win, features_json, label)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    worker,
                    symbol,
                    None if probability is None else float(probability),
                    json.dumps(payload),
                    None if label is None else int(label),
                ),
            )
            self.conn.commit()

    def update_ml_label(self, trade_id: str, label: int) -> None:
        pattern = f'%"trade_id": "{trade_id}"%'
        with self.lock:
            self.conn.execute(
                """
                UPDATE ml_scores
                SET label=?
                WHERE label IS NULL AND features_json LIKE ?
                """,
                (int(label), pattern),
            )
            self.conn.commit()

    def close(self) -> None:
        with self.lock:
            try:
                self.conn.close()
            except Exception:
                pass


__all__ = ["DashboardRecorder"]

