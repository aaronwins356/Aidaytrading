"""Data access layer for the trading dashboard.

This module centralises SQLite/YAML/JSON I/O and applies caching so the
Streamlit application can remain responsive. All reader functions fall back to
synthetic demo data if the underlying resources are missing.
"""
from __future__ import annotations

import glob
import json
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field, ValidationError, validator

__all__ = [
    "CONFIG_PATH",
    "DB_PAPER",
    "DB_LIVE",
    "LOG_DIR",
    "get_conn",
    "load_trades",
    "load_equity",
    "load_positions",
    "load_ml_scores",
    "load_logs",
    "load_config",
    "save_config",
    "seed_demo_data",
    "get_workers_from_trades",
]

CONFIG_PATH = "desk/configs/config.yaml"
DB_PAPER = "desk/db/paper_trading.sqlite"
DB_LIVE = "desk/db/live_trading.sqlite"
LOG_DIR = "desk/logs"

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]


def _st_cache(name: str, fallback_maxsize: int = 16):
    """Return a Streamlit cache decorator if available else a LRU cache."""

    try:
        import streamlit as st  # type: ignore

        return getattr(st, name)
    except Exception:  # pragma: no cover - executed only outside Streamlit

        def decorator(func):
            from functools import lru_cache

            return lru_cache(maxsize=fallback_maxsize)(func)

        return decorator


def _cached_path(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    if not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


@_st_cache("cache_resource")
def get_conn(db_path: str) -> sqlite3.Connection:
    """Return a cached SQLite connection with sensible defaults."""

    path = _cached_path(db_path)
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            trade_id TEXT PRIMARY KEY,
            opened_at TIMESTAMP,
            closed_at TIMESTAMP,
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
            ts TIMESTAMP PRIMARY KEY,
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
            opened_at TIMESTAMP,
            worker TEXT,
            mode TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ml_scores (
            ts TIMESTAMP,
            worker TEXT,
            symbol TEXT,
            proba_win REAL,
            features_json TEXT,
            label INTEGER
        )
        """
    )
    conn.commit()


@_st_cache("cache_data")
def load_trades(db_path: str, mode: Optional[str] = None) -> pd.DataFrame:
    """Load trades from the provided SQLite database.

    Parameters
    ----------
    db_path:
        SQLite path to read from.
    mode:
        Optional filter on the ``status`` column (Paper/Live/Both). If ``None``
        all trades are returned.
    """

    conn = get_conn(db_path)
    try:
        trades = pd.read_sql_query("SELECT * FROM trades", conn, parse_dates=["opened_at", "closed_at"])
    except Exception:
        trades = pd.DataFrame(columns=[
            "trade_id",
            "opened_at",
            "closed_at",
            "symbol",
            "side",
            "qty",
            "entry",
            "exit",
            "pnl",
            "fees",
            "worker",
            "status",
            "stop",
            "target",
            "note",
        ])
    if mode and mode != "Both":
        trades = trades[trades["status"].str.contains(mode, na=False)]
    trades.sort_values("opened_at", inplace=True, ascending=True)
    return trades.reset_index(drop=True)


@_st_cache("cache_data")
def load_equity(db_path: str) -> pd.DataFrame:
    conn = get_conn(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM equity", conn, parse_dates=["ts"])
    except Exception:
        df = pd.DataFrame(columns=["ts", "balance", "mode"])
    df.sort_values("ts", inplace=True)
    return df.reset_index(drop=True)


@_st_cache("cache_data")
def load_positions(db_path: str) -> pd.DataFrame:
    conn = get_conn(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM positions", conn, parse_dates=["opened_at"])
    except Exception:
        df = pd.DataFrame(
            columns=[
                "position_id",
                "symbol",
                "side",
                "qty",
                "avg_entry",
                "stop",
                "target",
                "unrealized",
                "opened_at",
                "worker",
                "mode",
            ]
        )
    return df.reset_index(drop=True)


@_st_cache("cache_data")
def load_ml_scores(db_path: str) -> pd.DataFrame:
    conn = get_conn(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM ml_scores", conn, parse_dates=["ts"])
    except Exception:
        df = pd.DataFrame(columns=["ts", "worker", "symbol", "proba_win", "features_json", "label"])
    return df.reset_index(drop=True)


@_st_cache("cache_data")
def load_logs(log_dir: str) -> pd.DataFrame:
    log_path = Path(log_dir)
    if not log_path.exists():
        return pd.DataFrame(columns=["ts", "level", "message", "worker", "payload"])

    records: List[Dict[str, Any]] = []
    for file in sorted(glob.glob(str(log_path / "**/*.json"), recursive=True)):
        with open(file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    payload = {"raw": line}
                ts = payload.get("timestamp") or payload.get("ts") or Path(file).stat().st_mtime
                try:
                    ts = pd.to_datetime(ts)
                except Exception:
                    ts = pd.Timestamp(Path(file).stat().st_mtime, unit="s")
                records.append(
                    {
                        "ts": ts,
                        "level": payload.get("level", "INFO").upper(),
                        "message": payload.get("message") or payload.get("msg") or "",
                        "worker": payload.get("worker"),
                        "payload": payload,
                        "file": os.path.basename(file),
                    }
                )
    df = pd.DataFrame(records)
    if not df.empty:
        df.sort_values("ts", inplace=True)
    return df


class WorkerConfig(BaseModel):
    enabled: bool = True
    max_position: float = 1.0
    risk_per_trade: float = 0.01
    symbols: List[str] = Field(default_factory=lambda: SYMBOLS)
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @validator("risk_per_trade")
    def _clamp_risk(cls, value: float) -> float:
        return float(max(0.0, min(value, 0.2)))


class RiskConfig(BaseModel):
    exposure_limit: float = 0.5
    leverage_limit: float = 2.0
    max_drawdown: float = 0.25
    slippage_bps: float = 5.0


class ReportingConfig(BaseModel):
    report_dir: str = "desk/reports"
    pdf_logo_path: Optional[str] = None
    timezone: str = "UTC"


class DeskConfig(BaseModel):
    mode: str = Field("Paper", pattern=r"^(Paper|Live|Both)$")
    base_currency: str = "USDT"
    symbols: List[str] = Field(default_factory=lambda: SYMBOLS)
    workers: Dict[str, WorkerConfig] = Field(default_factory=dict)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)


def _default_config() -> DeskConfig:
    return DeskConfig(
        workers={
            "alpha": WorkerConfig(parameters={"timeframe": "1m", "atr_multiplier": 1.5}),
            "beta": WorkerConfig(enabled=False, risk_per_trade=0.02, parameters={"timeframe": "5m"}),
        }
    )


def load_config(path: str = CONFIG_PATH) -> Tuple[DeskConfig, Dict[str, Any]]:
    """Load YAML configuration validating with Pydantic.

    Returns both the validated model and the raw dictionary for diff rendering.
    """

    path_obj = _cached_path(path)
    if not path_obj.exists():
        config = _default_config()
        save_config(config.dict(), path)
        return config, config.dict()

    with open(path_obj, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    defaults = _default_config().dict()
    merged = {**defaults, **data}
    merged["workers"] = {k: {**defaults["workers"].get(k, WorkerConfig().dict()), **v} for k, v in merged.get("workers", {}).items()} if merged.get("workers") else defaults["workers"]
    try:
        config = DeskConfig(**merged)
    except ValidationError as exc:
        raise ValueError(f"Invalid configuration: {exc}") from exc
    return config, data


def save_config(data: Dict[str, Any], path: str = CONFIG_PATH) -> Tuple[bool, str]:
    """Persist configuration after validating against the Pydantic model."""

    try:
        config = DeskConfig(**data)
    except ValidationError as exc:
        return False, str(exc)

    path_obj = _cached_path(path)
    with open(path_obj, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config.dict(), fh, sort_keys=False, allow_unicode=True)
    return True, "Configuration saved successfully."


def get_workers_from_trades(trades: pd.DataFrame) -> List[str]:
    workers = sorted({w for w in trades.get("worker", []) if isinstance(w, str) and w})
    return workers


def seed_demo_data(
    db_paths: Iterable[str] = (DB_PAPER, DB_LIVE),
    start: Optional[pd.Timestamp] = None,
    days: int = 60,
    seed: int = 7,
) -> None:
    """Populate the databases with synthetic demo data when empty."""

    rng = np.random.default_rng(seed)
    start = start or (pd.Timestamp.utcnow() - pd.Timedelta(days=days))
    date_index = pd.date_range(start=start, periods=days * 24, freq="H")

    for db in db_paths:
        conn = get_conn(db)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM trades")
        if cur.fetchone()[0] > 0:
            continue

        balances = 100_000 + np.cumsum(rng.normal(0, 250, size=len(date_index)))
        equity_rows = [(ts.to_pydatetime(), float(balance), "Paper" if "paper" in db else "Live") for ts, balance in zip(date_index, balances)]
        cur.executemany("INSERT OR REPLACE INTO equity(ts, balance, mode) VALUES (?, ?, ?)", equity_rows)

        rows = []
        for i in range(400):
            opened = start + timedelta(hours=int(rng.integers(0, days * 24)))
            length = rng.integers(1, 24)
            closed = opened + timedelta(hours=int(length))
            symbol = rng.choice(SYMBOLS)
            side = rng.choice(["LONG", "SHORT"])
            qty = rng.uniform(0.1, 2.0)
            entry = rng.uniform(20, 40_000)
            exit_price = entry * (1 + rng.normal(0, 0.01))
            pnl = (exit_price - entry) * qty * (1 if side == "LONG" else -1)
            fees = abs(pnl) * 0.001
            worker = rng.choice(["alpha", "beta", "gamma"])
            status = "Paper" if "paper" in db else "Live"
            stop = entry * (1 - rng.uniform(0.005, 0.03))
            target = entry * (1 + rng.uniform(0.005, 0.04))
            rows.append(
                (
                    f"{status}_{i}",
                    opened,
                    closed,
                    symbol,
                    side,
                    qty,
                    entry,
                    exit_price,
                    pnl,
                    fees,
                    worker,
                    status,
                    stop,
                    target,
                    "demo trade",
                )
            )
        cur.executemany(
            """
            INSERT OR REPLACE INTO trades (
                trade_id, opened_at, closed_at, symbol, side, qty, entry, exit, pnl, fees,
                worker, status, stop, target, note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

        ml_rows = []
        for worker in ["alpha", "beta", "gamma"]:
            for symbol in SYMBOLS:
                for ts in date_index[::6]:
                    proba = float(np.clip(rng.normal(0.55, 0.15), 0.05, 0.95))
                    label = int(proba > 0.5 and rng.random() > 0.4)
                    features = json.dumps({"rsi": rng.normal(50, 10), "atr": rng.normal(1.2, 0.2)})
                    ml_rows.append((ts.to_pydatetime(), worker, symbol, proba, features, label))
        cur.executemany(
            "INSERT INTO ml_scores (ts, worker, symbol, proba_win, features_json, label) VALUES (?, ?, ?, ?, ?, ?)",
            ml_rows,
        )
        conn.commit()

    log_dir = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "demo_log.json"
    if not log_file.exists():
        demo_entries = [
            {"timestamp": (datetime.utcnow() - timedelta(minutes=i)).isoformat(), "level": "INFO", "message": "Heartbeat", "worker": "alpha", "latency_ms": float(rng.normal(120, 30))}
            for i in range(120)
        ]
        with open(log_file, "w", encoding="utf-8") as fh:
            for entry in demo_entries:
                fh.write(json.dumps(entry) + "\n")


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod()
