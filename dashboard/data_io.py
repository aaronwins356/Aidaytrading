"""Data access layer for the trading dashboard.

The module centralises all persistence concerns (SQLite/YAML/log files) and
wraps them with caching, validation and extensive error handling. Every public
function is defensive: failures return empty DataFrames alongside structured
logging so the Streamlit UI can remain responsive and informative instead of
crashing.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import shutil
import sqlite3
from contextlib import suppress
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field, ValidationError

try:  # pragma: no cover - pydantic v2 path
    from pydantic import field_validator
except ImportError:  # pragma: no cover - pydantic v1 path
    field_validator = None
    from pydantic import validator

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
    "database_health",
    "DataHealth",
]

# --------------------------------------------------------------------------------------
# Path configuration
# --------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = str(BASE_DIR / "desk" / "configs" / "config.yaml")
DB_PAPER = str(BASE_DIR / "desk" / "db" / "paper_trading.sqlite")
DB_LIVE = str(BASE_DIR / "desk" / "db" / "live_trading.sqlite")
LOG_DIR = str(BASE_DIR / "desk" / "logs")
BACKUP_DIR = BASE_DIR / "desk" / "backups"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)
LAST_GOOD_CONFIG = BACKUP_DIR / "config.last_good.yaml"

# --------------------------------------------------------------------------------------
# Logging helpers
# --------------------------------------------------------------------------------------
LOGGER = logging.getLogger("dashboard.data_io")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]


def _asdict(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()  # type: ignore[no-any-return]
    return model.dict()  # type: ignore[no-any-return]


def _st_cache(name: str, fallback_maxsize: int = 16):
    """Return a Streamlit cache decorator if available, else a small LRU cache."""

    try:
        import streamlit as st  # type: ignore

        return getattr(st, name)
    except Exception:  # pragma: no cover - executed only outside Streamlit

        def decorator(func):
            from functools import lru_cache

            return lru_cache(maxsize=fallback_maxsize)(func)

        return decorator


_cache_data = _st_cache("cache_data")
_cache_resource = _st_cache("cache_resource")


def _cached_path(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    if not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


@_cache_resource(show_spinner=False)
def get_conn(db_path: str) -> sqlite3.Connection:
    """Return a cached SQLite connection with sensible defaults."""

    path = _cached_path(db_path)
    conn = sqlite3.connect(
        path,
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        check_same_thread=False,
    )
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


def _safe_read_sql(query: str, conn: sqlite3.Connection, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
    try:
        return pd.read_sql_query(query, conn, parse_dates=parse_dates)
    except Exception as exc:  # pragma: no cover - difficult to simulate consistently
        LOGGER.exception("SQL read failed", extra={"query": query, "error": str(exc)})
        return pd.DataFrame()


@_cache_data(show_spinner=False)
def load_trades(db_path: str, mode: Optional[str] = None) -> pd.DataFrame:
    """Load trades from the provided SQLite database.

    The function never raises – if the table is missing or unreadable an empty
    ``DataFrame`` with the expected schema is returned.
    """

    conn = get_conn(db_path)
    trades = _safe_read_sql("SELECT * FROM trades", conn, parse_dates=["opened_at", "closed_at"])
    if trades.empty:
        trades = pd.DataFrame(
            columns=[
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
            ]
        )
    if mode and mode != "Both" and "status" in trades:
        trades = trades[trades["status"].astype(str).str.contains(mode, na=False)]
    if "opened_at" in trades:
        trades.sort_values("opened_at", inplace=True, ascending=True)
    return trades.reset_index(drop=True)


@_cache_data(show_spinner=False)
def load_equity(db_path: str) -> pd.DataFrame:
    conn = get_conn(db_path)
    df = _safe_read_sql("SELECT * FROM equity", conn, parse_dates=["ts"])
    if df.empty:
        df = pd.DataFrame(columns=["ts", "balance", "mode"])
    if "ts" in df:
        df.sort_values("ts", inplace=True)
    return df.reset_index(drop=True)


@_cache_data(show_spinner=False)
def load_positions(db_path: str) -> pd.DataFrame:
    conn = get_conn(db_path)
    df = _safe_read_sql("SELECT * FROM positions", conn, parse_dates=["opened_at"])
    if df.empty:
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


@_cache_data(show_spinner=False)
def load_ml_scores(db_path: str) -> pd.DataFrame:
    conn = get_conn(db_path)
    df = _safe_read_sql("SELECT * FROM ml_scores", conn, parse_dates=["ts"])
    if df.empty:
        df = pd.DataFrame(columns=["ts", "worker", "symbol", "proba_win", "features_json", "label"])
    return df.reset_index(drop=True)


@_cache_data(show_spinner=False)
def load_logs(log_dir: str) -> pd.DataFrame:
    log_path = Path(log_dir)
    if not log_path.exists():
        return pd.DataFrame(columns=["ts", "level", "message", "worker", "payload", "file"])

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
                with suppress(Exception):
                    ts = pd.to_datetime(ts)
                records.append(
                    {
                        "ts": ts,
                        "level": str(payload.get("level", "INFO")).upper(),
                        "message": payload.get("message") or payload.get("msg") or "",
                        "worker": payload.get("worker"),
                        "payload": payload,
                        "file": os.path.basename(file),
                    }
                )
    df = pd.DataFrame(records)
    if not df.empty and "ts" in df:
        df.sort_values("ts", inplace=True)
    return df


class WorkerConfig(BaseModel):
    enabled: bool = True
    max_position: float = 1.0
    risk_per_trade: float = 0.01
    symbols: List[str] = Field(default_factory=lambda: SYMBOLS)
    parameters: Dict[str, Any] = Field(default_factory=dict)

    if field_validator is not None:  # pragma: no branch

        @field_validator("risk_per_trade")
        @classmethod
        def _clamp_risk(cls, value: float) -> float:
            return float(max(0.0, min(value, 0.2)))

    else:  # pragma: no cover - exercised under pydantic v1

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
    base_currency: str = "USD"
    symbols: List[str] = Field(default_factory=lambda: SYMBOLS)
    workers: Dict[str, WorkerConfig] = Field(default_factory=dict)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    kraken_api_key: str = ""
    kraken_api_secret: str = ""

    if field_validator:  # pragma: no branch - handled by import guard above

        @field_validator("mode")
        @classmethod
        def _validate_mode(cls, value: str) -> str:
            allowed = {"Paper", "Live", "Both"}
            if value not in allowed:
                raise ValueError(
                    "mode must match pattern ^(Paper|Live|Both)$"
                )
            return value

    else:  # pragma: no cover - exercised under pydantic v1

        @validator("mode")
        def _validate_mode(cls, value: str) -> str:
            allowed = {"Paper", "Live", "Both"}
            if value not in allowed:
                raise ValueError(
                    "mode must match pattern ^(Paper|Live|Both)$"
                )
            return value


def _default_config() -> DeskConfig:
    return DeskConfig(
        workers={
            "alpha": WorkerConfig(parameters={"timeframe": "1m", "atr_multiplier": 1.5}),
            "beta": WorkerConfig(enabled=False, risk_per_trade=0.02, parameters={"timeframe": "5m"}),
        }
    )


def _write_last_good(data: Dict[str, Any]) -> None:
    try:
        LAST_GOOD_CONFIG.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - best effort only
        LOGGER.warning("Unable to persist last good config", extra={"error": str(exc)})


def _load_last_good() -> Optional[Dict[str, Any]]:
    if LAST_GOOD_CONFIG.exists():
        with open(LAST_GOOD_CONFIG, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return None


def load_config(path: str = CONFIG_PATH) -> Tuple[DeskConfig, Dict[str, Any]]:
    """Load YAML configuration validating with Pydantic.

    Returns both the validated model and the raw dictionary for diff rendering.
    If the file is missing or invalid we fall back to defaults or the most
    recent known-good backup.
    """

    path_obj = _cached_path(path)
    if not path_obj.exists():
        config = _default_config()
        save_config(_asdict(config), path)
        return config, _asdict(config)

    try:
        with open(path_obj, "r", encoding="utf-8") as fh:
            raw_data = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        LOGGER.exception("Failed to parse config", extra={"path": str(path_obj)})
        backup = _load_last_good()
        if backup is not None:
            return DeskConfig(**backup), backup
        config = _default_config()
        save_config(_asdict(config), path)
        return config, _asdict(config)

    defaults = _asdict(_default_config())
    merged = {**defaults, **raw_data}
    workers = {}
    for name, worker_data in (merged.get("workers") or {}).items():
        base = defaults["workers"].get(name, _asdict(WorkerConfig()))
        workers[name] = {**base, **(worker_data or {})}
    merged["workers"] = workers if workers else defaults["workers"]

    try:
        config = DeskConfig(**merged)
    except ValidationError as exc:
        LOGGER.exception("Invalid configuration", extra={"error": str(exc)})
        backup = _load_last_good()
        if backup is not None:
            return DeskConfig(**backup), backup
        config = _default_config()
        save_config(_asdict(config), path)
        return config, _asdict(config)

    _write_last_good(_asdict(config))
    return config, raw_data


def save_config(data: Dict[str, Any], path: str = CONFIG_PATH) -> Tuple[bool, str]:
    """Persist configuration after validating against the Pydantic model."""

    try:
        config = DeskConfig(**data)
    except ValidationError as exc:
        LOGGER.warning("Config validation failed", extra={"error": str(exc)})
        return False, str(exc)

    path_obj = _cached_path(path)
    tmp_path = path_obj.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(_asdict(config), fh, sort_keys=False, allow_unicode=True)
    shutil.move(tmp_path, path_obj)
    _write_last_good(_asdict(config))
    return True, "Configuration saved successfully."


def get_workers_from_trades(trades: pd.DataFrame) -> List[str]:
    workers = sorted({w for w in trades.get("worker", []) if isinstance(w, str) and w})
    return workers


class DataHealth(BaseModel):
    name: str
    status: str
    detail: Optional[str] = None
    age_minutes: Optional[float] = None


def database_health(db_paths: Iterable[str]) -> List[DataHealth]:
    """Return simple health objects describing DB accessibility and data freshness."""

    health: List[DataHealth] = []
    for path in db_paths:
        try:
            conn = get_conn(path)
            cur = conn.cursor()
            cur.execute("SELECT MAX(ts) FROM equity")
            ts = cur.fetchone()[0]
            age = None
            if ts:
                ts_dt = pd.to_datetime(ts)
                age = (pd.Timestamp.utcnow() - ts_dt).total_seconds() / 60
            status = "ok" if ts else "stale"
            health.append(
                DataHealth(
                    name=Path(path).stem,
                    status=status,
                    detail=None if status == "ok" else "No equity rows found",
                    age_minutes=age,
                )
            )
        except Exception as exc:  # pragma: no cover - catastrophic failure path
            LOGGER.exception("DB health check failed", extra={"path": path})
            health.append(
                DataHealth(
                    name=Path(path).stem,
                    status="error",
                    detail=str(exc),
                    age_minutes=None,
                )
            )
    return health


def seed_demo_data(
    db_paths: Iterable[str] = (DB_PAPER, DB_LIVE),
    start: Optional[pd.Timestamp] = None,
    days: int = 60,
    seed: int = 7,
) -> None:
    """Populate the databases with synthetic demo data when empty."""

    rng = np.random.default_rng(seed)
    start = start or (pd.Timestamp.utcnow() - pd.Timedelta(days=days))
    date_index = pd.date_range(start=start, periods=days * 24, freq="h")

    for db in db_paths:
        conn = get_conn(db)
        cur = conn.cursor()
        try:
            cur.execute("SELECT COUNT(*) FROM trades")
            if cur.fetchone()[0] > 0:
                continue
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Unable to inspect trades table", extra={"db": db})
            continue

        balances = 100_000 + np.cumsum(rng.normal(0, 250, size=len(date_index)))
        equity_rows = [
            (ts.to_pydatetime(), float(balance), "Paper" if "paper" in db else "Live")
            for ts, balance in zip(date_index, balances)
        ]
        cur.executemany(
            "INSERT OR REPLACE INTO equity(ts, balance, mode) VALUES (?, ?, ?)",
            equity_rows,
        )

        rows = []
        for i in range(400):
            opened = start + timedelta(hours=int(rng.integers(0, days * 24)))
            length = rng.integers(1, 24)
            closed = opened + timedelta(hours=int(length))
            opened_dt = pd.Timestamp(opened).to_pydatetime()
            closed_dt = pd.Timestamp(closed).to_pydatetime()
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
                    opened_dt,
                    closed_dt,
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
            {
                "timestamp": (datetime.utcnow() - timedelta(minutes=i)).isoformat(),
                "level": "INFO",
                "message": "Heartbeat",
                "worker": "alpha",
                "latency_ms": float(rng.normal(120, 30)),
            }
            for i in range(120)
        ]
        with open(log_file, "w", encoding="utf-8") as fh:
            for entry in demo_entries:
                fh.write(json.dumps(entry) + "\n")


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod()
