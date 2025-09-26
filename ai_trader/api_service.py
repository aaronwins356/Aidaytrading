"""FastAPI service exposing runtime status, trades, and risk configuration."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict

from ai_trader.services.risk import RiskManager
from ai_trader.services.runtime_state import RuntimeStateStore
from ai_trader.services.trade_log import MemoryTradeLog, TradeLog

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "trades.db"
STATE_PATH = DATA_DIR / "runtime_state.json"

app = FastAPI(title="AI Trader Control API", version="1.0.0")

_RUNTIME_STATE: RuntimeStateStore = RuntimeStateStore(STATE_PATH)
_TRADE_LOG: TradeLog | MemoryTradeLog | None = None
_RISK_MANAGER: RiskManager | None = None


class RiskUpdate(BaseModel):
    """Payload schema for runtime risk configuration updates."""

    model_config = ConfigDict(extra="forbid")

    risk_per_trade: float | None = None
    max_drawdown_percent: float | None = None
    daily_loss_limit_percent: float | None = None
    max_open_positions: int | None = None
    max_position_duration_minutes: float | None = None
    confidence_relax_percent: float | None = None
    min_trades_per_day: int | None = None
    atr_stop_loss_multiplier: float | None = None
    atr_take_profit_multiplier: float | None = None
    min_stop_buffer: float | None = None


def get_runtime_state() -> RuntimeStateStore:
    """Return the global runtime state container."""

    return _RUNTIME_STATE


def attach_services(
    *,
    trade_log: TradeLog | MemoryTradeLog,
    runtime_state: RuntimeStateStore,
    risk_manager: RiskManager,
) -> None:
    """Attach live service instances provided by the trading runtime."""

    global _TRADE_LOG, _RISK_MANAGER, _RUNTIME_STATE
    _TRADE_LOG = trade_log
    _RISK_MANAGER = risk_manager
    _RUNTIME_STATE = runtime_state
    _RUNTIME_STATE.update_risk_settings(risk_manager.config_dict())


def reset_services(*, state_file: Path | None = STATE_PATH) -> None:
    """Reset service singletons – useful for test isolation."""

    global _TRADE_LOG, _RISK_MANAGER, _RUNTIME_STATE
    _TRADE_LOG = None
    _RISK_MANAGER = None
    _RUNTIME_STATE = RuntimeStateStore(state_file)


def _ensure_services() -> tuple[TradeLog | MemoryTradeLog, RuntimeStateStore, RiskManager]:
    global _TRADE_LOG, _RISK_MANAGER
    if _TRADE_LOG is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _TRADE_LOG = TradeLog(DB_PATH)
    if _RISK_MANAGER is None:
        _RISK_MANAGER = RiskManager()
    return _TRADE_LOG, _RUNTIME_STATE, _RISK_MANAGER


def _row_to_dict(row: Mapping[str, Any] | Iterable[Any]) -> Dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    if hasattr(row, "keys"):
        return {key: row[key] for key in row.keys()}
    ordered_keys = [
        "timestamp",
        "worker",
        "symbol",
        "side",
        "cash_spent",
        "entry_price",
        "exit_price",
        "pnl_percent",
        "pnl_usd",
        "win_loss",
        "reason",
        "metadata_json",
    ]
    values = list(row)
    return {key: values[idx] if idx < len(values) else None for idx, key in enumerate(ordered_keys)}


def _latest_trade_timestamp(trade_log: TradeLog | MemoryTradeLog) -> str | None:
    try:
        rows = list(trade_log.fetch_trades())
    except Exception:
        return None
    if not rows:
        return None
    first = _row_to_dict(rows[0])
    timestamp = first.get("timestamp")
    return str(timestamp) if timestamp is not None else None


def _format_trade(row: Mapping[str, Any] | Iterable[Any]) -> Dict[str, Any]:
    payload = _row_to_dict(row)
    metadata: Dict[str, Any] = {}
    metadata_raw = payload.get("metadata_json")
    if isinstance(metadata_raw, str) and metadata_raw:
        try:
            metadata = json.loads(metadata_raw)
        except json.JSONDecodeError:
            metadata = {}
    price = payload.get("exit_price") or payload.get("entry_price")
    price_value = float(price) if price is not None else None
    quantity = metadata.get("fill_quantity") or metadata.get("quantity")
    if quantity is None and price_value:
        try:
            quantity = float(payload.get("cash_spent", 0.0)) / price_value
        except (TypeError, ZeroDivisionError):
            quantity = None
    pnl_usd = payload.get("pnl_usd")
    pnl_percent = payload.get("pnl_percent")
    return {
        "timestamp": payload.get("timestamp"),
        "worker": payload.get("worker"),
        "pair": payload.get("symbol"),
        "side": payload.get("side"),
        "quantity": quantity,
        "price": price_value,
        "pnl": {
            "usd": float(pnl_usd) if pnl_usd is not None else None,
            "percent": float(pnl_percent) if pnl_percent is not None else None,
        },
    }


@app.get("/status")
async def get_status() -> Dict[str, Any]:
    trade_log, runtime_state, _ = _ensure_services()
    runtime_state.refresh_from_disk()
    snapshot = runtime_state.status_snapshot()
    if snapshot.get("last_trade_timestamp") is None:
        snapshot["last_trade_timestamp"] = _latest_trade_timestamp(trade_log)
    return snapshot


@app.get("/profit")
async def get_profit() -> Dict[str, Any]:
    _, runtime_state, _ = _ensure_services()
    runtime_state.refresh_from_disk()
    return runtime_state.profit_snapshot()


@app.get("/trades")
async def get_trades(limit: int = Query(10, ge=1, le=200)) -> Dict[str, Any]:
    trade_log, runtime_state, _ = _ensure_services()
    runtime_state.refresh_from_disk()
    try:
        rows = list(trade_log.fetch_trades())
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc
    trimmed = rows[:limit]
    return {
        "trades": [_format_trade(row) for row in trimmed],
        "count": len(trimmed),
    }


@app.get("/risk")
async def get_risk() -> Dict[str, Any]:
    _, runtime_state, risk_manager = _ensure_services()
    config = risk_manager.config_dict()
    runtime_state.update_risk_settings(config)
    runtime_state.refresh_from_disk()
    return runtime_state.risk_snapshot()


@app.post("/config")
async def update_config(payload: RiskUpdate) -> Dict[str, Any]:
    _, runtime_state, risk_manager = _ensure_services()
    updates = payload.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No settings provided")
    config = risk_manager.update_config(updates)
    runtime_state.update_risk_settings(config)
    return {"status": "ok", "config": config}


__all__ = [
    "app",
    "attach_services",
    "get_runtime_state",
    "reset_services",
]
