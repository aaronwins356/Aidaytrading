"""FastAPI service exposing runtime status, trades, and risk configuration."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict

from ai_trader.services.monitoring import get_monitoring_center
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
_ML_SERVICE: Any | None = None


class RiskUpdate(BaseModel):
    """Payload schema for runtime risk configuration updates."""

    model_config = ConfigDict(extra="forbid")

    risk_per_trade: float | None = None
    max_drawdown_percent: float | None = None
    daily_loss_limit_percent: float | None = None
    max_open_positions: int | None = None
    max_position_duration_minutes: float | None = None
    confidence_relax_percent: float | None = None
    min_trades_per_day: int | Dict[str, int] | None = None
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
    ml_service: Any | None = None,
) -> None:
    """Attach live service instances provided by the trading runtime."""

    global _TRADE_LOG, _RISK_MANAGER, _RUNTIME_STATE, _ML_SERVICE
    _TRADE_LOG = trade_log
    _RISK_MANAGER = risk_manager
    _RUNTIME_STATE = runtime_state
    _ML_SERVICE = ml_service
    revision: int | None = None
    fetch_latest = getattr(trade_log, "fetch_latest_risk_settings", None)
    if callable(fetch_latest):
        try:
            latest = fetch_latest()
        except Exception:  # noqa: BLE001 - API must remain resilient to DB errors
            latest = None
        if latest:
            revision, persisted_settings = latest
            current_config = risk_manager.config_dict()
            if any(current_config.get(key) != value for key, value in persisted_settings.items()):
                risk_manager.update_config(persisted_settings)
    _RUNTIME_STATE.update_risk_settings(risk_manager.config_dict(), revision=revision)


def reset_services(*, state_file: Path | None = STATE_PATH) -> None:
    """Reset service singletons – useful for test isolation."""

    global _TRADE_LOG, _RISK_MANAGER, _RUNTIME_STATE, _ML_SERVICE
    _TRADE_LOG = None
    _RISK_MANAGER = None
    _RUNTIME_STATE = RuntimeStateStore(state_file)
    _ML_SERVICE = None


def _ensure_services() -> tuple[TradeLog | MemoryTradeLog, RuntimeStateStore, RiskManager]:
    global _TRADE_LOG, _RISK_MANAGER
    if _TRADE_LOG is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _TRADE_LOG = TradeLog(DB_PATH)
    if _RISK_MANAGER is None:
        _RISK_MANAGER = RiskManager()
    return _TRADE_LOG, _RUNTIME_STATE, _RISK_MANAGER


def _load_validation_metrics_from_db() -> Dict[str, Dict[str, Any]]:
    if not DB_PATH.exists():
        return {}
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT symbol, precision, recall, win_rate, support, accuracy, f1_score,
                   reward, avg_confidence, threshold, trades, window, timestamp
            FROM ml_metrics
            WHERE mode = 'validation'
            ORDER BY timestamp DESC
            """
        ).fetchall()
    finally:
        conn.close()
    metrics: Dict[str, Dict[str, float]] = {}
    for row in rows:
        symbol = str(row["symbol"])
        if symbol in metrics:
            continue
        metrics[symbol] = {
            "precision": float(row["precision"] or 0.0),
            "recall": float(row["recall"] or 0.0),
            "win_rate": float(row["win_rate"] or 0.0),
            "support": float(row["support"] or 0.0),
            "accuracy": float(row["accuracy"] or 0.0),
            "f1_score": float(row["f1_score"] or 0.0),
            "reward": float(row["reward"] or 0.0),
            "avg_confidence": float(row["avg_confidence"] or 0.0),
            "threshold": float(row["threshold"] or 0.0),
            "trades": float(row["trades"] or 0.0),
            "window": float(row["window"] or 0.0),
            "timestamp": row["timestamp"],
        }
    return metrics


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
    monitoring = get_monitoring_center()
    snapshot["runtime_degraded"] = monitoring.runtime_degraded
    snapshot["runtime_degraded_reason"] = monitoring.degraded_reason
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


@app.get("/ml-metrics")
async def get_ml_metrics() -> Dict[str, Any]:
    metrics: Dict[str, Any]
    if _ML_SERVICE is not None and hasattr(_ML_SERVICE, "latest_validation_metrics"):
        try:
            metrics = _ML_SERVICE.latest_validation_metrics()  # type: ignore[attr-defined]
        except Exception:
            metrics = {}
    else:
        metrics = {}
    if not metrics:
        metrics = _load_validation_metrics_from_db()
    return {"metrics": metrics}


@app.get("/monitoring")
async def get_monitoring_events(limit: int = Query(50, ge=1, le=200)) -> Dict[str, Any]:
    center = get_monitoring_center()
    events = center.recent_events(limit)
    return {"events": events, "count": len(events)}


@app.get("/risk")
async def get_risk() -> Dict[str, Any]:
    _, runtime_state, risk_manager = _ensure_services()
    config = risk_manager.config_dict()
    runtime_state.update_risk_settings(config)
    runtime_state.refresh_from_disk()
    return runtime_state.risk_snapshot()


@app.post("/config")
async def update_config(payload: RiskUpdate) -> Dict[str, Any]:
    trade_log, runtime_state, risk_manager = _ensure_services()
    updates = payload.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No settings provided")
    config = risk_manager.update_config(updates)
    try:
        revision = trade_log.record_risk_settings(config)
    except Exception as exc:  # noqa: BLE001 - surface persistence issues to clients
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to persist risk settings: {exc}",
        ) from exc
    runtime_state.update_risk_settings(config, revision=revision)
    return {"status": "ok", "config": config, "revision": revision}


def _extract_equity_value(row: Mapping[str, Any] | Sequence[Any]) -> float:
    if isinstance(row, Mapping) or hasattr(row, "keys"):
        try:
            value = row.get("equity") if isinstance(row, Mapping) else row["equity"]
        except Exception:  # noqa: BLE001
            value = None
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0
    if isinstance(row, (list, tuple)):
        if len(row) >= 2:
            try:
                return float(row[1])
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _compute_max_drawdown(trade_log: TradeLog | MemoryTradeLog) -> float:
    try:
        rows = list(trade_log.fetch_equity_curve())
    except Exception:  # noqa: BLE001
        return 0.0
    if not rows:
        return 0.0
    peak_equity = 0.0
    max_drawdown = 0.0
    for row in rows:
        equity = _extract_equity_value(row)
        if equity <= 0.0:
            continue
        if equity > peak_equity:
            peak_equity = equity
        if peak_equity <= 0.0:
            continue
        drawdown = (equity - peak_equity) / peak_equity
        if drawdown < max_drawdown:
            max_drawdown = drawdown
    return abs(max_drawdown) * 100.0


def _latest_ml_accuracy() -> float:
    metrics: Dict[str, Dict[str, Any]] = {}
    if _ML_SERVICE is not None and hasattr(_ML_SERVICE, "latest_validation_metrics"):
        try:
            raw = _ML_SERVICE.latest_validation_metrics()  # type: ignore[attr-defined]
            if isinstance(raw, Mapping):
                metrics = {str(symbol): dict(values) for symbol, values in raw.items()}
        except Exception:  # noqa: BLE001
            metrics = {}
    if not metrics:
        metrics = _load_validation_metrics_from_db()
    best = 0.0
    for payload in metrics.values():
        try:
            value = float(payload.get("accuracy", 0.0))
        except (TypeError, ValueError):
            value = 0.0
        if value > best:
            best = value
    return best


def _watchdog_age_seconds(runtime_state: RuntimeStateStore) -> float:
    last_update = runtime_state.last_update_time()
    if last_update is None:
        return -1.0
    now = datetime.now(timezone.utc)
    return max((now - last_update).total_seconds(), 0.0)


def _websocket_reconnect_count() -> int:
    center = get_monitoring_center()
    events = center.recent_events()
    count = 0
    for event in events:
        try:
            if event.get("event_type") == "websocket_reconnect":
                count += 1
        except AttributeError:
            continue
    return count


@app.get("/metrics", response_class=PlainTextResponse)
async def get_metrics() -> str:
    trade_log, runtime_state, _ = _ensure_services()
    runtime_state.refresh_from_disk()
    status_snapshot = runtime_state.status_snapshot()
    equity = float(status_snapshot.get("equity") or 0.0)
    open_positions = status_snapshot.get("open_positions") or []
    drawdown = _compute_max_drawdown(trade_log)
    watchdog_age = _watchdog_age_seconds(runtime_state)
    accuracy = _latest_ml_accuracy()
    reconnects = _websocket_reconnect_count()
    lines = [
        "# HELP trader_equity_total Current account equity in USD.",
        "# TYPE trader_equity_total gauge",
        f"trader_equity_total {equity:.6f}",
        "# HELP trader_open_positions Number of open positions tracked by the runtime.",
        "# TYPE trader_open_positions gauge",
        f"trader_open_positions {len(open_positions)}",
        "# HELP trader_max_drawdown_percent Maximum recorded drawdown percent from equity curve.",
        "# TYPE trader_max_drawdown_percent gauge",
        f"trader_max_drawdown_percent {drawdown:.6f}",
        "# HELP trader_watchdog_last_update_age_seconds Age of the last runtime heartbeat in seconds.",
        "# TYPE trader_watchdog_last_update_age_seconds gauge",
        f"trader_watchdog_last_update_age_seconds {watchdog_age:.6f}",
        "# HELP trader_ml_validation_accuracy Latest ML validation accuracy.",
        "# TYPE trader_ml_validation_accuracy gauge",
        f"trader_ml_validation_accuracy {accuracy:.6f}",
        "# HELP trader_websocket_reconnect_total Count of websocket reconnect events since startup.",
        "# TYPE trader_websocket_reconnect_total counter",
        f"trader_websocket_reconnect_total {reconnects}",
    ]
    return "\n".join(lines) + "\n"


__all__ = [
    "app",
    "attach_services",
    "get_runtime_state",
    "reset_services",
]
