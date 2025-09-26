"""Utility helpers for running background backtests from the dashboard."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from ai_trader.backtester import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_BPS,
    BacktestTrade,
    run_backtest as _run_backtest_async,
)
from ai_trader.services.configuration import normalize_config, read_config_file
from ai_trader.services.logging import get_logger

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def _parse_date(value: str, *, label: str) -> datetime:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Missing {label} date for backtest")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:  # noqa: B904 - enrich error with user-friendly context
        raise ValueError(f"Invalid {label} date '{value}'. Use YYYY-MM-DD format.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _serialise_trade(trade: BacktestTrade) -> Dict[str, Any]:
    payload = asdict(trade)
    payload["open_time"] = trade.open_time.isoformat()
    if trade.close_time is not None:
        payload["close_time"] = trade.close_time.isoformat()
    if trade.metadata is None:
        payload["metadata"] = {}
    return payload


def _serialise_equity_curve(curve: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    serialised: list[Dict[str, Any]] = []
    for row in curve:
        timestamp = row.get("timestamp")
        if isinstance(timestamp, datetime):
            ts_value = timestamp.isoformat()
        else:
            ts_value = str(timestamp)
        serialised.append({**row, "timestamp": ts_value})
    return serialised


def run_backtest(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    config_path: Path | None = None,
    timeframe: str = "1h",
    fee_rate: float | None = None,
    slippage_bps: float | None = None,
) -> Dict[str, Any]:
    """Execute a backtest and return serialisable payload for dashboard display."""

    logger = get_logger(__name__)
    resolved_config_path = Path(config_path) if config_path else _CONFIG_PATH
    config_data = read_config_file(resolved_config_path)
    if not config_data:
        raise FileNotFoundError(f"Configuration file not found at {resolved_config_path}")
    config = normalize_config(config_data)

    start_dt = _parse_date(start_date, label="start")
    end_dt = _parse_date(end_date, label="end")
    if end_dt <= start_dt:
        raise ValueError("Backtest end date must be after the start date")

    pair = symbol.strip().upper()
    if "/" not in pair:
        raise ValueError("Symbol must be provided in BASE/QUOTE format, e.g. BTC/USD")

    logger.info(
        "Dashboard backtest requested for %s (%s to %s)",
        pair,
        start_dt.date().isoformat(),
        end_dt.date().isoformat(),
    )
    result = asyncio.run(
        _run_backtest_async(
            config,
            pair,
            start_dt,
            end_dt,
            timeframe=timeframe,
            fee_rate=fee_rate if fee_rate is not None else DEFAULT_FEE_RATE,
            slippage_bps=slippage_bps if slippage_bps is not None else DEFAULT_SLIPPAGE_BPS,
        )
    )

    trades = [_serialise_trade(trade) for trade in result.trades]
    payload: Dict[str, Any] = {
        "symbol": pair,
        "start": start_dt.date().isoformat(),
        "end": end_dt.date().isoformat(),
        "equity_curve": _serialise_equity_curve(result.equity_curve),
        "trades": trades,
        "metrics": dict(result.metrics),
    }
    return payload


__all__ = ["run_backtest"]
