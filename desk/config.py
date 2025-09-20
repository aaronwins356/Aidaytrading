"""Helpers for loading and persisting bot configuration."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict

try:  # pragma: no cover - import guard
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised in tests
    from desk._yaml_stub import yaml  # type: ignore

# Path to the repository root (two levels up from this file)
REPO_ROOT = Path(__file__).resolve().parents[1]
DESK_ROOT = REPO_ROOT / "desk"
CONFIG_PATH = DESK_ROOT / "configs" / "config.yaml"

_DEFAULT_CONFIG: Dict[str, Any] = {
    "settings": {
        "mode": "paper",
        "exchange": "kraken",
        "api_key": "",
        "api_secret": "",
        "balance": 1_000.0,
        "loop_delay": 60,
        "warmup_candles": 10,
        "timeframe": "1m",
        "lookback": 250,
        "feed_workers": None,
        "paper_fee_bps": 10.0,
        "paper_slippage_bps": 5.0,
        "paper_partial_fill_probability": 0.1,
        "paper_min_fill_ratio": 0.6,
        "paper_funding_rate_hourly": 0.0,
    },
    "feed": {
        "exchange": "binance",
        "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"],
        "timeframe": "1m",
    },
    "risk": {
        "fixed_risk_usd": 120.0,
        "rr_ratio": 2.0,
        "stop_loss_pct": 0.02,
        "max_hold_minutes": 15,
        "daily_dd": None,
        "weekly_dd": None,
        "max_concurrent": 8,
        "halt_on_dd": False,
        "retrain_every": 25,
        "ml_weight": 0.5,
        "trapdoor_pct": 0.02,
        "learning_risk": {
            "initial_multiplier": 1.8,
            "floor_multiplier": 0.9,
            "tighten_trades": 150,
            "target_win_rate": 0.6,
        },
    },
    "portfolio": {
        "min_weight": 0.01,
        "max_weight": 0.25,
        "epsilon": 0.1,
        "cooldown_minutes": 15,
    },
    "telemetry": {
        "endpoint": "",
        "flush_interval": 1.0,
        "max_backoff": 30.0,
    },
    "workers": [],
}


def _deep_merge(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``incoming`` into ``base`` without mutating either."""
    result: Dict[str, Any] = copy.deepcopy(base)
    for key, value in incoming.items():
        if (
            isinstance(value, dict)
            and key in result
            and isinstance(result[key], dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _ensure_config_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(_DEFAULT_CONFIG, handle, sort_keys=False)


def _parse_env_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        pass
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    prefix = "DESK_"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        path = key[len(prefix) :].split("__")
        cursor = overrides
        for part in path[:-1]:
            cursor = cursor.setdefault(part.lower(), {})
        cursor[path[-1].lower()] = _parse_env_value(value)
    if overrides:
        config = _deep_merge(config, overrides)
    return config


def load_config(path: Path | None = None) -> Dict[str, Any]:
    """Load the bot configuration, falling back to sane defaults."""
    path = path or CONFIG_PATH
    _ensure_config_file(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    # normalise keys to lowercase for env override compatibility
    merged = _deep_merge(_DEFAULT_CONFIG, raw)
    merged = _apply_env_overrides(merged)
    return merged


def save_config(config: Dict[str, Any], path: Path | None = None) -> None:
    """Persist the configuration back to disk."""
    path = path or CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
