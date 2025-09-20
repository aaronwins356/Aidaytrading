"""Helpers for loading and persisting bot configuration."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml

# Path to the repository root (two levels up from this file)
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"

_DEFAULT_CONFIG: Dict[str, Any] = {
    "settings": {
        "mode": "paper",
        "exchange": "kraken",
        "api_key": "",
        "api_secret": "",
        "balance": 1000.0,
        "loop_delay": 60,
        "warmup_candles": 10,
    },
    "risk": {
        "fixed_risk_usd": 50.0,
        "rr_ratio": 2.0,
        "stop_loss_pct": 0.02,
        "max_hold_minutes": 15,
        "daily_dd": None,
        "weekly_dd": None,
        "trade_stop_loss": 1.0,
        "max_concurrent": 8,
        "halt_on_dd": False,
        "retrain_every": 10,
        "ml_weight": 0.5,
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


def load_config(path: Path | None = None) -> Dict[str, Any]:
    """Load the bot configuration, falling back to sane defaults."""
    path = path or CONFIG_PATH
    _ensure_config_file(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return _deep_merge(_DEFAULT_CONFIG, raw)


def save_config(config: Dict[str, Any], path: Path | None = None) -> None:
    """Persist the configuration back to disk."""
    path = path or CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
