"""Helpers for loading and persisting bot configuration."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

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
        "mode": "live",
        "exchange": "kraken",
        "api_key": "",
        "api_secret": "",
        "balance": 500.0,
        "loop_delay": 45,
        "warmup_candles": 20,
        "timeframe": "1m",
        "lookback": 300,
        "feed_workers": None,
        "request_timeout": 30.0,
        "kraken_session": {},
    },
    "feed": {
        "exchange": "kraken",
        "symbols": ["BTC/USD", "ETH/USD", "SOL/USD"],
        "timeframe": "1m",
        "data_seeding": {
            "seed_length": 600,
            "max_stale_seconds": 600,
        },
    },
    "risk": {
        "fixed_risk_usd": 40.0,
        "rr_ratio": 2.0,
        "stop_loss_pct": 0.02,
        "max_hold_minutes": 20,
        "daily_dd": None,
        "weekly_dd": None,
        "max_concurrent": 5,
        "halt_on_dd": True,
        "retrain_every": 15,
        "ml_weight": 0.6,
        "trapdoor_pct": 0.015,
        "weekly_return_target": 1.0,
        "trading_days_per_week": 5.0,
        "expected_trades_per_day": None,
        "slippage_bps": 15.0,
        "balance_buffer_pct": 0.05,
        "duplicate_cooldown_seconds": 90.0,
        "learning_risk": {
            "initial_multiplier": 1.5,
            "floor_multiplier": 0.85,
            "tighten_trades": 100,
            "target_win_rate": 0.6,
        },
    },
    "portfolio": {
        "min_weight": 0.05,
        "max_weight": 0.3,
        "epsilon": 0.1,
        "cooldown_minutes": 10,
    },
    "telemetry": {
        "endpoint": "",
        "flush_interval": 1.0,
        "max_backoff": 30.0,
    },
    "ml": {
        "target_win_rate": 0.58,
        "min_samples": 150,
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


def _validate_positive(name: str, value: Any, *, allow_zero: bool = False) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric")
    if allow_zero and numeric == 0:
        return 0.0
    if numeric <= 0:
        raise ValueError(f"{name} must be positive")
    return numeric


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalise the runtime configuration.

    The function raises ``ValueError`` when unrecoverable validation errors are
    encountered.  A defensive copy of ``config`` is returned to ensure callers
    receive a stable dictionary that can be relied upon for live trading.
    """

    validated: Dict[str, Any] = copy.deepcopy(config)
    errors: List[str] = []

    settings = validated.get("settings")
    if not isinstance(settings, dict):
        errors.append("settings must be a mapping")
        settings = {}
    else:
        try:
            settings["loop_delay"] = _validate_positive("settings.loop_delay", settings.get("loop_delay"))
        except ValueError as exc:
            errors.append(str(exc))
        try:
            settings["warmup_candles"] = int(
                _validate_positive(
                    "settings.warmup_candles",
                    settings.get("warmup_candles"),
                )
            )
        except ValueError as exc:
            errors.append(str(exc))
    validated["settings"] = settings

    feed = validated.get("feed")
    if not isinstance(feed, dict):
        errors.append("feed must be a mapping")
        feed = {}
    symbols = feed.get("symbols") if isinstance(feed, dict) else None
    if not isinstance(symbols, Iterable) or isinstance(symbols, (str, bytes)):
        errors.append("feed.symbols must be a list of symbols")
        feed["symbols"] = []
    else:
        normalized_symbols = [str(symbol).upper() for symbol in symbols if str(symbol).strip()]
        if not normalized_symbols:
            errors.append("feed.symbols must contain at least one symbol")
        feed["symbols"] = normalized_symbols
    validated["feed"] = feed

    risk = validated.get("risk")
    if not isinstance(risk, dict):
        errors.append("risk must be a mapping")
        risk = {}
    else:
        for key in ("fixed_risk_usd", "stop_loss_pct"):
            if key in risk:
                try:
                    risk[key] = _validate_positive(f"risk.{key}", risk.get(key))
                except ValueError as exc:
                    errors.append(str(exc))
        if "balance_buffer_pct" in risk:
            try:
                risk["balance_buffer_pct"] = float(risk.get("balance_buffer_pct", 0.0))
                if risk["balance_buffer_pct"] < 0:
                    raise ValueError
            except (TypeError, ValueError):
                errors.append("risk.balance_buffer_pct must be non-negative")
        if "slippage_bps" in risk:
            try:
                risk["slippage_bps"] = _validate_positive(
                    "risk.slippage_bps", risk.get("slippage_bps"), allow_zero=True
                )
            except ValueError as exc:
                errors.append(str(exc))
        if "max_concurrent" in risk:
            try:
                risk["max_concurrent"] = int(_validate_positive("risk.max_concurrent", risk.get("max_concurrent")))
            except ValueError as exc:
                errors.append(str(exc))
        if "duplicate_cooldown_seconds" in risk:
            try:
                risk["duplicate_cooldown_seconds"] = _validate_positive(
                    "risk.duplicate_cooldown_seconds",
                    risk.get("duplicate_cooldown_seconds"),
                    allow_zero=True,
                )
            except ValueError as exc:
                errors.append(str(exc))
        if "max_position_value" in risk and risk.get("max_position_value") is not None:
            try:
                risk["max_position_value"] = _validate_positive(
                    "risk.max_position_value", risk.get("max_position_value")
                )
            except ValueError as exc:
                errors.append(str(exc))
        if "weekly_return_target" in risk and risk.get("weekly_return_target") is not None:
            try:
                risk["weekly_return_target"] = float(risk.get("weekly_return_target"))
            except (TypeError, ValueError):
                errors.append("risk.weekly_return_target must be numeric")
            else:
                if risk["weekly_return_target"] < 0:
                    errors.append("risk.weekly_return_target must not be negative")
        for key in ("trading_days_per_week", "expected_trades_per_day"):
            if key in risk and risk.get(key) is not None:
                try:
                    risk[key] = _validate_positive(f"risk.{key}", risk.get(key))
                except ValueError as exc:
                    errors.append(str(exc))
    validated["risk"] = risk

    workers = validated.get("workers", [])
    if not isinstance(workers, list):
        errors.append("workers must be a list")
        workers = []
    normalised_workers: List[Dict[str, Any]] = []
    for idx, worker in enumerate(workers):
        if not isinstance(worker, dict):
            errors.append(f"workers[{idx}] must be a mapping")
            continue
        missing = [field for field in ("name", "symbol", "strategy") if not worker.get(field)]
        if missing:
            errors.append(f"workers[{idx}] missing fields: {', '.join(missing)}")
            continue
        allocation = worker.get("allocation", worker.get("weight", 0.1))
        try:
            allocation_value = _validate_positive(
                f"workers[{idx}].allocation", allocation
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if allocation_value > 1.0:
            errors.append(
                f"workers[{idx}].allocation must not exceed 1.0 (got {allocation_value})"
            )
            continue
        worker = copy.deepcopy(worker)
        worker["allocation"] = allocation_value
        worker.setdefault("params", {})
        normalised_workers.append(worker)

    validated["workers"] = normalised_workers

    if errors:
        message = "; ".join(dict.fromkeys(errors))
        raise ValueError(f"Invalid configuration: {message}")

    return validated


def load_config(path: Path | None = None) -> Dict[str, Any]:
    """Load the bot configuration, falling back to sane defaults."""
    path = path or CONFIG_PATH
    _ensure_config_file(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    # normalise keys to lowercase for env override compatibility
    merged = _deep_merge(_DEFAULT_CONFIG, raw)
    merged = _apply_env_overrides(merged)
    return validate_config(merged)


def save_config(config: Dict[str, Any], path: Path | None = None) -> None:
    """Persist the configuration back to disk."""
    path = path or CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
