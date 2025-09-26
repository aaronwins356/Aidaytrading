"""Utilities for loading and normalising bot configuration files."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping

import yaml

from ai_trader.services.logging import get_logger

_ML_KEY_ALIASES: Dict[str, str] = {
    "learning_rate": "lr",
    "learningRate": "lr",
    "eta": "lr",
    "regularization_lambda": "regularization",
    "lambda": "regularization",
    "ensemble_trees": "forest_size",
    "n_trees": "forest_size",
    "forestModels": "forest_size",
    "forest_models": "forest_size",
    "ensemble_enabled": "ensemble",
    "threshold_probability": "threshold",
    "probability_threshold": "threshold",
}

_ML_DEFAULTS: Dict[str, float | bool] = {
    "lr": 0.03,
    "regularization": 0.0005,
    "forest_size": 10,
    "threshold": 0.5,
    "ensemble": True,
    "warmup_samples": 25,
}

_DEPRECATED_KEYS: Dict[str, str] = {
    "ensemble_enabled": "Use 'ensemble' under the ml section instead.",
    "ensemble_trees": "Use 'forest_size' for consistency with river.",
    "learning_rate": "Use 'lr' for SGD optimisers.",
}

_SYMBOL_ALIASES: Dict[str, str] = {
    "XBT": "BTC",
    "XXBT": "BTC",
    "BTC": "BTC",
    "XETH": "ETH",
    "ETH2": "ETH",
    "ETH": "ETH",
    "XETC": "ETC",
}


def normalize_symbol(symbol: object) -> str | None:
    """Return a canonical ``BASE/QUOTE`` string understood by Kraken."""

    if symbol is None:
        return None
    text = str(symbol).strip().upper()
    if not text or "/" not in text:
        return None
    base, quote = text.split("/", 1)
    base = base.split(".", 1)[0]
    quote = quote.split(".", 1)[0]
    base = _SYMBOL_ALIASES.get(base, base)
    quote = _SYMBOL_ALIASES.get(quote, quote)
    return f"{base}/{quote}"


def _normalise_symbol_sequence(symbols: object) -> list[str]:
    """Normalise an arbitrary symbol container into Kraken-friendly pairs."""

    normalised: list[str] = []
    seen: set[str] = set()
    if isinstance(symbols, (list, tuple, set)):
        iterable: Iterable[object] = symbols
    elif symbols is None:
        iterable = []
    else:
        iterable = [symbols]
    for candidate in iterable:
        symbol = normalize_symbol(candidate)
        if symbol and symbol not in seen:
            seen.add(symbol)
            normalised.append(symbol)
    return normalised


def read_config_file(path: Path) -> Dict[str, Any]:
    """Return a dictionary representation of the YAML configuration file."""

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, MutableMapping):
        raise ValueError(f"Configuration at {path} must be a mapping")
    return dict(payload)


def normalize_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalise configuration keys and warn on deprecated ML parameters."""

    logger = get_logger(__name__)
    normalised: Dict[str, Any] = deepcopy(dict(config))

    exchange_cfg = dict(normalised.get("exchange", {}))
    kraken_cfg = dict(normalised.get("kraken", {}))
    if not exchange_cfg and kraken_cfg:
        exchange_cfg = kraken_cfg
    if exchange_cfg.get("name", "kraken").lower() == "kraken":
        kraken_defaults = {
            "api_key": "",
            "api_secret": "",
            "rest_rate_limit": 0.5,
            "enable_rate_limit": True,
        }
        merged = {**kraken_defaults, **exchange_cfg, **kraken_cfg}
        merged["enable_rate_limit"] = bool(merged.get("enable_rate_limit", True))
        try:
            merged["rest_rate_limit"] = float(merged.get("rest_rate_limit", 0.5))
        except (TypeError, ValueError):  # pragma: no cover - configuration sanitation
            raise ValueError("Kraken rest_rate_limit must be numeric")
        normalised["exchange"] = merged
        normalised["kraken"] = {
            "api_key": merged.get("api_key", ""),
            "api_secret": merged.get("api_secret", ""),
            "rest_rate_limit": merged.get("rest_rate_limit", 0.5),
        }
    else:
        normalised["exchange"] = exchange_cfg

    ml_cfg: Dict[str, Any] = dict(normalised.get("ml", {}))

    # Record which deprecated keys were encountered so we only warn once per key.
    for deprecated_key, canonical_key in _ML_KEY_ALIASES.items():
        if deprecated_key in ml_cfg and canonical_key not in ml_cfg:
            logger.warning(
                "Deprecated ML config key '%s' detected; prefer '%s' instead.",
                deprecated_key,
                canonical_key,
            )
            ml_cfg[canonical_key] = ml_cfg.pop(deprecated_key)

    for key, default_value in _ML_DEFAULTS.items():
        ml_cfg.setdefault(key, default_value)

    for deprecated_key, guidance in _DEPRECATED_KEYS.items():
        if deprecated_key in normalised.get("ml", {}):
            logger.warning("%s", guidance)

    try:
        ml_cfg["lr"] = float(ml_cfg.get("lr", _ML_DEFAULTS["lr"]))
    except (TypeError, ValueError):  # pragma: no cover - configuration sanitation
        raise ValueError("ML configuration 'lr' must be numeric")
    # Expose a descriptive alias for downstream services that prefer the
    # ``learning_rate`` keyword while retaining backward compatibility with
    # existing configuration files using ``lr``.
    ml_cfg["learning_rate"] = ml_cfg["lr"]
    try:
        ml_cfg["regularization"] = float(
            ml_cfg.get("regularization", _ML_DEFAULTS["regularization"])
        )
    except (TypeError, ValueError):  # pragma: no cover - configuration sanitation
        raise ValueError("ML configuration 'regularization' must be numeric")
    try:
        ml_cfg["forest_size"] = int(ml_cfg.get("forest_size", _ML_DEFAULTS["forest_size"]))
    except (TypeError, ValueError):  # pragma: no cover - configuration sanitation
        raise ValueError("ML configuration 'forest_size' must be an integer")
    try:
        ml_cfg["threshold"] = float(ml_cfg.get("threshold", _ML_DEFAULTS["threshold"]))
    except (TypeError, ValueError):  # pragma: no cover - configuration sanitation
        raise ValueError("ML configuration 'threshold' must be numeric")
    ml_cfg["ensemble"] = bool(ml_cfg.get("ensemble", True))
    try:
        ml_cfg["warmup_samples"] = int(ml_cfg.get("warmup_samples", _ML_DEFAULTS["warmup_samples"]))
    except (TypeError, ValueError):  # pragma: no cover - configuration sanitation
        raise ValueError("ML configuration 'warmup_samples' must be an integer")

    normalised["ml"] = ml_cfg

    trading_cfg = dict(normalised.get("trading", {}))
    trading_cfg.setdefault("paper_trading", True)
    trading_cfg.setdefault("equity_allocation_percent", 2.0)
    trading_cfg.setdefault("paper_starting_equity", 25000.0)
    trading_cfg.setdefault("max_open_positions", 3)
    trading_cfg.setdefault("min_cash_per_trade", 0.0)
    trading_cfg.setdefault("max_cash_per_trade", 0.0)
    trading_cfg.setdefault("trade_confidence_min", 0.5)
    raw_symbols = trading_cfg.get("symbols", [])
    normalised_symbols = _normalise_symbol_sequence(raw_symbols)
    if raw_symbols and not normalised_symbols:
        logger.warning("No valid trading symbols were configured; check trading.symbols")
    trading_cfg["symbols"] = normalised_symbols
    trading_cfg.setdefault("allow_shorting", False)
    paper_flag = bool(trading_cfg.get("paper_trading", True))
    trading_cfg["paper_trading"] = paper_flag
    trading_cfg["mode"] = "paper" if paper_flag else "live"
    trading_cfg["allow_shorting"] = bool(trading_cfg.get("allow_shorting", False))
    trading_cfg["equity_allocation_percent"] = float(
        trading_cfg.get("equity_allocation_percent", 2.0)
    )
    trading_cfg["paper_starting_equity"] = float(trading_cfg.get("paper_starting_equity", 25000.0))
    trading_cfg["max_open_positions"] = int(trading_cfg.get("max_open_positions", 3))
    min_cash = max(0.0, float(trading_cfg.get("min_cash_per_trade", 0.0)))
    max_cash_raw = float(trading_cfg.get("max_cash_per_trade", 0.0))
    max_cash = max(min_cash, max_cash_raw) if max_cash_raw > 0 else max_cash_raw
    trading_cfg["min_cash_per_trade"] = min_cash
    trading_cfg["max_cash_per_trade"] = max_cash
    trading_cfg["trade_fee_percent"] = float(trading_cfg.get("trade_fee_percent", 0.0))
    confidence_floor = float(trading_cfg.get("trade_confidence_min", 0.5))
    trading_cfg["trade_confidence_min"] = max(0.0, min(1.0, confidence_floor))
    normalised["trading"] = trading_cfg

    risk_cfg = dict(normalised.get("risk", {}))
    risk_cfg.setdefault("max_drawdown_percent", 12.0)
    risk_cfg.setdefault("daily_loss_limit_percent", 2.0)
    risk_cfg.setdefault("max_position_duration_minutes", 120)
    risk_cfg["max_drawdown_percent"] = float(risk_cfg.get("max_drawdown_percent", 12.0))
    risk_cfg["daily_loss_limit_percent"] = float(risk_cfg.get("daily_loss_limit_percent", 2.0))
    risk_cfg["max_position_duration_minutes"] = float(
        risk_cfg.get("max_position_duration_minutes", 120)
    )
    normalised["risk"] = risk_cfg

    worker_cfg = dict(normalised.get("workers", {}))
    definitions = worker_cfg.get("definitions")
    if isinstance(definitions, dict):
        cleaned: Dict[str, dict[str, Any]] = {}
        for name, definition in definitions.items():
            if not isinstance(definition, MutableMapping):
                continue
            entry = dict(definition)
            entry["symbols"] = _normalise_symbol_sequence(entry.get("symbols"))
            cleaned[name] = entry
        worker_cfg["definitions"] = cleaned
    normalised["workers"] = worker_cfg

    researcher_cfg = dict(normalised.get("researcher", {}))
    if researcher_cfg:
        researcher_cfg["symbols"] = _normalise_symbol_sequence(researcher_cfg.get("symbols"))
        normalised["researcher"] = researcher_cfg

    return normalised


__all__ = ["normalize_config", "read_config_file", "normalize_symbol"]
