"""Utilities for loading and normalising bot configuration files."""

from __future__ import annotations

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
    "threshold_probability": "threshold",
}

_ML_DEFAULTS: Dict[str, float] = {
    "lr": 0.03,
    "regularization": 0.0005,
    "forest_size": 15,
    "threshold": 0.7,
}


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
    ml_cfg: Dict[str, Any] = dict(normalised.get("ml", {}))

    # Record which deprecated keys were encountered so we only warn once per key.
    for deprecated_key, canonical_key in _ML_KEY_ALIASES.items():
        if deprecated_key in ml_cfg and canonical_key not in ml_cfg:
            logger.warning(
                "Deprecated ML config key '%s' detected; prefer '%s' instead.",
                deprecated_key,
                canonical_key,
            )
            ml_cfg[canonical_key] = ml_cfg[deprecated_key]

    for key, default_value in _ML_DEFAULTS.items():
        ml_cfg.setdefault(key, default_value)

    normalised["ml"] = ml_cfg
    return normalised


__all__ = ["normalize_config", "read_config_file"]
