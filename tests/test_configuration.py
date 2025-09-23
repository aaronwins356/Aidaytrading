"""Tests for configuration normalisation utilities."""

from __future__ import annotations

import logging

from ai_trader.services.configuration import normalize_config


def test_normalize_config_maps_deprecated_keys(caplog) -> None:
    """Deprecated ML keys should map to the canonical names with warnings."""

    caplog.set_level(logging.WARNING)
    raw_config = {"ml": {"learning_rate": 0.1, "ensemble_trees": 25}}
    normalised = normalize_config(raw_config)

    ml_cfg = normalised["ml"]
    assert ml_cfg["lr"] == 0.1
    assert ml_cfg["forest_size"] == 25
    assert "learning_rate" in caplog.text
