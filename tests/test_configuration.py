"""Tests for configuration normalisation utilities."""

from __future__ import annotations

import logging

from ai_trader.services.configuration import normalize_config


def test_normalize_config_maps_deprecated_keys(caplog) -> None:
    """Deprecated ML keys should map to the canonical names with warnings."""

    caplog.set_level(logging.WARNING)
    raw_config = {
        "ml": {"learning_rate": 0.1, "ensemble_trees": 25, "ensemble_enabled": True}
    }
    normalised = normalize_config(raw_config)

    ml_cfg = normalised["ml"]
    assert ml_cfg["lr"] == 0.1
    assert ml_cfg["forest_size"] == 25
    assert ml_cfg["ensemble"] is True
    assert ml_cfg["threshold"] == 0.25
    assert "learning_rate" in caplog.text
    assert "ensemble_trees" in caplog.text


def test_normalize_config_sanitises_symbols() -> None:
    """Trading symbols should be upper-cased, de-duplicated, and XBT normalised."""

    config = {
        "trading": {"symbols": ["btc/usd", "SOL/USD", "xbt/usd", "ETH/USD", "eth/usd"]}
    }

    normalised = normalize_config(config)

    assert normalised["trading"]["symbols"] == ["BTC/USD", "SOL/USD", "ETH/USD"]
