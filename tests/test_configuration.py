"""Tests for configuration normalisation utilities."""

from __future__ import annotations

import logging

import pytest

from ai_trader.services.configuration import normalize_config


def test_normalize_config_maps_deprecated_keys(caplog) -> None:
    """Deprecated ML keys should map to the canonical names with warnings."""

    caplog.set_level(logging.WARNING)
    raw_config = {"ml": {"learning_rate": 0.1, "ensemble_trees": 25, "ensemble_enabled": True}}
    normalised = normalize_config(raw_config)

    ml_cfg = normalised["ml"]
    assert ml_cfg["lr"] == 0.1
    assert ml_cfg["forest_size"] == 25
    assert ml_cfg["ensemble"] is True
    assert ml_cfg["threshold"] == 0.5
    assert "learning_rate" in caplog.text
    assert "ensemble_trees" in caplog.text


def test_normalize_config_sanitises_symbols() -> None:
    """Trading symbols should be upper-cased, de-duplicated, and XBT normalised."""

    config = {"trading": {"symbols": ["btc/usd", "SOL/USD", "xbt/usd", "ETH/USD", "eth/usd"]}}

    normalised = normalize_config(config)

    assert normalised["trading"]["symbols"] == ["BTC/USD", "SOL/USD", "ETH/USD"]
    assert normalised["trading"]["trade_confidence_min"] == 0.5
    assert normalised["trading"]["max_cash_per_trade"] == 0.0


def test_normalize_config_clamps_trade_size_band() -> None:
    config = {"trading": {"min_cash_per_trade": 2.5, "max_cash_per_trade": 50.0}}

    normalised = normalize_config(config)
    trading = normalised["trading"]

    assert trading["min_cash_per_trade"] == pytest.approx(2.5)
    assert trading["max_cash_per_trade"] == pytest.approx(50.0)


def test_normalize_config_sets_trading_mode_and_worker_symbols() -> None:
    raw_config = {
        "trading": {"paper_trading": False, "symbols": ["xbt/usd"]},
        "workers": {
            "definitions": {
                "alpha": {
                    "module": "ai_trader.workers.momentum.MomentumWorker",
                    "symbols": ["xbt/usd"],
                }
            }
        },
    }

    normalised = normalize_config(raw_config)
    trading_cfg = normalised["trading"]
    assert trading_cfg["paper_trading"] is False
    assert trading_cfg["mode"] == "live"
    assert trading_cfg["symbols"] == ["BTC/USD"]

    worker_symbols = normalised["workers"]["definitions"]["alpha"]["symbols"]
    assert worker_symbols == ["BTC/USD"]
