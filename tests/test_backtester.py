from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict

import math
import pandas as pd
import pytest

from ai_trader.backtester import Backtester
from ai_trader.main import parse_args, prepare_config, run_backtest_cli


@pytest.fixture()
def sample_config() -> Dict[str, object]:
    return {
        "trading": {
            "symbols": ["BTC/USDT"],
            "paper_trading": True,
            "paper_starting_equity": 10000.0,
            "equity_allocation_percent": 10.0,
            "max_open_positions": 2,
            "min_cash_per_trade": 10.0,
            "max_cash_per_trade": 1000.0,
        },
        "risk": {
            "risk_per_trade": 0.1,
            "max_drawdown_percent": 90.0,
            "daily_loss_limit_percent": 90.0,
            "max_open_positions": 5,
            "min_trades_per_day": {"default": 1},
            "confidence_relax_percent": 0.5,
            "min_stop_buffer": 0.001,
        },
        "workers": {
            "definitions": {
                "momentum": {
                    "module": "ai_trader.workers.momentum.MomentumWorker",
                    "enabled": True,
                    "symbols": ["BTC/USDT"],
                    "parameters": {
                        "fast_window": 2,
                        "slow_window": 3,
                        "warmup_candles": 3,
                    },
                }
            }
        },
    }


def _sample_candles(start: datetime) -> list[dict[str, float | datetime]]:
    closes = [100.0, 101.0, 102.0, 103.0, 99.0, 98.0]
    candles = []
    for idx, close in enumerate(closes):
        timestamp = start + timedelta(minutes=idx)
        candles.append(
            {
                "timestamp": timestamp,
                "open": close - 0.5,
                "high": close + 0.5,
                "low": close - 1.0,
                "close": close,
                "volume": 10.0 + idx,
            }
        )
    return candles


def test_backtester_runs_on_sample_data(sample_config, tmp_path):
    start = datetime(2022, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(minutes=5)
    candles = _sample_candles(start)
    backtester = Backtester(
        sample_config,
        "BTC/USDT",
        start,
        end,
        candles=candles,
        fee_rate=0.0,
        slippage_bps=0.0,
        reports_dir=tmp_path,
        timeframe="1m",
    )
    result = asyncio.run(backtester.run())
    assert len(result.equity_curve) == len(candles)
    assert math.isclose(result.metrics["final_equity"], 10000.0, rel_tol=1e-6)
    assert math.isclose(result.metrics["net_profit"], 0.0, abs_tol=1e-6)
    assert result.report_paths["summary_json"].exists()


def test_backtester_metrics_are_finite(sample_config, tmp_path):
    start = datetime(2022, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(minutes=5)
    backtester = Backtester(
        sample_config,
        "BTC/USDT",
        start,
        end,
        candles=_sample_candles(start),
        fee_rate=0.0,
        slippage_bps=0.0,
        reports_dir=tmp_path,
        timeframe="1m",
    )
    result = asyncio.run(backtester.run())
    for value in result.metrics.values():
        assert not math.isnan(float(value))
    assert math.isclose(result.metrics["final_equity"], result.equity_curve[-1]["equity"])


def test_cli_backtest_and_live_modes(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
trading:
  symbols: [BTC/USDT]
  paper_trading: true
  paper_starting_equity: 10000
risk:
  risk_per_trade: 0.1
workers:
  definitions:
    momentum:
      module: ai_trader.workers.momentum.MomentumWorker
      enabled: true
      symbols: [BTC/USDT]
      parameters:
        fast_window: 2
        slow_window: 3
""",
        encoding="utf-8",
    )

    csv_path = tmp_path / "ohlcv.csv"
    start = datetime(2022, 1, 1, tzinfo=timezone.utc)
    candles = _sample_candles(start)
    frame = pd.DataFrame(candles)
    frame.to_csv(csv_path, index=False)

    args = parse_args(
        [
            "--mode",
            "backtest",
            "--config",
            str(config_path),
            "--pair",
            "BTC/USDT",
            "--start",
            "2022-01-01",
            "--end",
            "2022-01-02",
            "--backtest-csv",
            str(csv_path),
            "--reports-dir",
            str(tmp_path),
            "--backtest-fee",
            "0.0",
            "--backtest-slippage-bps",
            "0.0",
        ]
    )
    config = prepare_config(args)
    result = run_backtest_cli(args, config)
    assert result.report_paths["summary_json"].exists()
    assert math.isclose(result.metrics["final_equity"], result.equity_curve[-1]["equity"])

    live_called = {}

    async def fake_start(args, config):  # pragma: no cover - invoked in test
        live_called["invoked"] = True

    monkeypatch.setattr("ai_trader.main.start_trading", fake_start)
    monkeypatch.setattr("ai_trader.main._spawn_parallel_backtest", lambda *a, **k: None)
    monkeypatch.setattr("ai_trader.main.configure_logging", lambda: None)

    from ai_trader import main as main_module

    main_module.main(["--mode", "live", "--config", str(config_path)])
    assert live_called["invoked"]
