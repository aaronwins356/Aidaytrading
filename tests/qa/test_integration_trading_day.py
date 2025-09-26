"""Integration test simulating a trading day."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from ai_trader.backtester import Backtester


def _load_subset(csv_path: Path, start: datetime, periods: int = 24) -> list[dict[str, float]]:
    frame = pd.read_csv(csv_path).head(periods)
    candles: list[dict[str, float]] = []
    current = start
    for _, row in frame.iterrows():
        candles.append(
            {
                "timestamp": current,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
        )
        current += timedelta(hours=1)
    return candles


def test_trading_day_backtest(tmp_path: Path) -> None:
    start = datetime(2022, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(hours=23)
    csv_path = Path("tests/regression/data/btcusdt_2022-01-01_2022-02-28.csv")
    candles = _load_subset(csv_path, start, periods=48)
    config = {
        "trading": {
            "symbols": ["BTC/USDT"],
            "paper_trading": True,
            "paper_starting_equity": 10000.0,
            "equity_allocation_percent": 10.0,
            "max_open_positions": 2,
            "min_cash_per_trade": 10.0,
            "max_cash_per_trade": 500.0,
        },
        "risk": {
            "risk_per_trade": 0.05,
            "max_drawdown_percent": 20.0,
            "daily_loss_limit_percent": 10.0,
            "max_open_positions": 3,
            "confidence_relax_percent": 0.2,
            "min_trades_per_day": 1,
            "min_stop_buffer": 0.001,
        },
        "workers": {
            "definitions": {
                "momentum": {
                    "module": "ai_trader.workers.momentum.MomentumWorker",
                    "enabled": True,
                    "symbols": ["BTC/USDT"],
                    "parameters": {"fast_window": 5, "slow_window": 12},
                },
                "mean_reversion": {
                    "module": "ai_trader.workers.mean_reversion.MeanReversionWorker",
                    "enabled": True,
                    "symbols": ["BTC/USDT"],
                    "parameters": {"window": 20, "threshold": 0.01},
                },
            }
        },
    }
    tester = Backtester(
        config,
        "BTC/USDT",
        start,
        end,
        candles=candles,
        timeframe="1h",
        fee_rate=0.0,
        slippage_bps=0.0,
        reports_dir=tmp_path,
        label="qa_smoke",
    )
    result = asyncio.run(tester.run())
    expected_points = sum(1 for candle in candles if start <= candle["timestamp"] <= end)
    assert len(result.equity_curve) == expected_points
    assert result.metrics["net_profit"] is not None
    assert result.report_paths["summary_json"].exists()
    summary_json = result.report_paths["summary_json"].read_text(encoding="utf-8")
    assert "final_equity" in summary_json
