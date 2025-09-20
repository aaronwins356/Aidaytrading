from __future__ import annotations

import pandas as pd
import pytest

from dashboard import analytics


def sample_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_id": ["t1", "t2", "t3"],
            "opened_at": pd.date_range("2024-01-01", periods=3, freq="D"),
            "closed_at": pd.date_range("2024-01-02", periods=3, freq="D"),
            "symbol": ["BTC/USDT", "ETH/USDT", "BTC/USDT"],
            "worker": ["alpha", "beta", "alpha"],
            "qty": [1.0, 2.0, 1.5],
            "pnl": [100.0, -50.0, 200.0],
            "fees": [1.0, 1.5, 2.0],
        }
    )


def test_drawdown_series_handles_empty() -> None:
    empty = pd.DataFrame()
    result = analytics.drawdown_series(empty)
    assert result.empty


def test_aggregate_trade_kpis_basic() -> None:
    trades = sample_trades()
    metrics = analytics.aggregate_trade_kpis(trades, None)
    assert metrics["net_pnl"] == pytest.approx(250.0)
    assert metrics["profit_factor"] > 0
    assert metrics["hit_rate"] > 0


def test_correlation_matrix_not_empty() -> None:
    trades = sample_trades()
    corr = analytics.correlation_matrix(trades)
    assert "alpha" in corr.columns


def test_simulate_what_if_changes_returns() -> None:
    trades = sample_trades()
    result = analytics.simulate_what_if(trades, stop_multiplier=0.5, take_profit_multiplier=1.5, size_pct=0.5, risk_cap=0.8)
    assert result["net_pnl"] != 0
