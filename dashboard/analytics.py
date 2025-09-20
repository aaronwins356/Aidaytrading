"""Portfolio analytics utilities for the trading dashboard.

The functions in this module are intentionally pure/side-effect free so they can
be unit tested and reused by the Streamlit front-end. All functions accept
pandas DataFrames and return plain Python types.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf

__all__ = [
    "ensure_datetime_index",
    "max_drawdown",
    "drawdown_series",
    "sharpe_ratio",
    "sortino_ratio",
    "profit_factor",
    "expectancy",
    "hit_rate",
    "calc_var_cvar",
    "kelly_fraction",
    "aggregate_trade_kpis",
    "attribution_by",
    "correlation_matrix",
    "lead_lag_correlations",
    "rolling_sharpe",
    "simulate_what_if",
]


def ensure_datetime_index(df: pd.DataFrame, column: str = "ts") -> pd.DataFrame:
    """Return a copy with a datetime index."""

    if df.empty:
        return df.copy()
    out = df.copy()
    out[column] = pd.to_datetime(out[column])
    out.set_index(column, inplace=True)
    out.sort_index(inplace=True)
    return out


def max_drawdown(equity: pd.Series) -> Tuple[float, pd.Series]:
    """Compute the maximum drawdown and drawdown series."""

    if equity.empty:
        return 0.0, pd.Series(dtype=float)
    cummax = equity.cummax()
    dd = equity / cummax - 1.0
    return float(dd.min()), dd


def drawdown_series(equity_df: pd.DataFrame, balance_col: str = "balance") -> pd.DataFrame:
    if equity_df.empty:
        return equity_df
    s = equity_df[balance_col].astype(float)
    mdd, dd = max_drawdown(s)
    out = equity_df.copy()
    out["drawdown"] = dd
    out["rolling_max"] = s.cummax()
    out["max_drawdown"] = mdd
    return out


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    if returns.empty:
        return 0.0
    excess = returns - risk_free / periods_per_year
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / std)


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    if returns.empty:
        return 0.0
    downside = returns[returns < 0]
    if downside.empty:
        return np.inf
    downside_std = downside.std(ddof=1)
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    mean_excess = returns.mean() - risk_free / periods_per_year
    return float(np.sqrt(periods_per_year) * mean_excess / downside_std)


def profit_factor(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_loss = trades.loc[trades["pnl"] < 0, "pnl"].abs().sum()
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def expectancy(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    prob_win = len(wins) / len(trades)
    avg_win = wins["pnl"].mean() if not wins.empty else 0.0
    avg_loss = losses["pnl"].mean() if not losses.empty else 0.0
    return float(prob_win * avg_win + (1 - prob_win) * avg_loss)


def hit_rate(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    return float((trades["pnl"] > 0).mean())


def calc_var_cvar(returns: pd.Series, level: float = 0.95) -> Tuple[float, float]:
    if returns.empty:
        return 0.0, 0.0
    sorted_returns = returns.sort_values()
    var_idx = int((1 - level) * len(sorted_returns))
    var_idx = max(min(var_idx, len(sorted_returns) - 1), 0)
    var = sorted_returns.iloc[var_idx]
    cvar = sorted_returns.iloc[: var_idx + 1].mean()
    return float(var), float(cvar)


def kelly_fraction(win_prob: float, payoff_ratio: float) -> float:
    if payoff_ratio <= -1:
        return 0.0
    edge = win_prob * (payoff_ratio + 1) - 1
    denom = payoff_ratio
    if denom == 0:
        return 0.0
    frac = edge / denom
    return float(np.clip(frac, 0, 1))


def _daily_returns(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    df = trades.copy()
    df["closed_at"] = pd.to_datetime(df["closed_at"]).fillna(pd.to_datetime(df["opened_at"]))
    daily = df.groupby(df["closed_at"].dt.date)["pnl"].sum()
    return daily.div(daily.abs().mean() or 1.0)


def aggregate_trade_kpis(trades: pd.DataFrame, equity: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """Compute a suite of KPIs from a trades DataFrame."""

    metrics: Dict[str, float] = {k: 0.0 for k in [
        "net_pnl",
        "realized_pnl",
        "fees",
        "trades",
        "avg_trade",
        "median_trade",
        "profit_factor",
        "hit_rate",
        "sharpe",
        "sortino",
        "max_drawdown",
        "exposure",
        "payoff_ratio",
        "expectancy",
    ]}
    if trades.empty:
        return metrics

    trades = trades.copy()
    trades["pnl"] = trades["pnl"].astype(float)
    metrics["net_pnl"] = trades["pnl"].sum()
    metrics["realized_pnl"] = trades["pnl"].sum()
    metrics["fees"] = trades.get("fees", pd.Series(0.0, index=trades.index)).sum()
    metrics["trades"] = float(len(trades))
    metrics["avg_trade"] = trades["pnl"].mean()
    metrics["median_trade"] = trades["pnl"].median()
    metrics["profit_factor"] = profit_factor(trades)
    metrics["hit_rate"] = hit_rate(trades)
    avg_win = trades.loc[trades["pnl"] > 0, "pnl"].mean() or 0.0
    avg_loss = trades.loc[trades["pnl"] <= 0, "pnl"].mean() or 0.0
    metrics["payoff_ratio"] = float(abs(avg_win / avg_loss)) if avg_loss != 0 else np.inf
    metrics["expectancy"] = expectancy(trades)

    daily_returns = _daily_returns(trades)
    metrics["sharpe"] = sharpe_ratio(daily_returns)
    metrics["sortino"] = sortino_ratio(daily_returns)

    if equity is not None and not equity.empty:
        series = equity.set_index(pd.to_datetime(equity["ts"]))["balance"].astype(float)
        mdd, _ = max_drawdown(series)
        metrics["max_drawdown"] = abs(mdd)
        exposure = trades["qty"].abs().sum() / (len(trades) or 1)
        metrics["exposure"] = float(exposure)
    else:
        metrics["max_drawdown"] = 0.0
        metrics["exposure"] = float(trades["qty"].abs().mean() if "qty" in trades else 0.0)

    return metrics


def attribution_by(trades: pd.DataFrame, key: str) -> pd.DataFrame:
    if trades.empty or key not in trades:
        return pd.DataFrame(columns=[key, "pnl", "trades", "hit_rate"])
    grp = trades.groupby(key)
    out = grp["pnl"].agg(["sum", "count"])
    out.rename(columns={"sum": "pnl", "count": "trades"}, inplace=True)
    out["hit_rate"] = grp.apply(lambda x: (x["pnl"] > 0).mean())
    return out.reset_index()


def correlation_matrix(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    trades = trades.copy()
    trades["closed_at"] = pd.to_datetime(trades["closed_at"]).fillna(pd.to_datetime(trades["opened_at"]))
    pivot = trades.pivot_table(index=trades["closed_at"].dt.date, columns="worker", values="pnl", aggfunc="sum")
    return pivot.corr()


def lead_lag_correlations(trades: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    corr_rows = []
    workers = trades["worker"].dropna().unique().tolist()
    for worker in workers:
        series = trades.loc[trades["worker"] == worker]
        series = series.set_index(pd.to_datetime(series["closed_at"]))["pnl"].resample("1D").sum().fillna(0)
        acf_values = acf(series, nlags=max_lag, fft=False)
        for lag, value in enumerate(acf_values[1:], start=1):
            corr_rows.append({"worker": worker, "lag": lag, "autocorr": float(value)})
    return pd.DataFrame(corr_rows)


def rolling_sharpe(trades: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["ts", "worker", "sharpe"])
    trades = trades.copy()
    trades["closed_at"] = pd.to_datetime(trades["closed_at"]).fillna(pd.to_datetime(trades["opened_at"]))
    rows = []
    for worker, df in trades.groupby("worker"):
        daily = df.resample("1D", on="closed_at")["pnl"].sum().fillna(0)
        rolling = daily.rolling(window=window).apply(lambda x: sharpe_ratio(pd.Series(x)), raw=False)
        rows.append(pd.DataFrame({"ts": rolling.index, "worker": worker, "sharpe": rolling.values}))
    return pd.concat(rows, ignore_index=True)


def simulate_what_if(
    trades: pd.DataFrame,
    stop_multiplier: float = 1.0,
    take_profit_multiplier: float = 1.0,
    size_pct: float = 1.0,
    risk_cap: float = 1.0,
) -> Dict[str, float]:
    if trades.empty:
        return {"net_pnl": 0.0, "sharpe": 0.0, "sortino": 0.0, "hit_rate": 0.0}

    trades = trades.copy()
    trades["pnl"] = trades["pnl"].astype(float)
    pnl = trades["pnl"] * size_pct * risk_cap
    pnl *= np.where(trades["pnl"] > 0, take_profit_multiplier, stop_multiplier)
    simulated = trades.copy()
    simulated["pnl"] = pnl
    daily = _daily_returns(simulated)
    return {
        "net_pnl": float(simulated["pnl"].sum()),
        "sharpe": float(sharpe_ratio(daily)),
        "sortino": float(sortino_ratio(daily)),
        "hit_rate": float(hit_rate(simulated)),
    }


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod()
