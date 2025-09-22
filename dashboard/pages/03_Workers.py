"""Worker performance deep-dive."""
from __future__ import annotations

from typing import Dict

import pandas as pd
import streamlit as st

from analytics import aggregate_trade_kpis, correlation_matrix, rolling_sharpe
from components import download_chart_as_png, equity_with_drawdown, risk_return_scatter, rolling_sharpe_chart
from data_io import save_config
from ._shared import ensure_data_sources

st.set_page_config(page_title="Workers · Aurora Desk", page_icon="🛠")

st.title("Workers")
config = st.session_state.get("config")
if config is None:
    st.warning("Configuration not loaded. Please restart the application.")
    st.stop()

trades = ensure_data_sources().get("trades", pd.DataFrame())
if trades.empty:
    st.info("No trade history. Seed demo data or wait for executions.")
    st.stop()

if "worker" not in trades:
    st.warning("Trades dataset missing worker attribution.")
    st.stop()

worker_groups = trades.groupby("worker")
rows = []
for worker, df in worker_groups:
    metrics = aggregate_trade_kpis(df, None)
    avg_hold = (
        (pd.to_datetime(df.get("closed_at")) - pd.to_datetime(df.get("opened_at")))
        .dt.total_seconds()
        .mean()
        / 3600
        if not df.empty
        else 0
    )
    rows.append(
        {
            "worker": worker,
            "pnl": metrics.get("net_pnl", 0.0),
            "sharpe": metrics.get("sharpe", 0.0),
            "drawdown": metrics.get("max_drawdown", 0.0),
            "hit_rate": metrics.get("hit_rate", 0.0),
            "avg_hold_h": avg_hold,
            "expectancy": metrics.get("expectancy", 0.0),
            "trades": len(df),
            "slippage": df.get("slippage", pd.Series(dtype=float)).mean() if "slippage" in df else 0.0,
            "fees": df.get("fees", pd.Series(dtype=float)).sum() if "fees" in df else 0.0,
            "symbol": df["symbol"].mode().iloc[0] if "symbol" in df and not df["symbol"].empty else "",
        }
    )

summary_df = pd.DataFrame(rows)
if summary_df.empty:
    st.info("No worker aggregates available yet.")
    st.stop()

st.dataframe(summary_df)

risk_return = summary_df.rename(columns={"pnl": "return", "drawdown": "risk"})
st.plotly_chart(risk_return_scatter(risk_return), use_container_width=True)

def _safe_corr(df: pd.DataFrame) -> pd.DataFrame:
    corr = correlation_matrix(df)
    return corr.fillna(0.0)

corr = _safe_corr(trades)
if not corr.empty:
    st.subheader("Worker correlations")
    st.dataframe(corr)

rolling = rolling_sharpe(trades)
st.plotly_chart(rolling_sharpe_chart(rolling), use_container_width=True)

selected_worker = st.selectbox("Select worker", options=summary_df["worker"].tolist())
worker_trades = worker_groups.get_group(selected_worker)

st.subheader(f"{selected_worker} drill-down")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("PnL", f"{worker_trades['pnl'].sum():,.2f}")
with col2:
    st.metric("Trades", len(worker_trades))
with col3:
    wins = worker_trades["pnl"] > 0
    win_streak = wins.groupby((wins != wins.shift()).cumsum()).cumsum().max()
    losses = worker_trades["pnl"] <= 0
    loss_streak = losses.groupby((losses != losses.shift()).cumsum()).cumsum().max()
    st.metric("Win streak", int(win_streak or 0))
    st.metric("Loss streak", int(loss_streak or 0))

st.subheader("Worker equity curve")
worker_equity = worker_trades.sort_values("closed_at").copy()
worker_equity["balance"] = worker_equity["pnl"].cumsum()
worker_equity.rename(columns={"closed_at": "ts"}, inplace=True)
fig = equity_with_drawdown(worker_equity[["ts", "balance"]])
st.plotly_chart(fig, use_container_width=True)
download_chart_as_png(fig, f"equity_{selected_worker}")


def _worker_form(worker_key: str) -> Dict:
    worker_conf = config.workers.get(worker_key)
    enabled = st.checkbox("Enabled", value=worker_conf.enabled, key=f"enabled_{worker_key}")
    max_pos = st.number_input(
        "Max position",
        value=float(worker_conf.max_position),
        min_value=0.0,
        step=0.1,
        key=f"maxpos_{worker_key}",
    )
    risk = st.number_input(
        "Risk per trade",
        value=float(worker_conf.risk_per_trade),
        min_value=0.0,
        max_value=0.2,
        step=0.005,
        key=f"risk_{worker_key}",
    )
    symbols = st.multiselect("Symbols", config.symbols, default=worker_conf.symbols, key=f"symbols_{worker_key}")
    params = {}
    for key, val in worker_conf.parameters.items():
        params[key] = st.text_input(
            f"Param {key}",
            value=str(val),
            key=f"param_{worker_key}_{key}",
            help="Strings only. Strategy engine will coerce types.",
        )
    return {"enabled": enabled, "max_position": max_pos, "risk_per_trade": risk, "symbols": symbols, "parameters": params}


st.subheader("Configuration controls")
with st.form("worker_config"):
    updates = {}
    for worker_name in summary_df["worker"].tolist():
        st.markdown(f"### {worker_name}")
        updates[worker_name] = _worker_form(worker_name)
    submitted = st.form_submit_button("Save worker config", use_container_width=True)

if submitted:
    new_config = st.session_state["config"].dict()
    new_config["workers"].update(updates)
    success, message = save_config(new_config)
    if success:
        st.success(message)
        st.session_state["config"] = st.session_state["config"].__class__(**new_config)
        st.session_state["config_raw"] = new_config
    else:
        st.error(message)

st.download_button("Download worker summary CSV", summary_df.to_csv(index=False), file_name="worker_summary.csv", mime="text/csv")
