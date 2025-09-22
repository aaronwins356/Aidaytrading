"""Portfolio and risk diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from analytics import (
    aggregate_trade_kpis,
    attribution_by,
    calc_var_cvar,
    correlation_matrix,
    lead_lag_correlations,
    simulate_what_if,
)
from components import correlation_heatmap
from ._shared import ensure_data_sources

st.set_page_config(page_title="Portfolio Risk · Aurora Desk", page_icon="🛡")

st.title("Portfolio & Risk")
data_sources = ensure_data_sources()
trades = data_sources.get("trades", pd.DataFrame())
equity = data_sources.get("equity", pd.DataFrame())
if trades.empty:
    st.info("No trades to analyse. Seed demo data or adjust filters.")
    st.stop()

alloc_worker = attribution_by(trades, "worker")
alloc_symbol = attribution_by(trades, "symbol")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Allocation by worker")
    if not alloc_worker.empty:
        fig_worker = px.treemap(alloc_worker, path=["worker"], values="pnl", color="pnl", color_continuous_scale="Purples")
        st.plotly_chart(fig_worker, use_container_width=True)
    else:
        st.info("No worker attribution available.")
with col2:
    st.subheader("Allocation by symbol")
    if not alloc_symbol.empty:
        fig_symbol = px.pie(alloc_symbol, names="symbol", values="pnl", color="symbol")
        st.plotly_chart(fig_symbol, use_container_width=True)
    else:
        st.info("No symbol attribution available.")

st.subheader("Correlation matrix")
st.plotly_chart(correlation_heatmap(correlation_matrix(trades)), use_container_width=True)

st.subheader("Lead/Lag autocorrelation")
lead_df = lead_lag_correlations(trades)
if not lead_df.empty:
    st.dataframe(lead_df)
    st.download_button(
        "Export lead/lag CSV",
        data=lead_df.to_csv(index=False),
        file_name="lead_lag.csv",
        mime="text/csv",
    )
else:
    st.info("Insufficient data for lead/lag analysis.")

risk_metrics = aggregate_trade_kpis(trades, None)
returns = trades.groupby(trades["closed_at"].dt.date)["pnl"].sum() if "closed_at" in trades and "pnl" in trades else pd.Series(dtype=float)
var, cvar = calc_var_cvar(returns)
kelly = risk_metrics.get("payoff_ratio", 0.0)
metrics_cols = st.columns(3)
metrics_cols[0].metric("VaR 95%", f"{var:,.2f}")
metrics_cols[1].metric("CVaR 95%", f"{cvar:,.2f}")
metrics_cols[2].metric("Kelly fraction", f"{kelly:,.2f}")

st.subheader("Drawdown episodes")
if not equity.empty and "drawdown" in equity:
    st.line_chart(equity.set_index("ts")["drawdown"])
    equity = equity.copy()
    equity["episode"] = (equity["drawdown"].diff().fillna(0) >= 0).cumsum()
    dd_table = equity.groupby("episode").agg(start=("ts", "first"), end=("ts", "last"), depth=("drawdown", "min"))
    dd_table["recovery"] = dd_table["end"] - dd_table["start"]
    st.dataframe(dd_table.nsmallest(5, "depth"))
else:
    st.info("Drawdown data unavailable in current slice.")

st.subheader("What-if simulator")
with st.form("simulator"):
    stop = st.slider("Stop multiplier", 0.5, 1.5, 1.0, step=0.05)
    take = st.slider("Take-profit multiplier", 0.5, 2.0, 1.0, step=0.05)
    size = st.slider("Position size %", 0.25, 2.0, 1.0, step=0.05)
    risk_cap = st.slider("Risk cap", 0.5, 2.0, 1.0, step=0.1)
    run = st.form_submit_button("Run simulation")

if run:
    result = simulate_what_if(trades, stop_multiplier=stop, take_profit_multiplier=take, size_pct=size, risk_cap=risk_cap)
    st.json(result)

st.subheader("Exposure by symbol")
if "symbol" in trades and "qty" in trades:
    exposure = trades.groupby("symbol")["qty"].sum().reset_index()
    if not exposure.empty:
        st.bar_chart(exposure.set_index("symbol"))
        st.download_button(
            "Export exposure CSV",
            data=exposure.to_csv(index=False),
            file_name="exposure.csv",
            mime="text/csv",
        )
    else:
        st.info("No exposure data.")
else:
    st.info("Trades dataset missing symbol/qty fields.")
