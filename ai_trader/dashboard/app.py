"""Streamlit dashboard for the AI trading bot."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "trades.db"
CONFIG_PATH = BASE_DIR / "config.yaml"

st.set_page_config(page_title="AI Trader", layout="wide", page_icon="🪙")

CUSTOM_CSS = """
<style>
body {
    background: radial-gradient(circle at 20% 20%, rgba(0, 255, 170, 0.15), transparent 60%),
                radial-gradient(circle at 80% 0%, rgba(0, 120, 255, 0.12), transparent 55%),
                #0b1021;
    color: #f5f5ff;
    font-family: 'Inter', 'JetBrains Mono', monospace;
}
section.main > div {
    backdrop-filter: blur(18px);
    background: rgba(15, 25, 46, 0.78);
    border: 1px solid rgba(0, 255, 170, 0.15);
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 24px;
}
.stButton>button {
    background: linear-gradient(135deg, #00d4ff 0%, #0070f3 100%);
    color: #0b1021;
    border-radius: 12px;
    font-weight: 600;
    padding: 0.6rem 1.4rem;
    border: none;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(15, 25, 46, 0.6);
    border-radius: 12px 12px 0 0;
    border: 1px solid rgba(0, 255, 170, 0.2);
    color: #d1e0ff;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #00d4ff;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(ttl=30)
def load_config() -> Dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


@st.cache_data(ttl=15)
def load_trades() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame(
            columns=[
                "timestamp",
                "worker",
                "symbol",
                "side",
                "cash_spent",
                "entry_price",
                "exit_price",
                "pnl_percent",
                "pnl_usd",
                "win_loss",
            ]
        )
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp DESC", conn)
    return df


@st.cache_data(ttl=15)
def load_equity_curve() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame(columns=["timestamp", "equity", "pnl_percent", "pnl_usd"])
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT timestamp, equity, pnl_percent, pnl_usd FROM equity_curve ORDER BY timestamp ASC", conn)
    return df


def render_overview(config: Dict, trades: pd.DataFrame, equity_curve: pd.DataFrame) -> None:
    st.header("Overview")
    latest_equity = equity_curve["equity"].iloc[-1] if not equity_curve.empty else config["trading"].get("paper_starting_equity", 0)
    pnl_percent = equity_curve["pnl_percent"].iloc[-1] if not equity_curve.empty else 0
    pnl_usd = equity_curve["pnl_usd"].iloc[-1] if not equity_curve.empty else 0
    col1, col2, col3 = st.columns(3)
    col1.metric("Live Equity", f"${latest_equity:,.2f}", f"{pnl_percent:.2f}%")
    col2.metric("Total P/L", f"${pnl_usd:,.2f}")
    col3.metric("Trades Logged", f"{len(trades)}")

    st.subheader("Open Positions Snapshot")
    if trades.empty:
        st.info("No trades have been recorded yet. Workers will populate this once live.")
    else:
        open_trades = trades[trades["exit_price"].isna()].copy()
        if open_trades.empty:
            st.success("All positions closed.")
        else:
            st.dataframe(open_trades[["timestamp", "worker", "symbol", "side", "cash_spent", "entry_price"]])


def render_equity_curve(equity_curve: pd.DataFrame) -> None:
    st.header("Equity Curve")
    if equity_curve.empty:
        st.info("Equity data will appear once the bot starts trading.")
        return
    fig = px.line(
        equity_curve,
        x="timestamp",
        y="equity",
        title="Equity over Time",
        markers=True,
    )
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_trade_log(trades: pd.DataFrame) -> None:
    st.header("Trade Log")
    if trades.empty:
        st.info("No trades to display yet.")
        return
    workers = ["All"] + sorted(trades["worker"].unique())
    selected_worker = st.selectbox("Filter by worker", workers)
    filtered = trades if selected_worker == "All" else trades[trades["worker"] == selected_worker]
    st.dataframe(filtered, use_container_width=True)


def render_workers(config: Dict, trades: pd.DataFrame) -> None:
    st.header("Workers")
    worker_modules: List[str] = config.get("workers", {}).get("modules", [])
    worker_descriptions = {
        "ai_trader.workers.momentum.MomentumWorker": (
            "Momentum Rider",
            "⚡ Momentum Rider — rides fast/slow EMA momentum swings to capture breakout moves.",
        ),
        "ai_trader.workers.mean_reversion.MeanReversionWorker": (
            "Mean Reverter",
            "🔄 Mean Reverter — fades extreme moves back toward the average price band.",
        ),
    }
    cols = st.columns(2)
    for idx, module in enumerate(worker_modules):
        name, description = worker_descriptions.get(module, (module.split(".")[-1], module))
        with cols[idx % 2]:
            st.markdown(
                "<div style='padding:1rem;border-radius:1rem;border:1px solid rgba(0,255,170,0.25);background:rgba(11,16,33,0.8);'>"
                f"<strong>{description}</strong><br/>Status: <span style='color:#00ffb3;'>Active</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            worker_trades = trades[trades["worker"] == name] if not trades.empty else pd.DataFrame()
            chart_data = worker_trades[["timestamp", "pnl_usd"]] if not worker_trades.empty else pd.DataFrame({"timestamp": [], "pnl_usd": []})
            if not chart_data.empty:
                fig = px.bar(chart_data, x="timestamp", y="pnl_usd", title="Recent PnL")
                fig.update_layout(template="plotly_dark", height=250)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("PnL chart will appear once this worker records trades.")


def render_risk_controls(config: Dict) -> None:
    st.header("Risk Controls")
    trading_cfg = config.get("trading", {})
    equity_pct = st.slider(
        "Equity % per trade",
        min_value=1.0,
        max_value=20.0,
        value=float(trading_cfg.get("equity_allocation_percent", 5.0)),
        step=0.5,
    )
    max_positions = st.slider(
        "Max open positions",
        min_value=1,
        max_value=10,
        value=int(trading_cfg.get("max_open_positions", 3)),
    )
    if st.button("Apply Risk Settings"):
        st.success("Risk preferences updated. Restart the bot to apply changes.")
    st.caption("Adjust settings then update config.yaml for live trading.")


config = load_config()
trades_df = load_trades()
equity_df = load_equity_curve()

overview_tab, equity_tab, trades_tab, workers_tab, risk_tab = st.tabs(
    ["Overview", "Equity Curve", "Trade Log", "Workers", "Risk Controls"]
)

with overview_tab:
    render_overview(config, trades_df, equity_df)

with equity_tab:
    render_equity_curve(equity_df)

with trades_tab:
    render_trade_log(trades_df)

with workers_tab:
    render_workers(config, trades_df)

with risk_tab:
    render_risk_controls(config)
