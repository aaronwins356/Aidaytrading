"""Main entry point for the Streamlit crypto trading dashboard."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from analytics import drawdown_series
from components import download_chart_as_png, equity_with_drawdown
from data_io import (
    CONFIG_PATH,
    DB_LIVE,
    DB_PAPER,
    load_config,
    load_equity,
    load_trades,
    seed_demo_data,
)

st.set_page_config(
    page_title="Crypto Desk Control Center",
    layout="wide",
    page_icon="💹",
    initial_sidebar_state="expanded",
)

PRIMARY_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h"]
DATE_PRESETS = ["1D", "7D", "30D", "YTD", "Custom"]


@st.cache_data
def _load_styles() -> str:
    style_path = Path(__file__).with_name("styles.css")
    return style_path.read_text(encoding="utf-8")


def inject_styles() -> None:
    st.markdown(f"<style>{_load_styles()}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def fetch_trades(mode: str) -> pd.DataFrame:
    frames = []
    if mode in ("Paper", "Both"):
        frames.append(load_trades(DB_PAPER))
    if mode in ("Live", "Both"):
        frames.append(load_trades(DB_LIVE))
    if not frames:
        return pd.DataFrame()
    trades = pd.concat(frames, ignore_index=True)
    trades.sort_values("opened_at", inplace=True)
    return trades


@st.cache_data(show_spinner=False)
def fetch_equity(mode: str) -> pd.DataFrame:
    frames = []
    if mode in ("Paper", "Both"):
        frames.append(load_equity(DB_PAPER))
    if mode in ("Live", "Both"):
        frames.append(load_equity(DB_LIVE))
    if not frames:
        return pd.DataFrame()
    equity = pd.concat(frames, ignore_index=True)
    equity.sort_values("ts", inplace=True)
    return equity


def init_state() -> None:
    if "filters" not in st.session_state:
        now = date.today()
        st.session_state["filters"] = {
            "date_preset": "30D",
            "date_range": (now - timedelta(days=30), now),
            "mode": "Paper",
            "symbols": PRIMARY_SYMBOLS,
            "timeframe": "1h",
            "workers": [],
        }
    if "config" not in st.session_state:
        config, raw = load_config(CONFIG_PATH)
        st.session_state["config"] = config
        st.session_state["config_raw"] = raw
    if "data_sources" not in st.session_state:
        st.session_state["data_sources"] = {}


def apply_filters(trades: pd.DataFrame) -> pd.DataFrame:
    filters = st.session_state["filters"]
    if trades.empty:
        return trades
    start, end = filters["date_range"]
    trades = trades.copy()
    trades["opened_at"] = pd.to_datetime(trades["opened_at"])
    trades = trades[(trades["opened_at"].dt.date >= start) & (trades["opened_at"].dt.date <= end)]
    trades = trades[trades["symbol"].isin(filters["symbols"])]
    if filters["workers"]:
        trades = trades[trades["worker"].isin(filters["workers"])]
    return trades


def _update_date_range_from_preset(preset: str) -> Tuple[date, date]:
    today = date.today()
    if preset == "1D":
        return today - timedelta(days=1), today
    if preset == "7D":
        return today - timedelta(days=7), today
    if preset == "30D":
        return today - timedelta(days=30), today
    if preset == "YTD":
        return date(today.year, 1, 1), today
    return st.session_state["filters"].get("date_range", (today - timedelta(days=30), today))


def top_bar() -> None:
    inject_styles()
    st.markdown(
        """
        <div class="top-bar glass-card">
            <div class="title">
                <h2 style="margin:0;">⚡️ Aurora Crypto Desk</h2>
                <div class="badge">Multi-venue · ML assisted · Real-time</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    filters = st.session_state["filters"]
    with st.container():
        col1, col2 = st.columns([2, 3])
        with col1:
            preset_index = DATE_PRESETS.index(filters.get("date_preset", "30D")) if filters.get("date_preset", "30D") in DATE_PRESETS else DATE_PRESETS.index("30D")
            preset = st.selectbox("Date preset", DATE_PRESETS, index=preset_index)
            filters["date_preset"] = preset
            if preset != "Custom":
                filters["date_range"] = _update_date_range_from_preset(preset)
        with col2:
            filters["date_range"] = st.date_input("Date range", value=filters["date_range"], max_value=date.today())
            if isinstance(filters["date_range"], date):
                filters["date_range"] = (filters["date_range"], filters["date_range"])
        c1, c2, c3 = st.columns(3)
        with c1:
            filters["mode"] = st.selectbox("Mode", ["Paper", "Live", "Both"], index=["Paper", "Live", "Both"].index(filters["mode"]))
        with c2:
            filters["symbols"] = st.multiselect("Symbols", PRIMARY_SYMBOLS, default=filters.get("symbols", PRIMARY_SYMBOLS))
        with c3:
            filters["timeframe"] = st.selectbox("Timeframe", TIMEFRAMES, index=TIMEFRAMES.index(filters.get("timeframe", "1h")))
        worker_options = sorted(st.session_state["config"].workers.keys())
        filters["workers"] = st.multiselect("Workers", worker_options, default=filters.get("workers", []))
    st.session_state["filters"] = filters
    st.session_state.setdefault("keyboard_help", False)
    st.markdown(
        """
        <script>
        document.addEventListener('keydown', function(e){
            if(e.key === 'r' && !e.metaKey){window.dispatchEvent(new Event('streamlit:rerun'));}
            if(e.key === '?'){window.alert('Keyboard: r refresh · ? help');}
        });
        </script>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def get_demo_market(symbol: str, start: date, end: date, timeframe: str) -> pd.DataFrame:
    freq_map = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1H", "4h": "4H"}
    rng = pd.date_range(start=start, end=end, freq=freq_map.get(timeframe, "1H"))
    drift = np.linspace(0, 1, len(rng)) * 100
    noise = np.cumsum(np.random.normal(scale=20, size=len(rng)))
    base = 30_000 + drift + noise
    df = pd.DataFrame(
        {
            "ts": rng,
            "open": base,
            "high": base + np.abs(np.random.normal(40, 15, len(rng))),
            "low": base - np.abs(np.random.normal(40, 15, len(rng))),
            "close": base + np.random.normal(0, 15, len(rng)),
            "volume": np.abs(np.random.normal(1000, 250, len(rng))),
        }
    )
    return df


def refresh_data() -> None:
    filters = st.session_state["filters"]
    trades = fetch_trades(filters["mode"])
    st.session_state["data_sources"]["trades_raw"] = trades
    st.session_state["data_sources"]["trades"] = apply_filters(trades)
    equity = fetch_equity(filters["mode"])
    st.session_state["data_sources"]["equity_raw"] = equity
    st.session_state["data_sources"]["equity"] = drawdown_series(equity) if not equity.empty else equity


@st.cache_data(show_spinner=False)
def _generate_help() -> str:
    return json.dumps({"shortcuts": {"r": "Refresh data", "?": "Show help dialog"}}, indent=2)


def render_footer() -> None:
    st.markdown(
        """
        <div style="text-align:center; margin-top:3rem; opacity:0.6;">
            Built with ❤️ for high-velocity crypto teams · Keyboard: <span class='key'>r</span> refresh · <span class='key'>?</span> help
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    seed_demo_data()
    init_state()
    top_bar()
    refresh_data()
    equity = st.session_state["data_sources"].get("equity", pd.DataFrame())
    if not equity.empty:
        fig = equity_with_drawdown(equity)
        st.plotly_chart(fig, use_container_width=True)
        download_chart_as_png(fig, "equity_curve")
    else:
        st.info("No equity data available yet. Seed demo data via Settings if required.")

    with st.expander("Keyboard shortcuts & help", expanded=False):
        st.json(json.loads(_generate_help()))

    render_footer()


if __name__ == "__main__":
    main()
