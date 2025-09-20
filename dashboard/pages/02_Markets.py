from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import candles_with_trades, download_chart_as_png, volume_profile


@st.cache_data(show_spinner=False)
def _get_market_data(symbol: str, start: pd.Timestamp, end: pd.Timestamp, timeframe: str) -> pd.DataFrame:
    freq_map = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1H", "4h": "4H"}
    rng = pd.date_range(start=start, end=end, freq=freq_map.get(timeframe, "1H"))
    if rng.empty:
        return pd.DataFrame()
    drift = np.linspace(0, 1, len(rng)) * 50
    base = 30_000 + drift + np.cumsum(np.random.normal(0, 30, size=len(rng)))
    df = pd.DataFrame(
        {
            "ts": rng,
            "open": base + np.random.normal(0, 10, len(rng)),
            "high": base + np.abs(np.random.normal(40, 20, len(rng))),
            "low": base - np.abs(np.random.normal(40, 20, len(rng))),
            "close": base + np.random.normal(0, 12, len(rng)),
            "volume": np.abs(np.random.normal(1000, 320, len(rng))),
        }
    )
    return df


def _orderbook_stub(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["price", "bid", "ask"])
    last_price = df["close"].iloc[-1]
    levels = np.linspace(last_price * 0.99, last_price * 1.01, 10)
    bids = np.abs(np.random.normal(50, 10, len(levels)))
    asks = np.abs(np.random.normal(50, 10, len(levels)))
    return pd.DataFrame({"price": levels, "bid": bids, "ask": asks})


def _imbalance(orderbook: pd.DataFrame) -> float:
    if orderbook.empty:
        return 0.0
    denom = orderbook["bid"].sum() + orderbook["ask"].sum()
    if denom == 0:
        return 0.0
    return float((orderbook["bid"].sum() - orderbook["ask"].sum()) / denom)


st.set_page_config(page_title="Markets · Aurora Desk", page_icon="📈")

st.title("Markets")
filters = st.session_state.get("filters", {})
trades = st.session_state.get("data_sources", {}).get("trades", pd.DataFrame())
if trades.empty:
    st.info("No trades available; displaying synthetic market data.")

symbol = st.selectbox("Symbol", options=filters.get("symbols", ["BTC/USD"]))
timeframe = st.selectbox("Timeframe", options=["1m", "5m", "15m", "1h", "4h"], index=3)
start, end = filters.get(
    "date_range",
    (
        (datetime.utcnow() - pd.Timedelta(days=7)).date(),
        datetime.utcnow().date(),
    ),
)
market_df = _get_market_data(symbol, pd.to_datetime(start), pd.to_datetime(end), timeframe)

overlay_toggle = st.multiselect("Overlays", ["EMA", "Bollinger", "ATR"], default=["EMA", "Bollinger"])
bb_length = st.slider("Bollinger length", 10, 50, 20)
bb_dev = st.slider("Bollinger deviation", 1.0, 3.0, 2.0)
atr_length = st.slider("ATR length", 5, 30, 14)
atr_mult = st.slider("ATR multiplier", 1.0, 4.0, 2.0)

chart_trades = trades[trades.get("symbol") == symbol] if not trades.empty and "symbol" in trades else pd.DataFrame()
chart = candles_with_trades(market_df, chart_trades)
chart.update_layout(title=f"{symbol} Market Overview")

if "ATR" in overlay_toggle and not market_df.empty:
    tr1 = market_df["high"] - market_df["low"]
    tr2 = (market_df["high"] - market_df["close"].shift()).abs()
    tr3 = (market_df["low"] - market_df["close"].shift()).abs()
    atr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(atr_length).mean()
    chart.add_trace(
        go.Scatter(
            x=market_df["ts"],
            y=market_df["close"] + atr * atr_mult,
            name="ATR Upper",
            line=dict(color="#22d3ee", dash="dot"),
        )
    )
    chart.add_trace(
        go.Scatter(
            x=market_df["ts"],
            y=market_df["close"] - atr * atr_mult,
            name="ATR Lower",
            line=dict(color="#22d3ee", dash="dot"),
        )
    )

if "Bollinger" in overlay_toggle and not market_df.empty:
    mid = market_df["close"].rolling(bb_length).mean()
    std = market_df["close"].rolling(bb_length).std()
    chart.add_trace(
        go.Scatter(
            x=market_df["ts"],
            y=mid + bb_dev * std,
            name="BB Upper",
            line=dict(color="#f97316", dash="dot"),
        )
    )
    chart.add_trace(
        go.Scatter(
            x=market_df["ts"],
            y=mid - bb_dev * std,
            name="BB Lower",
            line=dict(color="#f97316", dash="dot"),
        )
    )

st.plotly_chart(chart, use_container_width=True)
download_chart_as_png(chart, f"market_{symbol}")

ob = _orderbook_stub(market_df)
imb = _imbalance(ob)
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("Synthetic order book")
    st.dataframe(ob)
with col2:
    st.subheader("Imbalance")
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=imb * 100,
            gauge={"axis": {"range": [-100, 100]}, "bar": {"color": "#7C3AED"}},
            title={"text": "Bid-Ask %"},
        )
    )
    st.plotly_chart(gauge, use_container_width=True)

st.subheader("Volume profile")
st.plotly_chart(volume_profile(market_df), use_container_width=True)

if not chart_trades.empty:
    st.subheader("Executed trades")
    cols = [c for c in ["trade_id", "opened_at", "side", "qty", "entry", "exit", "pnl", "worker"] if c in chart_trades]
    st.dataframe(chart_trades[cols].sort_values("opened_at", ascending=False))
    st.download_button(
        "Export trades CSV",
        data=chart_trades[cols].to_csv(index=False),
        mime="text/csv",
        file_name=f"{symbol}_trades.csv",
    )
else:
    st.caption("No executions for the selected symbol in current filters.")
