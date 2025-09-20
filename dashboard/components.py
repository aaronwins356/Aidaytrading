"""Reusable visual components for the Streamlit dashboard."""
from __future__ import annotations

import io
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

try:  # pragma: no cover - only imported inside Streamlit
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore

PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "rgba(15,17,23,0)",
        "plot_bgcolor": "rgba(15,17,23,0.65)",
        "font": {"family": "Inter", "color": "#E5E7EB"},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        "margin": dict(l=40, r=30, t=60, b=40),
    }
}


def apply_template(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    fig.update_xaxes(gridcolor="rgba(148, 163, 184,0.2)")
    fig.update_yaxes(gridcolor="rgba(148, 163, 184,0.15)")
    return fig


def kpi_tile(
    label: str,
    value: float | str,
    delta: Optional[float] = None,
    help_text: Optional[str] = None,
    trend: Optional[pd.Series] = None,
) -> None:
    if st is None:
        return
    class_name = "positive" if isinstance(value, (int, float)) and value >= 0 else "negative"
    value_fmt = f"{value:,.2f}" if isinstance(value, (int, float)) else value
    spark_chart = None
    if trend is not None and not trend.empty:
        spark_chart = sparkline(trend)
    with st.container():
        st.markdown(
            f"<div class='kpi-tile'><h3>{label}</h3><div class='kpi-value {class_name}'>{value_fmt}</div></div>",
            unsafe_allow_html=True,
        )
        if delta is not None:
            st.caption(f"Δ {delta:,.2f}")
        if help_text:
            st.caption(help_text)
        if spark_chart is not None:
            st.plotly_chart(spark_chart, use_container_width=True)


def sparkline(series: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(y=series.values, mode="lines", fill="tozeroy", line=dict(color="#7C3AED"))
    )
    fig.update_layout(height=80, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
    return apply_template(fig)


def equity_with_drawdown(equity: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not equity.empty:
        fig.add_trace(
            go.Scatter(
                x=equity["ts"],
                y=equity["balance"],
                name="Balance",
                line=dict(color="#7C3AED", width=3),
            )
        )
        if "rolling_max" in equity:
            fig.add_trace(
                go.Scatter(
                    x=equity["ts"],
                    y=equity["rolling_max"],
                    name="Rolling High",
                    line=dict(color="rgba(139,92,246,0.4)", dash="dot"),
                )
            )
        if "drawdown" in equity:
            fig.add_trace(
                go.Bar(
                    x=equity["ts"],
                    y=equity["drawdown"],
                    name="Drawdown",
                    marker_color="rgba(239,68,68,0.5)",
                    yaxis="y2",
                )
            )
            fig.update_layout(yaxis2=dict(overlaying="y", side="right"))
    fig.update_layout(title="Equity & Drawdown")
    return apply_template(fig)


def pnl_histogram(trades: pd.DataFrame) -> go.Figure:
    if trades.empty:
        return apply_template(go.Figure())
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(x=trades["pnl"], nbinsx=40, name="PnL", marker_color="#7C3AED", opacity=0.7)
    )
    fig.update_layout(barmode="overlay", title="Distribution of Trade Returns")
    return apply_template(fig)


def pnl_heatmap_hod_dow(trades: pd.DataFrame) -> go.Figure:
    if trades.empty:
        return apply_template(go.Figure())
    trades = trades.copy()
    trades["opened_at"] = pd.to_datetime(trades["opened_at"])
    pivot = (
        trades.pivot_table(
            index=trades["opened_at"].dt.day_name(),
            columns=trades["opened_at"].dt.hour,
            values="pnl",
            aggfunc="sum",
        )
        .fillna(0)
        .reindex(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            axis=0,
            fill_value=0,
        )
    )
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale="Viridis",
            colorbar=dict(title="PnL"),
        )
    )
    fig.update_layout(title="PnL by Day of Week / Hour of Day")
    return apply_template(fig)


def candles_with_trades(market: pd.DataFrame, trades: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not market.empty:
        fig.add_trace(
            go.Candlestick(
                x=market["ts"],
                open=market["open"],
                high=market["high"],
                low=market["low"],
                close=market["close"],
                name="Price",
            )
        )
        for length, color in zip([9, 21, 50, 200], ["#8B5CF6", "#F59E0B", "#10B981", "#6366F1"]):
            ma = market["close"].rolling(length, min_periods=1).mean()
            fig.add_trace(
                go.Scatter(x=market["ts"], y=ma, mode="lines", name=f"EMA {length}", line=dict(color=color))
            )
        if len(market) >= 20:
            bb_mid = market["close"].rolling(20, min_periods=20).mean()
            bb_std = market["close"].rolling(20, min_periods=20).std()
            fig.add_trace(
                go.Scatter(
                    x=market["ts"],
                    y=bb_mid + 2 * bb_std,
                    name="BB Upper",
                    line=dict(color="rgba(124,58,237,0.4)", dash="dot"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=market["ts"],
                    y=bb_mid - 2 * bb_std,
                    name="BB Lower",
                    line=dict(color="rgba(124,58,237,0.4)", dash="dot"),
                )
            )
    if not trades.empty and {"opened_at", "entry", "side"}.issubset(trades.columns):
        trades = trades.copy()
        trades["opened_at"] = pd.to_datetime(trades["opened_at"])
        buys = trades[trades["side"].str.upper() == "LONG"]
        sells = trades[trades["side"].str.upper() != "LONG"]
        fig.add_trace(
            go.Scatter(
                x=buys["opened_at"],
                y=buys["entry"],
                mode="markers",
                marker_symbol="triangle-up",
                marker_color="#10B981",
                marker_size=10,
                name="Buys",
                text=[
                    f"{row['worker']} size {row['qty']:.2f} PnL {row['pnl']:.2f}"
                    for _, row in buys.iterrows()
                ],
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sells["opened_at"],
                y=sells["entry"],
                mode="markers",
                marker_symbol="triangle-down",
                marker_color="#EF4444",
                marker_size=10,
                name="Sells",
                text=[
                    f"{row['worker']} size {row['qty']:.2f} PnL {row['pnl']:.2f}"
                    for _, row in sells.iterrows()
                ],
            )
        )
    fig.update_layout(title="Market Structure & Executions")
    return apply_template(fig)


def correlation_heatmap(matrix: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not matrix.empty:
        fig.add_trace(
            go.Heatmap(
                z=matrix.values,
                x=matrix.columns,
                y=matrix.index,
                colorscale="Magma",
                zmin=-1,
                zmax=1,
                colorbar=dict(title="ρ"),
            )
        )
    fig.update_layout(title="Worker Return Correlations")
    return apply_template(fig)


def risk_return_scatter(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return apply_template(go.Figure())
    fig = px.scatter(df, x="risk", y="return", size="trades", color="symbol", hover_name="worker")
    fig.update_layout(title="Risk vs Return by Worker")
    return apply_template(fig)


def rolling_sharpe_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not df.empty:
        for worker, group in df.groupby("worker"):
            fig.add_trace(go.Scatter(x=group["ts"], y=group["sharpe"], mode="lines", name=worker))
    fig.update_layout(title="Rolling Sharpe Ratios")
    return apply_template(fig)


def volume_profile(market: pd.DataFrame) -> go.Figure:
    if market.empty or "close" not in market:
        return apply_template(go.Figure())
    hist, bins = np.histogram(market["close"], bins=30, weights=market.get("volume"))
    fig = go.Figure(go.Bar(y=(bins[:-1] + bins[1:]) / 2, x=hist, orientation="h", marker_color="#7C3AED"))
    fig.update_layout(title="Volume Profile", yaxis=dict(title="Price"), xaxis=dict(title="Volume"))
    return apply_template(fig)


def download_chart_as_png(fig: go.Figure, label: str) -> None:
    if st is None:
        return
    buffer = io.BytesIO()
    file_name = f"{label}.png"
    mime = "image/png"
    try:
        import plotly.io as pio

        pio.write_image(fig, buffer, format="png")
        data = buffer.getvalue()
    except Exception:
        html_buffer = io.StringIO()
        fig.write_html(html_buffer)
        data = html_buffer.getvalue().encode("utf-8")
        file_name = f"{label}.html"
        mime = "text/html"
    st.download_button(f"Download {label}", data=data, file_name=file_name, mime=mime)
