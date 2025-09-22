"""Plotly figures used across the dashboard."""
from __future__ import annotations

from typing import Dict, Iterable, Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_dark"


def candlestick_chart(data: pd.DataFrame, *, title: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=data["time"],
                open=data["open"],
                high=data["high"],
                low=data["low"],
                close=data["close"],
                increasing_line_color="#34d399",
                decreasing_line_color="#f87171",
                name="Price",
            )
        ]
    )
    if "volume" in data:
        fig.add_trace(
            go.Bar(
                x=data["time"],
                y=data["volume"],
                name="Volume",
                marker_color="#38bdf8",
                opacity=0.3,
                yaxis="y2",
            )
        )
        fig.update_layout(
            yaxis=dict(title="Price"),
            yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
        )
    fig.update_layout(
        title=title,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_dark",
        height=420,
    )
    return fig


def sparkline_chart(values: Iterable[float]) -> go.Figure:
    values = list(values)
    fig = go.Figure(
        data=[
            go.Scatter(
                y=values,
                mode="lines",
                line=dict(color="#2dd4bf", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(45,212,191,0.2)",
            )
        ]
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=120,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def pie_chart(data: Dict[str, float], *, title: str) -> go.Figure:
    fig = go.Figure(
        go.Pie(
            labels=list(data.keys()),
            values=list(data.values()),
            hole=0.45,
            marker=dict(colors=["#38bdf8", "#34d399", "#f97316", "#a855f7", "#facc15"]),
        )
    )
    fig.update_layout(title=title, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def bar_chart(data: Dict[str, float], *, title: str) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=list(data.keys()),
            y=list(data.values()),
            marker_color="#60a5fa",
        )
    )
    fig.update_layout(title=title, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def gauge_chart(value: float, *, title: str, threshold: Optional[float] = None) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2dd4bf"},
                "steps": [
                    {"range": [0, 40], "color": "#1f2937"},
                    {"range": [40, 70], "color": "#334155"},
                    {"range": [70, 100], "color": "#0f172a"},
                ],
            },
        )
    )
    if threshold is not None:
        fig.data[0].gauge["threshold"] = {"line": {"color": "#f97316", "width": 4}, "thickness": 0.75, "value": threshold}
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def histogram(data: Iterable[float], *, title: str) -> go.Figure:
    fig = go.Figure(go.Histogram(x=list(data), marker_color="#f97316", opacity=0.85))
    fig.update_layout(title=title, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def equity_curve_chart(data: pd.DataFrame, *, title: str) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=data["timestamp"],
            y=data["equity"],
            mode="lines",
            line=dict(color="#2dd4bf", width=3),
            fill="tozeroy",
            fillcolor="rgba(45,212,191,0.15)",
        )
    )
    fig.update_layout(title=title, margin=dict(l=20, r=20, t=40, b=20))
    return fig
