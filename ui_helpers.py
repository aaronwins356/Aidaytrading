"""UI helper utilities for the Streamlit trading dashboard."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import streamlit as st


DARK_THEME_CSS = Path("dashboard/styles.css").read_text() if Path("dashboard/styles.css").exists() else ""


def initialize_session_state() -> None:
    """Ensure required Streamlit session state keys exist."""

    defaults: Mapping[str, object] = {
        "dark_mode": True,
        "trading_mode": "Paper",
        "risk_tolerance": 0.5,
        "max_drawdown_threshold": 0.2,
        "active_strategies": {"Momentum": True, "Mean Reversion": True, "Breakout": True, "Scalping": False},
        "equity_per_trade": 0.03,
        "stop_loss_default": 0.02,
        "take_profit_default": 0.05,
        "leverage": 1.0,
        "notifications": [],
        "last_refresh": 0.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def inject_custom_css(*, dark_mode: bool) -> None:
    """Inject custom CSS to style the dashboard."""

    base_css = """
    <style>
        :root {
            --accent-color: #2dd4bf;
            --accent-color-strong: #14b8a6;
            --card-bg-dark: rgba(17, 24, 39, 0.85);
            --card-bg-light: rgba(255, 255, 255, 0.85);
            --border-radius-large: 18px;
            --shadow-strong: 0 20px 45px rgba(15, 23, 42, 0.45);
            --shadow-soft: 0 12px 30px rgba(15, 23, 42, 0.35);
        }

        body {
            font-family: "Inter", "Roboto", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: radial-gradient(circle at top left, #0f172a, #020617);
        }

        .quant-card {
            border-radius: var(--border-radius-large);
            padding: 1.5rem;
            margin-bottom: 1.25rem;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            box-shadow: var(--shadow-soft);
            backdrop-filter: blur(12px);
        }

        .quant-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-strong);
        }

        .kpi-card h3 {
            font-size: 0.9rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            opacity: 0.7;
            margin-bottom: 0.35rem;
        }

        .kpi-value {
            font-size: 2.1rem;
            font-weight: 600;
        }

        .stMetric {
            background: transparent !important;
        }

        .css-ocqkz7, .css-1dp5vir, .css-1n76uvr, .css-1kyxreq {
            background: transparent !important;
        }

        .top-controls {
            display: flex;
            justify-content: flex-end;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .nav-tabs button {
            border-radius: 999px !important;
            padding: 0.75rem 1.5rem !important;
            font-weight: 600 !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.65rem;
            justify-content: center;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding-top: 0.6rem;
            padding-bottom: 0.6rem;
            font-weight: 600;
            background-color: rgba(148, 163, 184, 0.15);
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, var(--accent-color), var(--accent-color-strong));
            color: #0f172a !important;
            box-shadow: 0 10px 30px rgba(45, 212, 191, 0.35);
        }
    </style>
    """

    light_css = """
    <style>
        body {
            background: linear-gradient(180deg, #f5f7ff, #eaeef9);
            color: #0f172a;
        }
        .quant-card {
            background: var(--card-bg-light);
        }
    </style>
    """

    dark_css = """
    <style>
        body {
            color: #e2e8f0;
        }
        .quant-card {
            background: var(--card-bg-dark);
            border: 1px solid rgba(148, 163, 184, 0.08);
        }
    </style>
    """

    css = base_css + (dark_css if dark_mode else light_css) + DARK_THEME_CSS
    st.markdown(css, unsafe_allow_html=True)


def render_top_controls(*, dark_mode: bool) -> None:
    """Render the trading mode and theme toggles."""

    with st.container():
        cols = st.columns([0.6, 0.2, 0.2])
        with cols[1]:
            mode = st.selectbox(
                "Trading Mode",
                options=["Paper", "Live"],
                index=0 if st.session_state["trading_mode"] == "Paper" else 1,
                key="trading_mode_selector",
                help="Switch between paper trading and live execution modes.",
            )
            st.session_state["trading_mode"] = mode
        with cols[2]:
            dark = st.toggle("Dark Mode", value=dark_mode, key="dark_mode_toggle")
            st.session_state["dark_mode"] = dark


def kpi_card(title: str, value: str, *, delta: str | None = None, sparkline=None) -> None:
    """Render a KPI card with optional sparkline chart."""

    with st.container():
        card = st.container()
    card.markdown(
        f"""
        <div class="quant-card kpi-card">
            <h3>{title}</h3>
            <div class="kpi-value">{value}</div>
            {f'<div class="kpi-delta">{delta}</div>' if delta else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )
    if sparkline is not None:
        card.plotly_chart(sparkline, use_container_width=True, config={"displayModeBar": False})


def section_header(title: str, subtitle: str | None = None) -> None:
    """Render a stylised section header."""

    st.markdown(
        f"""
        <div style="display:flex;flex-direction:column;gap:0.25rem;margin:0.5rem 0 1rem 0;">
            <span style="letter-spacing:0.2em;text-transform:uppercase;font-size:0.75rem;opacity:0.6;">{subtitle or ''}</span>
            <h2 style="margin:0;font-weight:700;">{title}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )


def callout(message: str, *, variant: str = "info") -> None:
    """Render a callout banner used across tabs."""

    palette = {
        "info": ("#22d3ee", "#0f172a"),
        "warning": ("#f97316", "#111827"),
        "danger": ("#f43f5e", "#111827"),
        "success": ("#34d399", "#022c22"),
    }
    bg, fg = palette.get(variant, palette["info"])
    st.markdown(
        f"""
        <div style="border-radius:16px;padding:1rem 1.5rem;background:linear-gradient(120deg,{bg}33,{bg}55);color:{fg};border:1px solid {bg}99;margin-bottom:1rem;">
            {message}
        </div>
        """,
        unsafe_allow_html=True,
    )
