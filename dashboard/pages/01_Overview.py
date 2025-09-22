from __future__ import annotations

from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from fpdf import FPDF
from tabulate import tabulate

from analytics import aggregate_trade_kpis, attribution_by, calc_var_cvar, kelly_fraction, rolling_sharpe
from components import (
    download_chart_as_png,
    equity_with_drawdown,
    kpi_tile,
    pnl_heatmap_hod_dow,
    pnl_histogram,
    rolling_sharpe_chart,
)
from ._shared import ensure_data_sources


@st.cache_data(show_spinner=False)
def _get_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = ensure_data_sources()
    trades = data.get("trades", pd.DataFrame()).copy()
    equity = data.get("equity", pd.DataFrame()).copy()
    return trades, equity


def _empty_state() -> None:
    st.info("No trades available. Seed demo data from Settings or adjust filters.")


def _generate_pdf_report(kpis: Dict[str, float]) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Aurora Crypto Desk · Tear Sheet", ln=1)
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 6, f"Generated {datetime.utcnow().isoformat()} UTC")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Key Metrics", ln=1)
    pdf.set_font("Helvetica", size=10)
    table = tabulate(
        [[k, f"{v:,.2f}" if isinstance(v, (int, float)) else v] for k, v in kpis.items()],
        headers=["Metric", "Value"],
    )
    for line in table.splitlines():
        pdf.cell(0, 5, line, ln=1)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Charts", ln=1)
    pdf.set_font("Helvetica", size=9)
    pdf.multi_cell(0, 5, "Please refer to the Streamlit app for interactive visuals. Exported images are available separately.")
    return pdf.output(dest="S").encode("latin-1")


def render_kpis(trades: pd.DataFrame, equity: pd.DataFrame) -> Dict[str, float]:
    metrics = aggregate_trade_kpis(trades, equity)
    realized = metrics.get("realized_pnl", 0.0)
    unrealized = float(trades.get("unrealized", pd.Series(dtype=float)).sum()) if "unrealized" in trades else 0.0
    metrics.update({"unrealized_pnl": unrealized, "net_pnl": realized + unrealized})
    metrics["win_rate"] = metrics.get("hit_rate", 0.0)
    pnl_series = trades.get("pnl", pd.Series(dtype=float))
    var, cvar = calc_var_cvar(pnl_series) if not pnl_series.empty else (0.0, 0.0)
    metrics["VaR95"] = var
    metrics["CVaR95"] = cvar
    payoff = metrics.get("payoff_ratio", 0.0)
    metrics["Kelly"] = kelly_fraction(metrics.get("hit_rate", 0.0), payoff if payoff not in (np.inf, 0) else 1.0)

    tiles = [
        ("Net PnL", metrics.get("net_pnl", 0.0)),
        ("Realized PnL", realized),
        ("Unrealized PnL", unrealized),
        ("Win rate", metrics.get("win_rate", 0.0)),
        ("Profit factor", metrics.get("profit_factor", 0.0)),
        ("Sharpe", metrics.get("sharpe", 0.0)),
        ("Sortino", metrics.get("sortino", 0.0)),
        ("Max drawdown", metrics.get("max_drawdown", 0.0)),
        ("Exposure %", metrics.get("exposure", 0.0)),
        ("Avg trade", metrics.get("avg_trade", 0.0)),
        ("Payoff ratio", payoff),
        ("Expectancy", metrics.get("expectancy", 0.0)),
        ("Fees paid", metrics.get("fees", 0.0)),
        ("Trades", metrics.get("trades", 0.0)),
        ("VaR 95%", metrics.get("VaR95", 0.0)),
        ("CVaR 95%", metrics.get("CVaR95", 0.0)),
        ("Kelly fraction", metrics.get("Kelly", 0.0)),
    ]
    cols = st.columns(4)
    for idx, (label, value) in enumerate(tiles):
        with cols[idx % 4]:
            kpi_tile(label, value)
    return metrics


def top_trades_table(trades: pd.DataFrame) -> None:
    if trades.empty:
        return
    subset_cols = [c for c in ["trade_id", "symbol", "worker", "pnl", "opened_at", "closed_at", "note"] if c in trades]
    trades = trades.copy()
    trades["opened_at"] = pd.to_datetime(trades.get("opened_at"))
    trades["closed_at"] = pd.to_datetime(trades.get("closed_at"))
    trades["duration_h"] = (trades["closed_at"] - trades["opened_at"]).dt.total_seconds().div(3600).fillna(0)
    top = trades.nlargest(10, "pnl") if "pnl" in trades else trades.head(10)
    worst = trades.nsmallest(10, "pnl") if "pnl" in trades else trades.tail(10)
    st.subheader("Best trades")
    st.dataframe(top[subset_cols + ["duration_h"]])
    st.subheader("Tough trades")
    st.dataframe(worst[subset_cols + ["duration_h"]])


st.set_page_config(page_title="Overview · Aurora Desk", page_icon="🧭")

st.title("Overview")
trades, equity = _get_data()
if trades.empty:
    _empty_state()
else:
    metrics = render_kpis(trades, equity)
    dd_fig = equity_with_drawdown(equity)
    st.plotly_chart(dd_fig, use_container_width=True)
    download_chart_as_png(dd_fig, "overview_equity")

    pnl_fig = pnl_histogram(trades)
    st.plotly_chart(pnl_fig, use_container_width=True)
    download_chart_as_png(pnl_fig, "trade_distribution")

    heatmap_fig = pnl_heatmap_hod_dow(trades)
    st.plotly_chart(heatmap_fig, use_container_width=True)

    sharpe_df = rolling_sharpe(trades)
    st.plotly_chart(rolling_sharpe_chart(sharpe_df), use_container_width=True)

    attribution_symbol = attribution_by(trades, "symbol")
    if not attribution_symbol.empty:
        st.subheader("Attribution by symbol")
        st.dataframe(attribution_symbol)

    top_trades_table(trades)

    st.download_button("Download trades CSV", data=trades.to_csv(index=False), file_name="trades_overview.csv", mime="text/csv")

    pdf_bytes = _generate_pdf_report(metrics)
    st.download_button("Download tear sheet PDF", data=pdf_bytes, file_name="aurora_tearsheet.pdf", mime="application/pdf")
