"""Logs and reporting utilities."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from fpdf import FPDF

from data_io import LOG_DIR, load_logs

st.set_page_config(page_title="Logs & Reports · Aurora Desk", page_icon="🗂")

st.title("Logs & Reports")
logs = load_logs(LOG_DIR)
if logs.empty:
    st.info("No log files discovered yet.")
else:
    level = st.selectbox("Level", options=["ALL"] + sorted(logs["level"].dropna().unique().tolist()))
    keyword = st.text_input("Keyword", help="Filter log messages. Case-insensitive substring match.")
    filtered = logs.copy()
    if level != "ALL":
        filtered = filtered[filtered["level"] == level]
    if keyword:
        filtered = filtered[filtered["message"].astype(str).str.contains(keyword, case=False, na=False)]
    st.dataframe(filtered.tail(200))

    st.subheader("Health metrics")
    latencies = [payload.get("latency_ms") for payload in filtered["payload"].dropna() if isinstance(payload, dict) and payload.get("latency_ms")]
    if latencies:
        st.metric("Latency p95", f"{pd.Series(latencies).quantile(0.95):.2f} ms")
    st.metric("Last log entry", str(logs["ts"].max()))

    st.download_button("Export logs CSV", data=filtered.to_csv(index=False), file_name="filtered_logs.csv", mime="text/csv")

report_dir = Path("desk/reports")
report_dir.mkdir(parents=True, exist_ok=True)

st.subheader("Generate weekly PDF")
if st.button("Create report"):
    trades = st.session_state.get("data_sources", {}).get("trades", pd.DataFrame())
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Aurora Desk Weekly Report", ln=1)
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 8, f"Generated {datetime.utcnow().isoformat()} UTC", ln=1)
    pdf.ln(4)
    if not trades.empty and "symbol" in trades and "pnl" in trades:
        summary = trades.groupby("symbol")["pnl"].sum()
        for symbol, value in summary.items():
            pdf.cell(0, 6, f"{symbol}: {value:,.2f}", ln=1)
    else:
        pdf.cell(0, 6, "No trades this week", ln=1)
    output_path = report_dir / f"weekly_report_{datetime.utcnow().date()}.pdf"
    pdf.output(str(output_path))
    st.success(f"Report saved to {output_path}")
