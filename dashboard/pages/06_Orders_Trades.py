"""Orders and trades management page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Orders & Trades · Aurora Desk", page_icon="📜")

st.title("Orders & Trades")
trades = st.session_state.get("data_sources", {}).get("trades", pd.DataFrame())
if trades.empty:
    st.info("No trades to display.")
    st.stop()

mode = st.session_state.get("filters", {}).get("mode", "Paper")
search = st.text_input("Search", "", help="Filter trades by keyword. Case-insensitive.")
columns = st.multiselect("Columns", trades.columns.tolist(), default=trades.columns.tolist())
filtered = trades.copy()
if search:
    pattern = search.replace("%", "").strip()
    mask = pd.Series(False, index=filtered.index)
    for col in filtered.columns:
        mask |= filtered[col].astype(str).str.contains(pattern, case=False, na=False)
    filtered = filtered[mask]

if {"closed_at", "opened_at"}.issubset(filtered.columns):
    filtered["closed_at"] = pd.to_datetime(filtered["closed_at"])  # type: ignore
    filtered["opened_at"] = pd.to_datetime(filtered["opened_at"])  # type: ignore
else:
    filtered["closed_at"] = pd.to_datetime(filtered.get("closed_at"))
    filtered["opened_at"] = pd.to_datetime(filtered.get("opened_at"))

st.metric("Average slippage", f"{filtered.get('slippage', pd.Series(dtype=float)).mean():.4f}")
if "closed_at" in filtered and "opened_at" in filtered:
    median_minutes = (filtered["closed_at"] - filtered["opened_at"]).dt.total_seconds().median() / 60
else:
    median_minutes = 0
st.metric("Median time in trade", f"{median_minutes:.2f} mins")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Close all", disabled=mode == "Live"):
        st.success("Close all request sent to risk engine.")
with col2:
    if st.button("Close selected", disabled=mode == "Live"):
        st.success("Selected orders flagged for closure.")
with col3:
    st.download_button("Export CSV", data=filtered[columns].to_csv(index=False), file_name="orders_filtered.csv", mime="text/csv")

st.dataframe(filtered[columns])

st.subheader("Trade details")
selected_id = st.selectbox("Choose trade", options=filtered["trade_id"].astype(str).tolist())
trade_row = filtered[filtered["trade_id"].astype(str) == selected_id].iloc[0].to_dict()
# sanitize payload for display
for key, value in list(trade_row.items()):
    if isinstance(value, str) and len(value) > 500:
        trade_row[key] = value[:500] + "…"
st.json(trade_row)

st.subheader("Timeline")
with st.expander("Execution timeline"):
    st.write("Intent → Sent → Filled → Closed")
    st.write("This section will display detailed timestamps when available in the order payloads.")
