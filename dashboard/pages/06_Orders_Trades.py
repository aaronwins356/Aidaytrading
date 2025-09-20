"""Orders and trades management page."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Orders & Trades · Aurora Desk", page_icon="📜")

st.title("Orders & Trades")
trades = st.session_state.get("data_sources", {}).get("trades", pd.DataFrame())
if trades.empty:
    st.info("No trades to display.")
    st.stop()

mode = st.session_state.get("filters", {}).get("mode", "Paper")
search = st.text_input("Search", "")
columns = st.multiselect("Columns", trades.columns.tolist(), default=trades.columns.tolist())
filtered = trades.copy()
if search:
    mask = pd.Series(False, index=filtered.index)
    for col in filtered.columns:
        mask |= filtered[col].astype(str).str.contains(search, case=False, na=False)
    filtered = filtered[mask]

st.metric("Average slippage", f"{filtered.get('slippage', pd.Series(dtype=float)).mean():.4f}")
st.metric("Median time in trade", f"{(filtered['closed_at'] - filtered['opened_at']).dt.total_seconds().median()/60 if 'closed_at' in filtered else 0:.2f} mins")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Close all", disabled=mode == "Live"):
        st.success("Close all request sent to risk engine.")
with col2:
    if st.button("Close selected", disabled=mode == "Live"):
        st.success("Selected orders flagged for closure.")
with col3:
    st.download_button("Export CSV", data=filtered.to_csv(index=False), file_name="orders_filtered.csv", mime="text/csv")

st.dataframe(filtered[columns])

st.subheader("Trade details")
selected_id = st.selectbox("Choose trade", options=filtered["trade_id"].tolist())
trade_row = filtered[filtered["trade_id"] == selected_id].iloc[0].to_dict()
st.json(trade_row)

st.subheader("Timeline")
with st.expander("Execution timeline"):
    st.write("Intent → Sent → Filled → Closed")
    st.write("This section will display detailed timestamps when available in the order payloads.")

