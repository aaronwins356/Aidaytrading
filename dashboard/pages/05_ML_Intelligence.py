"""Machine learning intelligence page."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

from components import download_chart_as_png
from data_io import DB_LIVE, load_ml_scores

st.set_page_config(page_title="ML Intelligence · Aurora Desk", page_icon="🧠")

st.title("ML Intelligence")
ml_df = load_ml_scores(DB_LIVE)
if ml_df.empty:
    st.warning("No ML score logs detected. Ensure ml_scores table is populated or use the seeding utility.")
    st.stop()

ml_df["ts"] = pd.to_datetime(ml_df["ts"], errors="coerce")
ml_df.dropna(subset=["ts"], inplace=True)
ml_df.sort_values("ts", inplace=True)

workers = sorted(ml_df["worker"].dropna().unique().tolist())
if not workers:
    st.info("No worker identifiers present in ML scores.")
    st.stop()

worker = st.selectbox("Worker", workers)
worker_df = ml_df[ml_df["worker"] == worker]

if worker_df.empty:
    st.info("Selected worker has no ML scores yet.")
    st.stop()

st.subheader("Calibration")
bins = np.linspace(0, 1, 11)
worker_df["bin"] = pd.cut(worker_df["proba_win"], bins, include_lowest=True)
calibration = worker_df.groupby("bin")["label"].mean().fillna(0)
midpoints = [interval.mid for interval in calibration.index.categories] if hasattr(calibration.index, "categories") else bins[:-1] + 0.05
fig_cal = go.Figure()
fig_cal.add_trace(go.Scatter(x=midpoints, y=calibration.values, mode="lines+markers", name="Observed"))
fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ideal", line=dict(dash="dash")))
fig_cal.update_layout(title="Calibration Curve", xaxis_title="Predicted win probability", yaxis_title="Empirical win rate")
st.plotly_chart(fig_cal, use_container_width=True)
download_chart_as_png(fig_cal, f"calibration_{worker}")

st.subheader("ROC & PR curves")
try:
    roc_auc = roc_auc_score(worker_df["label"], worker_df["proba_win"])
except ValueError:
    roc_auc = float("nan")
if worker_df["label"].nunique() > 1:
    fpr, tpr, _ = roc_curve(worker_df["label"], worker_df["proba_win"])
else:
    fpr, tpr = np.array([0, 1]), np.array([0, 1])
prec, rec, _ = precision_recall_curve(worker_df["label"], worker_df["proba_win"])
pr_auc = auc(rec, prec)
fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC AUC {roc_auc:.2f}"))
fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash")))
fig_pr = go.Figure()
fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"PR AUC {pr_auc:.2f}"))
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_roc, use_container_width=True)
with col2:
    st.plotly_chart(fig_pr, use_container_width=True)

st.subheader("Score gating analysis")
threshold = st.slider("Minimum probability", 0.0, 1.0, 0.5, step=0.05)
filtered = worker_df[worker_df["proba_win"] >= threshold]
trades_df = st.session_state.get("data_sources", {}).get("trades", pd.DataFrame())
if not trades_df.empty:
    merged = worker_df.merge(trades_df, how="left", on=["worker", "symbol"], suffixes=("", "_trade"))
else:
    merged = worker_df
if not filtered.empty:
    win_rate = filtered["label"].mean()
    st.metric("Hit rate", f"{win_rate:.2%}")
    st.metric("Scores gated", len(filtered))
    if "pnl_trade" in merged:
        gated_pnl = merged.loc[filtered.index, "pnl_trade"].dropna()
        if not gated_pnl.empty:
            st.metric("Avg trade PnL", f"{gated_pnl.mean():,.2f}")
else:
    st.info("No scores above threshold. Lower the gate.")

st.subheader("Feature importance snapshot")
if "features_json" in worker_df:
    def _extract_features(row: str) -> dict:
        try:
            payload = json.loads(row)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    features = worker_df["features_json"].dropna().map(_extract_features)
    if not features.empty:
        feature_df = pd.DataFrame(features.tolist()).mean().sort_values(ascending=False).head(10)
        st.bar_chart(feature_df)
        st.download_button("Export features CSV", data=feature_df.to_csv(), file_name="feature_importance.csv", mime="text/csv")
    else:
        st.info("No feature payloads available.")
else:
    st.info("No feature payloads available.")
