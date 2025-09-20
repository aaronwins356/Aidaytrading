"""Machine learning intelligence page."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

from components import download_chart_as_png
from data_io import DB_LIVE, DB_PAPER, load_ml_scores

st.set_page_config(page_title="ML Intelligence · Aurora Desk", page_icon="🧠")

st.title("ML Intelligence")
mode = st.session_state.get("filters", {}).get("mode", "Paper")
dfs = []
if mode in ("Paper", "Both"):
    dfs.append(load_ml_scores(DB_PAPER))
if mode in ("Live", "Both"):
    dfs.append(load_ml_scores(DB_LIVE))
ml_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
if ml_df.empty:
    st.warning("No ML score logs detected. Ensure ml_scores table is populated or use the seeding utility.")
    st.stop()

ml_df["ts"] = pd.to_datetime(ml_df["ts"])
ml_df.sort_values("ts", inplace=True)

worker = st.selectbox("Worker", sorted(ml_df["worker"].dropna().unique().tolist()))
worker_df = ml_df[ml_df["worker"] == worker]

if worker_df.empty:
    st.info("Selected worker has no ML scores yet.")
    st.stop()

st.subheader("Calibration")
bins = np.linspace(0, 1, 11)
worker_df["bin"] = pd.cut(worker_df["proba_win"], bins)
calibration = worker_df.groupby("bin")["label"].mean()
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
fpr, tpr, _ = roc_curve(worker_df["label"], worker_df["proba_win"]) if worker_df["label"].nunique() > 1 else (np.array([0, 1]), np.array([0, 1]), None)
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
full_pnl = worker_df.merge(st.session_state.get("data_sources", {}).get("trades", pd.DataFrame()), how="left", on=["worker", "symbol"])
if not filtered.empty:
    win_rate = filtered["label"].mean()
    st.metric("Hit rate", f"{win_rate:.2%}")
    st.metric("Trades gated", len(filtered))
else:
    st.info("No scores above threshold. Lower the gate.")

st.subheader("Feature importance snapshot")
if "features_json" in worker_df:
    def _extract_features(row):
        try:
            return json.loads(row)
        except Exception:
            return {}

    features = worker_df["features_json"].dropna().map(_extract_features)
    feature_df = pd.DataFrame(features.tolist()).mean().sort_values(ascending=False).head(10)
    st.bar_chart(feature_df)
else:
    st.info("No feature payloads available.")
