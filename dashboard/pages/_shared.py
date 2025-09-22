"""Shared helpers for the multipage Streamlit dashboard."""
from __future__ import annotations

from typing import Dict, Iterable, cast

import pandas as pd
import streamlit as st

from dashboard.data_io import (
    DB_LIVE,
    load_equity,
    load_ml_scores,
    load_positions,
    load_trades,
    seed_demo_data,
)

DataSources = Dict[str, pd.DataFrame]
_REQUIRED_KEYS: Iterable[str] = ("trades", "equity", "positions", "ml_scores")


@st.cache_data(show_spinner=False)
def _load_sources() -> DataSources:
    """Load core data tables from SQLite, seeding demo rows if needed."""

    seed_demo_data()
    return {
        "trades": load_trades(DB_LIVE),
        "equity": load_equity(DB_LIVE),
        "positions": load_positions(DB_LIVE),
        "ml_scores": load_ml_scores(DB_LIVE),
    }


def ensure_data_sources() -> DataSources:
    """Ensure `st.session_state` exposes the latest cached datasets."""

    data = st.session_state.get("data_sources")
    if isinstance(data, dict) and all(key in data for key in _REQUIRED_KEYS):
        return cast(DataSources, data)

    sources = _load_sources()
    st.session_state["data_sources"] = sources
    return sources
