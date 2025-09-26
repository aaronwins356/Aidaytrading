"""Streamlit control center for the AI day trading bot."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

try:
    from ai_trader.services.configuration import normalize_config, read_config_file
except ModuleNotFoundError:  # pragma: no cover - allow running as a script
    import sys

    ROOT = Path(__file__).resolve().parent
    if str(ROOT.parent) not in sys.path:
        sys.path.append(str(ROOT.parent))
    from ai_trader.services.configuration import normalize_config, read_config_file

if hasattr(st, "cache_data"):
    cache_data = st.cache_data
else:  # pragma: no cover - fallback for minimal environments
    def cache_data(*_args: Any, **_kwargs: Any):
        def decorator(func):
            return func

        return decorator

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "trades.db"
TRADES_JSON_PATH = DATA_DIR / "trades.json"
EQUITY_JSON_PATH = DATA_DIR / "equity_curve.json"
CONFIG_PATH = BASE_DIR / "config.yaml"


def _set_page_style() -> None:
    if st.session_state.get("_page_ready"):
        return
    st.set_page_config(page_title="AI Trader Dashboard", page_icon="📊", layout="wide")
    st.markdown(
        """
        <style>
        body {background: #0b1021; color: #f8f9ff; font-family: "Inter", "Segoe UI", sans-serif;}
        .stMetric {background: rgba(15, 25, 46, 0.65); padding: 1rem; border-radius: 1rem;}
        .block-container {padding-top: 2rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state["_page_ready"] = True


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    )
    return cursor.fetchone() is not None


@cache_data(ttl=15)
def load_trades() -> pd.DataFrame:
    if DB_PATH.exists():
        with _connect() as conn:
            if _table_exists(conn, "trades"):
                frame = pd.read_sql_query(
                    "SELECT * FROM trades ORDER BY timestamp DESC", conn
                )
                return _prepare_trades(frame)
    if TRADES_JSON_PATH.exists():
        records = json.loads(TRADES_JSON_PATH.read_text(encoding="utf-8"))
        frame = pd.DataFrame(records)
        return _prepare_trades(frame)
    return pd.DataFrame()


def _prepare_trades(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if "metadata_json" in frame.columns:
        frame["metadata"] = frame["metadata_json"].apply(_safe_json_load)
    elif "metadata" not in frame.columns:
        frame["metadata"] = [{} for _ in range(len(frame))]
    float_cols = [
        "cash_spent",
        "entry_price",
        "exit_price",
        "pnl_percent",
        "pnl_usd",
        "confidence",
    ]
    for col in float_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def _safe_json_load(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value in (None, "", b""):
        return {}
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return {}


@cache_data(ttl=15)
def load_equity_curve() -> pd.DataFrame:
    if DB_PATH.exists():
        with _connect() as conn:
            if _table_exists(conn, "equity_curve"):
                frame = pd.read_sql_query(
                    "SELECT * FROM equity_curve ORDER BY timestamp ASC", conn
                )
                return _prepare_equity(frame)
    if EQUITY_JSON_PATH.exists():
        frame = pd.read_json(EQUITY_JSON_PATH)
        return _prepare_equity(frame)
    return pd.DataFrame()


def _prepare_equity(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    for column in ("equity", "pnl_percent", "pnl_usd"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    frame.set_index("timestamp", inplace=True)
    return frame


@cache_data(ttl=15)
def load_latest_account_snapshot() -> Dict[str, Any] | None:
    if not DB_PATH.exists():
        return None
    with _connect() as conn:
        if not _table_exists(conn, "account_snapshots"):
            return None
        row = conn.execute(
            """
            SELECT timestamp, equity, balances_json
            FROM account_snapshots
            ORDER BY timestamp DESC
            LIMIT 1
            """
        ).fetchone()
    if row is None:
        return None
    balances = _safe_json_load(row["balances_json"])
    return {
        "timestamp": row["timestamp"],
        "equity": float(row["equity"]),
        "balances": {k: float(v) for k, v in balances.items()},
    }


@cache_data(ttl=15)
def load_control_flags() -> Dict[str, str]:
    if not DB_PATH.exists():
        return {}
    with _connect() as conn:
        if not _table_exists(conn, "control_flags"):
            return {}
        rows = conn.execute("SELECT key, value FROM control_flags").fetchall()
    return {str(row["key"]): str(row["value"]) for row in rows}


def set_control_flag(key: str, value: str) -> None:
    if not DB_PATH.exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS control_flags (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            INSERT INTO control_flags(key, value, updated_at)
            VALUES(?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value,
                updated_at=excluded.updated_at
            """,
            (key, value),
        )
        conn.commit()
    load_control_flags.clear()


@cache_data(ttl=30)
def load_config() -> Dict[str, Any]:
    raw = read_config_file(CONFIG_PATH)
    return normalize_config(raw)


def save_config(config: Mapping[str, Any]) -> None:
    CONFIG_PATH.write_text(
        yaml.safe_dump(dict(config), sort_keys=False),
        encoding="utf-8",
    )
    load_config.clear()


def compute_drawdowns(equity: pd.Series) -> pd.Series:
    if equity.empty:
        return pd.Series(dtype="float64")
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return drawdown.fillna(0.0)


def compute_daily_returns(equity: pd.Series) -> pd.Series:
    if equity.empty:
        return pd.Series(dtype="float64")
    daily = equity.resample("1D").last().dropna()
    return daily.pct_change().dropna()


def render_portfolio_overview(trades: pd.DataFrame, equity: pd.DataFrame, account: Dict[str, Any] | None, config: Mapping[str, Any]) -> None:
    st.subheader("Portfolio Overview")
    latest_equity = float(equity["equity"].iloc[-1]) if not equity.empty else float(account["equity"]) if account else 0.0
    starting_equity = float(config.get("trading", {}).get("paper_starting_equity", latest_equity or 0.0))
    realized_pnl = float(trades.get("pnl_usd", pd.Series(dtype="float64")).sum())
    win_trades = trades[trades.get("win_loss") == "win"] if not trades.empty and "win_loss" in trades else pd.DataFrame()
    win_rate = (len(win_trades) / len(trades) * 100.0) if len(trades) else 0.0
    drawdown_series = compute_drawdowns(equity["equity"]) if not equity.empty else pd.Series(dtype="float64")
    max_drawdown = abs(drawdown_series.min() * 100.0) if not drawdown_series.empty else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Equity", f"${latest_equity:,.2f}")
    col2.metric("Realized PnL", f"${realized_pnl:,.2f}")
    col3.metric("Win Rate", f"{win_rate:.1f}%")
    col4.metric("Max Drawdown", f"{max_drawdown:.2f}%")

    equity_fig = go.Figure()
    if not equity.empty:
        equity_fig.add_trace(
            go.Scatter(x=equity.index, y=equity["equity"], mode="lines", name="Equity")
        )
    equity_fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=30, b=20),
        template="plotly_dark",
        title="Equity Curve",
    )
    st.plotly_chart(equity_fig, use_container_width=True)

    daily_returns = compute_daily_returns(equity["equity"]) if not equity.empty else pd.Series(dtype="float64")
    histogram = go.Figure()
    if not daily_returns.empty:
        histogram.add_trace(
            go.Histogram(x=daily_returns * 100.0, nbinsx=40, marker_color="#00d4ff")
        )
    histogram.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
        template="plotly_dark",
        title="Daily Returns (%)",
        xaxis_title="Return (%)",
        yaxis_title="Frequency",
    )
    st.plotly_chart(histogram, use_container_width=True)

    drawdown_fig = go.Figure()
    if not drawdown_series.empty:
        drawdown_fig.add_trace(
            go.Scatter(
                x=drawdown_series.index,
                y=drawdown_series * 100.0,
                fill="tozeroy",
                mode="lines",
                name="Drawdown",
            )
        )
    drawdown_fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
        template="plotly_dark",
        title="Drawdown (%)",
        yaxis_title="Drawdown %",
    )
    st.plotly_chart(drawdown_fig, use_container_width=True)


def render_trades_log(trades: pd.DataFrame) -> None:
    st.subheader("Trades Log")
    if trades.empty:
        st.info("No trades recorded yet.")
        return
    pairs = sorted(trades["symbol"].dropna().unique()) if "symbol" in trades else []
    workers = sorted(trades["worker"].dropna().unique()) if "worker" in trades else []
    with st.expander("Filters", expanded=False):
        selected_pairs = st.multiselect("Pairs", pairs, default=pairs)
        selected_workers = st.multiselect("Strategies", workers, default=workers)
        date_range = None
        if "timestamp" in trades:
            timestamps = trades["timestamp"].dropna()
            if not timestamps.empty:
                min_date = timestamps.min()
                max_date = timestamps.max()
                default_range = (min_date.date(), max_date.date())
                date_range = st.date_input("Date range", value=default_range)
    filtered = trades.copy()
    if selected_pairs:
        filtered = filtered[filtered["symbol"].isin(selected_pairs)]
    if selected_workers:
        filtered = filtered[filtered["worker"].isin(selected_workers)]
    if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
        start = pd.to_datetime(date_range[0])
        end = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
        filtered = filtered[(filtered["timestamp"] >= start) & (filtered["timestamp"] < end)]
    display_cols = [
        "timestamp",
        "worker",
        "symbol",
        "side",
        "cash_spent",
        "entry_price",
        "exit_price",
        "pnl_usd",
        "pnl_percent",
        "reason",
    ]
    available_cols = [col for col in display_cols if col in filtered.columns]
    st.dataframe(
        filtered[available_cols].sort_values("timestamp", ascending=False),
        use_container_width=True,
    )
    st.download_button(
        "Download CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="trades.csv",
        mime="text/csv",
    )


def render_risk_controls(control_flags: Mapping[str, str], config: MutableMapping[str, Any]) -> None:
    st.subheader("Risk Controls")
    risk_cfg = config.setdefault("risk", {})
    default_stop = float(control_flags.get("global::stop_loss_pct", risk_cfg.get("min_stop_buffer", 0.005) * 100))
    default_risk = float(risk_cfg.get("risk_per_trade", 0.02) * 100)
    default_drawdown = float(risk_cfg.get("max_drawdown_percent", 20.0))
    relax_enabled = control_flags.get("risk::confidence_relax", "on").lower() not in {"off", "false", "0"}

    with st.form("risk_controls"):
        stop_loss = st.slider("Stop-loss %", min_value=0.1, max_value=5.0, value=round(default_stop, 2), step=0.1)
        risk_per_trade = st.slider("Risk per trade %", min_value=0.1, max_value=10.0, value=round(default_risk, 2), step=0.1)
        max_drawdown = st.slider("Max drawdown %", min_value=1.0, max_value=60.0, value=round(default_drawdown, 1), step=0.5)
        relax_confidence = st.toggle("Relax ML confidence when idle", value=relax_enabled)
        submitted = st.form_submit_button("Apply risk controls")
    if submitted:
        set_control_flag("global::stop_loss_pct", f"{stop_loss:.2f}")
        set_control_flag("risk::risk_per_trade", f"{risk_per_trade:.2f}")
        set_control_flag("risk::confidence_relax", "on" if relax_confidence else "off")
        risk_cfg["risk_per_trade"] = risk_per_trade / 100.0
        risk_cfg["max_drawdown_percent"] = max_drawdown
        risk_cfg["min_stop_buffer"] = stop_loss / 100.0
        save_config(config)
        st.success("Risk controls updated.")


def render_strategy_manager(control_flags: Mapping[str, str], config: MutableMapping[str, Any]) -> None:
    st.subheader("Strategy Manager")
    definitions = (
        config.get("workers", {}).get("definitions", {})
        if isinstance(config.get("workers", {}), Mapping)
        else {}
    )
    if not definitions:
        st.info("No worker definitions available in configuration.")
        return
    entries: List[Dict[str, Any]] = []
    for key, definition in definitions.items():
        if not isinstance(definition, Mapping):
            continue
        module = str(definition.get("module", ""))
        is_ml = "ml" in module.lower()
        display = str(definition.get("display_name") or key)
        enabled = bool(definition.get("enabled", True))
        entries.append(
            {
                "key": key,
                "name": display,
                "module": module,
                "enabled": enabled,
                "is_ml": is_ml,
            }
        )
    rule_based = [entry for entry in entries if not entry["is_ml"]]
    ml_based = [entry for entry in entries if entry["is_ml"]]

    with st.form("strategy_manager"):
        st.markdown("### Rule-based workers")
        for entry in rule_based:
            entry["selected"] = st.checkbox(entry["name"], value=entry["enabled"], key=f"rule::{entry['key']}")
        st.markdown("### ML-driven workers")
        for entry in ml_based:
            entry["selected"] = st.checkbox(entry["name"], value=entry["enabled"], key=f"ml::{entry['key']}")
        submitted = st.form_submit_button("Update strategies")
    if submitted:
        for entry in entries:
            definition = definitions.get(entry["key"], {})
            definition["enabled"] = bool(entry.get("selected", entry["enabled"]))
            definitions[entry["key"]] = dict(definition)
            flag_value = "active" if definition["enabled"] else "disabled"
            set_control_flag(f"bot::{entry['name']}", flag_value)
            set_control_flag(f"bot::{entry['key']}", flag_value)
        save_config(config)
        st.success("Strategy configuration updated.")


def render_dashboard() -> None:
    _set_page_style()
    config = load_config()
    trades = load_trades()
    equity = load_equity_curve()
    account = load_latest_account_snapshot()
    control_flags = load_control_flags()

    tabs = st.tabs([
        "Portfolio Overview",
        "Trades Log",
        "Risk Controls",
        "Strategy Manager",
    ])

    with tabs[0]:
        render_portfolio_overview(trades, equity, account, config)
    with tabs[1]:
        render_trades_log(trades)
    with tabs[2]:
        render_risk_controls(control_flags, config)
    with tabs[3]:
        render_strategy_manager(control_flags, config)


def main() -> None:
    render_dashboard()


if __name__ == "__main__":
    main()
