# apps/dashboard.py
import os
import sqlite3
from pathlib import Path
from contextlib import closing

import pandas as pd
import yaml
import streamlit as st
import plotly.graph_objects as go

# -----------------------------
# Paths
# -----------------------------
from desk.config import DESK_ROOT, CONFIG_PATH

DB_PATH = DESK_ROOT / "logs" / "trades.db"
BAL_DB_PATH = DESK_ROOT / "logs" / "balances.db"

# -----------------------------
# DB bootstrap & migration
# -----------------------------
def init_db():
    with closing(sqlite3.connect(str(DB_PATH))) as conn, closing(conn.cursor()) as c:
        # ensure base schema exists
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                worker TEXT,
                symbol TEXT,
                side TEXT,
                qty REAL,
                entry_price REAL,
                exit_price REAL,
                exit_reason TEXT,
                pnl REAL
            )
            """
        )
        # migrate old schema if needed
        cols = {r[1] for r in c.execute("PRAGMA table_info(trades)").fetchall()}
        if "entry_price" not in cols:
            c.execute("ALTER TABLE trades ADD COLUMN entry_price REAL")
        if "exit_price" not in cols:
            c.execute("ALTER TABLE trades ADD COLUMN exit_price REAL")
        if "exit_reason" not in cols:
            c.execute("ALTER TABLE trades ADD COLUMN exit_reason TEXT")
        conn.commit()

    # balances db
    with closing(sqlite3.connect(str(BAL_DB_PATH))) as conn, closing(conn.cursor()) as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS balances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                usd REAL,
                positions TEXT
            )
            """
        )
        if not pd.read_sql("SELECT * FROM balances", conn).shape[0]:
            c.execute("INSERT INTO balances (usd, positions) VALUES (?, ?)", (1000.0, "{}"))
        conn.commit()

# -----------------------------
# Config helpers
# -----------------------------
def safe_read_yaml(path, default):
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or default
    except Exception:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(default, f)
        return default

def load_config():
    default_cfg = {
        "settings": {"balance": 1000, "loop_delay": 60, "warmup_candles": 10},
        "risk": {"fixed_risk_usd": 50, "rr_ratio": 2.0, "ml_weight": 0.5, "retrain_every": 10},
        "workers": [],
    }
    return safe_read_yaml(CONFIG_PATH, default_cfg)

def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(config, f)

# -----------------------------
# Data loaders
# -----------------------------
def load_trades():
    with closing(sqlite3.connect(str(DB_PATH))) as conn:
        trades = pd.read_sql("SELECT * FROM trades ORDER BY timestamp ASC", conn)
    if not trades.empty:
        trades["time"] = pd.to_datetime(trades["timestamp"], unit="s")
    return trades

def compute_stats(trades: pd.DataFrame, starting_balance: float):
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), starting_balance, 0, 0, 0, 0.0

    open_trades = trades[trades["exit_price"].isna()].copy()
    closed = trades[~trades["exit_price"].isna()].copy()

    if not closed.empty:
        closed["cum_pnl"] = closed["pnl"].cumsum()
        closed["equity"] = starting_balance + closed["cum_pnl"]
        closed["uptime_min"] = (closed["time"] - closed["time"].iloc[0]).dt.total_seconds() / 60.0
        equity = float(closed["equity"].iloc[-1])
        last_pnl = float(closed["pnl"].iloc[-1])
    else:
        equity = starting_balance
        last_pnl = 0.0

    total_trades = int(closed.shape[0])
    total_pnl = float(closed["pnl"].sum()) if not closed.empty else 0.0
    winrate = round(((closed["pnl"] > 0).sum() / total_trades) * 100, 2) if total_trades > 0 else 0.0

    if not closed.empty:
        worker_stats = (
            closed.groupby("worker")
            .agg(
                trades=("id", "count"),
                pnl_usd=("pnl", "sum"),
                wins=("pnl", lambda x: (x > 0).sum()),
                losses=("pnl", lambda x: (x <= 0).sum()),
            )
            .reset_index()
        )
        worker_stats["winrate"] = (worker_stats["wins"] / worker_stats["trades"] * 100).round(2)
    else:
        worker_stats = pd.DataFrame(columns=["worker", "trades", "pnl_usd", "wins", "losses", "winrate"])

    return open_trades, closed, worker_stats, equity, last_pnl, total_trades, total_pnl, winrate

# -----------------------------
# UI helpers
# -----------------------------
def inject_css():
    st.markdown(
        """
        <style>
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .appbar {position: sticky; top: 0; z-index: 5;
            background: linear-gradient(90deg, rgba(17,17,22,0.92), rgba(25,25,32,0.92));
            border-bottom: 1px solid rgba(255,255,255,0.08);
            padding: 14px 18px; margin: -1rem -1rem 1rem -1rem;}
        .appbar h1 {color: #e7e9ee; font-weight: 700; font-size: 1.1rem; margin: 0;}
        .appbar .sub {color: #a9b3c1; font-size: .85rem; margin-top: 2px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

def equity_chart(closed: pd.DataFrame, starting_balance: float):
    fig = go.Figure()
    if not closed.empty:
        fig.add_trace(
            go.Scatter(
                x=closed["uptime_min"],
                y=closed["equity"],
                mode="lines",
                name="Equity",
                line=dict(width=2),
                fill="tozeroy",
            )
        )
    fig.update_layout(template="plotly_dark", height=420, margin=dict(l=10, r=10, t=10, b=10))
    return fig

def reset_account_keep_ml(starting_balance: float):
    with closing(sqlite3.connect(DB_PATH)) as conn, closing(conn.cursor()) as c:
        c.execute("DELETE FROM trades")
        conn.commit()
    with closing(sqlite3.connect(BAL_DB_PATH)) as conn, closing(conn.cursor()) as c:
        c.execute("DELETE FROM balances")
        c.execute("INSERT INTO balances (usd, positions) VALUES (?, ?)", (starting_balance, "{}"))
        conn.commit()

# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="Trading Bot — Control Room", layout="wide")
inject_css()
init_db()

config = load_config()
starting_balance = float(config.get("settings", {}).get("balance", 1000))
trades = load_trades()
open_trades, closed, worker_stats, equity, last_pnl, total_trades, total_pnl, winrate = compute_stats(trades, starting_balance)

st.markdown(
    """
    <div class="appbar">
      <h1>Trading Bot — Control Room</h1>
      <div class="sub">Entry→Exit lifecycle • Realized PnL • Live config</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Equity", f"${equity:,.2f}", f"{last_pnl:+.2f}")
c2.metric("Total PnL", f"${total_pnl:,.2f}")
c3.metric("Win Rate", f"{winrate:.2f}%")
c4.metric("Closed Trades", total_trades)

# Tabs
tab_overview, tab_trades, tab_workers, tab_settings = st.tabs(["Overview", "Trades", "Workers", "Settings"])

with tab_overview:
    st.subheader("📈 Equity Curve (Realized PnL)")
    st.plotly_chart(equity_chart(closed, starting_balance), use_container_width=True)

with tab_trades:
    left, right = st.columns(2)
    with left:
        st.markdown("### 🟢 Open Trades")
        if open_trades.empty:
            st.info("No open trades.")
        else:
            st.dataframe(
                open_trades[
                    ["time","worker","symbol","side","qty","entry_price","pnl"]
                ].rename(columns={
                    "time":"Time","worker":"Worker","symbol":"Symbol",
                    "side":"Side","qty":"Qty","entry_price":"Entry","pnl":"PnL"
                }),
                use_container_width=True, hide_index=True
            )
    with right:
        st.markdown("### 📄 Closed Trades")
        if closed.empty:
            st.info("No closed trades yet.")
        else:
            view = closed[["time","worker","symbol","side","qty","entry_price","exit_price","exit_reason","pnl"]].tail(50)
            st.dataframe(
                view.rename(columns={
                    "time":"Time","worker":"Worker","symbol":"Symbol","side":"Side",
                    "qty":"Qty","entry_price":"Entry","exit_price":"Exit","exit_reason":"Reason","pnl":"PnL"
                }),
                use_container_width=True, hide_index=True
            )

with tab_workers:
    st.subheader("🤖 Worker Performance (Closed Trades)")
    if worker_stats.empty:
        st.info("No worker stats yet.")
    else:
        ws = worker_stats.copy().sort_values("pnl_usd", ascending=False)
        st.dataframe(
            ws.rename(columns={
                "worker":"Worker","trades":"Trades","pnl_usd":"PnL ($)",
                "wins":"Wins","losses":"Losses","winrate":"Winrate (%)"
            }),
            use_container_width=True, hide_index=True
        )

with tab_settings:
    st.subheader("⚙️ Live Config Editor")
    settings = config.get("settings", {})
    risk = config.get("risk", {})
    workers = config.get("workers", [])

    settings["balance"] = st.number_input("Starting Balance", value=float(settings.get("balance",1000)), step=100.0)
    settings["loop_delay"] = st.number_input("Loop Delay (s)", value=int(settings.get("loop_delay",60)), step=1)
    settings["warmup_candles"] = st.number_input("Warmup Candles", value=int(settings.get("warmup_candles",10)), step=1)

    risk["fixed_risk_usd"] = st.number_input("Fixed Risk per Trade ($)", value=float(risk.get("fixed_risk_usd",50.0)), step=1.0)
    risk["rr_ratio"] = st.number_input("Risk/Reward Ratio", value=float(risk.get("rr_ratio",2.0)), step=0.1)
    risk["ml_weight"] = st.slider("ML Weight", 0.0, 1.0, float(risk.get("ml_weight",0.5)), step=0.05)
    risk["retrain_every"] = st.number_input("Retrain Every N Trades", value=int(risk.get("retrain_every",10)), step=1)

    for w in workers:
        st.markdown(f"**{w.get('name','?')}** — {w.get('symbol','?')}")
        w["allocation"] = st.number_input(f"{w['name']} Allocation", value=float(w.get("allocation",0.1)), step=0.01)
        params = w.setdefault("params", {})
        params["risk_per_trade"] = st.number_input(f"{w['name']} Risk %", value=float(params.get("risk_per_trade",0.02)), step=0.01)
        st.divider()

    save_col, reset_col = st.columns(2)
    with save_col:
        if st.button("💾 Save Config"):
            config["settings"] = settings
            config["risk"] = risk
            config["workers"] = workers
            save_config(config)
            st.success("Config saved.")

    with reset_col:
        if st.button("♻️ Reset Account (Keep ML)"):
            reset_account_keep_ml(float(settings.get("balance",1000)))
            st.success("Account reset.")
            st.experimental_rerun()


