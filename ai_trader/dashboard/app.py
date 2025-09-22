"""Streamlit dashboard for the AI trading platform."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yaml

from ..services.ml import MLService

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "trades.db"
CONFIG_PATH = BASE_DIR / "config.yaml"
MODULE_DISPLAY_NAMES = {
    "ai_trader.workers.short_momentum.ShortMomentumWorker": "Velocity Short",
    "ai_trader.workers.short_mean_reversion.ShortMeanReversionWorker": "Reversion Raider",
    "ai_trader.workers.ml_short.MLShortWorker": "ML Short Alpha",
    "ai_trader.workers.researcher.MarketResearchWorker": "Research Sentinel",
}

st.set_page_config(page_title="AI Trader Control Center", layout="wide", page_icon="🧠")

CUSTOM_CSS = """
<style>
body {
    background: radial-gradient(circle at 20% 20%, rgba(0, 255, 170, 0.15), transparent 60%),
                radial-gradient(circle at 80% 0%, rgba(0, 120, 255, 0.12), transparent 55%),
                #0b1021;
    color: #f5f5ff;
    font-family: 'Inter', 'JetBrains Mono', monospace;
}
section.main > div {
    backdrop-filter: blur(18px);
    background: rgba(15, 25, 46, 0.78);
    border: 1px solid rgba(0, 255, 170, 0.15);
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 24px;
}
.stButton>button {
    background: linear-gradient(135deg, #00d4ff 0%, #0070f3 100%);
    color: #0b1021;
    border-radius: 12px;
    font-weight: 600;
    padding: 0.6rem 1.4rem;
    border: none;
}
.metric-card {
    padding: 1rem;
    border-radius: 1rem;
    border: 1px solid rgba(0, 255, 170, 0.25);
    background: rgba(11, 16, 33, 0.8);
    margin-bottom: 1rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _auto_refresh_script(interval_seconds: int) -> None:
    st.markdown(
        f"""
        <script>
        setTimeout(function() {{ window.location.reload(); }}, {interval_seconds * 1000});
        </script>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=5)
def load_config() -> Dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_config(config: Dict) -> None:
    with CONFIG_PATH.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    load_config.clear()


@st.cache_resource
def init_ml_service(config: Dict) -> MLService:
    ml_cfg = config.get("ml", {})
    feature_keys = ml_cfg.get(
        "feature_keys",
        [
            "momentum_1",
            "momentum_3",
            "momentum_5",
            "momentum_10",
            "rolling_volatility",
            "atr",
            "volume_delta",
            "volume_ratio",
            "volume_ratio_3",
            "volume_ratio_10",
            "body_pct",
            "upper_wick_pct",
            "lower_wick_pct",
            "wick_close_ratio",
            "range_pct",
            "ema_fast",
            "ema_slow",
            "macd",
            "macd_hist",
            "rsi",
            "zscore",
            "close_to_high",
            "close_to_low",
        ],
    )
    return MLService(
        db_path=DB_PATH,
        feature_keys=feature_keys,
        learning_rate=float(ml_cfg.get("learning_rate", 0.03)),
        regularization=float(ml_cfg.get("regularization", 0.0005)),
        threshold=float(ml_cfg.get("threshold", 0.7)),
        ensemble=bool(ml_cfg.get("ensemble_enabled", True)),
        forest_size=int(ml_cfg.get("ensemble_trees", 15)),
        random_state=int(ml_cfg.get("random_state", 7)),
    )


@st.cache_data(ttl=5)
def load_trades() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame(
            columns=[
                "timestamp",
                "worker",
                "symbol",
                "side",
                "cash_spent",
                "entry_price",
                "exit_price",
                "pnl_percent",
                "pnl_usd",
                "win_loss",
            ]
        )
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT timestamp, worker, symbol, side, cash_spent, entry_price, exit_price, pnl_percent, pnl_usd, win_loss FROM trades ORDER BY timestamp DESC",
            conn,
        )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


@st.cache_data(ttl=5)
def load_equity_curve() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame(columns=["timestamp", "equity", "pnl_percent", "pnl_usd"])
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT timestamp, equity, pnl_percent, pnl_usd FROM equity_curve ORDER BY timestamp ASC",
            conn,
        )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


@st.cache_data(ttl=5)
def load_bot_states() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame(columns=["worker", "symbol", "status", "last_signal", "indicators", "risk", "updated_at"])
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT worker, symbol, status, last_signal, indicators_json, risk_json, updated_at FROM bot_state"
        ).fetchall()
    records: List[Dict[str, object]] = []
    for row in rows:
        indicators = json.loads(row["indicators_json"] or "{}")
        risk = json.loads(row["risk_json"] or "{}")
        records.append(
            {
                "worker": row["worker"],
                "symbol": row["symbol"],
                "status": row["status"],
                "last_signal": row["last_signal"],
                "indicators": indicators,
                "risk": risk,
                "updated_at": pd.to_datetime(row["updated_at"], utc=True),
            }
        )
    return pd.DataFrame(records)


def load_control_flags() -> Dict[str, str]:
    if not DB_PATH.exists():
        return {}
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT key, value FROM control_flags").fetchall()
    return {row["key"]: row["value"] for row in rows}


def set_control_flag(key: str, value: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO control_flags(key, value, updated_at)
            VALUES(?, ?, datetime('now'))
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (key, value),
        )
        conn.commit()
    load_bot_states.clear()


@st.cache_data(ttl=5)
def load_market_features(symbol: str, limit: int = 720) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "features"])
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            """
            SELECT timestamp, open, high, low, close, volume, features_json, label
            FROM market_features
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            conn,
            params=(symbol, limit),
        )
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    numeric_cols = [col for col in ["open", "high", "low", "close", "volume"] if col in df.columns]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df["features"] = df["features_json"].apply(lambda x: json.loads(x or "{}"))
    return df.sort_values("timestamp")


@st.cache_data(ttl=5)
def load_ml_predictions(symbol: str, limit: int = 720) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame(columns=["timestamp", "worker", "confidence", "decision", "threshold"])
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            """
            SELECT timestamp, worker, confidence, decision, threshold
            FROM ml_predictions
            WHERE symbol = ? AND worker != 'researcher'
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            conn,
            params=(symbol, limit),
        )
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp")


@st.cache_data(ttl=5)
def load_ml_metrics() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame(columns=["timestamp", "symbol", "mode", "precision", "recall", "win_rate", "support"])
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            """
            SELECT timestamp, symbol, mode, precision, recall, win_rate, support
            FROM ml_metrics
            ORDER BY timestamp DESC
            """,
            conn,
        )
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


@st.cache_data(ttl=5)
def load_account_snapshot() -> Optional[Dict[str, object]]:
    if not DB_PATH.exists():
        return None
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
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
    return {
        "timestamp": pd.to_datetime(row["timestamp"], utc=True),
        "equity": float(row["equity"]),
        "balances": json.loads(row["balances_json"] or "{}"),
    }


def render_account_overview(
    config: Dict,
    equity_df: pd.DataFrame,
    trades: pd.DataFrame,
    account_snapshot: Optional[Dict[str, object]],
) -> None:
    st.header("Account Overview")
    latest_equity = equity_df["equity"].iloc[-1] if not equity_df.empty else config["trading"].get("paper_starting_equity", 0)
    pnl_percent = equity_df["pnl_percent"].iloc[-1] if not equity_df.empty else 0.0
    pnl_usd = equity_df["pnl_usd"].iloc[-1] if not equity_df.empty else 0.0
    starting_equity = config["trading"].get("paper_starting_equity", 0.0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Live Equity", f"${latest_equity:,.2f}", f"{pnl_percent:.2f}%")
    col2.metric("Total P/L", f"${pnl_usd:,.2f}")
    col3.metric("Starting Equity", f"${starting_equity:,.2f}")
    col4.metric("Recorded Trades", f"{len(trades)}")

    if account_snapshot:
        snapshot_time = account_snapshot["timestamp"].strftime("%Y-%m-%d %H:%M:%S UTC")
        st.caption(f"Equity values update every trading engine cycle | Broker snapshot: {snapshot_time}")
        balances = account_snapshot.get("balances", {})
        if balances:
            cols = st.columns(min(4, len(balances)))
            for idx, (asset, amount) in enumerate(sorted(balances.items())):
                cols[idx % len(cols)].metric(asset, f"{amount:,.6f}")
    else:
        st.caption("Equity values update every trading engine cycle (near real-time).")


def render_equity_curve(equity_df: pd.DataFrame) -> None:
    st.subheader("Equity Curve")
    if equity_df.empty:
        st.info("Equity data will populate as trades are logged.")
        return
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity_df["timestamp"],
            y=equity_df["equity"],
            mode="lines+markers",
            name="Equity",
            line=dict(color="#00d4ff", width=2),
        )
    )
    fig.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)


def render_bot_cards(config: Dict, bot_states: pd.DataFrame, ml_service: MLService) -> None:
    st.subheader("Strategy Stack")
    _ = ml_service  # clarity: cards already include ML stats via stored indicators
    if bot_states.empty:
        st.info("Workers will appear once the engine publishes their state.")
        return

    descriptions = {
        "Velocity Short": "Momentum-driven short seller that fades breakdown accelerations.",
        "Reversion Raider": "Contrarian short scalper targeting mean reversions after euphoric spikes.",
        "ML Short Alpha": "Machine learning assisted signal combiner focusing on asymmetric short setups.",
        "Research Sentinel": "Market researcher that constantly engineers features for models and traders.",
    }

    grouped = bot_states.groupby("worker")
    cols = st.columns(2)
    for idx, (worker, df) in enumerate(grouped):
        card_placeholder = cols[idx % 2].container()
        with card_placeholder:
            st.markdown(f"### {worker}")
            st.caption(descriptions.get(worker, ""))
            latest = df.sort_values("updated_at", ascending=False).iloc[0]
            status = latest.get("status", "unknown")
            last_signal = latest.get("last_signal") or "–"
            st.markdown(f"**Status:** `{status}` | **Last Signal:** `{last_signal}`")
            indicators = latest.get("indicators", {}) or {}
            risk = latest.get("risk", {}) or {}
            ml_confidence = indicators.get("ml_confidence")
            if indicators:
                pretty = {k: round(v, 4) if isinstance(v, (int, float)) else v for k, v in indicators.items()}
                st.json({"indicators": pretty})
            if risk:
                st.json({"risk": risk})
            if ml_confidence is not None:
                st.metric("ML Confidence", f"{ml_confidence:.3f}")
            st.markdown(f"_Last update: {latest['updated_at'].strftime('%H:%M:%S UTC')}_")


def build_market_figure(
    df: pd.DataFrame,
    symbol: str,
    trades: pd.DataFrame,
    confidence_df: pd.DataFrame,
) -> Optional[go.Figure]:
    if df.empty:
        return None
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.7, 0.3],
    )
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=symbol,
        ),
        row=1,
        col=1,
    )
    entries = trades[(trades["symbol"] == symbol) & trades["exit_price"].isna()]
    exits = trades[(trades["symbol"] == symbol) & trades["exit_price"].notna()]
    if not entries.empty:
        fig.add_trace(
            go.Scatter(
                x=entries["timestamp"],
                y=entries["entry_price"],
                mode="markers",
                marker=dict(symbol="triangle-down", size=12, color="#ff4d4d"),
                name="Short entries",
            ),
            row=1,
            col=1,
        )
    if not exits.empty:
        fig.add_trace(
            go.Scatter(
                x=exits["timestamp"],
                y=exits["exit_price"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=12, color="#00ffb3"),
                name="Covers",
            ),
            row=1,
            col=1,
        )
    if not confidence_df.empty:
        for worker, worker_df in confidence_df.groupby("worker"):
            fig.add_trace(
                go.Scatter(
                    x=worker_df["timestamp"],
                    y=worker_df["confidence"],
                    mode="lines",
                    name=f"{worker} confidence",
                ),
                row=2,
                col=1,
            )
        fig.update_yaxes(title_text="Confidence", range=[0, 1], row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_layout(
        template="plotly_dark",
        height=520,
        margin=dict(l=20, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def render_market_view(symbol: str, trades: pd.DataFrame, ml_service: MLService) -> None:
    st.subheader("Market Intelligence")
    feature_df = load_market_features(symbol)
    confidence_df = load_ml_predictions(symbol)
    fig = build_market_figure(feature_df, symbol, trades, confidence_df)
    if fig is None:
        st.info("Waiting for market data snapshots from the researcher bot.")
        return
    st.plotly_chart(fig, use_container_width=True)

    if not feature_df.empty:
        latest = feature_df.iloc[-1]
        st.caption("Latest engineered features")
        enriched = dict(latest["features"])
        for column in ["open", "high", "low", "close", "volume"]:
            if column in latest:
                enriched[column] = latest[column]
        st.json(enriched)
    top_features = ml_service.feature_importance(symbol)
    if top_features:
        st.caption("Top ML feature weights")
        feat_df = pd.DataFrame(
            {"feature": list(top_features.keys()), "weight": list(top_features.values())}
        )
        bar = go.Figure(
            data=[go.Bar(x=feat_df["feature"], y=feat_df["weight"], marker_color="#00d4ff")]
        )
        bar.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=40))
        st.plotly_chart(bar, use_container_width=True)


def render_trade_logs(trades: pd.DataFrame) -> None:
    st.subheader("Trade Logs")
    if trades.empty:
        st.info("Trades will appear once execution begins.")
        return
    workers = ["All"] + sorted(trades["worker"].unique())
    selected_worker = st.selectbox("Filter by strategy", workers)
    filtered = trades if selected_worker == "All" else trades[trades["worker"] == selected_worker]
    st.dataframe(filtered, use_container_width=True)


def render_risk_controls(config: Dict, ml_service: MLService, symbol: str) -> None:
    st.subheader("Risk & Deployment Controls")
    trading_cfg = config.get("trading", {})
    risk_cfg = config.get("risk", {})
    worker_cfg = config.get("workers", {}).get("definitions", {})
    control_flags = load_control_flags()

    with st.form("global_risk_form"):
        st.write("### Global Risk")
        max_drawdown = st.slider(
            "Max drawdown %",
            min_value=5.0,
            max_value=50.0,
            step=0.5,
            value=float(risk_cfg.get("max_drawdown_percent", 25.0)),
        )
        daily_limit = st.slider(
            "Daily loss limit %",
            min_value=1.0,
            max_value=20.0,
            step=0.5,
            value=float(risk_cfg.get("daily_loss_limit_percent", 5.0)),
        )
        allocation = st.slider(
            "Equity allocation per trade %",
            min_value=1.0,
            max_value=25.0,
            step=0.5,
            value=float(trading_cfg.get("equity_allocation_percent", 6.0)),
        )
        max_positions = st.slider(
            "Maximum concurrent positions",
            min_value=1,
            max_value=12,
            value=int(trading_cfg.get("max_open_positions", 4)),
        )
        submitted = st.form_submit_button("Update global risk")
        if submitted:
            config["risk"]["max_drawdown_percent"] = max_drawdown
            config["risk"]["daily_loss_limit_percent"] = daily_limit
            config["trading"]["equity_allocation_percent"] = allocation
            config["trading"]["max_open_positions"] = max_positions
            save_config(config)
            st.success("Global risk settings saved. Restart engine to apply immediately.")

    st.write("### Per-strategy tuning")
    for worker_key, definition in worker_cfg.items():
        with st.expander(definition.get("module", worker_key), expanded=False):
            symbols = definition.get("symbols", [])
            risk = definition.get("risk", {})
            params = definition.get("parameters", {})
            new_symbols = st.multiselect(
                "Symbols",
                options=config["trading"].get("symbols", symbols),
                default=symbols,
                key=f"symbols_{worker_key}",
            )
            position_size = st.slider(
                "Position size % of allocation",
                min_value=10.0,
                max_value=200.0,
                step=5.0,
                value=float(risk.get("position_size_pct", 100.0)),
                key=f"pos_{worker_key}",
            )
            leverage = st.slider(
                "Leverage",
                min_value=1.0,
                max_value=5.0,
                step=0.1,
                value=float(risk.get("leverage", 1.0)),
                key=f"lev_{worker_key}",
            )
            stop_loss = st.slider(
                "Stop loss %",
                min_value=0.5,
                max_value=5.0,
                step=0.1,
                value=float(risk.get("stop_loss_pct", 1.0)),
                key=f"sl_{worker_key}",
            )
            take_profit = st.slider(
                "Take profit %",
                min_value=0.5,
                max_value=6.0,
                step=0.1,
                value=float(risk.get("take_profit_pct", 2.0)),
                key=f"tp_{worker_key}",
            )
            trailing = st.slider(
                "Trailing stop %",
                min_value=0.0,
                max_value=5.0,
                step=0.1,
                value=float(risk.get("trailing_stop_pct", 0.0)),
                key=f"trail_{worker_key}",
            )
            if st.button("Save", key=f"save_{worker_key}"):
                definition["symbols"] = new_symbols
                definition.setdefault("risk", {})
                definition["risk"].update(
                    {
                        "position_size_pct": position_size,
                        "leverage": leverage,
                        "stop_loss_pct": stop_loss,
                        "take_profit_pct": take_profit,
                        "trailing_stop_pct": trailing,
                    }
                )
                definition.setdefault("parameters", {})
                definition["parameters"].update(params)
                config["workers"]["definitions"][worker_key] = definition
                save_config(config)
                st.success(f"Updated configuration for {worker_key}.")

    st.write("### ML Gating Overrides")
    gating_workers = [
        (worker_key, definition)
        for worker_key, definition in worker_cfg.items()
        if not definition.get("module", "").endswith("researcher.MarketResearchWorker")
    ]
    if gating_workers:
        cols = st.columns(min(3, len(gating_workers)))
        for idx, (worker_key, definition) in enumerate(gating_workers):
            module_path = definition.get("module", worker_key)
            worker_label = MODULE_DISPLAY_NAMES.get(module_path, module_path.split(".")[-1])
            flag_key = f"ml::{worker_label}"
            enabled = control_flags.get(flag_key, "on").lower() not in {"off", "false", "0", "disabled"}
            toggle = cols[idx % len(cols)].toggle(
                f"{worker_label} gating",
                value=enabled,
                key=f"ml_gate_{worker_key}",
            )
            if toggle != enabled:
                set_control_flag(flag_key, "on" if toggle else "off")
                st.success(f"Updated ML gating for {worker_label}.")
    else:
        st.info("No trading workers configured for ML gating.")

    st.write("### ML Validation & Backtests")
    metrics_df = load_ml_metrics()
    validation_cols = st.columns(3)
    symbol_options = config.get("trading", {}).get("symbols", [symbol])
    default_index = symbol_options.index(symbol) if symbol in symbol_options else 0
    selected_symbol = validation_cols[0].selectbox(
        "Symbol",
        options=symbol_options,
        index=default_index,
        key="ml_metric_symbol",
    )
    if validation_cols[1].button("Run backtest", key="run_ml_backtest"):
        results = ml_service.run_backtest(selected_symbol)
        load_ml_metrics.clear()
        st.success(
            "Backtest complete – precision: {precision:.2f}, recall: {recall:.2f}, win rate: {win_rate:.2f}".format(**results)
        )
    if not metrics_df.empty:
        filtered = metrics_df[metrics_df["symbol"] == selected_symbol]
        if filtered.empty:
            st.info("No metrics recorded yet for this symbol. Run a backtest to populate statistics.")
        else:
            st.dataframe(filtered.head(10), use_container_width=True)
    else:
        st.info("Metrics will appear once backtests or live evaluations are recorded.")


def render_sidebar(config: Dict) -> str:
    st.sidebar.title("Live Controls")
    refresh_interval = int(config.get("dashboard", {}).get("refresh_interval_seconds", 5))
    _auto_refresh_script(refresh_interval)

    control_flags = load_control_flags()

    trading_mode = st.sidebar.radio(
        "Trading Mode",
        options=["paper", "live"],
        index=0 if config["trading"].get("mode", "paper") == "paper" else 1,
    )
    if trading_mode != config["trading"].get("mode"):
        config["trading"]["mode"] = trading_mode
        config["trading"]["paper_trading"] = trading_mode == "paper"
        save_config(config)
        st.sidebar.success("Trading mode updated. Restart engine to take effect.")

    kill_switch = control_flags.get("kill_switch", "false") == "true"
    kill_toggle = st.sidebar.toggle("Global Kill Switch", value=kill_switch)
    if kill_toggle != kill_switch:
        set_control_flag("kill_switch", "true" if kill_toggle else "false")
        st.sidebar.success("Kill switch state updated.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Per-Bot Overrides")
    definitions = config.get("workers", {}).get("definitions", {})
    for worker_key, definition in definitions.items():
        module_path = definition.get("module", worker_key)
        if module_path.endswith("researcher.MarketResearchWorker"):
            continue
        worker_label = MODULE_DISPLAY_NAMES.get(module_path, module_path.split(".")[-1])
        flag_key = f"bot::{worker_label}"
        paused = control_flags.get(flag_key, "active") in {"paused", "disabled"}
        toggle = st.sidebar.toggle(f"{worker_label}", value=not paused)
        if toggle == paused:
            set_control_flag(flag_key, "active" if toggle else "paused")
            st.sidebar.success(f"Updated {worker_label} control flag.")

    st.sidebar.markdown("---")
    symbol = st.sidebar.selectbox(
        "Symbol",
        options=config["trading"].get("symbols", ["BTC/USD"]),
        index=0,
    )
    return symbol


def main() -> None:
    config = load_config()
    ml_service = init_ml_service(config)
    st.title("Quant Operations Terminal")
    symbol = render_sidebar(config)
    bot_states = load_bot_states()
    trades = load_trades()
    equity = load_equity_curve()
    account_snapshot = load_account_snapshot()

    render_account_overview(config, equity, trades, account_snapshot)
    render_equity_curve(equity)
    render_bot_cards(config, bot_states, ml_service)
    render_market_view(symbol, trades, ml_service)
    render_trade_logs(trades)
    render_risk_controls(config, ml_service, symbol)


if __name__ == "__main__":
    main()
