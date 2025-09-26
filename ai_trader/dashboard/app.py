"""Streamlit dashboard for the AI trading platform."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BASE_DIR.parent

try:
    from ai_trader.services.configuration import normalize_config, read_config_file
    from ai_trader.services.logging import get_logger
    from ai_trader.services.ml import MLService
    from ai_trader.services.schema import ALL_TABLES
except ModuleNotFoundError:  # pragma: no cover - fallback for direct script execution
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    from ai_trader.services.configuration import normalize_config, read_config_file
    from ai_trader.services.logging import get_logger
    from ai_trader.services.ml import MLService
    from ai_trader.services.schema import ALL_TABLES

if not st.session_state.get("_page_configured", False):
    st.set_page_config(page_title="AI Trader Control Center", layout="wide", page_icon="🧠")
    st.session_state["_page_configured"] = True

DB_PATH = BASE_DIR / "data" / "trades.db"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
MODULE_DISPLAY_NAMES = {
    "ai_trader.workers.momentum.MomentumWorker": "Momentum Scout",
    "ai_trader.workers.mean_reversion.MeanReversionWorker": "Mean Reverter",
    "ai_trader.workers.researcher.MarketResearchWorker": "Research Sentinel",
}

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

LOGGER = get_logger(__name__)


def _auto_refresh_script(interval_seconds: int) -> None:
    st.markdown(
        f"""
        <script>
        setTimeout(function() {{ window.location.reload(); }}, {interval_seconds * 1000});
        </script>
        """,
        unsafe_allow_html=True,
    )


def _display_name_for_definition(definition: Dict[str, Any]) -> str:
    module_path = str(definition.get("module", ""))
    fallback = module_path.split(".")[-1] if module_path else "Worker"
    return str(definition.get("display_name") or MODULE_DISPLAY_NAMES.get(module_path, fallback))


def _strategy_summary(module_path: str) -> str:
    if "momentum" in module_path:
        return "Tracks fast versus slow EMAs to ride bullish momentum while staying long-only."
    if "mean_reversion" in module_path:
        return "Buys pullbacks toward the average and exits once price snaps back."
    if module_path.endswith("researcher.MarketResearchWorker"):
        return "Captures market features and updates the ML gate for every symbol."
    return "Long-only strategy with ML-guided confidence gates."


def _human_signal(signal: Optional[str]) -> str:
    """Translate low-level strategy signals into friendly UI text."""

    mapping = {
        None: "Idle and monitoring conditions.",
        "buy": "Looking for a long entry.",
        "exit": "Preparing to lock in profits.",
        "sell": "Closing an existing long position.",
        "hold": "Holding steady; no action required.",
        "ml-block": "Waiting for ML confidence to exceed the gate.",
    }

    # Pandas-backed data frames sometimes deliver ``NaN`` instead of ``None``.
    if signal is None or (isinstance(signal, float) and pd.isna(signal)):
        return mapping[None]

    if isinstance(signal, str):
        normalized = signal.strip().lower()
        return mapping.get(normalized, normalized.capitalize())

    # Fall back to a string representation for any unexpected signal types.
    return str(signal)


@st.cache_data(ttl=5)
def load_config() -> Dict:
    raw_config = read_config_file(CONFIG_PATH)
    return normalize_config(raw_config)


def save_config(config: Dict) -> None:
    with CONFIG_PATH.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    load_config.clear()


@st.cache_resource
def init_ml_service(config: Dict) -> MLService:
    ensure_database_schema()
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
        learning_rate=float(ml_cfg.get("learning_rate", ml_cfg.get("lr", 0.03))),
        regularization=float(ml_cfg.get("regularization", 0.0005)),
        threshold=float(ml_cfg.get("threshold", 0.25)),
        ensemble=bool(ml_cfg.get("ensemble", True)),
        forest_size=int(ml_cfg.get("forest_size", 10)),
        random_state=int(ml_cfg.get("random_state", 7)),
        warmup_target=int(ml_cfg.get("warmup_target", 200)),
        warmup_samples=int(ml_cfg.get("warmup_samples", 25)),
        confidence_stall_limit=int(ml_cfg.get("confidence_stall_limit", 5)),
    )


@st.cache_resource
def ensure_database_schema() -> Path:
    """Create any missing SQLite tables before the dashboard queries data."""

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    missing: List[str] = []
    try:
        with sqlite3.connect(DB_PATH) as connection:
            cursor = connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing = {row[0] for row in cursor.fetchall()}
            missing = [name for name in ALL_TABLES if name not in existing]
            for table_name in missing:
                # Ensures first-run experiences don't crash when tables are absent.
                connection.execute(ALL_TABLES[table_name])
            connection.commit()
    except sqlite3.Error as exc:  # pragma: no cover - defensive guard for UI usage
        LOGGER.exception("Failed to initialise SQLite schema: %s", exc)
        st.error("Failed to initialise the SQLite database. Check application logs for details.")
        raise RuntimeError("SQLite schema initialisation failed") from exc
    if missing:
        LOGGER.info("Created missing SQLite tables: %s", ", ".join(sorted(missing)))
    return DB_PATH


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
            """
            SELECT timestamp, worker, symbol, side, cash_spent, entry_price, exit_price,
                   pnl_percent, pnl_usd, win_loss, reason, metadata_json
            FROM trades
            ORDER BY timestamp DESC
            """,
            conn,
        )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if "metadata_json" in df.columns:
        df["metadata"] = df["metadata_json"].apply(lambda value: json.loads(value or "{}"))
        df.drop(columns=["metadata_json"], inplace=True)
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
        return pd.DataFrame(
            columns=[
                "worker",
                "symbol",
                "status",
                "last_signal",
                "indicators",
                "risk",
                "updated_at",
            ]
        )
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT worker, symbol, status, last_signal, indicators_json, risk_json, updated_at FROM bot_state"
        ).fetchall()
    records: List[Dict[str, object]] = []
    for row in rows:
        indicators = json.loads(row["indicators_json"] or "{}")
        risk = json.loads(row["risk_json"] or "{}")
        ml_warmup = indicators.pop("ml_warming_up", None)
        records.append(
            {
                "worker": row["worker"],
                "symbol": row["symbol"],
                "status": row["status"],
                "last_signal": row["last_signal"],
                "indicators": indicators,
                "risk": risk,
                "updated_at": pd.to_datetime(row["updated_at"], utc=True),
                "ml_warming_up": ml_warmup,
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
def load_trade_events(limit: int = 200) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame(columns=["timestamp", "worker", "symbol", "event", "details"])
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            """
            SELECT timestamp, worker, symbol, event, details_json
            FROM trade_events
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            conn,
            params=(int(limit),),
        )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["details"] = df["details_json"].apply(lambda payload: json.loads(payload or "{}"))
    df.drop(columns=["details_json"], inplace=True)
    return df


@st.cache_data(ttl=5)
def load_market_features(symbol: str, limit: int = 720) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume", "features"]
        )
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
        return pd.DataFrame(
            columns=["timestamp", "symbol", "mode", "precision", "recall", "win_rate", "support"]
        )
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
    latest_equity = (
        equity_df["equity"].iloc[-1]
        if not equity_df.empty
        else config["trading"].get("paper_starting_equity", 0)
    )
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
        st.caption(
            f"Equity values update every trading engine cycle | Broker snapshot: {snapshot_time}"
        )
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


def render_runtime_status(config: Dict, bot_states: pd.DataFrame, ml_service: MLService) -> None:
    st.subheader("Runtime Status")
    trading_mode = str(config.get("trading", {}).get("mode", "paper")).lower()
    mode_label = "Live Trading" if trading_mode == "live" else "Paper Trading"
    badge = "🟢" if trading_mode == "live" else "🧪"
    st.markdown(f"**Mode:** {badge} {mode_label}")
    st.markdown("**Long-only policy:** ✅ Strategies and broker reject shorts by design.")

    symbols = config.get("trading", {}).get("symbols", [])
    if symbols:
        cols = st.columns(min(3, len(symbols)))
        for idx, symbol in enumerate(symbols):
            progress = ml_service.warmup_ratio(symbol)
            text = f"{symbol}: {progress * 100:.0f}%" if progress < 1 else f"{symbol}: ready"
            cols[idx % len(cols)].progress(progress, text)
    else:
        st.caption("No trading symbols configured.")

    definitions = config.get("workers", {}).get("definitions", {})
    research_names = [
        _display_name_for_definition(definition)
        for definition in definitions.values()
        if str(definition.get("module", "")).endswith("researcher.MarketResearchWorker")
    ]
    if research_names:
        st.caption("Research bots streaming features: " + ", ".join(sorted(set(research_names))))

    if not bot_states.empty:
        warming = bot_states[bot_states["ml_warming_up"].notnull()]
        if not warming.empty:
            warming_workers = warming[warming["ml_warming_up"] == True]  # noqa: E712
            if not warming_workers.empty:
                st.info(
                    "ML gating is still warming up for: "
                    + ", ".join(
                        sorted(
                            {f"{row.worker} {row.symbol}" for row in warming_workers.itertuples()}
                        )
                    )
                )


def render_strategy_pulse(config: Dict, bot_states: pd.DataFrame, ml_service: MLService) -> None:
    st.subheader("Strategy Pulse")
    if bot_states.empty:
        st.info("Signals will populate once workers publish state.")
        return

    definitions = config.get("workers", {}).get("definitions", {})
    name_lookup = {
        _display_name_for_definition(definition): definition for definition in definitions.values()
    }

    rows: List[Dict[str, object]] = []
    ordered = bot_states.sort_values("updated_at", ascending=False)
    for row in ordered.itertuples():
        definition = name_lookup.get(row.worker, {})
        module_path = str(definition.get("module", ""))
        warmup_state = getattr(row, "ml_warming_up", None)
        warmup_label = "warming" if warmup_state else "ready"
        indicators: Dict[str, object] = row.indicators or {}
        confidence = indicators.get("ml_confidence")
        threshold = indicators.get("ml_threshold") or definition.get("parameters", {}).get(
            "ml_threshold"
        )
        if confidence is None and row.symbol:
            confidence = ml_service.latest_confidence(row.symbol, worker=row.worker)
        posture = (
            "Research feed"
            if module_path.endswith("researcher.MarketResearchWorker")
            else "Long-only"
        )
        status_text = str(row.status or "unknown").lower()
        status_label = {
            "ready": "Ready to trade",
            "warmup": "Warming up",
            "paused": "Paused",
        }.get(status_text, str(row.status or "unknown").capitalize())
        rows.append(
            {
                "Worker": row.worker,
                "Symbol": row.symbol,
                "Status": status_label,
                "Decision": _human_signal(row.last_signal),
                "ML Confidence": (
                    f"{float(confidence):.3f}" if isinstance(confidence, (int, float)) else "–"
                ),
                "ML Gate": (
                    f"> {float(threshold):.2f}" if isinstance(threshold, (int, float)) else "auto"
                ),
                "Warmup": warmup_label if warmup_state is not None else "n/a",
                "Updated": row.updated_at.strftime("%H:%M:%S"),
                "Posture": posture,
            }
        )
    signal_df = pd.DataFrame(rows)
    display_cols = [
        "Worker",
        "Symbol",
        "Decision",
        "ML Confidence",
        "ML Gate",
        "Warmup",
        "Status",
        "Posture",
        "Updated",
    ]
    st.dataframe(signal_df[display_cols], use_container_width=True)


def render_bot_cards(
    config: Dict,
    bot_states: pd.DataFrame,
    ml_service: MLService,
    trades: pd.DataFrame,
) -> None:
    st.subheader("Strategy Stack")
    if bot_states.empty:
        st.info("Workers will appear once the engine publishes their state.")
        return

    definitions = config.get("workers", {}).get("definitions", {})
    name_lookup = {
        _display_name_for_definition(definition): definition for definition in definitions.values()
    }
    trading_mode = str(config.get("trading", {}).get("mode", "paper")).lower()
    mode_sentence = (
        "live trading with the broker" if trading_mode == "live" else "paper simulation mode"
    )

    def _status_sentence(raw_status: str) -> str:
        status_key = (raw_status or "").lower()
        mapping = {
            "ready": "actively scanning the market and ready to trade",
            "warmup": "collecting price history before taking new positions",
            "paused": "paused via the control panel",
        }
        return mapping.get(status_key, f"operating in {raw_status or 'an unknown state'}")

    def _performance_summary(worker_trades: pd.DataFrame) -> str:
        if worker_trades.empty:
            return "No closed trades yet."
        closed = worker_trades[worker_trades["exit_price"].notna()]
        if closed.empty:
            return "Positions opened, awaiting closes before we can score performance."
        wins = (
            closed[closed["win_loss"].str.lower() == "win"]
            if "win_loss" in closed
            else pd.DataFrame()
        )
        win_rate = (len(wins) / len(closed) * 100) if len(closed) else 0.0
        net_pnl = float(closed.get("pnl_usd", pd.Series(dtype=float)).fillna(0.0).sum())
        avg_return = float(closed.get("pnl_percent", pd.Series(dtype=float)).dropna().mean() or 0.0)
        return (
            "{count} closed trade(s), win rate {win:.0f}% with net P/L ${pnl:,.2f} "
            "and an average return of {avg:.2f}%."
        ).format(count=len(closed), win=win_rate, pnl=net_pnl, avg=avg_return)

    def _last_action(worker_trades: pd.DataFrame) -> str:
        if worker_trades.empty:
            return "No orders have been executed yet."
        latest_trade = worker_trades.sort_values("timestamp").iloc[-1]
        action_time = latest_trade["timestamp"].strftime("%Y-%m-%d %H:%M UTC")
        symbol = latest_trade.get("symbol", "the market")
        if pd.isna(latest_trade.get("exit_price")):
            entry = float(latest_trade.get("entry_price", 0.0))
            return ("Opened a long position on {symbol} at ${price:,.2f} on {time}.").format(
                symbol=symbol, price=entry, time=action_time
            )
        exit_price = float(latest_trade.get("exit_price", 0.0))
        pnl_percent = float(latest_trade.get("pnl_percent", 0.0))
        outcome = str(latest_trade.get("win_loss") or "result").lower()
        outcome_label = (
            "a win" if outcome == "win" else "a loss" if outcome == "loss" else "an outcome"
        )
        return (
            "Closed the latest {symbol} trade at ${price:,.2f} on {time}, "
            "locking {outcome} of {pnl:.2f}%."
        ).format(
            symbol=symbol,
            price=exit_price,
            time=action_time,
            outcome=outcome_label,
            pnl=pnl_percent,
        )

    grouped = bot_states.groupby("worker")
    cols = st.columns(2)
    for idx, (worker, df) in enumerate(grouped):
        card_placeholder = cols[idx % 2].container()
        with card_placeholder:
            st.markdown(f"### {worker}")
            definition = name_lookup.get(worker, {})
            module_path = str(definition.get("module", ""))
            indicators_frame = df.sort_values("updated_at", ascending=False).iloc[0]
            indicators = indicators_frame.get("indicators", {}) or {}
            strategy_brief = indicators.get("strategy_brief") or _strategy_summary(module_path)
            st.caption(strategy_brief)
            latest = indicators_frame
            symbols = sorted({str(sym) for sym in df["symbol"].dropna().unique()})
            primary_symbol = ", ".join(symbols) if symbols else "no assigned markets"
            status = str(latest.get("status", "unknown"))
            status_sentence = _status_sentence(status)
            st.markdown(
                f"{worker} is {status_sentence} while monitoring {primary_symbol} in {mode_sentence}."
            )
            last_signal = latest.get("last_signal")
            ml_confidence = indicators.get("ml_confidence")
            threshold = (
                indicators.get("ml_threshold")
                or definition.get("parameters", {}).get("ml_threshold")
                or ml_service.default_threshold
            )
            decision_text = _human_signal(last_signal)
            st.markdown(f"Latest signal assessment: {decision_text}")
            if module_path.endswith("researcher.MarketResearchWorker"):
                feature_snapshot = indicators.get("features")
                feature_count = len(feature_snapshot) if isinstance(feature_snapshot, dict) else 0
                ml_ready = bool(indicators.get("ml_ready"))
                st.markdown(
                    "**Research stream:** Captured {count} features this cycle; ML gate {state}.".format(
                        count=feature_count,
                        state="ready" if ml_ready else "is warming up",
                    )
                )
                if ml_confidence is not None:
                    st.markdown(
                        f"**ML broadcast:** confidence {float(ml_confidence):.3f} vs gate {float(threshold):.2f}."
                    )
            else:
                long_only = bool(indicators.get("long_only", False))
                posture_sentence = (
                    "Long-only guard enforced; shorts are blocked at worker and broker layers."
                    if long_only
                    else "Mixed positioning configured."
                )
                st.markdown(posture_sentence)
                if ml_confidence is None and latest.get("symbol"):
                    ml_confidence = ml_service.latest_confidence(
                        latest.get("symbol"), worker=worker
                    )
                if ml_confidence is not None:
                    st.markdown(
                        (
                            "ML gate is reading {confidence:.3f} versus a required {threshold:.2f}; "
                            "the strategy {state}."
                        ).format(
                            confidence=float(ml_confidence),
                            threshold=float(threshold),
                            state=(
                                "is cleared to trade"
                                if float(ml_confidence) >= float(threshold)
                                else "is waiting for more confirmation"
                            ),
                        )
                    )
                else:
                    st.markdown("Waiting for fresh ML features before taking new trades.")
                risk = latest.get("risk", {}) or {}
                if risk:
                    position_pct = float(risk.get("position_size_pct", 100.0))
                    stop_loss = float(risk.get("stop_loss_pct", 0.0))
                    take_profit = float(risk.get("take_profit_pct", 0.0))
                    trailing = float(risk.get("trailing_stop_pct", 0.0))
                    st.markdown(
                        (
                            "Risk guard rails: targeting {alloc:.0f}% of allocated capital with a "
                            "{sl:.2f}% protective stop, {tp:.2f}% profit objective, and {tr:.2f}% trailing buffer."
                        ).format(
                            alloc=position_pct,
                            sl=stop_loss,
                            tp=take_profit,
                            tr=trailing,
                        )
                    )
                worker_trades = (
                    trades[trades["worker"] == worker] if not trades.empty else pd.DataFrame()
                )
                st.markdown(f"Most recent action: {_last_action(worker_trades)}")
                st.markdown(f"Performance snapshot: {_performance_summary(worker_trades)}")
            last_updated = latest.get("updated_at")
            if isinstance(last_updated, pd.Timestamp):
                timestamp_text = (
                    last_updated.tz_convert("UTC") if last_updated.tzinfo else last_updated
                )
                st.caption(f"Last update received at {timestamp_text.strftime('%H:%M:%S UTC')}")
            else:
                st.caption("Awaiting the first status update from this worker.")


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
    entries = trades[
        (trades["symbol"] == symbol)
        & trades["exit_price"].isna()
        & (trades["side"].str.lower() == "buy")
    ]
    exits = trades[
        (trades["symbol"] == symbol)
        & trades["exit_price"].notna()
        & (trades["side"].str.lower() == "sell")
    ]
    if not entries.empty:
        fig.add_trace(
            go.Scatter(
                x=entries["timestamp"],
                y=entries["entry_price"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=12, color="#00ffb3"),
                name="Long entries",
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
                marker=dict(symbol="circle", size=10, color="#ff4d4d"),
                name="Exits",
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
    st.caption(
        "Markers show ▲ long entries and ● exits. The lower panel charts ML confidence feeding each strategy."
    )
    if not confidence_df.empty:
        latest_conf = confidence_df.sort_values("timestamp").iloc[-1]
        st.markdown(
            "**Latest ML reading:** {worker} confidence {conf:.3f} vs gate {thr:.2f}.".format(
                worker=latest_conf["worker"],
                conf=float(latest_conf["confidence"]),
                thr=float(latest_conf.get("threshold", ml_service.default_threshold)),
            )
        )

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


def render_ml_debug_panel(config: Dict, ml_service: MLService, bot_states: pd.DataFrame) -> None:
    st.subheader("ML Diagnostics")
    st.caption(
        f"Ensemble backend: {ml_service.ensemble_backend} | requested: {ml_service.ensemble_requested}"
    )
    with st.expander("Inspect live ML inputs", expanded=False):
        st.caption(
            "Use this panel to print the latest engineered features and confidence levels feeding each worker."
        )
        if st.button("Print ML snapshot", key="ml_debug_snapshot"):
            symbols = config.get("trading", {}).get("symbols", [])
            workers: List[str] = []
            if not bot_states.empty:
                workers = sorted(bot_states["worker"].unique())
            if not symbols:
                st.info("No symbols configured for trading.")
            for symbol in symbols:
                features = ml_service.latest_features(symbol) or {}
                st.markdown(f"**{symbol}**")
                if features:
                    st.json({"features": features})
                else:
                    st.info("No features observed yet for this symbol.")
                if workers:
                    confidence_payload = {
                        worker: round(
                            ml_service.latest_confidence(symbol, worker=worker),
                            4,
                        )
                        for worker in workers
                    }
                    st.json({"confidence": confidence_payload})
                else:
                    st.caption(
                        "Workers have not published state yet; confidence history unavailable."
                    )


def render_trade_logs(trades: pd.DataFrame, events: pd.DataFrame) -> None:
    st.subheader("Trade Logs")
    tab_exec, tab_events = st.tabs(["Executions", "Events"])

    with tab_exec:
        if trades.empty:
            st.info("Trades will appear once execution begins.")
        else:
            workers = ["All"] + sorted(trades["worker"].unique())
            worker_filter_key = "trade_logs_worker_filter"
            st.selectbox("Filter by strategy", workers, key=worker_filter_key)
            selected_worker = st.session_state.get(worker_filter_key, workers[0])
            filtered = (
                trades if selected_worker == "All" else trades[trades["worker"] == selected_worker]
            )
            display_cols = [
                "timestamp",
                "worker",
                "symbol",
                "side",
                "reason",
                "cash_spent",
                "entry_price",
                "exit_price",
                "pnl_usd",
                "pnl_percent",
                "win_loss",
            ]
            present_cols = [col for col in display_cols if col in filtered.columns]
            st.dataframe(filtered[present_cols], use_container_width=True)

    with tab_events:
        if events.empty:
            st.info("Trade lifecycle events (stops, trailing updates) will appear here.")
        else:
            st.dataframe(events, use_container_width=True)


def render_risk_controls(config: Dict, ml_service: MLService, symbol: str) -> None:
    st.subheader("Risk & Deployment Controls")
    trading_cfg = config.get("trading", {})
    risk_cfg = config.get("risk", {})
    worker_cfg = config.get("workers", {}).get("definitions", {})
    control_flags = load_control_flags()

    with st.form("global_risk_form"):
        st.write("### Global Risk")
        max_drawdown_key = "global_risk_max_drawdown"
        st.slider(
            "Max drawdown %",
            min_value=5.0,
            max_value=50.0,
            step=0.5,
            value=float(risk_cfg.get("max_drawdown_percent", 25.0)),
            key=max_drawdown_key,
        )
        daily_limit_key = "global_risk_daily_loss_limit"
        st.slider(
            "Daily loss limit %",
            min_value=1.0,
            max_value=20.0,
            step=0.5,
            value=float(risk_cfg.get("daily_loss_limit_percent", 5.0)),
            key=daily_limit_key,
        )
        allocation_key = "global_risk_allocation"
        st.slider(
            "Equity allocation per trade %",
            min_value=1.0,
            max_value=25.0,
            step=0.5,
            value=float(trading_cfg.get("equity_allocation_percent", 6.0)),
            key=allocation_key,
        )
        max_positions_key = "global_risk_max_positions"
        st.slider(
            "Maximum concurrent positions",
            min_value=1,
            max_value=12,
            value=int(trading_cfg.get("max_open_positions", 4)),
            key=max_positions_key,
        )
        submitted = st.form_submit_button("Update global risk")
        if submitted:
            max_drawdown = float(
                st.session_state.get(
                    max_drawdown_key,
                    float(risk_cfg.get("max_drawdown_percent", 25.0)),
                )
            )
            daily_limit = float(
                st.session_state.get(
                    daily_limit_key,
                    float(risk_cfg.get("daily_loss_limit_percent", 5.0)),
                )
            )
            allocation = float(
                st.session_state.get(
                    allocation_key,
                    float(trading_cfg.get("equity_allocation_percent", 6.0)),
                )
            )
            max_positions = int(
                st.session_state.get(
                    max_positions_key,
                    int(trading_cfg.get("max_open_positions", 4)),
                )
            )
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
            symbol_key = f"symbols_{worker_key}"
            st.multiselect(
                "Symbols",
                options=config["trading"].get("symbols", symbols),
                default=symbols,
                key=symbol_key,
            )
            position_size_key = f"pos_{worker_key}"
            st.slider(
                "Position size % of allocation",
                min_value=10.0,
                max_value=200.0,
                step=5.0,
                value=float(risk.get("position_size_pct", 100.0)),
                key=position_size_key,
            )
            leverage_key = f"lev_{worker_key}"
            st.slider(
                "Leverage",
                min_value=1.0,
                max_value=5.0,
                step=0.1,
                value=float(risk.get("leverage", 1.0)),
                key=leverage_key,
            )
            stop_loss_key = f"sl_{worker_key}"
            st.slider(
                "Stop loss %",
                min_value=0.5,
                max_value=5.0,
                step=0.1,
                value=float(risk.get("stop_loss_pct", 1.0)),
                key=stop_loss_key,
            )
            take_profit_key = f"tp_{worker_key}"
            st.slider(
                "Take profit %",
                min_value=0.5,
                max_value=6.0,
                step=0.1,
                value=float(risk.get("take_profit_pct", 2.0)),
                key=take_profit_key,
            )
            trailing_key = f"trail_{worker_key}"
            st.slider(
                "Trailing stop %",
                min_value=0.0,
                max_value=5.0,
                step=0.1,
                value=float(risk.get("trailing_stop_pct", 0.0)),
                key=trailing_key,
            )
            if st.button("Save", key=f"save_{worker_key}"):
                new_symbols = st.session_state.get(symbol_key, symbols)
                position_size = float(
                    st.session_state.get(
                        position_size_key, float(risk.get("position_size_pct", 100.0))
                    )
                )
                leverage = float(
                    st.session_state.get(leverage_key, float(risk.get("leverage", 1.0)))
                )
                stop_loss = float(
                    st.session_state.get(stop_loss_key, float(risk.get("stop_loss_pct", 1.0)))
                )
                take_profit = float(
                    st.session_state.get(take_profit_key, float(risk.get("take_profit_pct", 2.0)))
                )
                trailing = float(
                    st.session_state.get(trailing_key, float(risk.get("trailing_stop_pct", 0.0)))
                )
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
            worker_label = _display_name_for_definition(definition)
            flag_key = f"ml::{worker_label}"
            enabled = control_flags.get(flag_key, "on").lower() not in {
                "off",
                "false",
                "0",
                "disabled",
            }
            toggle_key = f"ml_gate_{worker_key}"
            initial_toggle = st.session_state.get(toggle_key, enabled)
            cols[idx % len(cols)].toggle(
                f"{worker_label} gating",
                value=initial_toggle,
                key=toggle_key,
            )
            toggle_state = st.session_state.get(toggle_key, initial_toggle)
            if toggle_state != enabled:
                set_control_flag(flag_key, "on" if toggle_state else "off")
                st.success(f"Updated ML gating for {worker_label}.")
    else:
        st.info("No trading workers configured for ML gating.")

    st.write("### ML Validation & Backtests")
    metrics_df = load_ml_metrics()
    validation_cols = st.columns(3)
    symbol_options = config.get("trading", {}).get("symbols", [symbol])
    default_index = symbol_options.index(symbol) if symbol in symbol_options else 0
    validation_cols[0].selectbox(
        "Symbol",
        options=symbol_options,
        index=default_index,
        key="ml_metric_symbol",
    )
    selected_symbol = st.session_state.get(
        "ml_metric_symbol",
        symbol_options[default_index] if symbol_options else symbol,
    )
    if validation_cols[1].button("Run backtest", key="run_ml_backtest"):
        results = ml_service.run_backtest(selected_symbol)
        load_ml_metrics.clear()
        st.success(
            (
                "Backtest complete – precision: {precision:.2f}, recall: {recall:.2f}, "
                "win rate: {win_rate:.2f}, accuracy: {accuracy:.2f}, F1: {f1_score:.2f}, "
                "trades evaluated: {trades:d}"
            ).format(**results)
        )
    if not metrics_df.empty:
        filtered = metrics_df[metrics_df["symbol"] == selected_symbol]
        if filtered.empty:
            st.info(
                "No metrics recorded yet for this symbol. Run a backtest to populate statistics."
            )
        else:
            st.dataframe(filtered.head(10), use_container_width=True)
    else:
        st.info("Metrics will appear once backtests or live evaluations are recorded.")


def render_sidebar(config: Dict) -> str:
    st.sidebar.title("Live Controls")
    refresh_interval = int(config.get("dashboard", {}).get("refresh_interval_seconds", 5))
    _auto_refresh_script(refresh_interval)

    control_flags = load_control_flags()

    trading_mode_default = config["trading"].get("mode", "paper")
    initial_mode = st.session_state.get("trading_mode_radio", trading_mode_default)
    if initial_mode not in {"paper", "live"}:
        initial_mode = trading_mode_default
    st.sidebar.radio(
        "Trading Mode",
        options=["paper", "live"],
        index=0 if initial_mode == "paper" else 1,
        key="trading_mode_radio",
    )
    trading_mode = st.session_state.get("trading_mode_radio", initial_mode)
    if trading_mode != config["trading"].get("mode"):
        config["trading"]["mode"] = trading_mode
        config["trading"]["paper_trading"] = trading_mode == "paper"
        save_config(config)
        st.sidebar.success("Trading mode updated. Restart engine to take effect.")

    kill_switch = control_flags.get("kill_switch", "false") == "true"
    kill_toggle_key = "kill_switch_toggle"
    kill_toggle_default = st.session_state.get(kill_toggle_key, kill_switch)
    st.sidebar.toggle(
        "Global Kill Switch",
        value=kill_toggle_default,
        key=kill_toggle_key,
    )
    kill_toggle = st.session_state.get(kill_toggle_key, kill_toggle_default)
    if kill_toggle != kill_switch:
        set_control_flag("kill_switch", "true" if kill_toggle else "false")
        st.sidebar.success("Kill switch state updated.")

    st.sidebar.info(
        "Long-only mode: sell orders automatically close longs; shorts are "
        "blocked at the strategy and broker layers."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Per-Bot Overrides")
    definitions = config.get("workers", {}).get("definitions", {})
    for worker_key, definition in definitions.items():
        module_path = definition.get("module", worker_key)
        if module_path.endswith("researcher.MarketResearchWorker"):
            continue
        worker_label = _display_name_for_definition(definition)
        flag_key = f"bot::{worker_label}"
        paused = control_flags.get(flag_key, "active") in {"paused", "disabled"}
        toggle_key = f"bot_toggle_{worker_key}"
        initial_toggle = st.session_state.get(toggle_key, not paused)
        st.sidebar.toggle(
            f"{worker_label}",
            value=initial_toggle,
            key=toggle_key,
        )
        toggle_state = st.session_state.get(toggle_key, initial_toggle)
        if toggle_state == paused:
            set_control_flag(flag_key, "active" if toggle_state else "paused")
            st.sidebar.success(f"Updated {worker_label} control flag.")

    st.sidebar.markdown("---")
    sidebar_symbol_key = "sidebar_symbol_select"
    st.sidebar.selectbox(
        "Symbol",
        options=config["trading"].get("symbols", ["BTC/USD"]),
        index=0,
        key=sidebar_symbol_key,
    )
    symbol_options = config["trading"].get("symbols", ["BTC/USD"])
    default_symbol = symbol_options[0] if symbol_options else "BTC/USD"
    return st.session_state.get(sidebar_symbol_key, default_symbol)


def main() -> None:
    config = load_config()
    ml_service = init_ml_service(config)
    st.title("Quant Operations Terminal")
    symbol = render_sidebar(config)
    bot_states = load_bot_states()
    trades = load_trades()
    equity = load_equity_curve()
    account_snapshot = load_account_snapshot()
    events = load_trade_events()

    render_account_overview(config, equity, trades, account_snapshot)
    render_equity_curve(equity)
    render_runtime_status(config, bot_states, ml_service)
    render_strategy_pulse(config, bot_states, ml_service)
    render_bot_cards(config, bot_states, ml_service, trades)
    render_market_view(symbol, trades, ml_service)
    render_ml_debug_panel(config, ml_service, bot_states)
    render_trade_logs(trades, events)
    render_risk_controls(config, ml_service, symbol)


if __name__ == "__main__":
    main()
