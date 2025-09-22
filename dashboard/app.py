"""Streamlit quant control center for AI day trading runtime."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from desk.services.risk import PositionSizingResult, RiskEngine

from .analytics import drawdown_series
from .components import apply_template, candles_with_trades, equity_with_drawdown, pnl_histogram
from .data_io import (
    CONFIG_PATH,
    DB_LIVE,
    LOG_DIR,
    DataHealth,
    database_health,
    load_config,
    load_equity,
    load_logs,
    load_ml_scores,
    load_positions,
    load_trades,
    save_config,
    seed_demo_data,
)
from data_feeds import KrakenMarketFeed

LOGGER = logging.getLogger("dashboard.quant_app")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)

MARKET_SYMBOLS: List[str] = ["BTC/USD", "ETH/USD", "SOL/USD", "MATIC/USD"]
REFRESH_SECONDS = 12
ORDERBOOK_DEPTH = 10


def _load_styles() -> str:
    style_path = Path(__file__).with_name("style.css")
    if style_path.exists():
        return style_path.read_text(encoding="utf-8")
    return ""


def inject_styles() -> None:
    css = _load_styles()
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_market_feed(symbols: Iterable[str]) -> KrakenMarketFeed:
    feed = KrakenMarketFeed(symbols)
    feed.start()
    return feed


@st.cache_data(show_spinner=False, ttl=REFRESH_SECONDS)
def load_all_data(version: int) -> Dict[str, pd.DataFrame]:
    trades = load_trades(DB_LIVE)
    equity = load_equity(DB_LIVE)
    positions = load_positions(DB_LIVE)
    ml_scores = load_ml_scores(DB_LIVE)
    logs = load_logs(LOG_DIR)
    return {
        "trades": trades,
        "equity": drawdown_series(equity) if not equity.empty else equity,
        "positions": positions,
        "ml_scores": ml_scores,
        "logs": logs,
    }


@st.cache_data(show_spinner=False, ttl=REFRESH_SECONDS)
def fetch_config_snapshot() -> Dict[str, Any]:
    _, raw = load_config(CONFIG_PATH)
    return json.loads(json.dumps(raw))  # deep copy for mutation safety


@dataclass
class MarketSnapshot:
    symbol: str
    candles: pd.DataFrame
    last_price: Optional[float]
    change_pct: Optional[float]
    updated_at: Optional[datetime]
    source: str


def schedule_autorefresh(interval_seconds: int = REFRESH_SECONDS) -> None:
    """Trigger a Streamlit rerun periodically to keep the UI reactive."""

    st_autorefresh(interval=interval_seconds * 1000, limit=None, key="global_autorefresh")


def init_session_state(config: Dict[str, Any]) -> None:
    state = st.session_state
    state.setdefault("data_version", 0)
    state.setdefault("runtime_status", "live")
    state.setdefault("control_log", [])
    state.setdefault("trade_console", [])
    state.setdefault("seen_trades", set())
    state.setdefault("allocation_overrides", {})
    risk_cfg = config.get("risk", {})
    state.setdefault(
        "risk_overrides",
        {
            "base_risk_pct": float(risk_cfg.get("base_risk_pct", 0.015)),
            "max_concurrent": int(risk_cfg.get("max_concurrent", 5)),
            "max_concurrent_risk_pct": float(risk_cfg.get("max_concurrent_risk_pct", 0.05)),
            "equity_floor": float(risk_cfg.get("equity_floor", 0.0) or 0.0),
            "risk_per_trade_pct": float(risk_cfg.get("risk_per_trade_pct", 0.015)),
            "min_trade_notional": float(risk_cfg.get("min_trade_notional", 10.0)),
        },
    )
    state["config_raw"] = config
    state.setdefault("market_snapshots", {})


def build_risk_engine(config: Dict[str, Any], equity: float) -> RiskEngine:
    risk_cfg = config.get("risk", {})
    engine = RiskEngine(
        daily_dd=risk_cfg.get("daily_dd"),
        weekly_dd=risk_cfg.get("weekly_dd"),
        default_stop_pct=float(risk_cfg.get("stop_loss_pct", 0.02)),
        max_concurrent=int(risk_cfg.get("max_concurrent", 5)),
        halt_on_dd=bool(risk_cfg.get("halt_on_dd", True)),
        trapdoor_pct=float(risk_cfg.get("trapdoor_pct", 0.02)),
        max_position_value=
            float(risk_cfg.get("max_position_value")) if risk_cfg.get("max_position_value") else None,
        equity_floor=float(risk_cfg.get("equity_floor", 0.0) or 0.0) or None,
        risk_per_trade_pct=float(risk_cfg.get("risk_per_trade_pct", 0.015)),
        max_risk_per_trade_pct=
            float(risk_cfg.get("max_risk_per_trade_pct")) if risk_cfg.get("max_risk_per_trade_pct") else None,
        min_notional=float(risk_cfg.get("min_trade_notional", 10.0)),
        base_risk_pct=float(risk_cfg.get("base_risk_pct", 0.015)),
        max_concurrent_risk_pct=float(risk_cfg.get("max_concurrent_risk_pct", 0.05)),
    )
    engine.initialise(equity)
    return engine


def compute_market_snapshot(feed: KrakenMarketFeed, symbol: str) -> MarketSnapshot:
    try:
        candles = feed.get_candles(symbol, limit=180)
        source = "websocket" if symbol in feed.latest_messages else "rest"
    except Exception as exc:  # pragma: no cover - network guard
        LOGGER.warning("Falling back to REST for %s: %s", symbol, exc)
        candles = pd.DataFrame()
        source = "rest"
    last_price: Optional[float] = None
    change_pct: Optional[float] = None
    updated_at: Optional[datetime] = None
    if not candles.empty:
        candles = candles.sort_values("time").rename(columns={"time": "ts"})
        last_price = float(candles["close"].iloc[-1])
        previous = candles["close"].iloc[-2] if len(candles) > 1 else candles["close"].iloc[-1]
        change_pct = float(((last_price - previous) / previous) * 100) if previous else None
        updated_at = pd.to_datetime(candles["ts"].iloc[-1]).to_pydatetime()
    return MarketSnapshot(
        symbol=symbol,
        candles=candles,
        last_price=last_price,
        change_pct=change_pct,
        updated_at=updated_at,
        source=source,
    )


def fetch_order_book(symbol: str, depth: int = ORDERBOOK_DEPTH) -> Dict[str, pd.DataFrame]:
    pair = symbol.replace("/", "")
    try:
        response = requests.get(
            "https://api.kraken.com/0/public/Depth",
            params={"pair": pair, "count": depth},
            timeout=5,
        )
        response.raise_for_status()
        data = response.json().get("result", {})
        if not data:
            raise ValueError("Empty response")
        key = next(iter(data.keys()))
        bids = pd.DataFrame(data[key].get("bids", [])[:depth], columns=["price", "qty", "timestamp"])
        asks = pd.DataFrame(data[key].get("asks", [])[:depth], columns=["price", "qty", "timestamp"])
        for frame in (bids, asks):
            if not frame.empty:
                frame["price"] = frame["price"].astype(float)
                frame["qty"] = frame["qty"].astype(float)
        return {"bids": bids, "asks": asks}
    except Exception as exc:  # pragma: no cover - network guard
        LOGGER.warning("Order book fetch failed for %s: %s", symbol, exc)
        return {"bids": pd.DataFrame(), "asks": pd.DataFrame()}


def render_header(equity_value: float, risk_engine: RiskEngine, health: Iterable[DataHealth]) -> None:
    st.markdown(
        """
        <div class="top-bar glass">
            <div class="title-block">
                <h2>🚀 Aurora Quant Command</h2>
                <p>Live Kraken connectivity · ML assisted execution · Risk governed sizing</p>
            </div>
            <div class="stat-block">
                <div class="stat">
                    <span>Equity</span>
                    <strong>${equity_value:,.0f}</strong>
                </div>
                <div class="stat">
                    <span>Risk Budget</span>
                    <strong>{risk_engine.base_risk_pct * 100:.2f}%</strong>
                </div>
                <div class="stat">
                    <span>Status</span>
                    <strong>{'HALTED' if risk_engine.halted else 'LIVE'}</strong>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(len(list(health)) or 1)
    for col, item in zip(cols, health):
        with col:
            badge = "ok" if item.status == "ok" else "warn"
            st.markdown(
                f"<div class='health-card {badge}'>"
                f"<span>{item.name}</span>"
                f"<strong>{item.status.upper()}</strong>"
                f"<small>{item.detail or 'Healthy'}</small>"
                f"</div>",
                unsafe_allow_html=True,
            )


def render_trade_alerts(trades: pd.DataFrame) -> None:
    if trades.empty:
        return
    alerts: List[str] = []
    seen: set[str] = st.session_state.get("seen_trades", set())
    for _, trade in trades.tail(8).iterrows():
        trade_id = str(trade.get("trade_id"))
        if trade_id in seen:
            continue
        action = "TRADE CLOSED" if pd.notna(trade.get("closed_at")) else "TRADE OPENED"
        pnl = trade.get("pnl", 0.0) or 0.0
        pnl_colour = "positive" if pnl >= 0 else "negative"
        size = trade.get("qty", 0.0)
        worker = trade.get("worker", "n/a")
        equity_pct = trade.get("equity_pct")
        pct_text = f" · {equity_pct:.2%} equity" if isinstance(equity_pct, (int, float)) else ""
        alerts.append(
            f"<div class='trade-alert glass {pnl_colour}'>"
            f"<strong>{action}</strong> · {trade.get('symbol', 'UNK')} @ {trade.get('entry', 0):,.2f}"\
            f" · size {size:,.4f}{pct_text}<br/><span>{worker} · PnL {pnl:,.2f}</span></div>"
        )
        seen.add(trade_id)
        console = st.session_state.get("trade_console", [])
        console.append(
            f"{datetime.utcnow().isoformat(timespec='seconds')} | {action} | "
            f"{trade.get('symbol')} size={size:.4f} pnl={pnl:,.2f} worker={worker}"
        )
        st.session_state["trade_console"] = console[-20:]
    st.session_state["seen_trades"] = seen
    if alerts:
        st.markdown("""<div class='trade-alerts'>%s</div>""" % "".join(alerts), unsafe_allow_html=True)
        st.code("\n".join(st.session_state.get("trade_console", [])[-6:]), language="text")


def render_portfolio_overview(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    positions: pd.DataFrame,
    risk_engine: RiskEngine,
) -> None:
    render_trade_alerts(trades)
    today = date.today()
    equity_value = float(equity["balance"].iloc[-1]) if not equity.empty else 0.0
    realized_pnl = float(trades["pnl"].sum()) if not trades.empty else 0.0
    daily_trades = trades[pd.to_datetime(trades["opened_at"]).dt.date == today] if not trades.empty else pd.DataFrame()
    daily_pnl = float(daily_trades["pnl"].sum()) if not daily_trades.empty else 0.0
    open_positions = positions if not positions.empty else pd.DataFrame()

    cols = st.columns(4)
    cols[0].markdown(
        (
            "<div class='kpi-card glass'><span>Total Equity</span>"
            f"<strong>${equity_value:,.0f}</strong></div>"
        ),
        unsafe_allow_html=True,
    )
    cols[1].markdown(
        (
            "<div class='kpi-card glass'><span>Realised PnL</span>"
            f"<strong class={'positive' if realized_pnl >= 0 else 'negative'}>"
            f"${realized_pnl:,.0f}</strong></div>"
        ),
        unsafe_allow_html=True,
    )
    cols[2].markdown(
        (
            "<div class='kpi-card glass'><span>Daily PnL</span>"
            f"<strong class={'positive' if daily_pnl >= 0 else 'negative'}>"
            f"${daily_pnl:,.0f}</strong></div>"
        ),
        unsafe_allow_html=True,
    )
    cols[3].markdown(
        f"<div class='kpi-card glass'><span>Open Trades</span><strong>{len(open_positions)}</strong></div>",
        unsafe_allow_html=True,
    )

    st.plotly_chart(equity_with_drawdown(equity), use_container_width=True)

    perf_cols = st.columns(2)
    with perf_cols[0]:
        st.plotly_chart(pnl_histogram(trades), use_container_width=True)
    with perf_cols[1]:
        if not trades.empty:
            win_rate = trades[trades["pnl"] > 0].shape[0] / trades.shape[0]
        else:
            win_rate = np.nan
        win_rate_text = f"{win_rate:.2%}" if not np.isnan(win_rate) else "n/a"
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=(1 - risk_engine.total_allocated_risk_pct()) * 100,
                title={"text": "Remaining Risk Budget (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#38bdf8"},
                    "steps": [
                        {"range": [0, 25], "color": "rgba(248,113,113,0.45)"},
                        {"range": [25, 60], "color": "rgba(251,191,36,0.35)"},
                        {"range": [60, 100], "color": "rgba(34,197,94,0.35)"},
                    ],
                },
            )
        )
        st.plotly_chart(apply_template(gauge), use_container_width=True)
        st.markdown(
            f"<div class='callout'>Win rate: {win_rate_text} · Base risk per trade:"
            f" {risk_engine.base_risk_pct:.2%}</div>",
            unsafe_allow_html=True,
        )

    with st.expander("Open positions", expanded=not open_positions.empty):
        if open_positions.empty:
            st.info("No active positions in the ledger.")
        else:
            view = open_positions.copy()
            view["opened_at"] = pd.to_datetime(view["opened_at"])
            st.dataframe(view, use_container_width=True)


def render_risk_controls(risk_engine: RiskEngine, trades: pd.DataFrame) -> None:
    overrides = st.session_state["risk_overrides"].copy()
    cols = st.columns(5)
    with cols[0]:
        overrides["base_risk_pct"] = st.slider(
            "Base risk %",
            min_value=0.001,
            max_value=0.05,
            value=float(overrides["base_risk_pct"]),
            step=0.001,
            help="Percentage of equity allocated per trade before guard rails.",
        )
    with cols[1]:
        overrides["risk_per_trade_pct"] = st.slider(
            "Max risk %",
            min_value=overrides["base_risk_pct"],
            max_value=0.15,
            value=float(overrides["risk_per_trade_pct"]),
            step=0.001,
        )
    with cols[2]:
        overrides["max_concurrent"] = st.number_input(
            "Max concurrent positions",
            min_value=1,
            max_value=20,
            value=int(overrides["max_concurrent"]),
        )
    with cols[3]:
        overrides["equity_floor"] = st.number_input(
            "Equity floor", min_value=0.0, value=float(overrides.get("equity_floor", 0.0)), step=1000.0
        )
    with cols[4]:
        overrides["min_trade_notional"] = st.number_input(
            "Min trade notional", min_value=0.0, value=float(overrides.get("min_trade_notional", 10.0)), step=1.0
        )
    caps = st.columns(1)
    with caps[0]:
        overrides["max_concurrent_risk_pct"] = st.slider(
            "Portfolio risk cap %",
            min_value=overrides["base_risk_pct"],
            max_value=0.5,
            value=float(overrides["max_concurrent_risk_pct"]),
            step=0.005,
        )

    st.session_state["risk_overrides"] = overrides

    save_col, warn_col = st.columns([1, 2])
    with save_col:
        if st.button("Persist risk config", type="primary", use_container_width=True):
            config = fetch_config_snapshot()
            risk_cfg = config.setdefault("risk", {})
            risk_cfg.update(overrides)
            save_config(config, CONFIG_PATH)
            fetch_config_snapshot.clear()
            st.success("Risk configuration updated. Runtime will pick up changes on next cycle.")
            st.session_state["data_version"] += 1
    with warn_col:
        if risk_engine.halted:
            st.error("Trading halted by risk engine. Review drawdown and restart once safe.")
        elif (
            risk_engine.current_equity
            and overrides.get("equity_floor")
            and risk_engine.current_equity < overrides["equity_floor"]
        ):
            st.warning("Equity floor breached – trades will be rejected until deposits restore balance.")

    st.markdown("### Position sizing playground")
    market = st.session_state.get("market_snapshots", {})
    default_symbol = MARKET_SYMBOLS[0]
    default_price = market.get(default_symbol, {}).get("last_price", 0.0)
    with st.form("position_sizer"):
        c1, c2, c3 = st.columns(3)
        symbol = c1.selectbox("Symbol", MARKET_SYMBOLS, index=0)
        side = c2.selectbox("Side", ["BUY", "SELL"], index=0)
        symbol_price = market.get(symbol, {}).get("last_price", default_price)
        price = c3.number_input(
            "Price",
            min_value=0.0,
            value=float(symbol_price or 0.0),
            format="%0.2f",
            key=f"sizer_price_{symbol}",
        )
        c4, c5, c6 = st.columns(3)
        stop_loss = c4.number_input("Stop loss", min_value=0.0, value=price * (0.99 if price else 0.0), format="%0.2f")
        allocation = c5.slider("Equity allocation", min_value=0.05, max_value=1.0, value=0.25, step=0.05)
        min_qty = c6.number_input("Minimum quantity", min_value=0.0, value=0.0, step=0.001)
        c7, c8, c9 = st.columns(3)
        precision = int(c7.number_input("Precision", min_value=0, max_value=8, value=3))
        min_notional = c8.number_input(
            "Exchange min notional",
            min_value=0.0,
            value=float(overrides["min_trade_notional"]),
            step=1.0,
        )
        prefer_round_down = c9.toggle("Prefer round down", value=True)
        submitted = st.form_submit_button("Run sizing", use_container_width=True)

    if submitted:
        sizing = risk_engine.get_position_size(
            symbol=symbol,
            price=price,
            side=side,
            stop_loss=stop_loss,
            allocation=allocation,
            minimum_qty=min_qty,
            precision=precision,
            min_notional=min_notional,
            prefer_round_down=prefer_round_down,
        )
        render_sizing_result(sizing)

    st.markdown("### Recent rejected/adjusted orders")
    rejected = (
        trades[trades.get("note", "").str.contains("reject", case=False, na=False)]
        if not trades.empty
        else pd.DataFrame()
    )
    if rejected.empty:
        st.info("No rejected orders recorded in the current window.")
    else:
        st.dataframe(rejected.tail(10), use_container_width=True)


def render_sizing_result(result: PositionSizingResult) -> None:
    cols = st.columns(4)
    cols[0].markdown(
        f"<div class='kpi-card glass'><span>Quantity</span><strong>{result.quantity:,.6f}</strong></div>",
        unsafe_allow_html=True,
    )
    cols[1].markdown(
        f"<div class='kpi-card glass'><span>Notional</span><strong>${result.notional:,.2f}</strong></div>",
        unsafe_allow_html=True,
    )
    cols[2].markdown(
        f"<div class='kpi-card glass'><span>Risk Amount</span><strong>${result.risk_amount:,.2f}</strong></div>",
        unsafe_allow_html=True,
    )
    cols[3].markdown(
        f"<div class='kpi-card glass'><span>Risk %</span><strong>{result.risk_pct * 100:.2f}%</strong></div>",
        unsafe_allow_html=True,
    )
    annotations = []
    if result.adjusted_for_minimum():
        annotations.append("Adjusted to satisfy exchange minimums.")
    if result.concurrency_adjusted:
        annotations.append("Allocation capped by portfolio concurrency budget.")
    if annotations:
        st.warning(" ".join(annotations))


def render_trade_blotter(trades: pd.DataFrame) -> None:
    st.markdown("### Filters")
    col1, col2, col3 = st.columns(3)
    symbols = sorted(trades["symbol"].dropna().unique()) if not trades.empty else []
    workers = sorted(trades["worker"].dropna().unique()) if not trades.empty else []
    status = sorted(trades["status"].dropna().unique()) if not trades.empty else []
    selected_symbol = col1.multiselect("Symbols", symbols, default=symbols)
    selected_worker = col2.multiselect("Workers", workers, default=workers)
    selected_status = col3.multiselect("Status", status, default=status)

    filtered = trades.copy()
    if selected_symbol:
        filtered = filtered[filtered["symbol"].isin(selected_symbol)]
    if selected_worker:
        filtered = filtered[filtered["worker"].isin(selected_worker)]
    if selected_status:
        filtered = filtered[filtered["status"].isin(selected_status)]

    if filtered.empty:
        st.warning("No trades match the selected filters.")
        return

    filtered["opened_at"] = pd.to_datetime(filtered["opened_at"])
    filtered["closed_at"] = pd.to_datetime(filtered["closed_at"])
    st.dataframe(filtered.sort_values("opened_at", ascending=False), use_container_width=True)

    summary = (
        filtered.groupby("worker")["pnl"]
        .agg(["count", "sum", "mean"])
        .rename(columns={"count": "trades", "sum": "total_pnl", "mean": "avg_pnl"})
    )
    st.markdown("### Worker performance")
    st.dataframe(summary, use_container_width=True)


def render_strategy_telemetry(trades: pd.DataFrame, ml_scores: pd.DataFrame, config: Dict[str, Any]) -> None:
    st.markdown("### Strategy signals & health")
    workers = [w.get("name") for w in config.get("workers", [])]
    metrics: List[Dict[str, Any]] = []
    for worker in workers:
        worker_trades = trades[trades["worker"] == worker] if not trades.empty else pd.DataFrame()
        trade_count = worker_trades.shape[0]
        win_rate = (worker_trades[worker_trades["pnl"] > 0].shape[0] / trade_count) if trade_count else np.nan
        last_trade = worker_trades.sort_values("opened_at").tail(1)
        last_side = last_trade["side"].iloc[0] if not last_trade.empty else "-"
        worker_scores = ml_scores[ml_scores["worker"] == worker] if not ml_scores.empty else pd.DataFrame()
        recent_score = worker_scores.sort_values("ts").tail(1)
        proba = float(recent_score["proba_win"].iloc[0]) if not recent_score.empty else np.nan
        sentiment = "Bullish" if proba and proba >= 0.55 else "Bearish" if proba and proba <= 0.45 else "Neutral"
        metrics.append(
            {
                "worker": worker,
                "trades": trade_count,
                "win_rate": win_rate,
                "last_side": last_side,
                "proba": proba,
                "sentiment": sentiment,
            }
        )

    grid = st.columns(3)
    for idx, metric in enumerate(metrics):
        column = grid[idx % 3]
        with column:
            sentiment_class = metric["sentiment"].lower()
            win_rate_display = f"{metric['win_rate']:.1%}" if not np.isnan(metric["win_rate"]) else "n/a"
            proba_display = f"{metric['proba']:.2f}" if not np.isnan(metric["proba"]) else "n/a"
            column.markdown(
                f"<div class='strategy-card glass {sentiment_class}'>"
                f"<h4>{metric['worker']}</h4>"
                f"<p>Trades: {metric['trades']} · Win rate {win_rate_display}</p>"
                f"<p>Last direction: {metric['last_side']}</p>"
                f"<strong>{metric['sentiment']}</strong> · Signal {proba_display}"
                f"</div>",
                unsafe_allow_html=True,
            )

    if not ml_scores.empty:
        trend = ml_scores.copy()
        trend["ts"] = pd.to_datetime(trend["ts"])
        trend = trend.sort_values("ts")
        fig = go.Figure()
        for worker, frame in trend.groupby("worker"):
            fig.add_trace(
                go.Scatter(x=frame["ts"], y=frame["proba_win"], mode="lines", name=worker)
            )
        fig.update_layout(title="Signal probability by worker", yaxis=dict(range=[0, 1]))
        st.plotly_chart(apply_template(fig), use_container_width=True)
    else:
        st.info("No ML telemetry captured yet.")


def render_market_data(feed: KrakenMarketFeed) -> None:
    st.markdown("### Live Kraken market data")
    symbol = st.selectbox("Symbol", MARKET_SYMBOLS, index=0, key="market_symbol")
    snapshot = st.session_state.get("market_snapshots", {}).get(symbol)
    if snapshot is None:
        st.warning("No market data available yet – establishing feeds.")
        return

    cols = st.columns(3)
    cols[0].markdown(
        (
            "<div class='kpi-card glass'><span>Last price</span>"
            f"<strong>${snapshot.get('last_price', 0):,.2f}</strong></div>"
        ),
        unsafe_allow_html=True,
    )
    change = snapshot.get("change_pct")
    change_class = "positive" if change and change >= 0 else "negative"
    cols[1].markdown(
        f"<div class='kpi-card glass'><span>Δ 1m</span><strong class='{change_class}'>"
        f"{change if change is not None else 0:.2f}%</strong></div>",
        unsafe_allow_html=True,
    )
    updated = snapshot.get("updated_at")
    source = snapshot.get("source", "rest").upper()
    cols[2].markdown(
        (
            "<div class='kpi-card glass'><span>Updated</span>"
            f"<strong>{updated.strftime('%H:%M:%S') if updated else 'n/a'}</strong>"
            f"<small>{source}</small></div>"
        ),
        unsafe_allow_html=True,
    )

    with st.container():
        trades = st.session_state.get("data_cache", {}).get("trades", pd.DataFrame())
        market = snapshot.get("candles", pd.DataFrame())
        figure = candles_with_trades(market, trades[trades["symbol"] == symbol] if not trades.empty else pd.DataFrame())
        st.plotly_chart(figure, use_container_width=True)

    book = fetch_order_book(symbol)
    order_cols = st.columns(2)
    with order_cols[0]:
        st.markdown("#### Asks")
        if book["asks"].empty:
            st.info("No ask data available.")
        else:
            st.dataframe(book["asks"][['price', 'qty']], use_container_width=True)
    with order_cols[1]:
        st.markdown("#### Bids")
        if book["bids"].empty:
            st.info("No bid data available.")
        else:
            st.dataframe(book["bids"][['price', 'qty']], use_container_width=True)


def render_allocations(config: Dict[str, Any], trades: pd.DataFrame) -> None:
    st.markdown("### Strategy weights")
    allocations = st.session_state.setdefault("allocation_overrides", {})
    workers = config.get("workers", [])
    total = 0.0
    for worker in workers:
        name = worker.get("name")
        default = float(worker.get("allocation", 0.0))
        allocations[name] = st.slider(
            f"{name}",
            min_value=0.0,
            max_value=0.5,
            value=float(allocations.get(name, default)),
            step=0.01,
            help="Target equity allocation for this strategy.",
        )
        total += allocations[name]
    st.session_state["allocation_overrides"] = allocations

    if total == 0:
        st.warning("Increase at least one allocation to view distribution.")
        return

    weights = {name: value / total for name, value in allocations.items() if value > 0}
    pie = go.Figure(go.Pie(labels=list(weights.keys()), values=list(weights.values()), hole=0.35))
    pie.update_traces(textinfo="label+percent", pull=[0.05] * len(weights))
    st.plotly_chart(apply_template(pie), use_container_width=True)

    if st.button("Save allocations to config", use_container_width=True):
        config_snapshot = fetch_config_snapshot()
        for worker in config_snapshot.get("workers", []):
            name = worker.get("name")
            if name in allocations:
                worker["allocation"] = float(allocations[name])
        save_config(config_snapshot, CONFIG_PATH)
        fetch_config_snapshot.clear()
        st.success("Allocations persisted. Portfolio optimiser will rebalance next run.")

    if not trades.empty:
        perf = trades.groupby("worker")["pnl"].sum().rename("realised_pnl")
        st.dataframe(perf.to_frame(), use_container_width=True)


def render_ml_health(ml_scores: pd.DataFrame) -> None:
    if ml_scores.empty:
        st.info("ML telemetry table is empty. Training jobs have not published metrics yet.")
        return

    df = ml_scores.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df.sort_values("ts", inplace=True)
    accuracy = (df["label"] == (df["proba_win"] > 0.5)).astype(int)
    df["accuracy"] = accuracy
    rolling = df.groupby("worker").rolling(window=50, on="ts").accuracy.mean().reset_index()
    fig = go.Figure()
    for worker, frame in rolling.groupby("worker"):
        fig.add_trace(go.Scatter(x=frame["ts"], y=frame["accuracy"], mode="lines", name=f"{worker} accuracy"))
    fig.update_layout(title="Rolling signal accuracy", yaxis=dict(range=[0, 1]))
    st.plotly_chart(apply_template(fig), use_container_width=True)

    loss_proxy = 1 - df.groupby("ts")["accuracy"].mean().rolling(20).mean()
    loss_fig = go.Figure(go.Scatter(x=loss_proxy.index, y=loss_proxy.values, mode="lines", name="loss"))
    loss_fig.update_layout(title="Training loss proxy (1 - accuracy)")
    st.plotly_chart(apply_template(loss_fig), use_container_width=True)

    latest = df.sort_values("ts").groupby("worker").tail(1)
    st.dataframe(latest[["worker", "symbol", "proba_win", "accuracy"]], use_container_width=True)


def render_execution_controls() -> None:
    st.markdown("### Execution switches")
    status = st.session_state.get("runtime_status", "live")
    col1, col2, col3 = st.columns(3)
    if col1.button("Pause trading", use_container_width=True):
        status = "paused"
        log_control_event("pause")
    if col2.button("Resume trading", use_container_width=True):
        status = "live"
        log_control_event("resume")
    if col3.button("Emergency stop", use_container_width=True, type="primary"):
        status = "halted"
        log_control_event("halt")
    st.session_state["runtime_status"] = status
    st.markdown(f"<div class='callout'>Runtime status: <strong>{status.upper()}</strong></div>", unsafe_allow_html=True)

    if st.button("Reload config", use_container_width=True):
        fetch_config_snapshot.clear()
        st.session_state["config_raw"] = fetch_config_snapshot()
        st.success("Configuration reloaded from disk.")

    st.markdown("### Control event log")
    log = st.session_state.get("control_log", [])
    if not log:
        st.info("No manual interventions recorded this session.")
    else:
        st.code("\n".join(log[-12:]), language="text")


def log_control_event(action: str) -> None:
    log = st.session_state.get("control_log", [])
    log.append(f"{datetime.utcnow().isoformat(timespec='seconds')} | {action.upper()}")
    st.session_state["control_log"] = log[-30:]


def render_backtests(trades: pd.DataFrame, config: Dict[str, Any]) -> None:
    workers = [w.get("name") for w in config.get("workers", [])]
    if not workers:
        st.info("No strategies configured.")
        return
    selected_worker = st.selectbox("Strategy", workers, index=0)
    lookback_days = st.slider("Lookback (days)", min_value=7, max_value=180, value=60, step=1)
    start_time = pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days)
    worker_trades = trades[(trades["worker"] == selected_worker)] if not trades.empty else pd.DataFrame()
    if worker_trades.empty:
        st.warning("No trade history for the selected worker. Upload backtest results to populate.")
        return
    worker_trades = worker_trades.copy()
    worker_trades["opened_at"] = pd.to_datetime(worker_trades["opened_at"])
    worker_trades = worker_trades[worker_trades["opened_at"] >= start_time]
    worker_trades.sort_values("opened_at", inplace=True)
    worker_trades["cum_pnl"] = worker_trades["pnl"].cumsum()
    baseline_equity = 100_000
    worker_trades["equity"] = baseline_equity + worker_trades["cum_pnl"]
    fig = go.Figure(go.Scatter(x=worker_trades["opened_at"], y=worker_trades["equity"], mode="lines", name="Equity"))
    fig.update_layout(title=f"Backtest equity curve · {selected_worker}")
    st.plotly_chart(apply_template(fig), use_container_width=True)

    metrics = {
        "Total Trades": worker_trades.shape[0],
        "Net PnL": worker_trades["pnl"].sum(),
        "Win Rate": worker_trades[worker_trades["pnl"] > 0].shape[0] / worker_trades.shape[0]
        if worker_trades.shape[0]
        else np.nan,
        "Max Drawdown": worker_trades["cum_pnl"].min(),
    }
    grid = st.columns(len(metrics))
    for column, (label, value) in zip(grid, metrics.items()):
        if isinstance(value, float):
            if label == "Win Rate" and not np.isnan(value):
                formatted = f"{value:.2%}"
            else:
                formatted = f"{value:,.2f}"
        else:
            formatted = str(value)
        column.markdown(
            f"<div class='kpi-card glass'><span>{label}</span><strong>{formatted}</strong></div>",
            unsafe_allow_html=True,
        )


def render_settings_logs(config: Dict[str, Any], logs: pd.DataFrame, health: Iterable[DataHealth]) -> None:
    st.markdown("### Runtime configuration")
    st.json(config)
    st.download_button(
        "Download config",
        data=json.dumps(config, indent=2),
        file_name="config.json",
        mime="application/json",
    )

    st.markdown("### Logs")
    if logs.empty:
        st.info("No log files parsed yet. Runtime will emit JSON logs into desk/logs.")
    else:
        logs = logs.copy()
        logs["ts"] = pd.to_datetime(logs["ts"])
        level_filter = st.multiselect("Levels", sorted(logs["level"].unique()), default=["INFO", "WARNING", "ERROR"])
        filtered = logs[logs["level"].isin(level_filter)]
        st.dataframe(filtered.tail(200), use_container_width=True)

    st.markdown("### Database health")
    cols = st.columns(len(list(health)) or 1)
    for col, item in zip(cols, health):
        with col:
            st.metric(item.name, item.status.upper(), help=item.detail or "")


def refresh_market_snapshots(feed: KrakenMarketFeed) -> None:
    snapshots: Dict[str, Dict[str, Any]] = {}
    for symbol in MARKET_SYMBOLS:
        snapshot = compute_market_snapshot(feed, symbol)
        snapshots[symbol] = {
            "candles": snapshot.candles,
            "last_price": snapshot.last_price,
            "change_pct": snapshot.change_pct,
            "updated_at": snapshot.updated_at,
            "source": snapshot.source,
        }
    st.session_state["market_snapshots"] = snapshots


def main() -> None:
    st.set_page_config(page_title="Aurora Quant Command", layout="wide", page_icon="💠")
    inject_styles()

    schedule_autorefresh(REFRESH_SECONDS)
    seed_demo_data()

    config = fetch_config_snapshot()
    init_session_state(config)

    data = load_all_data(st.session_state.get("data_version", 0))
    st.session_state["data_cache"] = data

    equity = data["equity"]
    equity_value = float(equity["balance"].iloc[-1]) if not equity.empty else 100_000.0
    risk_engine = build_risk_engine(config, equity_value)
    risk_engine.check_account(equity_value)

    feed = get_market_feed(MARKET_SYMBOLS)
    refresh_market_snapshots(feed)

    health = database_health([DB_LIVE])
    render_header(equity_value, risk_engine, health)

    tabs = st.tabs(
        [
            "Portfolio Overview",
            "Risk Controls",
            "Trade Blotter",
            "Strategy Telemetry",
            "Live Market Data",
            "Allocations",
            "ML Health",
            "Execution Controls",
            "Backtests",
            "Settings & Logs",
        ]
    )

    with tabs[0]:
        render_portfolio_overview(data["trades"], equity, data["positions"], risk_engine)
    with tabs[1]:
        render_risk_controls(risk_engine, data["trades"])
    with tabs[2]:
        render_trade_blotter(data["trades"])
    with tabs[3]:
        render_strategy_telemetry(data["trades"], data["ml_scores"], config)
    with tabs[4]:
        render_market_data(feed)
    with tabs[5]:
        render_allocations(config, data["trades"])
    with tabs[6]:
        render_ml_health(data["ml_scores"])
    with tabs[7]:
        render_execution_controls()
    with tabs[8]:
        render_backtests(data["trades"], config)
    with tabs[9]:
        render_settings_logs(config, data["logs"], health)


if __name__ == "__main__":
    main()
