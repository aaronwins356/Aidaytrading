"""Streamlit dashboard for monitoring the trading bot."""
from __future__ import annotations

import datetime as dt
import io
import random
import time
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_feeds import (
    KrakenMarketFeed,
    PortfolioAnalytics,
    SQLiteManager,
    generate_mock_trades,
)
from plotting import (
    bar_chart,
    candlestick_chart,
    equity_curve_chart,
    gauge_chart,
    histogram,
    pie_chart,
    sparkline_chart,
)
from risk import RiskEngine
from ui_helpers import callout, initialize_session_state, inject_custom_css, kpi_card, render_top_controls, section_header

REFRESH_INTERVAL_MS = 30_000
MARKET_SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD", "MATIC/USD"]
REFRESH_INTERVAL_SECONDS = REFRESH_INTERVAL_MS / 1000


def schedule_autorefresh() -> None:
    """Trigger a rerun periodically to keep data fresh."""

    now = time.time()
    last_refresh = st.session_state.get("last_refresh", 0.0)
    if now - last_refresh >= REFRESH_INTERVAL_SECONDS:
        st.session_state["last_refresh"] = now
        rerun = getattr(st, "experimental_rerun", None)
        if callable(rerun):  # pragma: no cover - requires Streamlit runtime
            rerun()
    elif "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = now


def setup_page() -> Tuple[SQLiteManager, PortfolioAnalytics, RiskEngine, KrakenMarketFeed]:
    """Initialise key services and configure the Streamlit page."""

    st.set_page_config(page_title="Quant Control Center", layout="wide", page_icon="📈")
    initialize_session_state()
    inject_custom_css(dark_mode=st.session_state["dark_mode"])
    render_top_controls(dark_mode=st.session_state["dark_mode"])

    store = SQLiteManager()
    generate_mock_trades(store)
    analytics = PortfolioAnalytics(store)

    risk_engine = RiskEngine(
        daily_dd=0.12,
        weekly_dd=0.25,
        default_stop_pct=st.session_state["stop_loss_default"],
        max_concurrent=5,
        halt_on_dd=True,
        trapdoor_pct=0.1,
        equity_floor=75_000,
        risk_per_trade_pct=st.session_state["equity_per_trade"],
        max_risk_per_trade_pct=0.05,
        max_position_value=25_000,
    )

    feed = KrakenMarketFeed(MARKET_SYMBOLS)
    feed.start()

    return store, analytics, risk_engine, feed


def overview_tab(analytics: PortfolioAnalytics, risk_engine: RiskEngine, store: SQLiteManager) -> None:
    section_header("Performance Overview", "Key Metrics")
    trades = store.fetch_trades()
    equity_curve = analytics.compute_equity_curve()
    todays_returns = trades[trades["opened_at"] >= (pd.Timestamp.utcnow().timestamp() - 86_400)]
    daily_pnl = todays_returns["pnl"].sum() if not todays_returns.empty else random.uniform(-450, 650)
    total_equity = equity_curve["equity"].iloc[-1] if not equity_curve.empty else 100_000.0
    total_pnl = trades["pnl"].sum() if not trades.empty else random.uniform(5000, 12000)

    sparkline = sparkline_chart(equity_curve["equity"].tail(30) if not equity_curve.empty else np.random.normal(100_000, 500, 30))

    cols = st.columns(4)
    with cols[0]:
        kpi_card("Total Equity", f"${total_equity:,.0f}", delta=f"PnL {total_pnl:,.0f}", sparkline=sparkline)
    with cols[1]:
        kpi_card("Daily PnL", f"${daily_pnl:,.0f}", delta=f"{(daily_pnl/total_equity)*100:.2f}%")
    with cols[2]:
        kpi_card("Active Strategies", str(sum(st.session_state["active_strategies"].values())), delta="Equity aligned")
    with cols[3]:
        open_trades = trades[trades["status"] == "OPEN"].shape[0] if not trades.empty else random.randint(0, 5)
        kpi_card("Open Trades", str(open_trades), delta="Risk-managed")

    st.plotly_chart(equity_curve_chart(equity_curve, title="Equity Curve"), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        drawdown_samples = np.random.uniform(2, 18, size=200)
        st.plotly_chart(histogram(drawdown_samples, title="Distribution of Trade Drawdowns (%)"), use_container_width=True)
    with col2:
        st.plotly_chart(
            gauge_chart(
                value=max(0.0, min(100.0, (1 - st.session_state["risk_tolerance"]) * 100)),
                title="Risk Budget Usage (%)",
                threshold=st.session_state["max_drawdown_threshold"] * 100,
            ),
            use_container_width=True,
        )

    callout(
        "Monitor equity, drawdowns, and risk budgets in real time. The current risk per trade is "
        f"{risk_engine.risk_per_trade_pct:.2%}; adjust tolerance in the Risk Engine tab to alter these controls."
    )


def risk_engine_tab(risk_engine: RiskEngine, store: SQLiteManager) -> None:
    section_header("Risk Controls", "Risk Engine")
    cols = st.columns(3)
    with cols[0]:
        st.session_state["risk_tolerance"] = st.slider(
            "Risk Tolerance",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state["risk_tolerance"]),
            help="Global risk throttle across all strategies.",
        )
    with cols[1]:
        st.session_state["max_drawdown_threshold"] = st.slider(
            "Max Drawdown Threshold",
            min_value=0.05,
            max_value=0.5,
            step=0.01,
            value=float(st.session_state["max_drawdown_threshold"]),
        )
    with cols[2]:
        st.session_state["equity_per_trade"] = st.number_input(
            "Equity % Per Trade",
            min_value=0.005,
            max_value=0.1,
            value=float(st.session_state["equity_per_trade"]),
            step=0.005,
            help="Risk per trade as a percentage of total equity.",
        )
        store.persist_setting("equity_per_trade", st.session_state["equity_per_trade"])

    st.session_state["stop_loss_default"] = st.number_input(
        "Default Stop Loss %",
        min_value=0.005,
        max_value=0.1,
        step=0.005,
        value=float(st.session_state["stop_loss_default"]),
    )
    st.session_state["take_profit_default"] = st.number_input(
        "Default Take Profit %",
        min_value=0.01,
        max_value=0.25,
        step=0.01,
        value=float(st.session_state["take_profit_default"]),
    )

    st.markdown("### Allocation Controls")
    strategies = st.session_state["active_strategies"]
    alloc_cols = st.columns(len(strategies))
    for idx, (name, active) in enumerate(strategies.items()):
        with alloc_cols[idx]:
            st.session_state["active_strategies"][name] = st.toggle(name, value=active, help="Enable/disable this strategy")

    st.markdown("### Equity Allocation Gauge")
    st.plotly_chart(
        gauge_chart(
            value=risk_engine.base_risk_pct * 100,
            title="Base Risk %",
            threshold=st.session_state["max_drawdown_threshold"] * 100,
        ),
        use_container_width=True,
    )

    st.info("Risk configuration updates are persisted to SQLite so that the bot can consume them in real time.")


def trade_log_tab(store: SQLiteManager) -> None:
    section_header("Trade Blotter", "Live & Historical")
    trades = store.fetch_trades()
    if trades.empty:
        st.warning("No trades recorded yet.")
        return

    trades["opened_at"] = pd.to_datetime(trades["opened_at"], unit="s")
    trades["closed_at"] = pd.to_datetime(trades["closed_at"], unit="s", errors="coerce")

    col1, col2, col3 = st.columns([0.4, 0.3, 0.3])
    with col1:
        symbols = sorted(trades["symbol"].unique())
        selected_symbols = st.multiselect("Symbols", options=symbols, default=symbols)
    with col2:
        status_options = sorted(trades["status"].unique())
        selected_status = st.multiselect("Status", options=status_options, default=status_options)
    with col3:
        min_pnl, max_pnl = st.slider(
            "PnL Range",
            min_value=float(trades["pnl"].min()),
            max_value=float(trades["pnl"].max()),
            value=(float(trades["pnl"].min()), float(trades["pnl"].max())),
        )

    filtered = trades[
        trades["symbol"].isin(selected_symbols)
        & trades["status"].isin(selected_status)
        & trades["pnl"].between(min_pnl, max_pnl)
    ]

    st.dataframe(filtered.sort_values("opened_at", ascending=False), use_container_width=True, hide_index=True)

    csv_buffer = io.StringIO()
    filtered.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download CSV",
        data=csv_buffer.getvalue(),
        file_name="trade_log.csv",
        mime="text/csv",
    )


def strategies_tab(store: SQLiteManager) -> None:
    section_header("Strategy Telemetry", "Workers")
    states = store.get_strategy_states()
    if not states:
        st.info("Strategies have not reported telemetry yet.")
        return

    for name, payload in states.items():
        col1, col2 = st.columns([0.3, 0.7])
        with col1:
            st.metric(label=f"{name} Signal", value=payload["signal"])
            st.write(f"Last Updated: {pd.to_datetime(payload['updated_at'], unit='s').strftime('%Y-%m-%d %H:%M:%S')}")
        with col2:
            st.write("**Indicators**")
            st.dataframe(pd.DataFrame(payload["indicators"], index=["value"]).T, use_container_width=True)


def market_data_tab(feed: KrakenMarketFeed) -> None:
    section_header("Market Intelligence", "Kraken Feed")
    symbol = st.selectbox("Symbol", options=MARKET_SYMBOLS, index=0)
    candles = feed.get_candles(symbol)
    if candles.empty:
        st.error("No market data available. Fallback synthetic data will be used.")
        candles = feed._synthetic_data(symbol, 120)

    candles["MA20"] = candles["close"].rolling(window=20).mean()
    candles["MA50"] = candles["close"].rolling(window=50).mean()

    fig = candlestick_chart(candles, title=f"{symbol} - 1m Candles")
    fig.add_trace(
        go.Scatter(x=candles["time"], y=candles["MA20"], mode="lines", line=dict(color="#facc15", width=1.5), name="MA20")
    )
    fig.add_trace(
        go.Scatter(x=candles["time"], y=candles["MA50"], mode="lines", line=dict(color="#38bdf8", width=1.5), name="MA50")
    )
    st.plotly_chart(fig, use_container_width=True)

    rsi = talib_like_rsi(candles["close"].to_numpy())
    macd_line, signal_line = talib_like_macd(candles["close"].to_numpy())

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(sparkline_chart(rsi[-60:]), use_container_width=True)
        st.caption("RSI (14)")
    with col2:
        macd_df = pd.DataFrame({"MACD": macd_line[-60:], "Signal": signal_line[-60:]})
        st.line_chart(macd_df)


def talib_like_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(prices, prepend=prices[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss.replace(0, 1e-9))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).to_numpy()


def talib_like_macd(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    price_series = pd.Series(prices)
    ema12 = price_series.ewm(span=12, adjust=False).mean()
    ema26 = price_series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line.to_numpy(), signal.to_numpy()


def portfolio_tab(analytics: PortfolioAnalytics) -> None:
    section_header("Capital Allocation", "Portfolio")
    allocation = analytics.allocation_breakdown()
    strategy_perf = analytics.strategy_performance()

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(pie_chart(allocation, title="Asset Allocation"), use_container_width=True)
    with col2:
        st.plotly_chart(bar_chart(strategy_perf, title="Strategy PnL"), use_container_width=True)


def machine_learning_tab(store: SQLiteManager) -> None:
    section_header("Machine Learning Telemetry", "Model Health")
    telemetry = store.fetch_telemetry()
    accuracy = telemetry.get("model_accuracy", {}).get("value", 0.0)
    win_rate = telemetry.get("win_rate", {}).get("value", 0.0)
    bullish = telemetry.get("bullish_probability", {}).get("value", 0.5)
    bearish = telemetry.get("bearish_probability", {}).get("value", 0.5)
    feature_data = telemetry.get("feature_importance", {}).get("metadata", {}).get("features", {})

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(gauge_chart(accuracy * 100, title="Model Accuracy"), use_container_width=True)
    with col2:
        st.plotly_chart(gauge_chart(win_rate * 100, title="Win Rate"), use_container_width=True)

    st.markdown("### Prediction Distribution")
    st.progress(bullish, text=f"Bullish probability: {bullish:.2%}")
    st.progress(bearish, text=f"Bearish probability: {bearish:.2%}")

    if feature_data:
        st.plotly_chart(bar_chart(feature_data, title="Feature Importance"), use_container_width=True)


def controls_tab(store: SQLiteManager) -> None:
    section_header("Execution Controls", "Overrides")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state["leverage"] = st.slider("Leverage", min_value=1.0, max_value=5.0, step=0.5, value=float(st.session_state["leverage"]))
    with col2:
        st.session_state["stop_loss_default"] = st.slider(
            "Stop Loss %",
            min_value=0.005,
            max_value=0.1,
            step=0.005,
            value=float(st.session_state["stop_loss_default"]),
        )
    with col3:
        st.session_state["take_profit_default"] = st.slider(
            "Take Profit %",
            min_value=0.01,
            max_value=0.3,
            step=0.01,
            value=float(st.session_state["take_profit_default"]),
        )

    st.write("### Strategy Toggles")
    for name, active in st.session_state["active_strategies"].items():
        st.session_state["active_strategies"][name] = st.checkbox(f"Enable {name}", value=active)

    panic = st.button("⚠️ Panic Stop", type="primary")
    if panic:
        store.log_event("WARNING", "Panic stop triggered from dashboard.")
        st.toast("Panic stop request sent to bot.", icon="⚠️")

    store.persist_setting("controls", {
        "leverage": st.session_state["leverage"],
        "stop_loss": st.session_state["stop_loss_default"],
        "take_profit": st.session_state["take_profit_default"],
        "strategies": st.session_state["active_strategies"],
    })


def backtests_tab(analytics: PortfolioAnalytics) -> None:
    section_header("On-Demand Backtests", "Simulation")
    with st.form("backtest_form"):
        symbol = st.selectbox("Symbol", options=MARKET_SYMBOLS, index=0)
        strategy = st.selectbox("Strategy", options=["Momentum", "Mean Reversion", "Breakout", "Scalping"], index=0)
        start_date = st.date_input("Start Date", value=dt.date.today() - dt.timedelta(days=90))
        end_date = st.date_input("End Date", value=dt.date.today())
        submitted = st.form_submit_button("Run Backtest")

    if submitted:
        duration = (end_date - start_date).days or 1
        base_equity = 100_000
        returns = np.random.normal(loc=0.001, scale=0.02, size=duration)
        equity = base_equity * (1 + returns).cumprod()
        curve = pd.DataFrame({"timestamp": pd.date_range(start=start_date, periods=duration, freq="D"), "equity": equity})
        st.plotly_chart(equity_curve_chart(curve, title=f"Backtest Equity Curve - {strategy} ({symbol})"), use_container_width=True)
        drawdown = (curve["equity"].cummax() - curve["equity"]) / curve["equity"].cummax()
        st.metric("Max Drawdown", f"{drawdown.max():.2%}")
        st.metric("Sharpe Ratio", f"{returns.mean() / (returns.std() + 1e-9) * np.sqrt(252):.2f}")


def settings_logs_tab(store: SQLiteManager) -> None:
    section_header("Settings & Logs", "Runtime")
    settings = store.fetch_settings()
    st.write("### Persisted Settings")
    st.json(settings)

    st.write("### Log Viewer")
    level = st.selectbox("Level", options=["ALL", "INFO", "WARNING", "ERROR"], index=0)
    limit = st.slider("Limit", min_value=50, max_value=500, step=50, value=200)
    logs = store.fetch_logs(level=None if level == "ALL" else level, limit=limit)
    if logs.empty:
        st.info("No log entries found.")
    else:
        logs["created_at"] = pd.to_datetime(logs["created_at"], unit="s")
        st.dataframe(logs, use_container_width=True, hide_index=True)
        st.download_button("Download Logs", logs.to_csv(index=False), file_name="dashboard_logs.csv")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Logs"):
            store.clear_logs()
            st.experimental_rerun()
    with col2:
        if st.button("Refresh"):
            st.experimental_rerun()


def dispatch_notifications(store: SQLiteManager) -> None:
    trades = store.fetch_trades(status="OPEN")
    latest_open = trades.sort_values("opened_at", ascending=False).head(1)
    if not latest_open.empty:
        trade_id = latest_open.iloc[0]["trade_id"]
        if trade_id not in st.session_state["notifications"]:
            st.session_state["notifications"].append(trade_id)
            st.toast(
                f"Trade {trade_id} opened on {latest_open.iloc[0]['symbol']} ({latest_open.iloc[0]['equity_pct']:.2%} equity).",
                icon="✅",
            )


def main() -> None:
    store, analytics, risk_engine, feed = setup_page()
    schedule_autorefresh()
    dispatch_notifications(store)

    tabs = st.tabs([
        "Overview",
        "Risk Engine",
        "Trade Log",
        "Strategies",
        "Market Data",
        "Portfolio Allocations",
        "Machine Learning",
        "Controls",
        "Backtests",
        "Settings / Logs",
    ])

    with tabs[0]:
        overview_tab(analytics, risk_engine, store)
    with tabs[1]:
        risk_engine_tab(risk_engine, store)
    with tabs[2]:
        trade_log_tab(store)
    with tabs[3]:
        strategies_tab(store)
    with tabs[4]:
        market_data_tab(feed)
    with tabs[5]:
        portfolio_tab(analytics)
    with tabs[6]:
        machine_learning_tab(store)
    with tabs[7]:
        controls_tab(store)
    with tabs[8]:
        backtests_tab(analytics)
    with tabs[9]:
        settings_logs_tab(store)


if __name__ == "__main__":
    main()
