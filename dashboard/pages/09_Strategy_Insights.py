from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import download_chart_as_png

st.set_page_config(page_title="Strategy Insights · Aurora Desk", page_icon="🧠")


@dataclass(frozen=True)
class StrategyInfo:
    """Display metadata for a strategy."""

    name: str
    category: str
    market_bias: str
    ideal_conditions: str
    description: str
    playbook: List[str]
    risk_notes: List[str]
    builder: Callable[[pd.DataFrame], Tuple[go.Figure, Dict[str, float]]]


@st.cache_data(show_spinner=False)
def _load_demo_market() -> pd.DataFrame:
    """Create a synthetic intraday market series for visualization."""

    periods = 320
    rng = pd.date_range(end=pd.Timestamp.utcnow(), periods=periods, freq="H")
    base = 24_000 + np.cumsum(np.random.normal(0, 45, periods))
    cyclical = 350 * np.sin(np.linspace(0, 4 * np.pi, periods))
    trend = np.linspace(-150, 420, periods)
    close = base + cyclical + trend
    high = close + np.abs(np.random.normal(18, 6, periods))
    low = close - np.abs(np.random.normal(18, 6, periods))
    df = pd.DataFrame({"ts": rng, "close": close, "high": high, "low": low})
    df["returns"] = df["close"].pct_change().fillna(0.0)
    return df


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def _rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / length, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / length, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def _performance_from_positions(df: pd.DataFrame, positions: pd.Series) -> Dict[str, float]:
    positions = positions.fillna(method="ffill").fillna(0)
    strat_ret = positions.shift().fillna(0) * df["returns"]
    equity = (1 + strat_ret).cumprod()
    trades = positions.diff().abs() > 0
    trade_count = int(trades.sum() / 2)
    positives = strat_ret[strat_ret > 0]
    non_zero = strat_ret[strat_ret != 0]
    hit_rate = float((positives.count() / max(non_zero.count(), 1)) * 100) if not non_zero.empty else 0.0
    sharpe = float(np.sqrt(24) * strat_ret.mean() / strat_ret.std()) if strat_ret.std() > 0 else 0.0
    return {
        "Net return %": (equity.iloc[-1] - 1) * 100,
        "Hit rate %": hit_rate,
        "Trades": trade_count,
        "Sharpe (sim)": sharpe,
    }


def _sma_crossover_chart(df: pd.DataFrame) -> Tuple[go.Figure, Dict[str, float]]:
    fast = _sma(df["close"], 18)
    slow = _sma(df["close"], 50)
    long = (fast > slow) & (fast.shift() <= slow.shift())
    short = (fast < slow) & (fast.shift() >= slow.shift())
    signals = pd.Series(0, index=df.index, dtype=float)
    signals[long] = 1
    signals[short] = -1
    positions = signals.replace(0, np.nan).ffill().fillna(0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["ts"], y=df["close"], name="Close", line=dict(color="#38bdf8")))
    fig.add_trace(go.Scatter(x=df["ts"], y=fast, name="Fast SMA", line=dict(color="#f97316", width=2)))
    fig.add_trace(go.Scatter(x=df["ts"], y=slow, name="Slow SMA", line=dict(color="#6366f1", width=2)))
    fig.add_trace(
        go.Scatter(
            x=df.loc[long, "ts"],
            y=df.loc[long, "close"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=12, color="#22c55e"),
            name="Bullish cross",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.loc[short, "ts"],
            y=df.loc[short, "close"],
            mode="markers",
            marker=dict(symbol="triangle-down", size=12, color="#ef4444"),
            name="Bearish cross",
        )
    )
    fig.update_layout(title="SMA crossover execution playbook", legend=dict(orientation="h", yanchor="bottom", y=1.02))
    stats = _performance_from_positions(df, positions)
    return fig, stats


def _rsi_mean_reversion_chart(df: pd.DataFrame) -> Tuple[go.Figure, Dict[str, float]]:
    rsi = _rsi(df["close"], 14)
    long = rsi < 30
    short = rsi > 70
    signals = pd.Series(0, index=df.index, dtype=float)
    signals[long] = 1
    signals[short] = -1
    positions = signals.replace(0, np.nan).ffill().fillna(0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["ts"], y=df["close"], name="Close", line=dict(color="#38bdf8")))
    fig.add_trace(
        go.Scatter(
            x=df["ts"],
            y=rsi,
            name="RSI",
            line=dict(color="#facc15"),
            yaxis="y2",
        )
    )
    fig.add_hrect(y0=70, y1=70, line_width=0, fillcolor="#ef4444", opacity=0.2, yref="y2")
    fig.add_hrect(y0=30, y1=30, line_width=0, fillcolor="#22c55e", opacity=0.2, yref="y2")
    fig.add_trace(
        go.Scatter(
            x=df.loc[long, "ts"],
            y=df.loc[long, "close"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=12, color="#22c55e"),
            name="RSI < 30",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.loc[short, "ts"],
            y=df.loc[short, "close"],
            mode="markers",
            marker=dict(symbol="triangle-down", size=12, color="#ef4444"),
            name="RSI > 70",
        )
    )
    fig.update_layout(
        title="RSI mean reversion context",
        yaxis2=dict(anchor="x", overlaying="y", side="right", range=[0, 100], title="RSI"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    stats = _performance_from_positions(df, positions)
    return fig, stats


def _ema_breakout_chart(df: pd.DataFrame) -> Tuple[go.Figure, Dict[str, float]]:
    ema = _ema(df["close"], 21)
    vol = df["close"].rolling(window=21, min_periods=21).std()
    upper = ema + vol
    lower = ema - vol
    long = df["close"] > upper
    short = df["close"] < lower
    signals = pd.Series(0, index=df.index, dtype=float)
    signals[long] = 1
    signals[short] = -1
    positions = signals.replace(0, np.nan).ffill().fillna(0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["ts"], y=df["close"], name="Close", line=dict(color="#38bdf8")))
    fig.add_trace(go.Scatter(x=df["ts"], y=ema, name="21 EMA", line=dict(color="#a855f7", width=2)))
    fig.add_trace(
        go.Scatter(
            x=df["ts"],
            y=upper,
            name="Breakout band",
            line=dict(color="#f97316", dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["ts"],
            y=lower,
            name="Breakdown band",
            line=dict(color="#f97316", dash="dot"),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.loc[long, "ts"],
            y=df.loc[long, "close"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=12, color="#22c55e"),
            name="Breakout long",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.loc[short, "ts"],
            y=df.loc[short, "close"],
            mode="markers",
            marker=dict(symbol="triangle-down", size=12, color="#ef4444"),
            name="Breakdown short",
        )
    )
    fig.update_layout(title="EMA breakout confirmation bands", legend=dict(orientation="h", yanchor="bottom", y=1.02))
    stats = _performance_from_positions(df, positions)
    return fig, stats


def _bollinger_reversion_chart(df: pd.DataFrame) -> Tuple[go.Figure, Dict[str, float]]:
    mid = _sma(df["close"], 20)
    std = df["close"].rolling(window=20, min_periods=20).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    long = df["close"] < lower
    short = df["close"] > upper
    signals = pd.Series(0, index=df.index, dtype=float)
    signals[long] = 1
    signals[short] = -1
    positions = signals.replace(0, np.nan).ffill().fillna(0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["ts"], y=df["close"], name="Close", line=dict(color="#38bdf8")))
    fig.add_trace(
        go.Scatter(
            x=df["ts"],
            y=upper,
            name="Upper band",
            line=dict(color="#f97316", dash="dot"),
        )
    )
    fig.add_trace(go.Scatter(x=df["ts"], y=mid, name="Middle band", line=dict(color="#a855f7", width=2)))
    fig.add_trace(
        go.Scatter(
            x=df["ts"],
            y=lower,
            name="Lower band",
            line=dict(color="#f97316", dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.loc[long, "ts"],
            y=df.loc[long, "close"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=12, color="#22c55e"),
            name="Lower band touch",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.loc[short, "ts"],
            y=df.loc[short, "close"],
            mode="markers",
            marker=dict(symbol="triangle-down", size=12, color="#ef4444"),
            name="Upper band touch",
        )
    )
    fig.update_layout(title="Bollinger band mean reversion map", legend=dict(orientation="h", yanchor="bottom", y=1.02))
    stats = _performance_from_positions(df, positions)
    return fig, stats


def _momentum_swing_chart(df: pd.DataFrame) -> Tuple[go.Figure, Dict[str, float]]:
    roc = df["close"].pct_change(periods=6) * 100
    long = roc > 1.5
    short = roc < -1.5
    signals = pd.Series(0, index=df.index, dtype=float)
    signals[long] = 1
    signals[short] = -1
    positions = signals.replace(0, np.nan).ffill().fillna(0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["ts"], y=df["close"], name="Close", line=dict(color="#38bdf8")))
    fig.add_trace(
        go.Bar(x=df["ts"], y=roc, name="Momentum %", marker_color="#facc15", yaxis="y2", opacity=0.5)
    )
    fig.add_shape(type="line", x0=df["ts"].iloc[0], x1=df["ts"].iloc[-1], y0=1.5, y1=1.5, yref="y2", line=dict(color="#22c55e", dash="dot"))
    fig.add_shape(type="line", x0=df["ts"].iloc[0], x1=df["ts"].iloc[-1], y0=-1.5, y1=-1.5, yref="y2", line=dict(color="#ef4444", dash="dot"))
    fig.add_trace(
        go.Scatter(
            x=df.loc[long, "ts"],
            y=df.loc[long, "close"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=12, color="#22c55e"),
            name="Momentum long",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.loc[short, "ts"],
            y=df.loc[short, "close"],
            mode="markers",
            marker=dict(symbol="triangle-down", size=12, color="#ef4444"),
            name="Momentum short",
        )
    )
    fig.update_layout(
        title="Momentum ignition bursts",
        yaxis2=dict(anchor="x", overlaying="y", side="right", title="ROC %"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    stats = _performance_from_positions(df, positions)
    return fig, stats


def _atr_trailing_chart(df: pd.DataFrame) -> Tuple[go.Figure, Dict[str, float]]:
    atr = _atr(df, 14)
    direction = _ema(df["close"], 30)
    stop_long = df["close"] - 2 * atr
    stop_short = df["close"] + 2 * atr
    long = df["close"] > direction
    short = df["close"] < direction
    positions = pd.Series(np.where(long, 1.0, np.where(short, -1.0, np.nan)), index=df.index)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["ts"], y=df["close"], name="Close", line=dict(color="#38bdf8")))
    fig.add_trace(go.Scatter(x=df["ts"], y=direction, name="Directional EMA", line=dict(color="#a855f7", width=2)))
    fig.add_trace(
        go.Scatter(
            x=df["ts"],
            y=stop_long,
            name="Trailing stop (long)",
            line=dict(color="#22c55e", dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["ts"],
            y=stop_short,
            name="Trailing stop (short)",
            line=dict(color="#ef4444", dash="dot"),
        )
    )
    fig.update_layout(title="ATR trailing stop discipline", legend=dict(orientation="h", yanchor="bottom", y=1.02))
    stats = _performance_from_positions(df, positions)
    return fig, stats


STRATEGY_LIBRARY: Dict[str, StrategyInfo] = {
    "sma_crossover": StrategyInfo(
        name="SMA Trend Crossover",
        category="Trend-following",
        market_bias="Expanding / directional",
        ideal_conditions="Strong momentum with orderly pullbacks",
        description=(
            "Tracks fast and slow moving averages to ride medium-term trends. Crossovers signal shifts in market structure "
            "while keeping the strategy disciplined against chop."
        ),
        playbook=[
            "Enter long when the fast average crosses above the slow average with volume confirmation.",
            "Scale out into strength; trail stops below the slow average.",
            "Flip short on bearish crosses during downtrends to capture continuation legs.",
        ],
        risk_notes=[
            "Susceptible to sideways whipsaws – widen filters or reduce size in range-bound markets.",
            "Lagging nature means late exits if volatility spikes sharply.",
        ],
        builder=_sma_crossover_chart,
    ),
    "rsi_mean_reversion": StrategyInfo(
        name="RSI Exhaustion Fade",
        category="Mean reversion",
        market_bias="Range-bound / oscillating",
        ideal_conditions="Choppy sessions with clearly defined support and resistance",
        description=(
            "Looks for momentum exhaustion by monitoring the Relative Strength Index. Extreme prints (overbought/oversold) "
            "often precede snap-back moves back to the range's midpoint."
        ),
        playbook=[
            "Fade stretched moves when RSI tags 70/30 or custom thresholds.",
            "Target the midpoint of the range; partial profit quickly to manage risk.",
            "Avoid trading against dominant higher timeframe trends.",
        ],
        risk_notes=[
            "Trending markets can keep RSI pinned – incorporate higher timeframe bias filters.",
            "Set hard stops; exhaustion trades fail fast when trend resumes.",
        ],
        builder=_rsi_mean_reversion_chart,
    ),
    "ema_breakout": StrategyInfo(
        name="EMA Volatility Breakout",
        category="Breakout",
        market_bias="High volatility expansion",
        ideal_conditions="Fresh catalysts with expanding ranges and elevated volume",
        description=(
            "Combines an adaptive exponential moving average with volatility bands. Closing outside the band flags impulse "
            "moves with potential for follow through before mean reversion kicks in."
        ),
        playbook=[
            "Wait for price to close outside the volatility band to confirm strength.",
            "Trail risk using the EMA minus volatility buffer to keep trades tight.",
            "Reduce exposure into volatility crushes or mean reversion signs.",
        ],
        risk_notes=[
            "False breakouts common in low liquidity sessions – pair with volume filters.",
            "Gap reversals can quickly invalidate signals; consider stop-loss at band re-entry.",
        ],
        builder=_ema_breakout_chart,
    ),
    "bollinger_band": StrategyInfo(
        name="Bollinger Band Reversion",
        category="Mean reversion",
        market_bias="Sideways drift",
        ideal_conditions="Low volatility consolidations with repeated band tests",
        description=(
            "Uses standard deviation envelopes around a moving average to identify stretched moves. Entries occur when price "
            "tags outer bands and momentum shows signs of cooling."
        ),
        playbook=[
            "Enter counter-trend on outer band tags with confirmation from oscillators.",
            "Take profits near the middle band or opposite band depending on volatility.",
            "Reduce exposure when bands expand sharply – trend may be forming.",
        ],
        risk_notes=[
            "Expanding volatility can turn mean reversion signals into breakout traps.",
            "Requires disciplined profit taking; holding for trend reversals reduces expectancy.",
        ],
        builder=_bollinger_reversion_chart,
    ),
    "momentum": StrategyInfo(
        name="Momentum Ignition",
        category="Momentum scalping",
        market_bias="News-driven bursts",
        ideal_conditions="Fast tape with liquid order books",
        description=(
            "Measures rate-of-change to catch acceleration phases. The strategy aims to position alongside liquidation "
            "events or news spikes before they exhaust."
        ),
        playbook=[
            "Trigger entries when rate-of-change exceeds the ignition threshold.",
            "Scale out quickly; momentum bursts are short lived.",
            "Drop size when spreads widen or liquidity disappears.",
        ],
        risk_notes=[
            "Late entries after the burst expose the strategy to immediate reversals.",
            "Requires active management; holding through consolidations erodes edge.",
        ],
        builder=_momentum_swing_chart,
    ),
    "atr_trailing_stop": StrategyInfo(
        name="ATR Trailing Rider",
        category="Trend-following risk management",
        market_bias="Persistent trends",
        ideal_conditions="Directional moves with consistent volatility",
        description=(
            "Focuses on staying with the dominant move while managing exits using an Average True Range trail. Entries follow "
            "directional bias filters, and stops adapt to volatility."
        ),
        playbook=[
            "Use higher timeframe EMA slope to define trade direction.",
            "Trail stops at a multiple of ATR to let winners breathe.",
            "Scale out when price accelerates far from the trailing stop.",
        ],
        risk_notes=[
            "Sudden volatility shocks can gap through the trailing stop level.",
            "Choppy sessions reduce edge; consider standing down when ATR compresses.",
        ],
        builder=_atr_trailing_chart,
    ),
}


def _render_strategy_details(info: StrategyInfo, stats: Dict[str, float]) -> None:
    st.subheader(info.name)
    st.caption(f"{info.category} · Bias: {info.market_bias}")
    st.markdown(info.description)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Ideal conditions**")
        st.info(info.ideal_conditions)
        st.markdown("**Playbook**")
        for bullet in info.playbook:
            st.markdown(f"- {bullet}")
    with col2:
        st.markdown("**Risk management notes**")
        for note in info.risk_notes:
            st.markdown(f"- {note}")
        st.markdown("**Simulated signal stats**")
        stat_cols = st.columns(len(stats))
        for (label, value), col in zip(stats.items(), stat_cols):
            if "%" in label:
                display = f"{value:.1f}%"
            elif label == "Trades":
                display = f"{int(round(value))}"
            else:
                display = f"{value:.2f}"
            col.metric(label, display)


st.title("Strategy Insights")

st.markdown(
    """
Get a visual breakdown of the core playbooks powering the Aurora Desk workers.
Synthetic sample data illustrates how signals are generated, what conditions
are preferred, and the type of risk management typically employed.
"""
)

demo_df = _load_demo_market()

options = list(STRATEGY_LIBRARY.keys())
selected = st.selectbox(
    "Select a strategy to explore",
    options,
    format_func=lambda key: STRATEGY_LIBRARY[key].name,
)

strategy_info = STRATEGY_LIBRARY[selected]

chart, statistics = strategy_info.builder(demo_df)

_render_strategy_details(strategy_info, statistics)

st.plotly_chart(chart, use_container_width=True)
download_chart_as_png(chart, f"strategy_{selected}")

stats_df = pd.DataFrame([statistics])
st.caption(
    "Performance metrics derive from synthetic sample data and should only be used for comparative education, not live expectancy."
)
st.download_button(
    "Export strategy stats CSV",
    data=stats_df.to_csv(index=False),
    file_name=f"{selected}_stats.csv",
    mime="text/csv",
)
