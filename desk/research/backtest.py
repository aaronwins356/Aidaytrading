"""Backtesting stubs for strategy smoke tests.

The routines here deliberately avoid live broker dependencies and instead use
in-memory candle streams to exercise strategy entry/exit logic.  They are not a
full-featured backtester but provide deterministic hooks for unit tests and
CI to verify that strategies remain self-contained and side-effect free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from desk.services.worker import Worker
from desk.strategies.base_strategy import Trade


@dataclass
class BacktestResult:
    """Container describing a toy backtest run."""

    entries: List[Dict[str, float]]
    exits: List[Dict[str, float]]
    final_position: Optional[Dict[str, float]]


def run_strategy_stub(
    strategy_name: str,
    candles: Iterable[Dict[str, float]],
    *,
    symbol: str = "BTC/USD",
    params: Optional[Dict[str, float]] = None,
    risk_budget: float = 100.0,
    max_history: Optional[int] = None,
) -> BacktestResult:
    """Run a lightweight candle stream against a strategy.

    The worker is initialised in isolation which ensures the strategy module
    can be imported and that its signal/extraction routines execute without
    raising exceptions.  The evaluation follows a simple state machine: the
    first approved intent opens a synthetic trade and subsequent candles call
    the strategy's ``check_exit`` routine to determine when to close it.
    """

    params = params or {}
    worker_config: Dict[str, object] = {
        "name": f"stub_{strategy_name}",
        "symbol": symbol,
        "strategy": strategy_name,
        "allocation": 1.0,
        "params": params,
        "risk_profile": {"initial_multiplier": 1.0, "floor_multiplier": 1.0},
    }

    worker = Worker(
        name=str(worker_config["name"]),
        symbol=symbol,
        strategy=strategy_name,
        params=worker_config,
    )

    entries: List[Dict[str, float]] = []
    exits: List[Dict[str, float]] = []
    active_trade: Optional[Dict[str, float]] = None
    history_limit = max_history or 500

    for candle in candles:
        worker.push_candle(candle, history_limit)
        if active_trade is None:
            intent = worker.build_intent(risk_budget)
            if intent and intent.approved and intent.qty > 0:
                entry = {
                    "timestamp": float(candle.get("timestamp", 0.0)),
                    "price": float(intent.price),
                    "side": 1.0 if intent.side == "BUY" else -1.0,
                    "qty": float(intent.qty),
                }
                entries.append(entry)
                active_trade = {
                    "side": intent.side,
                    "entry_price": float(intent.price),
                    "stop_loss": float(intent.stop_loss or intent.price),
                    "take_profit": float(intent.take_profit or intent.price),
                    "metadata": intent.plan_metadata,
                }
        else:
            df = worker._candles_df()
            trade = Trade(
                side=active_trade["side"].lower(),
                entry_price=float(active_trade["entry_price"]),
                stop_loss=float(active_trade["stop_loss"]),
                take_profit=float(active_trade["take_profit"]),
                meta=active_trade.get("metadata", {}),
            )
            exit_now, reason = worker.strategy.check_exit(trade, df)
            if exit_now:
                exit_price = float(df["close"].iloc[-1])
                pnl = (
                    exit_price - trade.entry_price
                    if trade.side == "buy"
                    else trade.entry_price - exit_price
                )
                exits.append(
                    {
                        "timestamp": float(df["timestamp"].iloc[-1]),
                        "price": exit_price,
                        "reason": reason or "exit",
                        "pnl": pnl,
                    }
                )
                active_trade = None

    return BacktestResult(entries=entries, exits=exits, final_position=active_trade)
