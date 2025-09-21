from __future__ import annotations

import pandas as pd

from desk.strategies.base_strategy import Trade
from desk.strategies.momentum import MomentumStrategy


def _df(prices):
    return pd.DataFrame({"close": list(prices)})


def test_momentum_strategy_buy_exit_with_uppercase_side():
    strategy = MomentumStrategy("BTC/USD", params={"length": 2})
    df = _df([110.0, 112.0, 111.0, 100.0])
    trade = Trade(
        side="BUY",
        entry_price=111.0,
        stop_loss=90.0,
        take_profit=150.0,
        meta={},
    )

    exit_now, reason = strategy.check_exit(trade, df)

    assert exit_now is True
    assert reason == "Momentum faded"


def test_momentum_strategy_sell_exit_with_uppercase_side():
    strategy = MomentumStrategy("BTC/USD", params={"length": 2})
    df = _df([100.0, 99.0, 98.0, 120.0])
    trade = Trade(
        side="SELL",
        entry_price=98.0,
        stop_loss=80.0,
        take_profit=200.0,
        meta={},
    )

    exit_now, reason = strategy.check_exit(trade, df)

    assert exit_now is True
    assert reason == "Momentum faded"
