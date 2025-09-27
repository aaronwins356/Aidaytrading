"""Unit coverage for fee-aware trade execution paths."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Iterable, Tuple
from unittest.mock import MagicMock

import pytest

from ai_trader.services.trade_engine import TradeEngine
from ai_trader.services.types import OpenPosition, TradeIntent


@dataclass
class _StubBroker:
    price: float = 50.0

    def __post_init__(self) -> None:
        self.last_order: Tuple[str, str, float] | None = None
        self.closed_order: Tuple[str, float] | None = None

    starting_equity: float = 10_000.0
    base_currency: str = "USD"
    is_paper_trading: bool = True

    async def place_order(
        self, symbol: str, side: str, cash_spent: float, *, reduce_only: bool | None = None
    ) -> tuple[float, float]:
        quantity = float(cash_spent) / self.price
        self.last_order = (symbol, side, float(cash_spent))
        return self.price, quantity

    async def close_position(self, symbol: str, side: str, amount: float) -> tuple[float, float]:
        self.closed_order = (symbol, float(amount))
        return self.price, amount


class _StubTradeLog:
    def __init__(self) -> None:
        self.trades: list[TradeIntent] = []
        self.events: list[tuple[str, Dict[str, object]]] = []

    def fetch_trades(self) -> Iterable[tuple]:  # pragma: no cover - unused but required
        return []

    def has_trade_entry(
        self, worker: str, symbol: str, entry_price: float, cash_spent: float
    ) -> bool:  # pragma: no cover - unused in tests
        return False

    def record_trade(self, trade: TradeIntent) -> None:
        self.trades.append(trade)

    def record_trade_event(
        self, *, worker: str, symbol: str, event: str, details: Dict[str, object]
    ) -> None:
        self.events.append((event, details))

    def fetch_control_flags(self) -> Dict[str, str]:  # pragma: no cover - defaults only
        return {}

    def fetch_latest_risk_settings(self):  # pragma: no cover - compatibility shim
        return None


def _build_engine(trade_log: _StubTradeLog, broker: _StubBroker, fee_rate: float) -> TradeEngine:
    risk_manager = MagicMock()
    risk_manager.config_dict.return_value = {}
    equity_engine = MagicMock()
    websocket_manager = MagicMock()
    engine = TradeEngine(
        broker=broker,
        websocket_manager=websocket_manager,
        workers=[],
        researchers=[],
        equity_engine=equity_engine,
        risk_manager=risk_manager,
        trade_log=trade_log,
        equity_allocation_percent=1.0,
        max_open_positions=10,
        refresh_interval=1.0,
        paper_trading=True,
        trade_fee_percent=fee_rate,
    )
    return engine


@pytest.mark.asyncio
async def test_open_trade_applies_fee():
    broker = _StubBroker(price=50.0)
    trade_log = _StubTradeLog()
    engine = _build_engine(trade_log, broker, fee_rate=0.0026)

    intent = TradeIntent(
        worker="worker",
        action="OPEN",
        symbol="BTC/USD",
        side="buy",
        cash_spent=100.0,
        entry_price=50.0,
        confidence=0.8,
    )

    await engine._open_trade(intent, (intent.worker, intent.symbol))

    assert broker.last_order is not None
    _, _, spendable_cash = broker.last_order
    # Executed notional must exclude fees, hence smaller than nominal cash target.
    assert spendable_cash < 100.0

    assert trade_log.trades, "expected trade log to store open trade"
    recorded = trade_log.trades[0]
    entry_fee = recorded.metadata["entry_fee"]  # type: ignore[index]
    fill_quantity = recorded.metadata["fill_quantity"]  # type: ignore[index]
    assert fill_quantity < (100.0 / broker.price)
    expected_fee = float(
        Decimal(str(fill_quantity)) * Decimal(str(broker.price)) * Decimal("0.0026")
    )
    assert entry_fee == pytest.approx(expected_fee, rel=1e-6)
    # Total cash captured should approximate the configured target (net of rounding).
    assert recorded.cash_spent == pytest.approx(100.0, rel=1e-5)


@pytest.mark.asyncio
async def test_close_trade_applies_fee():
    broker = _StubBroker(price=60.0)
    trade_log = _StubTradeLog()
    engine = _build_engine(trade_log, broker, fee_rate=0.0026)

    position = OpenPosition(
        worker="worker",
        symbol="BTC/USD",
        side="buy",
        quantity=1.5,
        entry_price=50.0,
        cash_spent=75.0 + (75.0 * 0.0026),
        fees_paid=75.0 * 0.0026,
    )
    engine._open_positions[(position.worker, position.symbol)] = position

    intent = TradeIntent(
        worker="worker",
        action="CLOSE",
        symbol="BTC/USD",
        side="sell",
        cash_spent=position.cash_spent,
        entry_price=position.entry_price,
        exit_price=broker.price,
        confidence=0.5,
    )

    await engine._close_trade(intent, (intent.worker, intent.symbol), position)

    assert trade_log.trades, "expected close to be recorded"
    recorded = trade_log.trades[0]
    exit_fee = recorded.metadata["exit_fee"]  # type: ignore[index]
    # Exit fee should reflect taker fee on the exit notional.
    expected_exit_fee = broker.price * position.quantity * 0.0026
    assert exit_fee == pytest.approx(expected_exit_fee, rel=1e-6)
    pnl = recorded.pnl_usd or 0.0
    gross = (broker.price - position.entry_price) * position.quantity
    expected_pnl = gross - position.fees_paid - expected_exit_fee
    assert pnl == pytest.approx(expected_pnl, rel=1e-6)
