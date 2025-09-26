"""Smoke test ensuring the trade engine operates in paper mode."""

from __future__ import annotations

import asyncio
from datetime import datetime
import json
from dataclasses import dataclass
import logging
import types
from typing import Any, Callable

import pytest

from ai_trader.broker.kraken_client import KrakenClient
from ai_trader.services.equity import EquityEngine
from ai_trader.services.risk import RiskManager
from ai_trader.services.trade_engine import TradeEngine
from ai_trader.services.trade_log import TradeLog
from ai_trader.services.types import MarketSnapshot, OpenPosition, TradeIntent
from ai_trader.workers.base import BaseWorker


@dataclass
class _DummyBroker:
    starting_equity: float = 10_000.0
    is_paper_trading: bool = True
    base_currency: str = "USD"
    fee_rate: float = 0.0

    def __post_init__(self) -> None:
        self.orders: list[tuple[str, str, float]] = []
        self.reduce_only: list[bool | None] = []
        self.cash_balance: float = float(self.starting_equity)
        self.min_notional: float = 0.0

    async def load_markets(self) -> None:  # pragma: no cover - trivial
        return None

    async def compute_equity(self, prices: dict[str, float]) -> tuple[float, dict[str, float]]:
        return self.cash_balance, {self.base_currency: self.cash_balance}

    async def fetch_balances(self) -> dict[str, float]:
        return {self.base_currency: self.cash_balance}

    async def ensure_market(self, symbol: str) -> None:  # pragma: no cover - trivial
        _ = symbol

    def min_order_value(self, symbol: str) -> float:  # pragma: no cover - trivial
        _ = symbol
        return self.min_notional

    async def place_order(
        self,
        symbol: str,
        side: str,
        cash_spent: float,
        *,
        reduce_only: bool | None = None,
    ) -> tuple[float, float]:
        price = 100.0
        quantity = float(cash_spent) / price
        self.orders.append((symbol, side, float(cash_spent)))
        self.reduce_only.append(reduce_only)
        if side == "buy":
            fee = float(cash_spent) * self.fee_rate
            self.cash_balance = max(0.0, self.cash_balance - float(cash_spent) - fee)
        else:
            fee = float(cash_spent) * self.fee_rate
            self.cash_balance += max(0.0, float(cash_spent) - fee)
        return price, quantity

    async def close_position(self, symbol: str, side: str, amount: float) -> tuple[float, float]:
        price = 100.0
        proceeds = price * float(amount)
        fee = proceeds * self.fee_rate
        self.cash_balance += max(0.0, proceeds - fee)
        return price, amount


class _DummyWebsocketManager:
    def __init__(self, symbol: str, price: float = 100.0) -> None:
        self._snapshot = MarketSnapshot(
            prices={symbol: price},
            history={symbol: [price, price * 1.01, price * 1.02]},
            candles={
                symbol: [
                    {
                        "close": price,
                        "open": price * 0.995,
                        "high": price * 1.01,
                        "low": price * 0.99,
                        "volume": 1.0,
                    }
                ]
            },
        )

    async def start(self) -> None:  # pragma: no cover - trivial
        return None

    async def stop(self) -> None:  # pragma: no cover - trivial
        return None

    def latest_snapshot(self) -> MarketSnapshot:
        return self._snapshot

    def update_price(self, symbol: str, price: float) -> None:
        self._snapshot.prices[symbol] = price
        self._snapshot.history[symbol].append(price)
        candle = self._snapshot.candles[symbol][-1]
        candle.update(
            {"close": price, "high": max(candle["high"], price), "low": min(candle["low"], price)}
        )


class _TestWorker(BaseWorker):
    name = "TestWorker"

    def __init__(self, symbol: str, trade_log: TradeLog) -> None:
        super().__init__([symbol], lookback=3, trade_log=trade_log)
        self._has_opened = False

    async def evaluate_signal(self, snapshot: MarketSnapshot) -> dict[str, str]:
        self.update_history(snapshot)
        signal = "buy" if not self._has_opened else "sell"
        return {self.symbols[0]: signal}

    async def generate_trade(
        self,
        symbol: str,
        signal: str | None,
        snapshot: MarketSnapshot,
        equity_per_trade: float,
        existing_position: OpenPosition | None = None,
    ) -> TradeIntent | None:
        price = snapshot.prices[symbol]
        if existing_position is None and signal == "buy":
            self._has_opened = True
            return TradeIntent(
                worker=self.name,
                action="OPEN",
                symbol=symbol,
                side="buy",
                cash_spent=equity_per_trade,
                entry_price=price,
                confidence=0.5,
            )
        if existing_position and signal == "sell":
            self._has_opened = False
            return TradeIntent(
                worker=self.name,
                action="CLOSE",
                symbol=symbol,
                side="sell",
                cash_spent=existing_position.cash_spent,
                entry_price=existing_position.entry_price,
                exit_price=price,
                confidence=0.5,
            )
        return None


class _ShortWorker(BaseWorker):
    name = "ShortWorker"

    def __init__(self, symbol: str, trade_log: TradeLog) -> None:
        super().__init__([symbol], lookback=3, trade_log=trade_log)
        self._has_opened = False

    async def evaluate_signal(self, snapshot: MarketSnapshot) -> dict[str, str]:
        self.update_history(snapshot)
        return {self.symbols[0]: "sell"}

    async def generate_trade(
        self,
        symbol: str,
        signal: str | None,
        snapshot: MarketSnapshot,
        equity_per_trade: float,
        existing_position: OpenPosition | None = None,
    ) -> TradeIntent | None:
        price = snapshot.prices[symbol]
        if not self._has_opened and existing_position is None and signal == "sell":
            self._has_opened = True
            return TradeIntent(
                worker=self.name,
                action="OPEN",
                symbol=symbol,
                side="sell",
                cash_spent=equity_per_trade,
                entry_price=price,
                confidence=0.5,
            )
        return None


class _StubExchange:
    """Minimal ccxt-like stub capturing reduce_only usage."""

    def __init__(self) -> None:
        self.order_calls: list[dict[str, Any]] = []

    def fetch_ticker(self, symbol: str) -> dict[str, Any]:  # pragma: no cover - trivial
        return {"last": 100.0}

    def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        self.order_calls.append(
            {
                "symbol": symbol,
                "type": order_type,
                "side": side,
                "amount": amount,
                "price": price,
                "params": params,
            }
        )
        return {"amount": amount, "average": 100.0}


class _StringCashWorker(BaseWorker):
    """Worker that deliberately returns string cash sizes to test coercion."""

    name = "StringCashWorker"

    def __init__(self, symbol: str, trade_log: TradeLog) -> None:
        super().__init__([symbol], lookback=3, trade_log=trade_log)
        self._submitted = False

    async def evaluate_signal(self, snapshot: MarketSnapshot) -> dict[str, str]:
        self.update_history(snapshot)
        return {self.symbols[0]: "buy"}

    async def generate_trade(
        self,
        symbol: str,
        signal: str | None,
        snapshot: MarketSnapshot,
        equity_per_trade: float,
        existing_position: OpenPosition | None = None,
    ) -> TradeIntent | None:
        if existing_position is not None or self._submitted or signal != "buy":
            return None
        self._submitted = True
        price = snapshot.prices[symbol]
        return TradeIntent(
            worker=self.name,
            action="OPEN",
            symbol=symbol,
            side="buy",
            cash_spent="3.59",
            entry_price=price,
            confidence=0.9,
        )


class _LargeCashWorker(BaseWorker):
    """Worker that requests oversized trades to trigger the $20 cap."""

    name = "LargeCashWorker"

    def __init__(self, symbol: str, trade_log: TradeLog) -> None:
        super().__init__([symbol], lookback=3, trade_log=trade_log)
        self._submitted = False

    async def evaluate_signal(self, snapshot: MarketSnapshot) -> dict[str, str]:
        self.update_history(snapshot)
        return {self.symbols[0]: "buy"}

    async def generate_trade(
        self,
        symbol: str,
        signal: str | None,
        snapshot: MarketSnapshot,
        equity_per_trade: float,
        existing_position: OpenPosition | None = None,
    ) -> TradeIntent | None:
        if existing_position is not None or self._submitted or signal != "buy":
            return None
        self._submitted = True
        price = snapshot.prices[symbol]
        return TradeIntent(
            worker=self.name,
            action="OPEN",
            symbol=symbol,
            side="buy",
            cash_spent=50.0,
            entry_price=price,
            confidence=0.9,
        )


class _LowConfidenceMLWorker(BaseWorker):
    """ML-style worker that should be skipped below the confidence floor."""

    name = "LowConfidenceMLWorker"

    def __init__(self, symbol: str, trade_log: TradeLog) -> None:
        super().__init__([symbol], lookback=3, trade_log=trade_log)
        self._submitted = False
        # Flag as ML worker so the trade engine applies ML-specific risk rules.
        self._ml_service = object()  # type: ignore[attr-defined]

    async def evaluate_signal(self, snapshot: MarketSnapshot) -> dict[str, str]:
        self.update_history(snapshot)
        return {self.symbols[0]: "buy"}

    async def generate_trade(
        self,
        symbol: str,
        signal: str | None,
        snapshot: MarketSnapshot,
        equity_per_trade: float,
        existing_position: OpenPosition | None = None,
    ) -> TradeIntent | None:
        if existing_position is not None or self._submitted or signal != "buy":
            return None
        self._submitted = True
        price = snapshot.prices[symbol]
        return TradeIntent(
            worker=self.name,
            action="OPEN",
            symbol=symbol,
            side="buy",
            cash_spent=equity_per_trade,
            entry_price=price,
            confidence=0.2,
        )


class _AtrSizingWorker(BaseWorker):
    """Worker that exposes ATR metadata so risk sizing can be asserted."""

    name = "AtrSizingWorker"

    def __init__(self, symbol: str, trade_log: TradeLog) -> None:
        super().__init__([symbol], lookback=3, trade_log=trade_log)
        self._submitted = False

    async def evaluate_signal(self, snapshot: MarketSnapshot) -> dict[str, str]:
        self.update_history(snapshot)
        return {self.symbols[0]: "buy"}

    async def generate_trade(
        self,
        symbol: str,
        signal: str | None,
        snapshot: MarketSnapshot,
        equity_per_trade: float,
        existing_position: OpenPosition | None = None,
    ) -> TradeIntent | None:
        price = snapshot.prices[symbol]
        if existing_position is not None:
            triggered, reason, risk_meta = self.check_risk_exit(symbol, price)
            if triggered:
                self.clear_risk_tracker(symbol)
                return TradeIntent(
                    worker=self.name,
                    action="CLOSE",
                    symbol=symbol,
                    side="sell",
                    cash_spent=existing_position.cash_spent,
                    entry_price=existing_position.entry_price,
                    exit_price=price,
                    confidence=0.0,
                    reason=reason,
                    metadata={
                        "trigger": reason,
                        **{k: v for k, v in risk_meta.items() if v is not None},
                    },
                )
            return None
        if self._submitted or signal != "buy":
            return None
        self._submitted = True
        risk_meta = self.prepare_entry_risk(symbol, "buy", price)
        metadata = {
            "atr": 1.0,
            **{k: v for k, v in risk_meta.items() if v is not None},
        }
        return TradeIntent(
            worker=self.name,
            action="OPEN",
            symbol=symbol,
            side="buy",
            cash_spent=equity_per_trade,
            entry_price=price,
            confidence=0.8,
            metadata=metadata,
        )


async def _run_trade_engine(tmp_path) -> None:
    """The trade engine should execute a simple open/close cycle in paper mode."""

    db_path = tmp_path / "engine.db"
    trade_log = TradeLog(db_path)
    broker = _DummyBroker()
    websocket_manager = _DummyWebsocketManager("BTC/USD")
    equity_engine = EquityEngine(trade_log, broker.starting_equity)
    risk_manager = RiskManager(
        {
            "max_drawdown_percent": 50,
            "daily_loss_limit_percent": 50,
            "max_position_duration_minutes": 5,
        }
    )
    worker = _TestWorker("BTC/USD", trade_log)

    engine = TradeEngine(
        broker=broker,
        websocket_manager=websocket_manager,
        workers=[worker],
        researchers=[],
        equity_engine=equity_engine,
        risk_manager=risk_manager,
        trade_log=trade_log,
        equity_allocation_percent=10.0,
        max_open_positions=1,
        refresh_interval=0.01,
        paper_trading=True,
        ml_service=None,
    )

    run_task = asyncio.create_task(engine.start())
    await asyncio.sleep(0.1)
    await engine.stop()
    await run_task

    assert broker.orders, "Paper broker should have recorded at least one order"


def test_trade_engine_paper_mode(tmp_path) -> None:
    asyncio.run(_run_trade_engine(tmp_path))


def test_rehydrate_open_positions(tmp_path) -> None:
    """Rehydrating broker positions should seed engine state and trade logs."""

    db_path = tmp_path / "rehydrate.db"
    trade_log = TradeLog(db_path)
    broker = _DummyBroker()
    websocket_manager = _DummyWebsocketManager("BTC/USD")
    equity_engine = EquityEngine(trade_log, broker.starting_equity)
    risk_manager = RiskManager(
        {
            "max_drawdown_percent": 50,
            "daily_loss_limit_percent": 50,
            "max_position_duration_minutes": 5,
        }
    )
    worker = _TestWorker("BTC/USD", trade_log)

    engine = TradeEngine(
        broker=broker,
        websocket_manager=websocket_manager,
        workers=[worker],
        researchers=[],
        equity_engine=equity_engine,
        risk_manager=risk_manager,
        trade_log=trade_log,
        equity_allocation_percent=10.0,
        max_open_positions=2,
        refresh_interval=0.01,
        paper_trading=True,
        ml_service=None,
    )

    broker_position = OpenPosition(
        worker="broker::unassigned",
        symbol="BTC/USD",
        side="buy",
        quantity=0.25,
        entry_price=100.0,
        cash_spent=25.0,
        opened_at=datetime.utcnow(),
    )

    asyncio.run(engine.rehydrate_open_positions([broker_position]))

    key = (worker.name, broker_position.symbol)
    assert key in engine._open_positions
    seeded = engine._open_positions[key]
    assert seeded.quantity == broker_position.quantity
    trades = list(trade_log.fetch_trades())
    assert any(row["reason"] == "rehydrated" for row in trades)
    events = list(trade_log.fetch_trade_events())
    assert events and events[0]["event"] == "rehydrate_open"

    # Rehydrating again with the same payload should not create duplicates.
    asyncio.run(engine.rehydrate_open_positions([broker_position]))
    assert len(list(trade_log.fetch_trades())) == len(trades)


async def _run_short_trade_engine(tmp_path, *, ml_enabled: bool = False) -> None:
    """Ensure short entries do not mark reduce_only on brokers allowing shorts."""

    db_path = tmp_path / "engine_short.db"
    trade_log = TradeLog(db_path)
    broker = _DummyBroker()
    websocket_manager = _DummyWebsocketManager("ETH/USD")
    equity_engine = EquityEngine(trade_log, broker.starting_equity)
    risk_manager = RiskManager(
        {
            "max_drawdown_percent": 50,
            "daily_loss_limit_percent": 50,
            "max_position_duration_minutes": 5,
        }
    )
    worker = _ShortWorker("ETH/USD", trade_log)
    if ml_enabled:
        worker._ml_service = object()  # type: ignore[attr-defined]

    engine = TradeEngine(
        broker=broker,
        websocket_manager=websocket_manager,
        workers=[worker],
        researchers=[],
        equity_engine=equity_engine,
        risk_manager=risk_manager,
        trade_log=trade_log,
        equity_allocation_percent=10.0,
        max_open_positions=1,
        refresh_interval=0.01,
        paper_trading=True,
        ml_service=None,
    )

    run_task = asyncio.create_task(engine.start())
    await asyncio.sleep(0.1)
    await engine.stop()
    await run_task

    assert not broker.orders, "Short trades should be blocked in long-only mode"


def test_trade_engine_short_sell_blocked(tmp_path, caplog) -> None:
    caplog.set_level(logging.INFO)
    asyncio.run(_run_short_trade_engine(tmp_path))
    assert "Short trade blocked" in caplog.text


def test_trade_engine_short_sell_logs_ml_block(tmp_path, caplog) -> None:
    caplog.set_level(logging.INFO)
    asyncio.run(_run_short_trade_engine(tmp_path, ml_enabled=True))
    assert "[RISK] Short signal blocked (symbol=ETH/USD, confidence=0.500)" in caplog.text


async def _run_engine_with_worker(
    tmp_path,
    worker_factory: Callable[[TradeLog], BaseWorker],
    *,
    starting_equity: float = 10_000.0,
    broker_factory: Callable[[], _DummyBroker] | None = None,
) -> tuple[list[tuple[str, str, float]], list[Any]]:
    """Run the trade engine with a supplied worker factory and return broker orders."""

    db_path = tmp_path / "engine_worker.db"
    trade_log = TradeLog(db_path)
    worker = worker_factory(trade_log)
    broker = broker_factory() if broker_factory else _DummyBroker(starting_equity=starting_equity)
    broker.cash_balance = starting_equity
    websocket_manager = _DummyWebsocketManager("BTC/USD")
    equity_engine = EquityEngine(trade_log, broker.starting_equity)
    risk_manager = RiskManager(
        {
            "max_drawdown_percent": 50,
            "daily_loss_limit_percent": 50,
            "max_position_duration_minutes": 5,
        }
    )
    engine = TradeEngine(
        broker=broker,
        websocket_manager=websocket_manager,
        workers=[worker],
        researchers=[],
        equity_engine=equity_engine,
        risk_manager=risk_manager,
        trade_log=trade_log,
        equity_allocation_percent=10.0,
        max_open_positions=1,
        refresh_interval=0.01,
        paper_trading=True,
        ml_service=None,
    )

    run_task = asyncio.create_task(engine.start())
    await asyncio.sleep(0.1)
    await engine.stop()
    await run_task

    trades = list(trade_log.fetch_trades())
    return broker.orders, trades


async def _run_string_cash_trade(
    tmp_path,
    starting_equity: float = 10_000.0,
) -> tuple[list[tuple[str, str, float]], list[Any]]:
    return await _run_engine_with_worker(
        tmp_path,
        lambda trade_log: _StringCashWorker("BTC/USD", trade_log),
        starting_equity=starting_equity,
    )


def test_trade_engine_casts_string_cash_from_workers(tmp_path) -> None:
    orders, trades = asyncio.run(_run_string_cash_trade(tmp_path))
    assert orders, "Engine should execute at least one order"
    _, _, cash_value = orders[-1]
    assert cash_value == pytest.approx(3.59)

    assert trades, "Trade log should capture the executed order"
    metadata = json.loads(trades[0]["metadata_json"])
    assert metadata.get("cash_floor_applied") is None
    assert metadata.get("original_cash_spent") == pytest.approx(3.59)


def test_trade_engine_dynamic_sizing_from_risk(tmp_path) -> None:
    async def runner() -> tuple[_DummyBroker, list[Any]]:
        db_path = tmp_path / "dynamic_sizing.db"
        trade_log = TradeLog(db_path)
        worker = _AtrSizingWorker("BTC/USD", trade_log)
        broker = _DummyBroker(starting_equity=200.0)
        broker.min_notional = 1.0
        websocket_manager = _DummyWebsocketManager("BTC/USD", price=1.0)
        equity_engine = EquityEngine(trade_log, broker.starting_equity)
        risk_manager = RiskManager({"risk_per_trade": 0.02, "atr_stop_loss_multiplier": 1.0})
        engine = TradeEngine(
            broker=broker,
            websocket_manager=websocket_manager,
            workers=[worker],
            researchers=[],
            equity_engine=equity_engine,
            risk_manager=risk_manager,
            trade_log=trade_log,
            equity_allocation_percent=100.0,
            max_open_positions=1,
            refresh_interval=0.01,
            paper_trading=True,
            min_cash_per_trade=0.0,
            max_cash_per_trade=0.0,
            trade_fee_percent=0.0,
        )
        run_task = asyncio.create_task(engine.start())
        await asyncio.sleep(0.1)
        await engine.stop()
        await run_task
        return broker, list(trade_log.fetch_trades())

    broker, trades = asyncio.run(runner())
    assert broker.orders, "Engine should submit at least one order"
    _, _, cash_value = broker.orders[-1]
    assert cash_value == pytest.approx(4.0, rel=1e-2)

    assert trades, "Trade log should capture the executed order"
    metadata = json.loads(trades[0]["metadata_json"])
    assert metadata.get("risk_amount") == pytest.approx(4.0, rel=1e-2)


def test_trade_engine_applies_trade_fees(tmp_path) -> None:
    fee_rate = 0.0026

    async def runner() -> tuple[_DummyBroker, list[Any]]:
        db_path = tmp_path / "fee_tracking.db"
        trade_log = TradeLog(db_path)
        worker = _TestWorker("BTC/USD", trade_log)
        broker = _DummyBroker(starting_equity=1_000.0, fee_rate=fee_rate)
        websocket_manager = _DummyWebsocketManager("BTC/USD")
        equity_engine = EquityEngine(trade_log, broker.starting_equity)
        risk_manager = RiskManager(
            {
                "max_drawdown_percent": 50,
                "daily_loss_limit_percent": 50,
                "risk_per_trade": 0.1,
            }
        )
        engine = TradeEngine(
            broker=broker,
            websocket_manager=websocket_manager,
            workers=[worker],
            researchers=[],
            equity_engine=equity_engine,
            risk_manager=risk_manager,
            trade_log=trade_log,
            equity_allocation_percent=10.0,
            max_open_positions=1,
            refresh_interval=0.01,
            paper_trading=True,
            min_cash_per_trade=0.0,
            max_cash_per_trade=0.0,
            trade_fee_percent=fee_rate,
        )
        run_task = asyncio.create_task(engine.start())
        await asyncio.sleep(0.2)
        await engine.stop()
        await run_task
        return broker, list(trade_log.fetch_trades())

    broker, trades = asyncio.run(runner())
    assert broker.orders, "Engine should submit orders with fees applied"
    assert trades, "Trade log should capture fills"
    ordered = sorted(trades, key=lambda row: row["timestamp"])
    open_rows = [row for row in ordered if row["exit_price"] is None]
    close_rows = [row for row in ordered if row["exit_price"] is not None]
    assert open_rows and close_rows, "Expected at least one open and one close trade"
    open_meta = json.loads(open_rows[0]["metadata_json"])
    close_meta = json.loads(close_rows[0]["metadata_json"])
    entry_fee = float(open_meta.get("entry_fee", 0.0))
    exit_fee = float(close_meta.get("exit_fee", 0.0))
    assert entry_fee == pytest.approx(open_meta["fees_total"], rel=1e-6)
    expected_exit_fee = close_meta["fill_price"] * close_meta["fill_quantity"] * fee_rate
    assert exit_fee == pytest.approx(expected_exit_fee, rel=1e-6)
    assert close_meta["fees_total"] == pytest.approx(entry_fee + exit_fee, rel=1e-6)
    expected_equity = 1_000.0 - (entry_fee + exit_fee)
    assert close_meta["pnl_usd"] == pytest.approx(-(entry_fee + exit_fee), rel=1e-6)


def test_trade_engine_enforces_atr_stop(tmp_path) -> None:
    async def runner() -> tuple[_DummyBroker, list[Any]]:
        db_path = tmp_path / "atr_stop.db"
        trade_log = TradeLog(db_path)
        worker = _AtrSizingWorker("BTC/USD", trade_log)
        broker = _DummyBroker(starting_equity=500.0)
        websocket_manager = _DummyWebsocketManager("BTC/USD", price=100.0)
        equity_engine = EquityEngine(trade_log, broker.starting_equity)
        risk_manager = RiskManager(
            {
                "risk_per_trade": 0.05,
                "atr_stop_loss_multiplier": 1.0,
                "max_drawdown_percent": 1000.0,
                "daily_loss_limit_percent": 1000.0,
            }
        )
        engine = TradeEngine(
            broker=broker,
            websocket_manager=websocket_manager,
            workers=[worker],
            researchers=[],
            equity_engine=equity_engine,
            risk_manager=risk_manager,
            trade_log=trade_log,
            equity_allocation_percent=50.0,
            max_open_positions=1,
            refresh_interval=0.01,
            paper_trading=True,
            min_cash_per_trade=0.0,
            max_cash_per_trade=0.0,
            trade_fee_percent=0.0,
        )
        run_task = asyncio.create_task(engine.start())
        await asyncio.sleep(0.1)
        websocket_manager.update_price("BTC/USD", 98.0)
        await asyncio.sleep(0.2)
        await engine.stop()
        await run_task
        return broker, list(trade_log.fetch_trades())

    broker, trades = asyncio.run(runner())
    assert broker.orders, "Engine should execute orders"
    ordered = sorted(trades, key=lambda row: row["timestamp"])
    close_rows = [row for row in ordered if row["exit_price"] is not None]
    assert close_rows, "Expected a close trade from ATR stop"
    close_meta = json.loads(close_rows[0]["metadata_json"])
    assert close_meta.get("reason") == "stop"
    assert close_meta.get("stop_price") is not None


def test_trade_engine_skips_trades_with_insufficient_balance(tmp_path, caplog) -> None:
    caplog.set_level(logging.INFO)
    orders, trades = asyncio.run(_run_string_cash_trade(tmp_path, starting_equity=5.0))
    assert orders, "Engine should downsize orders to allocation cap"
    _, _, cash_value = orders[-1]
    assert cash_value == pytest.approx(0.5)
    assert trades, "Trade log should capture the executed order"


def test_trade_engine_skips_low_confidence_ml_trades(tmp_path, caplog) -> None:
    caplog.set_level(logging.INFO)
    orders, trades = asyncio.run(
        _run_engine_with_worker(
            tmp_path,
            lambda trade_log: _LowConfidenceMLWorker("BTC/USD", trade_log),
        )
    )
    assert orders == []
    assert trades == []
    assert (
        "[ML] Skipped low-confidence trade (symbol=BTC/USD, confidence=0.200 < threshold=0.500)"
        in caplog.text
    )


async def _place_short_order_with_client() -> dict[str, Any]:
    """Invoke the Kraken client and capture the raw ccxt payload."""

    client = KrakenClient(
        api_key="",
        api_secret="",
        base_currency="USD",
        rest_rate_limit=0.0,
        paper_trading=False,
        allow_shorting=True,
    )
    exchange = _StubExchange()
    client._exchange = exchange  # type: ignore[attr-defined]

    async def _fake_with_retries(
        self: KrakenClient,
        func: Callable[..., Any],
        *args: Any,
        description: str,
        **kwargs: Any,
    ) -> Any:
        return func(*args, **kwargs)

    client._with_retries = types.MethodType(_fake_with_retries, client)

    await client.place_order("ETH/USD", "sell", 100.0, reduce_only=False)
    return exchange.order_calls[-1]


def test_kraken_client_sell_order_enforces_reduce_only_flag() -> None:
    order_payload = asyncio.run(_place_short_order_with_client())
    assert order_payload["params"] == {"reduce_only": True}
