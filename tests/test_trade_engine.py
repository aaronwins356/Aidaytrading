"""Smoke test ensuring the trade engine operates in paper mode."""

from __future__ import annotations

import asyncio
from datetime import datetime
from dataclasses import dataclass
import types
from typing import Any, Callable

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

    def __post_init__(self) -> None:
        self.orders: list[tuple[str, str, float]] = []
        self.reduce_only: list[bool | None] = []

    async def load_markets(self) -> None:  # pragma: no cover - trivial
        return None

    async def compute_equity(self, prices: dict[str, float]) -> tuple[float, dict[str, float]]:
        return self.starting_equity, {"USD": self.starting_equity}

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
        return price, quantity

    async def close_position(self, symbol: str, side: str, amount: float) -> tuple[float, float]:
        return 100.0, amount


class _DummyWebsocketManager:
    def __init__(self, symbol: str) -> None:
        self._snapshot = MarketSnapshot(
            prices={symbol: 100.0},
            history={symbol: [100.0, 101.0, 102.0]},
            candles={symbol: [{"close": 100.0, "open": 99.5, "high": 101.0, "low": 99.0, "volume": 1.0}]},
        )

    async def start(self) -> None:  # pragma: no cover - trivial
        return None

    async def stop(self) -> None:  # pragma: no cover - trivial
        return None

    def latest_snapshot(self) -> MarketSnapshot:
        return self._snapshot


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


async def _run_trade_engine(tmp_path) -> None:
    """The trade engine should execute a simple open/close cycle in paper mode."""

    db_path = tmp_path / "engine.db"
    trade_log = TradeLog(db_path)
    broker = _DummyBroker()
    websocket_manager = _DummyWebsocketManager("BTC/USD")
    equity_engine = EquityEngine(trade_log, broker.starting_equity)
    risk_manager = RiskManager({"max_drawdown_percent": 50, "daily_loss_limit_percent": 50, "max_position_duration_minutes": 5})
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
    risk_manager = RiskManager({
        "max_drawdown_percent": 50,
        "daily_loss_limit_percent": 50,
        "max_position_duration_minutes": 5,
    })
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


async def _run_short_trade_engine(tmp_path) -> None:
    """Ensure short entries do not mark reduce_only on brokers allowing shorts."""

    db_path = tmp_path / "engine_short.db"
    trade_log = TradeLog(db_path)
    broker = _DummyBroker()
    websocket_manager = _DummyWebsocketManager("ETH/USD")
    equity_engine = EquityEngine(trade_log, broker.starting_equity)
    risk_manager = RiskManager({"max_drawdown_percent": 50, "daily_loss_limit_percent": 50, "max_position_duration_minutes": 5})
    worker = _ShortWorker("ETH/USD", trade_log)

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
    )

    run_task = asyncio.create_task(engine.start())
    await asyncio.sleep(0.1)
    await engine.stop()
    await run_task

    assert broker.orders, "Short broker should record the open order"
    assert broker.reduce_only[-1] is False


def test_trade_engine_short_sell_without_reduce_only(tmp_path) -> None:
    asyncio.run(_run_short_trade_engine(tmp_path))


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


def test_kraken_client_sell_order_omits_reduce_only_when_shorting_allowed() -> None:
    order_payload = asyncio.run(_place_short_order_with_client())
    assert order_payload["params"] == {}
