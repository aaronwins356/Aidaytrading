"""Integration tests exercising the live trading loop with stub services."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai_trader.api_service import get_status, reset_services
from ai_trader.main import StartTradingOverrides, start_trading
from ai_trader.services.monitoring import get_monitoring_center
from ai_trader.services.runtime_state import RuntimeStateStore
from ai_trader.services.trade_log import TradeLog
from ai_trader.services.types import MarketSnapshot, TradeIntent
from ai_trader.workers.base import BaseWorker


class StubBroker:
    """In-memory broker that simulates balance changes for orders."""

    base_currency = "USD"
    starting_equity = 10_000.0
    is_paper_trading = True

    def __init__(self) -> None:
        self._balances: dict[str, float] = {self.base_currency: self.starting_equity}
        self._latest_prices: dict[str, float] = {}
        self.orders: list[tuple[str, str, float, float, float]] = []

    async def load_markets(self) -> None:  # pragma: no cover - interface compatibility
        return None

    async def ensure_market(self, symbol: str) -> None:  # pragma: no cover - always available
        self._latest_prices.setdefault(symbol, 0.0)

    def min_order_value(self, symbol: str) -> float:
        return 10.0

    async def fetch_open_positions(self) -> list[object]:
        return []

    async def fetch_balances(self) -> dict[str, float]:
        return dict(self._balances)

    async def compute_equity(self, prices: dict[str, float]) -> tuple[float, dict[str, float]]:
        equity = float(self._balances.get(self.base_currency, 0.0))
        for asset, amount in self._balances.items():
            if asset == self.base_currency or amount == 0:
                continue
            pair = f"{asset}/{self.base_currency}"
            price = prices.get(pair) or self._latest_prices.get(pair, 0.0)
            equity += float(amount) * float(price)
        return equity, dict(self._balances)

    async def place_order(
        self, symbol: str, side: str, cash_spent: float, *, reduce_only: bool | None = None
    ) -> tuple[float, float]:
        del reduce_only  # unused in the stub
        price = self._latest_prices.get(symbol, 50_000.0)
        if price <= 0:
            price = 50_000.0
        quantity = float(cash_spent) / float(price)
        base, quote = symbol.split("/")
        self._balances[quote] = float(self._balances.get(quote, 0.0) - float(cash_spent))
        if side == "buy":
            self._balances[base] = float(self._balances.get(base, 0.0) + quantity)
        else:
            self._balances[base] = float(self._balances.get(base, 0.0) - quantity)
            self._balances[quote] = float(self._balances.get(quote, 0.0) + float(cash_spent))
        self.orders.append((symbol, side, float(cash_spent), float(price), quantity))
        return float(price), float(quantity)

    async def close_position(self, symbol: str, side: str, quantity: float) -> tuple[float, float]:
        price = self._latest_prices.get(symbol, 50_000.0)
        base, quote = symbol.split("/")
        if side == "buy":
            self._balances[base] = float(self._balances.get(base, 0.0) - quantity)
            self._balances[quote] = float(self._balances.get(quote, 0.0) + quantity * price)
        else:
            self._balances[base] = float(self._balances.get(base, 0.0) + quantity)
            self._balances[quote] = float(self._balances.get(quote, 0.0) - quantity * price)
        return float(price), float(quantity)

    def update_price(self, symbol: str, price: float) -> None:
        self._latest_prices[symbol] = float(price)


class StubWebsocketManager:
    """Deterministic ticker feed for a single market pair."""

    def __init__(self, symbol: str, broker: StubBroker, interval: float = 0.5) -> None:
        self.symbols = [symbol]
        self._broker = broker
        self._interval = interval
        self._prices = deque([49_900.0, 50_050.0, 50_125.0, 50_200.0], maxlen=64)
        self._history = deque(maxlen=120)
        self._candles = deque(maxlen=120)
        self._snapshot = MarketSnapshot(prices={}, history={})
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._monitoring = get_monitoring_center()

    async def start(self) -> None:
        if self._task is None:
            self._stop_event.clear()
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop_event.set()
        with suppress(asyncio.CancelledError):
            await self._task
        self._task = None
        self._monitoring.record_event(
            "websocket_stopped",
            "INFO",
            "Stub websocket manager stopped",
            metadata={"symbols": list(self.symbols)},
        )

    def latest_snapshot(self) -> MarketSnapshot:
        return self._snapshot

    async def _run(self) -> None:
        symbol = self.symbols[0]
        index = 0
        self._monitoring.record_event(
            "websocket_connected",
            "INFO",
            "Stub websocket manager connected",
            metadata={"symbols": list(self.symbols)},
        )
        while not self._stop_event.is_set():
            price = list(self._prices)[index % len(self._prices)]
            index += 1
            self._history.append(price)
            candle = {
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1.0,
            }
            self._candles.append(candle)
            prices = {symbol: price}
            self._broker.update_price(symbol, price)
            self._snapshot = MarketSnapshot(
                prices=prices,
                history={symbol: list(self._history)},
                candles={symbol: list(self._candles)},
            )
            await asyncio.sleep(self._interval)


class StubNotifier:
    """Minimal Telegram notifier drop-in used for assertions."""

    def __init__(self, heartbeat_interval: float = 1.0) -> None:
        self.startup_payloads: list[dict[str, object]] = []
        self.trade_alerts: list[TradeIntent] = []
        self.heartbeats: list[float] = []
        self.watchdog_alerts: list[tuple[float, object | None]] = []
        self._heartbeat_interval = heartbeat_interval
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        if self._task is not None:
            return
        self._running = True
        self._task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _heartbeat_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self._heartbeat_interval)
            await self.send_heartbeat()

    async def send_startup_heartbeat(self, **payload: object) -> None:
        self.startup_payloads.append(payload)

    async def send_trade_alert(self, trade: TradeIntent) -> None:
        self.trade_alerts.append(trade)

    async def send_heartbeat(self) -> None:
        self.heartbeats.append(asyncio.get_running_loop().time())

    async def send_error(self, error: object) -> None:  # pragma: no cover - unused defensive stub
        self.trade_alerts.append(error)  # type: ignore[arg-type]

    async def send_watchdog_alert(self, timeout_seconds: float, last_update: object | None) -> None:
        self.watchdog_alerts.append((timeout_seconds, last_update))


class StubWatchdog:
    """Thread-less watchdog capturing alert attempts."""

    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.alerts: list[tuple[float, object | None]] = []

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def trigger(self, timeout: float, last_update: object | None) -> None:
        self.alerts.append((timeout, last_update))


class _StubMLService:
    """Minimal ML service implementation used for dependency injection."""

    default_threshold = 0.5

    def latest_features(self, symbol: str) -> dict[str, float]:  # pragma: no cover - deterministic
        del symbol
        return {}

    def register_trade_open(self, *args: object, **kwargs: object) -> None:  # pragma: no cover
        del args, kwargs

    def register_trade_close(self, *args: object, **kwargs: object) -> None:  # pragma: no cover
        del args, kwargs

    def latest_validation_metrics(self) -> dict[str, object]:  # pragma: no cover - deterministic
        return {}


class StubWorker(BaseWorker):
    """Deterministic worker that opens a single long trade on the first tick."""

    name = "StubAlpha"
    long_only = True

    def __init__(self, symbol: str) -> None:
        super().__init__([symbol], lookback=1, config={"warmup_candles": 1})
        self._opened = False

    async def evaluate_signal(self, snapshot: MarketSnapshot) -> dict[str, str]:
        self.update_history(snapshot)
        symbol = self.symbols[0]
        if not self.is_ready(symbol):
            return {symbol: "warmup"}
        if not self._opened:
            return {symbol: "buy"}
        return {symbol: "hold"}

    async def generate_trade(
        self,
        symbol: str,
        signal: str | None,
        snapshot: MarketSnapshot,
        equity_per_trade: float,
        existing_position: object | None = None,
    ) -> TradeIntent | None:
        del equity_per_trade
        if signal == "buy" and existing_position is None and not self._opened:
            self._opened = True
            price = float(snapshot.prices.get(symbol, 50_000.0))
            return TradeIntent(
                worker=self.name,
                action="OPEN",
                symbol=symbol,
                side="buy",
                cash_spent=100.0,
                entry_price=price,
                confidence=0.9,
            )
        return None


@dataclass(slots=True)
class RuntimeHarness:
    broker: StubBroker
    websocket: StubWebsocketManager
    notifier: StubNotifier
    trade_log: TradeLog
    runtime_state: RuntimeStateStore
    stop_event: asyncio.Event
    overrides: StartTradingOverrides
    watchdog_factory: Callable[..., StubWatchdog]
    watchdogs: list[StubWatchdog]


@pytest.fixture
def runtime_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RuntimeHarness:
    data_dir = tmp_path / "runtime"
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "trades.db"
    state_path = data_dir / "state.json"
    monkeypatch.setattr("ai_trader.main.DATA_DIR", data_dir)
    monkeypatch.setattr("ai_trader.main.DB_PATH", db_path)
    reset_services(state_file=state_path)
    monitoring = get_monitoring_center()
    monitoring.reset()

    broker = StubBroker()
    websocket = StubWebsocketManager("BTC/USD", broker, interval=0.2)
    notifier = StubNotifier(heartbeat_interval=0.5)
    trade_log = TradeLog(db_path)
    runtime_state = RuntimeStateStore(state_path)
    stop_event = asyncio.Event()
    watchdogs: list[StubWatchdog] = []
    ml_service = _StubMLService()

    def watchdog_factory(*_: object) -> StubWatchdog:
        watchdog = StubWatchdog()
        watchdog.start()
        watchdogs.append(watchdog)
        return watchdog

    overrides = StartTradingOverrides(
        broker=broker,
        websocket_manager=websocket,
        notifier=notifier,
        trade_log=trade_log,
        runtime_state=runtime_state,
        workers=[StubWorker("BTC/USD")],
        researchers=[],
        watchdog_factory=watchdog_factory,
        stop_event=stop_event,
        skip_validation=True,
        skip_warm_start=True,
        install_signal_handlers=False,
        ml_service=ml_service,
    )
    return RuntimeHarness(
        broker=broker,
        websocket=websocket,
        notifier=notifier,
        trade_log=trade_log,
        runtime_state=runtime_state,
        stop_event=stop_event,
        overrides=overrides,
        watchdog_factory=watchdog_factory,
        watchdogs=watchdogs,
    )


def _base_config() -> dict[str, object]:
    return {
        "trading": {
            "symbols": ["BTC/USD"],
            "paper_trading": True,
            "paper_starting_equity": 10_000,
            "equity_allocation_percent": 50,
            "max_open_positions": 2,
            "trade_confidence_min": 0.0,
        },
        "workers": {
            "refresh_interval_seconds": 0.2,
            "definitions": {},
        },
        "risk": {},
    }


async def _run_trading_for(duration: float, harness: RuntimeHarness) -> None:
    args = argparse.Namespace(mode="trade")
    config = _base_config()
    task = asyncio.create_task(start_trading(args, config, overrides=harness.overrides))
    try:
        await asyncio.sleep(duration)
    finally:
        harness.stop_event.set()
    await asyncio.wait_for(task, timeout=duration + 5)


async def _wait_for(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if condition():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("Condition not met within timeout")


@pytest.mark.asyncio
async def test_runtime_executes_trades(runtime_harness: RuntimeHarness) -> None:
    await _run_trading_for(3.0, runtime_harness)
    await _wait_for(lambda: bool(runtime_harness.broker.orders))
    trades = list(runtime_harness.trade_log.fetch_trades())
    assert trades, "expected at least one trade logged"

    status = await get_status()
    assert status.get("balance") is not None
    assert status.get("runtime_degraded") is False

    events = get_monitoring_center().recent_events()
    assert any(event["event_type"] == "websocket_connected" for event in events)


@pytest.mark.asyncio
async def test_runtime_notifier_and_watchdog(runtime_harness: RuntimeHarness) -> None:
    await _run_trading_for(4.0, runtime_harness)
    await _wait_for(lambda: bool(runtime_harness.notifier.startup_payloads))
    await _wait_for(lambda: len(runtime_harness.notifier.heartbeats) >= 1)

    assert runtime_harness.notifier.trade_alerts, "trade alert should be emitted"
    assert runtime_harness.notifier.watchdog_alerts == []
    assert runtime_harness.watchdogs, "watchdog factory should be invoked"
    assert runtime_harness.watchdogs[0].alerts == []
    assert runtime_harness.watchdogs[0].started is True
    assert runtime_harness.watchdogs[0].stopped is True

    events = get_monitoring_center().recent_events()
    assert all(event["event_type"] != "watchdog_stall" for event in events)
