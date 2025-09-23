"""Trading engine orchestrating workers, risk, and broker interactions."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timedelta
from typing import DefaultDict, Dict, Iterable, List, Tuple

from ai_trader.broker.kraken_client import KrakenClient
from ai_trader.broker.websocket_manager import KrakenWebsocketManager
from ai_trader.services.equity import EquityEngine
from ai_trader.services.logging import get_logger
from ai_trader.services.risk import RiskManager
from ai_trader.services.trade_log import TradeLog
from ai_trader.services.types import MarketSnapshot, OpenPosition, TradeIntent


class TradeEngine:
    """Coordinate market data, workers, and order execution."""

    def __init__(
        self,
        broker: KrakenClient,
        websocket_manager: KrakenWebsocketManager,
        workers: Iterable,
        researchers: Iterable,
        equity_engine: EquityEngine,
        risk_manager: RiskManager,
        trade_log: TradeLog,
        equity_allocation_percent: float,
        max_open_positions: int,
        refresh_interval: float,
    ) -> None:
        self._broker = broker
        self._websocket_manager = websocket_manager
        self._workers = list(workers)
        self._researchers = list(researchers)
        self._equity_engine = equity_engine
        self._risk_manager = risk_manager
        self._trade_log = trade_log
        self._equity_allocation_percent = equity_allocation_percent
        self._max_open_positions = max_open_positions
        self._refresh_interval = refresh_interval
        self._open_positions: Dict[Tuple[str, str], OpenPosition] = {}
        self._logger = get_logger(__name__)
        self._stop_event = asyncio.Event()
        self._control_flags: Dict[str, str] = {}
        self._kill_switch = False
        self._researcher_failures: DefaultDict[str, int] = defaultdict(int)

    async def start(self) -> None:
        await self._broker.load_markets()
        await self._websocket_manager.start()
        self._logger.info(
            "Trade engine started with %d workers and %d researcher(s)",
            len(self._workers),
            len(self._researchers),
        )
        await self._run()

    async def stop(self) -> None:
        self._stop_event.set()
        await self._websocket_manager.stop()

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            snapshot = self._websocket_manager.latest_snapshot()
            if not snapshot.prices:
                await asyncio.sleep(1.0)
                continue

            equity, balances = await self._broker.compute_equity(snapshot.prices)
            self._equity_engine.update(equity)
            equity_metrics = self._equity_engine.get_latest_metrics()
            self._trade_log.record_account_snapshot(balances, equity)
            equity_per_trade = equity * (self._equity_allocation_percent / 100)
            await self._run_researchers(snapshot, equity_metrics)
            self._update_control_flags()

            for worker in self._workers:
                if not getattr(worker, "active", True):
                    continue
                status_flag = self._control_flags.get(f"bot::{worker.name}")
                if status_flag in {"paused", "disabled"}:
                    self._logger.debug("Worker %s paused via control flag", worker.name)
                    continue
                try:
                    signals = await worker.evaluate_signal(snapshot)
                except Exception as exc:  # noqa: BLE001
                    self._logger.error("Worker %s evaluate failed: %s", worker.name, exc)
                    continue
                for symbol in worker.symbols:
                    signal = signals.get(symbol)
                    key = (worker.name, symbol)
                    open_position = self._open_positions.get(key)
                    state = worker.get_state_snapshot(symbol)
                    if open_position:
                        state.setdefault("indicators", {}).update(
                            {
                                "position": {
                                    "side": open_position.side,
                                    "entry": open_position.entry_price,
                                    "quantity": open_position.quantity,
                                    "cash": open_position.cash_spent,
                                }
                            }
                        )
                    self._trade_log.record_bot_state(worker.name, symbol, state)
                    try:
                        intent = await worker.generate_trade(
                            symbol,
                            signal,
                            snapshot,
                            equity_per_trade,
                            open_position,
                        )
                    except Exception as exc:  # noqa: BLE001
                        self._logger.error("Worker %s trade generation failed: %s", worker.name, exc)
                        continue
                    if intent is None:
                        continue
                    if self._kill_switch and intent.action == "OPEN":
                        self._logger.warning("Kill switch active. Blocking %s intent for %s", worker.name, symbol)
                        continue
                    if not self._risk_manager.check_trade(
                        intent,
                        equity_metrics,
                        len(self._open_positions),
                        self._max_open_positions,
                    ):
                        self._logger.warning("Risk manager blocked %s trade for %s", worker.name, symbol)
                        continue
                    await self._execute_intent(intent, key, open_position)

            await self._enforce_position_duration(snapshot)
            await asyncio.sleep(self._refresh_interval)

    async def _run_researchers(
        self,
        snapshot: MarketSnapshot,
        equity_metrics: Dict[str, float],
    ) -> None:
        if not self._researchers:
            return
        open_positions = list(self._open_positions.values())
        for researcher in self._researchers:
            try:
                await researcher.observe(snapshot, equity_metrics, open_positions)
                for symbol, state in researcher.get_all_state_snapshots().items():
                    self._trade_log.record_bot_state(researcher.name, symbol, state)
                self._researcher_failures.pop(researcher.name, None)
            except Exception as exc:  # noqa: BLE001
                self._researcher_failures[researcher.name] += 1
                failure_count = self._researcher_failures[researcher.name]
                self._logger.exception(
                    "Researcher %s failed (attempt %d): %s",
                    researcher.name,
                    failure_count,
                    exc,
                )
                if failure_count >= 3:
                    reset_hook = getattr(researcher, "reset", None) or getattr(
                        researcher, "reset_state", None
                    )
                    if callable(reset_hook):
                        try:
                            reset_hook()
                            self._researcher_failures.pop(researcher.name, None)
                            self._logger.warning(
                                "Researcher %s reset after repeated failures.",
                                researcher.name,
                            )
                        except Exception as reset_exc:  # noqa: BLE001
                            self._logger.error(
                                "Failed to reset researcher %s: %s",
                                researcher.name,
                                reset_exc,
                            )
                    else:
                        self._logger.error(
                            "Researcher %s has no reset hook; continuing after failure.",
                            researcher.name,
                        )

    async def _execute_intent(
        self,
        intent: TradeIntent,
        position_key: Tuple[str, str],
        existing_position: OpenPosition | None,
    ) -> None:
        if intent.action == "OPEN":
            await self._open_trade(intent, position_key)
        elif intent.action == "CLOSE" and existing_position is not None:
            await self._close_trade(intent, position_key, existing_position)
        else:
            self._logger.debug("Ignoring trade intent %s", intent)

    async def _open_trade(self, intent: TradeIntent, key: Tuple[str, str]) -> None:
        try:
            price, quantity = await self._broker.place_order(
                intent.symbol, intent.side, intent.cash_spent
            )
        except Exception as exc:  # noqa: BLE001 - broker errors should not crash engine
            self._logger.warning(
                "Order rejected for %s %s (cash %.2f): %s",
                intent.side.upper(),
                intent.symbol,
                intent.cash_spent,
                exc,
            )
            self._logger.debug("Failed intent payload: %s", intent, exc_info=True)
            return
        cost = price * quantity
        position = OpenPosition(
            worker=intent.worker,
            symbol=intent.symbol,
            side=intent.side,
            quantity=quantity,
            entry_price=price,
            cash_spent=cost,
        )
        self._open_positions[key] = position
        recorded_intent = TradeIntent(
            worker=intent.worker,
            action="OPEN",
            symbol=intent.symbol,
            side=intent.side,
            cash_spent=cost,
            entry_price=price,
            confidence=intent.confidence,
        )
        self._trade_log.record_trade(recorded_intent)
        self._logger.info("Trade opened: %s %s @ %.2f (qty %.6f)", intent.side.upper(), intent.symbol, price, quantity)

    async def _close_trade(self, intent: TradeIntent, key: Tuple[str, str], position: OpenPosition) -> None:
        try:
            price, quantity = await self._broker.close_position(
                intent.symbol, position.side, position.quantity
            )
        except Exception as exc:  # noqa: BLE001 - broker errors should not crash engine
            self._logger.warning(
                "Close order failed for %s %s (qty %.6f): %s",
                position.side.upper(),
                intent.symbol,
                position.quantity,
                exc,
            )
            self._logger.debug("Close intent payload: %s", intent, exc_info=True)
            return
        pnl = (price - position.entry_price) * position.quantity if position.side == "buy" else (position.entry_price - price) * position.quantity
        pnl_percent = pnl / position.cash_spent * 100 if position.cash_spent else 0.0
        recorded_intent = TradeIntent(
            worker=intent.worker,
            action="CLOSE",
            symbol=intent.symbol,
            side=intent.side,
            cash_spent=position.cash_spent,
            entry_price=position.entry_price,
            exit_price=price,
            pnl_percent=pnl_percent,
            pnl_usd=pnl,
            win_loss="win" if pnl > 0 else "loss",
        )
        self._trade_log.record_trade(recorded_intent)
        self._open_positions.pop(key, None)
        self._logger.info(
            "Trade closed: %s %s @ %.2f | PnL %.2f USD (%.2f%%)",
            intent.side.upper(),
            intent.symbol,
            price,
            pnl,
            pnl_percent,
        )

    async def _enforce_position_duration(self, snapshot: MarketSnapshot) -> None:
        now = datetime.utcnow()
        max_duration = timedelta(minutes=self._risk_manager.max_duration_minutes)
        for key, position in list(self._open_positions.items()):
            if now - position.opened_at > max_duration:
                price = snapshot.prices.get(position.symbol)
                if price is None:
                    continue
                pnl = position.unrealized_pnl(price)
                pnl_percent = pnl / position.cash_spent * 100 if position.cash_spent else 0.0
                intent = TradeIntent(
                    worker=key[0],
                    action="CLOSE",
                    symbol=position.symbol,
                    side="sell" if position.side == "buy" else "buy",
                    cash_spent=position.cash_spent,
                    entry_price=position.entry_price,
                    exit_price=price,
                    pnl_percent=pnl_percent,
                    pnl_usd=pnl,
                    win_loss="win" if pnl > 0 else "loss",
                )
                await self._close_trade(intent, key, position)

    def _update_control_flags(self) -> None:
        flags = self._trade_log.fetch_control_flags()
        if flags != self._control_flags:
            self._control_flags = flags
            kill_flag = flags.get("kill_switch") or flags.get("global::kill_switch")
            if kill_flag is None:
                self._kill_switch = False
            else:
                self._kill_switch = kill_flag.lower() in {"true", "on", "1"}
            for worker in self._workers:
                apply = getattr(worker, "apply_control_flags", None)
                if callable(apply):
                    apply(self._control_flags)
            for researcher in self._researchers:
                apply = getattr(researcher, "apply_control_flags", None)
                if callable(apply):
                    apply(self._control_flags)
