"""Trading engine orchestrating workers, risk, and broker interactions."""

from __future__ import annotations

import asyncio
from collections import defaultdict
import json
import math
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
        paper_trading: bool,
        min_cash_per_trade: float = 10.0,
        max_cash_per_trade: float = 20.0,
        trade_confidence_min: float = 0.5,
    ) -> None:
        self._broker = broker
        self._websocket_manager = websocket_manager
        self._workers = list(workers)
        self._researchers = list(researchers)
        self._equity_engine = equity_engine
        self._risk_manager = risk_manager
        self._trade_log = trade_log
        self._equity_allocation_percent = float(equity_allocation_percent)
        self._max_open_positions = max_open_positions
        self._refresh_interval = float(refresh_interval)
        self._open_positions: Dict[Tuple[str, str], OpenPosition] = {}
        self._logger = get_logger(__name__)
        self._stop_event = asyncio.Event()
        self._control_flags: Dict[str, str] = {}
        self._kill_switch = False
        self._researcher_failures: DefaultDict[str, int] = defaultdict(int)
        self._signal_only_mode = False
        self._signal_only_reason = ""
        self._paper_trading = paper_trading
        # Kraken rejects orders whose notional value is below the exchange minimum (~$10).
        # Enforcing a configurable floor here keeps all strategies broker-compliant even
        # when their internal position sizing logic suggests smaller allocations.
        self._min_cash_per_trade = max(0.0, float(min_cash_per_trade))
        self._max_cash_per_trade = max(self._min_cash_per_trade, float(max_cash_per_trade))
        self._trade_confidence_min = max(0.0, min(1.0, float(trade_confidence_min)))
        self._base_currency = getattr(broker, "base_currency", "USD")

    async def rehydrate_open_positions(self, positions: Iterable[OpenPosition]) -> None:
        """Reconcile broker open positions with the in-memory engine state."""

        broker_positions = list(positions or [])
        if not broker_positions:
            self._logger.info("Broker reports no open positions to rehydrate.")
            return

        worker_symbol_map: Dict[str, set[str]] = {}
        for worker in self._workers:
            name = getattr(worker, "name", "") or worker.__class__.__name__
            symbols = {str(symbol).upper() for symbol in getattr(worker, "symbols", [])}
            if symbols:
                worker_symbol_map[name] = symbols

        trade_history = list(self._trade_log.fetch_trades())
        reconciled = 0

        def match_worker(position: OpenPosition, candidates: List[str]) -> str | None:
            """Best effort worker resolution based on trade history and symbols."""

            if position.worker in candidates:
                return position.worker

            for row in trade_history:
                row_worker = str(row["worker"])
                if candidates and row_worker not in candidates:
                    continue
                if str(row["symbol"]).upper() != position.symbol.upper():
                    continue
                entry_price = float(row["entry_price"])
                cash_spent = float(row["cash_spent"])
                # Allow small float variance because fills are rounded differently
                # across Kraken endpoints and historical trade logs.
                if math.isclose(entry_price, position.entry_price, rel_tol=1e-5, abs_tol=1e-5) and math.isclose(
                    cash_spent, position.cash_spent, rel_tol=1e-4, abs_tol=1e-4
                ):
                    return row_worker
                metadata_json = None
                if hasattr(row, "keys") and "metadata_json" in row.keys():
                    metadata_json = row["metadata_json"]
                metadata: Dict[str, object] = {}
                if metadata_json:
                    try:
                        metadata = json.loads(metadata_json)
                    except (TypeError, ValueError):
                        metadata = {}
                quantity = metadata.get("fill_quantity") or metadata.get("quantity")
                if quantity is not None and math.isclose(
                    float(quantity), position.quantity, rel_tol=1e-6, abs_tol=1e-6
                ):
                    return row_worker
            return candidates[0] if candidates else None

        for broker_position in broker_positions:
            symbol_key = broker_position.symbol.upper()
            candidates = [
                worker for worker, symbols in worker_symbol_map.items() if symbol_key in symbols
            ]
            worker_name = match_worker(broker_position, candidates)
            if worker_name is None:
                self._logger.warning(
                    "Unable to reconcile broker position %s %s; skipping rehydration",
                    broker_position.side.upper(),
                    broker_position.symbol,
                )
                continue

            key = (worker_name, broker_position.symbol)
            if key in self._open_positions:
                self._logger.debug(
                    "Open position for %s on %s already tracked; skipping broker rehydrate",
                    worker_name,
                    broker_position.symbol,
                )
                continue

            position = OpenPosition(
                worker=worker_name,
                symbol=broker_position.symbol,
                side=broker_position.side,
                quantity=broker_position.quantity,
                entry_price=broker_position.entry_price,
                cash_spent=broker_position.cash_spent,
                opened_at=broker_position.opened_at,
            )
            self._open_positions[key] = position
            reconciled += 1

            if not self._trade_log.has_trade_entry(
                worker_name, position.symbol, position.entry_price, position.cash_spent
            ):
                intent = TradeIntent(
                    worker=worker_name,
                    action="OPEN",
                    symbol=position.symbol,
                    side=position.side,
                    cash_spent=position.cash_spent,
                    entry_price=position.entry_price,
                    reason="rehydrated",
                    metadata={
                        "mode": "rehydrated",
                        "quantity": position.quantity,
                        "opened_at": position.opened_at.isoformat(),
                    },
                )
                self._trade_log.record_trade(intent)
                self._trade_log.record_trade_event(
                    worker=worker_name,
                    symbol=position.symbol,
                    event="rehydrate_open",
                    details={
                        "price": position.entry_price,
                        "quantity": position.quantity,
                        "source": "broker",
                    },
                )

        if reconciled:
            self._logger.info("Rehydrated %d open position(s) from broker state", reconciled)
        else:
            self._logger.info("No broker open positions were reconciled with active workers")

    async def start(self) -> None:
        await self._broker.load_markets()
        for symbol in self._active_symbols():
            try:
                await self._broker.ensure_market(symbol)
            except Exception as exc:  # noqa: BLE001 - defensive logging
                self._logger.warning("Failed to verify market metadata for %s: %s", symbol, exc)
        await self._websocket_manager.start()
        self._logger.info(
            "Trade engine started with %d workers and %d researcher(s)",
            len(self._workers),
            len(self._researchers),
        )
        mode = "PAPER" if self._paper_trading else "LIVE"
        if self._signal_only_mode:
            self._logger.warning(
                "Signal-only mode active (%s). Orders will not reach the broker.",
                self._signal_only_reason,
            )
        self._logger.info("Operating in %s trading mode", mode)
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
            equity = float(equity)
            normalized_balances = {asset: float(amount) for asset, amount in balances.items()}
            self._equity_engine.update(equity, self._broker.starting_equity)
            equity_metrics = self._equity_engine.get_latest_metrics()
            self._trade_log.record_account_snapshot(normalized_balances, equity)
            equity_per_trade = equity * (self._equity_allocation_percent / 100.0)
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
                    intent = self._apply_trade_size_bounds(intent)
                    if self._violates_long_only(worker, intent, open_position):
                        continue
                    if (
                        intent.action == "OPEN"
                        and self._trade_confidence_min > 0.0
                        and self._is_ml_worker(worker)
                        and float(intent.confidence) < self._trade_confidence_min
                    ):
                        self._logger.info(
                            "[ML] Skipped low-confidence trade (symbol=%s, confidence=%.3f < threshold=%.3f)",
                            intent.symbol,
                            float(intent.confidence),
                            self._trade_confidence_min,
                        )
                        continue
                    if intent.action == "OPEN":
                        funding_checked = await self._ensure_sufficient_balance(intent)
                        if funding_checked is None:
                            continue
                        intent = funding_checked
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

    def enable_signal_only_mode(self, reason: str) -> None:
        """Prevent order placement while allowing research and logging."""

        self._signal_only_mode = True
        self._signal_only_reason = reason
        self._logger.warning("Signal-only mode enabled: %s", reason)

    def _active_symbols(self) -> list[str]:
        """Return the union of symbols known to the engine and websocket feed."""

        ordered: list[str] = []
        seen: set[str] = set()
        for symbol in getattr(self._websocket_manager, "symbols", []):
            if symbol not in seen:
                seen.add(symbol)
                ordered.append(symbol)
        for collection in (self._workers, self._researchers):
            for worker in collection:
                for symbol in getattr(worker, "symbols", []):
                    if symbol not in seen:
                        seen.add(symbol)
                        ordered.append(symbol)
        return ordered

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
            if self._signal_only_mode:
                self._logger.info(
                    "[TRADE] %s blocked from opening %s in signal-only mode (%s)",
                    intent.worker,
                    intent.symbol,
                    self._signal_only_reason,
                )
                return
            await self._open_trade(intent, position_key)
        elif intent.action == "CLOSE" and existing_position is not None:
            await self._close_trade(intent, position_key, existing_position)
        else:
            self._logger.debug("Ignoring trade intent %s", intent)

    def _is_ml_worker(self, worker: object) -> bool:
        return getattr(worker, "_ml_service", None) is not None

    def _violates_long_only(
        self,
        worker: object,
        intent: TradeIntent,
        existing_position: OpenPosition | None,
    ) -> bool:
        if intent.action == "OPEN" and intent.side == "sell":
            confidence = float(intent.confidence)
            if self._is_ml_worker(worker):
                self._logger.info(
                    "[RISK] Short signal blocked (symbol=%s, confidence=%.3f)",
                    intent.symbol,
                    confidence,
                )
            else:
                self._logger.info(
                    "Short trade blocked by long-only policy (%s -> %s)",
                    getattr(worker, "name", worker.__class__.__name__),
                    intent.symbol,
                )
            return True
        if intent.action == "CLOSE" and intent.side == "sell":
            if existing_position is None or existing_position.side != "buy":
                self._logger.info(
                    "Ignoring SELL close for %s – no matching long position to unwind",
                    intent.symbol,
                )
                return True
        return False

    async def _ensure_sufficient_balance(self, intent: TradeIntent) -> TradeIntent | None:
        try:
            balances = await self._broker.fetch_balances()
        except Exception as exc:  # noqa: BLE001 - defensive logging
            self._logger.warning("Failed to fetch balances prior to order: %s", exc)
            return None
        available = float(balances.get(self._base_currency, 0.0))
        required = float(intent.cash_spent)
        if available < self._min_cash_per_trade:
            self._logger.info(
                "[RISK] Skipped trade for %s – insufficient funds",
                intent.symbol,
            )
            return None
        if required > available:
            metadata = dict(intent.metadata or {})
            metadata.update(
                {
                    "balance_capped": True,
                    "available_cash": available,
                }
            )
            self._logger.info(
                "Adjusting %s order size to available cash %.2f (requested %.2f)",
                intent.symbol,
                available,
                required,
            )
            intent.metadata = metadata
            intent.cash_spent = available
        return intent

    def _apply_trade_size_bounds(self, intent: TradeIntent) -> TradeIntent:
        """Clamp open order cash sizing to the configured $10–$20 policy band."""

        if intent.action != "OPEN" or self._min_cash_per_trade <= 0:
            return intent

        requested_cash = float(intent.cash_spent)
        metadata = dict(intent.metadata or {})
        original_cash = requested_cash
        adjusted = False

        if requested_cash < self._min_cash_per_trade:
            self._logger.info(
                "Applying $%.2f cash floor to %s order from %s for %s (requested $%.2f)",
                self._min_cash_per_trade,
                intent.side.upper(),
                intent.worker,
                intent.symbol,
                requested_cash,
            )
            metadata.update(
                {
                    "cash_floor_applied": True,
                    "original_cash_spent": original_cash,
                    "min_cash_per_trade": self._min_cash_per_trade,
                }
            )
            requested_cash = self._min_cash_per_trade
            adjusted = True

        if requested_cash > self._max_cash_per_trade:
            self._logger.info(
                "[RISK] Capping %s order size to $%.2f (requested $%.2f exceeds ceiling)",
                intent.symbol,
                self._max_cash_per_trade,
                requested_cash,
            )
            metadata.setdefault("original_cash_spent", original_cash)
            metadata.update(
                {
                    "cash_cap_applied": True,
                    "max_cash_per_trade": self._max_cash_per_trade,
                }
            )
            requested_cash = self._max_cash_per_trade
            adjusted = True

        if adjusted:
            intent.metadata = metadata
            intent.cash_spent = requested_cash
        return intent

    async def _open_trade(self, intent: TradeIntent, key: Tuple[str, str]) -> None:
        mode = "paper" if self._paper_trading else "live"
        cash_spent = float(intent.cash_spent)
        confidence = float(intent.confidence)
        self._logger.info(
            "[TRADE] %s submitting %s order for %s size=$%.2f confidence=%.3f mode=%s",
            intent.worker,
            intent.side.upper(),
            intent.symbol,
            cash_spent,
            confidence,
            mode,
        )
        try:
            reduce_only = False if intent.side == "sell" else None
            price, quantity = await self._broker.place_order(
                intent.symbol,
                intent.side,
                cash_spent,
                reduce_only=reduce_only,
            )
        except Exception as exc:  # noqa: BLE001 - broker errors should not crash engine
            self._logger.warning(
                "Order rejected for %s %s (cash %.2f): %s",
                intent.side.upper(),
                intent.symbol,
                cash_spent,
                exc,
            )
            self._logger.debug("Failed intent payload: %s", intent, exc_info=True)
            return
        price = float(price)
        quantity = float(quantity)
        cost = float(price * quantity)
        position = OpenPosition(
            worker=intent.worker,
            symbol=intent.symbol,
            side=intent.side,
            quantity=quantity,
            entry_price=price,
            cash_spent=cost,
        )
        self._open_positions[key] = position
        metadata = dict(intent.metadata or {})
        metadata.update({"fill_price": price, "fill_quantity": quantity, "mode": mode})
        recorded_intent = TradeIntent(
            worker=intent.worker,
            action="OPEN",
            symbol=intent.symbol,
            side=intent.side,
            cash_spent=cost,
            entry_price=price,
            confidence=confidence,
            reason=intent.reason or "entry",
            metadata=metadata,
        )
        self._trade_log.record_trade(recorded_intent)
        self._trade_log.record_trade_event(
            worker=intent.worker,
            symbol=intent.symbol,
            event="execution_open",
            details={
                "price": price,
                "quantity": quantity,
                "confidence": confidence,
                "mode": mode,
            },
        )
        self._logger.info(
            "[TRADE] %s opened %s %s @ %.2f qty=%.6f cash=%.2f",
            intent.worker,
            intent.side.upper(),
            intent.symbol,
            price,
            quantity,
            cost,
        )

    async def _close_trade(self, intent: TradeIntent, key: Tuple[str, str], position: OpenPosition) -> None:
        mode = "paper" if self._paper_trading else "live"
        self._logger.info(
            "[TRADE] %s submitting CLOSE order for %s qty=%.6f mode=%s",
            intent.worker,
            intent.symbol,
            position.quantity,
            mode,
        )
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
        price = float(price)
        quantity = float(quantity)
        pnl = (
            (price - position.entry_price) * position.quantity
            if position.side == "buy"
            else (position.entry_price - price) * position.quantity
        )
        pnl_percent = pnl / position.cash_spent * 100 if position.cash_spent else 0.0
        reason = intent.reason or "exit"
        metadata = dict(intent.metadata or {})
        metadata.update({
            "fill_price": price,
            "fill_quantity": quantity,
            "mode": mode,
            "pnl_usd": pnl,
            "pnl_percent": pnl_percent,
            "reason": reason,
        })
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
            reason=reason,
            metadata=metadata,
        )
        self._trade_log.record_trade(recorded_intent)
        self._trade_log.record_trade_event(
            worker=intent.worker,
            symbol=intent.symbol,
            event="execution_close",
            details=metadata,
        )
        self._open_positions.pop(key, None)
        self._logger.info(
            "[TRADE] %s closed %s %s @ %.2f | PnL %.2f USD (%.2f%%) reason=%s",
            intent.worker,
            intent.side.upper(),
            intent.symbol,
            price,
            pnl,
            pnl_percent,
            reason,
        )
        if reason in {"stop", "target", "trail", "mean-hit", "prob-revert", "prob-floor"}:
            self._logger.info(
                "[TRADE] Risk exit triggered for %s on %s (%s)",
                intent.worker,
                intent.symbol,
                reason,
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
                    reason="duration",
                    metadata={"duration_minutes": self._risk_manager.max_duration_minutes},
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
