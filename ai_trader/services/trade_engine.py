"""Trading engine orchestrating workers, risk, and broker interactions."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from datetime import datetime, timedelta
from typing import DefaultDict, Dict, Iterable, List, Mapping, Tuple

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported for static type checking only
    from ai_trader.services.ml import MLService
    from ai_trader.notifier import Notifier
    from ai_trader.workers.base import BaseWorker

from ai_trader.broker.kraken_client import KrakenClient
from ai_trader.broker.websocket_manager import KrakenWebsocketManager
from ai_trader.services.equity import EquityEngine
from ai_trader.services.logging import get_logger
from ai_trader.services.risk import RiskManager
from ai_trader.services.runtime_state import RuntimeStateStore
from ai_trader.services.trade_log import TradeLog
from ai_trader.services.types import MarketSnapshot, OpenPosition, TradeIntent


@dataclass(slots=True)
class _WorkerSignalResult:
    """Container for asynchronous worker signal evaluation results."""

    worker: object
    signals: Mapping[str, object] | None
    error: BaseException | None


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
        min_cash_per_trade: float = 0.0,
        max_cash_per_trade: float = 0.0,
        trade_confidence_min: float = 0.5,
        trade_fee_percent: float = 0.0,
        ml_service: "MLService" | None = None,
        notifier: "Notifier" | None = None,
        runtime_state: RuntimeStateStore | None = None,
    ) -> None:
        self._broker = broker
        self._websocket_manager = websocket_manager
        self._workers = list(workers)
        self._researchers = list(researchers)
        self._worker_lookup: Dict[str, object] = {
            getattr(worker, "name", worker.__class__.__name__): worker for worker in self._workers
        }
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
        self._ml_service = ml_service
        self._notifier = notifier
        self._runtime_state = runtime_state
        self._min_cash_per_trade = max(0.0, float(min_cash_per_trade))
        max_cash_value = float(max_cash_per_trade)
        self._max_cash_per_trade = max_cash_value if max_cash_value > 0 else 0.0
        self._trade_confidence_min = max(0.0, min(1.0, float(trade_confidence_min)))
        self._effective_confidence_min: Dict[str, float] = {}
        self._trade_fee_rate = max(0.0, float(trade_fee_percent))
        self._base_currency = getattr(broker, "base_currency", "USD")
        self._latest_allocation_cap: float | None = None
        self._risk_revision: int = 0
        if self._runtime_state is not None:
            self._runtime_state.set_base_currency(self._base_currency)
            self._runtime_state.set_starting_equity(broker.starting_equity)
            self._runtime_state.update_risk_settings(self._risk_manager.config_dict())
        self._initialise_risk_settings_from_store()

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
                if math.isclose(
                    entry_price, position.entry_price, rel_tol=1e-5, abs_tol=1e-5
                ) and math.isclose(cash_spent, position.cash_spent, rel_tol=1e-4, abs_tol=1e-4):
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
        self._sync_runtime_positions()

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

    async def _collect_worker_signals(
        self,
        worker: "BaseWorker",
        snapshot: MarketSnapshot,
        timeout: float,
    ) -> _WorkerSignalResult:
        """Resolve a worker's signals with a timeout guard."""

        try:
            signals = await asyncio.wait_for(worker.evaluate_signal(snapshot), timeout=timeout)
        except asyncio.TimeoutError as exc:
            self._logger.warning(
                "Worker %s signal evaluation timed out after %.2fs", worker.name, timeout
            )
            return _WorkerSignalResult(worker=worker, signals=None, error=exc)
        except Exception as exc:  # noqa: BLE001 - defensive against strategy bugs
            self._logger.error("Worker %s evaluate failed: %s", worker.name, exc)
            return _WorkerSignalResult(worker=worker, signals=None, error=exc)
        return _WorkerSignalResult(worker=worker, signals=signals, error=None)

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            snapshot = self._websocket_manager.latest_snapshot()
            if not snapshot.prices:
                await asyncio.sleep(1.0)
                continue

            self._maybe_refresh_risk_settings()
            equity, balances = await self._broker.compute_equity(snapshot.prices)
            equity = float(equity)
            normalized_balances = {asset: float(amount) for asset, amount in balances.items()}
            self._equity_engine.update(equity, self._broker.starting_equity)
            equity_metrics = self._equity_engine.get_latest_metrics()
            if self._runtime_state is not None:
                self._runtime_state.update_account(
                    equity=equity,
                    balances=normalized_balances,
                    pnl_percent=float(equity_metrics.get("pnl_percent", 0.0)),
                    pnl_usd=float(equity_metrics.get("pnl_usd", 0.0)),
                    open_positions=self._open_positions.values(),
                    prices=snapshot.prices,
                    starting_equity=self._broker.starting_equity,
                )
            self._trade_log.record_account_snapshot(normalized_balances, equity)
            equity_per_trade = equity * (self._equity_allocation_percent / 100.0)
            self._latest_allocation_cap = equity_per_trade
            base_threshold = self._trade_confidence_min
            updated_thresholds: Dict[str, float] = {}
            for symbol in self._active_symbols():
                threshold, relaxed = self._risk_manager.effective_confidence_threshold(
                    base_threshold, symbol=symbol
                )
                updated_thresholds[symbol] = threshold
                if relaxed and base_threshold > 0:
                    self._logger.debug(
                        "[RISK] Confidence threshold relaxed for %s from %.3f to %.3f",
                        symbol,
                        base_threshold,
                        threshold,
                    )
            self._effective_confidence_min = updated_thresholds
            await self._run_researchers(snapshot, equity_metrics)
            self._update_control_flags()

            eligible_workers: list["BaseWorker"] = []
            for worker in self._workers:
                if not getattr(worker, "active", True):
                    continue
                status_flag = self._control_flags.get(f"bot::{worker.name}")
                if status_flag in {"paused", "disabled"}:
                    self._logger.debug("Worker %s paused via control flag", worker.name)
                    continue
                eligible_workers.append(worker)

            signal_results: list[_WorkerSignalResult] = []
            if eligible_workers:
                timeout = max(self._refresh_interval, 0.1)
                tasks = [
                    asyncio.create_task(
                        self._collect_worker_signals(worker, snapshot, timeout),
                        name=f"signals::{worker.name}",
                    )
                    for worker in eligible_workers
                ]
                signal_results = await asyncio.gather(*tasks)

            for result in signal_results:
                worker = result.worker  # original object preserved for sequential processing
                if result.error is not None or result.signals is None:
                    continue
                signals = dict(result.signals)
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
                    except Exception as exc:  # noqa: BLE001 - guard strategy failures
                        self._logger.error(
                            "Worker %s trade generation failed: %s", worker.name, exc
                        )
                        continue
                    if intent is None:
                        continue
                    if self._violates_long_only(worker, intent, open_position):
                        continue
                    threshold = self._effective_confidence_min.get(
                        intent.symbol, self._trade_confidence_min
                    )
                    if (
                        intent.action == "OPEN"
                        and threshold > 0.0
                        and self._is_ml_worker(worker)
                        and float(intent.confidence) < threshold
                    ):
                        self._logger.info(
                            "[ML] Skipped low-confidence trade (symbol=%s, confidence=%.3f < threshold=%.3f)",
                            intent.symbol,
                            float(intent.confidence),
                            threshold,
                        )
                        continue
                    if intent.action == "OPEN":
                        funding_checked = await self._ensure_sufficient_balance(intent)
                        if funding_checked is None:
                            continue
                        intent = funding_checked
                    if self._kill_switch and intent.action == "OPEN":
                        self._logger.warning(
                            "Kill switch active. Blocking %s intent for %s", worker.name, symbol
                        )
                        continue
                    assessment = self._risk_manager.evaluate_trade(
                        intent,
                        equity=equity,
                        equity_metrics=equity_metrics,
                        open_positions=len(self._open_positions),
                        max_open_positions=self._max_open_positions,
                        price=snapshot.prices.get(intent.symbol),
                        candles=snapshot.candles.get(intent.symbol),
                    )
                    intent = assessment.intent
                    if not assessment.allowed:
                        reason = assessment.reason or "risk_block"
                        self._logger.warning(
                            "[RISK] Blocked %s trade for %s (%s)",
                            worker.name,
                            symbol,
                            reason,
                        )
                        if reason in {"max_drawdown", "daily_loss_limit"}:
                            await self._notify_error(
                                f"Risk engine halted {worker.name} on {symbol}: {reason.replace('_', ' ')}"
                            )
                        continue
                    if assessment.adjustments:
                        self._logger.info(
                            "[RISK] Adjusted %s trade for %s -> %s",
                            worker.name,
                            symbol,
                            assessment.adjustments,
                        )
                    self._apply_global_risk_overrides(worker, intent)
                    intent = await self._apply_trade_sizing(intent)
                    if intent is None:
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

    def _is_ml_worker(self, worker: object | None = None, *, name: str | None = None) -> bool:
        if worker is not None:
            return getattr(worker, "_ml_service", None) is not None
        if name is not None:
            target = self._worker_lookup.get(name)
            if target is None:
                return False
            return getattr(target, "_ml_service", None) is not None
        return False

    def _violates_long_only(
        self,
        worker: object,
        intent: TradeIntent,
        existing_position: OpenPosition | None,
    ) -> bool:
        worker_name = getattr(worker, "name", worker.__class__.__name__)
        if intent.action == "OPEN" and intent.side == "sell":
            confidence = float(intent.confidence)
            if self._is_ml_worker(worker):
                self._logger.info(
                    "[RISK] Short signal blocked (symbol=%s, confidence=%.3f) [worker=%s]",
                    intent.symbol,
                    confidence,
                    worker_name,
                )
            else:
                self._logger.info(
                    "Short trade blocked by long-only policy (%s -> %s) [worker=%s]",
                    worker_name,
                    intent.symbol,
                    worker_name,
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
        min_notional = self._min_cash_per_trade
        try:
            await self._broker.ensure_market(intent.symbol)
            broker_floor = self._resolve_min_notional(intent.symbol)
            if broker_floor > min_notional:
                min_notional = broker_floor
        except Exception:  # pragma: no cover - defensive guard
            self._logger.debug(
                "Failed to resolve min notional for %s", intent.symbol, exc_info=True
            )
        if available < min_notional:
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

    async def _apply_trade_sizing(self, intent: TradeIntent) -> TradeIntent | None:
        if intent.action != "OPEN":
            return intent

        requested_cash = float(intent.cash_spent)
        metadata = dict(intent.metadata or {})
        adjusted = False

        allocation_cap = self._latest_allocation_cap or 0.0
        if allocation_cap > 0.0 and requested_cash > allocation_cap:
            metadata.setdefault("original_cash_spent", requested_cash)
            metadata.update(
                {
                    "allocation_cap_applied": True,
                    "equity_allocation_cap": allocation_cap,
                }
            )
            requested_cash = allocation_cap
            adjusted = True

        if self._max_cash_per_trade > 0.0 and requested_cash > self._max_cash_per_trade:
            metadata.setdefault("original_cash_spent", float(intent.cash_spent))
            metadata.update(
                {
                    "cash_cap_applied": True,
                    "max_cash_per_trade": self._max_cash_per_trade,
                }
            )
            requested_cash = self._max_cash_per_trade
            adjusted = True

        try:
            await self._broker.ensure_market(intent.symbol)
        except Exception:  # pragma: no cover - non-critical hint
            self._logger.debug(
                "Unable to ensure market metadata for %s before sizing",
                intent.symbol,
                exc_info=True,
            )
        min_notional = self._resolve_min_notional(intent.symbol)

        if min_notional > 0.0 and requested_cash < min_notional:
            self._logger.info(
                "[RISK] Skipping %s order from %s – notional %.2f below minimum %.2f",
                intent.symbol,
                intent.worker,
                requested_cash,
                min_notional,
            )
            return None

        if adjusted:
            intent.metadata = metadata
            intent.cash_spent = requested_cash
        return intent

    def _apply_global_risk_overrides(self, worker: object, intent: TradeIntent) -> None:
        if intent.action != "OPEN":
            return
        applier = getattr(worker, "apply_global_risk_overrides", None)
        if not callable(applier):
            return
        metadata = intent.metadata or {}
        stop = self._coerce_float(metadata.get("stop_price"))
        target = self._coerce_float(metadata.get("target_price"))
        trailing = self._coerce_float(metadata.get("trailing_price"))
        try:
            applier(
                intent.symbol,
                intent.side,
                float(intent.entry_price),
                stop=stop,
                target=target,
                trailing=trailing,
            )
        except Exception:  # pragma: no cover - risk overrides must never crash the loop
            self._logger.debug(
                "Failed to apply global risk overrides for %s via %s",
                intent.symbol,
                getattr(worker, "name", worker.__class__.__name__),
                exc_info=True,
            )

    def _resolve_min_notional(self, symbol: str) -> float:
        broker_floor = 0.0
        getter = getattr(self._broker, "min_order_value", None)
        if callable(getter):
            try:
                broker_floor = float(getter(symbol) or 0.0)
            except Exception:  # pragma: no cover - defensive guard
                broker_floor = 0.0
        configured = self._min_cash_per_trade
        if broker_floor > 0.0 and configured > 0.0:
            return max(broker_floor, configured)
        return broker_floor or configured

    @staticmethod
    def _coerce_float(value: object | None) -> float | None:
        if value is None:
            return None
        try:
            coerced = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(coerced) or math.isinf(coerced):
            return None
        return coerced

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
            await self._notify_error(f"Order rejected for {intent.symbol} ({intent.side}) — {exc}")
            return
        price = float(price)
        quantity = float(quantity)
        cost = float(price * quantity)
        entry_fee = cost * self._trade_fee_rate
        total_cost = cost + entry_fee
        position = OpenPosition(
            worker=intent.worker,
            symbol=intent.symbol,
            side=intent.side,
            quantity=quantity,
            entry_price=price,
            cash_spent=total_cost,
            fees_paid=entry_fee,
        )
        self._open_positions[key] = position
        self._sync_runtime_positions()
        metadata = dict(intent.metadata or {})
        metadata.update(
            {
                "fill_price": price,
                "fill_quantity": quantity,
                "mode": mode,
                "entry_fee": entry_fee,
                "fees_total": entry_fee,
            }
        )
        recorded_intent = TradeIntent(
            worker=intent.worker,
            action="OPEN",
            symbol=intent.symbol,
            side=intent.side,
            cash_spent=total_cost,
            entry_price=price,
            confidence=confidence,
            reason=intent.reason or "entry",
            metadata=metadata,
        )
        intent.cash_spent = total_cost
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
                "entry_fee": entry_fee,
            },
        )
        self._risk_manager.on_trade_executed(recorded_intent)
        if self._runtime_state is not None:
            self._runtime_state.record_trade(recorded_intent)
        self._record_ml_trade_open(recorded_intent, price, quantity)
        self._logger.info(
            "[TRADE] %s opened %s %s @ %.2f qty=%.6f cash=%.2f fees=%.4f",
            intent.worker,
            intent.side.upper(),
            intent.symbol,
            price,
            quantity,
            total_cost,
            entry_fee,
        )
        await self._notify_trade(recorded_intent)

    async def _close_trade(
        self, intent: TradeIntent, key: Tuple[str, str], position: OpenPosition
    ) -> None:
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
            await self._notify_error(
                f"Close order failed for {intent.symbol} ({position.side}) — {exc}"
            )
            return
        price = float(price)
        quantity = float(quantity)
        gross_pnl = (
            (price - position.entry_price) * position.quantity
            if position.side == "buy"
            else (position.entry_price - price) * position.quantity
        )
        exit_fee = price * quantity * self._trade_fee_rate
        total_fees = position.fees_paid + exit_fee
        pnl = gross_pnl - total_fees
        pnl_percent = pnl / position.cash_spent * 100 if position.cash_spent else 0.0
        reason = intent.reason or "exit"
        metadata = dict(intent.metadata or {})
        metadata.update(
            {
                "fill_price": price,
                "fill_quantity": quantity,
                "mode": mode,
                "pnl_usd": pnl,
                "pnl_percent": pnl_percent,
                "reason": reason,
                "exit_fee": exit_fee,
                "fees_total": total_fees,
            }
        )
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
        self._sync_runtime_positions()
        self._record_ml_trade_close(recorded_intent, position, price, pnl, pnl_percent, reason)
        self._logger.info(
            "[TRADE] %s closed %s %s @ %.2f | PnL %.2f USD (%.2f%%) fees=%.4f reason=%s",
            intent.worker,
            intent.side.upper(),
            intent.symbol,
            price,
            pnl,
            pnl_percent,
            total_fees,
            reason,
        )
        self._risk_manager.on_trade_executed(recorded_intent)
        if self._runtime_state is not None:
            self._runtime_state.record_trade(recorded_intent)
        if reason in {"stop", "target", "trail", "mean-hit", "prob-revert", "prob-floor"}:
            self._logger.info(
                "[TRADE] Risk exit triggered for %s on %s (%s)",
                intent.worker,
                intent.symbol,
                reason,
            )
        await self._notify_trade(recorded_intent)

    async def _notify_trade(self, intent: TradeIntent) -> None:
        if self._notifier is None:
            return
        try:
            await self._notifier.send_trade_alert(intent)
        except Exception:  # pragma: no cover - best-effort logging only
            self._logger.debug("Failed to push trade alert to notifier", exc_info=True)

    async def _notify_error(self, message: str) -> None:
        if self._notifier is None:
            return
        try:
            await self._notifier.send_error(message)
        except Exception:  # pragma: no cover - best-effort logging only
            self._logger.debug("Failed to push error alert to notifier", exc_info=True)

    def _record_ml_trade_open(
        self,
        intent: TradeIntent,
        fill_price: float,
        quantity: float,
    ) -> None:
        if self._ml_service is None or not self._is_ml_worker(name=intent.worker):
            return
        features = self._ml_service.latest_features(intent.symbol)
        if not features:
            self._logger.debug(
                "[ML] Skipping trade registration for %s on %s – no features available",
                intent.worker,
                intent.symbol,
            )
            return
        metadata = dict(intent.metadata or {})
        metadata.update(
            {
                "fill_price": fill_price,
                "fill_quantity": quantity,
                "mode": "paper" if self._paper_trading else "live",
            }
        )
        try:
            self._ml_service.register_trade_open(
                worker=intent.worker,
                symbol=intent.symbol,
                confidence=float(intent.confidence),
                entry_price=fill_price,
                quantity=quantity,
                features=features,
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001 - ML feedback must remain resilient
            self._logger.exception(
                "Failed to register ML trade open for %s on %s: %s",
                intent.worker,
                intent.symbol,
                exc,
            )

    def _record_ml_trade_close(
        self,
        intent: TradeIntent,
        position: OpenPosition,
        exit_price: float,
        pnl: float,
        pnl_percent: float,
        reason: str,
    ) -> None:
        if self._ml_service is None or not self._is_ml_worker(name=intent.worker):
            return
        metadata = dict(intent.metadata or {})
        metadata.update(
            {
                "entry_price": position.entry_price,
                "cash_spent": position.cash_spent,
                "quantity": position.quantity,
                "reason": reason,
            }
        )
        try:
            self._ml_service.register_trade_close(
                worker=intent.worker,
                symbol=intent.symbol,
                exit_price=exit_price,
                pnl=pnl,
                pnl_percent=pnl_percent,
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001 - logging only
            self._logger.exception(
                "Failed to register ML trade close for %s on %s: %s",
                intent.worker,
                intent.symbol,
                exc,
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

    def _sync_runtime_positions(self, prices: Mapping[str, float] | None = None) -> None:
        if self._runtime_state is None:
            return
        self._runtime_state.update_open_positions(self._open_positions.values(), prices)

    def _initialise_risk_settings_from_store(self) -> None:
        fetch_latest = getattr(self._trade_log, "fetch_latest_risk_settings", None)
        if not callable(fetch_latest):
            return
        try:
            latest = fetch_latest()
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Failed to load persisted risk settings: %s", exc)
            return
        if not latest:
            return
        revision, settings = latest
        current = self._risk_manager.config_dict()
        if all(current.get(key) == value for key, value in settings.items()):
            self._risk_revision = revision
            if self._runtime_state is not None:
                self._runtime_state.update_risk_settings(current, revision=revision)
            return
        try:
            updated = self._risk_manager.update_config(settings)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failed to apply persisted risk settings during startup (revision=%s): %s",
                revision,
                exc,
            )
            return
        self._risk_revision = revision
        if self._runtime_state is not None:
            self._runtime_state.update_risk_settings(updated, revision=revision)
        self._logger.info(
            "Risk configuration initialised from persisted settings (revision=%d)", revision
        )

    def _maybe_refresh_risk_settings(self) -> None:
        fetch_latest = getattr(self._trade_log, "fetch_latest_risk_settings", None)
        if not callable(fetch_latest):
            return
        try:
            latest = fetch_latest()
        except Exception as exc:  # noqa: BLE001
            self._logger.debug("Failed to poll persisted risk settings: %s", exc)
            return
        if not latest:
            return
        revision, settings = latest
        if revision <= self._risk_revision:
            return
        current = self._risk_manager.config_dict()
        if all(current.get(key) == value for key, value in settings.items()):
            self._risk_revision = revision
            if self._runtime_state is not None:
                self._runtime_state.update_risk_settings(current, revision=revision)
            return
        try:
            updated = self._risk_manager.update_config(settings)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failed to apply persisted risk settings revision %s: %s", revision, exc
            )
            return
        self._risk_revision = revision
        if self._runtime_state is not None:
            self._runtime_state.update_risk_settings(updated, revision=revision)
        self._logger.info(
            "Risk configuration refreshed from persisted settings (revision=%d)", revision
        )
