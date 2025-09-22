"""Execution engine that coordinates order lifecycle management."""

from __future__ import annotations

import csv
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

from desk.config import DESK_ROOT
from desk.data import candles_to_dataframe
from desk.services.logger import EventLogger

_FALLBACK_MINIMUM_ORDER_SIZES = {
    "BTC": 0.0001,
    "XBT": 0.0001,
    "ETH": 0.001,
    "SOL": 0.01,
    "MATIC": 1.0,
}

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from desk.services.dashboard_recorder import DashboardRecorder
    from desk.services.telemetry import TelemetryClient


@dataclass
class OpenTrade:
    """Normalized representation of an open position."""

    worker: str
    symbol: str
    side: str
    qty: float
    entry_price: float
    stop_loss: float
    take_profit: float
    max_hold_seconds: float
    opened_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    trade_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def unrealized_pnl(self, price: float) -> float:
        if self.side == "BUY":
            return (price - self.entry_price) * self.qty
        return (self.entry_price - price) * self.qty

    def to_record(self) -> Dict[str, object]:
        return {
            "trade_id": self.trade_id,
            "worker": self.worker,
            "symbol": self.symbol,
            "side": self.side,
            "qty": float(self.qty),
            "entry_price": float(self.entry_price),
            "stop_loss": float(self.stop_loss),
            "take_profit": float(self.take_profit),
            "max_hold_seconds": float(self.max_hold_seconds),
            "opened_at": float(self.opened_at),
            "metadata": json.dumps(self.metadata or {}),
        }

    @classmethod
    def from_record(cls, record: Dict[str, object]) -> "OpenTrade":
        metadata = record.get("metadata")
        if isinstance(metadata, str):
            try:
                metadata_dict = json.loads(metadata)
            except json.JSONDecodeError:
                metadata_dict = {}
        elif isinstance(metadata, dict):
            metadata_dict = metadata
        else:
            metadata_dict = {}

        return cls(
            trade_id=str(record["trade_id"]),
            worker=str(record["worker"]),
            symbol=str(record["symbol"]),
            side=str(record["side"]),
            qty=float(record["qty"]),
            entry_price=float(record["entry_price"]),
            stop_loss=float(record["stop_loss"]),
            take_profit=float(record["take_profit"]),
            max_hold_seconds=float(record["max_hold_seconds"]),
            opened_at=float(record["opened_at"]),
            metadata={str(k): v for k, v in metadata_dict.items()},
        )


class PositionStore:
    """Crash-safe persistence for open trades."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or DESK_ROOT / "logs" / "positions.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.lock = Lock()
        self._init()

    def _init(self) -> None:
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS positions (
                    trade_id TEXT PRIMARY KEY,
                    worker TEXT,
                    symbol TEXT,
                    side TEXT,
                    qty REAL,
                    entry_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    max_hold_seconds REAL,
                    opened_at REAL,
                    metadata TEXT
                )
                """
            )
            self.conn.commit()

    def persist(self, trade: OpenTrade) -> None:
        record = trade.to_record()
        with self.lock:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO positions
                (trade_id, worker, symbol, side, qty, entry_price, stop_loss,
                 take_profit, max_hold_seconds, opened_at, metadata)
                VALUES (:trade_id, :worker, :symbol, :side, :qty, :entry_price,
                        :stop_loss, :take_profit, :max_hold_seconds, :opened_at, :metadata)
                """,
                record,
            )
            self.conn.commit()

    def remove(self, trade_id: str) -> None:
        with self.lock:
            self.conn.execute(
                "DELETE FROM positions WHERE trade_id = ?",
                (trade_id,),
            )
            self.conn.commit()

    def load(self) -> List[OpenTrade]:
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM positions")
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
        trades = []
        for row in rows:
            record = dict(zip(columns, row))
            trades.append(OpenTrade.from_record(record))
        return trades

    def close(self) -> None:
        with self.lock:
            try:
                self.conn.close()
            except Exception:
                pass


class ExecutionEngine:
    """Handles live orders, monitors exits, and journals trades."""

    def __init__(
        self,
        broker,
        logger: EventLogger,
        risk_config: Dict[str, float],
        *,
        telemetry: Optional["TelemetryClient"] = None,
        dashboard_recorder: Optional["DashboardRecorder"] = None,
        position_store: Optional[PositionStore] = None,
    ) -> None:
        self.broker = broker
        self.logger = logger
        self.risk_config = risk_config
        self.telemetry = telemetry
        self.dashboard = dashboard_recorder
        self.position_store = position_store or PositionStore()
        self.open_positions: Dict[str, List[OpenTrade]] = {}
        self._load_persisted_positions()
        self.slippage_bps = float(self.risk_config.get("slippage_bps", 15.0))
        self.balance_buffer_pct = float(
            self.risk_config.get("balance_buffer_pct", 0.05)
        )
        self.duplicate_cooldown = float(
            self.risk_config.get("duplicate_cooldown_seconds", 90.0)
        )
        self._last_trade_times: Dict[Tuple[str, str, str], float] = {}
        self._journal_path = DESK_ROOT / "logs" / "trade_history.csv"
        self._ensure_journal()

    def _load_persisted_positions(self) -> None:
        for trade in self.position_store.load():
            self.open_positions.setdefault(trade.symbol, []).append(trade)

    def _ensure_journal(self) -> None:
        self._journal_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._journal_path.exists():
            with self._journal_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "timestamp",
                        "trade_id",
                        "worker",
                        "symbol",
                        "side",
                        "qty",
                        "entry_price",
                        "exit_price",
                        "stop_loss",
                        "take_profit",
                        "pnl",
                        "exit_reason",
                        "mode",
                    ],
                )
                writer.writeheader()

    @staticmethod
    def _normalize_asset(symbol: str) -> str:
        base = str(symbol or "").split("/")[0].split("-")[0].upper()
        if base.startswith("X") and len(base) > 3:
            base = base[1:]
        if base.startswith("Z") and len(base) > 3:
            base = base[1:]
        aliases = {"XBT": "BTC"}
        return aliases.get(base, base)

    def _minimum_order_size(self, symbol: str) -> float:
        broker_min = 0.0
        minimum_config = getattr(self.broker, "minimum_order_config", None)
        if callable(minimum_config):
            try:
                broker_min, _precision = minimum_config(symbol)
            except Exception:
                broker_min = 0.0
        fallback = _FALLBACK_MINIMUM_ORDER_SIZES.get(self._normalize_asset(symbol), 0.0)
        return max(float(broker_min or 0.0), float(fallback or 0.0))

    # ------------------------------------------------------------------
    # Trade lifecycle helpers
    # ------------------------------------------------------------------
    def _build_trade(
        self,
        worker_name: str,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        max_hold_minutes: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OpenTrade:
        side = side.upper()
        sl_pct = float(self.risk_config.get("stop_loss_pct", 0.02))
        rr = float(self.risk_config.get("rr_ratio", 2.0))
        hold_seconds = float(self.risk_config.get("max_hold_minutes", 15.0)) * 60

        if stop_loss is None or take_profit is None:
            if side == "BUY":
                stop_loss = price * (1 - sl_pct)
                take_profit = price * (1 + sl_pct * rr)
            else:
                stop_loss = price * (1 + sl_pct)
                take_profit = price * (1 - sl_pct * rr)

        if max_hold_minutes is not None:
            hold_seconds = float(max_hold_minutes) * 60

        return OpenTrade(
            worker=worker_name,
            symbol=symbol,
            side=side,
            qty=float(qty),
            entry_price=float(price),
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            max_hold_seconds=hold_seconds,
            metadata=dict(metadata or {}),
        )

    def open_position(
        self,
        worker,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        risk_amount: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        max_hold_minutes: Optional[float] = None,
        metadata: Optional[Dict[str, float]] = None,
        *,
        sizing_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[OpenTrade]:
        if qty <= 0:
            return None

        duplicate_key = (worker.name, symbol, side.upper())
        now = time.time()
        last_trade = self._last_trade_times.get(duplicate_key)
        if last_trade and (now - last_trade) < self.duplicate_cooldown:
            self.logger.write(
                {
                    "type": "trade_skipped",
                    "reason": "duplicate_guard",
                    "worker": worker.name,
                    "symbol": symbol,
                    "side": side,
                }
            )
            return None

        slippage = max(self.slippage_bps / 10_000.0, 0.0)
        reference_price = price * (1 + slippage if side.upper() == "BUY" else 1 - slippage)

        can_execute = True
        balance_check_error: Optional[str] = None
        balance_guard = getattr(self.broker, "can_execute_market_order", None)
        if callable(balance_guard):
            try:
                can_execute = bool(
                    balance_guard(
                        symbol,
                        side,
                        qty,
                        price=reference_price,
                        slippage=slippage + self.balance_buffer_pct,
                    )
                )
            except Exception as exc:  # pragma: no cover - defensive broker guard
                balance_check_error = str(exc)
                can_execute = False
        if not can_execute:
            self.logger.write(
                {
                    "type": "trade_skipped",
                    "reason": "insufficient_balance",
                    "worker": worker.name,
                    "symbol": symbol,
                    "side": side,
                    "detail": balance_check_error,
                }
            )
            return None

        minimum_qty = self._minimum_order_size(symbol)
        if minimum_qty > 0 and float(qty) < minimum_qty:
            self.logger.log_feed_event(
                "WARNING",
                symbol,
                "Order below Kraken minimum; skipping execution.",
                worker=worker.name,
                requested_qty=float(qty),
                minimum_qty=float(minimum_qty),
            )
            self.logger.write(
                {
                    "type": "trade_skipped",
                    "reason": "below_minimum",
                    "worker": worker.name,
                    "symbol": symbol,
                    "side": side,
                    "requested_qty": float(qty),
                    "minimum_qty": float(minimum_qty),
                }
            )
            return None

        notional_value = qty * reference_price
        sizing_payload = dict(sizing_info or {})
        sizing_payload.update(
            {
                "worker": worker.name,
                "symbol": symbol,
                "side": side.upper(),
                "qty": float(qty),
                "notional": float(notional_value),
                "risk_amount": float(risk_amount),
                "risk_pct": float(sizing_info.get("risk_pct", 0.0) if sizing_info else 0.0),
            }
        )
        if sizing_info and sizing_info.get("min_notional_applied"):
            sizing_payload["min_notional_adjustment"] = True
        if sizing_info and sizing_info.get("min_qty_applied"):
            sizing_payload["min_qty_adjustment"] = True
        self.logger.log_feed_event(
            "INFO",
            symbol,
            "Submitting market order.",
            worker=worker.name,
            qty=float(qty),
            usd_value=float(notional_value),
            equity_pct=float(sizing_payload.get("risk_pct", 0.0)),
            min_notional_adjustment=bool(sizing_payload.get("min_notional_adjustment", False)),
            min_qty_adjustment=bool(sizing_payload.get("min_qty_adjustment", False)),
        )

        trade = self._build_trade(
            worker.name,
            symbol,
            side,
            qty,
            reference_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_hold_minutes=max_hold_minutes,
            metadata=metadata,
        )
        placed_order = self.broker.market_order(
            symbol,
            side.lower(),
            qty,
            order_type="market",
            client_order_id=trade.trade_id,
            worker_name=worker.name,
        )
        if placed_order is None:
            return None
        if placed_order.get("status") != "ok":
            self.logger.write(
                {
                    "type": "trade_skipped",
                    "reason": "order_rejected",
                    "worker": worker.name,
                    "symbol": symbol,
                    "side": side,
                    "detail": placed_order.get("error"),
                }
            )
            return None

        fill_qty = qty
        fill_price = reference_price
        fee = 0.0
        if isinstance(placed_order, dict):
            fill_qty = float(placed_order.get("qty", qty) or qty)
            fill_price = float(placed_order.get("price", reference_price) or reference_price)
            remaining = float(placed_order.get("remaining_qty", 0.0) or 0.0)
            fee = float(placed_order.get("fee", 0.0) or 0.0)
            trade.metadata.setdefault("execution", {})
            trade.metadata["execution"].update(
                {
                    "requested_qty": float(qty),
                    "filled_qty": fill_qty,
                    "remaining_qty": remaining,
                    "fee": float(placed_order.get("fee", 0.0) or 0.0),
                    "slippage": float(placed_order.get("slippage", slippage) or slippage),
                }
            )
        if fill_qty <= 0:
            return None
        trade.qty = fill_qty
        trade.entry_price = fill_price

        self.open_positions.setdefault(symbol, []).append(trade)
        self.position_store.persist(trade)
        self.logger.log_trade(worker, symbol, side, fill_qty, fill_price, pnl=0.0)
        self.logger.write(
            {
                "type": "trade_opened",
                "worker": worker.name,
                "symbol": symbol,
                "side": side,
                "qty": fill_qty,
                "price": fill_price,
                "risk_amount": risk_amount,
                "notional": float(fill_qty * fill_price),
                "risk_pct": float(sizing_payload.get("risk_pct", 0.0)),
                "min_notional_adjustment": bool(
                    sizing_payload.get("min_notional_adjustment", False)
                ),
                "min_qty_adjustment": bool(sizing_payload.get("min_qty_adjustment", False)),
            }
        )
        if self.dashboard:
            features = (metadata or {}).get("features") if metadata else None
            ml_edge = (metadata or {}).get("ml_edge") if metadata else None
            probability: Optional[float]
            try:
                probability = None if ml_edge is None else float(ml_edge)
            except (TypeError, ValueError):
                probability = None
            self.dashboard.record_trade_open(trade, fee=fee, metadata=metadata)
            self.dashboard.record_ml_score(
                worker.name,
                symbol,
                probability=probability,
                features=features,
                trade_id=trade.trade_id,
            )
        if self.telemetry:
            self.telemetry.record_trade_open(
                {
                    "trade_id": trade.trade_id,
                    "worker": worker.name,
                    "symbol": symbol,
                    "side": side,
                    "qty": fill_qty,
                    "price": fill_price,
                    "order": placed_order,
                }
            )
        self.logger.write(
            {
                "type": "order_submitted",
                "worker": worker.name,
                "symbol": symbol,
                "side": side,
                "qty": fill_qty,
                "client_order_id": placed_order.get("client_order_id"),
                "txid": placed_order.get("txid"),
            }
        )
        self._last_trade_times[duplicate_key] = now
        return trade

    def positions_for_symbol(self, symbol: str) -> List[OpenTrade]:
        return list(self.open_positions.get(symbol, []))

    def _finalize_trade(self, trade: OpenTrade, exit_price: float, exit_reason: str) -> float:
        positions = self.open_positions.get(trade.symbol, [])
        if trade in positions:
            positions.remove(trade)
        self.position_store.remove(trade.trade_id)
        pnl = trade.unrealized_pnl(exit_price)
        self.logger.log_trade_end(trade.worker, trade.symbol, exit_price, exit_reason, pnl)
        self.logger.write(
            {
                "type": "trade_closed",
                "worker": trade.worker,
                "symbol": trade.symbol,
                "side": trade.side,
                "qty": trade.qty,
                "entry_price": trade.entry_price,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "pnl": pnl,
            }
        )
        if self.telemetry:
            self.telemetry.record_trade_close(
                {
                    "trade_id": trade.trade_id,
                    "worker": trade.worker,
                    "symbol": trade.symbol,
                    "side": trade.side,
                    "qty": trade.qty,
                    "entry_price": trade.entry_price,
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "pnl": pnl,
                }
            )
        try:
            with self._journal_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "timestamp",
                        "trade_id",
                        "worker",
                        "symbol",
                        "side",
                        "qty",
                        "entry_price",
                        "exit_price",
                        "stop_loss",
                        "take_profit",
                        "pnl",
                        "exit_reason",
                        "mode",
                    ],
                )
                writer.writerow(
                    {
                        "timestamp": time.time(),
                        "trade_id": trade.trade_id,
                        "worker": trade.worker,
                        "symbol": trade.symbol,
                        "side": trade.side,
                        "qty": trade.qty,
                        "entry_price": trade.entry_price,
                        "exit_price": exit_price,
                        "stop_loss": trade.stop_loss,
                        "take_profit": trade.take_profit,
                        "pnl": pnl,
                        "exit_reason": exit_reason,
                        "mode": getattr(self.broker, "mode", "live"),
                    }
                )
        except Exception:  # pragma: no cover - best effort logging
            pass
        if self.dashboard:
            self.dashboard.record_trade_close(
                trade,
                exit_price=exit_price,
                exit_reason=exit_reason,
                pnl=pnl,
            )
            label: Optional[int]
            if pnl > 0:
                label = 1
            elif pnl < 0:
                label = 0
            else:
                label = None
            if label is not None:
                self.dashboard.update_ml_label(trade.trade_id, label)
        return pnl

    # ------------------------------------------------------------------
    def reconcile(self, workers: Iterable, feed_handler, portfolio=None) -> None:
        """Re-evaluate persisted positions on startup to refresh metadata."""

        worker_map = {worker.name: worker for worker in workers}
        for symbol, trades in list(self.open_positions.items()):
            candles = feed_handler.fetch(symbol)
            for trade in trades:
                worker = worker_map.get(trade.worker)
                if worker is None:
                    continue
                worker.state.setdefault("trades", 0)
                if portfolio is not None:
                    portfolio.mark_routed(trade.worker)
                # Nothing else required; exit evaluation will handle stops on next loop.

    def evaluate_exits(
        self, symbol: str, candles: Iterable[Dict[str, float]]
    ) -> list[tuple[OpenTrade, float, str]]:
        df = candles_to_dataframe(candles)
        if df.empty:
            return []
        price = float(df["close"].iloc[-1])
        now = time.time()
        closed = []
        for trade in list(self.open_positions.get(symbol, [])):
            reason: Optional[str] = None
            if trade.side == "BUY":
                if price <= trade.stop_loss:
                    reason = "stop_loss"
                elif price >= trade.take_profit:
                    reason = "take_profit"
            else:
                if price >= trade.stop_loss:
                    reason = "stop_loss"
                elif price <= trade.take_profit:
                    reason = "take_profit"

            if reason is None and now - trade.opened_at >= trade.max_hold_seconds:
                reason = "time_stop"

            if reason is not None:
                pnl = self._finalize_trade(trade, price, reason)
                closed.append((trade, pnl, reason))
        return closed

    def close(self) -> None:
        """Release any persistent resources used by the execution engine."""

        self.position_store.close()

