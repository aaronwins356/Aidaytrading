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

if TYPE_CHECKING:  # pragma: no cover - type checking only
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
    """Handles paper/live orders, monitors exits, and journals trades."""

    def __init__(
        self,
        broker,
        logger: EventLogger,
        risk_config: Dict[str, float],
        *,
        telemetry: Optional["TelemetryClient"] = None,
        position_store: Optional[PositionStore] = None,
    ) -> None:
        self.broker = broker
        self.logger = logger
        self.risk_config = risk_config
        self.telemetry = telemetry
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
        metadata: Optional[Dict[str, float]] = None,
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
        placed_order = self.broker.market_order(symbol, side.lower(), qty)
        if placed_order is None:
            return None

        fill_qty = qty
        fill_price = reference_price
        if isinstance(placed_order, dict):
            fill_qty = float(placed_order.get("qty", qty) or qty)
            fill_price = float(placed_order.get("price", reference_price) or reference_price)
            remaining = float(placed_order.get("remaining_qty", 0.0) or 0.0)
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
        self.logger.log_trade(worker, symbol, side, qty, fill_price, pnl=0.0)
        self.logger.write(
            {
                "type": "trade_opened",
                "worker": worker.name,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price": fill_price,
                "risk_amount": risk_amount,
            }
        )
        if self.telemetry:
            self.telemetry.record_trade_open(
                {
                    "trade_id": trade.trade_id,
                    "worker": worker.name,
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "price": fill_price,
                    "order": placed_order,
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

