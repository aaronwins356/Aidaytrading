"""Execution engine that coordinates order lifecycle management."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from desk.data import candles_to_dataframe
from desk.services.logger import EventLogger


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
    metadata: Dict[str, float] = field(default_factory=dict)

    def unrealized_pnl(self, price: float) -> float:
        if self.side == "BUY":
            return (price - self.entry_price) * self.qty
        return (self.entry_price - price) * self.qty


class ExecutionEngine:
    """Handles paper/live orders, monitors exits, and journals trades."""

    def __init__(self, broker, logger: EventLogger, risk_config: Dict[str, float]):
        self.broker = broker
        self.logger = logger
        self.risk_config = risk_config
        self.open_positions: Dict[str, List[OpenTrade]] = {}

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
        metadata: Optional[Dict[str, float]] = None,
    ) -> OpenTrade:
        side = side.upper()
        sl_pct = float(self.risk_config.get("stop_loss_pct", 0.02))
        rr = float(self.risk_config.get("rr_ratio", 2.0))
        hold_seconds = float(self.risk_config.get("max_hold_minutes", 15.0)) * 60

        if side == "BUY":
            stop_loss = price * (1 - sl_pct)
            take_profit = price * (1 + sl_pct * rr)
        else:
            stop_loss = price * (1 + sl_pct)
            take_profit = price * (1 - sl_pct * rr)

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
        metadata: Optional[Dict[str, float]] = None,
    ) -> Optional[OpenTrade]:
        if qty <= 0:
            return None

        trade = self._build_trade(worker.name, symbol, side, qty, price, metadata=metadata)
        placed_order = self.broker.market_order(symbol, side.lower(), qty)
        if placed_order is None:
            return None

        self.open_positions.setdefault(symbol, []).append(trade)
        self.logger.log_trade(worker, symbol, side, qty, price, pnl=0.0)
        self.logger.write(
            {
                "type": "trade_opened",
                "worker": worker.name,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price": price,
                "risk_amount": risk_amount,
            }
        )
        return trade

    def positions_for_symbol(self, symbol: str) -> List[OpenTrade]:
        return list(self.open_positions.get(symbol, []))

    def _finalize_trade(self, trade: OpenTrade, exit_price: float, exit_reason: str) -> float:
        positions = self.open_positions.get(trade.symbol, [])
        if trade in positions:
            positions.remove(trade)
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
        return pnl

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

