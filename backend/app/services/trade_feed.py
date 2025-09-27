"""Helpers for publishing trade updates to subscribers."""
from __future__ import annotations

from app.models.trade import Trade
from app.services.events import TRADES_CHANNEL
from app.services.pubsub import event_bus


async def broadcast_trade(trade: Trade) -> None:
    payload = {
        "event": TRADES_CHANNEL,
        "data": {
            "id": trade.id,
            "symbol": trade.symbol,
            "side": trade.side.value,
            "size": str(trade.size),
            "pnl": str(trade.pnl),
            "timestamp": trade.timestamp.isoformat(),
        },
    }
    await event_bus.publish(TRADES_CHANNEL, payload)


__all__ = ["broadcast_trade"]

