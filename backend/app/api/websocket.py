"""WebSocket endpoints for live streaming."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import AsyncSessionLocal
from ..models.trade import BalanceSnapshot, BotState, EquityPoint, Trade
from ..security.jwt import TokenError, decode_token
from ..utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


class WebSocketManager:
    """Manage active websocket connections and deduplication per channel."""

    def __init__(self) -> None:
        self.connections: Dict[str, Set[WebSocket]] = {}
        self.last_payload: Dict[str, Any] = {}

    async def connect(self, key: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self.connections.setdefault(key, set()).add(websocket)

    def disconnect(self, key: str, websocket: WebSocket) -> None:
        if key in self.connections:
            self.connections[key].discard(websocket)
            if not self.connections[key]:
                self.connections.pop(key)

    async def broadcast(self, key: str, message: Any) -> None:
        if self.last_payload.get(key) == message:
            return
        self.last_payload[key] = message
        for websocket in list(self.connections.get(key, set())):
            try:
                await websocket.send_json(message)
            except Exception:
                self.disconnect(key, websocket)

    def connected_clients(self) -> int:
        return sum(len(connections) for connections in self.connections.values())


ws_manager = WebSocketManager()


async def authenticate_websocket(websocket: WebSocket) -> None:
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4401)
        raise WebSocketDisconnect(code=4401)
    payload = decode_token(token)
    if payload.get("type") != "access":
        await websocket.close(code=4403)
        raise WebSocketDisconnect(code=4403)


async def stream_endpoint(
    websocket: WebSocket,
    key: str,
    fetcher: Callable[[AsyncSession], Awaitable[Any]],
    interval: float,
) -> None:
    await authenticate_websocket(websocket)
    await ws_manager.connect(key, websocket)
    try:
        async with AsyncSessionLocal() as session:
            while True:
                data = await fetcher(session)
                await ws_manager.broadcast(key, data)
                await asyncio.sleep(interval)
    except WebSocketDisconnect:
        logger.debug("WebSocket disconnected", extra={"key": key})
    finally:
        ws_manager.disconnect(key, websocket)


async def fetch_status(session: AsyncSession) -> dict[str, Any]:
    state = await session.scalar(select(BotState))
    running = bool(state.running) if state else False
    mode = state.mode if state else "paper"
    uptime = None
    if state and state.running and state.uptime_started_at:
        uptime = (datetime.now(timezone.utc) - state.uptime_started_at).total_seconds()
    return {"running": running, "mode": mode, "uptime": uptime}


async def fetch_equity(session: AsyncSession) -> list[dict[str, Any]]:
    result = await session.execute(select(EquityPoint).order_by(EquityPoint.timestamp.desc()).limit(100))
    points = list(reversed(result.scalars().all()))
    return [{"timestamp": point.timestamp.isoformat(), "value": float(point.value)} for point in points]


async def fetch_trades(session: AsyncSession) -> list[dict[str, Any]]:
    result = await session.execute(select(Trade).order_by(Trade.executed_at.desc()).limit(20))
    trades = result.scalars().all()
    return [
        {
            "symbol": trade.symbol,
            "side": trade.side,
            "price": float(trade.price),
            "quantity": float(trade.quantity),
            "pnl": float(trade.pnl),
            "executed_at": trade.executed_at.isoformat(),
        }
        for trade in trades
    ]


async def fetch_balance(session: AsyncSession) -> dict[str, Any]:
    snapshot = await session.scalar(select(BalanceSnapshot).order_by(BalanceSnapshot.timestamp.desc()))
    if not snapshot:
        return {"balance": 0.0, "equity": 0.0, "timestamp": None}
    return {
        "balance": float(snapshot.balance),
        "equity": float(snapshot.equity),
        "timestamp": snapshot.timestamp.isoformat(),
    }


@router.websocket("/ws/status")
async def ws_status(websocket: WebSocket) -> None:
    await stream_endpoint(websocket, "status", fetch_status, interval=5.0)


@router.websocket("/ws/equity")
async def ws_equity(websocket: WebSocket) -> None:
    await stream_endpoint(websocket, "equity", fetch_equity, interval=10.0)


@router.websocket("/ws/trades")
async def ws_trades(websocket: WebSocket) -> None:
    await stream_endpoint(websocket, "trades", fetch_trades, interval=2.0)


@router.websocket("/ws/balance")
async def ws_balance(websocket: WebSocket) -> None:
    await stream_endpoint(websocket, "balance", fetch_balance, interval=5.0)
