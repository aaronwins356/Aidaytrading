"""WebSocket endpoints providing realtime updates."""
from __future__ import annotations

import datetime as dt

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from loguru import logger

from app.core.database import get_session_factory
from app.core.dependencies import resolve_user_from_token
from app.services.events import BALANCE_CHANNEL, EQUITY_CHANNEL, STATUS_CHANNEL, TRADES_CHANNEL
from app.services.metrics_source import get_metrics_source
from app.services.pubsub import event_bus
from app.services.reporting import get_latest_equity, get_system_status


class WebSocketAuthError(Exception):
    """Raised when WebSocket authentication fails."""


async def _extract_token(websocket: WebSocket) -> str:
    token = websocket.query_params.get("token")
    if token:
        return token
    auth_header = websocket.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header.split(" ", 1)[1]
    raise WebSocketAuthError("Missing access token")


async def _authenticate(websocket: WebSocket) -> None:
    token = await _extract_token(websocket)
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            await resolve_user_from_token(
                session,
                token,
                require_active=True,
                require_admin=False,
            )
        except HTTPException as exc:
            raise WebSocketAuthError(str(exc.detail)) from exc


async def _send_initial_status(websocket: WebSocket) -> None:
    session_factory = get_session_factory()
    async with session_factory() as session:
        status_row = await get_system_status(session)
    payload = {
        "event": STATUS_CHANNEL,
        "data": {
            "running": status_row.running,
            "started_at": status_row.started_at.isoformat() if status_row.started_at else None,
            "stopped_at": status_row.stopped_at.isoformat() if status_row.stopped_at else None,
        },
    }
    await websocket.send_json(payload)


async def _send_initial_equity(websocket: WebSocket) -> None:
    session_factory = get_session_factory()
    async with session_factory() as session:
        snapshot = await get_latest_equity(session)
    if snapshot is None:
        return
    payload = {
        "event": EQUITY_CHANNEL,
        "data": {
            "timestamp": snapshot.timestamp.isoformat(),
            "equity": format(snapshot.equity, "f"),
        },
    }
    await websocket.send_json(payload)


async def _send_initial_balance(websocket: WebSocket) -> None:
    metrics_source = get_metrics_source()
    balance = await metrics_source.fetch_current_balance()
    payload = {
        "event": BALANCE_CHANNEL,
        "data": {
            "balance": format(balance, "f"),
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    }
    await websocket.send_json(payload)


async def _stream_channel(websocket: WebSocket, channel: str) -> None:
    async with event_bus.subscribe(channel) as queue:
        while True:
            try:
                message = await queue.get()
                await websocket.send_json(message)
            except WebSocketDisconnect:
                break
            except Exception as exc:  # pragma: no cover - safety net
                logger.exception("websocket_publish_error", channel=channel)
                break


def register_websocket_routes(app: FastAPI) -> None:
    """Attach WebSocket endpoints to the FastAPI application."""

    @app.websocket("/ws/status")
    async def status_socket(websocket: WebSocket) -> None:
        try:
            await _authenticate(websocket)
        except WebSocketAuthError as exc:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=str(exc))
            return

        await websocket.accept()
        await _send_initial_status(websocket)
        await _stream_channel(websocket, STATUS_CHANNEL)

    @app.websocket("/ws/equity")
    async def equity_socket(websocket: WebSocket) -> None:
        try:
            await _authenticate(websocket)
        except WebSocketAuthError as exc:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=str(exc))
            return

        await websocket.accept()
        await _send_initial_equity(websocket)
        await _stream_channel(websocket, EQUITY_CHANNEL)

    @app.websocket("/ws/trades")
    async def trades_socket(websocket: WebSocket) -> None:
        try:
            await _authenticate(websocket)
        except WebSocketAuthError as exc:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=str(exc))
            return

        await websocket.accept()
        await _stream_channel(websocket, TRADES_CHANNEL)

    @app.websocket("/ws/balance")
    async def balance_socket(websocket: WebSocket) -> None:
        try:
            await _authenticate(websocket)
        except WebSocketAuthError as exc:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=str(exc))
            return

        await websocket.accept()
        await _send_initial_balance(websocket)
        await _stream_channel(websocket, BALANCE_CHANNEL)


__all__ = ["register_websocket_routes"]

