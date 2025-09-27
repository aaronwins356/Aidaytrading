from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.websockets import WebSocketDisconnect

from app.main import app
from app.models.user import UserStatus
from app.services.events import EQUITY_CHANNEL, STATUS_CHANNEL
from app.services.pubsub import event_bus
from app.tests.utils import create_equity_snapshot, create_user


@pytest.mark.asyncio
async def test_websocket_status_stream(client: AsyncClient, session: AsyncSession) -> None:
    await create_user(
        session,
        username="wsuser",
        email="ws@example.com",
        password="StrongPass1",
        status=UserStatus.ACTIVE,
    )
    token_response = await client.post(
        "/api/v1/login",
        json={"username": "wsuser", "password": "StrongPass1"},
    )
    token = token_response.json()["access_token"]

    with TestClient(app) as sync_client:
        with sync_client._transport.portal_factory() as portal:
            with sync_client.websocket_connect(f"/ws/status?token={token}") as websocket:
                initial = websocket.receive_json()
                assert initial["event"] == STATUS_CHANNEL
                portal.call(
                    event_bus.publish,
                    STATUS_CHANNEL,
                    {"event": STATUS_CHANNEL, "data": {"running": True}},
                )
                message = websocket.receive_json()
                assert message["data"]["running"] is True


@pytest.mark.asyncio
async def test_websocket_equity_requires_auth(client: AsyncClient) -> None:
    with TestClient(app) as sync_client:
        with pytest.raises(WebSocketDisconnect):
            with sync_client.websocket_connect("/ws/equity"):
                pass


@pytest.mark.asyncio
async def test_websocket_equity_broadcast(client: AsyncClient, session: AsyncSession) -> None:
    await create_user(
        session,
        username="equser",
        email="equser@example.com",
        password="StrongPass1",
        status=UserStatus.ACTIVE,
    )
    now = dt.datetime.now(dt.timezone.utc)
    await create_equity_snapshot(session, timestamp=now, equity=Decimal("10000"))
    token = (
        await client.post(
            "/api/v1/login",
            json={"username": "equser", "password": "StrongPass1"},
        )
    ).json()["access_token"]

    with TestClient(app) as sync_client:
        with sync_client._transport.portal_factory() as portal:
            with sync_client.websocket_connect(f"/ws/equity?token={token}") as websocket:
                initial = websocket.receive_json()
                assert initial["event"] == EQUITY_CHANNEL
                portal.call(
                    event_bus.publish,
                    EQUITY_CHANNEL,
                    {"event": EQUITY_CHANNEL, "data": {"timestamp": now.isoformat(), "equity": "10010"}},
                )
                message = websocket.receive_json()
                assert message["data"]["equity"] == "10010"
