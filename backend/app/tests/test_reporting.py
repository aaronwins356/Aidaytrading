from __future__ import annotations

import datetime as dt
from decimal import Decimal

import datetime as dt
from decimal import Decimal

import pytest
from httpx import AsyncClient
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.device_token import DevicePlatform, DeviceToken
from app.models.trade import Trade, TradeSide
from app.models.user import UserStatus
from app.tests.utils import create_equity_snapshot, create_trade, create_user

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]


async def _auth_headers(client: AsyncClient, username: str, password: str) -> dict[str, str]:
    response = await client.post(
        "/api/v1/login",
        json={"username": username, "password": password},
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_equity_curve_filters(client: AsyncClient, session: AsyncSession) -> None:
    await create_user(
        session,
        username="metricsuser",
        email="metrics@example.com",
        password="StrongPass1",
        status=UserStatus.ACTIVE,
    )
    now = dt.datetime.now(dt.timezone.utc)
    await create_equity_snapshot(session, timestamp=now - dt.timedelta(days=2), equity=Decimal("10000"))
    await create_equity_snapshot(session, timestamp=now - dt.timedelta(days=1), equity=Decimal("10100"))
    await create_equity_snapshot(session, timestamp=now, equity=Decimal("10200"))

    headers = await _auth_headers(client, "metricsuser", "StrongPass1")

    response = await client.get("/api/v1/equity-curve", headers=headers)
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 3
    timestamps = [dt.datetime.fromisoformat(point[0]) for point in payload]
    assert timestamps == sorted(timestamps)

    response = await client.get(
        "/api/v1/equity-curve",
        params={"limit": 2},
        headers=headers,
    )
    assert len(response.json()) == 2

    start = (now - dt.timedelta(days=1, hours=12)).isoformat()
    response = await client.get(
        "/api/v1/equity-curve",
        params={"start": start},
        headers=headers,
    )
    assert len(response.json()) == 2


@pytest.mark.asyncio
async def test_profit_and_win_rate(client: AsyncClient, session: AsyncSession) -> None:
    await create_user(
        session,
        username="profituser",
        email="profit@example.com",
        password="StrongPass1",
        status=UserStatus.ACTIVE,
    )
    now = dt.datetime.now(dt.timezone.utc)
    await create_equity_snapshot(session, timestamp=now - dt.timedelta(days=1), equity=Decimal("10000"))
    await create_equity_snapshot(session, timestamp=now, equity=Decimal("10500"))

    await create_trade(
        session,
        symbol="AAPL",
        side=TradeSide.BUY,
        size=Decimal("10"),
        pnl=Decimal("200"),
        timestamp=now - dt.timedelta(hours=2),
    )
    await create_trade(
        session,
        symbol="AAPL",
        side=TradeSide.SELL,
        size=Decimal("5"),
        pnl=Decimal("-50"),
        timestamp=now - dt.timedelta(hours=1),
    )
    await create_trade(
        session,
        symbol="TSLA",
        side=TradeSide.SELL,
        size=Decimal("2"),
        pnl=Decimal("0"),
        timestamp=now,
    )

    headers = await _auth_headers(client, "profituser", "StrongPass1")
    response = await client.get("/api/v1/profit", headers=headers)
    assert response.status_code == 200
    payload = response.json()
    assert Decimal(payload["current_balance"]) == Decimal("10500")
    assert Decimal(payload["total_pl_amount"]) == Decimal("500")
    assert Decimal(payload["total_pl_percent"]).quantize(Decimal("0.01")) == Decimal("5.00")
    assert payload["win_rate"] == pytest.approx(1 / 3, rel=1e-3)


@pytest.mark.asyncio
async def test_calendar_rollup(client: AsyncClient, session: AsyncSession) -> None:
    await create_user(
        session,
        username="calendaruser",
        email="calendar@example.com",
        password="StrongPass1",
        status=UserStatus.ACTIVE,
    )
    tz = ZoneInfo("America/Chicago")
    day_one = dt.datetime(2024, 4, 1, 10, 0, tzinfo=tz).astimezone(dt.timezone.utc)
    day_one_late = dt.datetime(2024, 4, 1, 22, 30, tzinfo=tz).astimezone(dt.timezone.utc)
    day_two = dt.datetime(2024, 4, 2, 9, 0, tzinfo=tz).astimezone(dt.timezone.utc)

    await create_trade(session, symbol="MSFT", side=TradeSide.BUY, size=Decimal("1"), pnl=Decimal("100"), timestamp=day_one)
    await create_trade(session, symbol="MSFT", side=TradeSide.SELL, size=Decimal("1"), pnl=Decimal("-20"), timestamp=day_one_late)
    await create_trade(session, symbol="GOOG", side=TradeSide.BUY, size=Decimal("1"), pnl=Decimal("-50"), timestamp=day_two)

    headers = await _auth_headers(client, "calendaruser", "StrongPass1")
    response = await client.get(
        "/api/v1/calendar",
        params={"start": "2024-04-01", "end": "2024-04-02"},
        headers=headers,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["2024-04-01"]["color"] == "green"
    assert Decimal(payload["2024-04-01"]["pnl"]) == Decimal("80")
    assert payload["2024-04-02"]["color"] == "red"
    assert Decimal(payload["2024-04-02"]["pnl"]) == Decimal("-50")


@pytest.mark.asyncio
async def test_status_endpoint(client: AsyncClient, session: AsyncSession) -> None:
    await create_user(
        session,
        username="statususer",
        email="status@example.com",
        password="StrongPass1",
        status=UserStatus.ACTIVE,
    )
    headers = await _auth_headers(client, "statususer", "StrongPass1")
    response = await client.get("/api/v1/status", headers=headers)
    assert response.status_code == 200
    payload = response.json()
    assert payload == {"running": False, "uptime_seconds": 0}


@pytest.mark.asyncio
async def test_trades_pagination_and_filters(client: AsyncClient, session: AsyncSession) -> None:
    await session.execute(delete(Trade))
    await session.commit()

    await create_user(
        session,
        username="tradeuser",
        email="trader@example.com",
        password="StrongPass1",
        status=UserStatus.ACTIVE,
    )
    now = dt.datetime.now(dt.timezone.utc)
    await create_trade(session, symbol="BTC", side=TradeSide.BUY, size=Decimal("1"), pnl=Decimal("100"), timestamp=now - dt.timedelta(minutes=2))
    await create_trade(session, symbol="BTC", side=TradeSide.SELL, size=Decimal("0.5"), pnl=Decimal("-30"), timestamp=now - dt.timedelta(minutes=1))
    await create_trade(session, symbol="ETH", side=TradeSide.SHORT, size=Decimal("2"), pnl=Decimal("50"), timestamp=now)

    headers = await _auth_headers(client, "tradeuser", "StrongPass1")
    response = await client.get(
        "/api/v1/trades",
        params={"page": 1, "page_size": 2},
        headers=headers,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 3
    assert len(payload["items"]) == 2
    timestamps = [item["timestamp"] for item in payload["items"]]
    assert timestamps == sorted(timestamps, reverse=True)

    response = await client.get(
        "/api/v1/trades",
        params={"symbol": "BTC"},
        headers=headers,
    )
    assert response.status_code == 200
    assert response.json()["total"] == 2


@pytest.mark.asyncio
async def test_register_device_upsert(client: AsyncClient, session: AsyncSession) -> None:
    user = await create_user(
        session,
        username="deviceuser",
        email="device@example.com",
        password="StrongPass1",
        status=UserStatus.ACTIVE,
    )
    headers = await _auth_headers(client, "deviceuser", "StrongPass1")

    response = await client.post(
        "/api/v1/register-device",
        json={"token": "abc123token", "platform": "ios"},
        headers=headers,
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Device token registered."

    devices = (
        await session.execute(select(DeviceToken).where(DeviceToken.user_id == user.id))
    ).scalars().all()
    assert len(devices) == 1

    response = await client.post(
        "/api/v1/register-device",
        json={"token": "abc123token", "platform": "android"},
        headers=headers,
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Device token updated."

    await session.refresh(devices[0])
    assert devices[0].platform == DevicePlatform.ANDROID
