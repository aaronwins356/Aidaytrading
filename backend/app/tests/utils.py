from __future__ import annotations

import datetime as dt
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password
from app.models.device_token import DevicePlatform, DeviceToken
from app.models.reporting import EquitySnapshot
from app.models.trade import Trade, TradeSide
from app.models.user import User, UserRole, UserStatus


async def create_user(
    session: AsyncSession,
    *,
    username: str,
    email: str,
    password: str,
    status: UserStatus = UserStatus.ACTIVE,
) -> User:
    user = User(
        username=username,
        email=email,
        email_canonical=email.lower(),
        password_hash=hash_password(password),
        role=UserRole.VIEWER,
        status=status,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


async def create_equity_snapshot(
    session: AsyncSession,
    *,
    timestamp: dt.datetime,
    equity: Decimal,
    source: str = "bot",
) -> EquitySnapshot:
    snapshot = EquitySnapshot(timestamp=timestamp, equity=equity, source=source)
    session.add(snapshot)
    await session.commit()
    await session.refresh(snapshot)
    return snapshot


async def create_trade(
    session: AsyncSession,
    *,
    symbol: str,
    side: TradeSide,
    size: Decimal,
    pnl: Decimal,
    timestamp: dt.datetime,
) -> Trade:
    trade = Trade(symbol=symbol, side=side, size=size, pnl=pnl, timestamp=timestamp)
    session.add(trade)
    await session.commit()
    await session.refresh(trade)
    return trade


async def register_device(
    session: AsyncSession,
    *,
    user_id: int,
    token: str,
    platform: DevicePlatform = DevicePlatform.IOS,
) -> DeviceToken:
    existing = (
        await session.execute(select(DeviceToken).where(DeviceToken.token == token))
    ).scalar_one_or_none()
    if existing:
        existing.user_id = user_id
        existing.platform = platform
        await session.commit()
        await session.refresh(existing)
        return existing

    device = DeviceToken(user_id=user_id, token=token, platform=platform)
    session.add(device)
    await session.commit()
    await session.refresh(device)
    return device

