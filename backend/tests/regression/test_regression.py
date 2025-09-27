from __future__ import annotations

from decimal import Decimal

from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.trade import BalanceSnapshot, DailyPnL, EquityPoint, Trade
from app.models.user import User, UserRole, UserStatus
from app.security.auth import create_tokens, hash_password


async def prepare_data(
    client: AsyncClient, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    async with session_factory() as session:
        user = User(
            username="investor",
            email="investor@example.com",
            hashed_password=hash_password("StrongPass123"),
            status=UserStatus.ACTIVE,
            role=UserRole.VIEWER,
        )
        session.add(user)
        session.add_all(
            [
                Trade(symbol="AAPL", side="buy", quantity=Decimal("1"), price=Decimal("150"), pnl=Decimal("10")),
                Trade(symbol="AAPL", side="sell", quantity=Decimal("1"), price=Decimal("160"), pnl=Decimal("-5")),
            ]
        )
        session.add(EquityPoint(value=Decimal("1000")))
        session.add(BalanceSnapshot(balance=Decimal("1000"), equity=Decimal("1010")))
        session.add(DailyPnL(trading_day="2024-01-01", pnl=Decimal("10"), pnl_percent=Decimal("1"), win_rate=Decimal("50")))
        await session.commit()
        await session.refresh(user)
        await create_tokens(session, user)


async def test_investor_endpoints(
    app_client: AsyncClient, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    await prepare_data(app_client, session_factory)
    async with session_factory() as session:
        trades = (await session.execute(select(Trade))).scalars().all()
        assert len(trades) == 2
        equity_points = (await session.execute(select(EquityPoint))).scalars().all()
        assert equity_points[0].value == Decimal("1000")
        balance = (await session.execute(select(BalanceSnapshot))).scalars().first()
        assert balance and balance.equity == Decimal("1010")
        calendar = (await session.execute(select(DailyPnL))).scalars().first()
        assert calendar and calendar.trading_day == "2024-01-01"
