from __future__ import annotations

from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api import bot as bot_api
from app.models.trade import BotState
from app.models.user import User, UserRole, UserStatus
from app.security.auth import hash_password


async def admin_user(session: AsyncSession) -> User:
    admin = User(
        username="botadmin",
        email="bot@example.com",
        hashed_password=hash_password("StrongPass123"),
        status=UserStatus.ACTIVE,
        role=UserRole.ADMIN,
    )
    session.add(admin)
    await session.commit()
    await session.refresh(admin)
    return admin


async def test_bot_start_stop(
    app_client: AsyncClient, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    async with session_factory() as session:
        admin = await admin_user(session)
        status = await bot_api.start_bot(admin, session=session)
        assert status.running is True

        mode = await bot_api.set_mode({"mode": "live"}, admin, session)
        assert mode.mode == "live"

        stopped = await bot_api.stop_bot(admin, session=session)
        assert stopped.running is False

        state = await session.scalar(select(BotState))
        assert state is not None and state.mode == "live"
