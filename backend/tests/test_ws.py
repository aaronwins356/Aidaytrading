from __future__ import annotations

from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.websocket import fetch_status
from app.models.user import User, UserStatus, UserRole
from app.security.auth import hash_password


async def ws_token(
    session: AsyncSession,
) -> None:
    user = User(
        username="wsuser",
        email="ws@example.com",
        hashed_password=hash_password("StrongPass123"),
        status=UserStatus.ACTIVE,
        role=UserRole.VIEWER,
    )
    session.add(user)
    await session.commit()


async def test_websocket_status_stream(
    app_client: AsyncClient, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    async with session_factory() as session:
        await ws_token(session)
        status = await fetch_status(session)
        assert "running" in status
        assert "mode" in status
