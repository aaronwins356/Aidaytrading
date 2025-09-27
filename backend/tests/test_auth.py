from __future__ import annotations

from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api import auth as auth_api
from app.models.user import User, UserRole, UserStatus
from app.schemas.auth import LoginRequest, LogoutRequest, RefreshRequest, SignupRequest


async def test_signup_pending_login_flow(
    app_client: AsyncClient, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    async with session_factory() as session:
        signup_tokens = await auth_api.signup(
            SignupRequest(username="alice", email="alice@example.com", password="StrongPass123"),
            session=session,
        )
        assert signup_tokens.access_token
        user = await session.scalar(select(User).where(User.username == "alice"))
        assert user is not None
        assert user.status == UserStatus.PENDING
        user.status = UserStatus.ACTIVE
        user.role = UserRole.ADMIN
        await session.commit()

        login_tokens = await auth_api.login(
            LoginRequest(username="alice", password="StrongPass123"), session=session
        )
        assert login_tokens.access_token
        refreshed = await auth_api.refresh(
            RefreshRequest(refresh_token=login_tokens.refresh_token), session=session
        )
        assert refreshed.access_token
        await auth_api.logout(LogoutRequest(refresh_token=login_tokens.refresh_token), session=session)
