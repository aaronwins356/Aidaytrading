from __future__ import annotations

from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api import users as users_api
from app.models.user import User, UserRole, UserStatus
from app.schemas.user import PasswordResetRequest, UserUpdate
from app.security.auth import hash_password


async def _create_admin(session: AsyncSession) -> User:
    admin = User(
        username="admin",
        email="admin@example.com",
        hashed_password=hash_password("StrongPass123"),
        status=UserStatus.ACTIVE,
        role=UserRole.ADMIN,
    )
    session.add(admin)
    await session.commit()
    await session.refresh(admin)
    return admin


async def test_user_admin_workflow(
    app_client: AsyncClient, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    async with session_factory() as session:
        admin = await _create_admin(session)
        bob = User(
            username="bob",
            email="bob@example.com",
            hashed_password=hash_password("StrongPass123"),
            status=UserStatus.PENDING,
            role=UserRole.VIEWER,
        )
        session.add(bob)
        await session.commit()
        await session.refresh(bob)

        users = await users_api.list_users(admin, session=session)
        assert len(users) == 2

        updated = await users_api.update_user(
            bob.id, UserUpdate(status=UserStatus.ACTIVE, role=UserRole.VIEWER), admin, session
        )
        assert updated.status == UserStatus.ACTIVE

        await users_api.trigger_password_reset(
            bob.id, PasswordResetRequest(reset_link="https://example.com/reset"), admin, session
        )
