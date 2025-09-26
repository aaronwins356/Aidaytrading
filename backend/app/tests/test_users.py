from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password
from app.models.user import User, UserRole, UserStatus


async def _create_active_user(session: AsyncSession, username: str, email: str, password: str) -> User:
    user = User(
        username=username,
        email=email,
        password_hash=hash_password(password),
        role=UserRole.VIEWER,
        status=UserStatus.ACTIVE,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


@pytest.mark.asyncio
async def test_me_returns_user_profile(client: AsyncClient, session: AsyncSession) -> None:
    await _create_active_user(session, "profileuser", "profile@example.com", "StrongPass1")
    login = await client.post(
        "/api/v1/login",
        json={"username": "profileuser", "password": "StrongPass1"},
    )
    access_token = login.json()["access_token"]

    response = await client.get(
        "/api/v1/me",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["username"] == "profileuser"
    assert payload["email"] == "profile@example.com"
    assert payload["role"] == "viewer"
    assert payload["status"] == "active"
    assert "created_at" in payload and "updated_at" in payload
