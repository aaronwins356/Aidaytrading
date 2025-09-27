from __future__ import annotations

import datetime as dt

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import jwt
from app.core.rate_limiter import login_rate_limiter
from app.core.security import hash_password
from app.models.token_blacklist import TokenBlacklist
from app.models.user import User, UserRole, UserStatus


@pytest.mark.asyncio
async def test_signup_success(client: AsyncClient, session: AsyncSession) -> None:
    response = await client.post(
        "/api/v1/signup",
        json={"username": "newuser", "email": "newuser@example.com", "password": "StrongPass1"},
    )
    assert response.status_code == 201
    payload = response.json()
    assert payload == {"message": "Signup received. Await approval.", "status": "pending"}

    result = await session.execute(select(User).where(User.username == "newuser"))
    user = result.scalar_one()
    assert user.status == UserStatus.PENDING


@pytest.mark.asyncio
async def test_signup_duplicate_username(client: AsyncClient) -> None:
    await client.post(
        "/api/v1/signup",
        json={"username": "dupuser", "email": "dup1@example.com", "password": "StrongPass1"},
    )
    response = await client.post(
        "/api/v1/signup",
        json={"username": "dupuser", "email": "dup2@example.com", "password": "StrongPass1"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "user_exists"


@pytest.mark.asyncio
async def test_signup_duplicate_email(client: AsyncClient) -> None:
    await client.post(
        "/api/v1/signup",
        json={"username": "emailuser1", "email": "dupemail@example.com", "password": "StrongPass1"},
    )
    response = await client.post(
        "/api/v1/signup",
        json={"username": "emailuser2", "email": "DUPEMAIL@example.com", "password": "StrongPass1"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "user_exists"


@pytest.mark.asyncio
async def test_signup_trimmed_email_case_insensitive(client: AsyncClient) -> None:
    first = await client.post(
        "/api/v1/signup",
        json={"username": "caseuser1", "email": "User@example.com ", "password": "StrongPass1"},
    )
    assert first.status_code == 201

    duplicate = await client.post(
        "/api/v1/signup",
        json={"username": "caseuser2", "email": " user@EXAMPLE.com", "password": "StrongPass1"},
    )
    assert duplicate.status_code == 400
    assert duplicate.json()["error"]["code"] == "user_exists"


@pytest.mark.asyncio
async def test_signup_invalid_email(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/signup",
        json={"username": "bademail", "email": "invalid-email", "password": "StrongPass1"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_email"


@pytest.mark.asyncio
async def test_signup_weak_password(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/signup",
        json={"username": "weakpass", "email": "weak@example.com", "password": "weak"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "weak_password"


async def _create_user(
    session: AsyncSession,
    *,
    username: str,
    email: str,
    password: str,
    status: UserStatus,
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


@pytest.mark.asyncio
async def test_login_active_user(client: AsyncClient, session: AsyncSession) -> None:
    await _create_user(
        session,
        username="activeuser",
        email="active@example.com",
        password="StrongPass1",
        status=UserStatus.ACTIVE,
    )
    response = await client.post(
        "/api/v1/login",
        json={"username": "activeuser", "password": "StrongPass1"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["token_type"] == "bearer"
    assert payload["access_token"]
    assert payload["refresh_token"]


@pytest.mark.asyncio
async def test_login_pending_user_fails(client: AsyncClient, session: AsyncSession) -> None:
    await _create_user(
        session,
        username="pendinguser",
        email="pending@example.com",
        password="StrongPass1",
        status=UserStatus.PENDING,
    )
    response = await client.post(
        "/api/v1/login",
        json={"username": "pendinguser", "password": "StrongPass1"},
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "inactive_account"


@pytest.mark.asyncio
async def test_login_disabled_user_fails(client: AsyncClient, session: AsyncSession) -> None:
    await _create_user(
        session,
        username="disableduser",
        email="disabled@example.com",
        password="StrongPass1",
        status=UserStatus.DISABLED,
    )
    response = await client.post(
        "/api/v1/login",
        json={"username": "disableduser", "password": "StrongPass1"},
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "inactive_account"


@pytest.mark.asyncio
async def test_refresh_token(client: AsyncClient, session: AsyncSession) -> None:
    await _create_user(
        session,
        username="refreshuser",
        email="refresh@example.com",
        password="StrongPass1",
        status=UserStatus.ACTIVE,
    )
    login = await client.post(
        "/api/v1/login",
        json={"username": "refreshuser", "password": "StrongPass1"},
    )
    refresh_token = login.json()["refresh_token"]

    response = await client.post("/api/v1/refresh", json={"refresh_token": refresh_token})
    assert response.status_code == 200
    assert response.json()["access_token"]


@pytest.mark.asyncio
async def test_cleanup_expired_tokens(session: AsyncSession) -> None:
    expired = TokenBlacklist(
        jti="expired",
        expires_at=dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=5),
    )
    active = TokenBlacklist(
        jti="active",
        expires_at=dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=5),
    )
    session.add_all([expired, active])
    await session.commit()

    removed = await jwt.cleanup_expired_tokens(session)
    await session.commit()

    assert removed == 1
    remaining = (await session.execute(select(TokenBlacklist))).scalars().all()
    assert len(remaining) == 1
    assert remaining[0].jti == "active"


@pytest.mark.asyncio
async def test_logout_blacklists_tokens(client: AsyncClient, session: AsyncSession) -> None:
    await _create_user(
        session,
        username="logoutuser",
        email="logout@example.com",
        password="StrongPass1",
        status=UserStatus.ACTIVE,
    )
    login = await client.post(
        "/api/v1/login",
        json={"username": "logoutuser", "password": "StrongPass1"},
    )
    access_token = login.json()["access_token"]
    refresh_token = login.json()["refresh_token"]

    response = await client.post(
        "/api/v1/logout",
        headers={"Authorization": f"Bearer {access_token}"},
        json={"refresh_token": refresh_token},
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Logged out"

    protected = await client.get(
        "/api/v1/me",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert protected.status_code == 401
    assert protected.json()["error"]["code"] == "token_revoked"


@pytest.mark.asyncio
async def test_login_rate_limited_after_excessive_attempts(
    client: AsyncClient, session: AsyncSession
) -> None:
    original_limit = login_rate_limiter.limit
    login_rate_limiter.limit = 2
    try:
        await _create_user(
            session,
            username="ratelimituser",
            email="ratelimit@example.com",
            password="StrongPass1",
            status=UserStatus.ACTIVE,
        )
        for _ in range(2):
            response = await client.post(
                "/api/v1/login",
                json={"username": "ratelimituser", "password": "wrong"},
            )
            assert response.status_code == 401

        blocked = await client.post(
            "/api/v1/login",
            json={"username": "ratelimituser", "password": "StrongPass1"},
        )
        assert blocked.status_code == 429
        assert blocked.headers.get("Retry-After") is not None
        payload = blocked.json()
        assert payload["error"]["code"] == "rate_limited"
    finally:
        login_rate_limiter.limit = original_limit


@pytest.mark.asyncio
async def test_refresh_revoked_after_token_version_bump(
    client: AsyncClient, session: AsyncSession
) -> None:
    user = await _create_user(
        session,
        username="versioned",
        email="versioned@example.com",
        password="StrongPass1",
        status=UserStatus.ACTIVE,
    )
    login = await client.post(
        "/api/v1/login",
        json={"username": "versioned", "password": "StrongPass1"},
    )
    refresh_token = login.json()["refresh_token"]

    user.token_version += 1
    await session.commit()

    response = await client.post("/api/v1/refresh", json={"refresh_token": refresh_token})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "token_revoked"


@pytest.mark.asyncio
async def test_pending_user_access_denied(client: AsyncClient, session: AsyncSession) -> None:
    user = await _create_user(
        session,
        username="pendingtoken",
        email="pendingtoken@example.com",
        password="StrongPass1",
        status=UserStatus.PENDING,
    )
    token_details = jwt.create_access_token(
        str(user.id),
        role=user.role.value,
        status=user.status.value,
        token_version=user.token_version,
    )
    response = await client.get(
        "/api/v1/me",
        headers={"Authorization": f"Bearer {token_details['token']}"},
    )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "inactive_user"
