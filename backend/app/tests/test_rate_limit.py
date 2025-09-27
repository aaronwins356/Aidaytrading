from __future__ import annotations

import datetime as dt

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import UserStatus
from app.services.ratelimiter import login_rate_limiter
from app.tests.utils import create_user


@pytest.mark.asyncio
async def test_login_rate_limit_enforced(client: AsyncClient, session: AsyncSession) -> None:
    await create_user(
        session,
        username="ratelimit",
        email="ratelimit@example.com",
        password="CorrectPass1",
        status=UserStatus.ACTIVE,
    )

    for _ in range(5):
        response = await client.post(
            "/api/v1/login",
            json={"username": "ratelimit", "password": "WrongPass"},
        )
        assert response.status_code == 401

    response = await client.post(
        "/api/v1/login",
        json={"username": "ratelimit", "password": "WrongPass"},
    )
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "too_many_attempts"

    # Force TTL expiry for deterministic testing
    entry = login_rate_limiter._entries.get("127.0.0.1")  # type: ignore[attr-defined]
    assert entry is not None
    entry.expires_at = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)

    response = await client.post(
        "/api/v1/login",
        json={"username": "ratelimit", "password": "CorrectPass1"},
    )
    assert response.status_code == 200
