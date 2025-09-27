from __future__ import annotations

import pytest
from freezegun import freeze_time
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.health import record_scheduler_tick
from app.core.metrics import record_push_event
from app.core.security import hash_password
from app.models.user import User, UserRole, UserStatus


@pytest.mark.asyncio
async def test_health_reports_scheduler_and_version(client: AsyncClient) -> None:
    with freeze_time("2024-05-01T12:00:00Z"):
        await record_scheduler_tick("equity_heartbeat")

    response = await client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert "uptime_seconds" in payload
    assert payload["db_status"]["state"] == "ok"
    scheduler_status = payload["scheduler_status"]
    scheduler = scheduler_status["equity_heartbeat"]
    assert scheduler["last_tick"].startswith("2024-05-01")
    assert scheduler["lag_seconds"] >= 0


@pytest.mark.asyncio
async def test_health_reports_database_failure(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _broken_connection() -> None:
        raise RuntimeError("db offline")

    monkeypatch.setattr("app.core.health.check_connection", _broken_connection)
    response = await client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["db_status"]["state"] == "down"
    assert "db offline" in payload["db_status"]["reason"]


async def _persist_user(
    session: AsyncSession,
    username: str,
    *,
    status: UserStatus,
    role: UserRole = UserRole.VIEWER,
) -> User:
    user = User(
        username=username,
        email=f"{username}@example.com",
        email_canonical=f"{username}@example.com",
        password_hash=hash_password("StrongPass1"),
        role=role,
        status=status,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


@pytest.mark.asyncio
async def test_metrics_expose_counters(client: AsyncClient, session: AsyncSession) -> None:
    await _persist_user(
        session,
        "metricsadmin",
        status=UserStatus.ACTIVE,
        role=UserRole.ADMIN,
    )

    success = await client.post(
        "/api/v1/login",
        json={"username": "metricsadmin", "password": "StrongPass1"},
    )
    assert success.status_code == 200

    failure = await client.post(
        "/api/v1/login",
        json={"username": "metricsadmin", "password": "WrongPass1"},
    )
    assert failure.status_code == 401

    record_push_event("email", "success")
    await client.get("/health")

    response = await client.get("/metrics")
    assert response.status_code == 200
    body = response.text
    lines = body.splitlines()
    metrics_lookup = {
        line.split(" ")[0]: line
        for line in lines
        if line and not line.startswith("#")
    }
    http_key = 'http_requests_total{path="/health",method="GET",status="200"}'
    http_line = metrics_lookup.get(http_key)
    if http_line is None:
        http_line = next(
            (
                line
                for line in lines
                if line.startswith("http_requests_total")
                and 'path="/health"' in line
                and 'method="GET"' in line
                and 'status="200"' in line
            ),
            None,
        )
    assert http_line is not None
    push_line = next(
        (
            line
            for line in lines
            if line.startswith("push_events_total")
            and 'type="email"' in line
            and 'outcome="success"' in line
        ),
        None,
    )
    assert push_line is not None
    assert push_line.strip().endswith(" 1.0")
    success_line = next(
        (
            line
            for line in lines
            if line.startswith("auth_logins_total")
            and 'outcome="success"' in line
        ),
        None,
    )
    failure_line = next(
        (
            line
            for line in lines
            if line.startswith("auth_logins_total")
            and 'outcome="failure"' in line
        ),
        None,
    )
    assert success_line is not None and success_line.strip().endswith(" 1.0")
    assert failure_line is not None and failure_line.strip().endswith(" 1.0")


@pytest.mark.asyncio
async def test_status_endpoint_reports_user_mix(client: AsyncClient, session: AsyncSession) -> None:
    await _persist_user(session, "pending-user", status=UserStatus.PENDING)
    await _persist_user(
        session,
        "active-user",
        status=UserStatus.ACTIVE,
        role=UserRole.VIEWER,
    )
    await _persist_user(session, "disabled-user", status=UserStatus.DISABLED)

    response = await client.get("/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["users"]["pending"] >= 1
    assert payload["users"]["active"] >= 1
    assert payload["users"]["disabled"] >= 1
