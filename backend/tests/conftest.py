from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import asyncio
from collections.abc import AsyncIterator

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import create_app
from app.services.brevo import brevo_service
from app.services.push import push_service
from app.services.notifications import notification_service


@pytest.fixture(scope="session")
def event_loop() -> AsyncIterator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def client_session(monkeypatch) -> AsyncIterator[tuple[AsyncClient, async_sessionmaker[AsyncSession]]]:
    test_engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    TestingSessionLocal = async_sessionmaker(test_engine, expire_on_commit=False)

    async def override_get_db() -> AsyncIterator[AsyncSession]:
        async with TestingSessionLocal() as session:
            yield session

    monkeypatch.setattr("app.database.engine", test_engine)
    monkeypatch.setattr("app.database.AsyncSessionLocal", TestingSessionLocal)
    monkeypatch.setattr(notification_service, "start", lambda: None)

    async def _send_signup(*args, **kwargs):
        return None

    async def _send_reset(*args, **kwargs):
        return None

    monkeypatch.setattr(brevo_service, "send_signup_approval_request", _send_signup)
    monkeypatch.setattr(brevo_service, "send_password_reset", _send_reset)

    async def _send_push(*args, **kwargs):
        return None

    monkeypatch.setattr(push_service, "send_push", _send_push)

    async def _shutdown() -> None:
        return None

    monkeypatch.setattr(notification_service, "shutdown", _shutdown)

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client, TestingSessionLocal

    await test_engine.dispose()


@pytest.fixture
async def app_client(client_session: tuple[AsyncClient, async_sessionmaker[AsyncSession]]) -> AsyncIterator[AsyncClient]:
    client, _ = client_session
    yield client


@pytest.fixture
async def session_factory(
    client_session: tuple[AsyncClient, async_sessionmaker[AsyncSession]]
) -> async_sessionmaker[AsyncSession]:
    _, factory = client_session
    return factory
