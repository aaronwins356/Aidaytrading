from __future__ import annotations

import os
from pathlib import Path
from typing import AsyncIterator, Iterator, cast

import pytest
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.types import ASGIApp

# Configure environment for tests before importing the app
os.environ.setdefault("DB_URL", "sqlite+aiosqlite:///./test_backend.db")
os.environ.setdefault("JWT_SECRET", "test_secret")
os.environ.setdefault("JWT_ALGORITHM", "HS256")
os.environ.setdefault("ACCESS_TOKEN_EXPIRES_MIN", "15")
os.environ.setdefault("REFRESH_TOKEN_EXPIRES_DAYS", "7")
os.environ.setdefault("ENV", "local")
os.environ.setdefault("BREVO_API_KEY", "test-brevo-key")
os.environ.setdefault("BREVO_SMTP_SERVER", "smtp.test.local")
os.environ.setdefault("BREVO_PORT", "587")
os.environ.setdefault("BREVO_SENDER_EMAIL", "alerts@example.com")
os.environ.setdefault("BREVO_SENDER_NAME", "Aidaytrading Alerts")
os.environ.setdefault("ADMIN_NOTIFICATION_EMAIL", "admin@example.com")

from app.core.database import get_session_factory  # noqa: E402
from app.main import app  # noqa: E402


def _apply_migrations() -> None:
    root_path = Path(__file__).resolve().parents[2]
    alembic_cfg = Config(str(root_path / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(root_path / "app" / "migrations"))
    command.upgrade(alembic_cfg, "head")


@pytest.fixture(scope="session", autouse=True)
def apply_migrations() -> Iterator[None]:
    db_path = Path("test_backend.db")
    if db_path.exists():
        db_path.unlink()
    _apply_migrations()
    yield
    if db_path.exists():
        db_path.unlink()


@pytest.fixture()
async def client() -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=cast(ASGIApp, app))  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.fixture()
async def session() -> AsyncIterator[AsyncSession]:
    session_factory = get_session_factory()
    async with session_factory() as session:
        yield session


@pytest.fixture(autouse=True)
def stub_email_notifications(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _noop(*_: object, **__: object) -> None:
        return None

    monkeypatch.setattr(
        "app.api.v1.auth._email_service.notify_admin_of_signup",
        _noop,
        raising=False,
    )
