from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import AsyncIterator, Iterator, cast

from loguru import logger

import pytest
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete
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
os.environ.setdefault("BASELINE_EQUITY", "10000")
os.environ.setdefault("CORS_ALLOWED_ORIGINS", '["http://testserver"]')
os.environ.setdefault("SCHEDULER_ENABLED", "false")
os.environ.setdefault("HEARTBEAT_CRON", "0 2,8,14,20 * * *")
os.environ.setdefault("FIREBASE_CREDENTIALS", "")
os.environ.setdefault("EQUITY_SNAPSHOT_INTERVAL_MINUTES", "10")
os.environ.setdefault("LOGURU_LEVEL", "ERROR")
os.environ.setdefault("DISABLE_REQUEST_LOGGING", "1")
logger.remove()


def _install_firebase_stub() -> None:
    stub = types.ModuleType("firebase_admin")
    app_container: dict[str, object] = {}

    class _App:
        pass

    def _get_app() -> _App:
        if "app" not in app_container:
            raise ValueError("no app")
        return cast(_App, app_container["app"])

    def _initialize_app(_: object) -> _App:
        app = _App()
        app_container["app"] = app
        return app

    stub.App = _App
    stub.get_app = _get_app
    stub.initialize_app = _initialize_app

    credentials_module = types.ModuleType("firebase_admin.credentials")

    def _certificate(_: object) -> object:
        return object()

    credentials_module.Certificate = _certificate  # type: ignore[attr-defined]

    messaging_module = types.ModuleType("firebase_admin.messaging")

    class _Notification:
        def __init__(self, title: str, body: str) -> None:
            self.title = title
            self.body = body

    class _MulticastMessage:
        def __init__(self, tokens: list[str], notification: _Notification, data: dict[str, str] | None = None) -> None:
            self.tokens = tokens
            self.notification = notification
            self.data = data or {}

    def _send_multicast(message: _MulticastMessage, app: _App | None = None):  # type: ignore[override]
        class _Response:
            def __init__(self, tokens: list[str]) -> None:
                self.success_count = len(tokens)
                self.failure_count = 0
                self.responses = [types.SimpleNamespace(success=True) for _ in tokens]

        return _Response(message.tokens)

    messaging_module.Notification = _Notification  # type: ignore[attr-defined]
    messaging_module.MulticastMessage = _MulticastMessage  # type: ignore[attr-defined]
    messaging_module.send_multicast = _send_multicast  # type: ignore[attr-defined]

    exceptions_module = types.ModuleType("firebase_admin.exceptions")

    class FirebaseError(Exception):
        pass

    exceptions_module.FirebaseError = FirebaseError  # type: ignore[attr-defined]

    stub.credentials = credentials_module
    stub.messaging = messaging_module
    stub.exceptions = exceptions_module

    sys.modules.setdefault("firebase_admin", stub)
    sys.modules.setdefault("firebase_admin.credentials", credentials_module)
    sys.modules.setdefault("firebase_admin.messaging", messaging_module)
    sys.modules.setdefault("firebase_admin.exceptions", exceptions_module)


_install_firebase_stub()


def _install_apscheduler_stub() -> None:
    scheduler_module = types.ModuleType("apscheduler.schedulers.asyncio")

    class AsyncIOScheduler:
        def __init__(self, timezone: str | None = None) -> None:
            self.timezone = timezone
            self.running = False

        def add_job(self, *args: object, **kwargs: object) -> None:
            return None

        def start(self) -> None:
            self.running = True

        def shutdown(self, wait: bool = False) -> None:
            self.running = False

    scheduler_module.AsyncIOScheduler = AsyncIOScheduler  # type: ignore[attr-defined]

    cron_module = types.ModuleType("apscheduler.triggers.cron")

    class CronTrigger:
        def __init__(self, **_: object) -> None:
            return

        @classmethod
        def from_crontab(cls, *_: object, **__: object) -> "CronTrigger":
            return cls()

    cron_module.CronTrigger = CronTrigger  # type: ignore[attr-defined]

    interval_module = types.ModuleType("apscheduler.triggers.interval")

    class IntervalTrigger:
        def __init__(self, **_: object) -> None:
            return

    interval_module.IntervalTrigger = IntervalTrigger  # type: ignore[attr-defined]

    sys.modules.setdefault("apscheduler", types.ModuleType("apscheduler"))
    sys.modules.setdefault("apscheduler.schedulers", types.ModuleType("apscheduler.schedulers"))
    sys.modules.setdefault("apscheduler.schedulers.asyncio", scheduler_module)
    sys.modules.setdefault("apscheduler.triggers", types.ModuleType("apscheduler.triggers"))
    sys.modules.setdefault("apscheduler.triggers.cron", cron_module)
    sys.modules.setdefault("apscheduler.triggers.interval", interval_module)


_install_apscheduler_stub()

from app.core.database import get_session_factory  # noqa: E402
from app.models.reporting import SystemStatus  # noqa: E402
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
async def reset_system_status_between_tests() -> AsyncIterator[None]:
    """Ensure each test begins with a clean trading system state."""

    session_factory = get_session_factory()
    async with session_factory() as session:
        await session.execute(delete(SystemStatus))
        await session.commit()
    yield


@pytest.fixture(autouse=True)
def stub_email_notifications(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _noop(*_: object, **__: object) -> None:
        return None

    monkeypatch.setattr(
        "app.api.v1.auth._email_service.notify_admin_of_signup",
        _noop,
        raising=False,
    )
    monkeypatch.setattr(
        "app.api.v1.admin._email_service.send_password_reset_email",
        _noop,
        raising=False,
    )
