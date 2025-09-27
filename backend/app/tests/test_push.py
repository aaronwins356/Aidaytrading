from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.device_token import DeviceToken
from app.models.user import UserStatus
from app.services.push import send_push_to_user
from app.tests.utils import create_user, register_device


class DummyFirebaseResponse:
    def __init__(self, *, success: int, failure: int, codes: list[str] | None = None) -> None:
        self.success_count = success
        self.failure_count = failure
        self.responses = []
        codes = codes or []
        for code in codes:
            exception = type("Exc", (), {"code": code})()
            self.responses.append(type("Resp", (), {"success": False, "exception": exception})())
        for _ in range(success):
            self.responses.append(type("Resp", (), {"success": True, "exception": None})())


@pytest.mark.asyncio
async def test_push_success(monkeypatch: pytest.MonkeyPatch, session: AsyncSession) -> None:
    user = await create_user(
        session,
        username="pushuser",
        email="push@example.com",
        password="StrongPass1",
        status=UserStatus.ACTIVE,
    )
    await register_device(session, user_id=user.id, token="token-a")

    async def fake_ensure() -> object:
        return object()

    monkeypatch.setattr("app.services.push._ensure_firebase_app", fake_ensure)

    def fake_send(message: Any, app: object | None = None) -> DummyFirebaseResponse:
        assert message.tokens == ["token-a"]
        return DummyFirebaseResponse(success=1, failure=0)

    monkeypatch.setattr("app.services.push.messaging.send_multicast", fake_send)

    result = await send_push_to_user(user.id, "Hello", "World", session=session)
    assert result.enabled is True
    assert result.success == 1
    assert result.failed == 0


@pytest.mark.asyncio
async def test_push_removes_invalid_tokens(monkeypatch: pytest.MonkeyPatch, session: AsyncSession) -> None:
    user = await create_user(
        session,
        username="pushbad",
        email="pushbad@example.com",
        password="StrongPass1",
        status=UserStatus.ACTIVE,
    )
    await register_device(session, user_id=user.id, token="token-b")

    async def fake_ensure() -> object:
        return object()

    monkeypatch.setattr("app.services.push._ensure_firebase_app", fake_ensure)

    def fake_send(message: Any, app: object | None = None) -> DummyFirebaseResponse:
        return DummyFirebaseResponse(success=0, failure=1, codes=["registration-token-not-registered"])

    monkeypatch.setattr("app.services.push.messaging.send_multicast", fake_send)

    result = await send_push_to_user(user.id, "Hello", "World", session=session)
    assert result.failed == 1

    remaining = (
        await session.execute(select(DeviceToken).where(DeviceToken.user_id == user.id))
    ).scalars().all()
    assert remaining == []
