from __future__ import annotations

from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api import devices as devices_api
from app.api import notifications as notifications_api
from app.models.user import DeviceToken, NotificationPreference, User, UserRole, UserStatus
from app.schemas.user import (
    DeviceDeregisterRequest,
    DeviceRegisterRequest,
    NotificationPreferencesUpdate,
)
from app.security.auth import hash_password
from app.services.notifications import notification_service


async def test_notification_preferences_and_devices(
    app_client: AsyncClient, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    async with session_factory() as session:
        notification_service.invalidate_cache()
        user = User(
            username="notify_user",
            email="notify@example.com",
            hashed_password=hash_password("StrongPass123"),
            status=UserStatus.ACTIVE,
            role=UserRole.VIEWER,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)

        prefs = await notifications_api.get_preferences(user=user, session=session)
        assert prefs.heartbeat_push is True
        assert prefs.trade_alert_push is True

        await devices_api.register_device(
            DeviceRegisterRequest(
                token="device-token-1",
                platform="ios",
                timezone="America/Chicago",
            ),
            user=user,
            session=session,
        )

        db_device = await session.scalar(
            select(DeviceToken).where(DeviceToken.token == "device-token-1")
        )
        assert db_device is not None
        assert db_device.timezone == "America/Chicago"

        tokens = await notification_service._fetch_tokens(session, "heartbeat_push")
        assert "device-token-1" in tokens

        updated = await notifications_api.update_preferences(
            NotificationPreferencesUpdate(heartbeat_push=False),
            user=user,
            session=session,
        )
        assert updated.heartbeat_push is False

        stored_pref = await session.get(NotificationPreference, user.id)
        assert stored_pref is not None
        assert stored_pref.preferences["heartbeat_push"] is False

        tokens_after_update = await notification_service._fetch_tokens(session, "heartbeat_push")
        assert "device-token-1" not in tokens_after_update

        await notifications_api.unregister_device(
            DeviceDeregisterRequest(token="device-token-1"),
            user=user,
            session=session,
        )

        removed_device = await session.scalar(
            select(DeviceToken).where(DeviceToken.token == "device-token-1")
        )
        assert removed_device is None

        assert not notification_service._token_cache
        assert not notification_service._preference_cache
