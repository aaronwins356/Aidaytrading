"""Notification management endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.user import DeviceToken, NotificationPreference, User
from ..schemas.user import (
    DeviceDeregisterRequest,
    NotificationPreferences,
    NotificationPreferencesUpdate,
)
from ..security.auth import get_current_user
from ..services.notifications import notification_service

router = APIRouter(prefix="/notifications", tags=["notifications"])


async def _get_or_create_preferences(
    session: AsyncSession, user_id: str
) -> NotificationPreference:
    preference = await session.get(NotificationPreference, user_id)
    if preference is None:
        preference = NotificationPreference(user_id=user_id)
        session.add(preference)
        await session.commit()
        await session.refresh(preference)
        notification_service.invalidate_cache(user_id)
    return preference


@router.get("/preferences", response_model=NotificationPreferences)
async def get_preferences(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> NotificationPreferences:
    preference = await _get_or_create_preferences(session, user.id)
    merged = preference.merged_preferences()
    return NotificationPreferences(**merged)


@router.put("/preferences", response_model=NotificationPreferences)
async def update_preferences(
    payload: NotificationPreferencesUpdate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> NotificationPreferences:
    preference = await _get_or_create_preferences(session, user.id)
    updates = payload.model_dump(exclude_unset=True)
    if updates:
        preference.preferences.update(updates)
        await session.commit()
        await session.refresh(preference)
        notification_service.invalidate_cache(user.id)
    merged = preference.merged_preferences()
    return NotificationPreferences(**merged)


@router.delete("/devices", status_code=status.HTTP_204_NO_CONTENT)
async def unregister_device(
    payload: DeviceDeregisterRequest,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> Response:
    device = await session.scalar(
        select(DeviceToken).where(
            DeviceToken.user_id == user.id, DeviceToken.token == payload.token
        )
    )
    if device:
        await session.delete(device)
        await session.commit()
        notification_service.invalidate_cache(user.id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
