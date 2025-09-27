"""Device registration endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.user import DeviceToken, User
from ..schemas.user import DeviceRegisterRequest
from ..security.auth import get_current_user

router = APIRouter(prefix="/devices", tags=["devices"])


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_device(
    payload: DeviceRegisterRequest,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    existing = await session.scalar(select(DeviceToken).where(DeviceToken.token == payload.token))
    if existing:
        if existing.user_id != user.id:
            existing.user_id = user.id
            existing.platform = payload.platform
            await session.commit()
        return {"status": "updated"}
    device = DeviceToken(user_id=user.id, token=payload.token, platform=payload.platform)
    session.add(device)
    await session.commit()
    return {"status": "registered"}
