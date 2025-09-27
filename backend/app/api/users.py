"""User management endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.user import User
from ..schemas.user import PasswordResetRequest, UserOut, UserUpdate
from ..security.auth import require_admin, to_user_out
from ..services.brevo import brevo_service

router = APIRouter(prefix="/users", tags=["users"])


@router.get("", response_model=list[UserOut])
async def list_users(
    _: User = Depends(require_admin), session: AsyncSession = Depends(get_db)
) -> list[UserOut]:
    result = await session.execute(select(User))
    users = result.scalars().all()
    return [to_user_out(user) for user in users]


@router.patch("/{user_id}", response_model=UserOut)
async def update_user(
    user_id: str,
    payload: UserUpdate,
    _: User = Depends(require_admin),
    session: AsyncSession = Depends(get_db),
) -> UserOut:
    user = await session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if payload.status is not None:
        user.status = payload.status
    if payload.role is not None:
        user.role = payload.role
    await session.commit()
    await session.refresh(user)
    return to_user_out(user)


@router.post("/{user_id}/reset-password", status_code=status.HTTP_202_ACCEPTED)
async def trigger_password_reset(
    user_id: str,
    payload: PasswordResetRequest,
    _: User = Depends(require_admin),
    session: AsyncSession = Depends(get_db),
) -> None:
    user = await session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    await brevo_service.send_password_reset(user.email, payload.reset_link)
