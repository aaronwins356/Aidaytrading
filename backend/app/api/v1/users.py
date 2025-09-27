"""User endpoints."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.core.dependencies import require_active_user
from app.models.user import User
from app.schemas.user import UserRead

router = APIRouter()


@router.get("/me", response_model=UserRead)
async def read_me(current_user: Annotated[User, Depends(require_active_user)]) -> UserRead:
    """Return the authenticated user's profile."""

    return UserRead.model_validate(current_user)
