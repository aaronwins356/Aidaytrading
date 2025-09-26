"""User Pydantic schemas."""
from __future__ import annotations

import datetime as dt

from pydantic import BaseModel

from app.models.user import UserRole, UserStatus


class UserRead(BaseModel):
    """Representation of a user returned to clients."""

    id: int
    username: str
    email: str
    role: UserRole
    status: UserStatus
    created_at: dt.datetime
    updated_at: dt.datetime

    model_config = {
        "from_attributes": True,
    }
