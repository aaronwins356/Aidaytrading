"""Schemas for admin APIs."""
from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, Field

from app.models.user import UserRole, UserStatus


class PendingUser(BaseModel):
    id: int
    username: str
    email: str
    created_at: dt.datetime

    model_config = {"from_attributes": True}


class StatusChangeResponse(BaseModel):
    message: str
    user_id: int
    status: UserStatus


class RoleUpdateRequest(BaseModel):
    role: UserRole = Field(description="Target role for the user")


class RoleChangeResponse(BaseModel):
    message: str
    user_id: int
    previous_role: UserRole
    new_role: UserRole


class AuditLogEntry(BaseModel):
    id: int
    admin_id: int
    action: str
    target_user_id: int
    metadata: dict[str, Any] | None = Field(alias="details")
    created_at: dt.datetime

    model_config = {"from_attributes": True, "populate_by_name": True}
