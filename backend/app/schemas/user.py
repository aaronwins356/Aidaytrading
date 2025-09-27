"""User related schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr, Field

from ..models.user import UserRole, UserStatus


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class UserUpdate(BaseModel):
    status: UserStatus | None = None
    role: UserRole | None = None


class UserOut(BaseModel):
    id: str
    username: str
    email: EmailStr
    status: UserStatus
    role: UserRole
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class UserProfile(BaseModel):
    id: str
    username: str
    email: EmailStr
    role: UserRole
    approval_status: UserStatus

    class Config:
        from_attributes = True


class PasswordResetRequest(BaseModel):
    reset_link: str


class DeviceRegisterRequest(BaseModel):
    token: str = Field(max_length=512)
    platform: Literal["ios", "android"] = "ios"
