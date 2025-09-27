"""Authentication schemas."""
from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field

from .user import UserProfile


class SignupRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    email: str
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenBundle(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_expires_in: int


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str | None = None


class AuthResponse(BaseModel):
    tokens: TokenBundle
    user: UserProfile


class AuthStatusResponse(BaseModel):
    status: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr
    reset_link: str = Field(min_length=1)
