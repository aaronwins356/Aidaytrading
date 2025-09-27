"""Authentication endpoints."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Security, status
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import EmailStr
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.user import User, UserRole, UserStatus
from ..schemas.auth import (
    AuthResponse,
    AuthStatusResponse,
    ForgotPasswordRequest,
    LoginRequest,
    LogoutRequest,
    RefreshRequest,
    SignupRequest,
    TokenBundle,
)
from ..security.auth import (
    authenticate_user,
    create_tokens,
    hash_password,
    oauth2_scheme,
    revoke_refresh_token,
    to_user_profile,
    validate_refresh_token,
)
from ..security.jwt import create_access_token
from ..services.brevo import brevo_service
from ..utils.logger import get_logger

router = APIRouter(prefix="/auth", tags=["auth"])
logger = get_logger(__name__)


@router.post("/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def signup(payload: SignupRequest, session: AsyncSession = Depends(get_db)) -> AuthResponse:
    existing = await session.execute(
        select(User).where((User.username == payload.username) | (User.email == payload.email))
    )
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User already exists")
    user = User(
        username=payload.username,
        email=payload.email,
        hashed_password=hash_password(payload.password),
        status=UserStatus.PENDING,
        role=UserRole.VIEWER,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    await brevo_service.send_signup_approval_request(user.username, user.email)
    tokens = await create_tokens(session, user)
    return AuthResponse(tokens=tokens, user=to_user_profile(user))


@router.post("/login", response_model=AuthResponse)
async def login(payload: LoginRequest, session: AsyncSession = Depends(get_db)) -> AuthResponse:
    user = await authenticate_user(session, payload.username, payload.password)
    tokens = await create_tokens(session, user)
    return AuthResponse(tokens=tokens, user=to_user_profile(user))


@router.post("/refresh", response_model=AuthResponse)
async def refresh(payload: RefreshRequest, session: AsyncSession = Depends(get_db)) -> AuthResponse:
    stored = await validate_refresh_token(session, payload.refresh_token)
    user = await session.get(User, stored.user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    access_token, _, access_exp = create_access_token(
        user.id, {"role": user.role.value, "username": user.username}
    )
    if isinstance(access_exp, (int, float)):
        access_exp_dt = datetime.fromtimestamp(access_exp, tz=timezone.utc)
    else:
        access_exp_dt = access_exp
    stored_expires = stored.expires_at if stored.expires_at.tzinfo else stored.expires_at.replace(tzinfo=timezone.utc)
    expires_in = max(int((access_exp_dt - datetime.now(timezone.utc)).total_seconds()), 0)
    refresh_expires_in = max(int((stored_expires - datetime.now(timezone.utc)).total_seconds()), 0)
    tokens = TokenBundle(
        access_token=access_token,
        refresh_token=payload.refresh_token,
        token_type="bearer",
        expires_in=expires_in,
        refresh_expires_in=refresh_expires_in,
    )
    return AuthResponse(tokens=tokens, user=to_user_profile(user))


@router.post("/logout", status_code=status.HTTP_200_OK)
async def logout(
    payload: LogoutRequest | None = None,
    credentials: HTTPAuthorizationCredentials | None = Security(oauth2_scheme),
    session: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    refresh_token_body = payload.refresh_token if payload and payload.refresh_token else None
    refresh_token = refresh_token_body or (credentials.credentials if credentials else None)
    if not refresh_token:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Refresh token required")
    stored = await validate_refresh_token(session, refresh_token)
    await revoke_refresh_token(session, stored.jti)
    logger.info("Refresh token revoked", extra={"jti": stored.jti})
    return {"status": "logged-out"}


@router.get("/status", response_model=AuthStatusResponse)
async def auth_status(
    username: Annotated[str | None, Query(min_length=3)] = None,
    email: Annotated[EmailStr | None, Query()] = None,
    session: AsyncSession = Depends(get_db),
) -> AuthStatusResponse:
    if not username and not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Provide a username or email",
        )
    query = select(User)
    filters = []
    if username:
        filters.append(User.username == username)
    if email:
        filters.append(User.email == email)
    condition = filters[0]
    if len(filters) > 1:
        condition = or_(*filters)
    result = await session.execute(query.where(condition))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return AuthStatusResponse(status=user.status.value)


@router.post("/forgot-password", status_code=status.HTTP_202_ACCEPTED)
async def forgot_password(
    payload: ForgotPasswordRequest, session: AsyncSession = Depends(get_db)
) -> dict[str, str]:
    result = await session.execute(select(User).where(User.email == payload.email))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    await brevo_service.send_password_reset(payload.email, payload.reset_link)
    return {"status": "password-reset-requested"}
