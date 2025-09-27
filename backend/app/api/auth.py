"""Authentication endpoints."""
from __future__ import annotations

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.user import User, UserRole, UserStatus
from ..schemas.auth import (
    LoginRequest,
    LogoutRequest,
    RefreshRequest,
    SignupRequest,
    TokenResponse,
)
from ..security.auth import (
    authenticate_user,
    create_tokens,
    hash_password,
    revoke_refresh_token,
    validate_refresh_token,
)
from ..security.jwt import create_access_token
from ..services.brevo import brevo_service
from ..utils.logger import get_logger

router = APIRouter(prefix="/auth", tags=["auth"])
logger = get_logger(__name__)


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def signup(payload: SignupRequest, session: AsyncSession = Depends(get_db)) -> TokenResponse:
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
    return TokenResponse(**tokens)


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, session: AsyncSession = Depends(get_db)) -> TokenResponse:
    user = await authenticate_user(session, payload.username, payload.password)
    tokens = await create_tokens(session, user)
    return TokenResponse(**tokens)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(payload: RefreshRequest, session: AsyncSession = Depends(get_db)) -> TokenResponse:
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
    return TokenResponse(
        access_token=access_token,
        refresh_token=payload.refresh_token,
        token_type="bearer",
        expires_in=expires_in,
        refresh_expires_in=refresh_expires_in,
    )


@router.post("/logout", status_code=status.HTTP_200_OK)
async def logout(payload: LogoutRequest, session: AsyncSession = Depends(get_db)) -> dict[str, str]:
    stored = await validate_refresh_token(session, payload.refresh_token)
    await revoke_refresh_token(session, stored.jti)
    logger.info("Refresh token revoked", extra={"jti": stored.jti})
    return {"status": "logged-out"}
