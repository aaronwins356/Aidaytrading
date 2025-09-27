"""Authentication endpoints."""
from __future__ import annotations

import datetime as dt
import re
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import or_, select

from app.core import jwt
from app.core.dependencies import DBSession
from app.core.logging import mask_email, record_validation_error
from app.core.security import (
    PasswordValidationError,
    hash_password,
    validate_email_format,
    verify_password,
)
from app.models.user import User, UserRole, UserStatus
from app.schemas import auth as auth_schema
from app.services.brevo_email import BrevoEmailService
from app.services.ratelimiter import login_rate_limiter

router = APIRouter()

_email_service = BrevoEmailService()


def get_email_service() -> BrevoEmailService:
    return _email_service


@router.post("/signup", response_model=auth_schema.SignupResponse, status_code=status.HTTP_201_CREATED)
async def signup(
    payload: auth_schema.SignupRequest,
    request: Request,
    session: DBSession,
    email_service: Annotated[BrevoEmailService, Depends(get_email_service)]
) -> auth_schema.SignupResponse:
    """Register a new user account with pending status."""

    username = payload.username.strip()
    if not re.fullmatch(r"^[A-Za-z0-9_]{3,30}$", username):
        record_validation_error(request, "invalid_username", {"username": payload.username})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"code": "invalid_username", "message": "Username must contain only letters, numbers, or underscores."}},
        )

    try:
        email_normalized = payload.email.strip().lower()
        validate_email_format(email_normalized)
        password_hash = hash_password(payload.password)
    except PasswordValidationError as exc:
        record_validation_error(request, "weak_password", {"username": payload.username})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"code": "weak_password", "message": str(exc)}},
        ) from exc
    except ValueError as exc:
        record_validation_error(request, "invalid_email", {"email": mask_email(email_normalized)})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"code": "invalid_email", "message": "Email address is invalid."}},
        ) from exc

    stmt = select(User).where(or_(User.username == username, User.email_canonical == email_normalized))
    existing = await session.execute(stmt)
    if existing.scalar_one_or_none():
        record_validation_error(request, "duplicate_user", {"username": username})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"code": "user_exists", "message": "Username or email already registered."}},
        )

    user = User(
        username=username,
        email=email_normalized,
        email_canonical=email_normalized,
        password_hash=password_hash,
        role=UserRole.VIEWER,
        status=UserStatus.PENDING,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)

    await email_service.notify_admin_of_signup(
        user_id=user.id, username=user.username, email=user.email
    )

    return auth_schema.SignupResponse(message="Signup received. Await approval.", status=user.status.value)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    if request.client:
        return request.client.host or "unknown"
    return "unknown"


@router.post("/login", response_model=auth_schema.TokenPairResponse)
async def login(
    payload: auth_schema.LoginRequest,
    request: Request,
    session: DBSession,
) -> auth_schema.TokenPairResponse:
    """Authenticate a user and return an access/refresh token pair."""

    key = _client_ip(request)
    allowed, retry_after = await login_rate_limiter.check(key)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": {
                    "code": "too_many_attempts",
                    "message": "Too many login attempts. Please try again later.",
                    "retry_after": retry_after,
                }
            },
        )

    username = payload.username.strip()
    stmt = select(User).where(User.username == username)
    result = await session.execute(stmt)
    user = result.scalar_one_or_none()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"code": "invalid_credentials", "message": "Invalid credentials or inactive account."}},
        )

    if user.status != UserStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"code": "inactive_account", "message": "Invalid credentials or inactive account."}},
        )

    access = jwt.create_access_token(
        str(user.id), role=user.role.value, status=user.status.value, token_version=user.token_version
    )
    refresh = jwt.create_refresh_token(
        str(user.id), role=user.role.value, status=user.status.value, token_version=user.token_version
    )

    await login_rate_limiter.reset(key)

    return auth_schema.TokenPairResponse(access_token=access["token"], refresh_token=refresh["token"])


def _extract_refresh_token(request: Request, payload: auth_schema.RefreshRequest) -> str:
    if payload.refresh_token:
        return payload.refresh_token
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header.split(" ", 1)[1]
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"error": {"code": "missing_token", "message": "Refresh token required."}},
    )


@router.post("/refresh", response_model=auth_schema.AccessTokenResponse)
async def refresh_token(
    payload: auth_schema.RefreshRequest,
    request: Request,
    session: DBSession,
) -> auth_schema.AccessTokenResponse:
    """Exchange a refresh token for a new access token."""

    token = _extract_refresh_token(request, payload)
    try:
        decoded = jwt.decode_token(token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"code": "invalid_token", "message": "Invalid or expired token."}},
        ) from exc

    if decoded.get("token_type") != jwt.TokenType.REFRESH.value:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"code": "invalid_token", "message": "Refresh token required."}},
        )

    jti = decoded.get("jti")
    if jti is None or await jwt.is_token_blacklisted(session, jti):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"code": "token_revoked", "message": "Token has been revoked."}},
        )

    user_id = decoded.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"code": "invalid_token", "message": "Invalid token payload."}},
        )
    user_id_int = int(user_id)
    stmt = select(User).where(User.id == user_id_int)
    user = (await session.execute(stmt)).scalar_one_or_none()
    if not user or user.status != UserStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"code": "inactive_account", "message": "Invalid credentials or inactive account."}},
        )

    token_version = decoded.get("token_version")
    if token_version is None or token_version != user.token_version:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"code": "token_revoked", "message": "Token has been revoked."}},
        )

    access = jwt.create_access_token(
        str(user.id),
        role=user.role.value,
        status=user.status.value,
        token_version=user.token_version,
    )
    return auth_schema.AccessTokenResponse(access_token=access["token"])


@router.post("/logout", response_model=dict)
async def logout(
    request: Request,
    payload: auth_schema.LogoutRequest,
    session: DBSession,
) -> dict[str, str]:
    """Invalidate the provided tokens by blacklisting their JTIs."""

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"code": "unauthorized", "message": "Missing credentials."}},
        )
    access_token = auth_header.split(" ", 1)[1]

    tokens_to_blacklist: list[tuple[str, dt.datetime]] = []

    try:
        access_payload = jwt.decode_token(access_token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"code": "invalid_token", "message": "Invalid or expired token."}},
        ) from exc

    if access_payload.get("token_type") != jwt.TokenType.ACCESS.value:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"code": "invalid_token", "message": "Invalid or expired token."}},
        )
    if access_payload.get("jti"):
        expires = dt.datetime.fromtimestamp(int(access_payload["exp"]), tz=dt.timezone.utc)
        tokens_to_blacklist.append((access_payload["jti"], expires))

    if payload.refresh_token:
        try:
            refresh_payload = jwt.decode_token(payload.refresh_token)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error": {"code": "invalid_token", "message": "Invalid or expired token."}},
            ) from exc
        if refresh_payload.get("token_type") != jwt.TokenType.REFRESH.value:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error": {"code": "invalid_token", "message": "Invalid or expired token."}},
            )
        if refresh_payload.get("jti"):
            expires = dt.datetime.fromtimestamp(int(refresh_payload["exp"]), tz=dt.timezone.utc)
            tokens_to_blacklist.append((refresh_payload["jti"], expires))

    for jti, expires_at in tokens_to_blacklist:
        if not await jwt.is_token_blacklisted(session, jti):
            await jwt.store_token_jti(session, jti, expires_at)

    await session.commit()
    return {"message": "Logged out"}
