"""Common FastAPI dependencies."""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import jwt
from app.core.database import get_db_session
from app.models.user import User, UserRole, UserStatus

DBSession = Annotated[AsyncSession, Depends(get_db_session)]


def _extract_bearer_token(request: Request, token: str | None) -> str:
    if token:
        return token
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"code": "unauthorized", "message": "Missing credentials."}},
        )
    return auth_header.split(" ", 1)[1]


async def _resolve_user(
    request: Request,
    session: AsyncSession,
    *,
    token: str | None,
    require_active: bool,
    require_admin: bool,
) -> User:
    raw_token = _extract_bearer_token(request, token)
    try:
        payload = jwt.decode_token(raw_token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"code": "invalid_token", "message": "Invalid or expired token."}},
        ) from exc

    if payload.get("token_type") != jwt.TokenType.ACCESS.value:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"code": "invalid_token", "message": "Access token required."}},
        )

    jti = payload.get("jti")
    if jti is None or await jwt.is_token_blacklisted(session, jti):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"code": "token_revoked", "message": "Token has been revoked."}},
        )

    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"code": "invalid_token", "message": "Invalid token payload."}},
        )

    stmt = select(User).where(User.id == int(user_id))
    result = await session.execute(stmt)
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"code": "user_not_found", "message": "User not found."}},
        )

    token_version = payload.get("token_version")
    if token_version is None or token_version != user.token_version:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"code": "token_revoked", "message": "Token has been revoked."}},
        )

    if require_active and user.status != UserStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": {"code": "inactive_user", "message": "Account is not active."}},
        )

    if require_admin and user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": {"code": "forbidden", "message": "Admin privileges required."}},
        )

    request.state.user = user
    request.state.user_id = user.id
    return user


async def require_active_user(
    request: Request,
    session: DBSession,
    token: str | None = None,
) -> User:
    """Dependency that returns the authenticated active user."""

    return await _resolve_user(
        request,
        session,
        token=token,
        require_active=True,
        require_admin=False,
    )


async def require_admin_active_user(
    request: Request,
    session: DBSession,
    token: str | None = None,
) -> User:
    """Dependency that enforces active admin access."""

    return await _resolve_user(
        request,
        session,
        token=token,
        require_active=True,
        require_admin=True,
    )


async def get_current_user(
    request: Request,
    session: DBSession,
    token: str | None = None,
) -> User:
    """Backward-compatible alias for require_active_user."""

    return await require_active_user(request, session, token)
