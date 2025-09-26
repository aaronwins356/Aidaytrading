"""Common FastAPI dependencies."""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import jwt
from app.core.database import get_db_session
from app.models.user import User, UserStatus

DBSession = Annotated[AsyncSession, Depends(get_db_session)]


async def get_current_user(
    request: Request,
    session: DBSession,
    token: str | None = None,
) -> User:
    """Resolve the currently authenticated user from the Authorization header."""

    if token is None:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error": {"code": "unauthorized", "message": "Missing credentials."}},
            )
        token = auth_header.split(" ", 1)[1]

    try:
        payload = jwt.decode_token(token)
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

    if user.status != UserStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": {"code": "inactive_user", "message": "Account is not active."}},
        )

    request.state.user = user
    return user
