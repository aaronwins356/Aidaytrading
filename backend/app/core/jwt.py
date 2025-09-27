"""JWT helper utilities."""
from __future__ import annotations

import datetime as dt
import uuid
from enum import Enum
from typing import TypedDict, cast

from jose import JWTError, jwt  # type: ignore[import-untyped]
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.token_blacklist import TokenBlacklist

settings = get_settings()


class TokenType(str, Enum):
    ACCESS = "access"
    REFRESH = "refresh"


class TokenDetails(TypedDict):
    token: str
    expires_at: dt.datetime
    jti: str


class TokenPayload(TypedDict, total=False):
    sub: str
    role: str
    status: str
    jti: str
    token_type: str
    exp: int
    iat: int
    token_version: int


def _create_token(
    subject: str,
    token_type: TokenType,
    *,
    role: str,
    status: str,
    token_version: int,
) -> TokenDetails:
    now = dt.datetime.now(dt.timezone.utc)
    if token_type is TokenType.ACCESS:
        expires = now + dt.timedelta(minutes=settings.access_token_expires_min)
    else:
        expires = now + dt.timedelta(days=settings.refresh_token_expires_days)

    jti = uuid.uuid4().hex
    payload: TokenPayload = {
        "sub": subject,
        "role": role,
        "status": status,
        "jti": jti,
        "token_type": token_type.value,
        "iat": int(now.timestamp()),
        "exp": int(expires.timestamp()),
        "token_version": token_version,
    }

    token = jwt.encode(payload, settings.jwt_secret.get_secret_value(), algorithm=settings.jwt_algorithm)
    return {"token": token, "expires_at": expires, "jti": jti}


def create_access_token(subject: str, *, role: str, status: str, token_version: int) -> TokenDetails:
    """Create an access token for the given identity."""

    return _create_token(subject, TokenType.ACCESS, role=role, status=status, token_version=token_version)


def create_refresh_token(subject: str, *, role: str, status: str, token_version: int) -> TokenDetails:
    """Create a refresh token for the given identity."""

    return _create_token(subject, TokenType.REFRESH, role=role, status=status, token_version=token_version)


def decode_token(token: str) -> TokenPayload:
    """Decode a JWT token and validate signature."""

    try:
        payload = jwt.decode(token, settings.jwt_secret.get_secret_value(), algorithms=[settings.jwt_algorithm])
        return cast(TokenPayload, payload)
    except JWTError as exc:  # pragma: no cover - specific error message not needed in tests
        raise ValueError("Invalid token.") from exc


async def is_token_blacklisted(session: AsyncSession, jti: str) -> bool:
    """Return True if the token JTI exists in the blacklist."""

    stmt = select(TokenBlacklist.id).where(TokenBlacklist.jti == jti)
    result = await session.execute(stmt)
    return result.scalar_one_or_none() is not None


async def store_token_jti(session: AsyncSession, jti: str, expires_at: dt.datetime) -> None:
    """Persist a token identifier into the blacklist."""

    record = TokenBlacklist(jti=jti, expires_at=expires_at)
    session.add(record)
    await session.flush()


async def cleanup_expired_tokens(session: AsyncSession) -> int:
    """Delete expired token blacklist entries. Returns count deleted."""

    now = dt.datetime.now(dt.timezone.utc)
    stmt = select(TokenBlacklist).where(TokenBlacklist.expires_at < now)
    result = await session.execute(stmt)
    expired = result.scalars().all()
    deleted = 0
    for entry in expired:
        await session.delete(entry)
        deleted += 1
    return deleted
