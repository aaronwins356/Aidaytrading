"""Authentication utilities and dependencies."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from passlib.context import CryptContext
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.user import RefreshToken, User, UserRole, UserStatus
from ..schemas.auth import TokenBundle
from ..schemas.user import UserOut, UserProfile
from ..utils.logger import get_logger
from .jwt import TokenError, create_access_token, create_refresh_token, decode_token

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = HTTPBearer(auto_error=False)
logger = get_logger(__name__)


async def get_user_by_username(session: AsyncSession, username: str) -> User | None:
    result = await session.execute(select(User).where(User.username == username))
    return result.scalar_one_or_none()


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)


def hash_token(token: str) -> str:
    return pwd_context.hash(token)


def verify_token_hash(token: str, token_hash: str) -> bool:
    return pwd_context.verify(token, token_hash)


async def authenticate_user(session: AsyncSession, username: str, password: str) -> User:
    user = await get_user_by_username(session, username)
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if user.status != UserStatus.ACTIVE:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User is not approved")
    return user


async def create_tokens(session: AsyncSession, user: User) -> TokenBundle:
    access_token, access_jti, access_exp = create_access_token(
        user.id, {"role": user.role.value, "username": user.username}
    )
    if isinstance(access_exp, (int, float)):
        access_exp_dt = datetime.fromtimestamp(access_exp, tz=timezone.utc)
    else:
        access_exp_dt = access_exp
    refresh_token, refresh_jti, refresh_exp = create_refresh_token(user.id)
    if isinstance(refresh_exp, (int, float)):
        refresh_exp_dt = datetime.fromtimestamp(refresh_exp, tz=timezone.utc)
    else:
        refresh_exp_dt = refresh_exp
    refresh_record = RefreshToken(
        user_id=user.id,
        token_hash=hash_token(refresh_token),
        jti=refresh_jti,
        expires_at=refresh_exp_dt,
    )
    session.add(refresh_record)
    await session.commit()
    access_expires_in = int((access_exp_dt - datetime.now(timezone.utc)).total_seconds())
    refresh_expires_in = int((refresh_exp_dt - datetime.now(timezone.utc)).total_seconds())
    return TokenBundle(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=max(access_expires_in, 0),
        refresh_expires_in=max(refresh_expires_in, 0),
    )


async def revoke_refresh_token(session: AsyncSession, jti: str) -> None:
    await session.execute(
        update(RefreshToken)
        .where(RefreshToken.jti == jti)
        .values(revoked=True, expires_at=datetime.now(timezone.utc))
    )
    await session.commit()


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(oauth2_scheme)],
    session: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing credentials")
    try:
        payload = decode_token(credentials.credentials)
    except (TokenError, JWTError) as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc
    if payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")
    user_id: str = payload.get("sub")
    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None or user.status != UserStatus.ACTIVE:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Inactive user")
    return user


async def require_admin(user: Annotated[User, Depends(get_current_user)]) -> User:
    if user.role != UserRole.ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required")
    return user


async def validate_refresh_token(session: AsyncSession, token: str) -> RefreshToken:
    try:
        payload = decode_token(token)
    except (TokenError, JWTError) as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Wrong token type")
    jti: str = payload.get("jti")
    user_id: str = payload.get("sub")
    result = await session.execute(
        select(RefreshToken).where(RefreshToken.jti == jti, RefreshToken.user_id == user_id)
    )
    stored = result.scalar_one_or_none()
    if stored is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token revoked")
    expires_at = stored.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if stored.revoked or expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    if not verify_token_hash(token, stored.token_hash):
        logger.warning("Refresh token hash mismatch", extra={"jti": jti})
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token compromised")
    return stored


def to_user_out(user: User) -> UserOut:
    """Serialize a user to schema."""

    return UserOut(
        id=user.id,
        username=user.username,
        email=user.email,
        status=user.status,
        role=user.role,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )


def to_user_profile(user: User) -> UserProfile:
    """Serialize a user into a lightweight profile response."""

    return UserProfile(
        id=user.id,
        username=user.username,
        email=user.email,
        role=user.role,
        approval_status=user.status,
    )
