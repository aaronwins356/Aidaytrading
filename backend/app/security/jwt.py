"""JWT helper utilities."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Tuple
from uuid import uuid4

from jose import JWTError, jwt

from ..config import get_settings


class TokenError(RuntimeError):
    """Raised when token decoding fails."""


def _build_payload(subject: str, token_type: str, expires_delta: timedelta) -> Dict[str, Any]:
    settings = get_settings()
    expire = datetime.now(timezone.utc) + expires_delta
    return {
        "sub": subject,
        "type": token_type,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "jti": uuid4().hex,
        "iss": settings.app_name,
    }


def create_access_token(subject: str, additional_claims: Dict[str, Any] | None = None) -> Tuple[str, str, datetime]:
    """Create a signed access token."""

    settings = get_settings()
    payload = _build_payload(subject, "access", timedelta(seconds=settings.jwt_access_expires))
    if additional_claims:
        payload.update(additional_claims)
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return token, payload["jti"], payload["exp"]


def create_refresh_token(subject: str, additional_claims: Dict[str, Any] | None = None) -> Tuple[str, str, datetime]:
    """Create a signed refresh token."""

    settings = get_settings()
    payload = _build_payload(subject, "refresh", timedelta(seconds=settings.jwt_refresh_expires))
    if additional_claims:
        payload.update(additional_claims)
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return token, payload["jti"], payload["exp"]


def decode_token(token: str) -> Dict[str, Any]:
    """Decode and validate a token."""

    settings = get_settings()
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm], issuer=settings.app_name)
    except JWTError as exc:  # pragma: no cover - jose already tested
        raise TokenError("Invalid token") from exc
