"""Security helpers including password policy enforcement."""
from __future__ import annotations

import re
from typing import Final, cast

from passlib.context import CryptContext  # type: ignore[import-untyped]

PASSWORD_REGEX: Final[re.Pattern[str]] = re.compile(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$")
EMAIL_REGEX: Final[re.Pattern[str]] = re.compile(
    r"^(?:[a-zA-Z0-9_'^&+/=?`{|}~-]+(?:\.[a-zA-Z0-9_'^&+/=?`{|}~-]+)*|\"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*\")@"
    r"(?:(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}|\[(?:[0-9]{1,3}\.){3}[0-9]{1,3}\])$"
)

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class PasswordValidationError(ValueError):
    """Raised when the provided password fails policy validation."""


def validate_password_strength(password: str) -> None:
    """Validate password against the defined policy."""

    if not PASSWORD_REGEX.match(password):
        raise PasswordValidationError(
            "Password must be at least 8 characters long and include one uppercase letter, one lowercase letter, and one digit."
        )


def hash_password(password: str) -> str:
    """Return a bcrypt hash for the provided password."""

    validate_password_strength(password)
    return cast(str, _pwd_context.hash(password))


def verify_password(password: str, password_hash: str) -> bool:
    """Validate a plaintext password against a bcrypt hash."""

    return bool(_pwd_context.verify(password, password_hash))


def validate_email_format(email: str) -> None:
    """Ensure email addresses resemble RFC 5322 format."""

    if not EMAIL_REGEX.match(email):
        raise ValueError("Email address is not valid.")
