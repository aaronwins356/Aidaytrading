"""Token blacklist model."""
from __future__ import annotations

import datetime as dt

from sqlalchemy import DateTime, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class TokenBlacklist(TimestampMixin, Base):
    """Blacklist entry for JWT revocation."""

    __tablename__ = "token_blacklist"
    __table_args__ = (Index("ix_token_blacklist_jti", "jti", unique=True),)

    id: Mapped[int] = mapped_column(primary_key=True)
    jti: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    expires_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
