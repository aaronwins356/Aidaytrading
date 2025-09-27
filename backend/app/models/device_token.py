"""Push notification device token model."""
from __future__ import annotations

import datetime as dt
from enum import Enum

from sqlalchemy import DateTime, Enum as PgEnum, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class DevicePlatform(str, Enum):
    """Enumerated device platforms for push notifications."""

    IOS = "ios"
    ANDROID = "android"


class DeviceToken(Base):
    """Registered push notification device token."""

    __tablename__ = "device_tokens"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    token: Mapped[str] = mapped_column(String(255), nullable=False)
    platform: Mapped[DevicePlatform] = mapped_column(
        PgEnum(DevicePlatform, name="device_platform"), nullable=False, default=DevicePlatform.IOS
    )
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    user = relationship("User", back_populates="device_tokens")

    __table_args__ = (
        UniqueConstraint("token", name="uq_device_tokens_token"),
    )

