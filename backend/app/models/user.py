"""Database models related to users and authentication."""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base


class UserStatus(str, enum.Enum):
    """Enumeration of lifecycle states for a user."""

    PENDING = "pending"
    ACTIVE = "active"
    DISABLED = "disabled"


class UserRole(str, enum.Enum):
    """Supported user roles."""

    VIEWER = "viewer"
    ADMIN = "admin"


class User(Base):
    """User record with authentication metadata."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[UserStatus] = mapped_column(Enum(UserStatus), default=UserStatus.PENDING, nullable=False)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.VIEWER, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    refresh_tokens: Mapped[list["RefreshToken"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", lazy="selectin"
    )
    devices: Mapped[list["DeviceToken"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", lazy="selectin"
    )


class RefreshToken(Base):
    """Stored refresh tokens with revocation support."""

    __tablename__ = "refresh_tokens"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"))
    token_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    jti: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    user: Mapped[User] = relationship(back_populates="refresh_tokens")


class DeviceToken(Base):
    """Registered push notification tokens for APNs/FCM."""

    __tablename__ = "device_tokens"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"))
    token: Mapped[str] = mapped_column(String(512), unique=True, nullable=False)
    platform: Mapped[str] = mapped_column(String(32), default="ios", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    user: Mapped[User] = relationship(back_populates="devices")


class NotificationLog(Base):
    """History of push and email notifications."""

    __tablename__ = "notification_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("users.id", ondelete="SET NULL"))
    channel: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(String(128), nullable=False)
    body: Mapped[str] = mapped_column(String(512), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    user: Mapped[User | None] = relationship()
