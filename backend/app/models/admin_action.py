"""Administrative audit log model."""
from __future__ import annotations

from typing import Any

from sqlalchemy import JSON, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class AdminAction(TimestampMixin, Base):
    """Append-only audit trail for privileged operations."""

    __tablename__ = "admin_actions"
    __table_args__ = (
        Index("ix_admin_actions_admin_id", "admin_id"),
        Index("ix_admin_actions_target_user_id", "target_user_id"),
        Index("ix_admin_actions_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    admin_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"), nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    target_user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"), nullable=False)
    details: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSON, nullable=True)
