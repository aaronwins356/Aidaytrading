"""Initial user and token blacklist tables."""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from app.models.user import UserRole, UserStatus

# revision identifiers, used by Alembic.
revision = "20240401_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("username", sa.String(length=30), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("role", sa.Enum(UserRole, name="userrole"), nullable=False, server_default=UserRole.VIEWER.value),
        sa.Column("status", sa.Enum(UserStatus, name="userstatus"), nullable=False, server_default=UserStatus.PENDING.value),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_users_username", "users", ["username"], unique=True)
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "token_blacklist",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("jti", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_token_blacklist_jti", "token_blacklist", ["jti"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_token_blacklist_jti", table_name="token_blacklist")
    op.drop_table("token_blacklist")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_index("ix_users_username", table_name="users")
    op.drop_table("users")
    sa.Enum(name="userrole").drop(op.get_bind(), checkfirst=False)  # type: ignore[no-untyped-call]
    sa.Enum(name="userstatus").drop(op.get_bind(), checkfirst=False)  # type: ignore[no-untyped-call]
