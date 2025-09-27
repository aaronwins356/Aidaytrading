"""Admin audit logs, email normalization, and token versioning."""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "20240415_0002"
down_revision = "20240401_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("users", schema=None) as batch:
        batch.add_column(sa.Column("email_canonical", sa.String(length=320), nullable=True))
        batch.add_column(sa.Column("token_version", sa.Integer(), nullable=False, server_default="0"))

    connection = op.get_bind()
    connection.execute(sa.text("UPDATE users SET email_canonical = lower(email)"))

    with op.batch_alter_table("users", schema=None) as batch:
        batch.alter_column("email_canonical", existing_type=sa.String(length=320), nullable=False)

    op.create_index("ix_users_email_canonical", "users", ["email_canonical"], unique=True)

    op.create_table(
        "admin_actions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("admin_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column(
            "target_user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_admin_actions_admin_id", "admin_actions", ["admin_id"])
    op.create_index("ix_admin_actions_target_user_id", "admin_actions", ["target_user_id"])
    op.create_index("ix_admin_actions_created_at", "admin_actions", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_admin_actions_created_at", table_name="admin_actions")
    op.drop_index("ix_admin_actions_target_user_id", table_name="admin_actions")
    op.drop_index("ix_admin_actions_admin_id", table_name="admin_actions")
    op.drop_table("admin_actions")

    op.drop_index("ix_users_email_canonical", table_name="users")

    with op.batch_alter_table("users", schema=None) as batch:
        batch.drop_column("token_version")
        batch.drop_column("email_canonical")
