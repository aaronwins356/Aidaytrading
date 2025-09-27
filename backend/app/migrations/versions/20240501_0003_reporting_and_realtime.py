"""Add reporting, trade, and device token tables."""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

from app.models.device_token import DevicePlatform
from app.models.reporting import DailyPnLColor
from app.models.trade import TradeSide


revision = "20240501_0003_reporting_and_realtime"
down_revision = "20240415_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "equity_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("equity", sa.Numeric(18, 2), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False, server_default="bot"),
    )

    op.create_index(
        "ix_equity_snapshots_timestamp",
        "equity_snapshots",
        ["timestamp"],
    )

    op.create_table(
        "daily_pnl",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("pnl_amount", sa.Numeric(18, 2), nullable=False),
        sa.Column("color", sa.Enum(DailyPnLColor, name="daily_pnl_color"), nullable=False),
        sa.UniqueConstraint("date", name="uq_daily_pnl_date"),
    )

    op.create_index("ix_daily_pnl_date", "daily_pnl", ["date"])

    op.create_table(
        "trades",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("side", sa.Enum(TradeSide, name="trade_side"), nullable=False),
        sa.Column("size", sa.Numeric(20, 8), nullable=False),
        sa.Column("pnl", sa.Numeric(18, 2), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_index("ix_trades_symbol_timestamp", "trades", ["symbol", "timestamp"])
    op.create_index("ix_trades_timestamp", "trades", ["timestamp"])

    op.create_table(
        "system_status",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("running", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("stopped_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_index("ix_system_status_singleton", "system_status", ["id"])

    op.create_table(
        "device_tokens",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("token", sa.String(length=255), nullable=False),
        sa.Column(
            "platform",
            sa.Enum(DevicePlatform, name="device_platform"),
            nullable=False,
            server_default=DevicePlatform.IOS.value,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("token", name="uq_device_tokens_token"),
    )

    op.create_index("ix_device_tokens_user_id", "device_tokens", ["user_id"])

    op.bulk_insert(
        sa.table(
            "system_status",
            sa.column("id", sa.Integer()),
            sa.column("running", sa.Boolean()),
            sa.column("started_at", sa.DateTime(timezone=True)),
            sa.column("stopped_at", sa.DateTime(timezone=True)),
        ),
        [
            {
                "id": 1,
                "running": False,
                "started_at": None,
                "stopped_at": None,
            }
        ],
    )


def downgrade() -> None:
    op.drop_index("ix_device_tokens_user_id", table_name="device_tokens")
    op.drop_table("device_tokens")

    op.drop_index("ix_system_status_singleton", table_name="system_status")
    op.drop_table("system_status")

    op.drop_index("ix_trades_symbol_timestamp", table_name="trades")
    op.drop_index("ix_trades_timestamp", table_name="trades")
    op.drop_table("trades")

    op.drop_index("ix_daily_pnl_date", table_name="daily_pnl")
    op.drop_table("daily_pnl")

    op.drop_index("ix_equity_snapshots_timestamp", table_name="equity_snapshots")
    op.drop_table("equity_snapshots")

    sa.Enum(name="device_platform").drop(op.get_bind(), checkfirst=False)  # type: ignore[call-arg]
    sa.Enum(name="daily_pnl_color").drop(op.get_bind(), checkfirst=False)  # type: ignore[call-arg]
    sa.Enum(name="trade_side").drop(op.get_bind(), checkfirst=False)  # type: ignore[call-arg]

