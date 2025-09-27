"""Background job implementations for APScheduler."""
from __future__ import annotations

import datetime as dt

from loguru import logger
from sqlalchemy import select

from app.core.database import get_session_factory
from app.models.device_token import DeviceToken
from app.models.user import User, UserStatus
from app.services.events import BALANCE_CHANNEL, EQUITY_CHANNEL, STATUS_CHANNEL
from app.services.metrics_source import get_metrics_source
from app.services.pubsub import event_bus
from app.services.push import send_push_to_user
from app.services.reporting import (
    CENTRAL_TZ,
    ProfitSummary,
    compute_profit_summary,
    get_system_status,
    record_equity_snapshot,
    upsert_daily_pnl,
)


async def capture_equity_snapshot_job() -> None:
    metrics_source = get_metrics_source()
    equity = await metrics_source.fetch_equity_snapshot()
    session_factory = get_session_factory()
    async with session_factory() as session:
        snapshot = await record_equity_snapshot(session, equity=equity)
        await session.commit()
        payload = {
            "event": EQUITY_CHANNEL,
            "data": {
                "timestamp": snapshot.timestamp.isoformat(),
                "equity": format(snapshot.equity, "f"),
            },
        }
        await event_bus.publish(EQUITY_CHANNEL, payload)


async def daily_pnl_rollup_job(target_date: dt.date | None = None) -> None:
    if target_date is None:
        now_central = dt.datetime.now(dt.timezone.utc).astimezone(CENTRAL_TZ)
        target_date = (now_central - dt.timedelta(days=1)).date()

    session_factory = get_session_factory()
    async with session_factory() as session:
        record = await upsert_daily_pnl(session, target_date)
        await session.commit()
        logger.info(
            "daily_pnl_rollup",
            date=record.date.isoformat(),
            pnl=format(record.pnl_amount, "f"),
            color=record.color.value,
        )


async def heartbeat_job() -> None:
    metrics_source = get_metrics_source()
    balance = await metrics_source.fetch_current_balance()

    session_factory = get_session_factory()
    async with session_factory() as session:
        summary: ProfitSummary = await compute_profit_summary(session)
        stmt = (
            select(DeviceToken.user_id)
            .join(User, DeviceToken.user_id == User.id)
            .where(User.status == UserStatus.ACTIVE)
            .distinct()
        )
        user_ids = [row[0] for row in await session.execute(stmt)]

        balance_str = format(balance, "f")
        pl_percent_str = format(summary.total_pl_percent, "f")
        for user_id in user_ids:
            await send_push_to_user(
                user_id,
                title="Trading heartbeat",
                body=f"Balance: ${balance_str}, P/L: {pl_percent_str}%",
                data={"balance": balance_str, "pl_percent": pl_percent_str},
                session=session,
            )

    payload = {
        "event": BALANCE_CHANNEL,
        "data": {
            "balance": balance_str,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    }
    await event_bus.publish(BALANCE_CHANNEL, payload)


async def broadcast_system_status() -> None:
    session_factory = get_session_factory()
    async with session_factory() as session:
        status_row = await get_system_status(session)
    payload = {
        "event": STATUS_CHANNEL,
        "data": {
            "running": status_row.running,
            "started_at": status_row.started_at.isoformat() if status_row.started_at else None,
            "stopped_at": status_row.stopped_at.isoformat() if status_row.stopped_at else None,
        },
    }
    await event_bus.publish(STATUS_CHANNEL, payload)


__all__ = [
    "capture_equity_snapshot_job",
    "daily_pnl_rollup_job",
    "heartbeat_job",
    "broadcast_system_status",
]

