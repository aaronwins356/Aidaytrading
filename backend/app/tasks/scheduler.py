"""Scheduler orchestration for recurring jobs."""
from __future__ import annotations

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from app.core.config import get_settings
from app.services.reporting import CENTRAL_TZ
from app.tasks import jobs


_scheduler = AsyncIOScheduler(timezone="UTC")
_configured = False


def configure_scheduler() -> None:
    global _configured
    settings = get_settings()
    if not settings.scheduler_enabled or _configured:
        return

    logger.info("scheduler_configure", equity_interval=settings.equity_snapshot_interval_minutes)

    _scheduler.add_job(
        jobs.capture_equity_snapshot_job,
        IntervalTrigger(minutes=settings.equity_snapshot_interval_minutes),
        id="equity_snapshot",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=60,
    )

    _scheduler.add_job(
        jobs.daily_pnl_rollup_job,
        CronTrigger(hour=0, minute=0, timezone=CENTRAL_TZ),
        id="daily_pnl_rollup",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=300,
    )

    heartbeat_trigger = CronTrigger.from_crontab(settings.heartbeat_cron, timezone=CENTRAL_TZ)
    _scheduler.add_job(
        jobs.heartbeat_job,
        heartbeat_trigger,
        id="heartbeat",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=120,
    )

    _configured = True


def start_scheduler() -> None:
    settings = get_settings()
    if not settings.scheduler_enabled:
        logger.info("scheduler_disabled")
        return

    configure_scheduler()
    if not _scheduler.running:
        _scheduler.start()
        logger.info("scheduler_started")


async def shutdown_scheduler() -> None:
    if _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("scheduler_stopped")


def get_scheduler() -> AsyncIOScheduler:
    return _scheduler


__all__ = ["start_scheduler", "shutdown_scheduler", "configure_scheduler", "get_scheduler"]

