"""Notification scheduling and dispatch."""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Iterable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import AsyncSessionLocal
from ..models.trade import BalanceSnapshot, BotState, SchedulerRun
from ..models.user import DeviceToken, NotificationLog, NotificationPreference
from ..utils.logger import get_logger
from .push import push_service

logger = get_logger(__name__)


class NotificationService:
    """Coordinates scheduled notifications."""

    def __init__(self) -> None:
        self.scheduler = AsyncIOScheduler()
        self._token_cache: dict[str, list[str]] = {}
        self._preference_cache: dict[str, dict[str, bool]] = {}
        self._configure_jobs()

    def _configure_jobs(self) -> None:
        hours = [8, 14, 20, 2]
        for hour in hours:
            trigger = CronTrigger(hour=hour, minute=0, timezone="America/Chicago")
            self.scheduler.add_job(self._heartbeat_job, trigger=trigger, id=f"heartbeat-{hour}")

    async def _heartbeat_job(self) -> None:
        async with AsyncSessionLocal() as session:
            try:
                bot_state = await session.scalar(select(BotState))
                latest_snapshot = await session.scalar(
                    select(BalanceSnapshot).order_by(BalanceSnapshot.timestamp.desc())
                )
                uptime_seconds = None
                if bot_state and bot_state.running and bot_state.uptime_started_at:
                    uptime_seconds = (datetime.utcnow() - bot_state.uptime_started_at).total_seconds()
                balance = float(latest_snapshot.balance) if latest_snapshot else 0.0
                equity = float(latest_snapshot.equity) if latest_snapshot else 0.0
                message = (
                    f"Bot {'running' if bot_state and bot_state.running else 'stopped'} | "
                    f"Balance: {balance:.2f} | Equity: {equity:.2f}"
                )
                if uptime_seconds is not None:
                    hours = uptime_seconds / 3600
                    message = f"{message} | Uptime: {hours:.1f}h"
                tokens = await self._fetch_tokens(session, "heartbeat_push")
                await push_service.send_push(tokens, "Heartbeat", message)
                await self._log_notification(session, None, "push", "Heartbeat", message)
                await self._record_scheduler_run(session, "heartbeat", "success", message)
            except Exception as exc:  # pragma: no cover - scheduler resilience
                logger.exception("Heartbeat job failed", exc_info=exc)
                await self._record_scheduler_run(session, "heartbeat", "error", str(exc))

    async def _fetch_tokens(self, session: AsyncSession, event_key: str) -> Iterable[str]:
        token_cache = await self._load_token_cache(session)
        preference_cache = await self._load_preference_cache(session)
        tokens: list[str] = []
        default_preferences = NotificationPreference.default_preferences()
        for user_id, user_tokens in token_cache.items():
            prefs = preference_cache.get(user_id, default_preferences)
            if prefs.get(event_key, True):
                tokens.extend(user_tokens)
        return tokens

    async def _load_token_cache(self, session: AsyncSession) -> dict[str, list[str]]:
        if not self._token_cache:
            result = await session.execute(select(DeviceToken.user_id, DeviceToken.token))
            tokens: dict[str, list[str]] = {}
            for user_id, token in result.all():
                tokens.setdefault(user_id, []).append(token)
            self._token_cache = tokens
        return self._token_cache

    async def _load_preference_cache(
        self, session: AsyncSession
    ) -> dict[str, dict[str, bool]]:
        if not self._preference_cache:
            result = await session.execute(select(NotificationPreference))
            for preference in result.scalars():
                self._preference_cache[preference.user_id] = preference.merged_preferences()
        return self._preference_cache

    async def _log_notification(
        self,
        session,
        user_id: str | None,
        channel: str,
        title: str,
        body: str,
    ) -> None:
        log = NotificationLog(user_id=user_id, channel=channel, title=title, body=body)
        session.add(log)
        await session.commit()

    async def _record_scheduler_run(
        self,
        session,
        job_name: str,
        status: str,
        detail: str | None = None,
    ) -> None:
        run = SchedulerRun(job_name=job_name, status=status, detail=detail)
        session.add(run)
        await session.commit()

    def invalidate_cache(self, user_id: str | None = None) -> None:
        """Invalidate cached token and preference snapshots."""

        self._token_cache.clear()
        self._preference_cache.clear()

    def start(self) -> None:
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("Notification scheduler started")

    async def shutdown(self) -> None:
        if self.scheduler.running:
            await asyncio.to_thread(self.scheduler.shutdown)
            logger.info("Notification scheduler stopped")


notification_service = NotificationService()
