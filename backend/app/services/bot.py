"""Bot runtime control service."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.trade import BotState


class BotService:
    """Control bot start/stop and operating mode."""

    async def get_state(self, session: AsyncSession) -> BotState:
        result = await session.execute(select(BotState))
        state = result.scalar_one_or_none()
        if state is None:
            state = BotState(running=False, mode="paper")
            session.add(state)
            await session.commit()
            await session.refresh(state)
        return state

    async def start(self, session: AsyncSession) -> BotState:
        state = await self.get_state(session)
        state.running = True
        state.uptime_started_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(state)
        return state

    async def stop(self, session: AsyncSession) -> BotState:
        state = await self.get_state(session)
        state.running = False
        await session.commit()
        await session.refresh(state)
        return state

    async def set_mode(self, session: AsyncSession, mode: str) -> BotState:
        if mode not in {"paper", "live"}:
            raise ValueError("mode must be 'paper' or 'live'")
        state = await self.get_state(session)
        state.mode = mode
        await session.commit()
        await session.refresh(state)
        return state


bot_service = BotService()
