"""Bot management API."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.trade import BotState
from ..models.user import DeviceToken, NotificationLog, User
from ..schemas.trade import StatusResponse
from ..security.auth import get_current_user, require_admin
from ..services.bot import bot_service
from ..services.push import push_service

router = APIRouter(prefix="/bot", tags=["bot"])


async def _state_to_status(state: BotState) -> StatusResponse:
    uptime = None
    if state.running and state.uptime_started_at:
        started = state.uptime_started_at
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        uptime = (datetime.now(timezone.utc) - started).total_seconds()
    return StatusResponse(running=state.running, mode=state.mode, uptime_seconds=uptime)


async def _notify(session: AsyncSession, title: str, body: str) -> None:
    result = await session.execute(select(DeviceToken.token))
    tokens = [row[0] for row in result.all()]
    await push_service.send_push(tokens, title, body)
    log = NotificationLog(user_id=None, channel="push", title=title, body=body)
    session.add(log)
    await session.commit()


@router.get("/status", response_model=StatusResponse)
async def get_status(session: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)) -> StatusResponse:
    state = await bot_service.get_state(session)
    return await _state_to_status(state)


@router.post("/start", response_model=StatusResponse)
async def start_bot(
    admin: User = Depends(require_admin),
    session: AsyncSession = Depends(get_db),
) -> StatusResponse:
    state = await bot_service.start(session)
    await _notify(session, "Bot started", f"Started by {admin.username}")
    return await _state_to_status(state)


@router.post("/stop", response_model=StatusResponse)
async def stop_bot(
    admin: User = Depends(require_admin),
    session: AsyncSession = Depends(get_db),
) -> StatusResponse:
    state = await bot_service.stop(session)
    await _notify(session, "Bot stopped", f"Stopped by {admin.username}")
    return await _state_to_status(state)


@router.patch("/mode", response_model=StatusResponse)
async def set_mode(
    payload: dict[str, str],
    admin: User = Depends(require_admin),
    session: AsyncSession = Depends(get_db),
) -> StatusResponse:
    mode = payload.get("mode")
    if mode not in {"paper", "live"}:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid mode")
    state = await bot_service.set_mode(session, mode)
    await _notify(session, "Bot mode changed", f"Mode set to {mode} by {admin.username}")
    return await _state_to_status(state)
