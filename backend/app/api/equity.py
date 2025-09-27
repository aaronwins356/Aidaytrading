"""Investor facing endpoints."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.trade import BalanceSnapshot, BotState, DailyPnL, EquityPoint, Trade
from ..schemas.trade import BalanceSnapshotOut, CalendarEntry, ProfitSummary, StatusResponse
from ..security.auth import get_current_user

router = APIRouter(tags=["investor"])


@router.get("/status", response_model=StatusResponse)
async def investor_status(
    session: AsyncSession = Depends(get_db), user=Depends(get_current_user)
) -> StatusResponse:
    state = await session.scalar(select(BotState))
    running = bool(state.running) if state else False
    mode = state.mode if state else "paper"
    uptime = None
    if state and state.running and state.uptime_started_at:
        started = state.uptime_started_at
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        uptime = (datetime.now(timezone.utc) - started).total_seconds()
    return StatusResponse(running=running, mode=mode, uptime_seconds=uptime)


@router.get("/profit", response_model=ProfitSummary)
async def profit_summary(
    session: AsyncSession = Depends(get_db), user=Depends(get_current_user)
) -> ProfitSummary:
    latest = await session.scalar(select(BalanceSnapshot).order_by(BalanceSnapshot.timestamp.desc()))
    if latest:
        equity = Decimal(latest.equity)
        balance = Decimal(latest.balance)
    else:
        equity = Decimal(0)
        balance = Decimal(0)
    result = await session.execute(
        select(func.sum(Trade.pnl), func.count(Trade.id)).where(Trade.pnl > 0)
    )
    pnl_sum, win_count = result.first() or (0, 0)
    total_trades = await session.scalar(select(func.count(Trade.id))) or 0
    win_rate = float(win_count) / total_trades * 100 if total_trades else 0.0
    pnl_absolute = equity - balance
    pnl_percent = (pnl_absolute / balance * 100) if balance else 0
    return ProfitSummary(
        current_balance=equity,
        total_pl_amount=pnl_absolute,
        total_pl_percent=pnl_percent,
        win_rate=win_rate,
    )


@router.get("/equity-curve", response_model=list[tuple[str, str]])
async def equity_curve(
    session: AsyncSession = Depends(get_db), user=Depends(get_current_user)
) -> list[tuple[str, str]]:
    result = await session.execute(select(EquityPoint).order_by(EquityPoint.timestamp))
    points = result.scalars().all()
    return [(point.timestamp.isoformat(), str(point.value)) for point in points]


@router.get("/calendar", response_model=list[CalendarEntry])
async def calendar(
    session: AsyncSession = Depends(get_db), user=Depends(get_current_user)
) -> list[CalendarEntry]:
    result = await session.execute(select(DailyPnL).order_by(DailyPnL.trading_day))
    days = result.scalars().all()
    return [CalendarEntry.model_validate(day) for day in days]


@router.get("/balance", response_model=list[BalanceSnapshotOut])
async def balance_history(
    session: AsyncSession = Depends(get_db), user=Depends(get_current_user)
) -> list[BalanceSnapshotOut]:
    result = await session.execute(select(BalanceSnapshot).order_by(BalanceSnapshot.timestamp))
    snapshots = result.scalars().all()
    return [BalanceSnapshotOut.model_validate(snapshot) for snapshot in snapshots]
