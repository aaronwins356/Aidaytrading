"""Reporting and realtime friendly API endpoints."""
from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import and_, func, select

from app.core.dependencies import DBSession, require_active_user
from app.models.device_token import DeviceToken
from app.models.trade import Trade, TradeSide
from app.schemas.reporting import (
    CalendarDay,
    CalendarResponse,
    DeviceRegistrationRequest,
    DeviceRegistrationResponse,
    EquityCurveResponse,
    ProfitResponse,
    StatusResponse,
    TradeRecord,
    TradesResponse,
)
from app.services.reporting import (
    CENTRAL_TZ,
    ProfitSummary,
    compute_profit_summary,
    get_calendar_heatmap,
    get_equity_curve,
    get_system_status,
)


router = APIRouter()


def _parse_datetime(value: str, *, param: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"code": "invalid_datetime", "message": f"Invalid datetime for {param}."}},
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _parse_date(value: str, *, param: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"code": "invalid_date", "message": f"Invalid date for {param}."}},
        ) from exc


@router.get("/equity-curve", response_model=EquityCurveResponse)
async def equity_curve(
    session: DBSession,
    user: Annotated[object, Depends(require_active_user)],
    start: str | None = Query(None),
    end: str | None = Query(None),
    limit: int | None = Query(None, ge=1, le=1000),
) -> EquityCurveResponse:
    now = dt.datetime.now(dt.timezone.utc)
    start_dt = _parse_datetime(start, param="start") if start else now - dt.timedelta(days=30)
    end_dt = _parse_datetime(end, param="end") if end else now
    if start_dt > end_dt:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"code": "invalid_range", "message": "start must be before end."}},
        )

    points = await get_equity_curve(session, start=start_dt, end=end_dt, limit=limit)
    response: EquityCurveResponse = [[ts.isoformat(), format(equity, "f")] for ts, equity in points]
    return response


@router.get("/profit", response_model=ProfitResponse)
async def profit(session: DBSession, user: Annotated[object, Depends(require_active_user)]) -> ProfitResponse:
    summary: ProfitSummary = await compute_profit_summary(session)
    return ProfitResponse(
        current_balance=summary.current_balance,
        total_pl_amount=summary.total_pl_amount,
        total_pl_percent=summary.total_pl_percent,
        win_rate=round(summary.win_rate, 4),
    )


@router.get("/calendar", response_model=CalendarResponse)
async def calendar(
    session: DBSession,
    user: Annotated[object, Depends(require_active_user)],
    start: str | None = Query(None),
    end: str | None = Query(None),
) -> CalendarResponse:
    today_central = dt.datetime.now(dt.timezone.utc).astimezone(CENTRAL_TZ).date()
    end_date = _parse_date(end, param="end") if end else today_central
    start_date = _parse_date(start, param="start") if start else end_date - dt.timedelta(days=29)
    if start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"code": "invalid_range", "message": "start must be before end."}},
        )

    calendar_map = await get_calendar_heatmap(session, start_date=start_date, end_date=end_date)
    return {day: CalendarDay(**payload) for day, payload in calendar_map.items()}


@router.get("/status", response_model=StatusResponse)
async def status_endpoint(
    session: DBSession, user: Annotated[object, Depends(require_active_user)]
) -> StatusResponse:
    status_row = await get_system_status(session)
    uptime = 0
    if status_row.running and status_row.started_at:
        uptime = int((dt.datetime.now(dt.timezone.utc) - status_row.started_at).total_seconds())
    return StatusResponse(running=status_row.running, uptime_seconds=max(uptime, 0))


@router.get("/trades", response_model=TradesResponse)
async def trades_endpoint(
    session: DBSession,
    user: Annotated[object, Depends(require_active_user)],
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    symbol: str | None = Query(None),
    side: TradeSide | None = Query(None),
    start: str | None = Query(None),
    end: str | None = Query(None),
) -> TradesResponse:
    filters = []
    if symbol:
        filters.append(Trade.symbol == symbol.upper())
    if side:
        filters.append(Trade.side == side)
    if start:
        filters.append(Trade.timestamp >= _parse_datetime(start, param="start"))
    if end:
        filters.append(Trade.timestamp <= _parse_datetime(end, param="end"))

    base_query = select(Trade)
    if filters:
        base_query = base_query.where(and_(*filters))

    count_stmt = select(func.count()).select_from(base_query.subquery())
    total = (await session.execute(count_stmt)).scalar_one()

    offset = (page - 1) * page_size
    stmt = base_query.order_by(Trade.timestamp.desc()).offset(offset).limit(page_size)
    records = (await session.execute(stmt)).scalars().all()

    items = [
        TradeRecord(
            id=trade.id,
            symbol=trade.symbol,
            side=trade.side,
            size=Decimal(trade.size),
            pnl=Decimal(trade.pnl),
            timestamp=trade.timestamp,
        )
        for trade in records
    ]

    return TradesResponse(items=items, page=page, page_size=page_size, total=total)


@router.post("/register-device", response_model=DeviceRegistrationResponse)
async def register_device(
    payload: DeviceRegistrationRequest,
    session: DBSession,
    user=Depends(require_active_user),
) -> DeviceRegistrationResponse:
    stmt = select(DeviceToken).where(DeviceToken.token == payload.token)
    existing = (await session.execute(stmt)).scalar_one_or_none()
    if existing:
        existing.user_id = user.id
        existing.platform = payload.platform
        message = "Device token updated."
    else:
        device = DeviceToken(user_id=user.id, token=payload.token, platform=payload.platform)
        session.add(device)
        message = "Device token registered."
    await session.commit()
    return DeviceRegistrationResponse(message=message)


__all__ = ["router"]

