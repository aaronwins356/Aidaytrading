"""Trade endpoints."""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.trade import Trade
from ..schemas.trade import TradeOut, TradePage
from ..security.auth import get_current_user

router = APIRouter(prefix="/trades", tags=["trades"])


@router.get("", response_model=TradePage)
async def list_trades(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    symbol: str | None = Query(None, min_length=1),
    side: str | None = Query(None, min_length=3, max_length=5),
    start: datetime | None = Query(None),
    end: datetime | None = Query(None),
    session: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
) -> TradePage:
    filters = []
    if symbol:
        filters.append(Trade.symbol == symbol)
    if side:
        filters.append(Trade.side == side.lower())
    if start:
        filters.append(Trade.executed_at >= start)
    if end:
        filters.append(Trade.executed_at <= end)

    base_query = select(Trade).order_by(Trade.executed_at.desc())
    count_query = select(func.count(Trade.id))
    if filters:
        base_query = base_query.where(*filters)
        count_query = count_query.where(*filters)

    offset = (page - 1) * page_size
    result = await session.execute(base_query.offset(offset).limit(page_size))
    trades = result.scalars().all()
    total = await session.scalar(count_query) or 0
    return TradePage(
        items=[TradeOut.model_validate(t) for t in trades],
        page=page,
        page_size=page_size,
        total=total,
    )
