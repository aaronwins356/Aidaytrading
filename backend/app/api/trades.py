"""Trade endpoints."""
from __future__ import annotations

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
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
) -> TradePage:
    result = await session.execute(
        select(Trade).order_by(Trade.executed_at.desc()).offset(offset).limit(limit)
    )
    trades = result.scalars().all()
    total = await session.scalar(select(func.count(Trade.id)))
    return TradePage(items=[TradeOut.model_validate(t) for t in trades], total=total or 0)
