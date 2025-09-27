"""Monitoring endpoints."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.trade import BalanceSnapshot, SchedulerRun, Trade
from ..models.user import NotificationLog
from ..utils.logger import get_logger
from .websocket import ws_manager

router = APIRouter(prefix="", tags=["monitoring"])
logger = get_logger(__name__)


@router.get("/health")
async def health_check(request: Request, session: AsyncSession = Depends(get_db)) -> dict[str, object]:
    start_time: datetime = request.app.state.start_time
    uptime = (datetime.now(timezone.utc) - start_time).total_seconds()
    last_snapshot = await session.scalar(select(BalanceSnapshot.timestamp).order_by(BalanceSnapshot.timestamp.desc()))
    db_ok = True
    return {
        "status": "ok" if db_ok else "degraded",
        "uptime_seconds": uptime,
        "last_snapshot": last_snapshot.isoformat() if last_snapshot else None,
    }


@router.get("/metrics")
async def metrics(session: AsyncSession = Depends(get_db)) -> Response:
    trades_total = await session.scalar(select(func.count(Trade.id))) or 0
    notifications_total = await session.scalar(
        select(func.count(NotificationLog.id)).where(NotificationLog.channel == "push")
    ) or 0
    scheduler_total = await session.scalar(select(func.count(SchedulerRun.id))) or 0
    ws_clients = ws_manager.connected_clients()
    body = (
        "# HELP trades_executed_total Total number of trades executed\n"
        "# TYPE trades_executed_total counter\n"
        f"trades_executed_total {trades_total}\n"
        "# HELP ws_clients_connected Active websocket clients\n"
        "# TYPE ws_clients_connected gauge\n"
        f"ws_clients_connected {ws_clients}\n"
        "# HELP push_notifications_sent_total Push notifications dispatched\n"
        "# TYPE push_notifications_sent_total counter\n"
        f"push_notifications_sent_total {notifications_total}\n"
        "# HELP scheduler_runs_total Scheduler job executions\n"
        "# TYPE scheduler_runs_total counter\n"
        f"scheduler_runs_total {scheduler_total}\n"
    )
    return Response(content=body, media_type="text/plain; version=0.0.4")
