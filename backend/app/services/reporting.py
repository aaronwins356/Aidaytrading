"""Reporting helpers for analytics endpoints and scheduled jobs."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Iterable

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.reporting import DailyPnL, DailyPnLColor, EquitySnapshot, SystemStatus
from app.models.trade import Trade

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python >=3.11 ships zoneinfo
    from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]


CENTRAL_TZ = ZoneInfo("America/Chicago")


@dataclass
class ProfitSummary:
    current_balance: Decimal
    total_pl_amount: Decimal
    total_pl_percent: Decimal
    win_rate: float


async def record_equity_snapshot(
    session: AsyncSession, *, equity: Decimal, source: str = "bot"
) -> EquitySnapshot:
    snapshot = EquitySnapshot(timestamp=dt.datetime.now(dt.timezone.utc), equity=equity, source=source)
    session.add(snapshot)
    await session.flush()
    await session.refresh(snapshot)
    return snapshot


async def get_latest_equity(session: AsyncSession) -> EquitySnapshot | None:
    stmt = select(EquitySnapshot).order_by(EquitySnapshot.timestamp.desc()).limit(1)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_equity_curve(
    session: AsyncSession,
    *,
    start: dt.datetime | None,
    end: dt.datetime | None,
    limit: int | None,
) -> list[tuple[dt.datetime, Decimal]]:
    stmt: Select[tuple[EquitySnapshot]] = select(EquitySnapshot).order_by(EquitySnapshot.timestamp.asc())
    if start is not None:
        stmt = stmt.where(EquitySnapshot.timestamp >= start)
    if end is not None:
        stmt = stmt.where(EquitySnapshot.timestamp <= end)
    if limit is not None:
        stmt = stmt.limit(limit)

    records = (await session.execute(stmt)).scalars().all()
    return [(row.timestamp, row.equity) for row in records]


async def get_baseline_equity(session: AsyncSession) -> Decimal:
    settings = get_settings()
    if settings.baseline_equity is not None:
        return settings.baseline_equity

    stmt = select(EquitySnapshot.equity).order_by(EquitySnapshot.timestamp.asc()).limit(1)
    result = await session.execute(stmt)
    value = result.scalar_one_or_none()
    return value or Decimal("0")


async def compute_win_rate(session: AsyncSession) -> float:
    total_stmt = select(func.count(Trade.id))
    won_stmt = select(func.count(Trade.id)).where(Trade.pnl > 0)

    total = (await session.execute(total_stmt)).scalar_one()
    if not total:
        return 0.0

    won = (await session.execute(won_stmt)).scalar_one()
    return float(won) / float(total)


async def compute_profit_summary(session: AsyncSession) -> ProfitSummary:
    baseline = await get_baseline_equity(session)
    latest = await get_latest_equity(session)
    current_balance = latest.equity if latest else baseline

    total_pl_amount = current_balance - baseline
    if baseline > 0:
        total_pl_percent = (total_pl_amount / baseline) * Decimal("100")
        total_pl_percent = total_pl_percent.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    else:
        total_pl_percent = Decimal("0")

    win_rate = await compute_win_rate(session)

    return ProfitSummary(
        current_balance=current_balance,
        total_pl_amount=total_pl_amount,
        total_pl_percent=total_pl_percent,
        win_rate=win_rate,
    )


def _determine_color(pnl_amount: Decimal) -> DailyPnLColor:
    if pnl_amount > 0:
        return DailyPnLColor.GREEN
    if pnl_amount < 0:
        return DailyPnLColor.RED
    return DailyPnLColor.NEUTRAL


def _date_range(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    delta = (end - start).days
    for offset in range(delta + 1):
        yield start + dt.timedelta(days=offset)


async def aggregate_trades_for_day(session: AsyncSession, target_date: dt.date) -> tuple[Decimal, DailyPnLColor]:
    start_central = dt.datetime.combine(target_date, dt.time.min, tzinfo=CENTRAL_TZ)
    end_central = start_central + dt.timedelta(days=1)

    start_utc = start_central.astimezone(dt.timezone.utc)
    end_utc = end_central.astimezone(dt.timezone.utc)

    stmt = select(func.coalesce(func.sum(Trade.pnl), 0)).where(
        Trade.timestamp >= start_utc, Trade.timestamp < end_utc
    )
    pnl_value = (await session.execute(stmt)).scalar_one()
    if isinstance(pnl_value, float):  # SQLite may return float
        pnl = Decimal(str(pnl_value))
    else:
        pnl = Decimal(pnl_value)
    color = _determine_color(pnl)
    return pnl, color


async def upsert_daily_pnl(session: AsyncSession, target_date: dt.date) -> DailyPnL:
    pnl_amount, color = await aggregate_trades_for_day(session, target_date)
    stmt = select(DailyPnL).where(DailyPnL.date == target_date)
    existing = (await session.execute(stmt)).scalar_one_or_none()
    if existing:
        existing.pnl_amount = pnl_amount
        existing.color = color
        await session.flush()
        return existing

    record = DailyPnL(date=target_date, pnl_amount=pnl_amount, color=color)
    session.add(record)
    await session.flush()
    await session.refresh(record)
    return record


async def get_calendar_heatmap(
    session: AsyncSession, *, start_date: dt.date, end_date: dt.date
) -> dict[str, dict[str, str | Decimal]]:
    stmt = select(DailyPnL).where(DailyPnL.date >= start_date, DailyPnL.date <= end_date)
    records = {row.date: row for row in (await session.execute(stmt)).scalars().all()}

    calendar: dict[str, dict[str, str | Decimal]] = {}
    for date in _date_range(start_date, end_date):
        record = records.get(date)
        if record is None:
            pnl_amount, color = await aggregate_trades_for_day(session, date)
        else:
            pnl_amount, color = record.pnl_amount, record.color
        calendar[date.isoformat()] = {"pnl": pnl_amount, "color": color.value}

    return calendar


async def get_system_status(session: AsyncSession) -> SystemStatus:
    stmt = select(SystemStatus).limit(1)
    status = (await session.execute(stmt)).scalar_one_or_none()
    if status is None:
        status = SystemStatus(running=False)
        session.add(status)
        await session.commit()
        await session.refresh(status)
    return status


async def set_system_status(
    session: AsyncSession,
    *,
    running: bool,
    started_at: dt.datetime | None = None,
    stopped_at: dt.datetime | None = None,
) -> SystemStatus:
    status = await get_system_status(session)
    status.running = running
    status.started_at = started_at
    status.stopped_at = stopped_at
    status.updated_at = dt.datetime.now(dt.timezone.utc)
    await session.flush()
    return status

