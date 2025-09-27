"""Health check helpers and scheduler telemetry."""
from __future__ import annotations

import asyncio
import datetime as dt
from dataclasses import dataclass
from typing import Any

from loguru import logger

from app.core.database import check_connection

_PROCESS_STARTED_AT = dt.datetime.now(dt.timezone.utc)
_SCHEDULER_LOCK = asyncio.Lock()


@dataclass(slots=True)
class _SchedulerProbe:
    name: str
    last_tick: dt.datetime | None = None

    def snapshot(self) -> dict[str, Any]:
        if self.last_tick is None:
            return {"last_tick": None, "lag_seconds": None}
        now = dt.datetime.now(dt.timezone.utc)
        age = max((now - self.last_tick).total_seconds(), 0.0)
        return {
            "last_tick": self.last_tick.isoformat(),
            "lag_seconds": round(age, 2),
        }


_SCHEDULER_PROBES: dict[str, _SchedulerProbe] = {
    "equity_heartbeat": _SchedulerProbe("equity_heartbeat"),
    "daily_rollup": _SchedulerProbe("daily_rollup"),
}


async def record_scheduler_tick(name: str, *, timestamp: dt.datetime | None = None) -> None:
    """Capture the last time a scheduler job completed."""

    probe = _SCHEDULER_PROBES.get(name)
    if probe is None:
        probe = _SchedulerProbe(name)
        _SCHEDULER_PROBES[name] = probe
    async with _SCHEDULER_LOCK:
        probe.last_tick = timestamp or dt.datetime.now(dt.timezone.utc)
        logger.bind(event="scheduler.tick", job=name).info("scheduler_tick_recorded")


def get_uptime_seconds() -> float:
    """Return the service uptime in seconds."""

    now = dt.datetime.now(dt.timezone.utc)
    return round((now - _PROCESS_STARTED_AT).total_seconds(), 2)


async def database_health(timeout_seconds: float = 2.0) -> dict[str, str]:
    """Run a lightweight query to confirm the database is reachable."""

    try:
        await asyncio.wait_for(check_connection(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        reason = f"Connection test exceeded {timeout_seconds} seconds"
        logger.warning("database_health_timeout", timeout=timeout_seconds)
        return {"state": "degraded", "reason": reason}
    except Exception as exc:
        logger.exception("database_health_failed")
        return {"state": "down", "reason": str(exc)}
    return {"state": "ok"}


async def scheduler_snapshot() -> dict[str, dict[str, Any]]:
    """Return last-tick information for known scheduler jobs."""

    async with _SCHEDULER_LOCK:
        return {name: probe.snapshot() for name, probe in _SCHEDULER_PROBES.items()}


async def build_health_payload(version: str | None) -> dict[str, Any]:
    """Compose the JSON payload for the /health endpoint."""

    db = await database_health()
    schedulers = await scheduler_snapshot()
    return {
        "uptime_seconds": get_uptime_seconds(),
        "db_status": db,
        "scheduler_status": schedulers,
        "version": version or "unknown",
    }


__all__ = [
    "build_health_payload",
    "database_health",
    "get_uptime_seconds",
    "record_scheduler_tick",
    "scheduler_snapshot",
]
