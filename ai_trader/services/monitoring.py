"""Centralised monitoring utilities for runtime observability."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from threading import RLock
from typing import Deque, Iterable, List, Mapping

from ai_trader.services.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class MonitoringEvent:
    """Structured representation of a monitoring event."""

    timestamp: datetime
    event_type: str
    severity: str
    message: str
    symbol: str | None = None
    metadata: Mapping[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        if self.metadata is not None:
            payload["metadata"] = dict(self.metadata)
        return payload


class MonitoringCenter:
    """Thread-safe event buffer with JSON logging hooks."""

    def __init__(self, max_events: int = 200) -> None:
        self._events: Deque[MonitoringEvent] = deque(maxlen=max_events)
        self._lock = RLock()
        self._runtime_degraded = False
        self._degraded_reason: str | None = None

    # ------------------------------------------------------------------
    # Event management
    # ------------------------------------------------------------------
    def record_event(
        self,
        event_type: str,
        severity: str,
        message: str,
        *,
        symbol: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> MonitoringEvent:
        severity_upper = severity.upper()
        now = datetime.now(timezone.utc)
        event = MonitoringEvent(
            timestamp=now,
            event_type=event_type,
            severity=severity_upper,
            message=message,
            symbol=symbol,
            metadata=dict(metadata) if metadata is not None else None,
        )
        with self._lock:
            self._events.appendleft(event)
        LOGGER.log(
            self._severity_to_level(severity_upper),
            json.dumps(
                {
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.event_type,
                    "severity": event.severity,
                    "symbol": event.symbol,
                    "message": event.message,
                    "metadata": dict(metadata) if metadata is not None else None,
                }
            ),
        )
        return event

    def recent_events(self, limit: int | None = None) -> List[dict[str, object]]:
        with self._lock:
            events: Iterable[MonitoringEvent]
            if limit is None:
                events = list(self._events)
            else:
                events = list(self._events)[:limit]
        return [event.to_dict() for event in events]

    # ------------------------------------------------------------------
    # Runtime health state
    # ------------------------------------------------------------------
    def set_runtime_degraded(self, degraded: bool, reason: str | None = None) -> None:
        with self._lock:
            self._runtime_degraded = degraded
            self._degraded_reason = reason

    @property
    def runtime_degraded(self) -> bool:
        with self._lock:
            return self._runtime_degraded

    @property
    def degraded_reason(self) -> str | None:
        with self._lock:
            return self._degraded_reason

    def status_flags(self) -> dict[str, object | None]:
        with self._lock:
            return {
                "runtime_degraded": self._runtime_degraded,
                "reason": self._degraded_reason,
            }

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------
    def reset(self) -> None:
        with self._lock:
            self._events.clear()
            self._runtime_degraded = False
            self._degraded_reason = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _severity_to_level(severity: str) -> int:
        if severity == "CRITICAL":
            return 50
        if severity == "ERROR":
            return 40
        if severity == "WARNING":
            return 30
        if severity == "DEBUG":
            return 10
        return 20


_MONITORING_CENTER = MonitoringCenter()


def get_monitoring_center() -> MonitoringCenter:
    """Return the singleton monitoring center."""

    return _MONITORING_CENTER


__all__ = ["MonitoringCenter", "MonitoringEvent", "get_monitoring_center"]
