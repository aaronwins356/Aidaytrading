"""Background watchdog monitoring runtime state freshness."""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional

from ai_trader.services.monitoring import MonitoringCenter, get_monitoring_center
from ai_trader.services.runtime_state import RuntimeStateStore


class RuntimeWatchdog:
    """Periodically ensure the trading runtime is still updating state."""

    def __init__(
        self,
        runtime_state: RuntimeStateStore,
        *,
        timeout_seconds: float = 60.0,
        check_interval: float | None = None,
        alert_callback: Callable[[float, datetime | None], Awaitable[None]] | None = None,
        event_loop: asyncio.AbstractEventLoop | None = None,
        monitoring_center: MonitoringCenter | None = None,
    ) -> None:
        self._runtime_state = runtime_state
        self._timeout = max(1.0, float(timeout_seconds))
        self._interval = max(0.5, float(check_interval or min(self._timeout / 3.0, 5.0)))
        self._alert_callback = alert_callback
        self._loop = event_loop
        self._monitoring = monitoring_center or get_monitoring_center()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._degraded = False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="runtime-watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._timeout)
            self._thread = None

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval):
            last_update = self._runtime_state.last_update_time()
            now = datetime.now(timezone.utc)
            stale = last_update is None or (now - last_update).total_seconds() > self._timeout
            if stale and not self._degraded:
                self._degraded = True
                message = f"Bot stalled: no runtime update in {self._timeout:.0f} seconds"
                self._monitoring.record_event(
                    "watchdog_stall",
                    "WARNING",
                    message,
                    metadata={
                        "timeout_seconds": self._timeout,
                        "last_update_time": last_update.isoformat() if last_update else None,
                    },
                )
                self._monitoring.set_runtime_degraded(True, message)
                self._dispatch_alert(last_update)
            elif not stale and self._degraded:
                self._degraded = False
                self._monitoring.set_runtime_degraded(False, None)
                self._monitoring.record_event(
                    "watchdog_recovered",
                    "INFO",
                    "Runtime heartbeat restored",
                    metadata={
                        "timeout_seconds": self._timeout,
                        "last_update_time": (
                            last_update.isoformat() if last_update is not None else None
                        ),
                    },
                )

    def _dispatch_alert(self, last_update: Optional[datetime]) -> None:
        if self._alert_callback is None or self._loop is None:
            return
        coro = self._alert_callback(self._timeout, last_update)
        try:
            asyncio.run_coroutine_threadsafe(coro, self._loop)
        except RuntimeError:
            # Event loop already closed; nothing else to do.
            return


__all__ = ["RuntimeWatchdog"]
