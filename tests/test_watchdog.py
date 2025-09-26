import asyncio

from ai_trader.services.monitoring import get_monitoring_center
from ai_trader.services.runtime_state import RuntimeStateStore
from ai_trader.services.watchdog import RuntimeWatchdog


class _DummyNotifier:
    def __init__(self) -> None:
        self.alerts: list[tuple[float, object]] = []

    async def send_watchdog_alert(self, timeout_seconds: float, last_update: object) -> None:
        self.alerts.append((timeout_seconds, last_update))


def test_watchdog_detects_stall_and_recovers() -> None:
    center = get_monitoring_center()
    center.reset()
    runtime_state = RuntimeStateStore(state_file=None)
    runtime_state.mark_runtime_update()
    notifier = _DummyNotifier()

    async def _runner() -> tuple[list[tuple[float, object]], dict[str, object | None]]:
        watchdog = RuntimeWatchdog(
            runtime_state,
            timeout_seconds=1.0,
            check_interval=0.2,
            alert_callback=notifier.send_watchdog_alert,
            event_loop=asyncio.get_running_loop(),
            monitoring_center=center,
        )
        watchdog.start()
        await asyncio.sleep(1.3)
        runtime_state.mark_runtime_update()
        await asyncio.sleep(0.4)
        watchdog.stop()
        return notifier.alerts, center.status_flags()

    alerts, status = asyncio.run(_runner())

    assert alerts, "watchdog should emit alert when runtime stalls"
    assert status["runtime_degraded"] is False
    events = center.recent_events()
    stall_events = [event for event in events if event["event_type"] == "watchdog_stall"]
    recovery_events = [event for event in events if event["event_type"] == "watchdog_recovered"]
    assert stall_events, "watchdog should log stall event"
    assert recovery_events, "watchdog should log recovery event"
