import threading

from desk.services.telemetry import TelemetryClient


def test_telemetry_flush_and_close_publishes_events():
    published: list[dict] = []

    def publisher(payload: dict) -> None:
        published.append(payload)

    client = TelemetryClient(endpoint=None, publisher=publisher, flush_interval=0.05)
    client.record_equity(123.0)
    client.record_latency("loop", 0.5)
    client.flush(timeout=1.0)
    client.close()

    assert any(event.get("event_type") == "equity_snapshot" for event in published)
    assert any(event.get("event_type") == "latency" for event in published)


def test_telemetry_flush_respects_timeout():
    published: list[dict] = []
    latch = threading.Event()

    def blocking_publisher(payload: dict) -> None:
        if not latch.is_set():
            raise RuntimeError("downstream unavailable")
        published.append(payload)

    client = TelemetryClient(endpoint=None, publisher=blocking_publisher, flush_interval=0.05, max_backoff=0.1)
    client.record_equity(42.0)
    client.flush(timeout=0.1)
    assert len(published) == 0
    latch.set()
    client.flush(timeout=1.0)
    client.close()
    assert len(published) == 1
