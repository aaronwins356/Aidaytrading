"""Prometheus metrics collectors and helpers."""
from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    PlatformCollector,
    ProcessCollector,
    generate_latest,
)

REGISTRY: CollectorRegistry
HTTP_REQUESTS_TOTAL: Counter
HTTP_REQUEST_DURATION: Histogram
PUSH_EVENTS_TOTAL: Counter
AUTH_LOGINS_TOTAL: Counter
WS_CLIENTS_GAUGE: Gauge


def _initialise_registry() -> None:
    global REGISTRY, HTTP_REQUESTS_TOTAL, HTTP_REQUEST_DURATION
    global PUSH_EVENTS_TOTAL, AUTH_LOGINS_TOTAL, WS_CLIENTS_GAUGE

    registry = CollectorRegistry(auto_describe=True)
    ProcessCollector(registry=registry)  # expose process CPU/memory stats
    PlatformCollector(registry=registry)  # platform/runtime metadata

    HTTP_REQUESTS_TOTAL = Counter(
        "http_requests_total",
        "Count of HTTP requests received",
        labelnames=("path", "method", "status"),
        registry=registry,
    )

    HTTP_REQUEST_DURATION = Histogram(
        "http_request_duration_seconds",
        "Histogram of request latency",
        labelnames=("path", "method"),
        registry=registry,
    )

    PUSH_EVENTS_TOTAL = Counter(
        "push_events_total",
        "Outcome of push notification dispatches",
        labelnames=("type", "outcome"),
        registry=registry,
    )

    AUTH_LOGINS_TOTAL = Counter(
        "auth_logins_total",
        "Authentication results",
        labelnames=("outcome",),
        registry=registry,
    )

    WS_CLIENTS_GAUGE = Gauge(
        "ws_clients_gauge",
        "Active WebSocket clients by channel",
        labelnames=("channel",),
        registry=registry,
    )

    REGISTRY = registry


_initialise_registry()


def render_metrics() -> bytes:
    """Return the current metrics snapshot in Prometheus format."""

    return generate_latest(REGISTRY)


def observe_request(path: str, method: str, status: int, latency_seconds: float) -> None:
    """Record HTTP request metrics in a thread-safe manner."""

    HTTP_REQUESTS_TOTAL.labels(path=path, method=method, status=str(status)).inc()
    HTTP_REQUEST_DURATION.labels(path=path, method=method).observe(latency_seconds)


def record_push_event(event_type: str, outcome: str) -> None:
    """Increment counters for push notification outcomes."""

    PUSH_EVENTS_TOTAL.labels(type=event_type, outcome=outcome).inc()


def record_auth_login(outcome: str) -> None:
    """Increment authentication outcome counter."""

    AUTH_LOGINS_TOTAL.labels(outcome=outcome).inc()


def set_ws_client_gauge(channel: str, value: int) -> None:
    """Explicitly set the number of active websocket clients for a channel."""

    WS_CLIENTS_GAUGE.labels(channel=channel).set(value)


def adjust_ws_client_gauge(channel: str, delta: int) -> None:
    """Increment or decrement websocket gauge by delta."""

    gauge = WS_CLIENTS_GAUGE.labels(channel=channel)
    gauge.inc(delta) if delta >= 0 else gauge.dec(abs(delta))


def reset_metrics() -> None:
    """Reset collectors; intended for deterministic tests."""

    _initialise_registry()


__all__ = [
    "CONTENT_TYPE_LATEST",
    "REGISTRY",
    "render_metrics",
    "observe_request",
    "record_push_event",
    "record_auth_login",
    "set_ws_client_gauge",
    "adjust_ws_client_gauge",
    "reset_metrics",
]
