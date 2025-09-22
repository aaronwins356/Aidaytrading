"""Telemetry publishing utilities for the trading desk."""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from desk.services.pretty_logger import pretty_logger

try:  # pragma: no cover - optional dependency guard
    from urllib import request
except Exception:  # pragma: no cover - exercised in tests
    request = None  # type: ignore


Publisher = Callable[[Dict[str, Any]], None]


def _http_publisher(endpoint: str, timeout: float = 2.0) -> Publisher:
    """Return a callable that POSTs payloads to an HTTP endpoint."""

    def _publish(payload: Dict[str, Any]) -> None:
        if request is None:  # pragma: no cover - import guard
            raise RuntimeError("urllib is not available for HTTP telemetry publishing")
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=timeout) as resp:
            resp.read()

    return _publish


class InMemoryPublisher:
    """Fallback publisher that buffers telemetry locally."""

    def __init__(self) -> None:
        self.events: list[Dict[str, Any]] = []

    def __call__(self, payload: Dict[str, Any]) -> None:
        self.events.append(dict(payload))


@dataclass
class TelemetryEvent:
    event_type: str
    payload: Dict[str, Any]


class TelemetryClient:
    """Asynchronous telemetry dispatcher.

    Events are queued in memory and flushed by a lightweight worker thread so we
    can continue trading even if the downstream telemetry stack is slow or
    temporarily unavailable. Failed deliveries are retried with exponential
    back-off to avoid hot-looping on persistent outages.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        publisher: Optional[Publisher] = None,
        flush_interval: float = 1.0,
        max_backoff: float = 30.0,
    ) -> None:
        self.flush_interval = max(0.1, float(flush_interval))
        self.max_backoff = max(1.0, float(max_backoff))
        self._queue: "queue.Queue[TelemetryEvent]" = queue.Queue()
        self._publisher = publisher
        if self._publisher is None:
            if endpoint:
                self._publisher = _http_publisher(endpoint)
            else:
                self._publisher = InMemoryPublisher()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._backoff = self.flush_interval
        self._drain_lock = threading.Lock()
        self._ensure_thread()

    # ------------------------------------------------------------------
    def _ensure_thread(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def flush(self, timeout: Optional[float] = None) -> None:
        """Synchronously drain the telemetry queue.

        The background worker already handles dispatching events, but during
        shutdown we want to minimise the likelihood of losing telemetry data.
        ``flush`` processes queued events in the current thread, respecting the
        same error-handling semantics as the async worker.  If publishing fails
        the event is re-queued and the flush stops early so we can honour the
        caller's timeout and avoid tight retry loops.
        """

        deadline = None if timeout is None else time.time() + max(0.0, timeout)
        while True:
            if deadline is not None and time.time() >= deadline:
                break
            try:
                event = self._queue.get_nowait()
            except queue.Empty:
                break

            with self._drain_lock:
                try:
                    self._publisher(event.payload)
                    self._backoff = self.flush_interval
                except Exception as exc:  # pragma: no cover - resiliency path
                    pretty_logger.warning(
                        f"[Telemetry] Failed to publish {event.event_type}: {exc}"
                    )
                    self._queue.put(event)
                    break

    def close(self, *, drain: bool = True, timeout: float = 1.0) -> None:
        if drain:
            self.flush(timeout=timeout)
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                event = self._queue.get(timeout=self.flush_interval)
            except queue.Empty:
                continue

            with self._drain_lock:
                try:
                    self._publisher(event.payload)
                    self._backoff = self.flush_interval
                except Exception as exc:  # pragma: no cover - resiliency path
                    # Push back into the queue and sleep with backoff.
                    pretty_logger.warning(
                        f"[Telemetry] Failed to publish {event.event_type}: {exc}"
                    )
                    time.sleep(self._backoff)
                    self._backoff = min(self._backoff * 2, self.max_backoff)
                    self._queue.put(event)

    # ------------------------------------------------------------------
    def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        payload = dict(payload)
        payload.setdefault("event_type", event_type)
        payload.setdefault("ts", time.time())
        self._queue.put(TelemetryEvent(event_type, payload))

    def record_trade_open(self, trade: Dict[str, Any]) -> None:
        self._emit("trade_open", trade)

    def record_trade_close(self, trade: Dict[str, Any]) -> None:
        self._emit("trade_close", trade)

    def record_equity(self, equity: float) -> None:
        self._emit("equity_snapshot", {"equity": float(equity)})

    def record_latency(self, operation: str, duration: float) -> None:
        self._emit(
            "latency",
            {"operation": operation, "duration": float(duration)},
        )

    @property
    def publisher(self) -> Publisher:
        return self._publisher  # type: ignore[return-value]


__all__ = ["TelemetryClient", "TelemetryEvent", "InMemoryPublisher"]

