"""Structured logging utilities."""
from __future__ import annotations

import sys
import time
import uuid
from collections.abc import MutableMapping

from fastapi import Request
from typing import Any
from starlette.types import ASGIApp, Receive, Scope, Send
from loguru import logger


def configure_logging() -> None:
    """Configure loguru to emit JSON-formatted, single-line logs."""

    logger.remove()
    logger.add(sys.stdout, level="INFO", serialize=True, enqueue=True, backtrace=False, diagnose=False)


class RequestLoggingMiddleware:
    """ASGI middleware that records structured request metrics."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid.uuid4())
        method = scope.get("method")
        path = scope.get("path")
        start = time.perf_counter()
        log = logger.bind(request_id=request_id, path=path, method=method)

        async def send_wrapper(message: MutableMapping[str, Any]) -> None:
            if message["type"] == "http.response.start":
                status_code = message.get("status")
                elapsed = (time.perf_counter() - start) * 1000
                log.bind(status_code=status_code, latency_ms=round(elapsed, 2)).info("request_completed")
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            elapsed = (time.perf_counter() - start) * 1000
            log.bind(latency_ms=round(elapsed, 2)).exception("request_failed")
            raise


def mask_email(email: str) -> str:
    """Return a partially masked email safe for logging."""

    if "@" not in email:
        return email
    name, domain = email.split("@", 1)
    if len(name) <= 2:
        masked = name[0] + "*" * max(0, len(name) - 1)
    else:
        masked = name[0] + "*" * (len(name) - 2) + name[-1]
    return f"{masked}@{domain}"


def record_validation_error(request: Request, error: str, details: Any | None = None) -> None:
    """Log validation issues without exposing PII."""

    logger.bind(path=request.url.path, method=request.method, details=details).warning(error)
