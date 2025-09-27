"""Structured logging utilities."""
from __future__ import annotations

import sys
import time
import uuid
from collections.abc import MutableMapping
from typing import Any, cast

from fastapi import Request
from loguru import logger
from starlette.types import ASGIApp, Receive, Scope, Send

from app.core.metrics import adjust_ws_client_gauge, observe_request


def configure_logging() -> None:
    """Configure loguru to emit JSON-formatted, single-line logs."""

    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        serialize=True,
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )


def _client_host(scope: Scope) -> str | None:
    client = scope.get("client")
    if client and isinstance(client, tuple) and client:
        return cast(str, client[0])
    return None


class RequestLoggingMiddleware:
    """ASGI middleware that records structured request metrics."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        scope_state = scope.setdefault("state", {})
        if scope["type"] == "websocket":
            request_id = str(uuid.uuid4())
            path = scope.get("path", "")
            channel = path or "unknown"
            adjust_ws_client_gauge(channel, 1)
            log = logger.bind(
                request_id=request_id,
                path=path,
                method="WS",
                channel=channel,
            )
            try:
                await self.app(scope, receive, send)
                log.bind(status_code=101).info("websocket_connection_opened")
            except Exception:
                log.exception("websocket_connection_error")
                raise
            finally:
                adjust_ws_client_gauge(channel, -1)
                log.info("websocket_connection_closed")
            return

        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid.uuid4())
        scope_state["request_id"] = request_id
        method = scope.get("method", "UNKNOWN")
        path = scope.get("path", "")
        route = scope.get("route")
        if route is not None:
            path_template = str(getattr(route, "path", path))
        else:
            path_template = path
        client_host = _client_host(scope)
        start = time.perf_counter()
        log = logger.bind(request_id=request_id, path=path, method=method)
        if client_host:
            scope_state["ip"] = client_host
            log = log.bind(ip=client_host)

        responded = False

        async def send_wrapper(message: MutableMapping[str, Any]) -> None:
            nonlocal responded
            if message["type"] == "http.response.start" and not responded:
                responded = True
                status_code = int(message.get("status", 500))
                elapsed_seconds = time.perf_counter() - start
                latency_ms = round(elapsed_seconds * 1000, 2)
                observe_request(path_template, method, status_code, elapsed_seconds)
                log_context: dict[str, Any] = {
                    "status_code": status_code,
                    "latency_ms": latency_ms,
                }
                if user_id := scope_state.get("user_id"):
                    log_context["user_id"] = user_id
                if status_code >= 500:
                    error_detail = scope_state.get("error_detail") or scope_state.get("error")
                    if error_detail:
                        log_context["error"] = error_detail
                log.bind(**log_context).info("request_completed")
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            elapsed_seconds = time.perf_counter() - start
            latency_ms = round(elapsed_seconds * 1000, 2)
            scope_state["error_detail"] = exc.__class__.__name__
            log.bind(
                latency_ms=latency_ms,
                error=exc.__class__.__name__,
            ).exception("request_failed")
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

    context: dict[str, Any] = {
        "path": request.url.path,
        "method": request.method,
        "details": details,
    }
    request_id = getattr(request.state, "request_id", None)
    if request_id:
        context["request_id"] = request_id
    logger.bind(**context).warning(error)
