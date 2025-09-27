"""FastAPI application entrypoint."""
from __future__ import annotations

import datetime as dt
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from loguru import logger
from sqlalchemy import func, select

from app.api.v1 import api_router
from app.core import jwt
from app.core.config import get_settings
from app.core.database import get_session_factory
from app.core.dependencies import DBSession
from app.core.health import build_health_payload
from app.core.logging import RequestLoggingMiddleware, configure_logging, record_validation_error
from app.core.metrics import CONTENT_TYPE_LATEST, render_metrics
from app.models.user import User, UserRole, UserStatus

configure_logging()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    session_factory = get_session_factory()
    async with session_factory() as session:
        deleted = await jwt.cleanup_expired_tokens(session)
        if deleted:
            logger.info("expired_tokens_cleaned", count=deleted)
        await session.commit()
    yield


app = FastAPI(title="Aidaytrading Backend", version="1.0.0", lifespan=lifespan)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        content = detail
    else:
        content = {"error": {"code": "http_error", "message": str(detail)}}
    if exc.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
        if isinstance(exc.detail, str):
            request.state.error_detail = exc.detail
        elif isinstance(exc.detail, dict):
            request.state.error_detail = exc.detail.get("code", "http_error")
    headers = exc.headers if exc.headers else None
    return JSONResponse(status_code=exc.status_code, content=content, headers=headers)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    record_validation_error(request, "validation_error", exc.errors())
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "validation_error",
                "message": "Request validation failed.",
                "details": exc.errors(),
            }
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request.state.error_detail = exc.__class__.__name__
    logger.exception("Unhandled application error")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": {"code": "server_error", "message": "Internal server error."}},
    )


app.include_router(api_router)


@app.get("/status", tags=["status"], response_model=dict)
async def service_status(session: DBSession) -> dict[str, Any]:
    """Expose business-facing service status information."""

    now = dt.datetime.now(dt.timezone.utc)
    counts: dict[str, int] = {}
    for status_value in UserStatus:
        stmt = (
            select(func.count())
            .select_from(User)
            .where(User.status == status_value)
        )
        result = await session.execute(stmt)
        counts[status_value.value] = int(result.scalar_one())

    admin_stmt = (
        select(func.count())
        .select_from(User)
        .where(User.role == UserRole.ADMIN, User.status == UserStatus.ACTIVE)
    )
    admin_result = await session.execute(admin_stmt)
    active_admins = int(admin_result.scalar_one())

    return {
        "timestamp": now.isoformat(),
        "environment": settings.env,
        "version": settings.git_sha or "unknown",
        "users": {
            "pending": counts.get(UserStatus.PENDING.value, 0),
            "active": counts.get(UserStatus.ACTIVE.value, 0),
            "disabled": counts.get(UserStatus.DISABLED.value, 0),
        },
        "admins_active": active_admins,
    }


@app.get("/health", tags=["health"], response_model=dict)
async def health() -> dict[str, object]:
    """Return infrastructure-focused health telemetry."""

    payload = await build_health_payload(settings.git_sha)
    return payload


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Expose Prometheus-formatted metrics."""

    return Response(content=render_metrics(), media_type=CONTENT_TYPE_LATEST)
