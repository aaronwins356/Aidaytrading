"""FastAPI application entrypoint."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.api.v1 import api_router
from app.api.ws import register_websocket_routes
from app.core import jwt
from app.core.config import get_settings
from app.core.database import get_session_factory
from app.core.logging import RequestLoggingMiddleware, configure_logging, record_validation_error
from app.tasks.scheduler import shutdown_scheduler, start_scheduler

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
    start_scheduler()
    try:
        yield
    finally:
        await shutdown_scheduler()


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
    return JSONResponse(status_code=exc.status_code, content=content)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
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
    logger.exception("Unhandled application error")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": {"code": "server_error", "message": "Internal server error."}},
    )


app.include_router(api_router)
register_websocket_routes(app)


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    """Simple service health check."""

    return {"status": "ok"}
