"""FastAPI application entry point."""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import auth, bot, devices, equity, monitoring, risk, trades, websocket, users
from .config import get_settings
from .database import Base, engine
from .services.notifications import notification_service
from .utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.start_time = datetime.now(timezone.utc)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    if settings.debug:
        logger.info("Application started with debug mode")
    notification_service.start()
    try:
        yield
    finally:
        await notification_service.shutdown()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[str(origin) for origin in settings.cors_origins],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.include_router(auth.router)
    app.include_router(users.router)
    app.include_router(risk.router)
    app.include_router(bot.router)
    app.include_router(trades.router)
    app.include_router(equity.router)
    app.include_router(devices.router)
    app.include_router(monitoring.router)
    app.include_router(websocket.router)
    return app


app = create_app()
