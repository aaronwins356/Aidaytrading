"""FastAPI application entry point."""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import APIRouter, FastAPI
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
    api_v1_router = APIRouter(prefix="/api/v1")

    investor_router = APIRouter(prefix="/investor", tags=["investor"])
    investor_router.include_router(equity.router)
    investor_router.include_router(trades.router)
    investor_router.include_router(devices.router)

    admin_router = APIRouter(prefix="/admin", tags=["admin"])
    admin_router.include_router(users.admin_router)
    admin_router.include_router(risk.router)
    admin_router.include_router(bot.router)
    admin_router.include_router(monitoring.router)

    api_v1_router.include_router(auth.router)
    api_v1_router.include_router(users.router)
    api_v1_router.include_router(investor_router)
    api_v1_router.include_router(admin_router)

    app.include_router(api_v1_router)
    app.include_router(websocket.router)
    return app


app = create_app()
