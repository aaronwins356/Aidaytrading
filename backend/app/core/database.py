"""Database session and engine management."""
from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings

_settings = get_settings()

_async_engine: AsyncEngine = create_async_engine(_settings.db_url, echo=False, future=True)
_async_session_factory = async_sessionmaker(_async_engine, expire_on_commit=False)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide an async SQLAlchemy session for request handling."""

    async with _async_session_factory() as session:
        yield session


async def dispose_engine() -> None:
    """Dispose the underlying engine (used in test teardown)."""

    await _async_engine.dispose()


async def check_connection() -> None:
    """Verify that the database connection is reachable."""

    async with _async_engine.connect() as connection:
        await connection.execute(text("SELECT 1"))


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the underlying async session factory."""

    return _async_session_factory
