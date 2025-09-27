"""Database session and initialization utilities."""
from __future__ import annotations

from typing import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from .config import get_settings


class Base(DeclarativeBase):
    """Declarative base for SQLAlchemy models."""

    pass


def _create_engine() -> AsyncEngine:
    """Create the async database engine."""

    settings = get_settings()
    return create_async_engine(settings.database_url, echo=settings.debug, future=True)


engine: AsyncEngine = _create_engine()
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncIterator[AsyncSession]:
    """Provide an async database session."""

    async with AsyncSessionLocal() as session:
        yield session
