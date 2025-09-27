from __future__ import annotations

import pytest
from fastapi import HTTPException
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api import risk as risk_api
from app.services import risk as risk_service
from app.models.user import User, UserRole, UserStatus
from app.schemas.risk import RiskSettings
from app.security.auth import hash_password


async def create_admin(session: AsyncSession) -> User:
    admin = User(
        username="riskadmin",
        email="risk@example.com",
        hashed_password=hash_password("StrongPass123"),
        status=UserStatus.ACTIVE,
        role=UserRole.ADMIN,
    )
    session.add(admin)
    await session.commit()
    await session.refresh(admin)
    return admin


async def test_risk_update_validation(
    app_client: AsyncClient, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    async with session_factory() as session:
        admin = await create_admin(session)
        current = await risk_api.get_risk(admin, session=session)
        assert current.max_drawdown_percent >= 5

        updated = await risk_api.update_risk(
            RiskSettings(
                max_drawdown_percent=25.0,
                daily_loss_limit_percent=current.daily_loss_limit_percent,
                risk_per_trade=0.02,
                max_open_positions=current.max_open_positions,
                atr_stop_loss_multiplier=current.atr_stop_loss_multiplier,
                atr_take_profit_multiplier=current.atr_take_profit_multiplier,
            ),
            admin,
            session,
        )
        assert updated.max_drawdown_percent == 25.0

        with pytest.raises(HTTPException):
            risk_service.validate_risk_payload(
                {"risk_per_trade": 0.5}
            )
