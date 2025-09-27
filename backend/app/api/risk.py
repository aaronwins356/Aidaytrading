"""Risk configuration endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.user import User
from ..schemas.risk import RiskSettings
from ..security.auth import require_admin
from ..services import risk as risk_service

router = APIRouter(prefix="/risk", tags=["risk"])


@router.get("", response_model=RiskSettings)
async def get_risk(
    _: User = Depends(require_admin), session: AsyncSession = Depends(get_db)
) -> RiskSettings:
    risk = await risk_service.get_risk_settings(session)
    return RiskSettings.model_validate(risk)


@router.patch("", response_model=RiskSettings)
async def update_risk(
    payload: RiskSettings,
    _: User = Depends(require_admin),
    session: AsyncSession = Depends(get_db),
) -> RiskSettings:
    risk = await risk_service.update_risk_settings(session, payload.model_dump())
    return RiskSettings.model_validate(risk)
