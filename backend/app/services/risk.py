"""Risk configuration business logic."""
from __future__ import annotations

from typing import Any

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.risk import RiskSetting

RISK_CONSTRAINTS = {
    "max_drawdown_percent": (5.0, 90.0),
    "daily_loss_limit_percent": (1.0, 50.0),
    "risk_per_trade": (0.005, 0.1),
    "max_open_positions": (1, 10),
    "atr_stop_loss_multiplier": (0.5, 3.0),
    "atr_take_profit_multiplier": (1.0, 5.0),
}


async def get_risk_settings(session: AsyncSession) -> RiskSetting:
    result = await session.execute(select(RiskSetting))
    risk = result.scalar_one_or_none()
    if risk is None:
        risk = RiskSetting()
        session.add(risk)
        await session.commit()
        await session.refresh(risk)
    return risk


def validate_risk_payload(payload: dict[str, Any]) -> dict[str, Any]:
    validated: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in RISK_CONSTRAINTS:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown field {key}")
        min_value, max_value = RISK_CONSTRAINTS[key]
        if value < min_value or value > max_value:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"{key} must be between {min_value} and {max_value}",
            )
        validated[key] = value
    return validated


async def update_risk_settings(session: AsyncSession, payload: dict[str, Any]) -> RiskSetting:
    risk = await get_risk_settings(session)
    validated = validate_risk_payload(payload)
    for key, value in validated.items():
        setattr(risk, key, value)
    await session.commit()
    await session.refresh(risk)
    return risk
