"""Administrative endpoints."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger
from sqlalchemy import func, select

from app.core.dependencies import DBSession, require_admin_active_user
from app.models.admin_action import AdminAction
from app.models.user import User, UserRole, UserStatus
from app.schemas import admin as admin_schema

router = APIRouter(prefix="/admin")


async def _get_user(session: DBSession, user_id: int) -> User:
    stmt = select(User).where(User.id == user_id)
    result = await session.execute(stmt)
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "user_not_found", "message": "User not found."}},
        )
    return user


async def _ensure_additional_admin(session: DBSession, exclude_user_id: int) -> None:
    stmt = (
        select(func.count())
        .select_from(User)
        .where(
            User.role == UserRole.ADMIN,
            User.status == UserStatus.ACTIVE,
            User.id != exclude_user_id,
        )
    )
    count = (await session.execute(stmt)).scalar_one()
    if count == 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": {
                    "code": "last_admin",
                    "message": "At least one active admin must remain.",
                }
            },
        )


@router.get("/pending-users", response_model=list[admin_schema.PendingUser])
async def list_pending_users(
    session: DBSession,
    current_admin: Annotated[User, Depends(require_admin_active_user)],
) -> list[admin_schema.PendingUser]:
    del current_admin  # dependency ensures privileges
    stmt = (
        select(User)
        .where(User.status == UserStatus.PENDING)
        .order_by(User.created_at.asc())
    )
    result = await session.execute(stmt)
    users = result.scalars().all()
    return [admin_schema.PendingUser.model_validate(user) for user in users]


@router.post("/approve/{user_id}", response_model=admin_schema.StatusChangeResponse)
async def approve_user(
    user_id: int,
    session: DBSession,
    current_admin: Annotated[User, Depends(require_admin_active_user)],
) -> admin_schema.StatusChangeResponse:
    try:
        user = await _get_user(session, user_id)
        if user.status == UserStatus.ACTIVE:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error": {
                        "code": "already_active",
                        "message": "User already active.",
                    }
                },
            )
        if user.status == UserStatus.DISABLED:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error": {
                        "code": "disabled_user",
                        "message": "Disabled accounts cannot be approved.",
                    }
                },
            )

        previous_status = user.status
        user.status = UserStatus.ACTIVE
        user.token_version += 1

        session.add(
            AdminAction(
                admin_id=current_admin.id,
                action="approve_user",
                target_user_id=user.id,
                details={
                    "previous_status": previous_status.value,
                    "new_status": user.status.value,
                },
            )
        )
        await session.commit()
    except Exception:
        await session.rollback()
        raise

    logger.bind(
        event="admin.action",
        admin_id=current_admin.id,
        target_user_id=user.id,
        action="approve_user",
        previous_status=previous_status.value,
        new_status=user.status.value,
    ).info("admin_user_approved")
    return admin_schema.StatusChangeResponse(
        message="User approved.",
        user_id=user_id,
        status=UserStatus.ACTIVE,
    )


@router.post("/disable/{user_id}", response_model=admin_schema.StatusChangeResponse)
async def disable_user(
    user_id: int,
    session: DBSession,
    current_admin: Annotated[User, Depends(require_admin_active_user)],
) -> admin_schema.StatusChangeResponse:
    if user_id == current_admin.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": {
                    "code": "forbidden",
                    "message": "You cannot disable your own account.",
                }
            },
        )

    try:
        user = await _get_user(session, user_id)
        if user.status == UserStatus.DISABLED:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error": {
                        "code": "already_disabled",
                        "message": "User already disabled.",
                    }
                },
            )

        if user.role == UserRole.ADMIN and user.status == UserStatus.ACTIVE:
            await _ensure_additional_admin(session, user.id)

        previous_status = user.status
        user.status = UserStatus.DISABLED
        user.token_version += 1

        session.add(
            AdminAction(
                admin_id=current_admin.id,
                action="disable_user",
                target_user_id=user.id,
                details={
                    "previous_status": previous_status.value,
                    "new_status": user.status.value,
                },
            )
        )
        await session.commit()
    except Exception:
        await session.rollback()
        raise

    logger.bind(
        event="admin.action",
        admin_id=current_admin.id,
        target_user_id=user.id,
        action="disable_user",
        previous_status=previous_status.value,
        new_status=user.status.value,
    ).info("admin_user_disabled")
    return admin_schema.StatusChangeResponse(
        message="User disabled.",
        user_id=user_id,
        status=UserStatus.DISABLED,
    )


@router.post("/role/{user_id}", response_model=admin_schema.RoleChangeResponse)
async def change_role(
    user_id: int,
    payload: admin_schema.RoleUpdateRequest,
    session: DBSession,
    current_admin: Annotated[User, Depends(require_admin_active_user)],
) -> admin_schema.RoleChangeResponse:
    if user_id == current_admin.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": {
                    "code": "forbidden",
                    "message": "You cannot change your own role.",
                }
            },
        )

    try:
        user = await _get_user(session, user_id)
        new_role = payload.role
        if user.role == new_role:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error": {
                        "code": "no_role_change",
                        "message": "User already in requested role.",
                    }
                },
            )

        if (
            user.role == UserRole.ADMIN
            and new_role != UserRole.ADMIN
            and user.status == UserStatus.ACTIVE
        ):
            await _ensure_additional_admin(session, user.id)

        previous_role = user.role
        user.role = new_role
        user.token_version += 1

        session.add(
            AdminAction(
                admin_id=current_admin.id,
                action="change_role",
                target_user_id=user.id,
                details={
                    "previous_role": previous_role.value,
                    "new_role": new_role.value,
                },
            )
        )
        await session.commit()
    except Exception:
        await session.rollback()
        raise

    logger.bind(
        event="admin.action",
        admin_id=current_admin.id,
        target_user_id=user.id,
        action="change_role",
        previous_role=previous_role.value,
        new_role=new_role.value,
    ).info("admin_user_role_changed")
    return admin_schema.RoleChangeResponse(
        message="Role updated.",
        user_id=user_id,
        previous_role=previous_role,
        new_role=new_role,
    )


@router.get("/audit-logs", response_model=list[admin_schema.AuditLogEntry])
async def list_audit_logs(
    session: DBSession,
    current_admin: Annotated[User, Depends(require_admin_active_user)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[admin_schema.AuditLogEntry]:
    del current_admin
    stmt = (
        select(AdminAction)
        .order_by(AdminAction.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    result = await session.execute(stmt)
    actions = result.scalars().all()
    return [admin_schema.AuditLogEntry.model_validate(action) for action in actions]
