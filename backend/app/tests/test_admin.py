from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.admin_action import AdminAction
from app.models.user import User, UserRole, UserStatus
from app.core.security import hash_password


async def _create_user(
    session: AsyncSession,
    *,
    username: str,
    email: str,
    password: str,
    role: UserRole,
    status: UserStatus,
) -> User:
    user = User(
        username=username,
        email=email,
        email_canonical=email.lower(),
        password_hash=hash_password(password),
        role=role,
        status=status,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


async def _login(client: AsyncClient, username: str, password: str) -> str:
    response = await client.post(
        "/api/v1/login",
        json={"username": username, "password": password},
    )
    assert response.status_code == 200
    return response.json()["access_token"]


@pytest.mark.asyncio
async def test_admin_endpoints_require_admin(client: AsyncClient, session: AsyncSession) -> None:
    viewer = await _create_user(
        session,
        username="viewer_admin_test",
        email="viewer-admin@example.com",
        password="StrongPass1",
        role=UserRole.VIEWER,
        status=UserStatus.ACTIVE,
    )
    await _create_user(
        session,
        username="pending_user",
        email="pending-admin@example.com",
        password="StrongPass1",
        role=UserRole.VIEWER,
        status=UserStatus.PENDING,
    )

    access_token = await _login(client, viewer.username, "StrongPass1")
    response = await client.get(
        "/api/v1/admin/pending-users",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"


@pytest.mark.asyncio
async def test_list_pending_users(client: AsyncClient, session: AsyncSession) -> None:
    admin = await _create_user(
        session,
        username="admin_pending",
        email="admin-pending@example.com",
        password="StrongPass1",
        role=UserRole.ADMIN,
        status=UserStatus.ACTIVE,
    )
    pending = await _create_user(
        session,
        username="pending_list",
        email="pending-list@example.com",
        password="StrongPass1",
        role=UserRole.VIEWER,
        status=UserStatus.PENDING,
    )

    access_token = await _login(client, admin.username, "StrongPass1")
    response = await client.get(
        "/api/v1/admin/pending-users",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert any(item["id"] == pending.id for item in payload)


@pytest.mark.asyncio
async def test_approve_user_creates_audit_log(client: AsyncClient, session: AsyncSession) -> None:
    admin = await _create_user(
        session,
        username="admin_approve",
        email="admin-approve@example.com",
        password="StrongPass1",
        role=UserRole.ADMIN,
        status=UserStatus.ACTIVE,
    )
    pending = await _create_user(
        session,
        username="pending_approve",
        email="pending-approve@example.com",
        password="StrongPass1",
        role=UserRole.VIEWER,
        status=UserStatus.PENDING,
    )

    access_token = await _login(client, admin.username, "StrongPass1")
    response = await client.post(
        f"/api/v1/admin/approve/{pending.id}",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "active"

    await session.refresh(pending)
    assert pending.status == UserStatus.ACTIVE
    assert pending.token_version == 1

    audit_entries = (
        await session.execute(select(AdminAction).where(AdminAction.target_user_id == pending.id))
    ).scalars().all()
    assert any(entry.action == "approve_user" for entry in audit_entries)


@pytest.mark.asyncio
async def test_disable_user_revokes_tokens(client: AsyncClient, session: AsyncSession) -> None:
    admin = await _create_user(
        session,
        username="admin_disable",
        email="admin-disable@example.com",
        password="StrongPass1",
        role=UserRole.ADMIN,
        status=UserStatus.ACTIVE,
    )
    member = await _create_user(
        session,
        username="member_disable",
        email="member-disable@example.com",
        password="StrongPass1",
        role=UserRole.VIEWER,
        status=UserStatus.ACTIVE,
    )

    member_token = await _login(client, member.username, "StrongPass1")
    admin_token = await _login(client, admin.username, "StrongPass1")

    response = await client.post(
        f"/api/v1/admin/disable/{member.id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "disabled"

    await session.refresh(member)
    assert member.status == UserStatus.DISABLED
    assert member.token_version == 1

    protected = await client.get(
        "/api/v1/me",
        headers={"Authorization": f"Bearer {member_token}"},
    )
    assert protected.status_code == 401
    assert protected.json()["error"]["code"] == "token_revoked"

    audit_entries = (
        await session.execute(select(AdminAction).where(AdminAction.target_user_id == member.id))
    ).scalars().all()
    assert any(entry.action == "disable_user" for entry in audit_entries)


@pytest.mark.asyncio
async def test_disable_self_forbidden(client: AsyncClient, session: AsyncSession) -> None:
    admin = await _create_user(
        session,
        username="admin_self_disable",
        email="admin-self-disable@example.com",
        password="StrongPass1",
        role=UserRole.ADMIN,
        status=UserStatus.ACTIVE,
    )
    admin_token = await _login(client, admin.username, "StrongPass1")

    response = await client.post(
        f"/api/v1/admin/disable/{admin.id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"


@pytest.mark.asyncio
async def test_change_role_promote_and_demote(client: AsyncClient, session: AsyncSession) -> None:
    admin = await _create_user(
        session,
        username="admin_role",
        email="admin-role@example.com",
        password="StrongPass1",
        role=UserRole.ADMIN,
        status=UserStatus.ACTIVE,
    )
    target = await _create_user(
        session,
        username="role_target",
        email="role-target@example.com",
        password="StrongPass1",
        role=UserRole.VIEWER,
        status=UserStatus.ACTIVE,
    )
    admin_token = await _login(client, admin.username, "StrongPass1")

    promote = await client.post(
        f"/api/v1/admin/role/{target.id}",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"role": "admin"},
    )
    assert promote.status_code == 200
    assert promote.json()["new_role"] == "admin"

    await session.refresh(target)
    assert target.role == UserRole.ADMIN
    assert target.token_version == 1

    demote = await client.post(
        f"/api/v1/admin/role/{target.id}",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"role": "viewer"},
    )
    assert demote.status_code == 200
    assert demote.json()["new_role"] == "viewer"

    await session.refresh(target)
    assert target.role == UserRole.VIEWER
    assert target.token_version == 2

    audit_entries = (
        await session.execute(
            select(AdminAction).where(AdminAction.target_user_id == target.id)
        )
    ).scalars().all()
    assert any(entry.action == "change_role" for entry in audit_entries)


@pytest.mark.asyncio
async def test_change_role_prevents_self_change(client: AsyncClient, session: AsyncSession) -> None:
    admin = await _create_user(
        session,
        username="admin_self_role",
        email="admin-self-role@example.com",
        password="StrongPass1",
        role=UserRole.ADMIN,
        status=UserStatus.ACTIVE,
    )
    admin_token = await _login(client, admin.username, "StrongPass1")

    response = await client.post(
        f"/api/v1/admin/role/{admin.id}",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"role": "viewer"},
    )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"


@pytest.mark.asyncio
async def test_audit_logs_endpoint(client: AsyncClient, session: AsyncSession) -> None:
    admin = await _create_user(
        session,
        username="admin_audit",
        email="admin-audit@example.com",
        password="StrongPass1",
        role=UserRole.ADMIN,
        status=UserStatus.ACTIVE,
    )
    target = await _create_user(
        session,
        username="audit_target",
        email="audit-target@example.com",
        password="StrongPass1",
        role=UserRole.VIEWER,
        status=UserStatus.ACTIVE,
    )
    admin_token = await _login(client, admin.username, "StrongPass1")

    await client.post(
        f"/api/v1/admin/disable/{target.id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    await client.post(
        f"/api/v1/admin/role/{target.id}",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"role": "viewer"},
    )

    response = await client.get(
        "/api/v1/admin/audit-logs",
        headers={"Authorization": f"Bearer {admin_token}"},
        params={"limit": 5, "offset": 0},
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) >= 2
    actions = {entry["action"] for entry in payload}
    assert {"disable_user", "change_role"}.issubset(actions)
