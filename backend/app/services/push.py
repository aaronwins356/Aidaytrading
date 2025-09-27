"""Firebase push notification helpers."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
import anyio
from firebase_admin import App, credentials, get_app, initialize_app, messaging
from firebase_admin.exceptions import FirebaseError
from loguru import logger
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_session_factory
from app.models.device_token import DeviceToken
from app.models.user import User, UserStatus


@dataclass
class PushDispatchResult:
    requested: int
    success: int
    failed: int
    enabled: bool


_firebase_app: App | None = None
_app_lock = anyio.Lock()


def _load_credentials(raw: str) -> credentials.Certificate:
    path = Path(raw)
    if path.exists():
        return credentials.Certificate(str(path))

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - configuration error
        raise ValueError("Invalid FIREBASE_CREDENTIALS value; must be JSON or file path") from exc
    return credentials.Certificate(data)


async def _ensure_firebase_app() -> App | None:
    settings = get_settings()
    creds = settings.firebase_credentials
    if not creds:
        return None

    async with _app_lock:
        global _firebase_app
        if _firebase_app is not None:
            return _firebase_app
        try:
            _firebase_app = get_app()
        except ValueError:
            credential = _load_credentials(creds)
            _firebase_app = initialize_app(credential)
        return _firebase_app


async def _dispatch(
    session: AsyncSession,
    *,
    user_id: int,
    title: str,
    body: str,
    data: dict[str, str] | None,
) -> PushDispatchResult:
    app = await _ensure_firebase_app()
    if app is None:
        logger.info("push_disabled", user_id=user_id, reason="missing_credentials")
        return PushDispatchResult(requested=0, success=0, failed=0, enabled=False)

    stmt = (
        select(DeviceToken.token)
        .join(User, DeviceToken.user_id == User.id)
        .where(User.id == user_id, User.status == UserStatus.ACTIVE)
    )
    tokens = [row[0] for row in await session.execute(stmt)]
    if not tokens:
        logger.info("push_skipped_no_tokens", user_id=user_id)
        return PushDispatchResult(requested=0, success=0, failed=0, enabled=True)

    payload_data = {key: str(value) for key, value in (data or {}).items()}
    message = messaging.MulticastMessage(
        tokens=tokens,
        notification=messaging.Notification(title=title, body=body),
        data=payload_data if payload_data else None,
    )

    try:
        response = await asyncio.to_thread(messaging.send_multicast, message, app=app)
    except FirebaseError:  # pragma: no cover - network failures
        logger.exception("push_dispatch_failed", user_id=user_id)
        return PushDispatchResult(requested=len(tokens), success=0, failed=len(tokens), enabled=True)

    invalid_tokens: list[str] = []
    for index, resp in enumerate(response.responses):
        if not resp.success:
            code = getattr(resp.exception, "code", "unknown")
            logger.warning(
                "push_delivery_failure",
                user_id=user_id,
                token=tokens[index],
                code=code,
            )
            if code in {"registration-token-not-registered", "invalid-argument"}:
                invalid_tokens.append(tokens[index])

    if invalid_tokens:
        await session.execute(delete(DeviceToken).where(DeviceToken.token.in_(invalid_tokens)))
        await session.commit()

    logger.info(
        "push_dispatch_result",
        user_id=user_id,
        requested=len(tokens),
        success=response.success_count,
        failed=response.failure_count,
    )

    return PushDispatchResult(
        requested=len(tokens),
        success=response.success_count,
        failed=response.failure_count,
        enabled=True,
    )


async def send_push_to_user(
    user_id: int,
    title: str,
    body: str,
    data: dict[str, str] | None = None,
    *,
    session: AsyncSession | None = None,
) -> PushDispatchResult:
    """Send a push notification to all active devices for a user."""

    if session is not None:
        return await _dispatch(session, user_id=user_id, title=title, body=body, data=data or {})

    session_factory = get_session_factory()
    async with session_factory() as db_session:
        return await _dispatch(db_session, user_id=user_id, title=title, body=body, data=data or {})


__all__ = ["PushDispatchResult", "send_push_to_user"]

