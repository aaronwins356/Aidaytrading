"""Push notification service leveraging Firebase Admin SDK."""
from __future__ import annotations

import asyncio
from typing import Iterable

import firebase_admin
from firebase_admin import credentials, messaging

from ..config import get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PushService:
    """Service for sending APNs/FCM pushes."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._initialize_app()

    def _initialize_app(self) -> None:
        if firebase_admin._apps:  # type: ignore[attr-defined]
            return
        try:
            cred = credentials.Certificate(self.settings.firebase_credentials_path)
            firebase_admin.initialize_app(cred)
        except FileNotFoundError:
            logger.warning("Firebase credentials not found; push notifications disabled")
        except ValueError:
            logger.debug("Firebase app already initialized")

    async def send_push(self, tokens: Iterable[str], title: str, body: str) -> None:
        if not tokens:
            logger.debug("No device tokens registered for push")
            return
        if not firebase_admin._apps:  # type: ignore[attr-defined]
            logger.warning("Firebase app not initialized; cannot send push")
            return
        message = messaging.MulticastMessage(
            notification=messaging.Notification(title=title, body=body),
            tokens=list(tokens),
        )
        await asyncio.to_thread(self._send_multicast, message)

    def _send_multicast(self, message: messaging.MulticastMessage) -> None:
        try:
            response = messaging.send_multicast(message, app=firebase_admin.get_app())
            logger.info(
                "Push notification dispatched",
                extra={"success": response.success_count, "failure": response.failure_count},
            )
        except Exception as exc:  # pragma: no cover - network errors
            logger.exception("Push notification failed", exc_info=exc)


push_service = PushService()
