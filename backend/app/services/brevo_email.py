"""Brevo email service interface placeholder."""
from __future__ import annotations

from loguru import logger


class BrevoEmailService:
    """Service responsible for outbound notifications."""

    async def notify_admin_of_signup(self, *, username: str, email: str) -> None:
        """Notify administrators about a new signup.

        The full integration will be implemented in Prompt 2. For now we log the event
        without leaking the raw email address.
        """

        logger.bind(service="brevo_email", username=username).info("signup_notification_queued")
