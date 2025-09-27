"""Integration with Brevo email service."""
from __future__ import annotations

from typing import Any, Dict

import httpx

from ..config import get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BrevoService:
    """Client for Brevo transactional emails."""

    base_url = "https://api.brevo.com/v3/smtp/email"

    def __init__(self) -> None:
        self.settings = get_settings()
        if not self.settings.brevo_api_key:
            logger.warning("Brevo API key missing; emails will not be sent")
        self._client = httpx.AsyncClient(timeout=10.0)

    async def _post(self, payload: Dict[str, Any]) -> None:
        headers = {"api-key": self.settings.brevo_api_key, "Content-Type": "application/json"}
        if not self.settings.brevo_api_key:
            logger.info("Skipping Brevo call due to missing API key", extra={"payload": payload})
            return
        response = await self._client.post(self.base_url, json=payload, headers=headers)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error("Brevo API error", extra={"status": exc.response.status_code, "body": exc.response.text})
            raise

    async def send_signup_approval_request(self, username: str, email: str) -> None:
        payload = {
            "to": [{"email": self.settings.owner_email}],
            "subject": f"New signup pending approval: {username}",
            "htmlContent": f"<p>User {username} ({email}) is awaiting approval.</p>",
        }
        await self._post(payload)

    async def send_password_reset(self, email: str, reset_link: str) -> None:
        payload = {
            "to": [{"email": email}],
            "subject": "Password reset",
            "htmlContent": f"<p>Reset your password using <a href='{reset_link}'>this link</a>.</p>",
        }
        await self._post(payload)

    async def close(self) -> None:
        await self._client.aclose()


brevo_service = BrevoService()
