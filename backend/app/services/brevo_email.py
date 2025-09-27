"""Brevo email service integration."""
from __future__ import annotations

import random
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import formataddr
from pathlib import Path
from typing import Any

import anyio
from jinja2 import Environment, FileSystemLoader, select_autoescape
from loguru import logger

from app.core.config import get_settings
from app.core.logging import mask_email


class EmailSendError(RuntimeError):
    """Raised when an email cannot be delivered."""


class TransientEmailError(EmailSendError):
    """Raised when a transient SMTP error occurs."""


@dataclass(slots=True)
class EmailClient:
    """Asynchronous SMTP client with retry support."""

    host: str
    port: int
    username: str
    password: str
    sender_email: str
    sender_name: str
    timeout: float = 30.0

    async def send_html_email(
        self,
        to: str,
        subject: str,
        html_body: str,
        *,
        retries: int = 3,
        backoff_base: float = 1.0,
    ) -> None:
        """Send an HTML email with exponential backoff and jitter."""

        message = self._compose_message(to=to, subject=subject, html_body=html_body)
        attempt = 0
        delay = backoff_base

        while True:
            try:
                await anyio.to_thread.run_sync(self._send_message, message)
                return
            except TransientEmailError as exc:
                if attempt >= retries:
                    raise EmailSendError("Exceeded retry attempts") from exc
                jitter = random.uniform(0, delay / 2)
                await anyio.sleep(delay + jitter)
                delay *= 2
                attempt += 1
            except EmailSendError:
                raise

    def _compose_message(self, *, to: str, subject: str, html_body: str) -> EmailMessage:
        message = EmailMessage()
        message["From"] = formataddr((self.sender_name, self.sender_email))
        message["To"] = to
        message["Subject"] = subject
        message.set_content("This message requires an HTML-capable email client.")
        message.add_alternative(html_body, subtype="html")
        return message

    def _send_message(self, message: EmailMessage) -> None:
        context = ssl.create_default_context()
        try:
            with smtplib.SMTP(self.host, self.port, timeout=self.timeout) as client:
                client.starttls(context=context)
                client.login(self.username, self.password)
                refused = client.send_message(message)
                if refused:
                    raise EmailSendError(f"Recipients refused: {refused}")
        except smtplib.SMTPResponseException as exc:  # pragma: no cover - exercised via subclass
            if 400 <= exc.smtp_code < 600:
                raise TransientEmailError(str(exc)) from exc
            raise EmailSendError(str(exc)) from exc
        except (
            smtplib.SMTPServerDisconnected,
            smtplib.SMTPConnectError,
            smtplib.SMTPHeloError,
            smtplib.SMTPDataError,
            OSError,
        ) as exc:
            raise TransientEmailError(str(exc)) from exc


_template_env = Environment(
    loader=FileSystemLoader(Path(__file__).resolve().parent / "templates"),
    autoescape=select_autoescape(["html", "xml"]),
)

settings = get_settings()


def _render_template(template_name: str, context: dict[str, Any]) -> str:
    template = _template_env.get_template(template_name)
    return template.render(**context)


class BrevoEmailService:
    """Service responsible for outbound notifications."""

    def __init__(self, client: EmailClient | None = None) -> None:
        self._client = client or EmailClient(
            host=settings.brevo_smtp_server,
            port=settings.brevo_port,
            username="apikey",
            password=settings.brevo_api_key.get_secret_value(),
            sender_email=settings.brevo_sender_email,
            sender_name=settings.brevo_sender_name,
        )

    async def notify_admin_of_signup(self, *, user_id: int, username: str, email: str) -> None:
        """Notify administrators about a new signup without interrupting the caller."""

        subject = "New User Signup - Approval Needed"
        origins = settings.cors_origins or ["https://aidaytrading.local"]
        admin_url = f"{origins[0].rstrip('/')}/admin"

        html_body = _render_template(
            "signup_admin_notification.html",
            {
                "brand": settings.brevo_sender_name,
                "username": username,
                "email": email,
                "status": "Pending",
                "subject": subject,
                "admin_panel_url": admin_url,
            },
        )

        log = logger.bind(event="email_signup_admin_notice", user_id=user_id)
        try:
            await self._client.send_html_email(
                settings.admin_notification_email,
                subject,
                html_body,
            )
        except EmailSendError as exc:
            log.bind(outcome="failure", reason=str(exc), recipient=mask_email(settings.admin_notification_email)).error(
                "email_delivery_failed"
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            log.bind(outcome="failure", reason=str(exc), recipient=mask_email(settings.admin_notification_email)).exception(
                "email_delivery_failed"
            )
        else:
            log.bind(outcome="success", recipient=mask_email(settings.admin_notification_email)).info(
                "email_delivery_sent"
            )
