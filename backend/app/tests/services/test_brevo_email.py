from __future__ import annotations

import smtplib
from typing import Any

import pytest

from app.services.brevo_email import BrevoEmailService, EmailClient, EmailSendError


class DummySMTP:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.calls = 0
        self.login_calls: list[tuple[str, str]] = []
        self.starttls_called = False

    def __enter__(self) -> "DummySMTP":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # type: ignore[override]
        return False

    def starttls(self, context: Any) -> None:
        self.starttls_called = True

    def login(self, username: str, password: str) -> None:
        self.login_calls.append((username, password))

    def send_message(self, message: Any) -> dict[str, tuple[int, bytes]]:
        self.calls += 1
        if self.calls == 1:
            raise smtplib.SMTPServerDisconnected("transient failure")
        return {}


@pytest.mark.asyncio
async def test_email_client_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = DummySMTP()

    async def fake_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("app.services.brevo_email.smtplib.SMTP", lambda *a, **k: dummy)
    monkeypatch.setattr("app.services.brevo_email.anyio.sleep", fake_sleep)

    client = EmailClient(
        host="smtp.example.com",
        port=587,
        username="apikey",
        password="secret",
        sender_email="from@example.com",
        sender_name="Example",
    )

    await client.send_html_email("to@example.com", "Subject", "<p>Hello</p>")

    assert dummy.calls == 2
    assert dummy.starttls_called is True
    assert len(dummy.login_calls) == 2
    assert all(call == ("apikey", "secret") for call in dummy.login_calls)


@pytest.mark.asyncio
async def test_email_client_exhausts_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    class AlwaysFailSMTP(DummySMTP):
        def send_message(self, message: Any) -> dict[str, tuple[int, bytes]]:
            raise smtplib.SMTPServerDisconnected("failure")

    async def fake_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("app.services.brevo_email.smtplib.SMTP", lambda *a, **k: AlwaysFailSMTP())
    monkeypatch.setattr("app.services.brevo_email.anyio.sleep", fake_sleep)

    client = EmailClient(
        host="smtp.example.com",
        port=587,
        username="apikey",
        password="secret",
        sender_email="from@example.com",
        sender_name="Example",
    )

    with pytest.raises(EmailSendError):
        await client.send_html_email("to@example.com", "Subject", "<p>Hello</p>", retries=1)


@pytest.mark.asyncio
async def test_brevo_service_handles_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []

    class FailingClient(EmailClient):
        async def send_html_email(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
            captured.append("called")
            raise EmailSendError("boom")

    service = BrevoEmailService(client=FailingClient(
        host="smtp.example.com",
        port=587,
        username="apikey",
        password="secret",
        sender_email="from@example.com",
        sender_name="Example",
    ))

    await service.notify_admin_of_signup(user_id=1, username="user", email="user@example.com")
    assert captured == ["called"]
