"""Regression tests that mirror the mobile API contract."""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from importlib import reload
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def investor_client(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Provide a TestClient seeded with representative investor data."""

    db_dir = tmp_path_factory.mktemp("api_contract")
    db_path = db_dir / "backend.sqlite"
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{db_path}"

    from backend.app import config

    config.get_settings.cache_clear()

    from backend.app import database as database_module

    database_module = reload(database_module)

    from backend.app.services import notifications as notifications_module

    notifications_module = reload(notifications_module)
    monkeypatch.setattr(notifications_module.notification_service, "start", lambda: None)

    async def _noop_shutdown() -> None:
        return None

    monkeypatch.setattr(notifications_module.notification_service, "shutdown", _noop_shutdown)

    from backend.app import main as main_module

    main_module = reload(main_module)
    app = main_module.create_app()

    from backend.app.security.auth import get_current_user
    from backend.app.models.user import UserRole

    dummy_user = SimpleNamespace(id="user-1", role=UserRole.VIEWER, username="viewer")

    async def _current_user_override() -> SimpleNamespace:
        return dummy_user

    app.dependency_overrides[get_current_user] = _current_user_override

    async def seed_data() -> None:
        from backend.app.models.trade import BalanceSnapshot, BotState, EquityPoint, Trade

        async with database_module.AsyncSessionLocal() as session:
            now = datetime.now(timezone.utc)
            session.add(
                BotState(running=True, mode="paper", uptime_started_at=now - timedelta(minutes=5))
            )
            session.add(
                BalanceSnapshot(
                    balance=Decimal("10000.00"),
                    equity=Decimal("10250.00"),
                    timestamp=now - timedelta(minutes=1),
                )
            )
            session.add_all(
                [
                    EquityPoint(timestamp=now - timedelta(days=2), value=Decimal("9800.00")),
                    EquityPoint(timestamp=now - timedelta(days=1), value=Decimal("10050.50")),
                    EquityPoint(timestamp=now, value=Decimal("10250.00")),
                ]
            )
            session.add_all(
                [
                    Trade(
                        symbol="AAPL",
                        side="buy",
                        quantity=Decimal("10"),
                        price=Decimal("150.00"),
                        pnl=Decimal("50.00"),
                        balance=Decimal("10100.00"),
                        executed_at=now - timedelta(days=1, hours=2),
                        strategy="swing",
                    ),
                    Trade(
                        symbol="AAPL",
                        side="sell",
                        quantity=Decimal("5"),
                        price=Decimal("155.00"),
                        pnl=Decimal("25.00"),
                        balance=Decimal("10250.00"),
                        executed_at=now - timedelta(hours=4),
                        strategy="swing",
                    ),
                    Trade(
                        symbol="TSLA",
                        side="buy",
                        quantity=Decimal("3"),
                        price=Decimal("210.00"),
                        pnl=Decimal("-15.00"),
                        balance=Decimal("10235.00"),
                        executed_at=now - timedelta(hours=1),
                        strategy="scalp",
                    ),
                ]
            )
            await session.commit()

    with TestClient(app) as client:
        asyncio.run(seed_data())
        yield client

    os.environ.pop("DATABASE_URL", None)
    config.get_settings.cache_clear()


def test_investor_status_contract(investor_client: TestClient) -> None:
    response = investor_client.get("/api/v1/investor/status")
    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {"running", "mode", "uptime_seconds"}
    assert payload["running"] is True
    assert payload["mode"] == "paper"
    assert payload["uptime_seconds"] > 0


def test_profit_summary_contract(investor_client: TestClient) -> None:
    response = investor_client.get("/api/v1/investor/profit")
    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {
        "current_balance",
        "total_pl_amount",
        "total_pl_percent",
        "win_rate",
    }
    as_decimal = lambda value: Decimal(str(value))
    assert as_decimal(payload["current_balance"]) == Decimal("10250.00")
    assert as_decimal(payload["total_pl_amount"]) == Decimal("250.00")
    assert as_decimal(payload["total_pl_percent"]) == Decimal("2.5")
    assert payload["win_rate"] == pytest.approx(66.6666, rel=1e-3)


def test_equity_curve_contract(investor_client: TestClient) -> None:
    response = investor_client.get("/api/v1/investor/equity-curve")
    assert response.status_code == 200
    payload = response.json()
    assert all(isinstance(point, list) and len(point) == 2 for point in payload)
    timestamps = [datetime.fromisoformat(point[0]) for point in payload]
    assert timestamps == sorted(timestamps)


def test_trades_pagination_contract(investor_client: TestClient) -> None:
    response = investor_client.get(
        "/api/v1/investor/trades",
        params={"page": 1, "page_size": 2, "symbol": "AAPL"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["page"] == 1
    assert payload["page_size"] == 2
    assert payload["total"] >= 2
    assert len(payload["items"]) == 2
    trade = payload["items"][0]
    assert set(trade.keys()) == {"id", "symbol", "side", "size", "pnl", "timestamp"}
    assert trade["symbol"] == "AAPL"
    datetime.fromisoformat(trade["timestamp"])
