"""Unit tests for the readiness reporting pipeline."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from desk.services.readiness import ReadinessChecker


class _FakeExchange:
    """Provides a deterministic `market` and `fetch_time` implementation."""

    def __init__(self, market: Dict[str, Any], *, server_time: float = 1_000.0) -> None:
        self._market = market
        self._server_time = server_time

    def market(self, symbol: str) -> Dict[str, Any]:
        if symbol != "XBT/USD":
            raise KeyError(symbol)
        return self._market

    def fetch_time(self) -> float:
        return self._server_time


class _FakeBroker:
    def __init__(self, market: Dict[str, Any], *, server_time: float = 1_000.0) -> None:
        self.exchange = _FakeExchange(market, server_time=server_time)

    def resolve_symbol(self, symbol: str) -> str:
        return "XBT/USD"


@pytest.fixture
def market_metadata() -> Dict[str, Any]:
    return {
        "precision": {"amount": 5},
        "limits": {
            "amount": {"min": 0.001},
            "cost": {"min": 5.0},
        },
    }


def test_readiness_ok(tmp_path, market_metadata) -> None:
    broker = _FakeBroker(market_metadata)
    checker = ReadinessChecker(
        broker=broker,
        symbols=["BTC/USD"],
        required_env={"KRAKEN_KEY": "abc", "KRAKEN_SECRET": "def"},
        db_path=tmp_path / "ready.sqlite",
        time_provider=lambda: 1_000.5,
        server_time_fetcher=broker.exchange.fetch_time,
        max_time_drift=2.0,
    )
    report = checker.run()
    assert report.overall_status() == "ok"
    rendered = report.render_console()
    assert "Readiness Report" in rendered
    assert any(issue.check == "ccxt" for issue in report.issues)


def test_readiness_time_drift_error(tmp_path, market_metadata) -> None:
    broker = _FakeBroker(market_metadata, server_time=100.0)
    checker = ReadinessChecker(
        broker=broker,
        symbols=["BTC/USD"],
        required_env={"KRAKEN_KEY": "abc", "KRAKEN_SECRET": "def"},
        db_path=tmp_path / "ready.sqlite",
        time_provider=lambda: 200.0,
        server_time_fetcher=broker.exchange.fetch_time,
        max_time_drift=10.0,
    )
    report = checker.run()
    assert report.overall_status() == "error"
    drift_issue = next(issue for issue in report.issues if issue.check == "time_sync")
    assert "drift" in drift_issue.message


def test_readiness_missing_database(tmp_path, market_metadata) -> None:
    broker = _FakeBroker(market_metadata)
    checker = ReadinessChecker(
        broker=broker,
        symbols=["BTC/USD"],
        required_env={"KRAKEN_KEY": "abc", "KRAKEN_SECRET": "def"},
        db_path=None,
        time_provider=lambda: 1_000.0,
        server_time_fetcher=None,
    )
    report = checker.run()
    assert report.overall_status() == "warning"
    db_issue = next(issue for issue in report.issues if issue.check == "database")
    assert db_issue.level == "warning"
