"""Tests for the Kraken client helpers."""

from __future__ import annotations

import asyncio

import pytest

from ai_trader.broker.kraken_client import KrakenClient


class DummyKrakenClient(KrakenClient):
    """Expose helpers for testing without hitting the network."""

    def __init__(self) -> None:
        super().__init__(
            api_key="", api_secret="", base_currency="USD", rest_rate_limit=0.0, paper_trading=True
        )
        # Reduce noise during tests while still validating logic.
        self._paper_balances = {"USD": 1000.0, "SOL.F": 2.0}


def test_compute_equity_normalises_dot_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DummyKrakenClient()

    def fake_safe_currency_code(currency_id: str) -> str:
        return {"SOL.F": "SOL"}.get(currency_id, currency_id)

    monkeypatch.setattr(client._exchange, "safe_currency_code", fake_safe_currency_code)

    equity, balances = asyncio.run(client.compute_equity({"SOL/USD": 25.0}))

    assert balances == {"USD": 1000.0, "SOL.F": 2.0}
    assert pytest.approx(equity) == 1050.0


def test_normalise_balance_asset_strips_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DummyKrakenClient()

    monkeypatch.setattr(client._exchange, "safe_currency_code", lambda code: code)

    assert client._normalise_balance_asset("SOL.F") == "SOL"
    assert client._normalise_balance_asset("eth.s") == "ETH"
    assert client._normalise_balance_asset(" usd ") == "USD"
    assert client._normalise_balance_asset("") == ""


def test_apply_precision_handles_fractional_steps() -> None:
    client = DummyKrakenClient()

    # Kraken can report the precision as a fractional step (e.g. 0.001). The
    # helper should trim to the nearest multiple of that step without raising.
    assert client._apply_precision(1.234567, 0.001) == pytest.approx(1.234)


def test_apply_precision_accepts_decimal_digits() -> None:
    client = DummyKrakenClient()

    # Precision values represented as strings should be treated as digit counts.
    assert client._apply_precision(0.123456789, "5") == pytest.approx(0.12345)
