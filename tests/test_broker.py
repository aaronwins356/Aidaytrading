from __future__ import annotations

import types
from typing import Any, Dict

import pytest

from desk.services import broker as broker_module


class DummyExchange:
    def __init__(self) -> None:
        self.ticker_price = 100.0
        self.balance_payload: Dict[str, Any] = {"total": {"USD": 500.0, "BTC": 0.01}}
        self.orders: list[Dict[str, Any]] = []
        self.token_calls = 0
        self.markets: Dict[str, Dict[str, Any]] = {
            "BTC/USD": {
                "symbol": "BTC/USD",
                "info": {"wsname": "XBT/USD", "altname": "XXBTZUSD"},
                "base": "BTC",
                "quote": "USD",
                "limits": {"amount": {"min": 0.0001}},
                "precision": {"amount": 4},
            }
        }
        self.markets_by_id: Dict[str, Dict[str, Any]] = {
            "XXBTZUSD": {"symbol": "BTC/USD"}
        }
        self.closed = False

    def load_markets(self):  # pragma: no cover - noop
        return None

    def market(self, symbol: str) -> Dict[str, Any]:
        return self.markets[symbol]

    def fetch_ohlcv(self, symbol, timeframe="1m", limit=50, since=None):
        return [[1, 100, 101, 99, 100, 5]]

    def fetch_ticker(self, symbol):
        return {"last": self.ticker_price}

    def privatePostGetWebSocketsToken(self):
        self.token_calls += 1
        return {"token": "ABC"}

    def fetch_balance(self):
        return self.balance_payload

    def close(self):
        self.closed = True


class DummyWebSocket:
    def __init__(self, *args, **kwargs) -> None:
        self.started = False
        self.stopped = False
        self.submissions: list[Dict[str, Any]] = []
        self.balances = {"USD": 750.0, "BTC": 0.05}

    def start(self) -> None:
        self.started = True

    def submit_order(self, pair: str, **params):
        self.submissions.append({"pair": pair, **params})
        return types.SimpleNamespace(
            status="ok",
            txid="TX123",
            client_order_id=params.get("client_order_id", "cid"),
        )

    def cancel_order(self, **params):
        return params

    def latest_balances(self):
        return dict(self.balances)

    def stop(self) -> None:
        self.stopped = True


def _patch_dependencies(monkeypatch, exchange: DummyExchange, websocket: DummyWebSocket) -> None:
    monkeypatch.setattr(broker_module, "ccxt", types.SimpleNamespace(kraken=lambda _: exchange))

    def factory(**kwargs):
        return websocket

    monkeypatch.setattr(broker_module, "KrakenWebSocketClient", factory)


def test_market_order_routes_via_websocket(monkeypatch):
    exchange = DummyExchange()
    websocket = DummyWebSocket()
    _patch_dependencies(monkeypatch, exchange, websocket)
    broker = broker_module.KrakenBroker(
        api_key="key",
        api_secret="secret",
        symbols=["BTC/USD"],
    )

    result = broker.market_order("BTC/USD", "buy", 2.5)

    assert websocket.submissions, "order should be routed via websocket"
    submission = websocket.submissions[-1]
    assert submission["pair"] == "XBT/USD"
    assert submission["side"] == "buy"
    assert result["status"] == "ok"


def test_account_equity_prefers_ws_balances(monkeypatch):
    exchange = DummyExchange()
    websocket = DummyWebSocket()
    websocket.balances = {"USD": 400.0, "BTC": 2.0}
    _patch_dependencies(monkeypatch, exchange, websocket)
    broker = broker_module.KrakenBroker(
        api_key="key",
        api_secret="secret",
        symbols=["BTC/USD"],
    )

    equity = broker.account_equity()

    assert equity == pytest.approx(402.0)
    assert websocket.started is True


def test_close_stops_websocket(monkeypatch):
    exchange = DummyExchange()
    websocket = DummyWebSocket()
    _patch_dependencies(monkeypatch, exchange, websocket)
    broker = broker_module.KrakenBroker(
        api_key="key",
        api_secret="secret",
        symbols=["BTC/USD"],
    )

    broker.close()

    assert websocket.stopped is True
    assert exchange.closed is True
