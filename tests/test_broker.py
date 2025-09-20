import types
from typing import Any, Dict

from desk.services import broker as broker_module


class DummyExchange:
    def __init__(self):
        self.ticker_price = 100.0
        self.orders: list[Dict[str, Any]] = []
        self.balance_payload: Dict[str, Any] = {"total": {"USD": 500.0, "BTC": 0.01}}
        self.closed = False

    def load_markets(self):  # pragma: no cover - noop for tests
        return None

    def fetch_ohlcv(self, symbol, timeframe="1m", limit=50, since=None):
        return [[1, 100, 101, 99, 100, 5]]

    def fetch_ticker(self, symbol):
        return {"last": self.ticker_price}

    def create_order(self, symbol, order_type, side, qty):
        order = {
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "amount": qty,
            "price": self.ticker_price,
            "cost": qty * self.ticker_price,
        }
        self.orders.append(order)
        return order

    def fetch_balance(self):
        return self.balance_payload

    def close(self):
        self.closed = True


def _patch_kraken(monkeypatch, exchange):
    fake_ccxt = types.SimpleNamespace()
    fake_ccxt.kraken = lambda config: exchange
    monkeypatch.setattr(broker_module, "ccxt", fake_ccxt)


def test_market_order_routes_to_kraken(monkeypatch):
    exchange = DummyExchange()
    _patch_kraken(monkeypatch, exchange)
    broker = broker_module.KrakenBroker(api_key="key", api_secret="secret")

    order = broker.market_order("BTC/USD", "buy", 2.5)

    assert exchange.orders, "market order should reach Kraken"
    assert order is not None
    assert order["requested_qty"] == 2.5
    assert order["price"] == exchange.ticker_price
    operations = {entry["operation"] for entry in broker.latency_log}
    assert {"fetch_price", "market_order"}.issubset(operations)


def test_account_equity_uses_total_balance(monkeypatch):
    exchange = DummyExchange()
    exchange.balance_payload = {"total": {"USD": 400.0, "ETH": 2.0}}
    _patch_kraken(monkeypatch, exchange)
    broker = broker_module.KrakenBroker(api_key="key", api_secret="secret")

    equity = broker.account_equity()

    assert equity == 402.0
    assert any(entry["operation"] == "fetch_balance" for entry in broker.latency_log)


def test_close_handles_exchange_close(monkeypatch):
    exchange = DummyExchange()
    _patch_kraken(monkeypatch, exchange)
    broker = broker_module.KrakenBroker(api_key="key", api_secret="secret")

    broker.close()

    assert exchange.closed is True
