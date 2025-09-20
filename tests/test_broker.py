from __future__ import annotations

from typing import Any, Dict
import types

from desk.services import broker as broker_module


class DummyExchange:
    def __init__(self, *_args, **_kwargs):
        self.ticker_price = 100.0
        self.orders: list[Dict[str, Any]] = []
        self.ohlcv = [[1, 100, 101, 99, 100, 5]]

    def fetch_ohlcv(self, symbol, timeframe="1m", limit=50, since=None):
        return self.ohlcv

    def fetch_ticker(self, symbol):
        return {"last": self.ticker_price}

    def create_order(self, symbol, order_type, side, qty):
        order = {"symbol": symbol, "type": order_type, "side": side, "qty": qty}
        self.orders.append(order)
        return order

    def fetch_balance(self):
        return {"USD": 1000}


def test_paper_market_order_updates_balance(monkeypatch):
    fake_ccxt = types.SimpleNamespace()
    fake_ccxt.kraken = lambda config: DummyExchange()
    monkeypatch.setattr(broker_module, "ccxt", fake_ccxt)
    broker = broker_module.BrokerCCXT(mode="paper", exchange_name="kraken", starting_balance=1000)
    trade = broker.market_order("BTC/USD", "buy", 1)
    assert trade is not None
    assert "fee" in trade
    assert trade["qty"] <= 1
    assert broker.balance()["USD"] < 1000
    assert any(entry["operation"] == "market_order" for entry in broker.latency_log)


def test_live_market_order_records_latency(monkeypatch):
    exchange = DummyExchange()
    fake_ccxt = types.SimpleNamespace()
    fake_ccxt.kraken = lambda config: exchange
    monkeypatch.setattr(broker_module, "ccxt", fake_ccxt)
    broker = broker_module.BrokerCCXT(mode="live", exchange_name="kraken", starting_balance=1000)
    broker.market_order("BTC/USD", "buy", 1)
    assert exchange.orders
    operations = {entry["operation"] for entry in broker.latency_log}
    assert {"fetch_price", "market_order"}.issubset(operations)


def test_broker_close_handles_exchange_close(monkeypatch):
    class ExchangeWithClose(DummyExchange):
        def __init__(self):
            super().__init__()
            self.closed = False

        def close(self):
            self.closed = True

    exchange = ExchangeWithClose()
    fake_ccxt = types.SimpleNamespace()
    fake_ccxt.kraken = lambda config: exchange
    monkeypatch.setattr(broker_module, "ccxt", fake_ccxt)

    broker = broker_module.BrokerCCXT(mode="live", exchange_name="kraken", starting_balance=1000)
    broker.close()

    assert exchange.closed is True
