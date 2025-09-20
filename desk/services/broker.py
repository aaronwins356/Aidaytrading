"""Broker facade built on top of CCXT."""

from __future__ import annotations

import ast
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional

try:  # pragma: no cover - import guard
    import ccxt  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised in tests
    class _MissingCCXT:
        def __getattr__(self, name: str):
            raise ModuleNotFoundError(
                "ccxt is required for live trading but is not installed"
            )

    ccxt = _MissingCCXT()  # type: ignore

from desk.config import DESK_ROOT


class BrokerCCXT:
    """Paper/live execution wrapper with simple balance persistence."""

    def __init__(
        self,
        mode: str = "paper",
        exchange_name: str = "kraken",
        api_key: str = "",
        api_secret: str = "",
        starting_balance: float = 1_000.0,
        db_path: str | Path | None = None,
    ) -> None:
        self.mode = mode
        self.exchange_name = exchange_name
        self.api_key = api_key
        self.api_secret = api_secret
        self.exchange = getattr(ccxt, exchange_name)({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True
        })

        self.starting_balance = starting_balance
        self.db_path = Path(db_path or DESK_ROOT / "logs" / "balances.db")
        self._init_db()
        self._latency_log: list[Dict[str, float]] = []

        if self.mode == "paper":
            bal = self._load_balance()
            if not bal:
                self.balance_data = {"USD": starting_balance, "positions": {}}
                self._save_balance()
            else:
                self.balance_data = bal
        else:
            self.balance_data = None

        print(f"[BrokerCCXT] Running in {mode.upper()} mode")

    @contextmanager
    def _measure_latency(self, operation: str):
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self._latency_log.append(
                {"operation": operation, "duration": duration, "timestamp": start}
            )

    @property
    def latency_log(self) -> list[Dict[str, float]]:
        return list(self._latency_log)

    # ----------------------------
    # Database persistence
    # ----------------------------
    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS balances (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            usd REAL,
            positions TEXT
        )
        """)
        conn.commit()
        conn.close()

    def _save_balance(self) -> None:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM balances")
        c.execute("INSERT INTO balances (usd, positions) VALUES (?, ?)",
                  (self.balance_data["USD"], json.dumps(self.balance_data["positions"])))
        conn.commit()
        conn.close()

    def _load_balance(self) -> Optional[Dict[str, object]]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT usd, positions FROM balances ORDER BY id DESC LIMIT 1")
        row = c.fetchone()
        conn.close()
        if not row:
            return None

        usd, positions = row
        if positions is None:
            return {"USD": usd, "positions": {}}

        try:
            pos_data = json.loads(positions)
        except json.JSONDecodeError:
            try:
                pos_data = ast.literal_eval(positions)
            except Exception:
                pos_data = {}
        if not isinstance(pos_data, dict):
            pos_data = {}

        return {"USD": usd, "positions": pos_data}

    # ----------------------------
    # Price + Candle Data
    # ----------------------------
    def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 50):
        with self._measure_latency("fetch_ohlcv"):
            return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    def fetch_price(self, symbol: str) -> float:
        with self._measure_latency("fetch_price"):
            ticker = self.exchange.fetch_ticker(symbol)
        return float(ticker["last"])

    # ----------------------------
    # Balance Management
    # ----------------------------
    def balance(self):
        if self.mode == "paper":
            return self.balance_data
        return self.exchange.fetch_balance()

    def update_balance(self, pnl, qty, side, price, symbol):
        if self.mode != "paper":
            return

        base, _quote = symbol.split("/")
        cost = qty * price

        if side == "buy":
            if self.balance_data["USD"] < cost:
                print(f"[BrokerCCXT] Not enough USD balance to buy {qty} {base}")
                return False
            self.balance_data["USD"] -= cost
            self.balance_data["positions"][base] = self.balance_data["positions"].get(base, 0.0) + qty

        elif side == "sell":
            if self.balance_data["positions"].get(base, 0.0) < qty:
                print(f"[BrokerCCXT] Not enough {base} to sell {qty}")
                return False
            self.balance_data["positions"][base] -= qty
            self.balance_data["USD"] += cost

        # Apply PnL adjustments
        self.balance_data["USD"] += pnl

        self._save_balance()

        if self.balance_data["USD"] < 0:
            print("[BrokerCCXT] WARNING: USD balance negative!")

        return True

    # ----------------------------
    # Orders
    # ----------------------------
    def market_order(self, symbol: str, side: str, qty: float):
        """Unified market order handler for paper + live mode."""
        price = self.fetch_price(symbol)

        if self.mode == "paper":
            trade = {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price": price,
                "timestamp": time.time(),
                "pnl": 0.0,
            }
            success = self.update_balance(pnl=0.0, qty=qty, side=side, price=price, symbol=symbol)
            if not success:
                return None
            with self._measure_latency("market_order"):
                pass
            return trade

        with self._measure_latency("market_order"):
            order = self.exchange.create_order(symbol, "market", side, qty)
        return order


