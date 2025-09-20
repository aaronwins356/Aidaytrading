import ccxt
import sqlite3
import os
import time
import json
import ast

class BrokerCCXT:
    def __init__(self, mode="paper", exchange_name="kraken",
                 api_key="", api_secret="", starting_balance=1000,
                 db_path="balances.db"):
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
        self.db_path = db_path
        self._init_db()

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

    # ----------------------------
    # Database persistence
    # ----------------------------
    def _init_db(self):
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

    def _save_balance(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM balances")
        c.execute("INSERT INTO balances (usd, positions) VALUES (?, ?)",
                  (self.balance_data["USD"], json.dumps(self.balance_data["positions"])))
        conn.commit()
        conn.close()

    def _load_balance(self):
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
    def fetch_ohlcv(self, symbol, timeframe="1m", limit=50):
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    def fetch_price(self, symbol):
        ticker = self.exchange.fetch_ticker(symbol)
        return ticker["last"]

    # ----------------------------
    # Balance Management
    # ----------------------------
    def balance(self):
        if self.mode == "paper":
            return self.balance_data
        else:
            return self.exchange.fetch_balance()

    def update_balance(self, pnl, qty, side, price, symbol):
        if self.mode != "paper":
            return

        base, quote = symbol.split("/")
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
    def market_order(self, symbol, side, qty):
        """Unified market order handler for paper + live mode"""
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
            return trade
        else:
            order = self.exchange.create_order(symbol, "market", side, qty)
            return order


