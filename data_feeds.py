"""Data feed and persistence layer for the trading dashboard."""
from __future__ import annotations

import asyncio
import json
import logging
import random
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import requests
import websockets
from pandas import DataFrame

logger = logging.getLogger(__name__)
DATABASE_PATH = Path("runtime/trading_dashboard.db")


@dataclass(slots=True)
class TradeRecord:
    """Model object representing a trade captured in the SQLite store."""

    trade_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    pnl: float
    opened_at: float
    closed_at: Optional[float]
    status: str
    reason: str
    equity_pct: float


class SQLiteManager:
    """Simple SQLite abstraction dedicated to dashboard needs."""

    def __init__(self, db_path: Path = DATABASE_PATH) -> None:
        self.db_path = db_path
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    pnl REAL DEFAULT 0,
                    opened_at REAL NOT NULL,
                    closed_at REAL,
                    status TEXT NOT NULL,
                    reason TEXT,
                    equity_pct REAL DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_states (
                    name TEXT PRIMARY KEY,
                    signal TEXT NOT NULL,
                    indicator_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS telemetry (
                    metric TEXT PRIMARY KEY,
                    value REAL,
                    metadata TEXT,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    def fetch_trades(self, *, status: Optional[str] = None) -> DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM trades"
            params: tuple[Any, ...] = ()
            if status:
                query += " WHERE status = ?"
                params = (status,)
            return pd.read_sql_query(query, conn, params=params)

    def upsert_trade(self, trade: TradeRecord) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO trades (trade_id, symbol, side, quantity, price, pnl, opened_at, closed_at, status, reason, equity_pct)
                VALUES (:trade_id, :symbol, :side, :quantity, :price, :pnl, :opened_at, :closed_at, :status, :reason, :equity_pct)
                ON CONFLICT(trade_id) DO UPDATE SET
                    quantity=excluded.quantity,
                    price=excluded.price,
                    pnl=excluded.pnl,
                    closed_at=excluded.closed_at,
                    status=excluded.status,
                    reason=excluded.reason,
                    equity_pct=excluded.equity_pct
                """,
                trade.__dict__,
            )
            conn.commit()

    def get_strategy_states(self) -> Dict[str, Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("SELECT * FROM strategy_states", conn)
        result: Dict[str, Dict[str, Any]] = {}
        for _, row in df.iterrows():
            indicators = json.loads(row["indicator_json"]) if row["indicator_json"] else {}
            result[row["name"]] = {
                "signal": row["signal"],
                "indicators": indicators,
                "updated_at": row["updated_at"],
            }
        return result

    def upsert_strategy_state(self, name: str, signal: str, indicators: Dict[str, Any]) -> None:
        payload = {
            "name": name,
            "signal": signal,
            "indicator_json": json.dumps(indicators),
            "updated_at": time.time(),
        }
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO strategy_states(name, signal, indicator_json, updated_at)
                VALUES (:name, :signal, :indicator_json, :updated_at)
                ON CONFLICT(name) DO UPDATE SET
                    signal=excluded.signal,
                    indicator_json=excluded.indicator_json,
                    updated_at=excluded.updated_at
                """,
                payload,
            )
            conn.commit()

    def log_event(self, level: str, message: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO logs(level, message, created_at) VALUES (?, ?, ?)",
                (level.upper(), message, time.time()),
            )
            conn.commit()

    def fetch_logs(self, *, level: Optional[str] = None, limit: int = 200) -> DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM logs"
            params: tuple[Any, ...] = ()
            if level:
                query += " WHERE level = ?"
                params = (level.upper(),)
            query += " ORDER BY created_at DESC LIMIT ?"
            params += (limit,)
            return pd.read_sql_query(query, conn, params=params)

    def persist_setting(self, key: str, value: Any) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO settings(key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (key, json.dumps(value), time.time()),
            )
            conn.commit()

    def fetch_settings(self) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("SELECT key, value FROM settings", conn)
        return {row["key"]: json.loads(row["value"]) for _, row in df.iterrows()}

    def fetch_telemetry(self) -> Dict[str, Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("SELECT * FROM telemetry", conn)
        output: Dict[str, Dict[str, Any]] = {}
        for _, row in df.iterrows():
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            output[row["metric"]] = {
                "value": row["value"],
                "metadata": metadata,
                "updated_at": row["updated_at"],
            }
        return output

    def upsert_telemetry(self, metric: str, value: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "metric": metric,
            "value": value,
            "metadata": json.dumps(metadata or {}),
            "updated_at": time.time(),
        }
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO telemetry(metric, value, metadata, updated_at)
                VALUES (:metric, :value, :metadata, :updated_at)
                ON CONFLICT(metric) DO UPDATE SET value=excluded.value, metadata=excluded.metadata, updated_at=excluded.updated_at
                """,
                payload,
            )
            conn.commit()

    def clear_logs(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM logs")
            conn.commit()


class KrakenMarketFeed:
    """Handles real-time Kraken data via WebSocket with graceful fallbacks."""

    def __init__(self, pairs: Iterable[str]) -> None:
        self.pairs = list(pairs)
        self.latest_messages: dict[str, dict[str, Any]] = {}
        self._ws_url = "wss://ws.kraken.com/v2"
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task: Optional[asyncio.Task[None]] = None

    async def _subscriber(self) -> None:
        try:
            async with websockets.connect(self._ws_url, ping_interval=30) as websocket:
                subscribe_message = {
                    "method": "subscribe",
                    "params": {
                        "channel": "ohlc/1",
                        "symbol": [f"{pair}" for pair in self.pairs],
                    },
                }
                await websocket.send(json.dumps(subscribe_message))
                async for message in websocket:
                    try:
                        payload = json.loads(message)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict) and payload.get("channel") == "ohlc":
                        symbol = payload.get("symbol", "")
                        self.latest_messages[symbol] = payload
        except Exception as exc:  # pragma: no cover - network guard
            logger.warning("Kraken WebSocket connection failed: %s", exc)

    def start(self) -> None:
        """Start the WebSocket feed in the current or background event loop."""

        if self._task and not self._task.done():
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            self._loop = loop

            import threading as _threading

            def runner() -> None:
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._subscriber())

            thread = _threading.Thread(target=runner, daemon=True)
            thread.start()
            return

        self._loop = loop
        self._task = loop.create_task(self._subscriber())

    def get_latest(self, symbol: str) -> dict[str, Any] | None:
        return self.latest_messages.get(symbol)

    def get_candles(self, symbol: str, limit: int = 120) -> DataFrame:
        latest = self.get_latest(symbol)
        if latest and "data" in latest:
            rows = latest["data"][-limit:]
            df = pd.DataFrame(rows, columns=["time", "end", "open", "high", "low", "close", "vwap", "volume", "count"])
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
            return df
        return self._fallback_candles(symbol, limit)

    def _fallback_candles(self, symbol: str, limit: int) -> DataFrame:
        try:
            response = requests.get(
                "https://api.kraken.com/0/public/OHLC",
                params={"pair": symbol.replace("/", ""), "interval": 1, "since": int(time.time()) - limit * 60},
                timeout=5,
            )
            response.raise_for_status()
            data = response.json()["result"]
            key = next(iter(data.keys()))
            candles = data[key][-limit:]
            df = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
            return df
        except Exception as exc:  # pragma: no cover - network guard
            logger.error("REST fallback failed for %s: %s", symbol, exc)
            return self._synthetic_data(symbol, limit)

    def _synthetic_data(self, symbol: str, limit: int) -> DataFrame:
        base_price = 100 + random.random() * 10
        timestamps = pd.date_range(end=pd.Timestamp.utcnow(), periods=limit, freq="1min")
        prices = np.cumsum(np.random.normal(0, 0.2, size=limit)) + base_price
        highs = prices + np.random.uniform(0.1, 0.5, size=limit)
        lows = prices - np.random.uniform(0.1, 0.5, size=limit)
        opens = prices + np.random.uniform(-0.2, 0.2, size=limit)
        closes = prices
        volume = np.random.uniform(5, 15, size=limit)
        df = pd.DataFrame({
            "time": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volume,
        })
        return df


class PortfolioAnalytics:
    """Aggregates portfolio level stats from the database."""

    def __init__(self, store: SQLiteManager) -> None:
        self.store = store

    def compute_equity_curve(self) -> DataFrame:
        trades = self.store.fetch_trades()
        if trades.empty:
            timestamps = pd.date_range(end=pd.Timestamp.utcnow(), periods=30, freq="1H")
            equity = np.cumsum(np.random.normal(0, 50, size=len(timestamps))) + 100000
            return pd.DataFrame({"timestamp": timestamps, "equity": equity})
        trades = trades.sort_values("opened_at")
        trades["timestamp"] = pd.to_datetime(trades["opened_at"], unit="s")
        trades["cum_pnl"] = trades["pnl"].cumsum()
        starting_equity = 100000
        trades["equity"] = starting_equity + trades["cum_pnl"]
        return trades[["timestamp", "equity"]]

    def allocation_breakdown(self) -> Dict[str, float]:
        trades = self.store.fetch_trades()
        if trades.empty:
            return {"BTC": 0.35, "ETH": 0.25, "SOL": 0.2, "MATIC": 0.1, "Cash": 0.1}
        allocations = trades.groupby("symbol")["equity_pct"].mean().to_dict()
        total = sum(allocations.values()) or 1.0
        return {asset: value / total for asset, value in allocations.items()}

    def strategy_performance(self) -> Dict[str, float]:
        trades = self.store.fetch_trades()
        if trades.empty:
            return {"Momentum": 0.12, "Mean Reversion": 0.08, "Breakout": 0.05, "Scalping": 0.02}
        return trades.groupby("reason")["pnl"].sum().to_dict()


def generate_mock_trades(store: SQLiteManager, *, count: int = 15) -> None:
    """Populate the SQLite store with sample data for demo usage."""

    if not store.fetch_trades().empty:
        return
    symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "MATIC/USD"]
    sides = ["BUY", "SELL"]
    for idx in range(count):
        symbol = random.choice(symbols)
        side = random.choice(sides)
        quantity = round(random.uniform(0.1, 1.5), 3)
        price = round(random.uniform(50, 50000), 2)
        pnl = round(random.uniform(-250, 500), 2)
        opened = time.time() - random.randint(1000, 100000)
        closed = opened + random.randint(10, 5000)
        status = "CLOSED" if idx % 3 else "OPEN"
        reason = random.choice(["Momentum", "Mean Reversion", "Breakout"])
        equity_pct = round(random.uniform(0.01, 0.12), 3)
        trade = TradeRecord(
            trade_id=f"T{idx:04d}",
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            pnl=pnl,
            opened_at=opened,
            closed_at=closed if status == "CLOSED" else None,
            status=status,
            reason=reason,
            equity_pct=equity_pct,
        )
        store.upsert_trade(trade)

    for strategy in ["Momentum", "Mean Reversion", "Breakout", "Scalping"]:
        signal = random.choice(["Bullish", "Bearish", "Neutral"])
        indicators = {
            "RSI": round(random.uniform(30, 70), 2),
            "MACD": round(random.uniform(-2, 2), 2),
            "SMA50": round(random.uniform(80, 120), 2),
        }
        store.upsert_strategy_state(strategy, signal, indicators)

    store.upsert_telemetry("model_accuracy", 0.71, {"window": "30d"})
    store.upsert_telemetry("win_rate", 0.58, {"trades": 125})
    store.upsert_telemetry("feature_importance", 0.0, {"features": {
        "RSI": 0.22,
        "Volume": 0.18,
        "Momentum": 0.32,
        "Spread": 0.12,
        "Funding": 0.16,
    }})
    store.upsert_telemetry("bullish_probability", 0.63, None)
    store.upsert_telemetry("bearish_probability", 0.37, None)

    store.log_event("INFO", "Bootstrapped dashboard with mock telemetry.")
