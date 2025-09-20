"""Background service responsible for maintaining a fresh OHLCV store."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:  # pragma: no cover - optional dependency guard
    import ccxt  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised in tests
    class _MissingCCXT:
        def __getattr__(self, name: str):
            raise ModuleNotFoundError(
                "ccxt is required for live data feeds but is not installed"
            )

    ccxt = _MissingCCXT()  # type: ignore

from desk.config import DESK_ROOT
from desk.data import normalize_ohlcv
from desk.services.logger import EventLogger


class CandleStore:
    """SQLite-backed persistence layer for OHLCV candles."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or DESK_ROOT / "data" / "ohlcv.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:  # pragma: no cover - defensive cleanup
                pass

    @staticmethod
    def _table_name(symbol: str) -> str:
        sanitized = symbol.replace("/", "_").replace("-", "_")
        return f"candles_{sanitized}".lower()

    def _ensure_table(self, symbol: str) -> None:
        table = self._table_name(symbol)
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    timestamp INTEGER PRIMARY KEY,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL
                )
                """
            )
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_ts ON {table} (timestamp)"
            )
            self._conn.commit()

    def append(self, symbol: str, candles: Iterable[Dict[str, float]]) -> int:
        candles_list = sorted(
            list(candles), key=lambda candle: float(candle.get("timestamp", 0.0))
        )
        if not candles_list:
            return 0
        self._ensure_table(symbol)
        table = self._table_name(symbol)
        records = [
            (
                int(float(candle["timestamp"])),
                float(candle["open"]),
                float(candle["high"]),
                float(candle["low"]),
                float(candle["close"]),
                float(candle["volume"]),
            )
            for candle in candles_list
        ]

        attempts = 0
        while True:
            attempts += 1
            try:
                with self._lock:
                    self._conn.executemany(
                        f"""
                        INSERT OR REPLACE INTO {table}
                        (timestamp, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        records,
                    )
                    self._conn.commit()
                return len(records)
            except sqlite3.OperationalError as exc:
                if "locked" in str(exc).lower() and attempts <= 5:
                    time.sleep(0.1 * attempts)
                    continue
                raise

    def latest_timestamp(self, symbol: str) -> Optional[int]:
        self._ensure_table(symbol)
        table = self._table_name(symbol)
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(f"SELECT MAX(timestamp) AS ts FROM {table}")
            row = cursor.fetchone()
        if not row:
            return None
        ts = row["ts"]
        return int(ts) if ts is not None else None

    def load(self, symbol: str, limit: int) -> List[Dict[str, float]]:
        self._ensure_table(symbol)
        table = self._table_name(symbol)
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                f"""
                SELECT timestamp, open, high, low, close, volume
                FROM {table}
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (int(limit),),
            )
            rows = cursor.fetchall()
        candles: List[Dict[str, float]] = []
        for row in reversed(rows):
            candles.append(
                {
                    "timestamp": float(row["timestamp"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
            )
        return candles

    def has_candles(self, symbol: str) -> bool:
        return self.latest_timestamp(symbol) is not None


class FeedUpdater:
    """Continuously refresh OHLCV candles for configured markets."""

    def __init__(
        self,
        *,
        exchange: str,
        symbols: Iterable[str],
        timeframe: str,
        mode: str = "paper",
        api_key: str = "",
        api_secret: str = "",
        interval_seconds: float = 60.0,
        logger: Optional[EventLogger] = None,
        store: Optional[CandleStore] = None,
        max_retries: int = 5,
        max_backoff: float = 30.0,
    ) -> None:
        self.exchange_name = exchange
        self.symbols = [str(symbol) for symbol in symbols]
        self.timeframe = timeframe
        self.mode = mode
        self.api_key = api_key
        self.api_secret = api_secret
        self.interval_seconds = max(5.0, float(interval_seconds))
        self.logger = logger
        self.store = store or CandleStore()
        self.max_retries = max(1, int(max_retries))
        self.max_backoff = max(5.0, float(max_backoff))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._exchange = self._build_exchange()

    def _build_exchange(self):  # pragma: no cover - network heavy
        exchange_cls = getattr(ccxt, self.exchange_name)
        return exchange_cls(
            {
                "apiKey": self.api_key,
                "secret": self.api_secret,
                "enableRateLimit": True,
            }
        )

    def close(self) -> None:
        self.stop()
        self.store.close()

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _log(self, level: str, symbol: str, message: str, **metadata) -> None:
        if self.logger is None:
            return
        payload = {"attempt": metadata.get("attempt"), "detail": metadata.get("detail")}
        payload = {k: v for k, v in payload.items() if v is not None}
        self.logger.log_feed_event(level=level, symbol=symbol, message=message, **payload)

    # ------------------------------------------------------------------
    # Seeding helpers
    # ------------------------------------------------------------------
    def seed_if_needed(self, *, candles: int = 1_000) -> None:
        if self.mode.lower() != "paper":
            return
        for symbol in self.symbols:
            if self.store.has_candles(symbol):
                continue
            try:
                fresh = self._fetch(symbol, limit=candles)
                if fresh:
                    inserted = self.store.append(symbol, fresh)
                    self._log(
                        "INFO",
                        symbol,
                        f"Seeded {inserted} candles for paper mode",
                    )
            except Exception as exc:  # pragma: no cover - network guard
                self._log(
                    "ERROR",
                    symbol,
                    "Failed to seed candles",
                    detail=str(exc),
                )

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------
    def _fetch(self, symbol: str, *, limit: int = 200, since: Optional[int] = None):
        backoff = 1.0
        attempt = 0
        while attempt < self.max_retries:
            attempt += 1
            try:
                raw = self._exchange.fetch_ohlcv(
                    symbol,
                    timeframe=self.timeframe,
                    limit=limit,
                    since=since,
                )
                return normalize_ohlcv(raw)
            except Exception as exc:  # pragma: no cover - network guard
                self._log(
                    "WARNING",
                    symbol,
                    "Exchange fetch failed",
                    attempt=attempt,
                    detail=str(exc),
                )
                if attempt >= self.max_retries:
                    break
                time.sleep(min(backoff, self.max_backoff))
                backoff *= 2
        raise RuntimeError(f"Failed to fetch candles for {symbol}")

    def _refresh_symbol(self, symbol: str) -> None:
        latest = self.store.latest_timestamp(symbol)
        since = int(latest * 1000) if latest is not None else None
        try:
            candles = self._fetch(symbol, limit=200 if since is None else 10, since=since)
        except Exception as exc:  # pragma: no cover - network guard
            self._log("ERROR", symbol, "Exhausted retries fetching candles", detail=str(exc))
            return

        if not candles:
            return

        filtered = []
        for candle in candles:
            ts = float(candle.get("timestamp", 0.0) or 0.0)
            if latest is None or ts > latest:
                filtered.append(candle)

        if not filtered:
            return

        inserted = self.store.append(symbol, filtered)
        if inserted:
            self._log(
                "INFO",
                symbol,
                f"Appended {inserted} new candles",
            )

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------
    def _run(self) -> None:  # pragma: no cover - threading
        while not self._stop_event.is_set():
            start = time.time()
            for symbol in self.symbols:
                try:
                    self._refresh_symbol(symbol)
                except Exception as exc:
                    self._log(
                        "ERROR",
                        symbol,
                        "Unhandled exception during refresh",
                        detail=str(exc),
                    )
            elapsed = time.time() - start
            remaining = self.interval_seconds - elapsed
            if remaining > 0:
                self._stop_event.wait(remaining)

    def start(self) -> None:  # pragma: no cover - threading
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="FeedUpdater", daemon=True)
        self._thread.start()

    def stop(self) -> None:  # pragma: no cover - threading
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.interval_seconds + 5)
        self._thread = None

