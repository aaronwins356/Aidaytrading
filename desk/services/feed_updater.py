"""Background service responsible for maintaining a fresh OHLCV store."""

from __future__ import annotations

import math
import random
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


class ExchangeBlockedError(RuntimeError):
    """Raised when an exchange rejects requests due to jurisdiction limits."""


class StaleDataError(RuntimeError):
    """Raised when an exchange keeps returning out-of-date candles."""


def _timeframe_to_seconds(timeframe: str) -> float:
    units = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
    try:
        value = float(timeframe[:-1])
        unit = timeframe[-1].lower()
        return value * units.get(unit, 60.0)
    except Exception:
        return 60.0


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
        fallback_exchanges: Optional[Iterable[str]] = None,
        seed_config: Optional[Dict[str, float]] = None,
    ) -> None:
        self.exchange_name = str(exchange)
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
        self.fallback_exchanges = [
            str(name)
            for name in (fallback_exchanges or [])
            if str(name or "").strip()
        ]
        self._candidate_exchanges = self._build_candidate_list()
        self._blocked_until: dict[str, float] = {}
        self._exchange_pool: dict[str, object] = {}
        self._active_exchange_name: Optional[str] = None
        self._exchange: Optional[object] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._timeframe_seconds = max(1.0, _timeframe_to_seconds(timeframe))

        default_seed: Dict[str, Any] = {
            "warm_candles": 500,
            "synthetic_candles": 200,
            "allow_synthetic": True,
            "max_stale_seconds": self._timeframe_seconds * 6,
            "blocked_retry_seconds": 900.0,
            "synthetic_drift": 0.0,
            "synthetic_volatility": 0.01,
            "synthetic_volume_mean": 10_000.0,
            "synthetic_volume_std": 2_500.0,
            "seed_length": 1_000,
            "seed_timeframe": "1m",
            "fallback_timeframes": [],
        }
        if seed_config:
            for key, value in seed_config.items():
                if value is None:
                    continue
                if key in {"warm_candles", "synthetic_candles", "seed_length"}:
                    try:
                        default_seed[key] = int(float(value))
                    except (TypeError, ValueError):
                        continue
                elif key in {"allow_synthetic"}:
                    default_seed[key] = bool(value)
                elif key in {"seed_timeframe"}:
                    default_seed[key] = str(value)
                elif key == "fallback_timeframes":
                    if isinstance(value, (list, tuple, set)):
                        default_seed[key] = [
                            str(item)
                            for item in value
                            if str(item or "").strip()
                        ]
                    elif isinstance(value, str):
                        default_seed[key] = [
                            part.strip()
                            for part in value.split(",")
                            if part.strip()
                        ]
                    else:
                        continue
                else:
                    try:
                        default_seed[key] = float(value)
                    except (TypeError, ValueError):
                        continue
        self.seed_config = default_seed
        try:
            self._seed_length = max(
                10, int(float(self.seed_config.get("seed_length", 1_000)))
            )
        except (TypeError, ValueError):
            self._seed_length = 1_000
        try:
            self._warm_candles = max(
                self._seed_length,
                10,
                int(float(self.seed_config.get("warm_candles", 500))),
            )
        except (TypeError, ValueError):
            self._warm_candles = max(self._seed_length, 500)
        try:
            self._synthetic_batch = max(
                1, int(float(self.seed_config.get("synthetic_candles", 200)))
            )
        except (TypeError, ValueError):
            self._synthetic_batch = 200
        self._blocked_retry_seconds = max(
            120.0, float(self.seed_config.get("blocked_retry_seconds", 900.0))
        )
        self._stale_threshold = float(
            self.seed_config.get("max_stale_seconds", self._timeframe_seconds * 6)
        )
        self._seed_timeframe = str(
            self.seed_config.get("seed_timeframe") or self.timeframe
        )
        self._timeframe_fallbacks = self._build_timeframe_fallbacks(self.timeframe)
        self._rng = random.Random()
        self._resolved_symbols: Dict[Tuple[str, str], str] = {}
        self._symbol_skip_until: Dict[str, float] = {}
        self._exchange = self._build_exchange()

    # ------------------------------------------------------------------
    # Exchange lifecycle helpers
    # ------------------------------------------------------------------
    def _build_candidate_list(self) -> List[str]:
        seen = set()
        candidates: List[str] = []
        for name in [self.exchange_name, *self.fallback_exchanges]:
            if not name:
                continue
            lname = name.lower()
            if lname in seen:
                continue
            seen.add(lname)
            candidates.append(name)
        if "kraken" not in seen:
            seen.add("kraken")
            candidates.append("kraken")
        return candidates

    def _build_timeframe_fallbacks(self, primary: str) -> List[str]:
        """Return a prioritized list of timeframes to try when fetching candles.

        Smaller intervals are prioritised because they can be aggregated back to
        the requested timeframe, ensuring we maintain consistent candle widths.
        User-provided fallbacks are respected ahead of automatic defaults.
        """

        cleaned_primary = str(primary or "").strip() or "1m"
        seen = {cleaned_primary}
        ordered: List[str] = [cleaned_primary]

        # Respect explicit configuration first so operators can tune behaviour
        # for niche markets or exchanges without relying on code changes.
        for candidate in self.seed_config.get("fallback_timeframes", []):
            text = str(candidate or "").strip()
            if not text:
                continue
            if text in seen:
                continue
            ordered.append(text)
            seen.add(text)

        # Kraken supports a small set of discrete granularities; ordering these
        # from highest to lowest resolution lets us degrade gracefully.
        default_order = [
            "15s",
            "30s",
            "1m",
            "3m",
            "5m",
            "15m",
            "30m",
            "1h",
            "2h",
            "4h",
            "6h",
            "12h",
            "1d",
        ]
        primary_seconds = _timeframe_to_seconds(cleaned_primary)
        fallback_defaults = sorted(
            (
                tf
                for tf in default_order
                if tf not in seen and _timeframe_to_seconds(tf) < primary_seconds
            ),
            key=lambda tf: primary_seconds - _timeframe_to_seconds(tf),
        )
        for tf in fallback_defaults:
            ordered.append(tf)
            seen.add(tf)
        return ordered

    @staticmethod
    def _error_message(exc: Exception) -> str:
        if not getattr(exc, "args", None):
            return str(exc)
        return " ".join(str(part) for part in exc.args if part)

    @staticmethod
    def _is_geo_blocked(exc: Exception) -> bool:
        message = FeedUpdater._error_message(exc).lower()
        if not message:
            return False
        if "us" not in message and "united states" not in message:
            return False
        keywords = ("restricted", "forbidden", "not available", "denied", "prohibited")
        return any(word in message for word in keywords)

    def _eligible_exchanges(self) -> List[str]:
        now = time.time()
        eligible: List[str] = []
        for name in self._candidate_exchanges:
            lname = name.lower()
            retry_at = self._blocked_until.get(lname)
            if retry_at and retry_at > now:
                continue
            if retry_at and retry_at <= now:
                self._blocked_until.pop(lname, None)
            eligible.append(name)
        return eligible

    def _mark_exchange_blocked(self, name: str, reason: str, *, symbol: str = "ALL") -> None:
        retry_at = time.time() + self._blocked_retry_seconds
        self._blocked_until[name.lower()] = retry_at
        self._log(
            "WARNING",
            symbol,
            f"Exchange {name} blocked until {retry_at:.0f}",
            detail=reason,
        )

    def _instantiate_exchange(self, name: str):  # pragma: no cover - network heavy
        exchange_cls = getattr(ccxt, name)
        exchange = exchange_cls(
            {
                "apiKey": self.api_key,
                "secret": self.api_secret,
                "enableRateLimit": True,
            }
        )
        load_markets = getattr(exchange, "load_markets", None)
        if callable(load_markets):
            try:
                load_markets()
            except Exception as exc:  # pragma: no cover - network guard
                if self._is_geo_blocked(exc):
                    close = getattr(exchange, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception:
                            pass
                    raise ExchangeBlockedError(self._error_message(exc)) from exc
                raise
        return exchange

    @staticmethod
    def _currency_aliases(code: str) -> List[str]:
        """Return a list of exchange-specific aliases for a currency code."""

        normalized = str(code or "").upper()
        aliases = {normalized}
        known = {
            "BTC": {"BTC", "XBT"},
            "XBT": {"XBT", "BTC"},
            "ETH": {"ETH", "XETH"},
            "XETH": {"ETH", "XETH"},
            "LTC": {"LTC", "XLTC"},
            "XLTC": {"LTC", "XLTC"},
            "BCH": {"BCH", "XBCH"},
            "XBCH": {"BCH", "XBCH"},
            "XRP": {"XRP", "XXRP"},
            "XXRP": {"XRP", "XXRP"},
            "ADA": {"ADA", "XADA"},
            "XADA": {"ADA", "XADA"},
            "XLM": {"XLM", "XXLM"},
            "XXLM": {"XLM", "XXLM"},
        }
        aliases.update(known.get(normalized, {normalized}))
        if normalized.startswith("X") and len(normalized) == 4:
            aliases.add(normalized[1:])
        if normalized.startswith("Z") and len(normalized) == 4:
            aliases.add(normalized[1:])
        if len(normalized) == 3:
            aliases.add(f"X{normalized}")
            aliases.add(f"Z{normalized}")
        return sorted({alias for alias in aliases if alias})

    def _resolve_symbol(self, exchange: object, symbol: str) -> str:
        """Translate user symbols into exchange-specific market symbols."""

        if exchange is None:
            return symbol
        exchange_id = str(getattr(exchange, "id", "") or self._active_exchange_name or "")
        cache_key = (exchange_id, symbol)
        cached = self._resolved_symbols.get(cache_key)
        if cached:
            return cached

        markets = getattr(exchange, "markets", None)
        if not isinstance(markets, dict) or not markets:
            self._resolved_symbols[cache_key] = symbol
            return symbol

        # Direct lookups first as they are the most common path once markets are
        # loaded by ccxt.
        direct = markets.get(symbol)
        if isinstance(direct, dict):
            resolved = str(direct.get("symbol") or symbol)
            self._resolved_symbols[cache_key] = resolved
            if resolved != symbol:
                self._log(
                    "INFO",
                    symbol,
                    f"Resolved exchange symbol to {resolved}",
                )
            return resolved

        normalized = symbol.lower()
        for market_symbol, market in markets.items():
            canonical = str(market.get("symbol") or market_symbol)
            aliases = {canonical.lower(), str(market_symbol).lower()}
            altname = market.get("altname")
            if altname:
                aliases.add(str(altname).lower())
            market_id = market.get("id")
            if market_id:
                aliases.add(str(market_id).lower())
            base = str(market.get("base") or "").upper()
            quote = str(market.get("quote") or "").upper()
            if base and quote:
                for b_alias in self._currency_aliases(base):
                    for q_alias in self._currency_aliases(quote):
                        aliases.add(f"{b_alias}/{q_alias}".lower())
            base_id = str(market.get("baseId") or "").upper()
            quote_id = str(market.get("quoteId") or "").upper()
            if base_id and quote_id:
                aliases.add(f"{base_id}/{quote_id}".lower())
            if normalized in aliases:
                self._resolved_symbols[cache_key] = canonical
                if canonical != symbol:
                    self._log(
                        "INFO",
                        symbol,
                        f"Resolved exchange symbol to {canonical}",
                    )
                return canonical

        # Fall back to ccxt's own safe_symbol helper if available.
        safe_symbol = getattr(exchange, "safe_symbol", None)
        if callable(safe_symbol):
            try:
                resolved = str(safe_symbol(symbol))
                if resolved:
                    self._resolved_symbols[cache_key] = resolved
                    if resolved != symbol:
                        self._log(
                            "INFO",
                            symbol,
                            f"Resolved exchange symbol to {resolved}",
                        )
                    return resolved
            except Exception:
                pass

        self._resolved_symbols[cache_key] = symbol
        return symbol

    def _candidate_timeframes(self, override: Optional[str] = None) -> List[str]:
        """Return ordered timeframes to try for a fetch request."""

        if override:
            cleaned = str(override).strip()
            if not cleaned:
                return list(self._timeframe_fallbacks)
            candidates = [cleaned]
            candidates.extend(
                tf for tf in self._timeframe_fallbacks if tf not in {cleaned}
            )
            return candidates
        return list(self._timeframe_fallbacks)

    def _resample_candles(
        self,
        candles: List[Dict[str, float]],
        source_timeframe: str,
        target_timeframe: str,
    ) -> List[Dict[str, float]]:
        """Aggregate higher resolution candles to the requested timeframe."""

        if not candles:
            return []
        source_seconds = max(1.0, _timeframe_to_seconds(source_timeframe))
        target_seconds = max(1.0, _timeframe_to_seconds(target_timeframe))
        if math.isclose(source_seconds, target_seconds):
            return candles
        if source_seconds > target_seconds:
            # Unable to reconstruct higher resolution candles from lower ones.
            return []

        buckets: Dict[int, List[Dict[str, float]]] = {}
        for candle in candles:
            ts = self._normalize_timestamp(float(candle.get("timestamp", 0.0)))
            bucket_start = int(ts // target_seconds * target_seconds)
            buckets.setdefault(bucket_start, []).append(candle)

        expected = max(1, int(round(target_seconds / source_seconds)))
        aggregated: List[Dict[str, float]] = []
        for bucket_start in sorted(buckets):
            bucket = sorted(
                buckets[bucket_start],
                key=lambda item: self._normalize_timestamp(
                    float(item.get("timestamp", 0.0))
                ),
            )
            if len(bucket) < expected:
                continue
            open_price = float(bucket[0].get("open", 0.0))
            high_price = max(float(item.get("high", 0.0)) for item in bucket)
            low_price = min(float(item.get("low", 0.0)) for item in bucket)
            close_price = float(bucket[-1].get("close", 0.0))
            volume = sum(float(item.get("volume", 0.0)) for item in bucket)
            aggregated.append(
                {
                    "timestamp": float(bucket_start * 1000.0),
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                }
            )
        return aggregated

    def _mark_symbol_skip(
        self, symbol: str, *, reason: str, duration: Optional[float] = None
    ) -> None:
        """Temporarily pause refreshing a misbehaving market symbol."""

        delay = duration if duration is not None else max(
            self.interval_seconds, self._timeframe_seconds
        )
        retry_at = time.time() + delay
        self._symbol_skip_until[symbol] = retry_at
        self._log(
            "WARNING",
            symbol,
            "Temporarily skipping symbol",
            detail=f"reason={reason} retry_at={retry_at:.0f}",
        )

    def _ensure_exchange(self, name: str, *, rebuild: bool = False):  # pragma: no cover - network heavy
        key = name.lower()
        existing = self._exchange_pool.get(key)
        if rebuild:
            preserved = existing
            try:
                exchange = self._instantiate_exchange(name)
            except Exception:
                if preserved is not None:
                    self._exchange_pool[key] = preserved
                raise
            if preserved is not None and preserved is not exchange:
                close = getattr(preserved, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
            self._exchange_pool[key] = exchange
            return exchange
        if existing is not None:
            return existing
        exchange = self._instantiate_exchange(name)
        self._exchange_pool[key] = exchange
        return exchange

    def _build_exchange(self):  # pragma: no cover - network heavy
        last_error: Exception | None = None
        for name in self._eligible_exchanges():
            try:
                exchange = self._ensure_exchange(name, rebuild=True)
            except ExchangeBlockedError as exc:
                self._mark_exchange_blocked(name, self._error_message(exc))
                last_error = exc
                continue
            except Exception as exc:
                self._log("ERROR", "ALL", f"Failed to initialize {name}", detail=str(exc))
                last_error = exc
                continue
            self._active_exchange_name = name
            if name != self._candidate_exchanges[0]:
                self._log("INFO", "ALL", f"Using fallback exchange {name}")
            return exchange
        if last_error is not None:
            raise last_error
        raise RuntimeError("Unable to initialise any exchange for feed updater")

    def _switch_exchange(self, symbol: str, reason: str) -> bool:  # pragma: no cover - network heavy
        for name in self._eligible_exchanges():
            if name == self._active_exchange_name:
                continue
            try:
                exchange = self._ensure_exchange(name, rebuild=True)
            except ExchangeBlockedError as exc:
                self._mark_exchange_blocked(name, self._error_message(exc), symbol=symbol)
                continue
            except Exception as exc:
                self._log("ERROR", symbol, f"Failed to switch to {name}", detail=str(exc))
                continue
            old = self._exchange
            self._exchange = exchange
            self._active_exchange_name = name
            self._log("INFO", symbol, f"Switched feed to {name}", detail=reason)
            if old is not None and old is not exchange:
                close = getattr(old, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
            return True
        if self._active_exchange_name:
            try:
                exchange = self._ensure_exchange(self._active_exchange_name, rebuild=True)
            except ExchangeBlockedError as exc:
                self._mark_exchange_blocked(
                    self._active_exchange_name,
                    self._error_message(exc),
                    symbol=symbol,
                )
                return False
            except Exception as exc:
                self._log(
                    "ERROR",
                    symbol,
                    "Failed to rebuild active exchange",
                    detail=str(exc),
                )
                return False
            self._exchange = exchange
            self._log(
                "INFO",
                symbol,
                f"Rebuilt exchange {self._active_exchange_name}",
                detail=reason,
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _log(self, level: str, symbol: str, message: str, **metadata) -> None:
        payload = {"attempt": metadata.get("attempt"), "detail": metadata.get("detail")}
        payload = {k: v for k, v in payload.items() if v is not None}
        meta_str = ""
        if payload:
            detail_parts = [f"{key}={value}" for key, value in payload.items()]
            meta_str = " | " + ", ".join(detail_parts)
        print(f"[FeedUpdater][{level.upper()}] {symbol}: {message}{meta_str}")
        if self.logger is None:
            return
        self.logger.log_feed_event(level=level, symbol=symbol, message=message, **payload)

    # ------------------------------------------------------------------
    # Seeding helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_timestamp(value: float) -> float:
        if value > 1e12:
            return value / 1000.0
        if value > 1e10:
            return value / 1000.0
        return value

    def _current_time_seconds(self, exchange: Optional[object] = None) -> float:
        """Return the best available notion of "now" in epoch seconds."""

        source = exchange or self._exchange
        milliseconds = getattr(source, "milliseconds", None)
        if callable(milliseconds):  # pragma: no branch - tiny helper
            try:
                return float(milliseconds()) / 1000.0
            except Exception:
                # Fallback to wall clock time if the exchange clock fails.
                pass
        return time.time()

    def _candles_are_stale(
        self, candles: List[Dict[str, float]], *, exchange: Optional[object] = None
    ) -> bool:
        if not candles:
            return True
        last_ts = self._normalize_timestamp(float(candles[-1].get("timestamp", 0.0)))
        now = self._current_time_seconds(exchange)
        age = now - last_ts
        return age > self._stale_threshold

    def _symbol_stale(self, latest: Optional[int]) -> bool:
        if latest is None:
            return True
        ts = self._normalize_timestamp(float(latest))
        now = self._current_time_seconds()
        return (now - ts) > self._stale_threshold

    def _default_price(self, symbol: str) -> float:
        sym = symbol.upper()
        if "BTC" in sym:
            return 30_000.0
        if "ETH" in sym:
            return 2_000.0
        if "SOL" in sym:
            return 25.0
        if "XRP" in sym:
            return 0.6
        return 100.0

    def _generate_synthetic_candles(self, symbol: str, count: int) -> List[Dict[str, float]]:
        if not self.seed_config.get("allow_synthetic", True):
            return []
        count = max(1, int(count))
        try:
            history = self.store.load(symbol, 1)
        except Exception:
            history = []
        if history:
            last_candle = history[-1]
            last_ts = self._normalize_timestamp(float(last_candle.get("timestamp", 0.0)))
            price = float(
                last_candle.get("close")
                or last_candle.get("open")
                or self._default_price(symbol)
            )
        else:
            last_ts = time.time() - self._timeframe_seconds * count
            price = self._default_price(symbol)
        candles: List[Dict[str, float]] = []
        local_rng = random.Random(self._rng.random())
        drift = float(self.seed_config.get("synthetic_drift", 0.0))
        volatility = max(0.0001, float(self.seed_config.get("synthetic_volatility", 0.01)))
        volume_mean = max(1.0, float(self.seed_config.get("synthetic_volume_mean", 10_000.0)))
        volume_std = max(1.0, float(self.seed_config.get("synthetic_volume_std", volume_mean / 4.0)))
        timestamp = last_ts
        for _ in range(count):
            timestamp += self._timeframe_seconds
            open_price = price
            move = local_rng.normalvariate(drift, volatility)
            close_price = max(0.0001, open_price * (1.0 + move))
            high = max(open_price, close_price) * (1.0 + abs(local_rng.normalvariate(0.0, volatility / 2.0)))
            low = min(open_price, close_price) * (1.0 - abs(local_rng.normalvariate(0.0, volatility / 2.0)))
            volume = max(1.0, local_rng.normalvariate(volume_mean, volume_std))
            candles.append(
                {
                    "timestamp": int(timestamp * 1000.0),
                    "open": float(open_price),
                    "high": float(high),
                    "low": float(low),
                    "close": float(close_price),
                    "volume": float(volume),
                }
            )
            price = close_price
        return candles

    def _fallback_candles(
        self,
        symbol: str,
        limit: int,
        *,
        allow_synthetic: bool = True,
    ) -> List[Dict[str, float]]:
        cached: List[Dict[str, float]] = []
        try:
            cached = self.store.load(symbol, limit)
        except Exception as exc:
            self._log(
                "ERROR",
                symbol,
                "Failed to load cached candles",
                detail=self._error_message(exc),
            )
        if cached:
            last_ts = float(cached[-1].get("timestamp", 0.0) or 0.0)
            age = time.time() - self._normalize_timestamp(last_ts)
            if not self._candles_are_stale(cached):
                self._log(
                    "INFO",
                    symbol,
                    "Using cached candles while exchange recovers",
                    detail=f"count={len(cached)}",
                )
                return cached
            self._log(
                "WARNING",
                symbol,
                "Cached candles are stale",
                detail=f"age={age:.2f}s",
            )
        if allow_synthetic and self.mode.lower() == "paper":
            synthetic = self._generate_synthetic_candles(
                symbol, min(limit, self._synthetic_batch)
            )
            if synthetic:
                self._log(
                    "WARNING",
                    symbol,
                    "Generated synthetic candles for fallback",
                    detail=f"count={len(synthetic)}",
                )
                return synthetic
        return []

    def _seed_from_kraken(self, symbol: str, limit: int) -> List[Dict[str, float]]:
        try:
            exchange = self._ensure_exchange("kraken")
        except Exception as exc:
            self._log(
                "WARNING",
                symbol,
                "Kraken unavailable for seeding",
                detail=self._error_message(exc),
            )
            return []
        try:
            raw = exchange.fetch_ohlcv(
                self._resolve_symbol(exchange, symbol),
                timeframe=self._seed_timeframe,
                limit=limit,
            )
        except Exception as exc:  # pragma: no cover - network guard
            self._log(
                "WARNING",
                symbol,
                "Kraken fetch failed for seeding",
                detail=self._error_message(exc),
            )
            return []
        candles = normalize_ohlcv(raw)
        if candles and self._candles_are_stale(candles, exchange=exchange):
            last_ts = self._normalize_timestamp(float(candles[-1].get("timestamp", 0.0)))
            age = self._current_time_seconds(exchange) - last_ts
            if age < 0:
                age = 0.0
            self._log(
                "WARNING",
                symbol,
                "Kraken returned stale seed candles",
                detail=f"age={age:.2f}s",
            )
            return []
        if candles:
            self._log(
                "INFO",
                symbol,
                "Seeded candles fetched from Kraken",
                detail=f"count={len(candles)}",
            )
        return candles[-limit:]

    def _maybe_seed_symbol(self, symbol: str, *, reason: str, count: Optional[int] = None) -> bool:
        if self.mode.lower() != "paper":
            return False
        batch = count if count is not None else self._synthetic_batch
        synthetic = self._generate_synthetic_candles(symbol, batch)
        if not synthetic:
            return False
        inserted = self.store.append(symbol, synthetic)
        if inserted:
            self._log(
                "WARNING",
                symbol,
                f"Seeded {inserted} synthetic candles ({reason})",
            )
            return True
        return False

    def seed_if_needed(self) -> None:
        for symbol in self.symbols:
            try:
                existing = self.store.load(symbol, self._seed_length)
            except Exception:
                existing = []
            latest = None
            if existing:
                latest = int(float(existing[-1].get("timestamp", 0.0)))
            has_enough = len(existing) >= self._seed_length
            if has_enough and latest is not None and not self._symbol_stale(latest):
                continue
            fresh: List[Dict[str, float]] = []
            try:
                fresh = self._seed_from_kraken(
                    symbol, max(self._warm_candles, self._seed_length)
                )
            except Exception as exc:  # pragma: no cover - defensive
                self._log(
                    "WARNING",
                    symbol,
                    "Unexpected error during Kraken seeding",
                    detail=self._error_message(exc),
                )
            if not fresh:
                try:
                    fresh = self._fetch(
                        symbol,
                        limit=max(self._warm_candles, self._seed_length),
                        since=None,
                        allow_synthetic=self.mode.lower() == "paper",
                        timeframe=self._seed_timeframe,
                    )
                except Exception as exc:
                    self._log(
                        "WARNING",
                        symbol,
                        "Failed to seed candles from exchange",
                        detail=self._error_message(exc),
                    )
                    if self.mode.lower() == "paper":
                        self._maybe_seed_symbol(
                            symbol,
                            reason="initial_seed",
                            count=max(self._seed_length, self._warm_candles),
                        )
                    continue
            if not fresh and self.mode.lower() == "paper":
                self._maybe_seed_symbol(
                    symbol,
                    reason="initial_seed_empty",
                    count=max(self._seed_length, self._warm_candles),
                )
                continue
            if fresh:
                if (
                    len(fresh) < self._seed_length
                    and self.mode.lower() == "paper"
                ):
                    deficit = self._seed_length - len(fresh)
                    synthetic_extra = self._generate_synthetic_candles(
                        symbol, deficit
                    )
                    if synthetic_extra:
                        self._log(
                            "WARNING",
                            symbol,
                            "Supplemented seed with synthetic candles",
                            detail=f"count={len(synthetic_extra)}",
                        )
                        fresh = fresh + synthetic_extra
                inserted = self.store.append(
                    symbol, fresh[-max(self._warm_candles, self._seed_length) :]
                )
                if inserted:
                    self._log("INFO", symbol, f"Seeded {inserted} candles for startup")

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------
    def _should_failover(self, exc: Exception, attempt: int) -> bool:
        message = self._error_message(exc).lower()
        if "stale market data" in message:
            return True
        transient_terms = (
            "timeout",
            "temporarily",
            "connection reset",
            "service unavailable",
            "network",
            "rate limit",
        )
        if any(term in message for term in transient_terms):
            return attempt >= 2
        return True

    def _fetch(
        self,
        symbol: str,
        *,
        limit: int = 200,
        since: Optional[int] = None,
        allow_synthetic: bool = True,
        timeframe: Optional[str] = None,
    ) -> List[Dict[str, float]]:
        if self._exchange is None:
            self._exchange = self._build_exchange()
        target_timeframe = timeframe or self.timeframe
        last_exc: Exception | None = None
        stale_details: List[str] = []

        for active_timeframe in self._candidate_timeframes(timeframe):
            attempt = 0
            backoff = 1.0
            while attempt < self.max_retries:
                attempt += 1
                try:
                    exchange_symbol = self._resolve_symbol(self._exchange, symbol)
                    raw = self._exchange.fetch_ohlcv(
                        exchange_symbol,
                        timeframe=active_timeframe,
                        limit=limit,
                        since=since,
                    )
                    candles = normalize_ohlcv(raw)
                    if not candles:
                        break
                    if self._candles_are_stale(
                        candles, exchange=self._exchange
                    ):
                        last_ts = self._normalize_timestamp(
                            float(candles[-1].get("timestamp", 0.0))
                        )
                        age = self._current_time_seconds() - last_ts
                        if age < 0:
                            age = 0.0
                        stale_details.append(f"{active_timeframe}:{age:.2f}s")
                        self._log(
                            "WARNING",
                            symbol,
                            "Received stale candles",
                            detail=f"timeframe={active_timeframe} age={age:.2f}s",
                        )
                        break
                    if active_timeframe != target_timeframe:
                        candles = self._resample_candles(
                            candles,
                            source_timeframe=active_timeframe,
                            target_timeframe=target_timeframe,
                        )
                        if not candles:
                            break
                    return candles
                except Exception as exc:  # pragma: no cover - network guard
                    last_exc = exc
                    detail = self._error_message(exc)
                    self._log(
                        "WARNING",
                        symbol,
                        "Exchange fetch failed",
                        attempt=attempt,
                        detail=detail,
                    )
                    if self._active_exchange_name and self._is_geo_blocked(exc):
                        self._mark_exchange_blocked(
                            self._active_exchange_name,
                            detail,
                            symbol=symbol,
                        )
                        if self._switch_exchange(symbol, detail):
                            break
                    elif self._should_failover(exc, attempt):
                        if self._switch_exchange(symbol, detail):
                            break
                    if attempt < self.max_retries:
                        time.sleep(min(backoff, self.max_backoff))
                        backoff = min(self.max_backoff, backoff * 2)
        fallback = self._fallback_candles(
            symbol, limit, allow_synthetic=allow_synthetic
        )
        if fallback:
            return fallback
        if stale_details:
            message = ", ".join(stale_details)
            raise StaleDataError(
                f"Exchange returned stale candles for {symbol} ({message})"
            ) from last_exc
        if last_exc is not None:
            raise RuntimeError(f"Failed to fetch candles for {symbol}") from last_exc
        raise RuntimeError(f"Failed to fetch candles for {symbol}")

    def _refresh_symbol(self, symbol: str) -> None:
        skip_until = self._symbol_skip_until.get(symbol)
        now = time.time()
        if skip_until and skip_until > now:
            self._log(
                "INFO",
                symbol,
                "Skipping refresh due to previous stale data",
                detail=f"retry_at={skip_until:.0f}",
            )
            return
        if skip_until and skip_until <= now:
            self._symbol_skip_until.pop(symbol, None)

        latest = self.store.latest_timestamp(symbol)
        since = int(latest + 1) if latest is not None else None
        limit = 250 if latest is None else 20
        try:
            candles = self._fetch(symbol, limit=limit, since=since)
        except StaleDataError as exc:
            self._log(
                "WARNING",
                symbol,
                "Stale candles detected; pausing symbol",
                detail=self._error_message(exc),
            )
            self._mark_symbol_skip(
                symbol,
                reason="stale_data",
                duration=max(self._timeframe_seconds * 2, self.interval_seconds),
            )
            if self.mode.lower() == "paper" and (
                latest is None or self._symbol_stale(latest)
            ):
                self._maybe_seed_symbol(symbol, reason="stale_skip")
            return
        except Exception as exc:  # pragma: no cover - network guard
            self._log(
                "ERROR",
                symbol,
                "Exhausted retries fetching candles",
                detail=self._error_message(exc),
            )
            if self.mode.lower() == "paper" and (
                latest is None or self._symbol_stale(latest)
            ):
                self._maybe_seed_symbol(symbol, reason="refresh_failure")
            return

        if not candles:
            if (
                self.mode.lower() == "paper"
                and latest is not None
                and self._symbol_stale(latest)
            ):
                self._maybe_seed_symbol(symbol, reason="empty_fetch")
            return

        filtered: List[Dict[str, float]] = []
        for candle in candles:
            ts = float(candle.get("timestamp", 0.0) or 0.0)
            if latest is None or ts > latest:
                filtered.append(candle)

        if not filtered:
            if (
                self.mode.lower() == "paper"
                and latest is not None
                and self._symbol_stale(latest)
            ):
                self._maybe_seed_symbol(symbol, reason="stale_store_no_updates")
            return

        inserted = self.store.append(symbol, filtered)
        if inserted:
            self._log("INFO", symbol, f"Appended {inserted} new candles")

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
        self._thread = threading.Thread(
            target=self._run, name="FeedUpdater", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:  # pragma: no cover - threading
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.interval_seconds + 5)
        self._thread = None

    def close(self) -> None:
        self.stop()
        for exchange in list(self._exchange_pool.values()):
            close = getattr(exchange, "close", None)
            if callable(close):  # pragma: no cover - network cleanup
                try:
                    close()
                except Exception:
                    pass
        self._exchange_pool.clear()
        self._exchange = None
        self.store.close()
