"""Broker facade built on top of CCXT."""

from __future__ import annotations

import ast
import json
import random
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Optional

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

try:  # pragma: no cover - optional import guard
    from typing import TYPE_CHECKING
except ImportError:  # pragma: no cover
    TYPE_CHECKING = False

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from desk.services.telemetry import TelemetryClient


class ExchangeBlockedError(RuntimeError):
    """Raised when an exchange rejects the connection due to geo restrictions."""


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
        *,
        telemetry: Optional["TelemetryClient"] = None,
        paper_params: Optional[Dict[str, float]] = None,
        fallback_exchanges: Optional[Iterable[str]] = None,
    ) -> None:
        normalized_mode = str(mode or "").strip().lower()
        if not normalized_mode:
            normalized_mode = "paper"
        self.mode = normalized_mode
        self.exchange_name = exchange_name
        self.api_key = api_key
        self.api_secret = api_secret
        self._fallback_exchanges = [
            str(name)
            for name in (fallback_exchanges or [])
            if str(name or "").strip()
        ]
        self._blocked_until: dict[str, float] = {}
        self._blocked_retry_seconds = 900.0
        self._candidate_exchanges = self._build_candidate_list(exchange_name)
        self.exchange = self._initialise_exchange()

        self.starting_balance = starting_balance
        self.db_path = Path(db_path or DESK_ROOT / "logs" / "balances.db")
        self._init_db()
        self._latency_log: list[Dict[str, float]] = []
        self.telemetry = telemetry
        self._rng = random.Random()
        self.paper_params = {
            "fee_bps": 10.0,
            "slippage_bps": 5.0,
            "partial_fill_probability": 0.1,
            "min_fill_ratio": 0.6,
            "funding_rate_hourly": 0.0,
        }
        if paper_params:
            self.paper_params.update({k: float(v) for k, v in paper_params.items()})

        if self.mode == "paper":
            bal = self._load_balance()
            if not bal:
                self.balance_data = {"USD": starting_balance, "positions": {}}
                self._save_balance()
            else:
                self.balance_data = bal
        else:
            self.balance_data = None

        print(f"[BrokerCCXT] Running in {self.mode.upper()} mode")

    @staticmethod
    def _error_message(exc: Exception) -> str:
        message = "".join(str(part) for part in getattr(exc, "args", []) if part)
        return message or str(exc)

    @staticmethod
    def _is_geo_blocked(exc: Exception) -> bool:
        message = BrokerCCXT._error_message(exc).lower()
        if not message:
            return False
        blockers = ("restricted", "forbidden", "not available", "denied", "prohibited")
        if "us" not in message and "united states" not in message:
            return False
        return any(token in message for token in blockers)

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        message = BrokerCCXT._error_message(exc).lower()
        transient_terms = (
            "timeout",
            "temporarily",
            "service unavailable",
            "connection reset",
            "network",
            "rate limit",
        )
        return any(term in message for term in transient_terms)

    def _build_candidate_list(self, primary: str) -> list[str]:
        seen = set()
        candidates: list[str] = []
        for name in [primary, *self._fallback_exchanges]:
            if not name:
                continue
            lname = name.lower()
            if lname in seen:
                continue
            seen.add(lname)
            candidates.append(name)
        if not candidates:
            candidates.append("kraken")
        return candidates

    def _mark_blocked(self, name: str, reason: str) -> None:
        retry_at = time.time() + self._blocked_retry_seconds
        self._blocked_until[name.lower()] = retry_at
        print(f"[BrokerCCXT] Exchange {name} blocked until {retry_at:.0f}: {reason}")

    def _is_blocked(self, name: str) -> bool:
        retry_at = self._blocked_until.get(name.lower())
        if not retry_at:
            return False
        if retry_at <= time.time():
            self._blocked_until.pop(name.lower(), None)
            return False
        return True

    def _create_exchange(self, name: str):
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
                    exchange.close() if hasattr(exchange, "close") else None
                    raise ExchangeBlockedError(self._error_message(exc)) from exc
                raise
        return exchange

    def _initialise_exchange(self):  # pragma: no cover - network heavy
        last_error: Exception | None = None
        for name in self._candidate_exchanges:
            if self._is_blocked(name):
                continue
            try:
                exchange = self._create_exchange(name)
            except ExchangeBlockedError as exc:
                self._mark_blocked(name, self._error_message(exc))
                last_error = exc
                continue
            except Exception as exc:  # pragma: no cover - network guard
                last_error = exc
                continue
            self.exchange_name = name
            if name != self._candidate_exchanges[0]:
                print(f"[BrokerCCXT] Using fallback exchange {name}")
            return exchange
        if last_error is not None:
            raise last_error
        raise RuntimeError("Unable to initialise any exchange")

    def _promote_fallback(self, *, reason: str) -> bool:  # pragma: no cover - network heavy
        for name in self._candidate_exchanges:
            if name == self.exchange_name:
                continue
            if self._is_blocked(name):
                continue
            try:
                exchange = self._create_exchange(name)
            except ExchangeBlockedError as exc:
                self._mark_blocked(name, self._error_message(exc))
                continue
            except Exception as exc:  # pragma: no cover - network guard
                print(f"[BrokerCCXT] Failed to promote {name}: {self._error_message(exc)}")
                continue
            old_exchange = self.exchange
            self.exchange = exchange
            self.exchange_name = name
            print(f"[BrokerCCXT] Switched to {name} due to: {reason}")
            try:
                old_exchange.close()
            except Exception:
                pass
            return True
        return False

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
            if self.telemetry:
                self.telemetry.record_latency(operation, duration)

    @property
    def latency_log(self) -> list[Dict[str, float]]:
        return list(self._latency_log)

    def _execute_with_retry(self, operation: str, func):  # pragma: no cover - network heavy
        attempts = 0
        backoff = 1.0
        last_exc: Exception | None = None
        while attempts < 5:
            attempts += 1
            try:
                with self._measure_latency(operation):
                    return func(self.exchange)
            except Exception as exc:
                last_exc = exc
                message = self._error_message(exc)
                if self._is_geo_blocked(exc):
                    self._mark_blocked(self.exchange_name, message)
                    if self._promote_fallback(reason=message):
                        backoff = 1.0
                        continue
                else:
                    switched = False
                    if not self._is_transient(exc) or attempts >= 2:
                        switched = self._promote_fallback(reason=message)
                    if switched:
                        backoff = 1.0
                        continue
                if attempts < 5:
                    time.sleep(min(backoff, 15.0))
                    backoff = min(self._blocked_retry_seconds / 2, backoff * 2)
                    continue
                break
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"{operation} failed without raising an exception")

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
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        *,
        limit: int = 50,
        since: int | float | None = None,
    ):
        normalized_since = int(since) if since is not None else None
        return self._execute_with_retry(
            "fetch_ohlcv",
            lambda exchange: exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                limit=limit,
                since=normalized_since,
            ),
        )

    def fetch_price(self, symbol: str) -> float:
        ticker = self._execute_with_retry(
            "fetch_price", lambda exchange: exchange.fetch_ticker(symbol)
        )
        return float(ticker["last"])

    # ----------------------------
    # Balance Management
    # ----------------------------
    def balance(self):
        if self.mode == "paper":
            return self.balance_data
        return self.exchange.fetch_balance()

    def _normalize_position(self, base: str) -> Dict[str, float]:
        raw = self.balance_data["positions"].get(base, {})
        if isinstance(raw, dict):
            qty = float(raw.get("qty", 0.0) or 0.0)
            cost_basis = float(raw.get("cost_basis", 0.0) or 0.0)
            last_funding = float(raw.get("last_funding", time.time()) or time.time())
        else:
            qty = float(raw or 0.0)
            cost_basis = 0.0
            last_funding = time.time()
        return {"qty": qty, "cost_basis": cost_basis, "last_funding": last_funding}

    def _apply_funding(self, base: str, mark_price: float) -> None:
        funding_rate = float(self.paper_params.get("funding_rate_hourly", 0.0))
        if funding_rate == 0:
            return
        position = self._normalize_position(base)
        qty = position["qty"]
        if qty == 0:
            return
        now = time.time()
        hours = max(0.0, (now - position["last_funding"]) / 3600.0)
        if hours <= 0:
            return
        funding = abs(qty) * mark_price * funding_rate * hours
        if qty > 0:
            self.balance_data["USD"] -= funding
        else:
            self.balance_data["USD"] += funding
        position["last_funding"] = now
        self.balance_data["positions"][base] = position

    def update_balance(
        self,
        pnl: float,
        qty: float,
        side: str,
        price: float,
        symbol: str,
        *,
        fee: float,
        remaining_qty: float,
    ):
        if self.mode != "paper":
            return

        base, _quote = symbol.split("/")
        self._apply_funding(base, price)
        cost = qty * price

        if side == "buy":
            if self.balance_data["USD"] < cost:
                print(f"[BrokerCCXT] Not enough USD balance to buy {qty} {base}")
                return False
            self.balance_data["USD"] -= cost + fee
            position = self._normalize_position(base)
            total_cost = position["cost_basis"] * position["qty"] + cost
            position["qty"] += qty
            if position["qty"] > 0:
                position["cost_basis"] = total_cost / position["qty"]
            position["last_funding"] = time.time()
            self.balance_data["positions"][base] = position

        elif side == "sell":
            position = self._normalize_position(base)
            if position["qty"] < qty:
                print(f"[BrokerCCXT] Not enough {base} to sell {qty}")
                return False
            position["qty"] -= qty
            position["last_funding"] = time.time()
            if position["qty"] <= 0:
                position["cost_basis"] = 0.0
            self.balance_data["positions"][base] = position
            self.balance_data["USD"] += cost - fee

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
            slippage = self.paper_params.get("slippage_bps", 0.0) / 10_000.0
            fee_bps = self.paper_params.get("fee_bps", 0.0) / 10_000.0
            fill_qty = qty
            remaining_qty = 0.0
            if self.paper_params.get("partial_fill_probability", 0.0) > 0:
                if self._rng.random() < self.paper_params["partial_fill_probability"]:
                    ratio = self._rng.uniform(
                        float(self.paper_params.get("min_fill_ratio", 0.5)),
                        1.0,
                    )
                    fill_qty = qty * ratio
                    remaining_qty = qty - fill_qty

            effective_price = price
            if slippage:
                slip = slippage * price
                effective_price = price + slip if side == "buy" else price - slip

            fee = abs(effective_price * fill_qty) * fee_bps

            trade = {
                "symbol": symbol,
                "side": side,
                "requested_qty": qty,
                "qty": fill_qty,
                "price": effective_price,
                "timestamp": time.time(),
                "fee": fee,
                "slippage": effective_price - price,
                "remaining_qty": remaining_qty,
                "pnl": 0.0,
            }
            success = self.update_balance(
                pnl=0.0,
                qty=fill_qty,
                side=side,
                price=effective_price,
                symbol=symbol,
                fee=fee,
                remaining_qty=remaining_qty,
            )
            if not success:
                return None
            with self._measure_latency("market_order"):
                pass
            return trade

        with self._measure_latency("market_order"):
            order = self.exchange.create_order(symbol, "market", side, qty)
        return order

    def close(self) -> None:
        """Close the underlying exchange session if supported."""

        close = getattr(self.exchange, "close", None)
        if callable(close):  # pragma: no cover - network resource cleanup
            try:
                close()
            except Exception:
                pass


