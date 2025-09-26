"""Thread-safe runtime state store shared between the trading loop and API."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Dict, Iterable, List, Mapping, Optional

from ai_trader.services.types import OpenPosition, TradeIntent


@dataclass(slots=True)
class PositionSnapshot:
    """Serializable representation of an open position."""

    worker: str
    symbol: str
    side: str
    quantity: float
    entry_price: float
    cash_spent: float
    opened_at: Optional[datetime]
    unrealized_pnl: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        opened_at = payload.get("opened_at")
        if isinstance(opened_at, datetime):
            payload["opened_at"] = opened_at.isoformat()
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "PositionSnapshot":
        opened = payload.get("opened_at")
        opened_at: Optional[datetime]
        if isinstance(opened, str):
            try:
                opened_at = datetime.fromisoformat(opened)
            except ValueError:
                opened_at = None
        elif isinstance(opened, datetime):
            opened_at = opened
        else:
            opened_at = None
        return cls(
            worker=str(payload.get("worker", "")),
            symbol=str(payload.get("symbol", "")),
            side=str(payload.get("side", "")),
            quantity=float(payload.get("quantity", 0.0)),
            entry_price=float(payload.get("entry_price", 0.0)),
            cash_spent=float(payload.get("cash_spent", 0.0)),
            opened_at=opened_at,
            unrealized_pnl=float(payload.get("unrealized_pnl", 0.0)),
        )


class RuntimeStateStore:
    """Centralised runtime state shared across services and the API layer."""

    def __init__(self, state_file: Path | None = None) -> None:
        self._lock = RLock()
        self._state_file = state_file
        self._balances: Dict[str, float] = {}
        self._equity: float = 0.0
        self._starting_equity: float | None = None
        self._base_currency: str = "USD"
        self._open_positions: List[PositionSnapshot] = []
        self._last_trade_timestamp: datetime | None = None
        self._last_update_time: datetime | None = None
        self._realized_pnl_usd: float = 0.0
        self._realized_pnl_percent: float = 0.0
        self._unrealized_pnl_usd: float = 0.0
        self._unrealized_pnl_percent: float = 0.0
        self._risk_settings: Dict[str, float | int] = {}
        self._risk_revision: int = 0
        if state_file and state_file.exists():
            self.refresh_from_disk()

    # ------------------------------------------------------------------
    # Public mutators
    # ------------------------------------------------------------------
    def set_base_currency(self, currency: str) -> None:
        with self._lock:
            if currency:
                self._base_currency = currency.upper()
            self._persist_locked()
            self._mark_updated_locked()

    def set_starting_equity(self, value: float | None) -> None:
        with self._lock:
            if value is not None and value > 0:
                self._starting_equity = float(value)
            self._persist_locked()
            self._mark_updated_locked()

    def update_account(
        self,
        *,
        equity: float,
        balances: Mapping[str, float],
        pnl_percent: float,
        pnl_usd: float,
        open_positions: Iterable[OpenPosition],
        prices: Mapping[str, float] | None = None,
        starting_equity: float | None = None,
    ) -> None:
        with self._lock:
            self._equity = float(equity)
            self._balances = {asset: float(amount) for asset, amount in balances.items()}
            if starting_equity is not None and starting_equity > 0:
                self._starting_equity = float(starting_equity)
            self._unrealized_pnl_usd = self._compute_unrealized_pnl(open_positions, prices)
            baseline = self._starting_equity or 0.0
            if baseline > 0:
                self._unrealized_pnl_percent = (self._unrealized_pnl_usd / baseline) * 100
                self._realized_pnl_percent = (self._realized_pnl_usd / baseline) * 100
            else:
                self._unrealized_pnl_percent = 0.0
                self._realized_pnl_percent = 0.0
            self._open_positions = self._serialize_positions(open_positions, prices)
            # PnL metrics derived from equity include realised + unrealised.
            # We keep the aggregate for completeness even though the API exposes
            # the breakdown separately.
            self._persist_locked()
            self._mark_updated_locked()

    def update_open_positions(
        self,
        positions: Iterable[OpenPosition],
        prices: Mapping[str, float] | None = None,
    ) -> None:
        with self._lock:
            self._open_positions = self._serialize_positions(positions, prices)
            self._unrealized_pnl_usd = self._compute_unrealized_pnl(positions, prices)
            baseline = self._starting_equity or 0.0
            if baseline > 0:
                self._unrealized_pnl_percent = (self._unrealized_pnl_usd / baseline) * 100
            else:
                self._unrealized_pnl_percent = 0.0
            self._persist_locked()
            self._mark_updated_locked()

    def record_trade(self, trade: TradeIntent) -> None:
        with self._lock:
            self._last_trade_timestamp = trade.created_at
            if trade.action == "CLOSE":
                pnl_usd = float(trade.pnl_usd or 0.0)
                self._realized_pnl_usd += pnl_usd
                baseline = self._starting_equity or 0.0
                if baseline > 0:
                    self._realized_pnl_percent = (self._realized_pnl_usd / baseline) * 100
            self._persist_locked()
            self._mark_updated_locked()

    def update_risk_settings(
        self, settings: Mapping[str, float | int | None], *, revision: int | None = None
    ) -> None:
        with self._lock:
            cleaned: Dict[str, float | int] = {}
            for key, value in settings.items():
                if value is None:
                    continue
                if isinstance(value, (int, float)):
                    cleaned[key] = value
                else:
                    try:
                        cleaned[key] = float(value)  # type: ignore[assignment]
                    except (TypeError, ValueError):
                        continue
            changed = False
            if cleaned:
                for key, value in cleaned.items():
                    if self._risk_settings.get(key) != value:
                        changed = True
                    self._risk_settings[key] = value
            if revision is not None and int(revision) != self._risk_revision:
                self._risk_revision = int(revision)
                changed = True
            if changed:
                self._persist_locked()
                self._mark_updated_locked()

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    def status_snapshot(self) -> Dict[str, object]:
        with self._lock:
            balance = self._balances.get(self._base_currency)
            if balance is None:
                balance = sum(self._balances.values())
            last_trade = (
                self._last_trade_timestamp.isoformat()
                if isinstance(self._last_trade_timestamp, datetime)
                else None
            )
            return {
                "balance": balance,
                "balances": dict(self._balances),
                "equity": self._equity,
                "open_positions": [pos.to_dict() for pos in self._open_positions],
                "last_trade_timestamp": last_trade,
                "last_update_time": (
                    self._last_update_time.isoformat()
                    if isinstance(self._last_update_time, datetime)
                    else None
                ),
                "risk_revision": self._risk_revision,
            }

    def profit_snapshot(self) -> Dict[str, object]:
        with self._lock:
            total_usd = self._realized_pnl_usd + self._unrealized_pnl_usd
            baseline = self._starting_equity or 0.0
            total_percent = (total_usd / baseline * 100) if baseline > 0 else 0.0
            return {
                "realized": {
                    "usd": self._realized_pnl_usd,
                    "percent": self._realized_pnl_percent,
                },
                "unrealized": {
                    "usd": self._unrealized_pnl_usd,
                    "percent": self._unrealized_pnl_percent,
                },
                "total": {"usd": total_usd, "percent": total_percent},
            }

    def risk_snapshot(self) -> Dict[str, float | int]:
        with self._lock:
            snapshot = dict(self._risk_settings)
            snapshot["revision"] = self._risk_revision
            return snapshot

    def refresh_from_disk(self) -> None:
        if not self._state_file or not self._state_file.exists():
            return
        try:
            raw = self._state_file.read_text(encoding="utf-8")
            payload = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            return
        with self._lock:
            balances = payload.get("balances")
            if isinstance(balances, Mapping):
                self._balances = {str(asset): float(amount) for asset, amount in balances.items()}
            self._equity = float(payload.get("equity", self._equity))
            self._starting_equity = self._coerce_optional_float(
                payload.get("starting_equity"), self._starting_equity
            )
            last_trade = payload.get("last_trade_timestamp")
            if isinstance(last_trade, str):
                try:
                    self._last_trade_timestamp = datetime.fromisoformat(last_trade)
                except ValueError:
                    self._last_trade_timestamp = None
            last_update = payload.get("last_update_time")
            if isinstance(last_update, str):
                try:
                    self._last_update_time = datetime.fromisoformat(last_update)
                except ValueError:
                    self._last_update_time = None
            realized = payload.get("realized")
            if isinstance(realized, Mapping):
                self._realized_pnl_usd = float(realized.get("usd", self._realized_pnl_usd))
                self._realized_pnl_percent = float(
                    realized.get("percent", self._realized_pnl_percent)
                )
            unrealized = payload.get("unrealized")
            if isinstance(unrealized, Mapping):
                self._unrealized_pnl_usd = float(unrealized.get("usd", self._unrealized_pnl_usd))
                self._unrealized_pnl_percent = float(
                    unrealized.get("percent", self._unrealized_pnl_percent)
                )
            risk_settings = payload.get("risk_settings")
            if isinstance(risk_settings, Mapping):
                cleaned: Dict[str, float | int] = {}
                for key, value in risk_settings.items():
                    name = str(key)
                    if isinstance(value, (int, float)):
                        cleaned[name] = float(value)
                        continue
                    try:
                        cleaned[name] = float(value)  # type: ignore[assignment]
                    except (TypeError, ValueError):
                        continue
                if cleaned:
                    self._risk_settings = cleaned
            revision = payload.get("risk_revision")
            if isinstance(revision, int):
                self._risk_revision = revision
            elif isinstance(revision, str):
                try:
                    self._risk_revision = int(revision)
                except ValueError:
                    pass
            base_currency = payload.get("base_currency")
            if isinstance(base_currency, str) and base_currency:
                self._base_currency = base_currency.upper()
            positions = payload.get("open_positions")
            if isinstance(positions, list):
                self._open_positions = [
                    PositionSnapshot.from_mapping(item)
                    for item in positions
                    if isinstance(item, Mapping)
                ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _serialize_positions(
        self,
        positions: Iterable[OpenPosition],
        prices: Mapping[str, float] | None,
    ) -> List[PositionSnapshot]:
        snapshots: List[PositionSnapshot] = []
        for position in positions:
            price = None
            if prices is not None:
                price = prices.get(position.symbol)
            unrealized = 0.0
            if price is not None:
                try:
                    unrealized = position.unrealized_pnl(float(price))
                except Exception:
                    unrealized = 0.0
            snapshots.append(
                PositionSnapshot(
                    worker=position.worker,
                    symbol=position.symbol,
                    side=position.side,
                    quantity=float(position.quantity),
                    entry_price=float(position.entry_price),
                    cash_spent=float(position.cash_spent),
                    opened_at=position.opened_at,
                    unrealized_pnl=unrealized,
                )
            )
        return snapshots

    def _compute_unrealized_pnl(
        self,
        positions: Iterable[OpenPosition],
        prices: Mapping[str, float] | None,
    ) -> float:
        total = 0.0
        if prices is None:
            return total
        for position in positions:
            price = prices.get(position.symbol)
            if price is None:
                continue
            try:
                total += position.unrealized_pnl(float(price))
            except Exception:
                continue
        return total

    def _persist_locked(self) -> None:
        if not self._state_file:
            return
        payload = {
            "balances": self._balances,
            "equity": self._equity,
            "starting_equity": self._starting_equity,
            "base_currency": self._base_currency,
            "open_positions": [pos.to_dict() for pos in self._open_positions],
            "last_trade_timestamp": (
                self._last_trade_timestamp.isoformat()
                if isinstance(self._last_trade_timestamp, datetime)
                else None
            ),
            "last_update_time": (
                self._last_update_time.isoformat()
                if isinstance(self._last_update_time, datetime)
                else None
            ),
            "realized": {
                "usd": self._realized_pnl_usd,
                "percent": self._realized_pnl_percent,
            },
            "unrealized": {
                "usd": self._unrealized_pnl_usd,
                "percent": self._unrealized_pnl_percent,
            },
            "risk_settings": dict(self._risk_settings),
            "risk_revision": self._risk_revision,
        }
        try:
            if self._state_file.parent:
                self._state_file.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self._state_file.with_suffix(".tmp")
            temp_path.write_text(json.dumps(payload, default=str), encoding="utf-8")
            temp_path.replace(self._state_file)
        except OSError:
            # Disk persistence is best-effort; runtime consumers still see in-memory state.
            return

    @staticmethod
    def _coerce_optional_float(value: object, fallback: float | None) -> float | None:
        if value is None:
            return fallback
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    def mark_runtime_update(self) -> None:
        """Record a manual runtime heartbeat timestamp."""

        with self._lock:
            self._mark_updated_locked()

    def last_update_time(self) -> datetime | None:
        with self._lock:
            return self._last_update_time

    def _mark_updated_locked(self) -> None:
        self._last_update_time = datetime.now(timezone.utc)


__all__ = ["RuntimeStateStore", "PositionSnapshot"]
