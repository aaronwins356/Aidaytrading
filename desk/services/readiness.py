"""Readiness checks that validate the trading runtime before going live."""

from __future__ import annotations

import math
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from desk.services.pretty_logger import pretty_logger


CheckFunc = Callable[[], List["ReadinessIssue"]]


@dataclass(frozen=True)
class ReadinessIssue:
    """Represents the outcome of a single readiness validation."""

    check: str
    level: str
    message: str

    def to_dict(self) -> MutableMapping[str, str]:
        return {"check": self.check, "level": self.level, "message": self.message}


@dataclass(frozen=True)
class ReadinessReport:
    """Collection of readiness issues gathered during startup."""

    generated_at: float
    issues: Sequence[ReadinessIssue] = field(default_factory=tuple)

    def summary(self) -> MutableMapping[str, int]:
        counts: MutableMapping[str, int] = {"ok": 0, "warning": 0, "error": 0}
        for issue in self.issues:
            counts[issue.level] = counts.get(issue.level, 0) + 1
        return counts

    def overall_status(self) -> str:
        for issue in self.issues:
            if issue.level == "error":
                return "error"
        for issue in self.issues:
            if issue.level == "warning":
                return "warning"
        return "ok"

    def to_dict(self) -> MutableMapping[str, object]:
        return {
            "generated_at": float(self.generated_at),
            "status": self.overall_status(),
            "counts": self.summary(),
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def render_console(self) -> str:
        emoji = {"ok": "✅", "warning": "🟡", "error": "🟥"}
        header = f"Readiness Report {emoji.get(self.overall_status(), '🟡')}"
        lines = [header]
        for issue in self.issues:
            prefix = emoji.get(issue.level, "🟡")
            lines.append(f"  {prefix} {issue.check}: {issue.message}")
        if len(self.issues) == 0:
            lines.append("  ✅ All readiness checks passed.")
        return "\n".join(lines)


class ReadinessChecker:
    """Performs a suite of runtime readiness checks.

    The checker is dependency-injected so that tests can substitute a fake
    broker, deterministic time provider, and synthetic market metadata.
    """

    def __init__(
        self,
        *,
        broker,
        symbols: Iterable[str],
        required_env: Optional[Mapping[str, str]] = None,
        db_path: str | Path | None = None,
        time_provider: Callable[[], float] = time.time,
        server_time_fetcher: Optional[Callable[[], float]] = None,
        max_time_drift: float = 2.0,
        extra_checks: Optional[Sequence[CheckFunc]] = None,
    ) -> None:
        self.broker = broker
        self.symbols = [str(symbol) for symbol in symbols]
        self.required_env = dict(required_env or {})
        self.db_path = Path(db_path) if db_path else None
        self.time_provider = time_provider
        self.server_time_fetcher = server_time_fetcher
        self.max_time_drift = max(0.0, float(max_time_drift))
        self.extra_checks = list(extra_checks or [])

    # ------------------------------------------------------------------
    def run(self) -> ReadinessReport:
        issues: List[ReadinessIssue] = []
        for checker in (
            self._check_ccxt,
            self._check_symbols,
            self._check_market_precision,
            self._check_credentials,
            self._check_database,
            self._check_time_sync,
        ):
            try:
                issues.extend(checker())
            except Exception as exc:  # pragma: no cover - defensive guard
                pretty_logger.error(f"Readiness check {checker.__name__} failed: {exc}")
                issues.append(
                    ReadinessIssue(
                        check=checker.__name__,
                        level="error",
                        message=str(exc),
                    )
                )
        for checker in self.extra_checks:
            try:
                issues.extend(checker())
            except Exception as exc:  # pragma: no cover - defensive guard
                pretty_logger.error(f"Readiness extra check failed: {exc}")
                issues.append(
                    ReadinessIssue(
                        check=getattr(checker, "__name__", "extra_check"),
                        level="error",
                        message=str(exc),
                    )
                )
        timestamp = self.time_provider()
        return ReadinessReport(generated_at=float(timestamp), issues=tuple(issues))

    # ------------------------------------------------------------------
    def _check_ccxt(self) -> List[ReadinessIssue]:
        if getattr(self.broker, "exchange", None) is None:
            return [
                ReadinessIssue(
                    check="ccxt",
                    level="error",
                    message="Kraken broker exchange handle is unavailable.",
                )
            ]
        return [
            ReadinessIssue(
                check="ccxt",
                level="ok",
                message="ccxt exchange handle initialised.",
            )
        ]

    def _check_symbols(self) -> List[ReadinessIssue]:
        issues: List[ReadinessIssue] = []
        resolver = getattr(self.broker, "resolve_symbol", None)
        market_getter = getattr(self.broker, "exchange", None)
        for symbol in self.symbols:
            resolved = None
            if callable(resolver):
                try:
                    resolved = resolver(symbol)
                except Exception:
                    resolved = None
            resolved_symbol = resolved or str(symbol)
            market = None
            if market_getter is not None:
                try:
                    market = market_getter.market(resolved_symbol)
                except Exception:
                    market = getattr(self.broker, "_market_cache", {}).get(resolved_symbol)
            if not market:
                issues.append(
                    ReadinessIssue(
                        check=f"symbol:{symbol}",
                        level="error",
                        message="Market metadata not loaded for symbol.",
                    )
                )
                continue
            issues.append(
                ReadinessIssue(
                    check=f"symbol:{symbol}",
                    level="ok",
                    message=f"Resolved to {resolved_symbol} with market metadata present.",
                )
            )
        return issues

    def _check_market_precision(self) -> List[ReadinessIssue]:
        issues: List[ReadinessIssue] = []
        market_getter = getattr(self.broker, "exchange", None)
        for symbol in self.symbols:
            try:
                resolved = self.broker.resolve_symbol(symbol)
            except Exception:
                resolved = str(symbol)
            market = None
            if market_getter is not None:
                try:
                    market = market_getter.market(resolved)
                except Exception:
                    market = getattr(self.broker, "_market_cache", {}).get(resolved)
            precision = None
            min_notional = None
            min_amount = None
            if isinstance(market, Mapping):
                precision = market.get("precision", {}).get("amount")
                limits = market.get("limits") or {}
                if isinstance(limits, Mapping):
                    amount = limits.get("amount")
                    price = limits.get("price")
                    cost = limits.get("cost")
                    if isinstance(amount, Mapping):
                        min_amount = amount.get("min")
                    if isinstance(cost, Mapping):
                        min_notional = cost.get("min")
                    if min_notional is None and isinstance(price, Mapping) and min_amount:
                        try:
                            min_price = price.get("min")
                            if min_price:
                                min_notional = float(min_price) * float(min_amount)
                        except (TypeError, ValueError):
                            min_notional = None
            if precision is None and min_amount is None and min_notional is None:
                issues.append(
                    ReadinessIssue(
                        check=f"precision:{symbol}",
                        level="warning",
                        message="Precision or minimums missing; trades may be rejected.",
                    )
                )
                continue
            message_parts = []
            if precision is not None:
                message_parts.append(f"precision {precision}")
            if min_amount:
                message_parts.append(f"min_qty {min_amount}")
            if min_notional:
                message_parts.append(f"min_notional {min_notional}")
            issues.append(
                ReadinessIssue(
                    check=f"precision:{symbol}",
                    level="ok",
                    message="; ".join(message_parts) or "Precision metadata loaded.",
                )
            )
        return issues

    def _check_credentials(self) -> List[ReadinessIssue]:
        issues: List[ReadinessIssue] = []
        for key, value in self.required_env.items():
            env_value = os.environ.get(key, value)
            if not env_value:
                issues.append(
                    ReadinessIssue(
                        check=f"env:{key}",
                        level="error",
                        message="Missing environment variable.",
                    )
                )
            else:
                issues.append(
                    ReadinessIssue(
                        check=f"env:{key}",
                        level="ok",
                        message="Environment variable detected.",
                    )
                )
        return issues

    def _check_database(self) -> List[ReadinessIssue]:
        if not self.db_path:
            return [
                ReadinessIssue(
                    check="database",
                    level="warning",
                    message="Dashboard database path not configured.",
                )
            ]
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA user_version;")
        except Exception as exc:
            return [
                ReadinessIssue(
                    check="database",
                    level="error",
                    message=f"SQLite unavailable: {exc}",
                )
            ]
        return [
            ReadinessIssue(
                check="database",
                level="ok",
                message=f"SQLite ready at {self.db_path}",
            )
        ]

    def _check_time_sync(self) -> List[ReadinessIssue]:
        if not self.server_time_fetcher:
            return [
                ReadinessIssue(
                    check="time_sync",
                    level="warning",
                    message="Server time fetcher unavailable; drift unchecked.",
                )
            ]
        try:
            server_time = float(self.server_time_fetcher())
        except Exception as exc:
            return [
                ReadinessIssue(
                    check="time_sync",
                    level="warning",
                    message=f"Unable to fetch server time: {exc}",
                )
            ]
        local_time = float(self.time_provider())
        drift = abs(local_time - server_time)
        if math.isnan(drift) or math.isinf(drift):
            return [
                ReadinessIssue(
                    check="time_sync",
                    level="warning",
                    message="Invalid drift calculation.",
                )
            ]
        if drift > self.max_time_drift:
            return [
                ReadinessIssue(
                    check="time_sync",
                    level="error",
                    message=f"Clock drift {drift:.2f}s exceeds {self.max_time_drift:.2f}s limit.",
                )
            ]
        return [
            ReadinessIssue(
                check="time_sync",
                level="ok",
                message=f"Clock drift {drift:.2f}s within tolerance.",
            )
        ]


__all__ = ["ReadinessChecker", "ReadinessReport", "ReadinessIssue"]
