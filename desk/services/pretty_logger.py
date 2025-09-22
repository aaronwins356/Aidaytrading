"""Console logger tailored for concise trading desk output."""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class _ColourScheme:
    info: str = "\033[36m"
    warning: str = "\033[33m"
    error: str = "\033[31m"
    trade: str = "\033[32m"
    reset: str = "\033[0m"


class PrettyLogger:
    """Formats runtime events into human-friendly console output."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._colours = _ColourScheme()
        self._dedupe_cache: Dict[str, str] = {}
        self._rest_fallback_logged = False
        self._verbose = False

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def configure(self, *, verbose: Optional[bool] = None) -> None:
        if verbose is not None:
            self._verbose = bool(verbose)

    # ------------------------------------------------------------------
    # Core logging primitives
    # ------------------------------------------------------------------
    def info(self, message: str, *, dedupe_key: str | None = None) -> None:
        self._emit("INFO", message, dedupe_key=dedupe_key)

    def warning(self, message: str, *, dedupe_key: str | None = None) -> None:
        self._emit("WARNING", message, dedupe_key=dedupe_key)

    def error(self, message: str, *, dedupe_key: str | None = None) -> None:
        self._emit("ERROR", message, dedupe_key=dedupe_key)

    def debug(self, message: str) -> None:
        if self._verbose:
            self._emit("INFO", message, force=True)

    def _emit(
        self,
        level: str,
        message: str,
        *,
        dedupe_key: str | None = None,
        force: bool = False,
    ) -> None:
        if not message:
            return
        upper_level = level.upper()
        if not self._verbose and not force and dedupe_key:
            cached = self._dedupe_cache.get(dedupe_key)
            if cached == message:
                return
            self._dedupe_cache[dedupe_key] = message
        text = f"[{upper_level}] {message}"
        colour = self._resolve_colour(upper_level)
        if colour:
            text = f"{colour}{text}{self._colours.reset}"
        with self._lock:
            print(text, file=sys.stdout, flush=True)

    def _resolve_colour(self, level: str) -> str:
        if level == "INFO":
            return self._colours.info
        if level == "WARNING":
            return self._colours.warning
        if level == "ERROR":
            return self._colours.error
        if level == "TRADE":
            return self._colours.trade
        return ""

    # ------------------------------------------------------------------
    # Domain specific helpers
    # ------------------------------------------------------------------
    def rest_fallback_notice(self) -> None:
        if self._rest_fallback_logged and not self._verbose:
            return
        self._rest_fallback_logged = True
        self.info("Using REST fallback", dedupe_key="rest_fallback")

    def warmup_progress(self, symbol: str, ready: int, total: int) -> None:
        self.info(
            f"[{symbol}] Warming candles ({ready}/{total})",
            dedupe_key=f"warmup:{symbol}",
        )

    def worker_live(self, worker_name: str) -> None:
        self.info(f"[Worker {worker_name} LIVE]", dedupe_key=f"worker_live:{worker_name}")

    def feed_bootstrap(self, symbol: str, candles: int) -> None:
        self.info(
            f"[Feed] {symbol}: {candles} candles bootstrapped",
            dedupe_key=f"feed_bootstrap:{symbol}",
        )

    def trade_message(self, message: str) -> None:
        self._emit("TRADE", message, force=True)

    def trade_opened(
        self,
        symbol: str,
        side: str,
        qty: float,
        usd_value: float,
        equity_pct: float,
        price: float,
    ) -> None:
        direction = "LONG" if side.upper() in {"BUY", "LONG"} else "SHORT"
        base_asset = symbol.split("/")[0] if "/" in symbol else symbol
        qty_str = f"{qty:.6f}".rstrip("0").rstrip(".") or "0"
        equity_display = f"{equity_pct:.1%}" if equity_pct > 0 else "0.0%"
        message = (
            f"TRADE OPENED [{symbol}] {direction} {qty_str} {base_asset} "
            f"(${usd_value:,.2f}, {equity_display} equity) @ ${price:,.2f}"
        )
        self._emit(
            "TRADE",
            message,
            dedupe_key=f"trade_open:{symbol}:{direction}:{qty:.6f}:{price}",
        )

    def trade_closed(
        self,
        symbol: str,
        side: str,
        price: float,
        pnl_pct: float,
        pnl_usd: float,
    ) -> None:
        direction = "LONG" if side.upper() in {"BUY", "LONG"} else "SHORT"
        sign = "+" if pnl_pct >= 0 else ""
        usd_sign = "+" if pnl_usd >= 0 else ""
        message = (
            f"TRADE CLOSED [{symbol}] {direction} @ ${price:,.2f} | "
            f"PnL: {sign}{pnl_pct:.2f}% ({usd_sign}${pnl_usd:,.2f})"
        )
        self._emit("TRADE", message)


pretty_logger = PrettyLogger()

__all__ = ["pretty_logger", "PrettyLogger"]

