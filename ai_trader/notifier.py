"""Telegram notifier for live trading alerts and heartbeats."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Mapping, MutableMapping, Optional

from telegram import Bot

from ai_trader.services.types import TradeIntent

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TelegramConfig:
    """Configuration payload required to talk to Telegram."""

    token: str
    chat_id: str


class Notifier:
    """Send rich Telegram alerts for trade activity and runtime health."""

    HEARTBEAT_HOURS = (2, 8, 14, 20)

    def __init__(
        self,
        *,
        token: str,
        chat_id: str,
        bot: Optional[Bot] = None,
    ) -> None:
        if not token or not chat_id:
            raise ValueError("Telegram notifier requires both token and chat_id")
        self._config = TelegramConfig(token=token, chat_id=str(chat_id))
        self._bot: Bot = bot or Bot(token=token)
        self._lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._last_heartbeat: datetime | None = None

    async def start(self) -> None:
        """Begin the background heartbeat scheduler."""

        if self._heartbeat_task is None:
            self._stop_event.clear()
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            LOGGER.info("Telegram notifier heartbeat loop started")

    async def stop(self) -> None:
        """Stop the heartbeat task and wait for completion."""

        self._stop_event.set()
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            finally:
                self._heartbeat_task = None
            LOGGER.info("Telegram notifier heartbeat loop stopped")

    async def send_trade_alert(self, trade: TradeIntent | Mapping[str, object]) -> None:
        """Send a formatted trade alert to Telegram."""

        payload = self._normalize_trade(trade)
        action = payload.get("action", "trade")
        verb = "Opened" if action == "OPEN" else "Closed"
        symbol = payload.get("symbol", "?")
        side = payload.get("side", "?").upper()
        price = payload.get("price")
        pnl_usd = payload.get("pnl_usd")
        pnl_pct = payload.get("pnl_percent")
        worker = payload.get("worker", "Unknown")
        confidence = payload.get("confidence")
        reason = payload.get("reason")
        metadata = payload.get("metadata", {})
        mode = metadata.get("mode") or payload.get("mode")

        lines = [
            f"📈 <b>{verb} {side}</b> {symbol} via <b>{worker}</b>",
            f"Price: <code>{price:,.2f}</code>" if isinstance(price, (int, float)) else None,
            (
                f"PnL: <code>{pnl_usd:,.2f} USD</code> ({pnl_pct:.2f}%)"
                if isinstance(pnl_usd, (int, float)) and isinstance(pnl_pct, (int, float))
                else None
            ),
            f"Confidence: {confidence:.2f}" if isinstance(confidence, (int, float)) else None,
            f"Mode: {mode}" if mode else None,
            f"Reason: {reason}" if reason else None,
        ]
        body = "\n".join(line for line in lines if line)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        message = f"{body}\n<sub>{timestamp}</sub>"
        await self._send_message(message, parse_mode="HTML")

    async def send_error(self, error: Exception | str) -> None:
        """Send a critical error notification."""

        if isinstance(error, Exception):
            message = f"⚠️ <b>Trading bot error:</b> {error}"
        else:
            message = f"⚠️ <b>Trading bot error:</b> {error}"
        await self._send_message(message, parse_mode="HTML")

    async def send_heartbeat(self) -> None:
        """Send a periodic heartbeat to confirm liveness."""

        now = datetime.now(timezone.utc)
        self._last_heartbeat = now
        message = now.strftime("✅ Heartbeat — bot online at %H:%M UTC (%Y-%m-%d)")
        await self._send_message(message)

    async def _send_message(self, text: str, *, parse_mode: str | None = None) -> None:
        try:
            async with self._lock:
                await self._bot.send_message(
                    chat_id=self._config.chat_id,
                    text=text,
                    parse_mode=parse_mode,
                    disable_web_page_preview=True,
                )
        except Exception as exc:  # noqa: BLE001 - logging only; notifier must not crash loop
            LOGGER.warning("Failed to send Telegram message: %s", exc)

    def _normalize_trade(
        self, trade: TradeIntent | Mapping[str, object]
    ) -> MutableMapping[str, object]:
        if isinstance(trade, TradeIntent):
            metadata = dict(trade.metadata or {})
            fill_price = metadata.get("fill_price")
            if fill_price is None:
                fill_price = trade.exit_price if trade.exit_price else trade.entry_price
            metadata.setdefault(
                "mode",
                metadata.get("mode")
                or ("paper" if trade.cash_spent and trade.cash_spent < 1_000 else "live"),
            )
            return {
                "worker": trade.worker,
                "action": trade.action,
                "symbol": trade.symbol,
                "side": trade.side,
                "price": float(fill_price) if fill_price is not None else float(trade.entry_price),
                "pnl_usd": trade.pnl_usd,
                "pnl_percent": trade.pnl_percent,
                "confidence": trade.confidence,
                "reason": trade.reason,
                "metadata": metadata,
            }
        return dict(trade)

    async def _heartbeat_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                delay = self._seconds_until_next_heartbeat()
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    await self.send_heartbeat()
        except asyncio.CancelledError:
            raise

    def _seconds_until_next_heartbeat(self) -> float:
        now = datetime.now(timezone.utc)
        for hour in self.HEARTBEAT_HOURS:
            target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            if target > now:
                return (target - now).total_seconds()
        tomorrow = (now + timedelta(days=1)).replace(
            hour=self.HEARTBEAT_HOURS[0], minute=0, second=0, microsecond=0
        )
        return max((tomorrow - now).total_seconds(), 0.0)
