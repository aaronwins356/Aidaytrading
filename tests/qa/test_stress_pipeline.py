"""Stress harness covering trade logging, runtime state, and notifier."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import cast

import pytest

from ai_trader.notifier import Notifier
from ai_trader.services.runtime_state import RuntimeStateStore
from ai_trader.services.trade_log import TradeLog
from ai_trader.services.types import TradeIntent, TradeAction


class DummyBot:
    """Simple asynchronous bot stub used during stress tests."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send_message(
        self,
        chat_id: str,
        text: str,
        parse_mode: str | None = None,
        disable_web_page_preview: bool = True,
    ) -> None:  # noqa: D401
        del chat_id, parse_mode, disable_web_page_preview
        self.messages.append(text)


@pytest.mark.timeout(20)
def test_trade_logging_and_notifier_stress(tmp_path: Path) -> None:
    db_path = tmp_path / "trades.db"
    state_path = tmp_path / "runtime_state.json"
    trade_log = TradeLog(db_path)
    runtime_state = RuntimeStateStore(state_path)
    runtime_state.set_starting_equity(10000.0)
    dummy_bot = DummyBot()

    async def _exercise() -> float:
        notifier = Notifier(token="dummy", chat_id="123", bot=dummy_bot)
        await notifier.start()
        total_pnl = 0.0
        for idx in range(120):
            action = cast(TradeAction, "CLOSE" if idx % 2 else "OPEN")
            exit_price = 21000.0 + idx if action == "CLOSE" else None
            pnl_usd = (idx % 5) * 1.5 if action == "CLOSE" else None
            trade = TradeIntent(
                worker="stress",
                action=action,
                symbol="BTC/USDT",
                side="buy" if idx % 3 else "sell",
                cash_spent=400.0 + idx,
                entry_price=20000.0 + idx,
                exit_price=exit_price,
                pnl_usd=pnl_usd,
                pnl_percent=(pnl_usd or 0.0) / (400.0 + idx) * 100 if pnl_usd else None,
                confidence=0.6,
                metadata={"fill_quantity": 0.02 + idx * 0.0001},
            )
            trade_log.record_trade(trade)
            runtime_state.record_trade(trade)
            await notifier.send_trade_alert(trade)
            if pnl_usd is not None:
                total_pnl += pnl_usd
        await notifier.stop()
        return total_pnl

    total_pnl = asyncio.run(_exercise())

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    assert count == 120
    assert len(dummy_bot.messages) == 120
    snapshot = runtime_state.profit_snapshot()
    realized = snapshot["realized"]["usd"]
    assert realized == pytest.approx(total_pnl, rel=1e-6)
