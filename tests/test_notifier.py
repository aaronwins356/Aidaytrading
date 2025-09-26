from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from ai_trader.notifier import Notifier
from ai_trader.services.monitoring import get_monitoring_center
from ai_trader.services.types import TradeIntent


class DummyBot:
    def __init__(self) -> None:
        self.send_message = AsyncMock()


def test_send_trade_alert_uses_bot() -> None:
    get_monitoring_center().reset()
    bot = DummyBot()
    notifier = Notifier(token="token", chat_id="123", bot=bot)  # type: ignore[arg-type]
    trade = TradeIntent(
        worker="Momentum Scout",
        action="OPEN",
        symbol="BTC/USD",
        side="buy",
        cash_spent=100.0,
        entry_price=25000.0,
        confidence=0.7,
    )

    asyncio.run(notifier.send_trade_alert(trade))

    assert bot.send_message.await_count == 1
    _, kwargs = bot.send_message.call_args
    assert "BTC/USD" in kwargs["text"]


def test_send_error_pushes_message() -> None:
    get_monitoring_center().reset()
    bot = DummyBot()
    notifier = Notifier(token="token", chat_id="123", bot=bot)  # type: ignore[arg-type]

    asyncio.run(notifier.send_error("drawdown breach"))

    assert bot.send_message.await_count == 1
    assert "drawdown" in bot.send_message.call_args.kwargs["text"]


def test_start_and_stop_heartbeat() -> None:
    get_monitoring_center().reset()
    bot = DummyBot()
    notifier = Notifier(token="token", chat_id="123", bot=bot)  # type: ignore[arg-type]

    async def _runner() -> None:
        await notifier.start()
        assert notifier._heartbeat_task is not None  # type: ignore[attr-defined]
        await notifier.stop()
        assert notifier._heartbeat_task is None  # type: ignore[attr-defined]

    asyncio.run(_runner())


def test_startup_heartbeat_emitted_before_scheduler() -> None:
    center = get_monitoring_center()
    center.reset()
    bot = DummyBot()
    notifier = Notifier(token="token", chat_id="123", bot=bot)  # type: ignore[arg-type]

    asyncio.run(
        notifier.send_startup_heartbeat(
            equity=1250.5,
            open_positions=2,
            mode="paper",
            currency="USD",
        )
    )

    assert bot.send_message.await_count == 1
    payload = bot.send_message.call_args.kwargs["text"]
    assert "Startup heartbeat" in payload
    events = center.recent_events()
    assert any(event["event_type"] == "notifier_startup" for event in events)
