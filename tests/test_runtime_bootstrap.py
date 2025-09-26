import asyncio
import logging
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest

if "colorama" not in sys.modules:
    colorama_stub = types.ModuleType("colorama")
    colorama_stub.Fore = types.SimpleNamespace(
        BLUE="",
        GREEN="",
        YELLOW="",
        RED="",
        MAGENTA="",
        RESET="",
    )
    colorama_stub.Style = types.SimpleNamespace(RESET_ALL="")

    def _noop_init(*_: object, **__: object) -> None:
        return None

    colorama_stub.init = _noop_init
    sys.modules["colorama"] = colorama_stub

from ai_trader.runtime import (
    RuntimeConfigBundle,
    initialise_notifier,
    prepare_runtime_config,
    start_watchdog,
    warm_start_workers,
)
from ai_trader.runtime import bootstrap


class _DummyRuntimeState:
    def __init__(self) -> None:
        self._last_update = datetime.now(timezone.utc)

    def last_update_time(self) -> datetime:
        return self._last_update


class _DummyNotifier:
    async def send_watchdog_alert(self, timeout: float, last_update: datetime | None) -> None:
        return None


def _bundle_for_tests(config: dict[str, object] | None = None) -> RuntimeConfigBundle:
    runtime_config: dict[str, object] = config or {"trading": {}, "notifications": {}}
    trading_cfg = runtime_config.setdefault("trading", {})
    return RuntimeConfigBundle(
        config=runtime_config,
        trading=trading_cfg,
        risk=runtime_config.setdefault("risk", {}),
        workers=runtime_config.setdefault("workers", {}),
        symbols=list(trading_cfg.get("symbols", [])),
        paper_mode=True,
        trading_mode="PAPER",
    )


def test_prepare_runtime_config_no_symbols_raises(tmp_path: Path) -> None:
    logger = logging.getLogger("runtime-bootstrap-test")
    with pytest.raises(SystemExit):
        prepare_runtime_config({}, data_dir=tmp_path, logger=logger)


def test_initialise_notifier_failure_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    bundle = _bundle_for_tests(
        {
            "notifications": {
                "telegram": {"enabled": True, "bot_token": "token", "chat_id": "chat"}
            },
            "trading": {},
        }
    )

    class BoomNotifier:
        def __init__(self, *_: object, **__: object) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(bootstrap, "Notifier", BoomNotifier)
    logger = logging.getLogger("runtime-bootstrap-test")

    notifier = initialise_notifier(bundle, logger=logger)
    assert notifier is None


def test_warm_start_workers_ignores_malformed_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    symbol = "BTC/USD"
    bundle = RuntimeConfigBundle(
        config={"trading": {"symbols": [symbol]}},
        trading={"symbols": [symbol]},
        risk={},
        workers={},
        symbols=[symbol],
        paper_mode=True,
        trading_mode="PAPER",
    )

    cache_path = tmp_path / "btc_usd.csv"
    cache_path.write_text("open,high,low,volume\n1,2,3,4\n", encoding="utf-8")

    class DummyResearcherBase:
        pass

    class DummyResearcher(DummyResearcherBase):
        def __init__(self) -> None:
            self.preloaded: list[tuple[str, list[dict[str, float]]]] = []

        def preload_candles(
            self, symbol_name: str, candles: list[dict[str, float]]
        ) -> None:
            self.preloaded.append((symbol_name, candles))

    monkeypatch.setattr(bootstrap, "MarketResearchWorker", DummyResearcherBase)

    worker = type("DummyWorker", (), {"price_history": {}, "lookback": 5})()
    researcher = DummyResearcher()
    logger = logging.getLogger("runtime-bootstrap-test")

    warm_start_workers(
        bundle,
        workers=[worker],
        researchers=[researcher],
        data_dir=tmp_path,
        logger=logger,
    )

    assert worker.price_history == {}
    assert researcher.preloaded == []


def test_start_watchdog_without_running_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _bundle_for_tests()

    monkeypatch.setattr(
        asyncio,
        "get_running_loop",
        lambda: (_ for _ in ()).throw(RuntimeError("no loop")),
    )

    runtime_state = _DummyRuntimeState()
    notifier = _DummyNotifier()

    watchdog = start_watchdog(bundle, runtime_state, notifier)
    assert watchdog is not None
    watchdog.stop()
