"""Runtime entry point for the AI trading bot."""

from __future__ import annotations

import asyncio
import signal
from pathlib import Path
from typing import Any, Dict

import yaml

from .broker.kraken_client import KrakenClient
from .broker.websocket_manager import KrakenWebsocketManager
from .services.equity import EquityEngine
from .services.logging import configure_logging, get_logger
from .services.risk import RiskManager
from .services.trade_engine import TradeEngine
from .services.trade_log import TradeLog
from .services.worker_loader import WorkerLoader

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
DATA_DIR = Path(__file__).resolve().parent / "data"
DB_PATH = DATA_DIR / "trades.db"


def load_config() -> Dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


async def start_bot() -> None:
    config = load_config()
    configure_logging()
    logger = get_logger(__name__)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    trading_cfg = config.get("trading", {})
    risk_cfg = config.get("risk", {})
    worker_cfg = config.get("workers", {})

    broker = KrakenClient(
        api_key=config.get("kraken", {}).get("api_key", ""),
        api_secret=config.get("kraken", {}).get("api_secret", ""),
        base_currency=trading_cfg.get("base_currency", "USD"),
        rest_rate_limit=float(config.get("kraken", {}).get("rest_rate_limit", 0.5)),
        paper_trading=bool(trading_cfg.get("paper_trading", True)),
        paper_starting_equity=float(trading_cfg.get("paper_starting_equity", 10000.0)),
    )

    trade_log = TradeLog(DB_PATH)
    equity_engine = EquityEngine(trade_log, broker.starting_equity)
    risk_manager = RiskManager(risk_cfg)

    symbols = trading_cfg.get("symbols", [])
    websocket_manager = KrakenWebsocketManager(symbols)
    worker_loader = WorkerLoader(worker_cfg.get("modules", []), symbols)
    workers = worker_loader.load()

    engine = TradeEngine(
        broker=broker,
        websocket_manager=websocket_manager,
        workers=workers,
        equity_engine=equity_engine,
        risk_manager=risk_manager,
        trade_log=trade_log,
        equity_allocation_percent=float(trading_cfg.get("equity_allocation_percent", 5.0)),
        max_open_positions=int(trading_cfg.get("max_open_positions", 3)),
        refresh_interval=float(worker_cfg.get("refresh_interval_seconds", 30)),
    )

    stop_event = asyncio.Event()

    def _shutdown(*_: int) -> None:
        logger.info("Shutdown signal received. Closing bot...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(sig, _shutdown, sig)
        except NotImplementedError:
            signal.signal(sig, lambda *_: asyncio.create_task(engine.stop()))

    bot_task = asyncio.create_task(engine.start())
    await stop_event.wait()
    await engine.stop()
    await bot_task


def main() -> None:
    asyncio.run(start_bot())


if __name__ == "__main__":
    main()
