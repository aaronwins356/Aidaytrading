"""Runtime entry point for the AI trading bot."""

from __future__ import annotations

import asyncio
import signal
import sqlite3
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

import yaml

from ai_trader.broker.kraken_client import KrakenClient
from ai_trader.broker.websocket_manager import KrakenWebsocketManager
from ai_trader.services.equity import EquityEngine
from ai_trader.services.logging import configure_logging, get_logger
from ai_trader.services.ml import MLService
from ai_trader.services.risk import RiskManager
from ai_trader.services.trade_engine import TradeEngine
from ai_trader.services.trade_log import TradeLog
from ai_trader.services.worker_loader import WorkerLoader

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
        allow_shorting=bool(trading_cfg.get("allow_shorting", False)),
    )

    trade_log = TradeLog(DB_PATH)
    ml_cfg = config.get("ml", {})
    feature_keys = ml_cfg.get(
        "feature_keys",
        [
            "momentum_1",
            "momentum_3",
            "momentum_5",
            "momentum_10",
            "rolling_volatility",
            "atr",
            "volume_delta",
            "volume_ratio",
            "volume_ratio_3",
            "volume_ratio_10",
            "body_pct",
            "upper_wick_pct",
            "lower_wick_pct",
            "wick_close_ratio",
            "range_pct",
            "ema_fast",
            "ema_slow",
            "macd",
            "macd_hist",
            "rsi",
            "zscore",
            "close_to_high",
            "close_to_low",
        ],
    )
    ml_service = MLService(
        db_path=DB_PATH,
        feature_keys=feature_keys,
        learning_rate=float(ml_cfg.get("learning_rate", 0.03)),
        regularization=float(ml_cfg.get("regularization", 0.0005)),
        threshold=float(ml_cfg.get("threshold", 0.7)),
        ensemble=bool(ml_cfg.get("ensemble_enabled", True)),
        forest_size=int(ml_cfg.get("ensemble_trees", 15)),
        random_state=int(ml_cfg.get("random_state", 7)),
    )
    equity_engine = EquityEngine(trade_log, broker.starting_equity)
    risk_manager = RiskManager(risk_cfg)

    symbols = trading_cfg.get("symbols", [])
    websocket_manager = KrakenWebsocketManager(symbols)
    worker_loader = WorkerLoader(worker_cfg, symbols, researcher_config=config.get("researcher"))
    shared_services = {"ml_service": ml_service, "trade_log": trade_log}
    workers, researchers = worker_loader.load(shared_services)

    _validate_startup(
        workers,
        researchers,
        ml_service,
        symbols,
        logger,
    )

    engine = TradeEngine(
        broker=broker,
        websocket_manager=websocket_manager,
        workers=workers,
        researchers=researchers,
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
    try:
        asyncio.run(start_bot())
    except RuntimeError as exc:
        print(f"Startup validation failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()


def _validate_startup(
    workers: Sequence[object],
    researchers: Sequence[object],
    ml_service: MLService,
    symbols: Iterable[str],
    logger: Logger,
) -> None:
    """Ensure critical dependencies are ready before starting the engine."""

    issues: List[str] = []

    if not researchers:
        issues.append(
            "No MarketResearchWorker instances were loaded. Configure `workers.researcher` to "
            "enable ML feature engineering."
        )

    missing_tables = _missing_tables(DB_PATH, _required_tables())
    if missing_tables:
        issues.append(
            "Missing database tables detected: "
            + ", ".join(sorted(missing_tables))
            + ". Recreate the SQLite schema by running the bot once or clearing the database."
        )

    ml_workers = [worker for worker in workers if getattr(worker, "_ml_service", None)]
    if ml_workers and not ml_service.has_feature_history(symbols):
        issues.append(
            "ML gating is enabled but no engineered features were found in the database. "
            "Allow the researcher to run first or disable ML gating."
        )

    if issues:
        for message in issues:
            logger.error(message)
        raise RuntimeError("see log output for details")

    logger.info("Startup validation passed with %d researcher worker(s).", len(researchers))


def _required_tables() -> Set[str]:
    """Return the set of SQLite tables the bot depends on."""

    return {
        "trades",
        "equity_curve",
        "market_features",
        "account_snapshots",
        "bot_state",
        "control_flags",
        "ml_models_state",
        "ml_predictions",
        "ml_metrics",
        "ml_models",
    }


def _missing_tables(db_path: Path, required: Set[str]) -> Set[str]:
    """Compare required tables against the SQLite schema."""

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing = {row[0] for row in cursor.fetchall()}
    except sqlite3.Error as exc:  # pragma: no cover - defensive logging
        raise RuntimeError(f"Unable to inspect database schema: {exc}") from exc
    return required - existing
