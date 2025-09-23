"""Runtime entry point for the AI trading bot."""

from __future__ import annotations

import asyncio
import os
import signal
import sqlite3
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Sequence

from ai_trader.broker.kraken_client import KrakenClient
from ai_trader.broker.websocket_manager import KrakenWebsocketManager
from ai_trader.services.configuration import normalize_config, read_config_file
from ai_trader.services.equity import EquityEngine
from ai_trader.services.logging import configure_logging, get_logger
from ai_trader.services.ml import MLService
from ai_trader.services.risk import RiskManager
from ai_trader.services.trade_engine import TradeEngine
from ai_trader.services.trade_log import TradeLog
from ai_trader.services.worker_loader import WorkerLoader
from ai_trader.workers.researcher import MarketResearchWorker

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
CONFIG_LIVE_PATH = Path(__file__).resolve().parent / "config.live.yaml"
DATA_DIR = Path(__file__).resolve().parent / "data"
DB_PATH = DATA_DIR / "trades.db"


def load_config() -> Dict[str, Any]:
    """Load and normalise runtime configuration from YAML sources."""

    logger = get_logger(__name__)
    base_config = read_config_file(CONFIG_PATH)
    profile = os.getenv("AI_TRADER_ENV", os.getenv("AI_TRADER_MODE", "")).lower()
    if profile == "live":
        live_overrides = read_config_file(CONFIG_LIVE_PATH)
        if live_overrides:
            logger.info("Applying live configuration overrides from %s", CONFIG_LIVE_PATH)
            base_config = _merge_dicts(base_config, live_overrides)
        else:
            logger.warning(
                "AI_TRADER_ENV=live but %s is missing – proceeding with base configuration",
                CONFIG_LIVE_PATH,
            )
    normalised = normalize_config(base_config)
    return normalised


def _merge_dicts(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overrides`` into ``base`` without mutating inputs."""

    merged: Dict[str, Any] = {**base}
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _validate_startup(
    engine: TradeEngine,
    workers: Sequence[object],
    config: Dict[str, Any],
    ml_service: MLService,
) -> None:
    """Ensure critical services and configuration are ready before trading."""

    logger = get_logger(__name__)

    researcher_cfg = config.get("researcher", {})
    if not bool(researcher_cfg.get("enabled", False)):
        logger.error(
            "Researcher worker must remain enabled when ML services are active."
        )
        raise SystemExit(1)

    ml_workers = [
        worker
        for worker in workers
        if getattr(worker, "_ml_service", None) is not None
    ]
    if ml_workers:
        researchers = list(getattr(engine, "_researchers", []))
        has_researcher = any(isinstance(researcher, MarketResearchWorker) for researcher in researchers)
        if not has_researcher:
            logger.error(
                "ML gating is enabled for %d worker(s) but no MarketResearchWorker was loaded. "
                "Enable a researcher in configuration or disable ML gating to continue.",
                len(ml_workers),
            )
            raise SystemExit(1)
        if not ml_service.has_feature_history():
            logger.warning(
                "ML workers detected but no market feature history found. "
                "Researcher will build the initial dataset before trades execute."
            )

    _ensure_sqlite_schema(logger)

    try:
        with sqlite3.connect(DB_PATH) as connection:
            exists = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='market_features'"
            ).fetchone()
    except sqlite3.Error as exc:
        logger.error("Failed to verify market_features table: %s", exc)
        raise SystemExit(1) from exc
    if exists is None:
        logger.error(
            "Required SQLite table 'market_features' is missing. Run initialisation before starting the bot."
        )
        raise SystemExit(1)

    trading_cfg = config.get("trading", {})
    trading_mode = str(trading_cfg.get("mode", "paper")).lower()
    if trading_mode == "live":
        kraken_cfg = config.get("kraken", {})
        api_key = os.getenv("KRAKEN_API_KEY", str(kraken_cfg.get("api_key", "")).strip())
        api_secret = os.getenv(
            "KRAKEN_API_SECRET", str(kraken_cfg.get("api_secret", "")).strip()
        )
        missing_keys = [
            name
            for name, value in {"api_key": api_key, "api_secret": api_secret}.items()
            if not value or "YOUR_KRAKEN" in value.upper()
        ]
        if missing_keys:
            logger.warning(
                "Live trading mode detected but missing Kraken credential(s): %s. "
                "Populate config.yaml before running with real funds.",
                ", ".join(missing_keys),
            )
            engine.enable_signal_only_mode(
                "Missing Kraken API credentials while in live mode"
            )

    if ml_workers:
        try:
            with sqlite3.connect(DB_PATH) as connection:
                cursor = connection.execute(
                    "SELECT COUNT(1) FROM market_features"
                )
                count = cursor.fetchone()[0]
        except sqlite3.Error as exc:
            logger.error("Failed to inspect market_features table: %s", exc)
            raise SystemExit(1) from exc
        if count == 0:
            logger.warning(
                "Market feature store is empty. Allow the researcher to warm up before expecting ML-driven trades."
            )

    try:
        ml_service.probe()
    except Exception as exc:  # noqa: BLE001 - defensive initialisation guard
        logger.exception("Failed to initialise ML pipeline: %s", exc)
        raise SystemExit(1) from exc

    if ml_service.ensemble_requested and not ml_service.ensemble_available:
        logger.warning("⚠️ ML fallback active – logistic regression only")
    else:
        logger.info("✅ ML ready (forest ensemble active)")

    if ml_service.ensemble_available:
        logger.info("River ensemble backend detected: %s", ml_service.ensemble_backend)

    logger.info("✅ Startup validation passed")


def _ensure_sqlite_schema(logger: Logger) -> None:
    """Create any missing SQLite tables required for the trading runtime."""

    table_statements: Dict[str, str] = {
        "trades": """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                worker TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                cash_spent REAL NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                pnl_percent REAL,
                pnl_usd REAL,
                win_loss TEXT
            )
        """,
        "equity_curve": """
            CREATE TABLE IF NOT EXISTS equity_curve (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                equity REAL NOT NULL,
                pnl_percent REAL NOT NULL,
                pnl_usd REAL NOT NULL
            )
        """,
        "bot_state": """
            CREATE TABLE IF NOT EXISTS bot_state (
                worker TEXT NOT NULL,
                symbol TEXT NOT NULL,
                status TEXT,
                last_signal TEXT,
                indicators_json TEXT,
                risk_json TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(worker, symbol)
            )
        """,
        "market_features": """
            CREATE TABLE IF NOT EXISTS market_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                features_json TEXT NOT NULL,
                label REAL
            )
        """,
        "ml_predictions": """
            CREATE TABLE IF NOT EXISTS ml_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                worker TEXT,
                confidence REAL NOT NULL,
                decision INTEGER NOT NULL,
                threshold REAL NOT NULL
            )
        """,
        "ml_metrics": """
            CREATE TABLE IF NOT EXISTS ml_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                mode TEXT NOT NULL,
                precision REAL,
                recall REAL,
                win_rate REAL,
                support INTEGER
            )
        """,
        "account_snapshots": """
            CREATE TABLE IF NOT EXISTS account_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                equity REAL NOT NULL,
                balances_json TEXT NOT NULL
            )
        """,
        "control_flags": """
            CREATE TABLE IF NOT EXISTS control_flags (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """,
    }

    try:
        with sqlite3.connect(DB_PATH) as connection:
            cursor = connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing = {row[0] for row in cursor.fetchall()}
            missing = [name for name in table_statements if name not in existing]
            for table_name in missing:
                connection.execute(table_statements[table_name])
            if missing:
                logger.info("Created missing SQLite tables: %s", ", ".join(sorted(missing)))
            connection.commit()
    except sqlite3.Error as exc:
        logger.error("Failed to validate SQLite schema: %s", exc)
        raise SystemExit(1) from exc


async def start_bot() -> None:
    config = load_config()
    configure_logging()
    logger = get_logger(__name__)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    trading_cfg = config.get("trading", {})
    risk_cfg = config.get("risk", {})
    worker_cfg = config.get("workers", {})

    trading_mode = str(trading_cfg.get("mode", "paper")).lower()
    live_trading = trading_mode == "live"
    if live_trading:
        logger.info("🚀 Running in LIVE trading mode – Kraken API credentials will be pulled from environment variables.")
    else:
        logger.info("🧪 Running in PAPER trading mode – no real funds at risk.")

    broker = KrakenClient(
        api_key=os.getenv("KRAKEN_API_KEY", config.get("kraken", {}).get("api_key", "")),
        api_secret=os.getenv(
            "KRAKEN_API_SECRET", config.get("kraken", {}).get("api_secret", "")
        ),
        base_currency=trading_cfg.get("base_currency", "USD"),
        rest_rate_limit=float(config.get("kraken", {}).get("rest_rate_limit", 0.5)),
        paper_trading=False if live_trading else bool(trading_cfg.get("paper_trading", True)),
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
        lr=float(ml_cfg.get("lr", 0.03)),
        regularization=float(ml_cfg.get("regularization", 0.0005)),
        threshold=float(ml_cfg.get("threshold", 0.25)),
        ensemble=bool(ml_cfg.get("ensemble", True)),
        forest_size=int(ml_cfg.get("forest_size", 10)),
        random_state=int(ml_cfg.get("random_state", 7)),
        warmup_target=int(ml_cfg.get("warmup_target", 200)),
        warmup_samples=int(ml_cfg.get("warmup_samples", 25)),
    )
    equity_engine = EquityEngine(trade_log, broker.starting_equity)
    risk_manager = RiskManager(risk_cfg)

    symbols = trading_cfg.get("symbols", [])
    websocket_manager = KrakenWebsocketManager(symbols)
    worker_loader = WorkerLoader(worker_cfg, symbols, researcher_config=config.get("researcher"))
    shared_services = {"ml_service": ml_service, "trade_log": trade_log}
    workers, researchers = worker_loader.load(shared_services)

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
        paper_trading=broker.is_paper_trading,
    )

    _validate_startup(engine, workers, config, ml_service)

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
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - catch-all for a clean shutdown message
        print(f"Fatal error starting bot: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
