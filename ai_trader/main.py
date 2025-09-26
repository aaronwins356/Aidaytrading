"""Runtime entry point for the AI trading bot."""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import signal
import sqlite3
from collections import deque
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from ai_trader.broker.kraken_client import KrakenClient
from ai_trader.broker.websocket_manager import KrakenWebsocketManager
from ai_trader.services.configuration import normalize_config, read_config_file
from ai_trader.services.equity import EquityEngine
from ai_trader.services.logging import configure_logging, get_logger
from ai_trader.services.ml import MLService
from ai_trader.services.risk import RiskManager
from ai_trader.services.schema import ALL_TABLES
from ai_trader.services.trade_engine import TradeEngine
from ai_trader.services.trade_log import MemoryTradeLog, TradeLog
from ai_trader.services.worker_loader import WorkerLoader
from ai_trader.workers.researcher import MarketResearchWorker

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
CONFIG_LIVE_PATH = Path(__file__).resolve().parent / "config.live.yaml"
DATA_DIR = Path(__file__).resolve().parent / "data"
DB_PATH = DATA_DIR / "trades.db"


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and normalise runtime configuration from YAML sources."""

    logger = get_logger(__name__)
    base_config = read_config_file(config_path)
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


def _normalize_symbol(symbol: object) -> str | None:
    """Return a broker-friendly ``BASE/QUOTE`` pair or ``None`` if invalid."""

    if symbol is None:
        return None
    text = str(symbol).strip().upper()
    if not text or "/" not in text:
        return None
    base, quote = text.split("/", 1)
    if base == "XBT":
        base = "BTC"
    return f"{base}/{quote}"


def _collect_all_symbols(config: Dict[str, Any]) -> list[str]:
    """Gather the union of symbols referenced across the configuration."""

    collected: list[str] = []
    seen: set[str] = set()

    def _ingest(candidate: object) -> None:
        if candidate is None:
            return
        if isinstance(candidate, (list, tuple, set)):
            for item in candidate:
                _ingest(item)
            return
        if isinstance(candidate, dict):
            _ingest(candidate.get("symbols"))
            return
        normalised = _normalize_symbol(candidate)
        if normalised and normalised not in seen:
            seen.add(normalised)
            collected.append(normalised)

    trading_cfg = config.get("trading", {})
    _ingest(trading_cfg.get("symbols"))

    worker_cfg = config.get("workers", {})
    definitions = worker_cfg.get("definitions") if isinstance(worker_cfg, dict) else None
    if isinstance(definitions, dict):
        for definition in definitions.values():
            if isinstance(definition, dict):
                _ingest(definition.get("symbols"))

    researcher_cfg = config.get("researcher")
    if isinstance(researcher_cfg, dict):
        _ingest(researcher_cfg.get("symbols"))

    return collected


def _cache_path_for_symbol(symbol: str) -> Path:
    """Return the expected cache path for a trading symbol."""

    sanitized = symbol.replace("/", "_").replace("-", "_").lower()
    return DATA_DIR / f"{sanitized}.csv"


def _load_cached_candles(
    symbol: str,
    logger: Logger,
    *,
    limit: int = 500,
) -> list[dict[str, float]]:
    """Load cached OHLCV candles from disk if present."""

    path = _cache_path_for_symbol(symbol)
    if not path.exists():
        return []
    candles: list[dict[str, float]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row:
                    continue
                normalized = {str(key).strip().lower(): value for key, value in row.items() if key}
                try:
                    candle = {
                        "open": float(normalized["open"]),
                        "high": float(normalized["high"]),
                        "low": float(normalized["low"]),
                        "close": float(normalized["close"]),
                        "volume": float(normalized.get("volume", 0.0)),
                    }
                except (KeyError, TypeError, ValueError):
                    logger.debug("Skipping malformed cache row for %s: %s", symbol, row)
                    continue
                if "timestamp" in normalized:
                    try:
                        candle["timestamp"] = float(normalized["timestamp"])
                    except (TypeError, ValueError):
                        logger.debug(
                            "Invalid timestamp in cache for %s: %s", symbol, normalized["timestamp"]
                        )
                candles.append(candle)
    except OSError as exc:
        logger.warning("Unable to read cache for %s at %s: %s", symbol, path, exc)
        return []
    if not candles:
        return []
    if limit and len(candles) > limit:
        candles = candles[-limit:]
    logger.info(
        "Loaded %d cached candles for %s from %s", len(candles), symbol, path
    )
    return candles


def _seed_worker_histories(
    workers: Iterable[object],
    cached_candles: Dict[str, list[dict[str, float]]],
) -> None:
    """Populate worker price history deques using cached candles."""

    for worker in workers:
        history_map = getattr(worker, "price_history", None)
        lookback = getattr(worker, "lookback", None)
        if history_map is None or lookback is None:
            continue
        for symbol, candles in cached_candles.items():
            series = history_map.setdefault(symbol, deque(maxlen=lookback))
            closes = [float(candle.get("close", 0.0)) for candle in candles]
            max_len = series.maxlen or len(closes)
            for close in closes[-max_len:]:
                series.append(float(close))


def _warm_start_researchers(
    researchers: Iterable[object],
    cached_candles: Dict[str, list[dict[str, float]]],
) -> None:
    """Seed MarketResearchWorker instances with cached candles."""

    for researcher in researchers:
        if isinstance(researcher, MarketResearchWorker):
            for symbol, candles in cached_candles.items():
                researcher.preload_candles(symbol, candles)


def _validate_strategy_topology(
    workers: Sequence[object],
    researchers: Sequence[object],
    configured_symbols: Iterable[str],
) -> None:
    """Ensure each market has two long-only workers and one researcher."""

    logger = get_logger(__name__)
    symbols = [str(symbol).upper() for symbol in configured_symbols or []]
    if not symbols:
        logger.error("No trading symbols configured – unable to validate strategy topology.")
        raise SystemExit(1)

    expected_worker_count = len(symbols) * 2
    strategy_workers = [
        worker for worker in workers if not getattr(worker, "is_researcher", False)
    ]
    if len(strategy_workers) != expected_worker_count:
        logger.error(
            "Strategy stack requires exactly %d workers (%d symbols × 2). Found %d.",
            expected_worker_count,
            len(symbols),
            len(strategy_workers),
        )
        raise SystemExit(1)

    assignment: Dict[str, list[str]] = {symbol: [] for symbol in symbols}
    for worker in strategy_workers:
        worker_symbols = [str(sym).upper() for sym in getattr(worker, "symbols", [])]
        if len(worker_symbols) != 1:
            logger.error(
                "Worker %s must focus on exactly one symbol but is configured for %s.",
                getattr(worker, "name", worker.__class__.__name__),
                ", ".join(worker_symbols) or "no symbols",
            )
            raise SystemExit(1)
        symbol = worker_symbols[0]
        if symbol not in assignment:
            logger.error(
                "Worker %s references unsupported symbol %s.",
                getattr(worker, "name", worker.__class__.__name__),
                symbol,
            )
            raise SystemExit(1)
        if not getattr(worker, "long_only", False):
            logger.error(
                "Worker %s must declare long_only=True to comply with the long-only mandate.",
                getattr(worker, "name", worker.__class__.__name__),
            )
            raise SystemExit(1)
        assignment[symbol].append(getattr(worker, "name", worker.__class__.__name__))

    missing_assignments = {
        symbol: names for symbol, names in assignment.items() if len(names) != 2
    }
    if missing_assignments:
        for symbol, names in missing_assignments.items():
            logger.error(
                "Symbol %s requires exactly two strategy workers but resolved %d (%s).",
                symbol,
                len(names),
                ", ".join(names) or "none",
            )
        raise SystemExit(1)

    researcher_map: Dict[str, list[str]] = {symbol: [] for symbol in symbols}
    for researcher in researchers:
        researcher_symbols = [
            str(sym).upper() for sym in getattr(researcher, "symbols", [])
        ]
        if len(researcher_symbols) != 1:
            logger.error(
                "Researcher %s must track exactly one symbol but is configured for %s.",
                getattr(researcher, "name", researcher.__class__.__name__),
                ", ".join(researcher_symbols) or "no symbols",
            )
            raise SystemExit(1)
        symbol = researcher_symbols[0]
        if symbol not in researcher_map:
            logger.error(
                "Researcher %s references unsupported symbol %s.",
                getattr(researcher, "name", researcher.__class__.__name__),
                symbol,
            )
            raise SystemExit(1)
        researcher_map[symbol].append(
            getattr(researcher, "name", researcher.__class__.__name__)
        )

    missing_researchers = {
        symbol: names for symbol, names in researcher_map.items() if len(names) != 1
    }
    if missing_researchers:
        for symbol, names in missing_researchers.items():
            logger.error(
                "Symbol %s requires one dedicated research bot but resolved %d (%s).",
                symbol,
                len(names),
                ", ".join(names) or "none",
            )
        raise SystemExit(1)

    logger.info(
        "Validated strategy topology: %d long-only workers and %d research bots across %d symbols.",
        len(strategy_workers),
        len(researchers),
        len(symbols),
    )


def _validate_startup(
    engine: TradeEngine,
    workers: Sequence[object],
    researchers: Sequence[object],
    config: Dict[str, Any],
    ml_service: MLService,
    *,
    dryrun: bool = False,
) -> None:
    """Ensure critical services and configuration are ready before trading."""

    logger = get_logger(__name__)

    _validate_strategy_topology(workers, researchers, config.get("trading", {}).get("symbols", []))

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

    if not dryrun:
        _ensure_sqlite_schema(logger)

    if not dryrun:
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
    logger.info("Trading mode: %s", trading_mode.upper())
    if trading_mode == "live":
        api_key = os.getenv("KRAKEN_API_KEY", "").strip()
        api_secret = os.getenv("KRAKEN_API_SECRET", "").strip()
        if not api_key or not api_secret:
            logger.error(
                "Live trading requires KRAKEN_API_KEY and KRAKEN_API_SECRET environment variables."
            )
            raise SystemExit(1)

    if ml_workers and not dryrun:
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

    try:
        with sqlite3.connect(DB_PATH) as connection:
            cursor = connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing = {row[0] for row in cursor.fetchall()}
            missing = [name for name in ALL_TABLES if name not in existing]
            for table_name in missing:
                connection.execute(ALL_TABLES[table_name])
            if missing:
                logger.info("Created missing SQLite tables: %s", ", ".join(sorted(missing)))
            connection.commit()
    except sqlite3.Error as exc:
        logger.error("Failed to validate SQLite schema: %s", exc)
        raise SystemExit(1) from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Day Trading Bot")
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Enable dry-run mode (no database writes or live orders)",
    )
    parser.add_argument(
        "--workers",
        nargs="*",
        help="Subset of worker definition keys to enable",
    )
    parser.add_argument("--risk-per-trade", type=float, dest="risk_per_trade")
    parser.add_argument("--risk-max-drawdown", type=float, dest="risk_max_drawdown")
    parser.add_argument("--risk-daily-loss-limit", type=float, dest="risk_daily_loss")
    parser.add_argument(
        "--risk-min-trades-per-day", type=int, dest="risk_min_trades"
    )
    parser.add_argument(
        "--risk-confidence-relax", type=float, dest="risk_confidence_relax"
    )
    parser.add_argument(
        "--risk-max-open-positions", type=int, dest="risk_max_open_positions"
    )
    parser.add_argument("--ml-window-size", type=int, dest="ml_window_size")
    parser.add_argument("--ml-retrain-interval", type=int, dest="ml_retrain_interval")
    return parser.parse_args(argv)


def _apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> None:
    risk_cfg = config.setdefault("risk", {})
    if args.risk_per_trade is not None:
        risk_cfg["risk_per_trade"] = float(args.risk_per_trade)
    if args.risk_max_drawdown is not None:
        risk_cfg["max_drawdown_percent"] = float(args.risk_max_drawdown)
    if args.risk_daily_loss is not None:
        risk_cfg["daily_loss_limit_percent"] = float(args.risk_daily_loss)
    if args.risk_min_trades is not None:
        risk_cfg["min_trades_per_day"] = int(args.risk_min_trades)
    if args.risk_confidence_relax is not None:
        risk_cfg["confidence_relax_percent"] = float(args.risk_confidence_relax)
    if args.risk_max_open_positions is not None:
        risk_cfg["max_open_positions"] = int(args.risk_max_open_positions)

    if args.ml_window_size is not None or args.ml_retrain_interval is not None:
        worker_cfg = config.setdefault("workers", {}).setdefault("definitions", {})
        ml_worker = worker_cfg.get("ml_ensemble") or worker_cfg.get("ml_ensemble_worker")
        if ml_worker is None:
            ml_worker = worker_cfg.setdefault(
                "ml_ensemble",
                {
                    "module": "ai_trader.workers.ml_ensemble_worker.EnsembleMLWorker",
                    "enabled": True,
                },
            )
        params = ml_worker.setdefault("parameters", {})
        if args.ml_window_size is not None:
            params["window_size"] = int(args.ml_window_size)
        if args.ml_retrain_interval is not None:
            params["retrain_interval"] = int(args.ml_retrain_interval)

    if args.workers:
        worker_cfg = config.setdefault("workers", {})
        definitions = worker_cfg.get("definitions", {})
        enabled_keys = set(args.workers)
        filtered = {
            key: value
            for key, value in definitions.items()
            if key in enabled_keys or value.get("display_name") in enabled_keys
        }
        if not filtered:
            raise SystemExit(
                f"No worker definitions match CLI override: {', '.join(args.workers)}"
            )
        worker_cfg["definitions"] = filtered


async def start_bot(args: argparse.Namespace) -> None:
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    _apply_cli_overrides(config, args)
    configure_logging()
    logger = get_logger(__name__)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    symbols = _collect_all_symbols(config)
    if not symbols:
        logger.error("No trading symbols configured. Add at least one market pair to config.yaml")
        raise SystemExit(1)
    config.setdefault("trading", {})["symbols"] = symbols
    trading_cfg = config.get("trading", {})
    risk_cfg = config.get("risk", {})
    worker_cfg = config.get("workers", {})
    logger.info("Tracking markets: %s", ", ".join(symbols))

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

    dryrun = bool(args.dryrun)
    if dryrun:
        trade_log = MemoryTradeLog()
    else:
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
        learning_rate=float(ml_cfg.get("learning_rate", ml_cfg.get("lr", 0.03))),
        regularization=float(ml_cfg.get("regularization", 0.0005)),
        threshold=float(ml_cfg.get("threshold", 0.25)),
        ensemble=bool(ml_cfg.get("ensemble", True)),
        forest_size=int(ml_cfg.get("forest_size", 10)),
        random_state=int(ml_cfg.get("random_state", 7)),
        warmup_target=int(ml_cfg.get("warmup_target", 200)),
        warmup_samples=int(ml_cfg.get("warmup_samples", 25)),
        confidence_stall_limit=int(ml_cfg.get("confidence_stall_limit", 5)),
    )
    equity_engine = EquityEngine(trade_log, broker.starting_equity)
    risk_manager = RiskManager(risk_cfg)

    symbols = trading_cfg.get("symbols", [])
    websocket_manager = KrakenWebsocketManager(symbols)
    worker_loader = WorkerLoader(worker_cfg, symbols, researcher_config=config.get("researcher"))
    shared_services = {"ml_service": ml_service, "trade_log": trade_log}
    workers, researchers = worker_loader.load(shared_services)

    cached_histories: Dict[str, list[dict[str, float]]] = {}
    for symbol in symbols:
        candles = _load_cached_candles(symbol, logger)
        if candles:
            cached_histories[symbol] = candles
    if cached_histories:
        _warm_start_researchers(researchers, cached_histories)
        non_research_workers = [
            worker for worker in workers if not isinstance(worker, MarketResearchWorker)
        ]
        _seed_worker_histories(non_research_workers, cached_histories)
    else:
        logger.info("No cached candle files found – live warmup will proceed normally.")

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
        min_cash_per_trade=float(trading_cfg.get("min_cash_per_trade", 10.0)),
        max_cash_per_trade=float(trading_cfg.get("max_cash_per_trade", 20.0)),
        trade_confidence_min=float(trading_cfg.get("trade_confidence_min", 0.5)),
        ml_service=ml_service,
    )

    try:
        broker_positions = await broker.fetch_open_positions()
    except Exception as exc:  # noqa: BLE001 - network/broker failures shouldn't abort startup
        logger.warning("Failed to fetch broker open positions during startup: %s", exc)
        broker_positions = []
    else:
        logger.info("Broker returned %d open position(s) for reconciliation", len(broker_positions))
    await engine.rehydrate_open_positions(broker_positions)

    _validate_startup(engine, workers, researchers, config, ml_service, dryrun=dryrun)

    stop_event = asyncio.Event()

    def _shutdown(*_: int) -> None:
        logger.info("Shutdown signal received. Closing bot...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(sig, _shutdown, sig)
        except NotImplementedError:
            signal.signal(sig, lambda *_: asyncio.create_task(engine.stop()))

    if dryrun:
        engine.enable_signal_only_mode("dry-run")
    bot_task = asyncio.create_task(engine.start())
    await stop_event.wait()
    await engine.stop()
    await bot_task


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        asyncio.run(start_bot(args))
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - catch-all for a clean shutdown message
        print(f"Fatal error starting bot: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
