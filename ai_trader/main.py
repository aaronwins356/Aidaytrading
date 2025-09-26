"""Runtime entry point for the AI trading bot."""

from __future__ import annotations

import argparse
import asyncio
import copy
import os
import signal
import sqlite3
import threading
from datetime import datetime, timezone
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from ai_trader.api_service import attach_services, get_runtime_state
from ai_trader.backtester import Backtester, BacktestResult
from ai_trader.broker.websocket_manager import KrakenWebsocketManager
from ai_trader.runtime import (
    create_broker,
    initialise_notifier,
    load_workers,
    prepare_runtime_config,
    start_watchdog,
    warm_start_workers,
)
from ai_trader.services.configuration import normalize_config, read_config_file
from ai_trader.services.equity import EquityEngine
from ai_trader.services.logging import configure_logging, get_logger
from ai_trader.services.ml import MLService
from ai_trader.services.risk import RiskManager
from ai_trader.services.schema import ALL_TABLES
from ai_trader.services.trade_engine import TradeEngine
from ai_trader.services.trade_log import TradeLog
from ai_trader.workers.researcher import MarketResearchWorker

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
DATA_DIR = Path(__file__).resolve().parent / "data"
DB_PATH = DATA_DIR / "trades.db"


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and normalise runtime configuration from YAML sources."""

    logger = get_logger(__name__)
    base_config = read_config_file(config_path)
    if not base_config:
        logger.warning("Configuration file %s was empty; using defaults", config_path)
    normalised = normalize_config(base_config)
    return normalised


def _merge_dicts(
    base: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:  # pragma: no cover
    """Backward-compatible helper retained for legacy imports."""

    merged: Dict[str, Any] = {**base}
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


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
    strategy_workers = [worker for worker in workers if not getattr(worker, "is_researcher", False)]
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

    missing_assignments = {symbol: names for symbol, names in assignment.items() if len(names) != 2}
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
        researcher_symbols = [str(sym).upper() for sym in getattr(researcher, "symbols", [])]
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
        researcher_map[symbol].append(getattr(researcher, "name", researcher.__class__.__name__))

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
    paper_trading: bool = True,
) -> None:
    """Ensure critical services and configuration are ready before trading."""

    logger = get_logger(__name__)

    _validate_strategy_topology(workers, researchers, config.get("trading", {}).get("symbols", []))

    researcher_cfg = config.get("researcher", {})
    if not bool(researcher_cfg.get("enabled", False)):
        logger.error("Researcher worker must remain enabled when ML services are active.")
        raise SystemExit(1)

    ml_workers = [worker for worker in workers if getattr(worker, "_ml_service", None) is not None]
    if ml_workers:
        researchers = list(getattr(engine, "_researchers", []))
        has_researcher = any(
            isinstance(researcher, MarketResearchWorker) for researcher in researchers
        )
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
    trading_mode = "PAPER" if paper_trading else "LIVE"
    logger.info("Trading mode: %s", trading_mode)
    if not paper_trading:
        exchange_cfg = config.get("exchange", {})
        api_key = (
            os.getenv("KRAKEN_API_KEY", "").strip() or str(exchange_cfg.get("api_key", "")).strip()
        )
        api_secret = (
            os.getenv("KRAKEN_API_SECRET", "").strip()
            or str(exchange_cfg.get("api_secret", "")).strip()
        )
        if not api_key or not api_secret:
            logger.error(
                "Live trading requires Kraken API credentials via environment variables or configs.exchange."
            )
            raise SystemExit(1)

    if ml_workers:
        try:
            with sqlite3.connect(DB_PATH) as connection:
                cursor = connection.execute("SELECT COUNT(1) FROM market_features")
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
        "--mode",
        choices=["live", "backtest", "api"],
        default="live",
        help="Execution mode: live trading, API service, or historical backtest",
    )
    parser.add_argument(
        "--workers",
        nargs="*",
        help="Subset of worker definition keys to enable",
    )
    parser.add_argument("--risk-per-trade", type=float, dest="risk_per_trade")
    parser.add_argument("--risk-max-drawdown", type=float, dest="risk_max_drawdown")
    parser.add_argument("--risk-daily-loss-limit", type=float, dest="risk_daily_loss")
    parser.add_argument("--risk-min-trades-per-day", type=int, dest="risk_min_trades")
    parser.add_argument("--risk-confidence-relax", type=float, dest="risk_confidence_relax")
    parser.add_argument("--risk-max-open-positions", type=int, dest="risk_max_open_positions")
    parser.add_argument("--ml-window-size", type=int, dest="ml_window_size")
    parser.add_argument("--ml-retrain-interval", type=int, dest="ml_retrain_interval")
    parser.add_argument("--pair", type=str, help="Trading pair for backtests (e.g. BTC/USDT)")
    parser.add_argument("--start", type=str, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument(
        "--timeframe", type=str, default="1h", help="Backtest timeframe for OHLCV data"
    )
    parser.add_argument(
        "--backtest-csv",
        dest="backtest_csv",
        type=str,
        help="Optional CSV file supplying OHLCV candles for backtesting",
    )
    parser.add_argument(
        "--backtest-fee",
        dest="backtest_fee",
        type=float,
        default=0.0026,
        help="Taker fee rate applied during backtests (default 0.0026)",
    )
    parser.add_argument(
        "--backtest-slippage-bps",
        dest="backtest_slippage_bps",
        type=float,
        default=1.0,
        help="Slippage applied during backtests in basis points",
    )
    parser.add_argument(
        "--reports-dir",
        dest="reports_dir",
        type=str,
        help="Override the reports output directory",
    )
    parser.add_argument(
        "--parallel-backtest",
        action="store_true",
        help="Run a background backtest while live trading",
    )
    parser.add_argument(
        "--parallel-backtest-pair",
        dest="parallel_backtest_pair",
        type=str,
        help="Pair for parallel backtest (defaults to primary pair)",
    )
    parser.add_argument(
        "--parallel-backtest-start",
        dest="parallel_backtest_start",
        type=str,
        help="Start date for parallel backtest",
    )
    parser.add_argument(
        "--parallel-backtest-end",
        dest="parallel_backtest_end",
        type=str,
        help="End date for parallel backtest",
    )
    parser.add_argument(
        "--parallel-backtest-timeframe",
        dest="parallel_backtest_timeframe",
        type=str,
        help="Timeframe for the parallel backtest",
    )
    parser.add_argument(
        "--parallel-backtest-csv",
        dest="parallel_backtest_csv",
        type=str,
        help="CSV source for the parallel backtest",
    )
    parser.add_argument(
        "--parallel-backtest-fee",
        dest="parallel_backtest_fee",
        type=float,
        help="Fee rate for the parallel backtest",
    )
    parser.add_argument(
        "--parallel-backtest-slippage-bps",
        dest="parallel_backtest_slippage",
        type=float,
        help="Slippage basis points for the parallel backtest",
    )
    parser.add_argument(
        "--parallel-backtest-label",
        dest="parallel_backtest_label",
        type=str,
        help="Custom label for parallel backtest report artefacts",
    )
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
            raise SystemExit(f"No worker definitions match CLI override: {', '.join(args.workers)}")
        worker_cfg["definitions"] = filtered


def prepare_config(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    _apply_cli_overrides(config, args)
    return config


async def start_trading(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    logger = get_logger(__name__)
    bundle = prepare_runtime_config(config, data_dir=DATA_DIR, logger=logger)
    runtime_config = bundle.config
    trading_cfg = bundle.trading
    risk_cfg = bundle.risk
    worker_cfg = bundle.workers

    if bundle.paper_mode:
        logger.info("🧪 Running in PAPER trading mode – no real funds at risk.")
    else:
        logger.info(
            "🚀 Running in LIVE trading mode – Kraken orders will be routed to the exchange."
        )

    broker = create_broker(bundle, logger=logger)
    trade_log = TradeLog(DB_PATH)
    notifier = initialise_notifier(bundle, logger=logger)
    ml_cfg = runtime_config.get("ml", {})
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
    runtime_state = get_runtime_state()
    runtime_state.set_base_currency(broker.base_currency)
    runtime_state.set_starting_equity(broker.starting_equity)
    runtime_state.update_risk_settings(risk_manager.config_dict())
    attach_services(trade_log=trade_log, runtime_state=runtime_state, risk_manager=risk_manager)

    websocket_manager = KrakenWebsocketManager(bundle.symbols)
    shared_services = {"ml_service": ml_service, "trade_log": trade_log}
    workers, researchers = load_workers(bundle, shared_services)

    warm_start_workers(bundle, workers, researchers, data_dir=DATA_DIR, logger=logger)

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
        trade_fee_percent=float(trading_cfg.get("trade_fee_percent", 0.0)),
        ml_service=ml_service,
        notifier=notifier,
        runtime_state=runtime_state,
    )

    try:
        broker_positions = await broker.fetch_open_positions()
    except Exception as exc:  # noqa: BLE001 - network/broker failures shouldn't abort startup
        logger.warning("Failed to fetch broker open positions during startup: %s", exc)
        broker_positions = []
    else:
        logger.info("Broker returned %d open position(s) for reconciliation", len(broker_positions))
    await engine.rehydrate_open_positions(broker_positions)
    runtime_state.mark_runtime_update()

    if notifier is not None:
        snapshot = runtime_state.status_snapshot()
        equity_value = snapshot.get("equity") or snapshot.get("balance") or broker.starting_equity
        try:
            equity = float(equity_value)
        except (TypeError, ValueError):
            equity = broker.starting_equity
        open_positions_count = len(snapshot.get("open_positions") or [])
        await notifier.send_startup_heartbeat(
            equity=equity,
            open_positions=open_positions_count,
            mode=bundle.trading_mode.lower(),
            currency=broker.base_currency,
        )
        await notifier.start()

    watchdog = start_watchdog(bundle, runtime_state, notifier)

    _validate_startup(
        engine,
        workers,
        researchers,
        runtime_config,
        ml_service,
        paper_trading=bundle.paper_mode,
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
    try:
        await stop_event.wait()
    finally:
        await engine.stop()
        if notifier is not None:
            await notifier.stop()
        if watchdog is not None:
            watchdog.stop()
    await bot_task


def _resolve_path(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser().resolve()


def run_api_server(config: Dict[str, Any]) -> None:
    """Start the FastAPI service for monitoring and risk controls."""

    runtime_config = copy.deepcopy(config)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    trading_cfg = runtime_config.get("trading", {})
    risk_cfg = runtime_config.get("risk", {})
    runtime_state = get_runtime_state()
    runtime_state.set_base_currency(trading_cfg.get("base_currency", "USD"))
    starting_equity = trading_cfg.get("paper_starting_equity") or trading_cfg.get("starting_equity")
    try:
        if starting_equity is not None:
            runtime_state.set_starting_equity(float(starting_equity))
    except (TypeError, ValueError):
        pass
    risk_manager = RiskManager(risk_cfg)
    trade_log = TradeLog(DB_PATH)
    attach_services(trade_log=trade_log, runtime_state=runtime_state, risk_manager=risk_manager)

    api_cfg = runtime_config.get("api", {})
    host = os.getenv("AI_TRADER_API_HOST") or api_cfg.get("host", "0.0.0.0")
    port_value = os.getenv("AI_TRADER_API_PORT") or str(api_cfg.get("port", 8000))
    try:
        port = int(port_value)
    except ValueError:
        port = 8000
    reload_env = os.getenv("AI_TRADER_API_RELOAD")
    reload_value = reload_env if reload_env is not None else str(api_cfg.get("reload", False))
    reload_flag = str(reload_value).lower() in {"1", "true", "on"}

    logger = get_logger(__name__)
    logger.info("Starting API server on %s:%d", host, port)

    import uvicorn

    from ai_trader.api_service import app as api_app

    uvicorn.run(api_app, host=host, port=port, reload=reload_flag)


def _parse_date(value: str | None, field_name: str) -> datetime:
    if not value:
        raise SystemExit(f"Missing required {field_name} date for backtest")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:  # noqa: B904 - re-raise with friendly hint
        raise SystemExit(f"Invalid {field_name} date '{value}'. Use YYYY-MM-DD format.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed


def run_backtest_cli(args: argparse.Namespace, config: Dict[str, Any]) -> BacktestResult:
    logger = get_logger(__name__)
    runtime_config = copy.deepcopy(config)
    trading_cfg = runtime_config.get("trading", {})
    default_pair = None
    symbols = trading_cfg.get("symbols")
    if isinstance(symbols, (list, tuple)) and symbols:
        default_pair = symbols[0]
    pair = args.pair or default_pair
    if not pair:
        raise SystemExit("Backtest mode requires --pair or a symbol in configuration")
    start = _parse_date(args.start, "start")
    end = _parse_date(args.end, "end")
    if end <= start:
        raise SystemExit("Backtest end date must be after the start date")

    csv_path = _resolve_path(args.backtest_csv)
    reports_dir = _resolve_path(args.reports_dir)
    backtester = Backtester(
        runtime_config,
        pair,
        start,
        end,
        timeframe=args.timeframe or "1h",
        fee_rate=float(args.backtest_fee or 0.0),
        slippage_bps=float(args.backtest_slippage_bps or 0.0),
        csv_path=csv_path,
        reports_dir=reports_dir,
    )
    result = asyncio.run(backtester.run())
    summary_path = result.report_paths.get("summary_json")
    if summary_path:
        logger.info("Backtest summary saved to %s", summary_path)
    return result


def _spawn_parallel_backtest(
    args: argparse.Namespace, config: Dict[str, Any]
) -> threading.Thread | None:
    if not args.parallel_backtest:
        return None
    logger = get_logger(__name__)
    runtime_config = copy.deepcopy(config)
    trading_cfg = runtime_config.get("trading", {})
    default_pair = None
    symbols = trading_cfg.get("symbols")
    if isinstance(symbols, (list, tuple)) and symbols:
        default_pair = symbols[0]
    pair = args.parallel_backtest_pair or args.pair or default_pair
    start_value = args.parallel_backtest_start or args.start
    end_value = args.parallel_backtest_end or args.end
    if not (pair and start_value and end_value):
        logger.warning(
            "Parallel backtest requested but pair/start/end parameters are incomplete. Skipping shadow run."
        )
        return None
    try:
        start = _parse_date(start_value, "parallel backtest start")
        end = _parse_date(end_value, "parallel backtest end")
    except SystemExit as exc:
        logger.warning("Parallel backtest parameter error: %s", exc)
        return None
    if end <= start:
        logger.warning(
            "Parallel backtest end date must be after start date. Skipping background run."
        )
        return None

    timeframe = args.parallel_backtest_timeframe or args.timeframe or "1h"
    csv_path = _resolve_path(args.parallel_backtest_csv)
    reports_dir = _resolve_path(args.reports_dir)
    fee = (
        float(args.parallel_backtest_fee)
        if args.parallel_backtest_fee is not None
        else float(args.backtest_fee or 0.0)
    )
    slippage = (
        float(args.parallel_backtest_slippage)
        if args.parallel_backtest_slippage is not None
        else float(args.backtest_slippage_bps or 0.0)
    )
    label = args.parallel_backtest_label or "parallel"

    def _runner() -> None:
        thread_logger = get_logger(__name__ + ".parallel")
        try:
            backtester = Backtester(
                runtime_config,
                pair,
                start,
                end,
                timeframe=timeframe,
                fee_rate=fee,
                slippage_bps=slippage,
                csv_path=csv_path,
                reports_dir=reports_dir,
                label=label,
            )
            asyncio.run(backtester.run())
            thread_logger.info("Parallel backtest '%s' finished", label)
        except Exception as exc:  # noqa: BLE001 - never crash main loop due to background run
            thread_logger.exception("Parallel backtest '%s' failed: %s", label, exc)

    thread = threading.Thread(target=_runner, name="parallel-backtest", daemon=True)
    thread.start()
    logger.info(
        "Spawned parallel backtest thread for %s (%s to %s)",
        pair,
        start.date(),
        end.date(),
    )
    return thread


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = prepare_config(args)
    configure_logging()
    try:
        if args.mode == "backtest":
            run_backtest_cli(args, config)
        elif args.mode == "api":
            run_api_server(config)
        else:
            _spawn_parallel_backtest(args, config)
            asyncio.run(start_trading(args, config))
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - catch-all for a clean shutdown message
        print(f"Fatal error starting bot: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
