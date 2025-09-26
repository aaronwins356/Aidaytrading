"""Helpers for bootstrapping the live trading runtime."""

from __future__ import annotations

import asyncio
import copy
import csv
import os
from collections import deque
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple

from ai_trader.broker.kraken_client import KrakenClient
from ai_trader.notifier import Notifier
from ai_trader.services.monitoring import get_monitoring_center
from ai_trader.services.worker_loader import WorkerLoader
from ai_trader.services.watchdog import RuntimeWatchdog
from ai_trader.workers.researcher import MarketResearchWorker


@dataclass(frozen=True)
class RuntimeConfigBundle:
    """Container with normalised runtime configuration details."""

    config: Dict[str, Any]
    trading: Dict[str, Any]
    risk: Dict[str, Any]
    workers: Dict[str, Any]
    symbols: list[str]
    paper_mode: bool
    trading_mode: str


def prepare_runtime_config(
    config: Dict[str, Any], *, data_dir: Path, logger: Logger
) -> RuntimeConfigBundle:
    """Normalise configuration and derive common runtime metadata."""

    runtime_config = copy.deepcopy(config)
    data_dir.mkdir(parents=True, exist_ok=True)

    symbols = _collect_all_symbols(runtime_config)
    if not symbols:
        logger.error("No trading symbols configured. Add at least one market pair to config.yaml")
        raise SystemExit(1)
    runtime_config.setdefault("trading", {})["symbols"] = symbols

    trading_cfg = runtime_config.get("trading", {})
    risk_cfg = runtime_config.get("risk", {})
    worker_cfg = runtime_config.get("workers", {})

    paper_mode = bool(trading_cfg.get("paper_trading", True))
    trading_mode = "PAPER" if paper_mode else "LIVE"
    trading_cfg["mode"] = trading_mode.lower()

    logger.info("Tracking markets: %s", ", ".join(symbols))

    return RuntimeConfigBundle(
        config=runtime_config,
        trading=trading_cfg,
        risk=risk_cfg,
        workers=worker_cfg,
        symbols=symbols,
        paper_mode=paper_mode,
        trading_mode=trading_mode,
    )


def create_broker(bundle: RuntimeConfigBundle, *, logger: Logger) -> KrakenClient:
    """Instantiate the Kraken broker client based on configuration and env vars."""

    exchange_cfg = bundle.config.get("exchange", {})
    default_rate_limit = bundle.config.get("kraken", {}).get("rest_rate_limit", 0.5)
    rest_rate_limit = float(exchange_cfg.get("rest_rate_limit", default_rate_limit))

    env_api_key = os.getenv("KRAKEN_API_KEY", "").strip()
    env_api_secret = os.getenv("KRAKEN_API_SECRET", "").strip()

    config_api_key = str(
        exchange_cfg.get("api_key", bundle.config.get("kraken", {}).get("api_key", ""))
    ).strip()
    config_api_secret = str(
        exchange_cfg.get("api_secret", bundle.config.get("kraken", {}).get("api_secret", ""))
    ).strip()

    broker_api_key = env_api_key or config_api_key
    broker_api_secret = env_api_secret or config_api_secret

    if not bundle.paper_mode and (not broker_api_key or not broker_api_secret):
        logger.error(
            "Live trading requires Kraken API credentials via environment variables or configs.exchange."
        )
        raise SystemExit(1)

    return KrakenClient(
        api_key=broker_api_key,
        api_secret=broker_api_secret,
        base_currency=bundle.trading.get("base_currency", "USD"),
        rest_rate_limit=rest_rate_limit,
        paper_trading=bundle.paper_mode,
        paper_starting_equity=float(bundle.trading.get("paper_starting_equity", 10000.0)),
        allow_shorting=bool(bundle.trading.get("allow_shorting", False)),
        fee_rate=float(bundle.trading.get("trade_fee_percent", 0.0)),
    )


def initialise_notifier(bundle: RuntimeConfigBundle, *, logger: Logger) -> Notifier | None:
    """Attempt to build the Telegram notifier, falling back gracefully on failure."""

    monitoring_center = get_monitoring_center()
    monitoring_center.set_runtime_degraded(False, None)

    telegram_cfg = bundle.config.get("notifications", {}).get("telegram", {})
    telegram_token = (
        os.getenv("TELEGRAM_TOKEN", "").strip() or str(telegram_cfg.get("bot_token", "")).strip()
    )
    telegram_chat_id = (
        os.getenv("TELEGRAM_CHAT_ID", "").strip() or str(telegram_cfg.get("chat_id", "")).strip()
    )
    telegram_enabled = bool(telegram_cfg.get("enabled", True))

    if not (telegram_enabled and telegram_token and telegram_chat_id):
        logger.info("Telegram notifier disabled – configure notifications.telegram or env vars")
        return None

    try:
        notifier = Notifier(token=telegram_token, chat_id=telegram_chat_id)
    except Exception as exc:  # noqa: BLE001 - network/setup issues should not abort startup
        logger.warning("Failed to initialise Telegram notifier: %s", exc)
        return None

    logger.info("Telegram notifier enabled for chat %s", telegram_chat_id)
    return notifier


def load_workers(
    bundle: RuntimeConfigBundle,
    shared_services: Dict[str, Any],
) -> Tuple[list[object], list[object]]:
    """Load worker and researcher instances defined in configuration."""

    worker_loader = WorkerLoader(
        bundle.workers, bundle.symbols, researcher_config=bundle.config.get("researcher")
    )
    workers, researchers = worker_loader.load(shared_services)
    return workers, researchers


def warm_start_workers(
    bundle: RuntimeConfigBundle,
    workers: Sequence[object],
    researchers: Sequence[object],
    *,
    data_dir: Path,
    logger: Logger,
) -> None:
    """Seed workers and researchers with cached candle data when available."""

    cached_histories: Dict[str, list[dict[str, float]]] = {}
    for symbol in bundle.symbols:
        candles = _load_cached_candles(symbol, data_dir=data_dir, logger=logger)
        if candles:
            cached_histories[symbol] = candles

    if not cached_histories:
        logger.info("No cached candle files found – live warmup will proceed normally.")
        return

    _warm_start_researchers(researchers, cached_histories)
    non_research_workers = [
        worker for worker in workers if not isinstance(worker, MarketResearchWorker)
    ]
    _seed_worker_histories(non_research_workers, cached_histories)


def start_watchdog(
    bundle: RuntimeConfigBundle,
    runtime_state: Any,
    notifier: Notifier | None,
) -> RuntimeWatchdog | None:
    """Create and start the runtime watchdog if configured."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    watchdog_timeout = float(bundle.config.get("watchdog_timeout_seconds", 60.0))
    watchdog = RuntimeWatchdog(
        runtime_state,
        timeout_seconds=watchdog_timeout,
        alert_callback=notifier.send_watchdog_alert if notifier is not None else None,
        event_loop=loop,
    )
    watchdog.start()
    return watchdog


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


def _normalize_symbol(candidate: object) -> str | None:
    if candidate is None:
        return None
    text = str(candidate).strip()
    if not text:
        return None
    return text.upper()


def _cache_path_for_symbol(symbol: str, *, data_dir: Path) -> Path:
    sanitized = symbol.replace("/", "_").replace("-", "_").lower()
    return data_dir / f"{sanitized}.csv"


def _load_cached_candles(
    symbol: str,
    *,
    data_dir: Path,
    logger: Logger,
    limit: int = 500,
) -> list[dict[str, float]]:
    path = _cache_path_for_symbol(symbol, data_dir=data_dir)
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

    logger.info("Loaded %d cached candles for %s from %s", len(candles), symbol, path)
    return candles


def _seed_worker_histories(
    workers: Iterable[object],
    cached_candles: Dict[str, list[dict[str, float]]],
) -> None:
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
    for researcher in researchers:
        if isinstance(researcher, MarketResearchWorker):
            for symbol, candles in cached_candles.items():
                researcher.preload_candles(symbol, candles)
