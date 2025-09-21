"""Runtime bootstrap for the AI day-trading desk."""

from __future__ import annotations

import concurrent.futures
import copy
import signal
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from desk.config import CONFIG_PATH, load_config
from desk.services import (
    KrakenBroker,
    EventLogger,
    ExecutionEngine,
    DashboardRecorder,
    FeedHandler,
    FeedUpdater,
    Learner,
    PortfolioManager,
    RiskEngine,
    TelemetryClient,
    Worker,
)


@dataclass
class RuntimeState:
    running: bool = True


class TradingRuntime:
    """Coordinates services to run the Kraken trading loop in live mode."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config_path = config_path or CONFIG_PATH
        self.config = load_config(self.config_path)
        self.state = RuntimeState()
        self.logger = EventLogger()
        ml_cfg = self.config.get("ml", {})
        self.learner = Learner(
            target_win_rate=float(ml_cfg.get("target_win_rate", 0.58)),
            min_samples=int(ml_cfg.get("min_samples", 120)),
        )
        telemetry_cfg = self.config.get("telemetry", {})
        self.telemetry = TelemetryClient(
            telemetry_cfg.get("endpoint"),
            flush_interval=float(telemetry_cfg.get("flush_interval", 1.0)),
            max_backoff=float(telemetry_cfg.get("max_backoff", 30.0)),
        )
        settings = self.config.get("settings", {})
        feed_cfg = self.config.get("feed", {})
        risk_cfg = self.config.get("risk", {})
        portfolio_cfg = self.config.get("portfolio", {})

        mode = str(settings.get("mode", "live")).strip().lower()
        if mode != "live":
            raise ValueError(
                "settings.mode must be 'live' now that paper trading is disabled"
            )
        # Persist the canonical casing expected by downstream services.
        self.mode = "live"

        self.dashboard = DashboardRecorder(self.mode)

        feed_workers = settings.get("feed_workers")
        try:
            feed_workers = int(feed_workers) if feed_workers else None
        except (TypeError, ValueError):
            feed_workers = None

        api_key = str(settings.get("api_key", ""))
        api_secret = str(settings.get("api_secret", ""))

        if not api_key or not api_secret:
            raise ValueError(
                "Live trading requires non-empty Kraken API credentials."
            )

        request_timeout = float(settings.get("request_timeout", 30.0))
        session_config = settings.get("kraken_session") or {}
        if not isinstance(session_config, dict):
            session_config = {}
        data_seed_cfg = dict(feed_cfg.get("data_seeding") or {})
        feed_timeframe = str(
            feed_cfg.get(
                "timeframe",
                settings.get("timeframe", "1m"),
            )
        )
        feed_symbols_cfg = feed_cfg.get("symbols")
        if feed_symbols_cfg:
            feed_symbols = [str(symbol) for symbol in feed_symbols_cfg]
        else:
            feed_symbols = [
                worker_cfg.get("symbol")
                for worker_cfg in self.config.get("workers", [])
                if worker_cfg.get("symbol")
            ]
        if not feed_symbols:
            feed_symbols = ["BTC/USD"]

        self.feed_updater = FeedUpdater(
            exchange="kraken",
            symbols=feed_symbols,
            timeframe=feed_timeframe,
            mode=self.mode,
            api_key=api_key,
            api_secret=api_secret,
            interval_seconds=float(settings.get("loop_delay", 60)),
            logger=self.logger,
            fallback_exchanges=[],
            seed_config=data_seed_cfg,
        )

        self.broker = KrakenBroker(
            api_key=api_key,
            api_secret=api_secret,
            telemetry=self.telemetry,
            event_logger=self.logger,
            request_timeout=request_timeout,
            session_config=session_config,
        )

        self._services_started = False

        self.feed = FeedHandler(
            self.broker,
            timeframe=settings.get("timeframe", "1m"),
            lookback=int(settings.get("lookback", 250)),
            max_workers=feed_workers,
            local_store=self.feed_updater.store,
        )
        self._fixed_risk_usd = float(risk_cfg.get("fixed_risk_usd", 50.0) or 0.0)
        self._weekly_return_target = float(risk_cfg.get("weekly_return_target", 0.0) or 0.0)
        self._trading_days_per_week = float(risk_cfg.get("trading_days_per_week", 5.0) or 5.0)
        expected_trades_per_day = risk_cfg.get("expected_trades_per_day")
        try:
            self._expected_trades_per_day: Optional[float]
            self._expected_trades_per_day = (
                float(expected_trades_per_day)
                if expected_trades_per_day is not None
                else None
            )
        except (TypeError, ValueError):
            self._expected_trades_per_day = None

        self.risk_engine = RiskEngine(
            daily_dd=risk_cfg.get("daily_dd"),
            weekly_dd=risk_cfg.get("weekly_dd"),
            default_stop_pct=float(
                risk_cfg.get("stop_loss_pct", risk_cfg.get("trade_stop_loss", 0.02))
            ),
            max_concurrent=int(risk_cfg.get("max_concurrent", 8)),
            halt_on_dd=bool(risk_cfg.get("halt_on_dd", True)),
            trapdoor_pct=float(risk_cfg.get("trapdoor_pct", 0.02)),
            max_position_value=(
                float(risk_cfg.get("max_position_value"))
                if risk_cfg.get("max_position_value")
                else None
            ),
        )
        self.executor = ExecutionEngine(
            self.broker,
            self.logger,
            risk_cfg,
            telemetry=self.telemetry,
            dashboard_recorder=self.dashboard,
        )
        self.portfolio = PortfolioManager(
            min_weight=float(portfolio_cfg.get("min_weight", 0.01)),
            max_weight=float(portfolio_cfg.get("max_weight", 0.25)),
            epsilon=float(portfolio_cfg.get("epsilon", 0.1)),
            cooldown_minutes=float(portfolio_cfg.get("cooldown_minutes", 15)),
        )
        self._ml_weight = float(risk_cfg.get("ml_weight", 0.5))
        self.workers = self._load_workers()

        self.executor.reconcile(self.workers, self.feed, portfolio=self.portfolio)

        self.loop_delay = float(settings.get("loop_delay", 60))
        self.warmup = int(settings.get("warmup_candles", 10))
        self._warm_progress: Dict[str, int] = {}
        self._retrain_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self._risk_config = risk_cfg

    def start_services(self) -> None:
        if self._services_started:
            return
        try:
            self.feed_updater.seed_if_needed()
        except Exception as exc:  # pragma: no cover - defensive guard
            print(f"[RUNTIME] Feed seeding failed: {exc}")
        self.feed_updater.start()
        self._services_started = True

    def _shutdown(self) -> None:
        """Flush and close long-lived resources."""

        try:
            self.executor.close()
        except Exception:
            pass
        try:
            self.logger.close()
        except Exception:
            pass
        try:
            self.telemetry.close()
        except Exception:
            pass
        try:
            self.dashboard.close()
        except Exception:
            pass
        try:
            self.broker.close()
        except Exception:
            pass
        try:
            self.feed_updater.close()
        except Exception:
            pass
        try:
            self._retrain_executor.shutdown(wait=False)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _load_workers(self) -> List[Worker]:
        workers = []
        risk_profile = self.config.get("risk", {}).get("learning_risk", {})
        for cfg in self.config.get("workers", []):
            try:
                worker_cfg = copy.deepcopy(cfg)
                if risk_profile and not worker_cfg.get("risk_profile"):
                    worker_cfg["risk_profile"] = copy.deepcopy(risk_profile)
                worker = Worker(
                    name=worker_cfg["name"],
                    symbol=worker_cfg["symbol"],
                    strategy=worker_cfg["strategy"],
                    params=worker_cfg,
                    logger=self.logger,
                    learner=self.learner,
                    risk_engine=self.risk_engine,
                    ml_weight=self._ml_weight,
                )
                workers.append(worker)
            except Exception as exc:  # pragma: no cover - defensive bootstrap
                print(f"[RUNTIME] Failed to load worker {cfg.get('name')}: {exc}")
        return workers

    # ------------------------------------------------------------------
    def _handle_signal(self, signum, frame) -> None:  # pragma: no cover - OS signal
        print(f"[RUNTIME] Received signal {signum}. Shutting down...")
        self.state.running = False

    @staticmethod
    def _sanitize_features(features: Dict[str, float]) -> Dict[str, float]:
        cleaned: Dict[str, float] = {}
        for key, value in features.items():
            try:
                cleaned[key] = float(value)
            except (TypeError, ValueError):
                continue
        return cleaned

    def _resolve_expected_trades_per_day(self) -> float:
        if self._expected_trades_per_day and self._expected_trades_per_day > 0:
            return float(self._expected_trades_per_day)
        return float(max(len(self.workers), 1))

    def _compute_base_risk(self, equity: float) -> float:
        """Derive the per-trade risk budget from account equity."""

        if equity <= 0:
            return max(self._fixed_risk_usd, 0.0)

        weekly_target = max(self._weekly_return_target, 0.0)
        if weekly_target <= 0:
            return max(self._fixed_risk_usd, 0.0)

        trading_days = max(self._trading_days_per_week, 1.0)
        expected_trades = max(self._resolve_expected_trades_per_day(), 1.0)

        daily_target = (1.0 + weekly_target) ** (1.0 / trading_days) - 1.0
        if daily_target <= 0:
            return max(self._fixed_risk_usd, 0.0)

        base_risk = equity * (daily_target / expected_trades)
        if base_risk <= 0:
            return max(self._fixed_risk_usd, 0.0)
        return base_risk

    def run(self) -> None:
        if not self.workers:
            print("[RUNTIME] No workers configured. Exiting.")
            self._shutdown()
            return

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        self.start_services()

        symbols = {worker.symbol for worker in self.workers}
        print(f"[RUNTIME] Starting loop with {len(self.workers)} workers")

        while self.state.running:
            try:
                equity = float(self.broker.account_equity())
            except Exception as exc:
                print(f"[RUNTIME] Failed to fetch account equity: {exc}")
                equity = 0.0
            self.logger.write_equity(equity)
            self.dashboard.record_equity(equity)
            self.telemetry.record_equity(equity)
            self.risk_engine.check_account(equity)
            if self.risk_engine.halted:
                print("[RUNTIME] Risk engine halted trading. Exiting loop.")
                self.state.running = False
                break

            base_risk = self._compute_base_risk(equity)

            snapshot = self.feed.snapshot(symbols)
            intents = []

            def _evaluate(worker):
                candles = snapshot.get(worker.symbol, [])
                if not candles:
                    return None
                worker.state["candles"] = candles
                if len(candles) < self.warmup:
                    progress = self._warm_progress.get(worker.name)
                    if progress != len(candles):
                        print(
                            f"[Worker] {worker.name} warming: ({len(candles)}/{self.warmup} candles)"
                        )
                        self._warm_progress[worker.name] = len(candles)
                    return None
                if worker.name in self._warm_progress:
                    self._warm_progress.pop(worker.name, None)
                intent = worker.build_intent(base_risk * worker.allocation)
                if not intent or not intent.approved:
                    return None
                return intent

            if self.workers:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(len(self.workers), 8)
                ) as pool:
                    futures = {pool.submit(_evaluate, worker): worker for worker in self.workers}
                    for future in concurrent.futures.as_completed(futures):
                        worker = futures[future]
                        try:
                            intent = future.result()
                        except Exception as exc:
                            print(f"[RUNTIME] Worker {worker.name} evaluation failed: {exc}")
                            continue
                        if intent:
                            intents.append(intent)

            if intents:
                intents.sort(key=lambda intent: intent.score, reverse=True)
                eligible_intents, allocations = self._allocate_eligible_intents(intents)
                all_positions = [
                    pos
                    for positions in self.executor.open_positions.values()
                    for pos in positions
                ]

                for intent in eligible_intents:
                    risk_budget = (
                        base_risk
                        * intent.worker.allocation
                        * allocations.get(intent.worker.name, 0.0)
                    )
                    if risk_budget <= 0:
                        continue
                    if not self.risk_engine.enforce_position_limits(all_positions):
                        break
                    qty = intent.worker.compute_quantity(
                        intent.price,
                        risk_budget,
                        stop_loss=intent.stop_loss,
                        side=intent.side,
                    )
                    if qty <= 0:
                        continue
                    plan_metadata = intent.plan_metadata or {}

                    trade = self.executor.open_position(
                        intent.worker,
                        intent.symbol,
                        intent.side,
                        qty,
                        intent.price,
                        risk_budget,
                        stop_loss=intent.stop_loss,
                        take_profit=intent.take_profit,
                        max_hold_minutes=intent.max_hold_minutes,
                        metadata={
                            "features": intent.features,
                            "score": intent.score,
                            "ml_edge": intent.ml_score,
                            "plan": plan_metadata,
                        },
                    )
                    if trade:
                        self.portfolio.mark_routed(intent.worker.name)
                        all_positions.append(trade)
                        self.learner.observe(
                            {
                                "worker": intent.worker.name,
                                "symbol": intent.symbol,
                                "side": intent.side,
                                "qty": qty,
                                "entry_price": intent.price,
                                "timestamp": time.time(),
                                "features": intent.features,
                                "ml_edge": intent.ml_score,
                                "score": intent.score,
                            }
                        )

            for symbol, candles in snapshot.items():
                closed = self.executor.evaluate_exits(symbol, candles)
                for trade, pnl, reason in closed:
                    worker = next((w for w in self.workers if w.name == trade.worker), None)
                    if worker:
                        worker.record_trade(pnl)
                        self.portfolio.update_stats(worker.name, pnl)
                        row = {
                            **self._sanitize_features(trade.metadata.get("features", {})),
                            "entry_price": trade.entry_price,
                            "stop_loss": trade.stop_loss,
                            "take_profit": trade.take_profit,
                            "qty": trade.qty,
                            "score": trade.metadata.get("score", 0.0),
                            "ml_edge": trade.metadata.get("ml_edge", 0.5),
                            "hold_time": time.time() - trade.opened_at,
                            "pnl": pnl,
                        }
                        self.learner.record_result(worker.name, row)

                        retrain_every = int(self._risk_config.get("retrain_every", 25))
                        if (
                            retrain_every > 0
                            and worker.state.get("trades", 0) % retrain_every == 0
                        ):
                            history = self.learner.load_trade_history(worker.name)

                            def _do_retrain():
                                try:
                                    self.learner.retrain_worker(worker, history)
                                except Exception as exc:
                                    print(
                                        f"[RUNTIME] Retrain failed for {worker.name}: {exc}"
                                    )

                            self._retrain_executor.submit(_do_retrain)

            time.sleep(self.loop_delay)

        print("[RUNTIME] Shutdown complete.")
        self._shutdown()

    def _allocate_eligible_intents(
        self, intents: List["Intent"]
    ) -> Tuple[List["Intent"], Dict[str, float]]:
        eligible: List["Intent"] = []
        for intent in intents:
            if self.portfolio.eligible(intent.worker.name):
                eligible.append(intent)
        if not eligible:
            return [], {}

        allocations = self.portfolio.allocate([intent.worker for intent in eligible])
        filtered = {
            intent.worker.name: float(allocations.get(intent.worker.name, 0.0))
            for intent in eligible
        }
        total = sum(filtered.values())
        if total <= 0:
            weight = 1.0 / len(filtered)
            return eligible, {name: weight for name in filtered}

        normalized = {name: value / total for name, value in filtered.items()}
        return eligible, normalized

