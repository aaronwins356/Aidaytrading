"""Runtime bootstrap for the AI day-trading desk."""

from __future__ import annotations

import concurrent.futures
import copy
import signal
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from desk.config import CONFIG_PATH, load_config
from desk.services import (
    BrokerCCXT,
    EventLogger,
    ExecutionEngine,
    FeedHandler,
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
    """Coordinates services to run the live/paper trading loop."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config_path = config_path or CONFIG_PATH
        self.config = load_config(self.config_path)
        self.state = RuntimeState()
        self.logger = EventLogger()
        self.learner = Learner()
        telemetry_cfg = self.config.get("telemetry", {})
        self.telemetry = TelemetryClient(
            telemetry_cfg.get("endpoint"),
            flush_interval=float(telemetry_cfg.get("flush_interval", 1.0)),
            max_backoff=float(telemetry_cfg.get("max_backoff", 30.0)),
        )

        settings = self.config.get("settings", {})
        risk_cfg = self.config.get("risk", {})
        portfolio_cfg = self.config.get("portfolio", {})
        paper_params = {
            "fee_bps": float(settings.get("paper_fee_bps", 10.0)),
            "slippage_bps": float(settings.get("paper_slippage_bps", 5.0)),
            "partial_fill_probability": float(
                settings.get("paper_partial_fill_probability", 0.1)
            ),
            "min_fill_ratio": float(settings.get("paper_min_fill_ratio", 0.6)),
            "funding_rate_hourly": float(settings.get("paper_funding_rate_hourly", 0.0)),
        }
        feed_workers = settings.get("feed_workers")
        try:
            feed_workers = int(feed_workers) if feed_workers else None
        except (TypeError, ValueError):
            feed_workers = None

        self.broker = BrokerCCXT(
            mode=settings.get("mode", "paper"),
            exchange_name=settings.get("exchange", "kraken"),
            api_key=settings.get("api_key", ""),
            api_secret=settings.get("api_secret", ""),
            starting_balance=float(settings.get("balance", 1_000.0)),
            telemetry=self.telemetry,
            paper_params=paper_params,
        )
        self.feed = FeedHandler(
            self.broker,
            timeframe=settings.get("timeframe", "1m"),
            lookback=int(settings.get("lookback", 250)),
            max_workers=feed_workers,
        )
        self.risk_engine = RiskEngine(
            daily_dd=risk_cfg.get("daily_dd"),
            weekly_dd=risk_cfg.get("weekly_dd"),
            default_stop_pct=float(
                risk_cfg.get("stop_loss_pct", risk_cfg.get("trade_stop_loss", 0.02))
            ),
            max_concurrent=int(risk_cfg.get("max_concurrent", 8)),
            halt_on_dd=bool(risk_cfg.get("halt_on_dd", True)),
            trapdoor_pct=float(risk_cfg.get("trapdoor_pct", 0.02)),
        )
        self.executor = ExecutionEngine(
            self.broker,
            self.logger,
            risk_cfg,
            telemetry=self.telemetry,
        )
        self.portfolio = PortfolioManager(
            min_weight=float(portfolio_cfg.get("min_weight", 0.01)),
            max_weight=float(portfolio_cfg.get("max_weight", 0.25)),
            epsilon=float(portfolio_cfg.get("epsilon", 0.1)),
            cooldown_minutes=float(portfolio_cfg.get("cooldown_minutes", 15)),
        )
        self.workers = self._load_workers()

        self.executor.reconcile(self.workers, self.feed, portfolio=self.portfolio)

        self.loop_delay = float(settings.get("loop_delay", 60))
        self.warmup = int(settings.get("warmup_candles", 10))

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

    def run(self) -> None:
        if not self.workers:
            print("[RUNTIME] No workers configured. Exiting.")
            return

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        symbols = {worker.symbol for worker in self.workers}
        risk_cfg = self.config.get("risk", {})
        base_risk = float(risk_cfg.get("fixed_risk_usd", 50.0))

        print(f"[RUNTIME] Starting loop with {len(self.workers)} workers")

        while self.state.running:
            snapshot = self.feed.snapshot(symbols)
            intents = []

            def _evaluate(worker):
                candles = snapshot.get(worker.symbol, [])
                if not candles:
                    return None
                worker.state["candles"] = candles
                if len(candles) < self.warmup:
                    return None
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
                        intent = future.result()
                        if intent:
                            intents.append(intent)

            if intents:
                intents.sort(key=lambda intent: intent.score, reverse=True)
                allocations = self.portfolio.allocate([intent.worker for intent in intents])
                all_positions = [pos for positions in self.executor.open_positions.values() for pos in positions]

                for intent in intents:
                    if not self.portfolio.eligible(intent.worker.name):
                        continue
                    risk_budget = base_risk * intent.worker.allocation * allocations.get(intent.worker.name, 0.0)
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

                        retrain_every = int(risk_cfg.get("retrain_every", 25))
                        if retrain_every > 0 and worker.state.get("trades", 0) % retrain_every == 0:
                            history = self.learner.load_trade_history(worker.name)
                            self.learner.retrain_worker(worker, history)

            balance = self.broker.balance()
            equity = 0.0
            if isinstance(balance, dict):
                equity = float(balance.get("USD", balance.get("total", 0.0)) or 0.0)
            self.logger.write_equity(equity)
            self.telemetry.record_equity(equity)
            self.risk_engine.check_account(equity)
            if self.risk_engine.halted:
                print("[RUNTIME] Risk engine halted trading. Exiting loop.")
                self.state.running = False

            time.sleep(self.loop_delay)

        print("[RUNTIME] Shutdown complete.")
        self.telemetry.close()

