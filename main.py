import signal
import time
from typing import Dict, List

from core.broker_ccxt import BrokerCCXT
from core.executor import Executor
from core.learner import Learner
from core.logger import EventLogger
from core.riskguard import RiskGuard
from core.worker import Worker
from utils.config_loader import load_config
from utils.data_utils import normalize_ohlcv

running = True


def signal_handler(sig, frame):
    """Flip the run flag so the main loop exits gracefully."""
    global running
    print("[MAIN] Gracefully shutting down...")
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def _init_broker(settings: Dict[str, object]) -> BrokerCCXT:
    return BrokerCCXT(
        mode=str(settings.get("mode", "paper")),
        exchange_name=str(settings.get("exchange", "kraken")),
        api_key=str(settings.get("api_key", "")),
        api_secret=str(settings.get("api_secret", "")),
        starting_balance=float(settings.get("balance", 1000.0)),
    )


def _init_risk_guard(risk_cfg: Dict[str, object]) -> RiskGuard:
    return RiskGuard(
        daily_dd=risk_cfg.get("daily_dd"),
        weekly_dd=risk_cfg.get("weekly_dd"),
        trade_stop_loss=float(risk_cfg.get("trade_stop_loss", 1.0)),
        max_concurrent=int(risk_cfg.get("max_concurrent", 8)),
        halt_on_dd=bool(risk_cfg.get("halt_on_dd", False)),
    )


def main() -> None:
    global running

    config = load_config()
    settings: Dict[str, object] = config.get("settings", {})
    risk_cfg: Dict[str, object] = config.get("risk", {})
    workers_cfg: List[Dict[str, object]] = config.get("workers", [])

    warmup_candles = int(settings.get("warmup_candles", 10))

    broker = _init_broker(settings)
    logger = EventLogger()
    learner = Learner()
    risk = _init_risk_guard(risk_cfg)
    initial_balance = broker.balance()
    if isinstance(initial_balance, dict) and "USD" in initial_balance:
        risk.set_start_equity(float(initial_balance.get("USD", 0.0)))
    else:
        risk.set_start_equity(float(settings.get("balance", 0.0)))
    executor = Executor(broker, logger, config)

    workers: List[Worker] = []
    for wcfg in workers_cfg:
        try:
            worker = Worker(
                wcfg["name"],
                wcfg["symbol"],
                wcfg["strategy"],
                wcfg,
                logger=logger,
                config=config,
            )
            worker.state.setdefault("candles", [])
            workers.append(worker)
        except Exception as exc:  # pragma: no cover - defensive guard
            print(f"[MAIN] Failed to load worker {wcfg.get('name', '?')}: {exc}")

    print(f"[MAIN] Loaded {len(workers)} workers")

    while running:
        try:
            for worker in workers:
                candles: List[Dict[str, float]] = worker.state.setdefault("candles", [])

                try:
                    raw_candles = broker.fetch_ohlcv(worker.symbol, "1m", 1)
                except Exception as fetch_err:  # pragma: no cover - network guard
                    print(
                        f"[MAIN] Failed to fetch candles for {worker.symbol}: {fetch_err}"
                    )
                    continue
                if not raw_candles:
                    continue
                latest = normalize_ohlcv([raw_candles[-1]])
                if not latest:
                    continue
                candle = latest[0]
                if candles and candle["timestamp"] <= candles[-1]["timestamp"]:
                    continue

                candles.append(candle)
                max_history = int(worker.params.get("params", {}).get("max_history", 1000))
                if len(candles) > max_history:
                    del candles[: len(candles) - max_history]

                if len(candles) < warmup_candles:
                    print(
                        f"[MAIN] {worker.name} warming up "
                        f"({len(candles)}/{warmup_candles} candles)"
                    )
                    continue

                if len(candles) == warmup_candles:
                    print(f"[MAIN] {worker.name} ready – trading enabled!")

                order = worker.generate_signal(candles, risk_cfg)
                if order:
                    side, qty, price = order
                    if qty > 0:
                        risk_budget = float(risk_cfg.get("fixed_risk_usd", 0.0)) * worker.allocation
                        if risk_budget <= 0:
                            risk_budget = float(risk_cfg.get("fixed_risk_usd", 0.0))
                        trade = executor.open_trade(worker, side, qty, price, risk_budget)
                        if trade:
                            learner.observe(trade)

                for open_trade in list(executor.get_open_trades(worker.symbol)):
                    exit_reason, exit_price, pnl = worker.check_exit(candles, open_trade)
                    if exit_reason:
                        executor.close_trade(open_trade, exit_price, exit_reason, pnl)

            balance_snapshot = broker.balance()
            if isinstance(balance_snapshot, dict):
                equity = float(balance_snapshot.get("USD", 0.0) or 0.0)
            else:
                equity = float(settings.get("balance", 0.0) or 0.0)
            risk.check(equity)
            if risk.halted:
                print("[MAIN] RiskGuard triggered halt due to drawdown limits")
                running = False

            time.sleep(float(settings.get("loop_delay", 60)))

        except Exception as exc:  # pragma: no cover - defensive guard
            print(f"[MAIN] Fatal loop error: {exc}")
            time.sleep(5)


if __name__ == "__main__":
    main()

