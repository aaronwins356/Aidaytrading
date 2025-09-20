import time
import signal
import sys
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
    global running
    print("[MAIN] Gracefully shutting down...")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main():
    global running

    # Load initial config
    config = load_config()
    settings = config.get("settings", {})
    risk_cfg = config.get("risk", {})
    workers_cfg = config.get("workers", [])

    warmup_candles = settings.get("warmup_candles", 10)

    broker = BrokerCCXT(settings)
    logger = EventLogger()
    learner = Learner()
    risk = RiskGuard(risk_cfg)
    executor = Executor(broker, logger, config)

    # Initialize workers
    workers = []
    for wcfg in workers_cfg:
        workers.append(Worker(wcfg["name"], wcfg["symbol"], wcfg["strategy"], wcfg))

    print(f"[MAIN] Loaded {len(workers)} workers")

    while running:
        try:
            for w in workers:
                # keep candle history in worker state
                if "candles" not in w.state:
                    w.state["candles"] = []

                # fetch just 1 new candle (latest only)
                raw_candle = broker.fetch_ohlcv(w.symbol, "1m", 1)[-1]
                norm_candle = normalize_ohlcv([raw_candle])[0]
                w.state["candles"].append(norm_candle)

                candles = w.state["candles"]

                # warm-up logging
                if len(candles) < warmup_candles:
                    print(f"[MAIN] {w.name} warming up ({len(candles)}/{warmup_candles} candles)")
                    continue

                if len(candles) == warmup_candles:
                    print(f"[MAIN] {w.name} ready – trading enabled!")

                # --- normal trading after warm-up ---
                signal_out = w.generate_signal(candles)

                if signal_out:
                    side, qty, price = signal_out
                    weight = getattr(w, "allocation", w.params.get("allocation", 0.1))
                    trade = executor.open_trade(
                        w, side, qty, price, risk_cfg["fixed_risk_usd"] * weight
                    )
                    if trade:
                        learner.observe(trade)

                # check open trades for exits
                for t in executor.get_open_trades(w.symbol):
                    exit_reason, exit_price, pnl = w.check_exit(candles, t)
                    if exit_reason:
                        executor.close_trade(t, exit_price, exit_reason, pnl)

            time.sleep(settings.get("loop_delay", 60))

        except Exception as e:
            print(f"[MAIN] Fatal loop error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()

