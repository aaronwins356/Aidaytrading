# 📈 Automated Trading Desk

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CCXT](https://img.shields.io/badge/powered%20by-CCXT-orange.svg)](https://github.com/ccxt/ccxt)
[![Build](https://img.shields.io/badge/tests-pytest-brightgreen.svg)](#)
[![Status](https://img.shields.io/badge/status-active-success.svg)](#)

An extensible **live cryptocurrency trading desk** built around Kraken. The runtime pulls
market data, evaluates strategy workers in parallel, enforces risk limits, executes
orders, and streams telemetry so the system can run unattended. A Streamlit
“Control Room” dashboard reads the same event store for monitoring and light-weight
operations.

---

## 🔍 Project highlights

- **Live-first runtime** – `TradingRuntime` boots a Kraken broker, feed updater, risk
  engine, execution engine, learner, and telemetry pipeline before entering the main
  trading loop. The loop continuously snapshots candles, scores worker intents, routes
  orders, and records results for retraining.【F:main.py†L1-L12】【F:desk/runtime.py†L20-L227】【F:desk/runtime.py†L244-L399】
- **Structured logging & telemetry** – every trade, equity update, and feed event is
  persisted to `desk/logs` as both JSONL and SQLite. Telemetry is dispatched
  asynchronously with retry and optional HTTP publishing.【F:desk/services/logger.py†L1-L119】【F:desk/services/telemetry.py†L1-L134】
- **Concurrent workers with ML feedback** – strategies are isolated modules loaded by
  `Worker`, which handles candle buffering, risk multipliers, ML scoring, and intent
  vetos before forwarding orders to the execution engine.【F:desk/services/worker.py†L1-L186】
- **Streamlit Control Room** – the dashboard app surfaces live equity curves, trade
  ledgers, worker attribution, and editable YAML configuration backed by the same
  SQLite database.【F:desk/apps/dashboard.py†L1-L152】

---

## 🗂️ Repository layout

```
main.py                  # CLI entry point
desk/
  runtime.py             # Runtime orchestrator
  configs/config.yaml    # Default live-trading configuration
  services/              # Broker, execution, feed, risk, learner, telemetry, logging
  strategies/            # Individual trading strategies (momentum, RSI, MACD, ...)
  apps/dashboard.py      # Legacy Streamlit dashboard implementation
  ...
dashboard/               # Streamlit Control Room (new dashboard)
run                      # Helper command wrapper (currently dashboard only)
run_dashboard.sh         # Launches dashboard/app.py inside .venv
requirements.txt         # Python dependencies
```

---

## 🚀 Quick start

1. **Clone & create a virtual environment**

   ```bash
   cd "C:\Users\moe\Desktop\AI Trader"
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt

   ```

2. **Configure credentials and runtime settings**

   Edit `desk/configs/config.yaml` to add your Kraken API key/secret and adjust
   runtime parameters. The default file is configured for live trading with
   five example workers and 1 minute candles.【F:desk/configs/config.yaml†L1-L97】

3. **Run the trading runtime**

   ```bash
   python main.py
   ```

   The runtime validates configuration, seeds historical candles if needed, and
   then enters the live loop. It will exit automatically if the risk engine halts
   trading or when you send SIGINT/SIGTERM (Ctrl+C).【F:desk/runtime.py†L85-L167】【F:desk/runtime.py†L244-L318】

4. **Launch the dashboard (optional)**

   Activate the virtual environment and either use the helper script or call
   Streamlit directly:

   ```bash
   streamlit run dashboard/app.py
   ```

   The dashboard reads `desk/logs/trades.db` and `desk/configs/config.yaml` to
   display performance metrics and allow inline config edits.【F:run_dashboard.sh†L1-L15】【F:dashboard/app.py†L1-L120】

---

## ⚙️ Configuration reference

The YAML configuration drives every subsystem. Key sections in
`desk/configs/config.yaml` include:

- **`settings`** – core runtime switches such as `mode` (must remain `live`), Kraken
  credentials, candle `timeframe`, lookback depth, and loop cadence.【F:desk/configs/config.yaml†L1-L13】
- **`feed`** – candle seeding and subscription options for the feed updater, including
  symbols, timeframe, and historical seed lengths.【F:desk/configs/config.yaml†L15-L26】
- **`risk`** – global guardrails (fixed USD risk, stop-loss %, concurrent limit,
  trapdoor equity lock, ML weight, retrain cadence) and defaults for adaptive
  risk profiles applied to each worker.【F:desk/configs/config.yaml†L28-L51】
- **`portfolio`** – position sizing parameters used by the portfolio manager when
  allocating risk budgets across workers.【F:desk/configs/config.yaml†L53-L57】
- **`telemetry` & `ml`** – optional HTTP endpoint for telemetry delivery and learner
  sizing targets.【F:desk/configs/config.yaml†L59-L64】
- **`workers`** – list of strategy instances, each with a name, symbol, strategy slug,
  allocation weight, strategy parameters, and (optional) custom risk profile overrides.
  Strategies must exist under `desk/strategies/<slug>.py` and expose a `*Strategy`
  class that matches the slug.【F:desk/configs/config.yaml†L66-L97】【F:desk/services/worker.py†L125-L186】

Configuration values are read once during startup, so restart the runtime after
changes. The dashboard writes updates back to the same file.【F:desk/apps/dashboard.py†L64-L80】

---

## 🧠 Runtime lifecycle

1. **Bootstrap** – load configuration, instantiate the Kraken broker, event logger,
   telemetry client, feed updater, execution engine, risk engine, learner, and
   portfolio manager.【F:desk/runtime.py†L20-L227】
2. **Data seeding** – warm the local feed store with the configured seed length and
   start the asynchronous updater thread.【F:desk/runtime.py†L127-L166】
3. **Main loop** – fetch account equity, enforce risk halts, build a candle snapshot,
   evaluate workers concurrently, and sort approved intents by score.【F:desk/runtime.py†L244-L318】
4. **Execution** – allocate risk across eligible intents, compute quantities based on
   stop levels, open positions through the execution engine, and record ML features for
   retraining.【F:desk/runtime.py†L318-L372】
5. **Monitoring & learning** – evaluate exits on every candle update, persist trade
   results, update worker stats, and retrain strategies based on the configured cadence.【F:desk/runtime.py†L372-L412】
6. **Shutdown** – flush telemetry, logger, broker, and feed updater resources.【F:desk/runtime.py†L168-L215】【F:desk/runtime.py†L414-L431】

---

## 🧾 Logging & observability

- **Event store** – JSONL events and `trades.db`/`equity` tables are written under
  `desk/logs`. These logs drive the dashboard and provide an audit trail for every
  trade and feed event.【F:desk/services/logger.py†L14-L119】
- **Telemetry** – the asynchronous telemetry client buffers equity snapshots, latency
  metrics, and trade events. Configure an HTTP endpoint to forward events to your
  observability stack or rely on the in-memory collector during development.【F:desk/services/telemetry.py†L1-L134】
- **Stdout tracing** – the broker, runtime, and logger emit human-readable updates to
  the console for quick inspection.【F:desk/services/broker.py†L32-L120】【F:desk/runtime.py†L244-L318】

---

## 📊 Streamlit Control Room

The Streamlit application in `dashboard/` provides:

- Live equity curve with uptime axis
- Open/closed trade ledger and per-worker attribution
- Config editor with YAML persistence and account reset utilities
- Utility components for analytics, PDF reporting, and theming

It reads directly from the SQLite databases populated by the runtime, so you can run
it alongside the live bot or offline for analysis.【F:dashboard/app.py†L1-L160】

---

## 🧪 Testing

The repository ships with pytest suites that cover configuration loading, broker
fail-safes, runtime orchestration, risk logic, worker behaviour, telemetry, and
shutdown hooks. Run all tests locally before deploying changes:

```bash
pytest
```

The tests assume `PYTHONPATH` includes the project root (activating the virtual
environment or exporting `PYTHONPATH=$(pwd)` achieves this).【F:tests/test_runtime.py†L1-L170】

---

## 🧩 Extending the desk

1. **Add or customise strategies** – drop a new module in `desk/strategies/` with a
   `<Name>Strategy` class and register it in the YAML `workers` list. The `Worker`
   loader infers the class name from the slug and handles candle buffering, ML scoring,
   and adaptive risk multipliers.【F:desk/services/worker.py†L125-L186】
2. **Adjust risk controls** – tune stop-loss defaults, max concurrent positions, trapdoor
   equity locks, and weekly targets under the `risk` section. The runtime recalculates
   per-trade risk budgets based on realised equity and targets.【F:desk/runtime.py†L283-L332】
3. **Hook in telemetry** – point the telemetry client at your collector by providing an
   HTTP endpoint in the configuration or inject a custom publisher when instantiating
   `TradingRuntime`.【F:desk/runtime.py†L50-L72】【F:desk/services/telemetry.py†L26-L58】

---

## ⚖️ Disclaimer

This project is provided for **educational and research purposes only**. Trading digital
assets involves substantial risk, including the potential loss of all invested capital.
Operate the software entirely at your own discretion.
