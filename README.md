# AI Day Trading Bot

A modular crypto trading bot for Kraken that runs multiple strategy workers, enforces risk guardrails, records activity in SQLite, and surfaces live performance through a Streamlit dashboard.

## Key Features

- **Unified configuration** – all runtime settings (exchange credentials, risk, workers, notifications, dashboard) live in [`configs/config.yaml`](configs/config.yaml).
- **Paper + live trading** – flip `trading.paper_trading` to simulate fills or submit orders to Kraken with the same runtime.
- **Streamlit operations hub** – monitor equity, trades, and worker status with live control flags plus on-demand backtesting.
- **FastAPI monitoring API** – expose bot status, profit snapshots, and risk controls for external integrations.
- **Extensible strategies** – drop new worker classes in `ai_trader/workers/` and wire them up via configuration.
- **SQLite-first persistence** – capture trades, equity history, and control flags for audits and analytics.
- **Walk-forward validation** – ML models continuously self-check using rolling validation windows and surface accuracy/reward metrics via the API and dashboard.

## Project Layout

```
ai_trader/
├── main.py             # Runtime loop and orchestration
├── broker/             # Kraken REST + WebSocket integrations
├── workers/            # Strategy workers implementing BaseWorker
├── services/           # Trade engine, equity, logging, risk, trade log
├── dashboard/          # Streamlit dashboard helpers
├── streamlit_app.py    # Streamlit UI entry point
└── data/               # SQLite database and generated artefacts
configs/
└── config.yaml         # Unified runtime configuration
```

## Configuration

The bot loads settings from [`configs/config.yaml`](configs/config.yaml). Key sections include:

- `exchange`: Kraken REST keys and rate limits. Leave `api_key`/`api_secret` empty in git and provide real credentials via environment variables (`KRAKEN_API_KEY`, `KRAKEN_API_SECRET`) for production.
- `trading`: base currency, enabled symbols, sizing rules, and the critical `paper_trading` flag. When `paper_trading: true`, the broker simulates fills and you can run without API keys.
- `risk`: global guardrails such as `max_drawdown_percent`, `daily_loss_limit_percent`, ATR stops, and minimum trades per day. `min_trades_per_day` now accepts a mapping so you can set a default plus per-symbol overrides (e.g. require 10 BTC trades before relaxing ML confidence but allow ETH to relax after 5).
- `workers`: strategy definitions with per-worker risk overrides.
- `workers`: strategy definitions with per-worker risk overrides. ML-centric workers support additional parameters like `shadow_mode` to run in signal-only mode until validation metrics are satisfactory.
- `researcher` / `ml`: market feature extraction and ML ensemble configuration.
- `notifications.telegram`: bot token, chat ID, and heartbeat cadence. Credentials can also be supplied via `TELEGRAM_TOKEN` / `TELEGRAM_CHAT_ID`.
- `streamlit` and `api`: dashboard refresh cadence, theme, and FastAPI host/port overrides.
- `watchdog_timeout_seconds`: optional override for the runtime stall watchdog (defaults to 60 seconds).

Symbols are normalised so Kraken aliases like `XBT/USD` resolve to `BTC/USD`. Update `trading.symbols` and the corresponding worker symbol lists together.

## One-click Launchers (Windows)

Double-click any of the provided batch files from Explorer:

- `trade.bat` – activate `.venv` and start the trading loop with `configs\config.yaml`.
- `dash.bat` – launch the Streamlit dashboard.
- `api.bat` – start the FastAPI monitoring service.

Each script pauses at the end so you can review logs before closing the console window.

## Containerised deployment

Build a reproducible runtime image and orchestrate the trading bot, FastAPI service, and Streamlit dashboard with Docker Compose:

```bash
docker compose up --build
```

The compose file provisions three services that share the SQLite database via the `trader_state` volume:

- **`trader`** – default entrypoint (`python -m ai_trader.main --mode live`).
- **`api`** – FastAPI monitoring service exposed on port `8000`.
- **`dashboard`** – Streamlit dashboard exposed on port `8501`.

SQLite artefacts and runtime state are mounted at `/app/ai_trader/data`, so all services read and write the same ledger. Override image commands or environment variables per service as needed for your deployment target.

## Dependency tiers

Dependencies are split to keep production images light:

- `requirements-core.txt` – FastAPI, Streamlit, ccxt, and other runtime essentials.
- `requirements-ml.txt` – optional heavy ML stack (`torch`, `scikit-learn`, `river`).
- `requirements-dev.txt` – formatting, linting, pytest, and HTTP client tooling.

Install only the layers you need, or pull everything for full test coverage:

```bash
pip install -r requirements-core.txt      # minimal runtime footprint
pip install -r requirements-ml.txt        # enable ML workers & backtests
pip install -r requirements-dev.txt       # linting + pytest utilities
```

## Manual Commands

### Run the trading loop

```bash
python -m ai_trader.main --mode live --config configs/config.yaml
```

- Paper mode (`trading.paper_trading: true`) skips real orders and uses the built-in simulator.
- Set `trading.paper_trading: false` (and provide API keys) to trade live funds.

Override risk knobs on the fly:

```bash
python -m ai_trader.main --mode live --config configs/config.yaml \
  --risk-per-trade 0.015 --risk-max-drawdown 12
```

### Historical backtests (CLI)

```bash
python -m ai_trader.main --mode backtest \
  --config configs/config.yaml \
  --pair BTC/USD \
  --start 2024-01-01 \
  --end 2024-03-01 \
  --timeframe 1h \
  --backtest-fee 0.0026 \
  --backtest-slippage-bps 1.0
```

Reports (CSV/JSON/PNG) are written to `reports/` by default.

### Monitoring API

```bash
python -m ai_trader.main --mode api --config configs/config.yaml
```

Use `AI_TRADER_API_HOST`, `AI_TRADER_API_PORT`, and `AI_TRADER_API_RELOAD` to customise the server without editing YAML.

The API exposes `/ml-metrics` to retrieve the latest walk-forward validation snapshot (accuracy, reward, support, and confidence) for each symbol.

Operational metrics for Prometheus are exported via `/metrics` (text format). Exposed gauges include:

- `trader_equity_total`
- `trader_open_positions`
- `trader_max_drawdown_percent`
- `trader_watchdog_last_update_age_seconds`
- `trader_ml_validation_accuracy`
- `trader_websocket_reconnect_total`

Point Prometheus or any compatible scraper at `http://<api-host>:8000/metrics` to populate dashboards.

### Runtime watchdog & telemetry

- **Watchdog** – a background thread inspects the shared runtime state and raises a warning if no updates occur within `watchdog_timeout_seconds`. When a stall is detected the `/status` endpoint returns `"runtime_degraded": true` and a Telegram alert (`⚠️ Bot stalled: ...`) is dispatched.
- **Startup heartbeat** – Telegram notifications emit an immediate startup summary (mode, equity, open positions) before the scheduled heartbeat cycle starts.
- **/monitoring endpoint** – `GET /monitoring` returns the 50 most recent monitoring events (watchdog, notifier, WebSocket reconnects) in structured JSON. Example payload:

```json
{
  "count": 3,
  "events": [
    {
      "timestamp": "2024-05-15T18:42:10.123456+00:00",
      "event_type": "websocket_reconnect",
      "severity": "WARNING",
      "message": "Kraken WebSocket reconnect scheduled",
      "metadata": {"attempt": 2, "delay_seconds": 3.5, "error": "connection dropped"}
    }
  ]
}
```

## Streamlit Dashboard Backtesting

Launch the dashboard:

```bash
streamlit run ai_trader/streamlit_app.py
```

The sidebar now includes:

- **ML Validation** – live accuracy/reward/support metrics aggregated from the walk-forward validator. Values update automatically as fresh labels arrive.
- **Quick Backtest** – run ad-hoc historical replays without interrupting the live loop.

For backtests:

1. Select a symbol from `trading.symbols`.
2. Choose how many days to replay (default 30).
3. Click **Run Backtest**.

An asynchronous task spins up a shared `Backtester` instance on a background thread so the live trading loop remains responsive. Results stream directly into the dashboard:

- **Equity curve** rendered via Plotly.
- **Recent trades** table with ISO timestamps.
- **Key metrics** (`PnL %`, `Win rate`, `Max drawdown`) surfaced as `st.metric` widgets.

No artefacts are written to disk; everything is rendered in-memory inside Streamlit.

## Regression backtests

Deterministic seven-day backtests guard against strategy drift. Baselines live in [`tests/regression/baselines/`](tests/regression/baselines/) and cover BTC/USDT and ETH/USDT pairs with fixed seeds. Run the suite locally with:

```bash
pytest tests/regression -q
```

CI executes the same tests and fails if equity, PnL, or the equity curve deviates by more than 1% from the stored baselines.

## QA Pipeline

The CI workflow (`.github/workflows/qa.yml`) installs runtime + dev dependencies, then runs:

1. `docker build --target runtime -t aidaytrading:ci .`
2. `flake8 . --max-line-length=100 --exclude=.venv`
3. `black --check .`
4. `pytest -q --maxfail=1 --disable-warnings --timeout=20`

Run the same steps locally before opening a pull request:

```bash
python -m pip install --upgrade pip
pip install -r requirements-core.txt
pip install -r requirements-ml.txt
pip install -r requirements-dev.txt
docker build --target runtime -t aidaytrading:ci .
flake8 . --max-line-length=100 --exclude=.venv
black --check .
pytest -q --maxfail=1 --disable-warnings --timeout=20
```

## Requirements

- Python 3.11+
- Kraken account (live trading)
- Optional: Telegram bot token + chat ID for notifications

See [`requirements-core.txt`](requirements-core.txt) and [`requirements-ml.txt`](requirements-ml.txt) for pinned runtime dependencies.

## Disclaimer

This project is for educational purposes only. Cryptocurrency trading is risky—use paper mode to validate behaviour and proceed with real funds at your own discretion.
