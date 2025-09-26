# AI Day Trading Bot

A streamlined, modular crypto trading bot that connects to Kraken, executes multiple strategy workers, logs trades in SQLite, and surfaces live performance in a neon-themed Streamlit dashboard.

## Features

- **Modular workers** – plug-and-play strategies in `ai_trader/workers/` implement a shared interface.
- **Trading engine** – routes trades to Kraken via `ccxt`, sizes positions by equity %, enforces risk limits, and records activity in SQLite.
- **Risk-aware** – configurable drawdown, daily loss, and position-duration guardrails.
- **Equity engine** – tracks live account equity (paper or live), stores history, and calculates total P/L.
- **Dashboard** – Streamlit UI with overview metrics, equity curve, trade log, worker cards, and adjustable risk controls.
- **Paper + live trading** – toggle via `config.yaml`. Paper mode simulates balances; live mode submits real orders.
- **Historical backtesting** – simulate strategies against Kraken OHLCV data or local CSVs with configurable fees/slippage.

## Project Layout

```
ai_trader/
├── main.py             # Runtime loop and orchestration
├── config.yaml         # API keys, trading symbols, risk config
├── broker/             # Kraken REST + WebSocket integrations
├── workers/            # Strategy workers implementing BaseWorker
├── services/           # Trade engine, equity, logging, risk, trade log
├── dashboard/          # Streamlit dashboard
└── data/               # SQLite database and generated artefacts
```


1. **Run the bot**

```bash
.venv\\Scripts\\Activate.ps1
python -m ai_trader.main
```

**Dry run (no SQLite or live orders)**

```bash
python -m ai_trader.main --dryrun --config ai_trader/config.yaml
```

**Enable the ML ensemble worker only**

```bash
python -m ai_trader.main --workers ml_ensemble
```

Risk parameters such as `risk_per_trade`, `max_drawdown_percent`, and the ML ensemble lookback can be overridden from the CLI:

```bash
python -m ai_trader.main --risk-per-trade 0.015 --risk-max-drawdown 12 --ml-window-size 200
```

**Run a historical backtest and export reports**

```bash
python -m ai_trader.main \
  --mode backtest \
  --config ai_trader/config.yaml \
  --pair BTC/USDT \
  --start 2022-01-01 \
  --end 2022-12-31 \
  --timeframe 1h \
  --backtest-fee 0.0026 \
  --backtest-slippage-bps 1.0
```

Backtest runs stream performance metrics to the console and persists CSV/JSON/PNG artefacts under `reports/` (override with `--reports-dir`).

**Shadow backtest alongside live trading**

```bash
python -m ai_trader.main --mode live --parallel-backtest --parallel-backtest-start 2023-01-01 --parallel-backtest-end 2023-03-01
```

When `--parallel-backtest` is enabled, a daemon thread replays historical data with alternate parameters while the live engine keeps trading. Results land in the same `reports/` directory using the label provided by `--parallel-backtest-label` (default `parallel`).

2. **Launch the dashboard** (in a new terminal)

   ```bash
   .venv\\Scripts\\Activate.ps1
   streamlit run ai_trader/streamlit_app.py
   ```

3. **Run automated tests**

   ```bash
   .venv\\Scripts\\Activate.ps1
   pytest
   ```

## Dummy Trade Test

Before pointing at a live account, keep `paper_trading: true` and verify that:

- Workers emit intents in the console (buy/sell logs).
- Paper orders create rows in `ai_trader/data/trades.db`.
- Equity curve updates in the dashboard.

The included paper mode initializes with `$10,000` USD and honours Kraken minimum order sizes to exercise the full pipeline safely.

## Extending the Bot

- **Add a worker:** drop a new module in `ai_trader/workers/`, subclass `BaseWorker`, and append the dotted path to `workers.modules` in `config.yaml`.
- **Leverage ML:** create workers that learn from the SQLite trade history. The logging schema stores entry/exit data ready for pandas modelling.
- **Risk tuning:** adjust sliders in the dashboard, then update `config.yaml` to persist changes for the runtime loop.

## Monitoring & Notifications

### Streamlit operations dashboard

- Start the UI with `streamlit run ai_trader/streamlit_app.py`.
- The **Portfolio Overview** tab surfaces equity, PnL, daily return histogram, and drawdown curve.
- **Trades Log** exposes rich filtering (symbol, strategy, date) with export to CSV.
- **Risk Controls** persists stop-loss, risk-per-trade, and drawdown guardrails directly into `config.yaml` and live control flags.
- **Strategy Manager** toggles rule-based and ML workers on/off while updating runtime control flags.
- Screenshot the dashboard after launching with seeded data for runbooks or documentation (e.g. `streamlit run ...` then capture via your OS screenshot tool).

### Telegram live alerts

Set the following environment variables before launching `python -m ai_trader.main` to enable live notifications:

```bash
export TELEGRAM_TOKEN="<bot token>"
export TELEGRAM_CHAT_ID="<chat id>"
```

With credentials in place the bot will:

- push every executed trade (open and close) with price, PnL, and confidence metrics;
- broadcast heartbeats daily at 02:00, 08:00, 14:00, and 20:00 UTC;
- escalate risk halts and broker/API failures via `send_error` alerts.

## Requirements

- Python 3.11+
- Kraken account (for live trading)
- Streamlit for the dashboard
- Telegram bot token & chat ID for live notifications (optional)

See `requirements.txt` for all Python dependencies.

## Disclaimer

This project is provided for educational purposes only. Trading cryptocurrencies carries significant risk. Use paper trading to validate behaviour and deploy to live markets at your own discretion.
