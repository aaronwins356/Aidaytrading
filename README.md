# AI Day Trading Bot

A streamlined, modular crypto trading bot that connects to Kraken, executes multiple strategy workers, logs trades in SQLite, and surfaces live performance in a neon-themed Streamlit dashboard.

## Features

- **Modular workers** – plug-and-play strategies in `ai_trader/workers/` implement a shared interface.
- **Trading engine** – routes trades to Kraken via `ccxt`, sizes positions by equity %, enforces risk limits, and records activity in SQLite.
- **Risk-aware** – configurable drawdown, daily loss, and position-duration guardrails.
- **Equity engine** – tracks live account equity (paper or live), stores history, and calculates total P/L.
- **Dashboard** – Streamlit UI with overview metrics, equity curve, trade log, worker cards, and adjustable risk controls.
- **Paper + live trading** – toggle via `config.yaml`. Paper mode simulates balances; live mode submits real orders.

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

## Quick Start

1. **Clone & install**

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure**

   Edit `ai_trader/config.yaml` with Kraken API keys, symbols, and risk preferences. Set `paper_trading: false` only when ready for live trading.

3. **Run the bot**

   ```bash
   python -m ai_trader.main
   ```

4. **Launch the dashboard** (in a new terminal)

   ```bash
   streamlit run ai_trader/dashboard/app.py
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

## Requirements

- Python 3.11+
- Kraken account (for live trading)
- Streamlit for the dashboard

See `requirements.txt` for all Python dependencies.

## Disclaimer

This project is provided for educational purposes only. Trading cryptocurrencies carries significant risk. Use paper trading to validate behaviour and deploy to live markets at your own discretion.
