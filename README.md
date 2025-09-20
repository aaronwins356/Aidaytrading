
# 📈 Automated Trading Desk

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CCXT](https://img.shields.io/badge/powered%20by-CCXT-orange.svg)](https://github.com/ccxt/ccxt)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)
[![Status](https://img.shields.io/badge/status-active-success.svg)](#)

An extensible **algorithmic trading system** designed to operate like a self-driving trading desk.  
It continuously ingests market data, evaluates strategies, manages portfolio risk, and executes trades in both **simulation (paper)** and **live** modes.

---

## 🌐 Overview

The trading desk is centered on a **runtime controller** (`main.py`), which boots the environment, loads configuration, and orchestrates services. The architecture is:

- **Modular** – plug in new strategies, swap brokers, extend services easily.  
- **Safe** – risk limits, drawdown guards, and profit locks built in.  
- **Transparent** – JSON + SQLite logging of every trade and equity update.  
- **Interactive** – Streamlit dashboard for live monitoring, reporting, and configuration.  

---

## 🚀 Quick Start

### 1. Install
```bash
git clone <your-private-repo-url>
cd trader
pip install -r requirements.txt
````

### 2. Configure

Edit `desk/configs/config.yaml`. Example:

```yaml
mode: paper
exchange: binance
starting_cash: 10000
workers:
  - rsi_mean_reversion
  - macd_trend
  - bollinger_band
```

### 3. Run

```bash
python main.py
```

### 4. Dashboard

```bash
streamlit run dashboard/app.py
```

Windows shortcuts:

* `run_bot.bat` → start trading
* `launch_dashboard.bat` → open dashboard

---

## 🏗️ System Architecture

```
Config (YAML) → Runtime Controller
    ├── Broker (CCXT wrapper, real or paper)
    ├── Market Feed (candles, cleaning, caching)
    ├── Workers (strategies + ML scoring)
    ├── Portfolio Manager (capital allocation)
    ├── Risk Engine (guardrails, loss limits)
    └── Execution Engine (orders, stops, PnL)
```

* **Broker** – unified CCXT interface for live or simulated trading
* **Market Feed** – OHLCV retrieval, anomaly cleaning, caching
* **Workers** – strategy modules producing buy/sell “intents”
* **Execution Engine** – validates intents, places orders, sets stop/take profit
* **Portfolio Manager** – allocates capital based on risk budgets and recent performance
* **Risk Engine** – drawdown caps, max concurrent positions, profit-locking “trapdoor”
* **Learner** – ML module (random forests today, extensible later)
* **Dashboard** – Streamlit “Control Room” for live equity curves, trade history, and worker performance

---

## 🎯 Strategies Included

* Moving Average Crossovers
* EMA Breakouts
* RSI Mean Reversion
* MACD Trend Following
* Pure Momentum
* ATR Trailing Stops
* Bollinger Bands
* Stochastic Oscillator

Each worker is isolated, outputs standardized trade intents, and can blend ML probability scores.

---

## 🔄 Trading Loop

1. Load config & defaults
2. Pull latest candles from broker
3. Workers generate trade intents
4. Risk filters + ML scoring
5. Portfolio manager allocates position sizes
6. Execution engine submits trades
7. Monitor open positions & exits
8. Log results to JSON + SQLite
9. Retrain models after N trades

---

## 🛡️ Safety Features

* Paper trading sandbox
* Automatic stop-loss & take-profit
* Daily & weekly loss caps
* Max concurrent positions
* Trapdoor equity lock to secure profits

---

## 📊 Dashboard

The Streamlit **Control Room** provides:

* Equity curve visualization
* Open & closed trade ledger
* Per-strategy performance attribution
* Portfolio allocation breakdown
* Editable configurations from browser

---

## 🔮 Roadmap

* Smarter execution (limit, iceberg, slippage control)
* Hyperparameter optimization (Optuna, Ray Tune)
* Advanced ML models (XGBoost, LightGBM, ensembles)
* Cross-asset correlation risk management
* Docker/Kubernetes deployment for distributed trading
* Enhanced dashboard with annotations and attribution reports

---

## ⚖️ Disclaimer

This project is for **educational and research purposes only**.
Cryptocurrency trading is highly volatile and risky. No warranty is provided.
Use entirely at your own discretion.
