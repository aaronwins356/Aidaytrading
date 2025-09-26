from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import yaml

from ai_trader import streamlit_app


def _create_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE trades (
                timestamp TEXT,
                worker TEXT,
                symbol TEXT,
                side TEXT,
                cash_spent REAL,
                entry_price REAL,
                exit_price REAL,
                pnl_percent REAL,
                pnl_usd REAL,
                win_loss TEXT,
                reason TEXT,
                metadata_json TEXT
            )
            """
        )
        now = datetime.utcnow()
        trades = [
            (
                (now - timedelta(minutes=30)).isoformat(),
                "Momentum Scout",
                "BTC/USD",
                "buy",
                100.0,
                25000.0,
                25250.0,
                1.0,
                100.0,
                "win",
                "entry",
                json.dumps({"mode": "paper"}),
            ),
            (
                (now - timedelta(minutes=10)).isoformat(),
                "Mean Reverter",
                "ETH/USD",
                "sell",
                150.0,
                1800.0,
                1780.0,
                -1.1,
                -16.5,
                "loss",
                "exit",
                json.dumps({"mode": "paper"}),
            ),
        ]
        conn.executemany(
            """
            INSERT INTO trades (
                timestamp, worker, symbol, side, cash_spent, entry_price, exit_price,
                pnl_percent, pnl_usd, win_loss, reason, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            trades,
        )
        conn.execute(
            """
            CREATE TABLE equity_curve (
                timestamp TEXT,
                equity REAL,
                pnl_percent REAL,
                pnl_usd REAL
            )
            """
        )
        equity_rows = [
            ((now - timedelta(days=2)).isoformat(), 10000.0, 0.0, 0.0),
            ((now - timedelta(days=1)).isoformat(), 10100.0, 1.0, 100.0),
            (now.isoformat(), 10050.0, -0.5, -50.0),
        ]
        conn.executemany(
            "INSERT INTO equity_curve(timestamp, equity, pnl_percent, pnl_usd) VALUES (?, ?, ?, ?)",
            equity_rows,
        )
        conn.execute(
            """
            CREATE TABLE account_snapshots (
                timestamp TEXT,
                equity REAL,
                balances_json TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO account_snapshots(timestamp, equity, balances_json) VALUES (?, ?, ?)",
            (now.isoformat(), 10050.0, json.dumps({"USD": 2000.0, "BTC": 0.25})),
        )
        conn.commit()


class DummyContext:
    def __enter__(self) -> "DummyContext":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class DummyColumn:
    def metric(self, *_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - UI stub
        return None


class DummyStreamlit:
    def __init__(self) -> None:
        self.session_state: dict[str, Any] = {}
        self.sidebar = self

    def set_page_config(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def markdown(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def caption(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def header(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def subheader(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def columns(self, count: int) -> Iterable[DummyColumn]:
        return [DummyColumn() for _ in range(count)]

    def plotly_chart(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def tabs(self, labels: Iterable[str]) -> list[DummyContext]:
        return [DummyContext() for _ in labels]

    def expander(self, *_args: Any, **_kwargs: Any) -> DummyContext:
        return DummyContext()

    def selectbox(
        self, _label: str, options: Iterable[Any], index: int = 0, **_kwargs: Any
    ) -> Any:
        choices = list(options)
        if not choices:
            return None
        index = max(0, min(index, len(choices) - 1))
        return choices[index]

    def number_input(self, _label: str, value: Any = 0, **_kwargs: Any) -> Any:
        return value

    def button(self, *_args: Any, **_kwargs: Any) -> bool:
        return False

    def multiselect(
        self, _label: str, options: Iterable[Any], default: Iterable[Any] | None = None
    ):
        return list(default) if default is not None else list(options)

    def date_input(self, _label: str, value: Any):
        return value

    def dataframe(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def download_button(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def info(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def form(self, *_args: Any, **_kwargs: Any) -> DummyContext:
        return DummyContext()

    def slider(self, _label: str, **kwargs: Any) -> Any:
        return kwargs.get("value")

    def toggle(self, _label: str, value: Any = False, **_kwargs: Any) -> Any:
        return value

    def form_submit_button(self, *_args: Any, **_kwargs: Any) -> bool:
        return False

    def checkbox(self, _label: str, value: bool = False, **_kwargs: Any) -> bool:
        return value

    def success(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def metric(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def test_streamlit_dashboard_smoke(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "trades.db"
    _create_db(db_path)
    config_path = tmp_path / "config.yaml"
    config = {
        "trading": {"paper_starting_equity": 10000.0},
        "risk": {"risk_per_trade": 0.02, "max_drawdown_percent": 20.0, "min_stop_buffer": 0.01},
        "workers": {
            "definitions": {
                "momentum": {
                    "module": "ai_trader.workers.momentum.MomentumWorker",
                    "display_name": "Momentum",
                    "enabled": True,
                },
                "ml_ensemble": {
                    "module": "ai_trader.workers.ml_ensemble_worker.EnsembleMLWorker",
                    "display_name": "ML Ensemble",
                    "enabled": False,
                },
            }
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    monkeypatch.setattr(streamlit_app, "DB_PATH", db_path, raising=False)
    monkeypatch.setattr(streamlit_app, "DATA_DIR", tmp_path, raising=False)
    monkeypatch.setattr(streamlit_app, "TRADES_JSON_PATH", tmp_path / "trades.json", raising=False)
    monkeypatch.setattr(streamlit_app, "EQUITY_JSON_PATH", tmp_path / "equity.json", raising=False)
    monkeypatch.setattr(streamlit_app, "CONFIG_PATH", config_path, raising=False)

    dummy_st = DummyStreamlit()
    monkeypatch.setattr(streamlit_app, "st", dummy_st, raising=False)
    monkeypatch.setattr(
        streamlit_app, "cache_data", lambda *args, **kwargs: (lambda func: func), raising=False
    )

    # clear any cached data that may have been initialised with the original paths
    for fn in (
        streamlit_app.load_trades,
        streamlit_app.load_equity_curve,
        streamlit_app.load_latest_account_snapshot,
        streamlit_app.load_control_flags,
        streamlit_app.load_config,
    ):
        clear = getattr(fn, "clear", None)
        if callable(clear):
            clear()

    streamlit_app.render_dashboard()
