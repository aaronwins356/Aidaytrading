from __future__ import annotations

import json
from pathlib import Path

import pytest

from dashboard import data_io


def _dump(model) -> dict:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def test_load_config_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config, raw = data_io.load_config(str(config_path))
    assert config.mode == "Live"
    assert raw == _dump(config)


def test_save_config_validation(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    invalid = {"mode": "INVALID"}
    success, message = data_io.save_config(invalid, str(config_path))
    assert not success
    assert "Live" in message

    valid = _dump(data_io._default_config())
    success, message = data_io.save_config(valid, str(config_path))
    assert success
    assert "successfully" in message


@pytest.mark.parametrize("days", [3])
def test_seed_and_load_sqlite(tmp_path: Path, days: int) -> None:
    db_path = tmp_path / "demo.sqlite"
    data_io.seed_demo_data(db_paths=[str(db_path)], days=days, seed=42)
    trades = data_io.load_trades(str(db_path))
    equity = data_io.load_equity(str(db_path))
    positions = data_io.load_positions(str(db_path))
    ml_scores = data_io.load_ml_scores(str(db_path))

    assert not trades.empty
    assert not equity.empty
    assert "drawdown" not in equity  # raw loader
    assert list(positions.columns) == [
        "position_id",
        "symbol",
        "side",
        "qty",
        "avg_entry",
        "stop",
        "target",
        "unrealized",
        "opened_at",
        "worker",
        "mode",
    ]
    assert not ml_scores.empty


def test_load_logs(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    payload = {"timestamp": "2024-01-01T00:00:00", "level": "info", "message": "hello"}
    (log_dir / "example.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    df = data_io.load_logs(str(log_dir))
    assert not df.empty
    assert df.iloc[0]["level"] == "INFO"


def test_database_health(tmp_path: Path) -> None:
    db_path = tmp_path / "health.sqlite"
    data_io.seed_demo_data(db_paths=[str(db_path)], days=1, seed=1)
    health = data_io.database_health([str(db_path)])
    assert health
    assert health[0].name == "health"
    assert health[0].status in {"ok", "stale"}
