"""Smoke tests ensuring configuration and worker instantiation succeed."""

from __future__ import annotations

from pathlib import Path

from ai_trader.services.configuration import normalize_config, read_config_file
from ai_trader.services.ml import MLService
from ai_trader.services.trade_log import TradeLog
from ai_trader.services.worker_loader import WorkerLoader


def test_worker_loader_smoke(tmp_path) -> None:
    """Ensure the default configuration produces active workers."""

    config_path = Path(__file__).resolve().parents[1] / "ai_trader" / "config.yaml"
    raw_config = read_config_file(config_path)
    config = normalize_config(raw_config)

    db_path = tmp_path / "trades.db"
    trade_log = TradeLog(db_path)
    ml_cfg = config.get("ml", {})
    feature_keys = ml_cfg.get("feature_keys", ["momentum_1", "rsi"])
    ml_service = MLService(
        db_path=db_path,
        feature_keys=feature_keys,
        learning_rate=0.05,
        ensemble=False,
        forest_size=1,
        warmup_target=5,
    )

    loader = WorkerLoader(
        config.get("workers"),
        config.get("trading", {}).get("symbols", []),
        researcher_config=config.get("researcher"),
    )
    workers, researchers = loader.load({"ml_service": ml_service, "trade_log": trade_log})

    assert workers, "Expected at least one trading worker from configuration"
    assert researchers, "Researcher worker should be provisioned for ML services"
