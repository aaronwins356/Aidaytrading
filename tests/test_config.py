from pathlib import Path

import pytest

from desk import config


def test_load_config_merges_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
settings:
  mode: live
risk:
  max_concurrent: 2
        """
    )
    loaded = config.load_config(cfg_path)
    assert loaded["settings"]["mode"] == "live"
    assert loaded["risk"]["max_concurrent"] == 2
    # Ensure defaults still present
    assert loaded["portfolio"]["cooldown_minutes"] == config._DEFAULT_CONFIG["portfolio"][
        "cooldown_minutes"
    ]
    assert loaded["ml"]["target_win_rate"] == config._DEFAULT_CONFIG["ml"]["target_win_rate"]


def test_env_overrides_take_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("{}")
    monkeypatch.setenv("DESK_RISK__MAX_CONCURRENT", "3")
    monkeypatch.setenv("DESK_SETTINGS__BALANCE", "2500")
    loaded = config.load_config(cfg_path)
    assert loaded["risk"]["max_concurrent"] == 3
    assert loaded["settings"]["balance"] == 2500


def test_invalid_config_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
settings:
  loop_delay: 0
feed:
  symbols: []
workers:
  - name: bad
    symbol: BTC/USD
    strategy: breakout_strategy
    allocation: 2.0
        """
    )
    with pytest.raises(ValueError) as excinfo:
        config.load_config(cfg_path)
    message = str(excinfo.value)
    assert "loop_delay" in message
    assert "symbols" in message
    assert "allocation" in message
