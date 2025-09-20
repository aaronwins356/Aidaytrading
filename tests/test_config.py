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


def test_env_overrides_take_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("{}")
    monkeypatch.setenv("DESK_RISK__MAX_CONCURRENT", "3")
    monkeypatch.setenv("DESK_SETTINGS__BALANCE", "2500")
    loaded = config.load_config(cfg_path)
    assert loaded["risk"]["max_concurrent"] == 3
    assert loaded["settings"]["balance"] == 2500
