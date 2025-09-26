"""Pytest wrapper for deterministic backtest regression scenarios."""

from __future__ import annotations

import pytest

from tests.regression.compare_regression import (
    RegressionScenario,
    load_scenarios,
    run_regression_scenario,
)

SCENARIOS = load_scenarios()


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda scenario: scenario.name)
def test_regression_scenarios(scenario: RegressionScenario) -> None:
    run_regression_scenario(scenario)
