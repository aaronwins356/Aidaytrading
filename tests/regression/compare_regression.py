"""Deterministic regression backtests validated during CI."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from ai_trader.backtester import run_backtest

BASE_DIR = Path(__file__).resolve().parent
BASELINES_DIR = BASE_DIR / "baselines"
DATA_DIR = BASE_DIR / "data"
DEFAULT_TOLERANCE = 0.01  # 1%


def _parse_datetime(value: str) -> datetime:
    text = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


@dataclass(slots=True)
class RegressionScenario:
    """Configuration bundle describing one deterministic regression test."""

    name: str
    pair: str
    timeframe: str
    start: datetime
    end: datetime
    csv_path: Path
    fee_rate: float
    tolerance: float
    config: Dict[str, Any]
    baseline_metrics: Dict[str, Any]
    baseline_equity: List[Dict[str, Any]]

    @classmethod
    def from_file(cls, path: Path) -> "RegressionScenario":
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload.get("config")
        baseline = payload.get("baseline") or {}
        metrics = baseline.get("metrics", {})
        equity = baseline.get("equity_curve", [])
        tolerance = float(payload.get("tolerance", DEFAULT_TOLERANCE))
        fee_rate = float(payload.get("fee_rate", 0.0))
        return cls(
            name=path.stem,
            pair=str(payload["pair"]),
            timeframe=str(payload.get("timeframe", "1h")),
            start=_parse_datetime(str(payload["start"])),
            end=_parse_datetime(str(payload["end"])),
            csv_path=(path.parent / payload["csv"]).resolve(),
            fee_rate=fee_rate,
            tolerance=tolerance,
            config=dict(config or {}),
            baseline_metrics=dict(metrics),
            baseline_equity=list(equity),
        )


@dataclass(slots=True)
class RegressionResult:
    """Minimal summary returned after executing a regression scenario."""

    metrics: Dict[str, Any]
    equity_curve: List[Dict[str, Any]]


async def _run_backtest_for_scenario(scenario: RegressionScenario) -> RegressionResult:
    result = await run_backtest(
        scenario.config,
        scenario.pair,
        scenario.start,
        scenario.end,
        timeframe=scenario.timeframe,
        csv_path=scenario.csv_path,
        fee_rate=scenario.fee_rate,
        slippage_bps=0.0,
    )
    return RegressionResult(
        metrics=result.metrics,
        equity_curve=[
            {
                "timestamp": (
                    point["timestamp"]
                    if isinstance(point["timestamp"], str)
                    else point["timestamp"].isoformat()
                ),
                "equity": float(point["equity"]),
                "pnl_percent": float(point.get("pnl_percent", 0.0)),
            }
            for point in result.equity_curve
        ],
    )


def _compare_metric_drift(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    tolerance: float,
) -> List[str]:
    messages: List[str] = []
    keys = ("final_equity", "net_profit", "return_percent")
    for key in keys:
        base_value = float(baseline.get(key, 0.0))
        cand_value = float(candidate.get(key, 0.0))
        if base_value == 0.0 and cand_value == 0.0:
            continue
        diff = abs(cand_value - base_value)
        limit = abs(base_value) * tolerance + 1e-9
        if diff > limit:
            messages.append(
                (
                    f"Metric {key} deviated: baseline={base_value:.4f} "
                    f"candidate={cand_value:.4f} tolerance={limit:.4f}"
                )
            )
    return messages


def _compare_equity_curves(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    tolerance: float,
    *,
    extra_margin: float = 0.0,
) -> List[str]:
    messages: List[str] = []
    if len(baseline) != len(candidate):
        messages.append(
            f"Equity curve length mismatch: baseline={len(baseline)} candidate={len(candidate)}"
        )
        return messages
    for idx, (base_point, cand_point) in enumerate(zip(baseline, candidate)):
        base_equity = float(base_point.get("equity", 0.0))
        cand_equity = float(cand_point.get("equity", 0.0))
        if base_equity == 0.0 and cand_equity == 0.0:
            continue
        diff = abs(cand_equity - base_equity)
        effective_tolerance = tolerance + max(extra_margin, 0.0)
        limit = max(abs(base_equity) * effective_tolerance, 1e-6)
        if diff > limit:
            messages.append(
                (
                    "Equity curve divergence at index "
                    f"{idx}: baseline={base_equity:.4f} candidate={cand_equity:.4f} "
                    f"tolerance={limit:.4f}"
                )
            )
            break
    return messages


def run_regression_scenario(scenario: RegressionScenario) -> None:
    """Execute one regression scenario and raise ``AssertionError`` on drift."""

    if not scenario.csv_path.exists():
        raise FileNotFoundError(f"Historical dataset missing: {scenario.csv_path}")
    if not scenario.baseline_metrics:
        raise ValueError(f"Baseline metrics missing in scenario: {scenario.name}")

    candidate = asyncio.run(_run_backtest_for_scenario(scenario))
    errors = _compare_metric_drift(scenario.baseline_metrics, candidate.metrics, scenario.tolerance)
    errors.extend(
        _compare_equity_curves(
            scenario.baseline_equity,
            candidate.equity_curve,
            scenario.tolerance,
            extra_margin=scenario.fee_rate,
        )
    )

    if errors:
        error_lines = "\n".join(f" - {msg}" for msg in errors)
        raise AssertionError(f"Regression comparison failed for {scenario.name}:\n{error_lines}")


def load_scenarios() -> List[RegressionScenario]:
    files = sorted(BASELINES_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError("No regression baseline configurations found")
    scenarios: List[RegressionScenario] = []
    for file in files:
        scenarios.append(RegressionScenario.from_file(file))
    return scenarios


def main() -> int:
    scenarios = load_scenarios()
    for scenario in scenarios:
        run_regression_scenario(scenario)
    last = scenarios[-1]
    final_equity = float(last.baseline_metrics.get("final_equity", 0.0))
    net_profit = float(last.baseline_metrics.get("net_profit", 0.0))
    print(
        "Regression comparison passed. Final equity="
        f"{final_equity:.2f} net_profit={net_profit:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
