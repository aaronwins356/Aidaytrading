"""Run deterministic backtest and compare against stored baseline."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_trader.backtester import run_backtest

DATA_PATH = BASE_DIR / "data" / "btcusdt_2022-01-01_2022-02-28.csv"
BASELINE_PATH = BASE_DIR / "baseline_equity.json"
TOLERANCE = 0.005  # 0.5%

CONFIG: Dict[str, Any] = {
    "trading": {
        "symbols": ["BTC/USDT"],
        "paper_trading": True,
        "paper_starting_equity": 10000.0,
        "equity_allocation_percent": 10.0,
        "max_open_positions": 2,
        "min_cash_per_trade": 10.0,
        "max_cash_per_trade": 500.0,
    },
    "risk": {
        "risk_per_trade": 0.05,
        "max_drawdown_percent": 25.0,
        "daily_loss_limit_percent": 10.0,
        "max_open_positions": 3,
        "min_trades_per_day": {"default": 1},
        "confidence_relax_percent": 0.2,
        "atr_stop_loss_multiplier": 1.8,
        "atr_take_profit_multiplier": 3.2,
        "min_stop_buffer": 0.001,
    },
    "workers": {
        "definitions": {
            "momentum": {
                "module": "ai_trader.workers.momentum.MomentumWorker",
                "enabled": True,
                "symbols": ["BTC/USDT"],
                "parameters": {"fast_window": 5, "slow_window": 12},
            },
            "mean_reversion": {
                "module": "ai_trader.workers.mean_reversion.MeanReversionWorker",
                "enabled": True,
                "symbols": ["BTC/USDT"],
                "parameters": {"window": 20, "threshold": 0.01},
            },
        }
    },
}


async def _run_candidate() -> Dict[str, Any]:
    start = datetime(2022, 1, 1, tzinfo=timezone.utc)
    end = datetime(2022, 2, 28, 23, tzinfo=timezone.utc)
    result = await run_backtest(
        CONFIG,
        "BTC/USDT",
        start,
        end,
        timeframe="1h",
        csv_path=DATA_PATH,
        fee_rate=0.0,
        slippage_bps=0.0,
    )
    return {
        "metrics": result.metrics,
        "equity_curve": [
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
    }


def _load_baseline() -> Dict[str, Any]:
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"Baseline file missing: {BASELINE_PATH}")
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _compare_metrics(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> List[str]:
    messages: List[str] = []
    for key in ("final_equity", "net_profit", "return_percent"):
        base_value = float(baseline.get(key, 0.0))
        cand_value = float(candidate.get(key, 0.0))
        if base_value == 0.0 and cand_value == 0.0:
            continue
        diff = abs(cand_value - base_value)
        limit = abs(base_value) * TOLERANCE + 1e-9
        if diff > limit:
            messages.append(
                f"Metric {key} deviated: baseline={base_value:.4f} candidate={cand_value:.4f} tolerance={limit:.4f}"
            )
    return messages


def _compare_equity_curve(
    baseline: List[Dict[str, Any]], candidate: List[Dict[str, Any]]
) -> List[str]:
    messages: List[str] = []
    if len(baseline) != len(candidate):
        messages.append(
            f"Equity curve length mismatch: baseline={len(baseline)} candidate={len(candidate)}"
        )
        return messages
    for idx, (base_point, cand_point) in enumerate(zip(baseline, candidate)):
        base_equity = float(base_point["equity"])
        cand_equity = float(cand_point["equity"])
        if base_equity == 0.0 and cand_equity == 0.0:
            continue
        diff = abs(cand_equity - base_equity)
        limit = max(abs(base_equity) * TOLERANCE, 1e-6)
        if diff > limit:
            messages.append(
                f"Equity curve divergence at index {idx}: baseline={base_equity:.4f} candidate={cand_equity:.4f} tolerance={limit:.4f}"
            )
            break
    return messages


def main() -> int:
    if not DATA_PATH.exists():
        print(f"Historical dataset missing: {DATA_PATH}", file=sys.stderr)
        return 2
    baseline = _load_baseline()
    candidate = asyncio.run(_run_candidate())

    baseline_metrics: Dict[str, Any] = baseline.get("metrics", {})
    candidate_metrics: Dict[str, Any] = candidate.get("metrics", {})
    baseline_curve: List[Dict[str, Any]] = baseline.get("equity_curve", [])
    candidate_curve: List[Dict[str, Any]] = candidate.get("equity_curve", [])

    errors = _compare_metrics(baseline_metrics, candidate_metrics)
    errors.extend(_compare_equity_curve(baseline_curve, candidate_curve))

    if errors:
        print("Regression comparison FAILED:")
        for error in errors:
            print(f" - {error}")
        return 1

    final_equity = float(candidate_metrics.get("final_equity", 0.0))
    net_profit = float(candidate_metrics.get("net_profit", 0.0))
    print(
        f"Regression comparison passed. Final equity={final_equity:.2f} net_profit={net_profit:.2f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
