"""Historical backtesting utilities for the AI Day Trading bot."""

from __future__ import annotations

import asyncio
import copy
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

import matplotlib

# ``Agg`` backend allows matplotlib to render charts in headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:  # pragma: no cover - ccxt import guarded for environments without dependency
    import ccxt  # type: ignore

    _CCXT_AVAILABLE = True
except Exception:  # pragma: no cover - if ccxt missing we continue with CSV-only mode
    ccxt = None  # type: ignore
    _CCXT_AVAILABLE = False

from ai_trader.services.equity import EquityEngine
from ai_trader.services.logging import get_logger
from ai_trader.services.risk import RiskManager
from ai_trader.services.trade_log import MemoryTradeLog
from ai_trader.services.types import MarketSnapshot, OpenPosition, TradeIntent
from ai_trader.services.worker_loader import WorkerLoader


REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

DEFAULT_FEE_RATE = 0.0026  # Kraken taker fee of 0.26%
DEFAULT_SLIPPAGE_BPS = 1.0


@dataclass(slots=True)
class BacktestTrade:
    """Container describing a completed backtest trade."""

    worker: str
    symbol: str
    side: str
    quantity: float
    open_time: datetime
    entry_price: float
    entry_fee: float
    cash_spent: float
    close_time: datetime | None = None
    exit_price: float | None = None
    exit_fee: float = 0.0
    pnl: float = 0.0
    pnl_percent: float = 0.0
    win: bool = False
    reason: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def realised(self) -> bool:
        return self.close_time is not None


@dataclass(slots=True)
class BacktestResult:
    """Summary of a backtest run including metrics and artefact paths."""

    metrics: Dict[str, float]
    trades: List[BacktestTrade]
    equity_curve: List[Dict[str, Any]]
    report_paths: Dict[str, Path]


@dataclass(slots=True)
class SimulatedPosition:
    """Internal representation of a paper trade open during the backtest."""

    worker: str
    symbol: str
    side: str
    quantity: float
    intent_price: float
    fill_price: float
    entry_fee: float
    entry_cash_flow: float
    opened_at: datetime

    def to_open_position(self) -> OpenPosition:
        return OpenPosition(
            worker=self.worker,
            symbol=self.symbol,
            side=self.side,
            quantity=self.quantity,
            entry_price=self.fill_price,
            cash_spent=abs(self.quantity * self.fill_price),
            opened_at=self.opened_at,
        )


def _parse_timestamp(value: Any) -> datetime:
    """Coerce a CSV timestamp value into a timezone-aware ``datetime``."""

    if isinstance(value, (int, float)):
        # Value may be in seconds or milliseconds. Assume milliseconds when large.
        if float(value) > 10**11:
            return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    text = str(value).strip()
    if not text:
        raise ValueError("Empty timestamp value")
    # pandas handles ISO8601 parsing gracefully
    parsed = pd.to_datetime(text, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Unable to parse timestamp value: {value}")
    return parsed.to_pydatetime()


def _load_csv_ohlcv(
    csv_path: Path, start: datetime, end: datetime, logger_name: str
) -> list[dict[str, Any]]:
    logger = get_logger(logger_name)
    try:
        frame = pd.read_csv(csv_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"CSV file not found for backtest: {csv_path}") from exc

    if frame.empty:
        raise ValueError("CSV file contains no rows for backtesting")

    # Normalise timestamp column
    ts_column: str | None = None
    for candidate in ("timestamp", "time", "datetime", "date"):
        if candidate in frame.columns:
            ts_column = candidate
            break

    timestamps: List[datetime] = []
    if ts_column is None:
        logger.warning(
            "CSV missing timestamp column – assuming sequential candles starting at %s",
            start.isoformat(),
        )
        timestamps = [start + timedelta(minutes=i) for i in range(len(frame))]
    else:
        for raw in frame[ts_column].tolist():
            try:
                timestamps.append(_parse_timestamp(raw))
            except ValueError:
                timestamps.append(datetime.min.replace(tzinfo=timezone.utc))
        frame = frame.drop(columns=[ts_column])

    candles: List[dict[str, Any]] = []
    for idx, row in enumerate(frame.to_dict(orient="records")):
        timestamp = timestamps[idx] if idx < len(timestamps) else start
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        if timestamp < start or timestamp > end:
            continue
        try:
            candle = {
                "timestamp": timestamp,
                "open": float(row.get("open")),
                "high": float(row.get("high")),
                "low": float(row.get("low")),
                "close": float(row.get("close")),
                "volume": float(row.get("volume", 0.0)),
            }
        except (TypeError, ValueError) as exc:
            logger.debug("Skipping malformed CSV row %d: %s (%s)", idx, row, exc)
            continue
        candles.append(candle)

    candles.sort(key=lambda item: item["timestamp"])
    if not candles:
        raise ValueError(
            f"CSV data between {start.date()} and {end.date()} contains no valid candles"
        )
    return candles


def load_ohlcv_history(
    pair: str,
    start: datetime,
    end: datetime,
    *,
    timeframe: str = "1h",
    csv_path: Path | None = None,
    logger_name: str = __name__,
) -> list[dict[str, Any]]:
    """Return OHLCV candles for the requested period using CSV or CCXT."""

    logger = get_logger(logger_name)
    if csv_path is not None:
        return _load_csv_ohlcv(csv_path, start, end, logger_name)

    if not _CCXT_AVAILABLE:
        raise RuntimeError("ccxt is not installed – provide --backtest-csv to run a backtest")

    exchange = ccxt.kraken({"enableRateLimit": True})  # type: ignore[call-arg]
    since = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
    ohlcv: list[dict[str, Any]] = []

    logger.info("Fetching OHLCV from Kraken via CCXT (%s, timeframe=%s)", pair, timeframe)
    while since <= end_ms:
        try:
            batch = exchange.fetch_ohlcv(pair, timeframe=timeframe, since=since, limit=500)
        except Exception as exc:  # noqa: BLE001 - propagate after logging
            logger.error("CCXT fetch failed for %s: %s", pair, exc)
            raise
        if not batch:
            break
        for entry in batch:
            timestamp_ms = int(entry[0])
            if timestamp_ms > end_ms:
                break
            candle_dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
            if candle_dt < start:
                continue
            ohlcv.append(
                {
                    "timestamp": candle_dt,
                    "open": float(entry[1]),
                    "high": float(entry[2]),
                    "low": float(entry[3]),
                    "close": float(entry[4]),
                    "volume": float(entry[5]) if len(entry) > 5 else 0.0,
                }
            )
        since = batch[-1][0] + timeframe_ms
        if batch[-1][0] >= end_ms:
            break

    if not ohlcv:
        raise ValueError(f"No OHLCV data returned for {pair} between {start} and {end}")

    ohlcv.sort(key=lambda item: item["timestamp"])
    return ohlcv


class Backtester:
    """Replay historical candles against configured workers and risk controls."""

    def __init__(
        self,
        config: Mapping[str, Any],
        pair: str,
        start: datetime,
        end: datetime,
        *,
        timeframe: str = "1h",
        fee_rate: float = DEFAULT_FEE_RATE,
        slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
        csv_path: Path | None = None,
        candles: Sequence[Mapping[str, Any]] | None = None,
        reports_dir: Path | None = None,
        label: str | None = None,
    ) -> None:
        if start >= end:
            raise ValueError("Backtest start date must be earlier than end date")

        self._config = copy.deepcopy(dict(config))
        self._pair = pair.upper()
        self._start = start.astimezone(timezone.utc)
        self._end = end.astimezone(timezone.utc)
        self._timeframe = timeframe
        self._fee_rate = float(fee_rate)
        self._slippage_bps = float(slippage_bps)
        self._slippage = max(0.0, self._slippage_bps / 10_000.0)
        self._csv_path = Path(csv_path).expanduser().resolve() if csv_path else None
        self._logger = get_logger(__name__)
        self._candles_input = list(candles) if candles is not None else None
        self._label = label
        self._reports_dir = reports_dir or REPORTS_DIR
        self._reports_dir.mkdir(parents=True, exist_ok=True)

        trading_cfg = self._config.setdefault("trading", {})
        trading_cfg["symbols"] = [self._pair]
        self._starting_equity = float(trading_cfg.get("paper_starting_equity", 10_000.0))
        self._equity_allocation_percent = float(trading_cfg.get("equity_allocation_percent", 5.0))
        self._max_open_positions = int(
            trading_cfg.get("max_open_positions", trading_cfg.get("max_positions", 3))
        )
        self._min_cash_per_trade = max(0.0, float(trading_cfg.get("min_cash_per_trade", 10.0)))
        self._max_cash_per_trade = max(
            self._min_cash_per_trade,
            float(trading_cfg.get("max_cash_per_trade", self._starting_equity)),
        )

        risk_cfg = self._config.get("risk", {})
        self._risk_manager = RiskManager(risk_cfg)
        self._trade_log = MemoryTradeLog()
        self._equity_engine = EquityEngine(self._trade_log, self._starting_equity)

        loader = WorkerLoader(
            self._config.get("workers"),
            [self._pair],
            researcher_config=self._config.get("researcher"),
        )
        shared_services = {"trade_log": self._trade_log}
        workers, researchers = loader.load(shared_services)
        self._workers = [
            worker for worker in workers if self._pair in getattr(worker, "symbols", [])
        ]
        self._researchers = [
            worker for worker in researchers if self._pair in getattr(worker, "symbols", [])
        ]
        if not self._workers:
            raise ValueError(
                "No trading workers available for backtest – ensure configuration matches the requested pair"
            )

        self._candles: list[dict[str, Any]] = []
        self._cash_balance: float = self._starting_equity
        self._open_positions: Dict[tuple[str, str], SimulatedPosition] = {}
        self._open_trades: Dict[tuple[str, str], BacktestTrade] = {}
        self._closed_trades: list[BacktestTrade] = []
        self._equity_curve: list[Dict[str, Any]] = []
        self._returns: list[float] = []
        self._drawdowns: list[float] = []
        self._wins = 0
        self._losses = 0

    async def run(self) -> BacktestResult:
        """Execute the full backtest and return performance metrics."""

        await self._load_candles()
        self._logger.info(
            "Starting backtest for %s from %s to %s (%d candles)",
            self._pair,
            self._start.date(),
            self._end.date(),
            len(self._candles),
        )
        if not self._candles:
            raise ValueError("Backtest has no candles to process")

        history_prices: list[float] = []
        history_candles: list[dict[str, float]] = []
        prev_equity = self._starting_equity
        peak_equity = self._starting_equity
        current_day: datetime | None = None

        for candle in self._candles:
            timestamp: datetime = candle["timestamp"].astimezone(timezone.utc)
            price = float(candle["close"])
            history_prices.append(price)
            history_candles.append(candle)
            snapshot = MarketSnapshot(
                prices={self._pair: price},
                history={self._pair: list(history_prices)},
                candles={self._pair: list(history_candles[-500:])},
                timestamp=timestamp,
            )

            if current_day is None or timestamp.date() != current_day.date():
                self._risk_manager.reset_daily_limits(now=timestamp)
                current_day = timestamp

            await self._run_researchers(snapshot)
            await self._run_workers(snapshot)

            positions_value = self._mark_to_market(price)
            equity = self._cash_balance + positions_value
            self._equity_engine.update(equity, self._starting_equity)
            metrics = self._equity_engine.get_latest_metrics()

            peak_equity = max(peak_equity, equity)
            drawdown = (equity / peak_equity) - 1.0 if peak_equity else 0.0
            self._drawdowns.append(drawdown)
            ret = (equity - prev_equity) / prev_equity if prev_equity else 0.0
            self._returns.append(ret)
            prev_equity = equity

            self._equity_curve.append(
                {
                    "timestamp": timestamp,
                    "equity": equity,
                    "cash": self._cash_balance,
                    "positions_value": positions_value,
                    "drawdown": drawdown,
                    "pnl_percent": metrics.get("pnl_percent", 0.0),
                }
            )

        result = self._build_result()
        self._notify_summary(result)
        return result

    async def _run_researchers(self, snapshot: MarketSnapshot) -> None:
        if not self._researchers:
            return
        equity_metrics = self._equity_engine.get_latest_metrics()
        open_positions = [position.to_open_position() for position in self._open_positions.values()]
        for researcher in self._researchers:
            try:
                await researcher.observe(snapshot, equity_metrics, open_positions)
            except Exception as exc:  # noqa: BLE001
                self._logger.exception(
                    "Researcher %s failed during backtest: %s",
                    getattr(researcher, "name", researcher.__class__.__name__),
                    exc,
                )

    async def _run_workers(self, snapshot: MarketSnapshot) -> None:
        equity_metrics = self._equity_engine.get_latest_metrics()
        equity = float(equity_metrics.get("equity", self._starting_equity))
        equity_per_trade = equity * (self._equity_allocation_percent / 100.0)

        for worker in self._workers:
            if not getattr(worker, "active", True):
                continue
            try:
                signals = await worker.evaluate_signal(snapshot)
            except Exception as exc:  # noqa: BLE001
                self._logger.error("Worker %s signal evaluation failed: %s", worker, exc)
                continue

            for symbol in getattr(worker, "symbols", []):
                if symbol != self._pair:
                    continue
                signal = signals.get(symbol)
                key = (getattr(worker, "name", worker.__class__.__name__), symbol)
                position = self._open_positions.get(key)
                try:
                    intent = await worker.generate_trade(
                        symbol,
                        signal,
                        snapshot,
                        equity_per_trade,
                        position.to_open_position() if position else None,
                    )
                except Exception as exc:  # noqa: BLE001
                    self._logger.error(
                        "Worker %s trade generation failed: %s",
                        getattr(worker, "name", worker.__class__.__name__),
                        exc,
                    )
                    continue
                if intent is None:
                    continue
                if worker.long_only and intent.action == "OPEN" and intent.side != "buy":
                    self._logger.debug("Skipping short trade for long-only worker %s", worker)
                    continue

                assessment = self._risk_manager.evaluate_trade(
                    intent,
                    equity=equity,
                    equity_metrics=equity_metrics,
                    open_positions=len(self._open_positions),
                    max_open_positions=self._max_open_positions,
                    price=snapshot.prices.get(symbol),
                    candles=snapshot.candles.get(symbol),
                )
                intent = assessment.intent
                if not assessment.allowed:
                    continue

                if intent.action == "OPEN":
                    await self._execute_open(intent, snapshot.timestamp)
                elif intent.action == "CLOSE" and position is not None:
                    await self._execute_close(intent, snapshot.timestamp, position)

    async def _execute_open(self, intent: TradeIntent, timestamp: datetime) -> None:
        key = (intent.worker, intent.symbol)
        available_cash = self._cash_balance
        requested_cash = float(intent.cash_spent)
        requested_cash = max(self._min_cash_per_trade, requested_cash)
        requested_cash = min(requested_cash, self._max_cash_per_trade)
        if requested_cash <= 0 or available_cash <= 0:
            return
        cash_allocation = min(requested_cash, available_cash)
        if cash_allocation < self._min_cash_per_trade:
            return

        fill_price = self._apply_slippage(intent.side, float(intent.entry_price))
        quantity = cash_allocation / max(fill_price, 1e-9)
        notional = quantity * fill_price
        fee = notional * self._fee_rate
        if intent.side == "buy":
            cash_flow = -(notional + fee)
        else:
            cash_flow = notional - fee
        self._cash_balance += cash_flow

        position = SimulatedPosition(
            worker=intent.worker,
            symbol=intent.symbol,
            side=intent.side,
            quantity=quantity,
            intent_price=float(intent.entry_price),
            fill_price=fill_price,
            entry_fee=fee,
            entry_cash_flow=cash_flow,
            opened_at=timestamp,
        )
        self._open_positions[key] = position

        trade = BacktestTrade(
            worker=intent.worker,
            symbol=intent.symbol,
            side=intent.side,
            quantity=quantity,
            open_time=timestamp,
            entry_price=fill_price,
            entry_fee=fee,
            cash_spent=notional,
            metadata=dict(intent.metadata or {}),
        )
        self._open_trades[key] = trade

        recorded_intent = TradeIntent(
            worker=intent.worker,
            action="OPEN",
            symbol=intent.symbol,
            side=intent.side,
            cash_spent=notional,
            entry_price=fill_price,
            confidence=float(intent.confidence),
            reason=intent.reason or "entry",
            metadata={**(intent.metadata or {}), "mode": "backtest", "fill_quantity": quantity},
        )
        self._trade_log.record_trade(recorded_intent)
        self._risk_manager.on_trade_executed(recorded_intent)

    async def _execute_close(
        self, intent: TradeIntent, timestamp: datetime, position: SimulatedPosition
    ) -> None:
        key = (intent.worker, intent.symbol)
        fill_price = self._apply_slippage(
            intent.side, float(intent.exit_price or intent.entry_price)
        )
        quantity = position.quantity
        notional = quantity * fill_price
        fee = notional * self._fee_rate
        if position.side == "buy":
            cash_flow = notional - fee
        else:
            cash_flow = -(notional + fee)
        self._cash_balance += cash_flow

        total_cash_flow = position.entry_cash_flow + cash_flow
        pnl = total_cash_flow
        base = abs(position.entry_cash_flow)
        pnl_percent = (pnl / base * 100.0) if base else 0.0
        win = pnl > 0
        if win:
            self._wins += 1
        else:
            self._losses += 1

        trade = self._open_trades.pop(key, None)
        if trade is None:
            trade = BacktestTrade(
                worker=intent.worker,
                symbol=intent.symbol,
                side=intent.side,
                quantity=quantity,
                open_time=position.opened_at,
                entry_price=position.fill_price,
                entry_fee=position.entry_fee,
                cash_spent=abs(position.entry_cash_flow),
            )
        trade.close_time = timestamp
        trade.exit_price = fill_price
        trade.exit_fee = fee
        trade.pnl = pnl
        trade.pnl_percent = pnl_percent
        trade.win = win
        trade.reason = intent.reason or "exit"
        trade.metadata.update(intent.metadata or {})
        self._closed_trades.append(trade)
        self._open_positions.pop(key, None)

        recorded_intent = TradeIntent(
            worker=intent.worker,
            action="CLOSE",
            symbol=intent.symbol,
            side=intent.side,
            cash_spent=trade.cash_spent,
            entry_price=trade.entry_price,
            exit_price=fill_price,
            pnl_percent=pnl_percent,
            pnl_usd=pnl,
            win_loss="win" if win else "loss",
            reason=intent.reason or "exit",
            metadata={
                **(intent.metadata or {}),
                "mode": "backtest",
                "fill_quantity": quantity,
                "fill_price": fill_price,
                "fees": position.entry_fee + fee,
            },
        )
        self._trade_log.record_trade(recorded_intent)
        self._risk_manager.on_trade_executed(recorded_intent)

    def _mark_to_market(self, price: float) -> float:
        valuation = 0.0
        for position in self._open_positions.values():
            notional = position.quantity * price
            if position.side == "buy":
                valuation += notional
            else:
                valuation -= notional
        return valuation

    def _apply_slippage(self, side: str, price: float) -> float:
        if self._slippage <= 0:
            return price
        if side == "buy":
            return price * (1.0 + self._slippage)
        return price * (1.0 - self._slippage)

    async def _load_candles(self) -> None:
        if self._candles_input is not None:
            prepared: list[dict[str, Any]] = []
            for candle in self._candles_input:
                timestamp = candle.get("timestamp")
                if not isinstance(timestamp, datetime):
                    timestamp = _parse_timestamp(timestamp)
                if timestamp < self._start or timestamp > self._end:
                    continue
                prepared.append(
                    {
                        "timestamp": timestamp.astimezone(timezone.utc),
                        "open": float(candle["open"]),
                        "high": float(candle["high"]),
                        "low": float(candle["low"]),
                        "close": float(candle["close"]),
                        "volume": float(candle.get("volume", 0.0)),
                    }
                )
            prepared.sort(key=lambda item: item["timestamp"])
            self._candles = prepared
            return

        self._candles = load_ohlcv_history(
            self._pair,
            self._start,
            self._end,
            timeframe=self._timeframe,
            csv_path=self._csv_path,
        )

    def _build_result(self) -> BacktestResult:
        latest_equity = (
            self._equity_curve[-1]["equity"] if self._equity_curve else self._starting_equity
        )
        if self._equity_curve:
            final_equity = latest_equity
        else:
            last_price = float(self._candles[-1]["close"]) if self._candles else 0.0
            final_equity = self._cash_balance + self._mark_to_market(last_price)
        net_profit = final_equity - self._starting_equity
        total_return_pct = (
            (net_profit / self._starting_equity * 100.0) if self._starting_equity else 0.0
        )
        max_drawdown = abs(min(self._drawdowns)) * 100.0 if self._drawdowns else 0.0
        returns_array = np.array([r for r in self._returns if math.isfinite(r)])
        sharpe = 0.0
        sortino = 0.0
        if returns_array.size > 1 and returns_array.std(ddof=1) > 0:
            sharpe = returns_array.mean() / returns_array.std(ddof=1) * math.sqrt(252)
        downside = returns_array[returns_array < 0]
        if downside.size > 0 and downside.std(ddof=1) > 0:
            sortino = returns_array.mean() / downside.std(ddof=1) * math.sqrt(252)

        total_trades = len(self._closed_trades)
        win_rate = (self._wins / total_trades * 100.0) if total_trades else 0.0

        metrics = {
            "final_equity": final_equity,
            "net_profit": net_profit,
            "return_percent": total_return_pct,
            "max_drawdown_percent": max_drawdown,
            "total_trades": float(total_trades),
            "win_rate": win_rate,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "fees_paid": sum(trade.entry_fee + trade.exit_fee for trade in self._closed_trades),
        }

        report_paths = self._export_reports(metrics)
        equity_curve = [
            {
                **row,
                "timestamp": row["timestamp"].isoformat(),
            }
            for row in self._equity_curve
        ]

        return BacktestResult(
            metrics=metrics,
            trades=self._closed_trades.copy(),
            equity_curve=equity_curve,
            report_paths=report_paths,
        )

    def _export_reports(self, metrics: Mapping[str, float]) -> Dict[str, Path]:
        timestamp_label = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        pair_label = self._pair.replace("/", "-")
        base_label = self._label or f"{pair_label}_{timestamp_label}"

        equity_path = self._reports_dir / f"backtest_{base_label}_equity.csv"
        trades_path = self._reports_dir / f"backtest_{base_label}_trades.csv"
        summary_path = self._reports_dir / f"backtest_{base_label}_summary.json"
        equity_png = self._reports_dir / f"backtest_{base_label}_equity.png"
        drawdown_png = self._reports_dir / f"backtest_{base_label}_drawdown.png"
        returns_png = self._reports_dir / f"backtest_{base_label}_returns.png"

        equity_frame = pd.DataFrame(self._equity_curve)
        if not equity_frame.empty:
            equity_frame["timestamp"] = equity_frame["timestamp"].map(lambda ts: ts.isoformat())
            equity_frame.to_csv(equity_path, index=False)
        trades_frame = pd.DataFrame(
            [
                {
                    "worker": trade.worker,
                    "symbol": trade.symbol,
                    "side": trade.side,
                    "quantity": trade.quantity,
                    "open_time": trade.open_time.isoformat(),
                    "close_time": trade.close_time.isoformat() if trade.close_time else None,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "entry_fee": trade.entry_fee,
                    "exit_fee": trade.exit_fee,
                    "pnl": trade.pnl,
                    "pnl_percent": trade.pnl_percent,
                    "win": trade.win,
                    "reason": trade.reason,
                }
                for trade in self._closed_trades
            ]
        )
        if not trades_frame.empty:
            trades_frame.to_csv(trades_path, index=False)

        summary_payload: Dict[str, Any] = {
            "pair": self._pair,
            "start": self._start.isoformat(),
            "end": self._end.isoformat(),
            "timeframe": self._timeframe,
            "fee_rate": self._fee_rate,
            "slippage_bps": self._slippage_bps,
            "metrics": dict(metrics),
            "trades": len(self._closed_trades),
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

        self._plot_equity(equity_png)
        self._plot_drawdown(drawdown_png)
        self._plot_returns(returns_png)

        return {
            "equity_csv": equity_path,
            "trades_csv": trades_path,
            "summary_json": summary_path,
            "equity_png": equity_png,
            "drawdown_png": drawdown_png,
            "returns_png": returns_png,
        }

    def _plot_equity(self, output_path: Path) -> None:
        if not self._equity_curve:
            return
        plt.figure(figsize=(10, 4))
        plt.plot(
            [row["timestamp"] for row in self._equity_curve],
            [row["equity"] for row in self._equity_curve],
        )
        plt.title(f"Equity Curve – {self._pair}")
        plt.xlabel("Time")
        plt.ylabel("Equity")
        plt.tight_layout()
        plt.gcf().autofmt_xdate()
        plt.savefig(output_path)
        plt.close()

    def _plot_drawdown(self, output_path: Path) -> None:
        if not self._equity_curve:
            return
        plt.figure(figsize=(10, 4))
        plt.plot(
            [row["timestamp"] for row in self._equity_curve], [dd * 100 for dd in self._drawdowns]
        )
        plt.title(f"Drawdown Curve – {self._pair}")
        plt.xlabel("Time")
        plt.ylabel("Drawdown (%)")
        plt.tight_layout()
        plt.gcf().autofmt_xdate()
        plt.savefig(output_path)
        plt.close()

    def _plot_returns(self, output_path: Path) -> None:
        if not self._returns:
            return
        plt.figure(figsize=(8, 4))
        plt.hist([r * 100 for r in self._returns], bins=20, edgecolor="black")
        plt.title("Distribution of Period Returns (%)")
        plt.xlabel("Return (%)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def _notify_summary(self, result: BacktestResult) -> None:
        metrics = result.metrics
        self._logger.info(
            "Backtest complete: trades=%d net=%.2fUSD return=%.2f%% maxDD=%.2f%% sharpe=%.2f sortino=%.2f",
            int(metrics.get("total_trades", 0)),
            metrics.get("net_profit", 0.0),
            metrics.get("return_percent", 0.0),
            metrics.get("max_drawdown_percent", 0.0),
            metrics.get("sharpe_ratio", 0.0),
            metrics.get("sortino_ratio", 0.0),
        )


async def run_backtest(
    config: Mapping[str, Any],
    pair: str,
    start: datetime,
    end: datetime,
    *,
    timeframe: str = "1h",
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    csv_path: Path | None = None,
    reports_dir: Path | None = None,
    label: str | None = None,
) -> BacktestResult:
    """Convenience wrapper that instantiates and executes :class:`Backtester`."""

    tester = Backtester(
        config,
        pair,
        start,
        end,
        timeframe=timeframe,
        fee_rate=fee_rate,
        slippage_bps=slippage_bps,
        csv_path=csv_path,
        reports_dir=reports_dir,
        label=label,
    )
    return await tester.run()
