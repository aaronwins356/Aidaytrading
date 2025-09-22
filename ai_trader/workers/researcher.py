"""Non-trading researcher bot that feeds the ML pipeline."""

from __future__ import annotations

import math
from statistics import fmean, pstdev
from typing import Dict, Iterable, List, Optional

from .base import BaseWorker
from ..services.ml_pipeline import MLPipeline
from ..services.trade_log import TradeLog
from ..services.types import MarketSnapshot, OpenPosition


class MarketResearchWorker(BaseWorker):
    """Observes markets, engineers features, and persists research data."""

    name = "Research Sentinel"
    emoji = "🔬"
    is_researcher = True

    def __init__(
        self,
        symbols,
        trade_log: TradeLog,
        pipeline: MLPipeline,
        config: Optional[Dict] = None,
    ) -> None:
        lookback = max(200, int((config or {}).get("lookback", 200)))
        super().__init__(symbols=symbols, lookback=lookback, config=config)
        self._trade_log = trade_log
        self._pipeline = pipeline
        cfg = config or {}
        self.feature_windows: List[int] = list(cfg.get("feature_windows", [5, 14, 30]))
        self.timeframe: str = cfg.get("timeframe", "1m")
        self.log_every_n = max(1, int(cfg.get("log_every_n_snapshots", 1)))
        self.volatility_window = max(5, int(cfg.get("volatility_window", 20)))
        self.ohlc_window = max(5, int(cfg.get("ohlc_window", 20)))
        self._snapshot_counter = 0

    async def evaluate_signal(self, snapshot: MarketSnapshot) -> Dict[str, str]:
        # Researcher does not emit trade signals but we keep interface parity.
        await self.observe(snapshot)
        return {}

    async def generate_trade(self, *args, **kwargs):  # type: ignore[override]
        return None

    async def observe(
        self,
        snapshot: MarketSnapshot,
        equity_metrics: Optional[Dict[str, float]] = None,
        open_positions: Optional[List[OpenPosition]] = None,
    ) -> None:
        self.update_history(snapshot)
        self._snapshot_counter += 1
        if self._snapshot_counter % self.log_every_n != 0:
            return
        for symbol in self.symbols:
            history = snapshot.history.get(symbol, [])
            if len(history) < max(max(self.feature_windows), self.warmup_candles) + 5:
                continue
            features = self._build_features(history)
            ohlcv = self._build_ohlcv(history)
            label = 1.0 if features.get("return_5", 0.0) < 0 and features.get("return_1", 0.0) < 0 else 0.0
            payload = {
                "symbol": symbol,
                "timeframe": self.timeframe,
                "open": ohlcv["open"],
                "high": ohlcv["high"],
                "low": ohlcv["low"],
                "close": ohlcv["close"],
                "volume": ohlcv["volume"],
                "features": features,
                "label": label,
            }
            self._trade_log.record_market_features(payload)
            self._pipeline.train(symbol)
            state_payload = {"features": features, "label": label}
            state_payload.update({f"ohlcv_{key}": value for key, value in ohlcv.items()})
            self.update_signal_state(symbol, None, state_payload)

    def _build_features(self, history: List[float]) -> Dict[str, float]:
        latest = history[-1]
        prev = history[-2]
        returns = [math.log(history[idx] / history[idx - 1]) for idx in range(1, len(history)) if history[idx - 1] > 0]
        return_1 = math.log(latest / prev) if prev > 0 else 0.0
        return_5 = math.log(latest / history[-6]) if len(history) > 5 and history[-6] > 0 else return_1
        return_15 = math.log(latest / history[-16]) if len(history) > 15 and history[-16] > 0 else return_5
        volatility = (
            pstdev(returns[-self.volatility_window :])
            if len(returns) >= self.volatility_window
            else pstdev(returns)
            if returns
            else 0.0
        )
        ema_fast = self._ema(history, 12)
        ema_slow = self._ema(history, 26)
        macd = ema_fast - ema_slow
        signal = self._ema(history, 9)
        macd_hist = macd - signal
        rsi = self._rsi(history, period=14)
        mean_price = fmean(history[-30:]) if len(history) >= 30 else fmean(history)
        std_dev = pstdev(history[-30:]) if len(history) >= 30 else pstdev(history) if len(history) > 1 else 0.0
        zscore = (latest - mean_price) / std_dev if std_dev else 0.0
        features = {
            "return_1": return_1,
            "return_5": return_5,
            "return_15": return_15,
            "volatility": volatility,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "macd": macd,
            "macd_hist": macd_hist,
            "rsi": rsi,
            "zscore": zscore,
        }
        return features

    @staticmethod
    def _ema(history: Iterable[float], window: int) -> float:
        prices = list(history[-window * 3 :]) if window else list(history)
        if not prices:
            return 0.0
        k = 2 / (window + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = price * k + ema * (1 - k)
        return ema

    @staticmethod
    def _rsi(history: List[float], period: int = 14) -> float:
        if len(history) < period + 1:
            return 50.0
        gains = []
        losses = []
        for idx in range(1, period + 1):
            delta = history[-idx] - history[-idx - 1]
            if delta >= 0:
                gains.append(delta)
            else:
                losses.append(abs(delta))
        avg_gain = sum(gains) / period if gains else 0.0
        avg_loss = sum(losses) / period if losses else 1e-9
        rs = avg_gain / avg_loss if avg_loss else 0.0
        rsi = 100 - (100 / (1 + rs))
        return max(0.0, min(100.0, rsi))

    def _build_ohlcv(self, history: List[float]) -> Dict[str, float]:
        window = history[-self.ohlc_window :]
        if not window:
            price = history[-1]
            return {"open": price, "high": price, "low": price, "close": price, "volume": 0.0}
        open_price = window[0]
        high_price = max(window)
        low_price = min(window)
        close_price = window[-1]
        return {
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": 0.0,
        }
