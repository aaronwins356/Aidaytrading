"""Non-trading researcher bot that feeds the online ML service."""

from __future__ import annotations

import math
from statistics import fmean, pstdev
from typing import Dict, Iterable, List, Optional

from ai_trader.services.ml import MLService
from ai_trader.services.trade_log import TradeLog
from ai_trader.services.types import MarketSnapshot, OpenPosition
from ai_trader.workers.base import BaseWorker


class MarketResearchWorker(BaseWorker):
    """Observes markets, engineers features, and persists research data."""

    name = "Research Sentinel"
    emoji = "🔬"
    is_researcher = True

    def __init__(
        self,
        symbols,
        trade_log: TradeLog,
        ml_service: MLService,
        config: Optional[Dict] = None,
    ) -> None:
        lookback = max(200, int((config or {}).get("lookback", 200)))
        super().__init__(symbols=symbols, lookback=lookback, config=config)
        self._trade_log = trade_log
        self._ml_service = ml_service
        cfg = config or {}
        self.feature_windows: List[int] = list(cfg.get("feature_windows", [5, 14, 30]))
        self.timeframe: str = cfg.get("timeframe", "1m")
        self.log_every_n = max(1, int(cfg.get("log_every_n_snapshots", 1)))
        self.volatility_window = max(5, int(cfg.get("volatility_window", 20)))
        self.ohlc_window = max(5, int(cfg.get("ohlc_window", 20)))
        self.atr_window = max(5, int(cfg.get("atr_window", 14)))
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
            candles = snapshot.candles.get(symbol, [])
            required_candles = max(self.warmup_candles, 3)
            if len(candles) < required_candles:
                self._logger.debug(
                    "Skipping feature snapshot for %s – need %d candles, have %d",
                    symbol,
                    required_candles,
                    len(candles),
                )
                continue
            features = self._build_features(candles)
            ohlcv = dict(candles[-1]) if candles else self._build_ohlcv([])
            label = self._derive_label(candles)
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
            confidence = self._ml_service.update(
                symbol,
                features,
                label=label,
                timestamp=snapshot.timestamp,
            )
            self._logger.info(
                "Snapshot saved for %s | label=%.0f features=%d confidence=%.4f",
                symbol,
                float(label) if label is not None else -1.0,
                len(features),
                confidence,
            )
            state_payload = {"features": features, "label": label, "ml_confidence": confidence}
            state_payload.update({f"ohlcv_{key}": value for key, value in ohlcv.items()})
            self.update_signal_state(symbol, None, state_payload)

    def _build_features(self, candles: List[dict[str, float]]) -> Dict[str, float]:
        closes = [float(candle.get("close", 0.0)) for candle in candles]
        volumes = [float(candle.get("volume", 0.0)) for candle in candles]
        latest_close = closes[-1]
        returns = [
            math.log(closes[idx] / closes[idx - 1])
            for idx in range(1, len(closes))
            if closes[idx - 1] > 0
        ]

        features: Dict[str, float] = {}
        for window in (1, 3, 5, 10):
            if len(closes) > window and closes[-window - 1] > 0:
                features[f"momentum_{window}"] = math.log(latest_close / closes[-window - 1])
            else:
                features[f"momentum_{window}"] = 0.0

        volatility_window = min(self.volatility_window, len(returns))
        if volatility_window > 1:
            features["rolling_volatility"] = pstdev(returns[-volatility_window:])
        else:
            features["rolling_volatility"] = pstdev(returns) if len(returns) > 1 else 0.0

        features["atr"] = self._atr(candles, self.atr_window)

        volume_delta = volumes[-1] - volumes[-2] if len(volumes) > 1 else 0.0
        features["volume_delta"] = volume_delta
        prev_volume = volumes[-2] if len(volumes) > 1 else 1.0
        features["volume_ratio"] = volumes[-1] / prev_volume if prev_volume else 1.0
        volume_window = volumes[-4:-1] if len(volumes) > 3 else volumes[:-1]
        volume_mean = fmean(volume_window) if volume_window else 1.0
        features["volume_ratio_3"] = volumes[-1] / volume_mean if volume_mean else 1.0
        longer_window = volumes[-11:-1] if len(volumes) > 10 else volumes[:-1]
        longer_mean = fmean(longer_window) if longer_window else volume_mean
        features["volume_ratio_10"] = volumes[-1] / longer_mean if longer_mean else 1.0

        last_candle = candles[-1]
        body = float(last_candle.get("close", 0.0)) - float(last_candle.get("open", 0.0))
        high_low_range = max(float(last_candle.get("high", 0.0)) - float(last_candle.get("low", 0.0)), 1e-8)
        upper_wick = float(last_candle.get("high", 0.0)) - max(float(last_candle.get("open", 0.0)), float(last_candle.get("close", 0.0)))
        lower_wick = min(float(last_candle.get("open", 0.0)), float(last_candle.get("close", 0.0))) - float(last_candle.get("low", 0.0))
        features["body_pct"] = abs(body) / high_low_range
        features["upper_wick_pct"] = max(upper_wick, 0.0) / high_low_range
        features["lower_wick_pct"] = max(lower_wick, 0.0) / high_low_range
        close_price = float(last_candle.get("close", 0.0)) or 1e-8
        features["wick_close_ratio"] = (max(upper_wick, 0.0) + max(lower_wick, 0.0)) / abs(close_price)
        features["range_pct"] = high_low_range / abs(close_price)

        features["ema_fast"] = self._ema(closes, 12)
        features["ema_slow"] = self._ema(closes, 26)
        macd = features["ema_fast"] - features["ema_slow"]
        signal = self._ema(closes, 9)
        features["macd"] = macd
        features["macd_hist"] = macd - signal
        features["rsi"] = self._rsi(closes, period=14)
        mean_price = fmean(closes[-30:]) if len(closes) >= 30 else fmean(closes)
        std_dev = pstdev(closes[-30:]) if len(closes) >= 30 else pstdev(closes) if len(closes) > 1 else 0.0
        features["zscore"] = (latest_close - mean_price) / std_dev if std_dev else 0.0
        features["close_to_high"] = (float(last_candle.get("high", 0.0)) - latest_close) / latest_close if latest_close else 0.0
        features["close_to_low"] = (latest_close - float(last_candle.get("low", 0.0))) / latest_close if latest_close else 0.0
        return features

    def _derive_label(self, candles: List[dict[str, float]]) -> float:
        if len(candles) < 2:
            return 0.0
        prev_close = float(candles[-2].get("close", 0.0))
        latest_close = float(candles[-1].get("close", 0.0))
        return 1.0 if latest_close < prev_close else 0.0

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
        if not history:
            return {"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0.0}
        window = history[-self.ohlc_window :]
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

    def _atr(self, candles: List[dict[str, float]], window: int) -> float:
        if len(candles) < 2:
            return 0.0
        true_ranges: List[float] = []
        for idx in range(1, len(candles)):
            current = candles[idx]
            prev = candles[idx - 1]
            high = float(current.get("high", 0.0))
            low = float(current.get("low", 0.0))
            prev_close = float(prev.get("close", 0.0))
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close),
            )
            true_ranges.append(tr)
        if not true_ranges:
            return 0.0
        return fmean(true_ranges[-window:]) if len(true_ranges) >= window else fmean(true_ranges)
