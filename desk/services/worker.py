"""Worker orchestration – wraps individual strategy bots."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

try:  # pragma: no cover - import guard
    import pandas as pd  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised in tests
    pd = None  # type: ignore

from desk.data import candles_to_dataframe
from desk.services.learner import Learner
from desk.services.logger import EventLogger

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from desk.services.risk import RiskEngine


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


@dataclass
class VetoResult:
    name: str
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Intent:
    worker: "Worker"
    symbol: str
    side: str
    qty: float
    price: float
    score: float
    vetoes: List[VetoResult]
    features: Dict[str, float]
    ml_score: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_hold_minutes: Optional[float] = None
    plan_metadata: Dict[str, float] = field(default_factory=dict)

    @property
    def approved(self) -> bool:
        return all(v.passed for v in self.vetoes)


class Worker:
    """Loads a strategy module and evaluates trading intents."""

    ACRONYMS = {"sma", "ema", "macd", "rsi", "atr", "vwma", "vwap"}

    def __init__(
        self,
        name: str,
        symbol: str,
        strategy: str,
        params: Dict[str, Any],
        logger: Optional[EventLogger] = None,
        learner: Optional[Learner] = None,
        risk_engine: Optional["RiskEngine"] = None,
    ) -> None:
        self.name = name
        self.symbol = symbol
        self.strategy_name = strategy
        self.params = params
        self.logger = logger or EventLogger()
        self.learner = learner or Learner()
        self.risk_engine = risk_engine

        self.strategy = self._load_strategy()
        self.state: Dict[str, Any] = {
            "candles": [],
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "pnl": 0.0,
        }
        self.allocation = float(params.get("allocation", params.get("weight", 0.1)))
        self.risk_profile: Dict[str, Any] = dict(params.get("risk_profile", {}))

    # ------------------------------------------------------------------
    def _learning_risk_multiplier(self) -> float:
        profile = self.risk_profile or {}
        initial = float(profile.get("initial_multiplier", 1.0) or 1.0)
        floor = float(profile.get("floor_multiplier", initial) or initial)
        tighten_trades = int(profile.get("tighten_trades", 0) or 0)
        target_win_rate = float(profile.get("target_win_rate", 0.0) or 0.0)

        # Ensure sane ordering between floor and ceiling multipliers.
        if initial <= 0:
            initial = 1.0
        floor = max(0.0, min(floor, initial))

        trades = max(int(self.state.get("trades", 0) or 0), 0)
        progress = 0.0
        if tighten_trades > 0:
            progress = min(1.0, trades / float(tighten_trades))

        wins = int(self.state.get("wins", 0) or 0)
        losses = int(self.state.get("losses", 0) or 0)
        total = wins + losses
        if total > 0 and target_win_rate > 0:
            win_rate = wins / float(total)
            if win_rate >= target_win_rate:
                bonus = (win_rate - target_win_rate) / max(1.0 - target_win_rate, 1e-6)
                progress = min(1.0, progress + bonus)

        multiplier = initial - (initial - floor) * progress
        return max(floor, round(multiplier, 6))

    # ------------------------------------------------------------------
    @staticmethod
    def _apply_learning_risk(
        side: str,
        price: float,
        plan: Dict[str, Any],
        multiplier: float,
    ) -> Dict[str, Any]:
        if not plan:
            return {}

        scaled: Dict[str, Any] = dict(plan)
        multiplier = max(multiplier, 0.1)

        def _scale_level(level: Any, *, is_stop: bool) -> Optional[float]:
            if not _is_number(level):
                return None
            value = float(level)
            if is_stop:
                if side == "BUY":
                    distance = price - value
                    if distance <= 0:
                        return value
                    return max(price - distance * multiplier, 0.0)
                else:
                    distance = value - price
                    if distance <= 0:
                        return value
                    return price + distance * multiplier
            else:
                if side == "BUY":
                    distance = value - price
                    if distance <= 0:
                        return value
                    return price + distance * multiplier
                else:
                    distance = price - value
                    if distance <= 0:
                        return value
                    return price - distance * multiplier

        stop_loss = _scale_level(plan.get("stop_loss"), is_stop=True)
        if stop_loss is not None:
            scaled["stop_loss"] = stop_loss

        take_profit = _scale_level(plan.get("take_profit"), is_stop=False)
        if take_profit is not None:
            scaled["take_profit"] = take_profit

        return scaled

    # ------------------------------------------------------------------
    def _load_strategy(self):
        module = importlib.import_module(f"desk.strategies.{self.strategy_name}")
        parts = self.strategy_name.split("_")
        class_name = "".join(
            part.upper() if part in self.ACRONYMS else part.title() for part in parts
        ) + "Strategy"
        if not hasattr(module, class_name):
            raise ImportError(f"Strategy class {class_name} not found in {module}")
        strategy_cls = getattr(module, class_name)
        return strategy_cls(self.symbol, self.params.get("params", {}))

    def push_candle(self, candle: Dict[str, Any], max_history: int) -> None:
        history = self.state.setdefault("candles", [])
        history.append(candle)
        if len(history) > max_history:
            del history[: len(history) - max_history]

    # ------------------------------------------------------------------
    def _candles_df(self) -> pd.DataFrame:
        return candles_to_dataframe(self.state.get("candles", []))

    def _generate_signal(self, df: pd.DataFrame) -> Optional[str]:
        signal = self.strategy.generate_signals(df)
        if signal not in {"buy", "sell"}:
            return None
        return signal.upper()

    def _veto_checks(self, df: pd.DataFrame) -> List[VetoResult]:
        latest = df.iloc[-1]
        vetoes = []

        # Spread/volatility gating – placeholder using candle body vs ATR.
        atr = df["high"].rolling(14).max() - df["low"].rolling(14).min()
        body = abs(latest["close"] - latest["open"])
        vetoes.append(
            VetoResult(
                name="volatility_window",
                passed=body <= atr.iloc[-1] * 1.5 if not atr.isna().iloc[-1] else True,
                details={"body": float(body)},
            )
        )

        # Session filter – avoid trading during the first 5 minutes of hour as placeholder.
        vetoes.append(
            VetoResult(
                name="session_filter",
                passed=(int(latest["timestamp"]) // 60) % 60 >= 5,
            )
        )

        # Momentum sanity – ensure price not vertical.
        mom = df["close"].pct_change().tail(5).abs().mean()
        vetoes.append(
            VetoResult(
                name="momentum_spike",
                passed=mom < 0.05,
                details={"avg_pct_change": float(mom)},
            )
        )

        return vetoes

    def compute_quantity(
        self,
        price: float,
        risk_budget: float,
        *,
        stop_loss: Optional[float] = None,
        side: str = "BUY",
    ) -> float:
        params = self.params.get("params", {})
        if self.risk_engine is not None:
            qty = self.risk_engine.position_size(
                price,
                risk_budget,
                stop_loss=stop_loss,
                side=side,
            )
        else:
            qty = risk_budget / max(price, 1e-9)
        min_qty = float(params.get("min_qty", 0.0) or 0.0)
        if min_qty > 0:
            qty = max(qty, min_qty)
        precision = int(params.get("qty_precision", 6) or 6)
        return round(qty, precision)

    def build_intent(self, risk_budget: float) -> Optional[Intent]:
        df = self._candles_df()
        if df.empty:
            return None

        side = self._generate_signal(df)
        if side is None:
            return None

        price = float(df["close"].iloc[-1])
        vetoes = self._veto_checks(df)

        features = self._collect_features(df)
        ml_score = self.learner.predict_edge(self, features)
        score = 0.6 + 0.4 * ml_score

        trade_plan: Dict[str, float] = {}
        try:
            trade_plan = self.strategy.plan_trade(side.lower(), df) or {}
        except Exception:
            trade_plan = {}

        risk_multiplier = self._learning_risk_multiplier()
        adaptive_risk_budget = risk_budget * risk_multiplier
        scaled_plan = self._apply_learning_risk(side, price, trade_plan, risk_multiplier)

        qty = self.compute_quantity(
            price,
            adaptive_risk_budget,
            stop_loss=scaled_plan.get("stop_loss"),
            side=side,
        )

        plan_metadata = scaled_plan.get("metadata")
        if isinstance(plan_metadata, dict):
            metadata_dict = {str(k): float(v) for k, v in plan_metadata.items() if _is_number(v)}
        else:
            metadata_dict = {}

        enriched_features = dict(features)
        enriched_features["ml_edge"] = ml_score
        enriched_features["combined_score"] = score
        enriched_features["proposed_qty"] = qty
        enriched_features["risk_budget"] = adaptive_risk_budget
        enriched_features["learning_risk_multiplier"] = risk_multiplier
        enriched_features["side"] = 1.0 if side == "BUY" else -1.0
        if "stop_loss" in scaled_plan:
            enriched_features["plan_stop_loss"] = float(scaled_plan.get("stop_loss", 0.0) or 0.0)
        if "take_profit" in scaled_plan:
            enriched_features["plan_take_profit"] = float(scaled_plan.get("take_profit", 0.0) or 0.0)
        if "max_hold_minutes" in scaled_plan:
            enriched_features["plan_max_hold_minutes"] = float(scaled_plan.get("max_hold_minutes", 0.0) or 0.0)
        for key, value in metadata_dict.items():
            enriched_features[f"plan_meta_{key}"] = value

        return Intent(
            worker=self,
            symbol=self.symbol,
            side=side,
            qty=qty,
            price=price,
            score=score,
            vetoes=vetoes,
            features=enriched_features,
            ml_score=ml_score,
            stop_loss=scaled_plan.get("stop_loss"),
            take_profit=scaled_plan.get("take_profit"),
            max_hold_minutes=scaled_plan.get("max_hold_minutes"),
            plan_metadata=metadata_dict,
        )

    # ------------------------------------------------------------------
    def _collect_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compile candle + strategy features for downstream learning."""

        latest = df.iloc[-1]
        features: Dict[str, float] = {}

        for column in ("open", "high", "low", "close", "volume"):
            if column in df.columns:
                try:
                    features[f"candle_{column}"] = float(latest[column])
                except (TypeError, ValueError):
                    continue

        # Derived features for richer ML context.
        if "close" in df.columns:
            returns = df["close"].pct_change().fillna(0.0)
            short_window = returns.tail(5)
            long_window = returns.tail(20)
            features["signal_return_volatility_short"] = float(short_window.std() or 0.0)
            features["signal_return_volatility_long"] = float(long_window.std() or 0.0)
            long_vol = features["signal_return_volatility_long"] or 1e-9
            features["signal_volatility_cluster_ratio"] = float(
                features["signal_return_volatility_short"] / long_vol
            )
        if "volume" in df.columns and len(df) > 1:
            prev_volume = df["volume"].iloc[-2]
            features["signal_volume_delta"] = float(latest["volume"] - prev_volume)

        try:
            strat_features = self.strategy.extract_features(df) or {}
        except Exception:  # pragma: no cover - defensive, strategy bug shouldn't crash
            strat_features = {}

        for key, value in strat_features.items():
            try:
                features[f"signal_{key}"] = float(value)
            except (TypeError, ValueError):
                continue

        return features

    # ------------------------------------------------------------------
    def record_trade(self, pnl: float) -> None:
        self.state["trades"] += 1
        self.state["pnl"] += pnl
        if pnl > 0:
            self.state["wins"] += 1
        else:
            self.state["losses"] += 1

