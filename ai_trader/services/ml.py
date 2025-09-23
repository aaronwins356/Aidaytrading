"""Online machine learning service for streaming trade intelligence."""

from __future__ import annotations

import json
import pickle
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Tuple

try:  # pragma: no cover - import guard for optional forest backend
    from river import forest
except ImportError:  # pragma: no cover - river build without forest module
    forest = None  # type: ignore[assignment]

from river import compose, linear_model, optim, preprocessing

from ai_trader.services.logging import get_logger


FeatureMapping = Mapping[str, float]


@dataclass(slots=True)
class _ModelBundle:
    """Container holding the live online models for a trading symbol."""

    pipeline: compose.Pipeline
    forest: Optional[Any]

    def learn(self, features: FeatureMapping, label: int) -> None:
        """Update the online models with the observed label."""

        self.pipeline.learn_one(features, label)
        if self.forest is not None:
            self.forest.learn_one(features, label)

    def predict_proba(self, features: FeatureMapping) -> float:
        """Return a blended probability using the configured ensemble."""

        probability = self.pipeline.predict_proba_one(features).get(True, 0.0)
        if self.forest is not None:
            forest_probability = self.forest.predict_proba_one(features).get(True, 0.0)
            probability = (probability + forest_probability) / 2
        return float(max(0.0, min(1.0, probability)))

    def feature_importances(self) -> Dict[str, float]:
        """Expose the logistic regression coefficients for explainability."""

        weights: Dict[str, float] = {}
        try:
            model: linear_model.LogisticRegression = self.pipeline[-1]
        except (IndexError, TypeError, AttributeError):  # pragma: no cover - defensive fallback
            return weights
        for feature, weight in model.weights.items():
            weights[str(feature)] = float(weight)
        return weights


class MLService:
    """Stateful online learning engine shared across workers."""

    def __init__(
        self,
        db_path: Path,
        feature_keys: Iterable[str],
        learning_rate: float = 0.03,
        regularization: float = 0.0005,
        threshold: float = 0.2,
        ensemble: bool = True,
        forest_size: int = 15,
        random_state: int = 7,
    ) -> None:
        self._db_path = db_path
        self._feature_keys = list(feature_keys)
        self._learning_rate = learning_rate
        self._regularization = regularization
        self._threshold = threshold
        self._logger = get_logger(__name__)
        self._ensemble_requested = bool(ensemble)
        self._forest_backend = "river.forest.ARFClassifier" if forest and hasattr(forest, "ARFClassifier") else None
        self._use_ensemble = self._ensemble_requested and self._forest_backend is not None
        self._forest_size = forest_size
        self._random_state = random_state
        self._models: Dict[str, _ModelBundle] = {}
        self._latest_features: Dict[str, Dict[str, float]] = {}
        self._latest_confidence: Dict[Tuple[str, str], float] = {}
        if self._ensemble_requested and not self._use_ensemble:
            self._logger.warning(
                "ML ensemble requested but river.forest.ARFClassifier is unavailable. Falling back to logistic regression only."
            )
        self._init_db()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ml_models_state (
                    symbol TEXT PRIMARY KEY,
                    logistic BLOB,
                    forest BLOB,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    worker TEXT,
                    confidence REAL NOT NULL,
                    decision INTEGER NOT NULL,
                    threshold REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ml_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    precision REAL,
                    recall REAL,
                    win_rate REAL,
                    support INTEGER
                )
                """
            )
            conn.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Model lifecycle management
    # ------------------------------------------------------------------
    def _build_pipeline(self) -> compose.Pipeline:
        optimizer = optim.SGD(lr=self._learning_rate)
        logistic = linear_model.LogisticRegression(optimizer=optimizer, l2=self._regularization)
        return compose.Pipeline(preprocessing.StandardScaler(), logistic)

    def _build_forest(self) -> forest.ARFClassifier:
        if forest is None or not hasattr(forest, "ARFClassifier"):
            raise RuntimeError("river.forest.ARFClassifier is unavailable in this environment")
        return forest.ARFClassifier(
            n_models=self._forest_size,
            max_features="sqrt",
            seed=self._random_state,
        )

    def _load_model(self, symbol: str) -> _ModelBundle:
        if symbol in self._models:
            return self._models[symbol]

        pipeline = self._build_pipeline()
        forest = self._build_forest() if self._use_ensemble else None

        with self._connect() as conn:
            row = conn.execute(
                "SELECT logistic, forest FROM ml_models_state WHERE symbol = ?",
                (symbol,),
            ).fetchone()
        if row and row["logistic"]:
            try:
                pipeline = pickle.loads(row["logistic"])
            except Exception as exc:  # noqa: BLE001 - fallback to fresh model
                self._logger.warning("Failed to load logistic model for %s: %s", symbol, exc)
                pipeline = self._build_pipeline()
        if row and row["forest"] and self._use_ensemble:
            try:
                forest = pickle.loads(row["forest"])
            except Exception as exc:  # noqa: BLE001 - fallback to fresh model
                self._logger.warning("Failed to load forest model for %s: %s", symbol, exc)
                forest = self._build_forest()

        bundle = _ModelBundle(pipeline=pipeline, forest=forest)
        self._models[symbol] = bundle
        return bundle

    def _persist_model(self, symbol: str, bundle: _ModelBundle) -> None:
        logistic_blob = pickle.dumps(bundle.pipeline)
        forest_blob = pickle.dumps(bundle.forest) if bundle.forest is not None else None
        metadata = json.dumps(
            {"ensemble": self._use_ensemble, "backend": self._forest_backend}
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ml_models_state(symbol, logistic, forest, updated_at, metadata_json)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    logistic = excluded.logistic,
                    forest = excluded.forest,
                    updated_at = excluded.updated_at,
                    metadata_json = excluded.metadata_json
                """,
                (
                    symbol,
                    logistic_blob,
                    forest_blob,
                    datetime.utcnow().isoformat(),
                    metadata,
                ),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def feature_keys(self) -> Iterable[str]:
        return tuple(self._feature_keys)

    @property
    def default_threshold(self) -> float:
        return self._threshold

    @property
    def ensemble_requested(self) -> bool:
        return self._ensemble_requested

    @property
    def ensemble_available(self) -> bool:
        return self._use_ensemble

    @property
    def ensemble_backend(self) -> str:
        return self._forest_backend or "disabled"

    def has_feature_history(self, symbols: Optional[Iterable[str]] = None) -> bool:
        """Return True when at least one engineered feature row exists."""

        query = "SELECT 1 FROM market_features"
        params: Tuple[str, ...] = ()
        if symbols:
            symbol_list = tuple(str(symbol) for symbol in symbols)
            if not symbol_list:
                return False
            placeholders = ", ".join(["?"] * len(symbol_list))
            query += f" WHERE symbol IN ({placeholders})"
            params = symbol_list
        query += " LIMIT 1"
        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()
        return row is not None

    def update(
        self,
        symbol: str,
        features: Mapping[str, float],
        label: Optional[float] = None,
        *,
        persist: bool = True,
        timestamp: Optional[datetime] = None,
    ) -> float:
        """Update the models with a new feature vector and optional label.

        Returns the current confidence so researcher workers can monitor drift.
        """

        cleaned = self._sanitize_features(features)
        feature_count = len(cleaned)
        self._logger.debug(
            "Received %d engineered features for %s", feature_count, symbol
        )
        self._latest_features[symbol] = cleaned
        model = self._load_model(symbol)

        probability = model.predict_proba(cleaned)
        if label is None:
            self._logger.warning(
                "No label supplied for %s – skipping learning step (confidence=%.4f)",
                symbol,
                probability,
            )
        else:
            model.learn(cleaned, int(label))
            probability = model.predict_proba(cleaned)
            if persist:
                self._persist_model(symbol, model)

        decision = int(probability >= self._threshold)
        self._latest_confidence[("researcher", symbol)] = probability
        self._logger.info(f"[ML] {symbol} confidence={probability:.3f} decision={decision}")
        self._logger.info(
            "ML UPDATE | symbol=%s confidence=%.2f decision=%d features=%d",
            symbol,
            probability,
            decision,
            feature_count,
        )
        self._record_prediction(
            symbol=symbol,
            worker="researcher",
            probability=probability,
            decision=decision,
            threshold=self._threshold,
            timestamp=timestamp or datetime.utcnow(),
        )
        return probability

    def predict(
        self,
        symbol: str,
        features: Optional[Mapping[str, float]] = None,
        *,
        worker: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """Return (decision, confidence) for the supplied feature vector."""

        model = self._load_model(symbol)
        feature_payload: Mapping[str, float] | None = features or self._latest_features.get(symbol)
        if feature_payload is None:
            self._logger.warning(
                "Prediction skipped for %s – no features available yet", symbol
            )
            return False, 0.0

        cleaned = self._sanitize_features(feature_payload)
        probability = model.predict_proba(cleaned)
        gate = threshold if threshold is not None else self._threshold
        decision = probability >= gate
        worker_name = worker or "worker"
        self._latest_confidence[(worker_name, symbol)] = probability
        self._logger.info(f"[ML] {symbol} confidence={probability:.3f} decision={int(decision)}")
        feature_count = len(cleaned)
        self._logger.info(
            "ML PREDICT | worker=%s symbol=%s confidence=%.2f threshold=%.2f decision=%d features=%d",
            worker_name,
            symbol,
            probability,
            gate,
            int(decision),
            feature_count,
        )
        self._record_prediction(
            symbol=symbol,
            worker=worker_name,
            probability=probability,
            decision=int(decision),
            threshold=gate,
            timestamp=datetime.utcnow(),
        )
        return decision, probability

    def latest_confidence(self, symbol: str, worker: Optional[str] = None) -> float:
        """Expose the last scored confidence for dashboards."""

        key = (worker or "worker", symbol)
        return self._latest_confidence.get(key, 0.0)

    def latest_features(self, symbol: str) -> Optional[Dict[str, float]]:
        """Return the last seen feature vector for a symbol."""

        return self._latest_features.get(symbol)

    def feature_importance(self, symbol: str, top_n: int = 10) -> Dict[str, float]:
        """Return the top-N absolute weighted features for the symbol."""

        model = self._load_model(symbol)
        weights = model.feature_importances()
        if not weights:
            return {}
        sorted_items = sorted(weights.items(), key=lambda item: abs(item[1]), reverse=True)
        return dict(sorted_items[:top_n])

    def confidence_history(self, symbol: str, limit: int = 200) -> list[tuple[datetime, float, str]]:
        """Fetch recent confidence readings for charting."""

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, confidence, worker
                FROM ml_predictions
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (symbol, limit),
            ).fetchall()
        history: list[tuple[datetime, float, str]] = []
        for row in rows:
            history.append((datetime.fromisoformat(row["timestamp"]), float(row["confidence"]), row["worker"]))
        return list(reversed(history))

    def run_backtest(
        self,
        symbol: str,
        limit: int = 1500,
        *,
        threshold: Optional[float] = None,
        warmup: int = 25,
    ) -> Dict[str, float]:
        """Replay stored features and return expanded classification metrics.

        Args:
            symbol: Trading symbol whose historical features should be replayed.
            limit: Maximum number of stored feature rows to inspect.
            threshold: Optional gating threshold override for the probability output.
            warmup: Number of initial samples used only to prime the model without scoring.
        """

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, features_json, label
                FROM market_features
                WHERE symbol = ? AND label IS NOT NULL
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (symbol, limit),
            ).fetchall()

        if not rows:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "win_rate": 0.0,
                "support": 0,
                "accuracy": 0.0,
                "f1_score": 0.0,
                "avg_confidence": 0.0,
                "threshold": float(threshold if threshold is not None else self._threshold),
                "trades": 0,
                "warmup": max(0, warmup),
            }

        test_model = _ModelBundle(pipeline=self._build_pipeline(), forest=self._build_forest() if self._use_ensemble else None)
        gate = threshold if threshold is not None else self._threshold

        stats = _ClassificationStats()
        probabilities: list[float] = []
        warmup_span = max(0, warmup)

        for index, row in enumerate(rows):
            payload = json.loads(row["features_json"])
            features = self._sanitize_features(payload)
            label = int(row["label"])
            if index < warmup_span:
                # Warm-up period: let the models adjust to the most recent distribution
                # before we start evaluating the trading signals.
                test_model.learn(features, label)
                continue

            proba = test_model.predict_proba(features)
            probabilities.append(float(proba))
            prediction = int(proba >= gate)
            stats.update(prediction=prediction, label=label)
            test_model.learn(features, label)

        precision = stats.precision()
        recall = stats.recall()
        win_rate = stats.win_rate()
        accuracy = stats.accuracy()
        f1_score = stats.f1_score()
        avg_confidence = sum(probabilities) / len(probabilities) if probabilities else 0.0

        metrics_payload = {
            "precision": precision,
            "recall": recall,
            "win_rate": win_rate,
            "support": int(stats.support),
            "accuracy": accuracy,
            "f1_score": f1_score,
            "avg_confidence": avg_confidence,
            "threshold": float(gate),
            "trades": int(stats.trades),
            "warmup": warmup_span,
        }

        self._logger.info(
            "Backtest completed for %s – precision=%.3f recall=%.3f win_rate=%.3f accuracy=%.3f f1=%.3f trades=%d warmup=%d",
            symbol,
            precision,
            recall,
            win_rate,
            accuracy,
            f1_score,
            stats.trades,
            warmup_span,
        )

        self._record_metrics(symbol, "backtest", metrics_payload)
        return metrics_payload

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _sanitize_features(self, features: Mapping[str, float]) -> Dict[str, float]:
        sanitized: Dict[str, float] = {}
        for key in self._feature_keys:
            value = float(features.get(key, 0.0))
            if value != value:  # NaN check without importing math
                value = 0.0
            sanitized[key] = value
        return sanitized

    def _record_prediction(
        self,
        *,
        symbol: str,
        worker: str,
        probability: float,
        decision: int,
        threshold: float,
        timestamp: datetime,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ml_predictions(timestamp, symbol, worker, confidence, decision, threshold)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp.isoformat(),
                    symbol,
                    worker,
                    float(probability),
                    int(decision),
                    float(threshold),
                ),
            )
            conn.commit()

    def _record_metrics(self, symbol: str, mode: str, metrics: Mapping[str, float]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ml_metrics(timestamp, symbol, mode, precision, recall, win_rate, support)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.utcnow().isoformat(),
                    symbol,
                    mode,
                    float(metrics.get("precision", 0.0)),
                    float(metrics.get("recall", 0.0)),
                    float(metrics.get("win_rate", 0.0)),
                    int(metrics.get("support", 0)),
                ),
            )
            conn.commit()

@dataclass(slots=True)
class _ClassificationStats:
    """Lightweight container to keep track of classification outcomes."""

    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0

    def update(self, *, prediction: int, label: int) -> None:
        """Update the confusion counts based on a single prediction."""

        if prediction == 1 and label == 1:
            self.true_positive += 1
        elif prediction == 1 and label == 0:
            self.false_positive += 1
        elif prediction == 0 and label == 0:
            self.true_negative += 1
        else:
            self.false_negative += 1

    @property
    def support(self) -> int:
        """Return the total number of scored examples."""

        return self.true_positive + self.false_positive + self.true_negative + self.false_negative

    @property
    def trades(self) -> int:
        """Return the number of trade signals emitted by the model."""

        return self.true_positive + self.false_positive

    def precision(self) -> float:
        """Return the ratio of profitable trades to all executed trades."""

        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    def recall(self) -> float:
        """Return the hit rate on all opportunities present in the data."""

        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    def accuracy(self) -> float:
        """Return the fraction of correct predictions across the sample."""

        return (self.true_positive + self.true_negative) / self.support if self.support else 0.0

    def win_rate(self) -> float:
        """Return the same notion of win rate used by the dashboard."""

        return self.precision()

    def f1_score(self) -> float:
        """Return the harmonic mean of precision and recall."""

        precision = self.precision()
        recall = self.recall()
        return (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

