"""Lightweight online ML pipeline for trading signals."""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np


@dataclass(slots=True)
class FeatureVector:
    """Structured representation of engineered features."""

    symbol: str
    timestamp: str
    features: List[float]
    label: Optional[float]


class OnlineLogisticModel:
    """Simple online logistic regression for streaming updates."""

    def __init__(self, feature_dim: int, learning_rate: float, regularization: float) -> None:
        self.feature_dim = feature_dim
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.weights = np.zeros(feature_dim, dtype=float)
        self.bias = 0.0

    def predict(self, features: Sequence[float]) -> float:
        vector = np.asarray(features, dtype=float)
        z = float(np.dot(self.weights, vector) + self.bias)
        z = max(min(z, 35.0), -35.0)  # numerical stability
        return 1.0 / (1.0 + math.exp(-z))

    def update(self, features: Sequence[float], label: float) -> None:
        vector = np.asarray(features, dtype=float)
        prediction = self.predict(vector)
        error = prediction - label
        self.weights -= self.learning_rate * (error * vector + self.regularization * self.weights)
        self.bias -= self.learning_rate * error


class MLPipeline:
    """Persist features and keep models updated for downstream workers."""

    def __init__(
        self,
        db_path: Path,
        feature_keys: Sequence[str],
        learning_rate: float,
        regularization: float,
        batch_size: int,
        max_training_rows: int,
    ) -> None:
        self._db_path = db_path
        self._feature_keys = list(feature_keys)
        self._learning_rate = learning_rate
        self._regularization = regularization
        self._batch_size = batch_size
        self._max_training_rows = max_training_rows
        self._models: Dict[str, OnlineLogisticModel] = {}
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ml_models (
                    symbol TEXT PRIMARY KEY,
                    weights TEXT NOT NULL,
                    bias REAL NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _feature_vector_from_row(self, row: sqlite3.Row) -> FeatureVector:
        payload = json.loads(row["features_json"])
        features = [float(payload.get(key, 0.0)) for key in self._feature_keys]
        label = row["label"]
        label_value = float(label) if label is not None else None
        return FeatureVector(
            symbol=row["symbol"], timestamp=row["timestamp"], features=features, label=label_value
        )

    def fetch_recent_features(self, symbol: str, limit: int = 500) -> List[FeatureVector]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT timestamp, symbol, features_json, label
                FROM market_features
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (symbol, limit),
            )
            rows = cursor.fetchall()
        return [self._feature_vector_from_row(row) for row in rows]

    def latest_feature(self, symbol: str) -> Optional[FeatureVector]:
        features = self.fetch_recent_features(symbol, limit=1)
        return features[0] if features else None

    def _load_model(self, symbol: str, feature_dim: int) -> OnlineLogisticModel:
        model = self._models.get(symbol)
        if model:
            return model
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT weights, bias FROM ml_models WHERE symbol = ?", (symbol,)
            ).fetchone()
        if row:
            weights = np.asarray(json.loads(row["weights"]), dtype=float)
            bias = float(row["bias"])
            model = OnlineLogisticModel(feature_dim, self._learning_rate, self._regularization)
            if len(weights) == feature_dim:
                model.weights = weights
            model.bias = bias
        else:
            model = OnlineLogisticModel(feature_dim, self._learning_rate, self._regularization)
        self._models[symbol] = model
        return model

    def _persist_model(self, symbol: str, model: OnlineLogisticModel) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ml_models(symbol, weights, bias, updated_at)
                VALUES(?, ?, ?, datetime('now'))
                ON CONFLICT(symbol) DO UPDATE SET
                    weights = excluded.weights,
                    bias = excluded.bias,
                    updated_at = excluded.updated_at
                """,
                (symbol, json.dumps(model.weights.tolist()), model.bias),
            )
            conn.commit()

    def train(self, symbol: str) -> None:
        dataset = self.fetch_recent_features(symbol, limit=self._max_training_rows)
        if not dataset:
            return
        model = self._load_model(symbol, len(self._feature_keys))
        batch = list(reversed(dataset[: self._batch_size]))
        for vector in batch:
            if vector.label is None:
                continue
            model.update(vector.features, vector.label)
        self._persist_model(symbol, model)

    def predict(self, symbol: str, features: Sequence[float]) -> float:
        model = self._load_model(symbol, len(features))
        return model.predict(features)

    @property
    def feature_keys(self) -> Sequence[str]:
        return self._feature_keys
