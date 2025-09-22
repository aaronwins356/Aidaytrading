"""ML learner scaffolding for nightly retraining."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pickle

try:  # pragma: no cover - import guard
    import joblib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised in tests
    class _JoblibShim:
        @staticmethod
        def dump(obj, path):
            with open(path, "wb") as handle:
                pickle.dump(obj, handle)

        @staticmethod
        def load(path):
            with open(path, "rb") as handle:
                return pickle.load(handle)

    joblib = _JoblibShim()  # type: ignore

try:  # pragma: no cover - import guard
    import numpy as np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised in tests
    np = None  # type: ignore

try:  # pragma: no cover - import guard
    import pandas as pd  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised in tests
    pd = None  # type: ignore

try:  # pragma: no cover - import guard
    from sklearn.ensemble import RandomForestClassifier  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised in tests
    class RandomForestClassifier:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "scikit-learn is required for model training but is not installed"
            )

from desk.config import DESK_ROOT
from desk.services.pretty_logger import pretty_logger

class Learner:
    """Lightweight model registry + inference helper."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        target_win_rate: float = 0.55,
        min_samples: int = 50,
    ) -> None:
        model_dir = Path(model_dir or DESK_ROOT / "models")
        model_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = model_dir
        self._observations_path = self.model_dir / "observations.csv"
        self._history_cache: Dict[str, Path] = {}
        self.target_win_rate = max(0.0, min(1.0, float(target_win_rate)))
        self.min_samples = max(20, int(min_samples))
        self._threshold_cache: Dict[str, float] = {}

    def retrain_worker(self, worker, trade_history: Optional[pd.DataFrame] = None) -> None:
        """
        Retrain model for a worker every N trades.
        trade_history: pandas.DataFrame with features and pnl outcome.
        """
        history = trade_history if trade_history is not None else self.load_trade_history(worker.name)
        if history is None or history.empty:
            pretty_logger.info(
                f"[Learner] No history available for {worker.name}, skipping retrain."
            )
            return

        if "pnl" not in history.columns:
            pretty_logger.warning(
                f"[Learner] History for {worker.name} missing pnl column; skipping retrain."
            )
            return

        feature_cols = [
            col
            for col in history.columns
            if col.startswith("candle_")
            or col.startswith("signal_")
            or col in {"ml_edge", "combined_score", "proposed_qty", "risk_budget", "side"}
        ]
        if not feature_cols:
            pretty_logger.warning(
                f"[Learner] No usable features for {worker.name}, skipping retrain."
            )
            return

        features = history[feature_cols]
        features = features.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        labels = (history["pnl"] > 0).astype(int)

        if len(features) < self.min_samples:
            pretty_logger.info(
                f"[Learner] Not enough data for {worker.name}, skipping retrain."
            )
            return

        if np is None:
            pretty_logger.warning(
                f"[Learner] numpy unavailable, cannot retrain {worker.name} right now."
            )
            return

        feature_matrix = features.values
        label_array = labels.values

        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(feature_matrix, label_array)

        path = self.model_dir / f"{worker.name}.pkl"
        joblib.dump(model, path)

        feature_path = self.model_dir / f"{worker.name}_features.json"
        feature_path.write_text(json.dumps(list(features.columns)))

        probabilities = model.predict_proba(feature_matrix)[:, 1]
        threshold = self._calibrate_threshold(probabilities, label_array)
        if threshold is not None:
            self._threshold_cache[worker.name] = threshold
            self._write_threshold(worker.name, threshold)
            pretty_logger.info(
                f"[Learner] Calibrated {worker.name} threshold to {threshold:.4f}"
            )
        else:
            self._threshold_cache.pop(worker.name, None)
            self._write_threshold(worker.name, None)

        pretty_logger.info(
            f"[Learner] Retrained model for {worker.name}, saved to {path}"
        )

    def observe(self, trade: Dict[str, object]) -> None:
        """Persist basic trade information for future ML retraining."""
        if not trade:
            return

        required_keys = {"timestamp", "symbol", "side", "qty", "entry_price"}
        if not required_keys.issubset(trade.keys()):
            return

        features = trade.get("features") or {}
        row = {
            "timestamp": float(trade.get("timestamp", 0.0)),
            "symbol": trade.get("symbol"),
            "side": trade.get("side"),
            "qty": float(trade.get("qty", 0.0)),
            "entry_price": float(trade.get("entry_price", 0.0)),
            "worker": trade.get("worker"),
        }
        if "ml_edge" in trade:
            try:
                row["ml_edge"] = float(trade["ml_edge"])
            except (TypeError, ValueError):
                pass
        if "score" in trade:
            try:
                row["score"] = float(trade["score"])
            except (TypeError, ValueError):
                pass
        for key, value in features.items():
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                continue

        try:
            if pd is None:
                return
            frame = pd.DataFrame([row])
            path = self._observations_path
            frame.to_csv(
                path,
                index=False,
                mode="a",
                header=not path.exists(),
            )
        except Exception as exc:  # pragma: no cover - best effort logging
            pretty_logger.warning(f"[Learner] Failed to persist observation: {exc}")


    def _threshold_path(self, worker_name: str) -> Path:
        return self.model_dir / f"{worker_name}_threshold.json"

    def _write_threshold(self, worker_name: str, value: float | None) -> None:
        path = self._threshold_path(worker_name)
        if value is None:
            if path.exists():
                path.unlink()
            return
        payload = {
            "threshold": float(value),
            "target": self.target_win_rate,
        }
        path.write_text(json.dumps(payload))

    def _load_threshold(self, worker_name: str) -> Optional[float]:
        if worker_name in self._threshold_cache:
            return self._threshold_cache[worker_name]
        path = self._threshold_path(worker_name)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
        threshold = payload.get("threshold")
        if threshold is None:
            return None
        try:
            value = float(threshold)
        except (TypeError, ValueError):
            return None
        self._threshold_cache[worker_name] = value
        return value

    def _calibrate_threshold(self, probabilities, labels) -> Optional[float]:
        if np is None or probabilities is None:
            return None
        try:
            probs = np.asarray(probabilities, dtype=float)
            lbls = np.asarray(labels, dtype=float)
        except Exception:
            return None
        if probs.size == 0:
            return None
        order = np.argsort(probs)[::-1]
        sorted_probs = probs[order]
        sorted_labels = lbls[order]
        cumulative_wins = np.cumsum(sorted_labels)
        counts = np.arange(1, len(sorted_probs) + 1)
        win_rates = np.divide(
            cumulative_wins, counts, out=np.zeros_like(cumulative_wins), where=counts != 0
        )
        mask = win_rates >= self.target_win_rate if self.target_win_rate > 0 else np.array([])
        if mask.size and mask.any():
            idx = int(np.argmax(mask))
            return float(sorted_probs[idx])
        return float(np.median(sorted_probs))

    def predict_edge(self, worker, features: Dict[str, float]) -> float:
        """
        Predict trade edge using worker's ML model.
        ``features`` should match the columns used during training.
        """
        path = self.model_dir / f"{worker.name}.pkl"
        if not path.exists():
            return 0.5  # neutral if no model yet

        model = joblib.load(path)
        feature_path = self.model_dir / f"{worker.name}_features.json"
        if not feature_path.exists():
            return 0.5

        try:
            required = json.loads(feature_path.read_text())
        except json.JSONDecodeError:
            return 0.5

        vector = []
        for key in required:
            try:
                vector.append(float(features.get(key, 0.0)))
            except (TypeError, ValueError):
                vector.append(0.0)

        if not vector:
            return 0.5

        if np is None:
            return 0.5

        x = np.array([vector])
        prob = float(model.predict_proba(x)[0, 1])  # probability of win
        threshold = self._load_threshold(worker.name)
        if threshold is None:
            return prob
        adjusted = 0.5 + (prob - threshold)
        return float(max(0.0, min(1.0, adjusted)))

    # ------------------------------------------------------------------
    def load_trade_history(self, worker_name: str) -> Optional[pd.DataFrame]:
        if pd is None:
            return None
        path = self._history_path(worker_name)
        if not path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(path)
            return df.apply(pd.to_numeric, errors="ignore")
        except Exception as exc:  # pragma: no cover - defensive I/O guard
            pretty_logger.warning(
                f"[Learner] Failed to load history for {worker_name}: {exc}"
            )
            return pd.DataFrame()

    def record_result(self, worker_name: str, row: Dict[str, float]) -> None:
        if pd is None:
            return
        path = self._history_path(worker_name)
        data = pd.DataFrame([row])
        try:
            if path.exists():
                existing = pd.read_csv(path)
                updated = pd.concat([existing, data], ignore_index=True)
            else:
                updated = data
            updated.to_csv(path, index=False)
        except Exception as exc:  # pragma: no cover - best effort logging
            pretty_logger.warning(
                f"[Learner] Failed to persist history for {worker_name}: {exc}"
            )

    def _history_path(self, worker_name: str) -> Path:
        if worker_name not in self._history_cache:
            safe_name = worker_name.replace("/", "_").replace(" ", "_")
            self._history_cache[worker_name] = self.model_dir / f"{safe_name}_history.csv"
        return self._history_cache[worker_name]
