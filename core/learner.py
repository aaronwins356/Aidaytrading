import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class Learner:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self._observations_path = os.path.join(model_dir, "observations.csv")

    def retrain_worker(self, worker, trade_history=None):
        """
        Retrain model for a worker every N trades.
        trade_history: pandas.DataFrame with features and pnl outcome.
        """
        if trade_history is None or trade_history.empty:
            print(f"[LEARNER] No history available for {worker.name}, skipping retrain.")
            return

        # Build features (simplified: last candle OHLCV)
        features = trade_history[["open", "high", "low", "close", "volume"]].values
        labels = (trade_history["pnl"] > 0).astype(int)  # 1 = win, 0 = loss

        if len(features) < 20:
            print(f"[LEARNER] Not enough data for {worker.name}, skipping retrain.")
            return

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(features, labels)

        path = os.path.join(self.model_dir, f"{worker.name}.pkl")
        joblib.dump(model, path)

        print(f"[LEARNER] Retrained model for {worker.name}, saved to {path}")

    def observe(self, trade):
        """Persist basic trade information for future ML retraining."""
        if not trade:
            return

        required_keys = {"timestamp", "symbol", "side", "qty", "entry_price"}
        if not required_keys.issubset(trade.keys()):
            return

        row = {
            "timestamp": float(trade.get("timestamp", 0.0)),
            "symbol": trade.get("symbol"),
            "side": trade.get("side"),
            "qty": float(trade.get("qty", 0.0)),
            "entry_price": float(trade.get("entry_price", 0.0)),
            "worker": trade.get("worker"),
        }

        try:
            if os.path.exists(self._observations_path):
                existing = pd.read_csv(self._observations_path)
                updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
            else:
                updated = pd.DataFrame([row])
            updated.to_csv(self._observations_path, index=False)
        except Exception as exc:  # pragma: no cover - best effort logging
            print(f"[LEARNER] Failed to persist observation: {exc}")

    def predict_edge(self, worker, latest_candle):
        """
        Predict trade edge using worker's ML model.
        latest_candle: dict {open, high, low, close, volume}
        """
        path = os.path.join(self.model_dir, f"{worker.name}.pkl")
        if not os.path.exists(path):
            return 0.5  # neutral if no model yet

        model = joblib.load(path)
        x = np.array([[latest_candle["open"], latest_candle["high"],
                       latest_candle["low"], latest_candle["close"],
                       latest_candle["volume"]]])
        prob = model.predict_proba(x)[0, 1]  # probability of win
        return prob
