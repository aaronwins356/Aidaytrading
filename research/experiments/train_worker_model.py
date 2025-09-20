import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

from services.model_store import ModelStore
from strategies.vwap_revert import VWAPRevertStrategy
from strategies.momentum_breakout import MomentumBreakoutStrategy

# Map worker strategies to their feature extractors
STRATEGY_FEATURES = {
    "vwap_revert": VWAPRevertStrategy,
    "momentum_breakout": MomentumBreakoutStrategy,
}

LOGS_PATH = Path("logs/events.jsonl")
MODELS_PATH = Path("models")

def load_logs():
    """Read logs/events.jsonl into a DataFrame."""
    if not LOGS_PATH.exists():
        print("No logs found.")
        return pd.DataFrame()

    rows = []
    with open(LOGS_PATH, "r") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return pd.DataFrame(rows)


def build_training_data(df):
    """
    Turn logs into supervised ML data.
    Assumes 'order_fill' events can be joined with pnl outcomes later.
    """
    if df.empty:
        return {}

    # For now, we simulate labels: profitable if direction*qty*random factor
    # In a real pipeline, you'd join fills with portfolio PnL outcomes.
    worker_datasets = {}
    for worker, wdf in df.groupby("worker"):
        features, labels = [], []

        for _, row in wdf.iterrows():
            if row["event"] != "order_fill":
                continue

            # Strategy-driven features
            strat_type = "vwap_revert" if "vwap" in worker else "momentum_breakout"
            strat_cls = STRATEGY_FEATURES[strat_type]
            strat = strat_cls(row["symbol"], {})
            # Here you would normally re-extract features from price history.
            # Placeholder: fake features from qty/price
            feat = [row["qty"], row["price"]]
            label = 1 if row["qty"] > 0 else 0

            features.append(feat)
            labels.append(label)

        if features:
            worker_datasets[worker] = (np.array(features), np.array(labels))

    return worker_datasets


def train_and_save_models(worker_datasets):
    store = ModelStore()
    MODELS_PATH.mkdir(parents=True, exist_ok=True)

    for worker, (X, y) in worker_datasets.items():
        if len(set(y)) < 2:
            print(f"Skipping {worker}, not enough class variety.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = LogisticRegression()
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"{worker}: trained LogisticRegression, acc={acc:.2f}")

        # Save with ModelStore
        store.save(worker, model)


def main():
    df = load_logs()
    worker_datasets = build_training_data(df)
    train_and_save_models(worker_datasets)


if __name__ == "__main__":
    main()
