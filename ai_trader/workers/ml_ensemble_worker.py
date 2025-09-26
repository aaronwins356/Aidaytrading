"""Machine learning ensemble worker combining classical and deep models."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from ai_trader.services.types import MarketSnapshot, OpenPosition, TradeIntent
from ai_trader.workers.base import BaseWorker


torch.set_num_threads(1)


class _SequenceClassifier(nn.Module):
    """Small LSTM-based classifier for directional forecasting."""

    def __init__(self, feature_dim: int, hidden_size: int = 16) -> None:
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        output, _ = self.lstm(x)
        final = output[:, -1, :]
        logits = self.fc(final)
        return self.sigmoid(logits)


@dataclass
class _ModelBundle:
    logistic: LogisticRegression
    forest: RandomForestClassifier
    lstm: _SequenceClassifier
    scaler: StandardScaler
    sequence_length: int


class EnsembleMLWorker(BaseWorker):
    """Worker that ensembles multiple ML models to produce trade intents."""

    name = "ML Ensemble"
    emoji = "🧠"
    long_only = True
    strategy_brief = "Combines logistic regression, random forest, and LSTM forecasts using Sharpe-weighted averaging."

    def __init__(
        self,
        symbols: Iterable[str],
        *,
        window_size: int = 150,
        retrain_interval: int = 20,
        sequence_length: int = 12,
        min_history: int = 80,
        ml_service: Optional[object] = None,
    ) -> None:
        super().__init__(symbols=symbols, lookback=window_size, ml_service=ml_service)
        self.window_size = max(window_size, 40)
        self.retrain_interval = max(5, retrain_interval)
        self.sequence_length = max(4, sequence_length)
        self.min_history = max(min_history, self.sequence_length + 10)
        self._bundles: Dict[str, _ModelBundle] = {}
        self._weights: Dict[str, Dict[str, float]] = {}
        self._trained_length: Dict[str, int] = {}
        self._latest_prob: Dict[str, float] = {}
        self._feature_cache: Dict[str, deque] = {
            symbol: deque(maxlen=self.window_size) for symbol in self.symbols
        }
        self._lstm_feature_cache: Dict[str, deque] = {}

    async def evaluate_signal(self, snapshot: MarketSnapshot) -> Dict[str, str]:
        self.update_history(snapshot)
        signals: Dict[str, str] = {}
        for symbol in self.symbols:
            candles = snapshot.candles.get(symbol) or []
            if len(candles) < self.min_history:
                self.update_signal_state(symbol, None, {"status": "warming"})
                continue
            features = self._build_dataset(candles)
            if features is None or features[0].empty:
                self.update_signal_state(symbol, None, {"status": "insufficient-data"})
                continue
            df, label_returns = features
            latest_length = len(df)
            need_train = (
                symbol not in self._bundles
                or latest_length != self._trained_length.get(symbol)
                or latest_length % self.retrain_interval == 0
            )
            if need_train:
                bundle = self._train_models(symbol, df, label_returns)
                if bundle is None:
                    self.update_signal_state(symbol, None, {"status": "training-failed"})
                    continue
                self._bundles[symbol] = bundle
                self._trained_length[symbol] = latest_length
            bundle = self._bundles.get(symbol)
            if bundle is None:
                continue
            prob = self._predict_latest(symbol, df.iloc[-1], bundle)
            self._latest_prob[symbol] = prob
            weights = self._weights.get(symbol, {})
            signal = self._decide_signal(prob)
            indicators = {
                "ensemble_probability": prob,
                "weights": weights,
                "ema_fast": df.iloc[-1]["ema_fast"],
                "ema_slow": df.iloc[-1]["ema_slow"],
                "bollinger_z": df.iloc[-1]["bollinger_z"],
                "volatility": df.iloc[-1]["volatility"],
            }
            self.update_signal_state(symbol, signal, indicators)
            if signal != "hold":
                signals[symbol] = signal
        return signals

    async def generate_trade(
        self,
        symbol: str,
        signal: Optional[str],
        snapshot: MarketSnapshot,
        equity_per_trade: float,
        existing_position: Optional[OpenPosition] = None,
    ) -> Optional[TradeIntent]:
        if signal is None or signal == "hold":
            return None
        price = snapshot.prices.get(symbol)
        if price is None:
            return None
        prob = self._latest_prob.get(symbol, 0.5)
        if signal == "buy" and existing_position is None:
            metadata = {
                "signal": signal,
                "probability": prob,
                "weights": self._weights.get(symbol, {}),
            }
            return TradeIntent(
                worker=self.name,
                action="OPEN",
                symbol=symbol,
                side="buy",
                cash_spent=float(equity_per_trade),
                entry_price=price,
                confidence=prob,
                metadata=metadata,
            )
        if signal == "sell" and existing_position is not None:
            pnl = (price - existing_position.entry_price) * existing_position.quantity
            pnl_percent = (
                pnl / existing_position.cash_spent * 100 if existing_position.cash_spent else 0.0
            )
            metadata = {
                "signal": signal,
                "probability": prob,
                "weights": self._weights.get(symbol, {}),
            }
            return TradeIntent(
                worker=self.name,
                action="CLOSE",
                symbol=symbol,
                side="sell",
                cash_spent=existing_position.cash_spent,
                entry_price=existing_position.entry_price,
                exit_price=price,
                pnl_percent=pnl_percent,
                pnl_usd=pnl,
                confidence=prob,
                metadata=metadata,
            )
        return None

    # ------------------------------------------------------------------
    # Feature engineering and training
    # ------------------------------------------------------------------
    def _build_dataset(
        self, candles: Sequence[Mapping[str, float]]
    ) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        if len(candles) < self.min_history:
            return None
        df = pd.DataFrame(candles).astype(float)
        df = df.tail(self.window_size + 5).reset_index(drop=True)
        df["return_1"] = df["close"].pct_change()
        df["return_5"] = df["close"].pct_change(5)
        df["return_10"] = df["close"].pct_change(10)
        df["ema_fast"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=26, adjust=False).mean()
        df["ema_cross"] = df["ema_fast"] - df["ema_slow"]
        rolling_mean = df["close"].rolling(window=20).mean()
        rolling_std = df["close"].rolling(window=20).std(ddof=0).replace(0, np.nan)
        df["bollinger_z"] = (df["close"] - rolling_mean) / rolling_std
        df["volatility"] = df["return_1"].rolling(window=20).std(ddof=0)
        df["future_return"] = df["close"].shift(-1) / df["close"] - 1
        df["target"] = (df["future_return"] > 0).astype(int)
        df = df.dropna().reset_index(drop=True)
        if len(df) < self.sequence_length + 5:
            return None
        labels = df["target"].astype(int)
        returns = df["future_return"].astype(float)
        feature_cols = [
            "return_1",
            "return_5",
            "return_10",
            "ema_fast",
            "ema_slow",
            "ema_cross",
            "bollinger_z",
            "volatility",
        ]
        df_features = df[feature_cols]
        df_features = df_features.replace([np.inf, -np.inf], np.nan).dropna()
        labels = labels.loc[df_features.index]
        returns = returns.loc[df_features.index]
        if len(df_features) < self.sequence_length + 5:
            return None
        df_features = df_features.reset_index(drop=True)
        labels = labels.reset_index(drop=True)
        returns = returns.reset_index(drop=True)
        df_features["future_return"] = returns
        return df_features, returns

    def _train_models(
        self,
        symbol: str,
        dataset: pd.DataFrame,
        future_returns: pd.Series,
    ) -> Optional[_ModelBundle]:
        if dataset.empty:
            return None
        feature_cols = [col for col in dataset.columns if col != "future_return"]
        features = dataset[feature_cols]
        labels = (dataset["future_return"] > 0).astype(int)
        if labels.sum() == 0 or labels.sum() == len(labels):
            # Avoid degenerate training sets.
            return None
        scaler = StandardScaler()
        X = scaler.fit_transform(features.values)
        y = labels.values
        logistic = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=400,
            random_state=7,
        )
        logistic.fit(X, y)
        forest = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=7,
            class_weight="balanced",
        )
        forest.fit(X, y)
        lstm = _SequenceClassifier(feature_dim=X.shape[1], hidden_size=16)
        self._train_lstm(lstm, X, y, sequence_length=self.sequence_length)
        bundle = _ModelBundle(
            logistic=logistic,
            forest=forest,
            lstm=lstm,
            scaler=scaler,
            sequence_length=self.sequence_length,
        )
        self._lstm_feature_cache.pop(symbol, None)
        self._update_model_weights(symbol, bundle, X, y, future_returns.values)
        return bundle

    def _train_lstm(
        self,
        model: _SequenceClassifier,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sequence_length: int,
        epochs: int = 6,
    ) -> None:
        if len(X) <= sequence_length:
            return
        sequences, targets = self._build_sequences(X, y, sequence_length)
        if len(sequences) == 0:
            return
        dataset = TensorDataset(
            torch.from_numpy(sequences).float(), torch.from_numpy(targets).float()
        )
        loader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        for _ in range(epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                preds = model(batch_x).squeeze()
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()

    def _build_sequences(
        self, X: np.ndarray, y: np.ndarray, sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        sequences: List[np.ndarray] = []
        targets: List[np.ndarray] = []
        for idx in range(sequence_length, len(X)):
            window = X[idx - sequence_length : idx]
            sequences.append(window)
            targets.append(np.array([y[idx]]))
        if not sequences:
            return np.empty((0, sequence_length, X.shape[1])), np.empty((0,))
        seq_array = np.stack(sequences)
        target_array = np.concatenate(targets)
        return seq_array, target_array

    def _update_model_weights(
        self,
        symbol: str,
        bundle: _ModelBundle,
        X: np.ndarray,
        y: np.ndarray,
        future_returns: np.ndarray,
    ) -> None:
        weights: Dict[str, float] = {}
        metrics: Dict[str, float] = {}
        preds_log = bundle.logistic.predict_proba(X)[:, 1]
        preds_for = bundle.forest.predict_proba(X)[:, 1]
        preds_lstm = self._predict_lstm_sequence(bundle, X, y)
        returns = future_returns
        metrics["logistic"] = self._sharpe_ratio(preds_log, returns)
        metrics["forest"] = self._sharpe_ratio(preds_for, returns)
        metrics["lstm"] = self._sharpe_ratio(preds_lstm, returns)
        total = sum(max(0.0, value) for value in metrics.values())
        if total <= 0:
            weights = {model: 1 / len(metrics) for model in metrics}
        else:
            weights = {model: max(0.0, value) / total for model, value in metrics.items()}
        self._weights[symbol] = weights

    def _predict_lstm_sequence(
        self, bundle: _ModelBundle, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        if len(X) <= bundle.sequence_length:
            return np.full(len(X), 0.5)
        sequences, _ = self._build_sequences(X, y, bundle.sequence_length)
        if len(sequences) == 0:
            return np.full(len(X), 0.5)
        loader = DataLoader(TensorDataset(torch.from_numpy(sequences).float()), batch_size=64)
        preds: List[float] = []
        bundle.lstm.eval()
        with torch.no_grad():
            for (batch_x,) in loader:
                probs = bundle.lstm(batch_x).squeeze().numpy()
                probs = np.atleast_1d(probs)
                preds.extend(probs.tolist())
        pad = len(X) - len(preds)
        if pad > 0:
            preds = [0.5] * pad + preds
        return np.array(preds[: len(X)])

    def _predict_latest(
        self,
        symbol: str,
        latest_row: pd.Series,
        bundle: _ModelBundle,
    ) -> float:
        feature_cols = [col for col in latest_row.index if col != "future_return"]
        vector = latest_row[feature_cols].values.reshape(1, -1)
        X = bundle.scaler.transform(vector)
        proba_lr = bundle.logistic.predict_proba(X)[0, 1]
        proba_rf = bundle.forest.predict_proba(X)[0, 1]
        proba_lstm = self._predict_lstm_live(symbol, bundle, X)
        weights = self._weights.get(symbol, {})
        if not weights:
            weights = {"logistic": 1 / 3, "forest": 1 / 3, "lstm": 1 / 3}
        blended = (
            weights.get("logistic", 0.0) * proba_lr
            + weights.get("forest", 0.0) * proba_rf
            + weights.get("lstm", 0.0) * proba_lstm
        )
        return float(blended)

    def _predict_lstm_live(self, symbol: str, bundle: _ModelBundle, X: np.ndarray) -> float:
        if len(X) == 0:
            return 0.5
        sequence_length = bundle.sequence_length
        symbol_cache = self._lstm_feature_cache.setdefault(symbol, deque(maxlen=sequence_length))
        symbol_cache.append(X.squeeze())
        if len(symbol_cache) < sequence_length:
            return 0.5
        sequence = np.stack(symbol_cache)
        tensor = torch.from_numpy(sequence).float().unsqueeze(0)
        bundle.lstm.eval()
        with torch.no_grad():
            prob = bundle.lstm(tensor).item()
        return float(prob)

    def _decide_signal(self, probability: float) -> str:
        if probability >= 0.55:
            return "buy"
        if probability <= 0.45:
            return "sell"
        return "hold"

    def _sharpe_ratio(self, predictions: np.ndarray, returns: np.ndarray) -> float:
        if len(predictions) == 0 or len(predictions) != len(returns):
            return 0.0
        positions = np.where(predictions >= 0.5, 1.0, -1.0)
        perf = positions * returns
        if np.allclose(perf.std(ddof=1) if len(perf) > 1 else perf.std(), 0):
            return 0.0
        mean = perf.mean()
        std = perf.std(ddof=1) if len(perf) > 1 else perf.std()
        if std == 0:
            return 0.0
        return float(mean / std * math.sqrt(len(perf)))


__all__ = ["EnsembleMLWorker"]
