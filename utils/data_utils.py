"""Utility helpers for working with OHLCV data."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd

OHLCV_KEYS = ("timestamp", "open", "high", "low", "close", "volume")


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def normalize_ohlcv(candles: Iterable[Sequence[Any] | Mapping[str, Any]]) -> List[Dict[str, float]]:
    """Normalize raw CCXT-style candles into a list of dictionaries."""
    normalized: List[Dict[str, float]] = []
    for candle in candles:
        if isinstance(candle, Mapping):
            data = {key: candle.get(key) for key in OHLCV_KEYS}
        else:
            seq = list(candle)
            if len(seq) < 6:
                continue
            data = dict(zip(OHLCV_KEYS, seq[:6]))
        normalized.append({key: _coerce_float(data.get(key)) for key in OHLCV_KEYS})
    return normalized


def candles_to_dataframe(candles: Iterable[Sequence[Any] | Mapping[str, Any]]) -> pd.DataFrame:
    """Convert raw candles to a Pandas DataFrame sorted by timestamp."""
    normalized = normalize_ohlcv(candles)
    if not normalized:
        return pd.DataFrame(columns=OHLCV_KEYS)
    df = pd.DataFrame(normalized)
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    df.reset_index(drop=True, inplace=True)
    return df
