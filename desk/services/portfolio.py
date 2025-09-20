"""Capital allocator with a contextual multi-armed bandit flavour."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable


@dataclass
class WorkerStats:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0
    last_promotion: float = field(default_factory=time.time)

    @property
    def win_rate(self) -> float:
        total = self.trades
        return (self.wins / total) if total else 0.0

    @property
    def expectancy(self) -> float:
        return self.pnl / self.trades if self.trades else 0.0


class PortfolioManager:
    """Simple epsilon-greedy allocator for promoting/demoting workers."""

    def __init__(self, min_weight: float, max_weight: float, epsilon: float, cooldown_minutes: float):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.epsilon = epsilon
        self.cooldown_seconds = cooldown_minutes * 60
        self.state: Dict[str, WorkerStats] = {}

    def mark_routed(self, worker_name: str) -> None:
        stats = self.state.setdefault(worker_name, WorkerStats())
        stats.last_promotion = time.time()

    def update_stats(self, worker_name: str, pnl: float) -> None:
        stats = self.state.setdefault(worker_name, WorkerStats())
        stats.trades += 1
        stats.pnl += pnl
        if pnl > 0:
            stats.wins += 1
        else:
            stats.losses += 1
        stats.last_promotion = time.time()

    def _score(self, worker) -> float:
        stats = self.state.get(worker.name)
        if not stats:
            return 0.5
        # Combine expectancy and win rate with a bounded activation.
        return 0.5 * stats.win_rate + 0.5 * (math.tanh(stats.expectancy / 50.0) + 1) / 2

    def allocate(self, workers: Iterable) -> Dict[str, float]:
        workers = list(workers)
        if not workers:
            return {}

        if random.random() < self.epsilon:
            weights = {w.name: 1.0 / len(workers) for w in workers}
        else:
            scores = {w.name: max(self._score(w), 0.001) for w in workers}
            total = sum(scores.values())
            weights = {name: score / total for name, score in scores.items()}

        # Enforce hard caps and re-normalize.
        clipped = {
            name: min(self.max_weight, max(self.min_weight, weight))
            for name, weight in weights.items()
        }
        total = sum(clipped.values())
        if total <= 0:
            return {w.name: 1.0 / len(workers) for w in workers}
        normalized = {name: weight / total for name, weight in clipped.items()}
        return normalized

    def eligible(self, worker_name: str) -> bool:
        stats = self.state.get(worker_name)
        if not stats:
            return True
        return (time.time() - stats.last_promotion) >= self.cooldown_seconds

