import numpy as np
from core.worker import Worker
from services.logger import EventLogger


class PortfolioManager:
    """
    Portfolio Manager (PM) routes capital to workers.
    Implements a bandit-style allocator with risk constraints.
    """

    def __init__(self, risk_budget: float = 0.02, logger: EventLogger = None):
        """
        :param risk_budget: Fraction of portfolio equity to risk per trade.
        """
        self.risk_budget = risk_budget
        self.logger = logger

    def allocate(self, workers: list[Worker], equity: float) -> dict:
        """
        Capital allocation across workers.
        :return: dict {worker_name: allocation_weight}
        """
        scores = []
        for w in workers:
            score = w.win_rate() * 0.5 + (w.avg_pnl() > 0) * 0.5  # hybrid score
            scores.append((w.name, score))

        # Normalize
        total = sum(s for _, s in scores) or 1.0
        weights = {name: s / total for name, s in scores}

        # Scale by risk budget
        allocs = {name: weights[name] * self.risk_budget for name, _ in scores}

        if self.logger:
            self.logger.write({
                "event": "allocation",
                "allocs": allocs
            })

        return allocs

