import numpy as np

class PortfolioAllocator:
    """
    Allocates portfolio weights across workers based on performance.
    """

    def __init__(self, min_weight=0.0, max_weight=0.2):
        self.min_weight = min_weight
        self.max_weight = max_weight

    def allocate(self, workers):
        """
        Return a dict mapping worker.name -> weight.
        Allocation is proportional to a score combining win rate and avg PnL.
        """
        scores = {}
        for w in workers:
            win = w.win_rate()
            pnl = w.avg_pnl()
            # Score balances win rate and pnl (normalize pnl to avoid huge swings)
            score = 0.7 * win + 0.3 * np.tanh(pnl / 100.0)
            scores[w.name] = max(score, 0.0)

        total = sum(scores.values())
        if total == 0:
            # Equal allocation fallback
            return {w.name: 1.0 / len(workers) for w in workers}

        # Normalize to weights
        raw_weights = {k: v / total for k, v in scores.items()}
        # Clip within min/max constraints
        clipped = {k: np.clip(v, self.min_weight, self.max_weight) for k, v in raw_weights.items()}
        # Re-normalize
        s = sum(clipped.values())
        return {k: v / s for k, v in clipped.items()}
