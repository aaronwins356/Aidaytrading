class RiskManager:
    """
    Risk guardrails for the desk.
    """

    def __init__(self, dd_limit=0.02, max_positions=5):
        self.halted = False
        self.dd_limit = dd_limit
        self.max_positions = max_positions

    def check_drawdown(self, equity, equity_high):
        """
        Halt trading if drawdown exceeds limit.
        """
        if equity_high <= 0:
            return
        dd = (equity_high - equity) / equity_high
        if dd > self.dd_limit:
            self.halted = True

    def check_positions(self, portfolio):
        """
        Halt trading if too many concurrent positions.
        """
        invested = sum(1 for kvp in portfolio if kvp.Value.Invested)
        if invested > self.max_positions:
            self.halted = True
