class ExecutionService:
    """
    Wraps QuantConnect execution methods for a unified interface.
    """

    def __init__(self, algo):
        self.algo = algo

    def market_order(self, symbol, qty):
        """
        Place a market order.
        """
        return self.algo.MarketOrder(symbol, qty)

    def limit_order(self, symbol, qty, limit_price):
        """
        Place a limit order.
        """
        return self.algo.LimitOrder(symbol, qty, limit_price)

    def stop_market_order(self, symbol, qty, stop_price):
        """
        Place a stop-market order.
        """
        return self.algo.StopMarketOrder(symbol, qty, stop_price)

    def liquidate(self, symbol=None):
        """
        Liquidate all or a specific position.
        """
        return self.algo.Liquidate(symbol)
