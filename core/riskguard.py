class RiskGuard:
    def __init__(self, daily_dd=None, weekly_dd=None, trade_stop_loss=1.0, max_concurrent=5, halt_on_dd=False):
        """
        Risk management layer for the trading bot.

        Args:
            daily_dd (float or None): Max daily drawdown fraction (e.g. 0.2 = 20%). None/0 disables check.
            weekly_dd (float or None): Max weekly drawdown fraction. None/0 disables check.
            trade_stop_loss (float): Fractional stop loss per trade (1.0 = 100%).
            max_concurrent (int): Max concurrent trades allowed.
            halt_on_dd (bool): Halt trading if drawdown exceeded.
        """
        self.daily_dd = daily_dd
        self.weekly_dd = weekly_dd
        self.trade_stop_loss = trade_stop_loss
        self.max_concurrent = max_concurrent
        self.halt_on_dd = halt_on_dd

        self.start_equity = None
        self.halted = False

    def set_start_equity(self, equity):
        """Set the baseline equity for drawdown calculations."""
        self.start_equity = equity

    def check_trade(self, price, qty):
        """
        Compute maximum allowed loss for a trade.

        Args:
            price (float): Trade price.
            qty (float): Quantity of asset.
        Returns:
            float: Max allowed loss in USD.
        """
        return price * qty * self.trade_stop_loss

    def check(self, equity):
        """
        Check global risk limits against current equity.
        """
        if self.start_equity is None:
            self.start_equity = equity

        # Daily DD check
        if self.daily_dd is not None and self.daily_dd > 0:
            if equity < self.start_equity * (1 - self.daily_dd):
                print("[RISK] Daily drawdown exceeded!")
                if self.halt_on_dd:
                    self.halted = True

        # Weekly DD check
        if self.weekly_dd is not None and self.weekly_dd > 0:
            if equity < self.start_equity * (1 - self.weekly_dd):
                print("[RISK] Weekly drawdown exceeded!")
                if self.halt_on_dd:
                    self.halted = True

    def update_limits(self, risk_cfg):
        """Update risk limits dynamically from config dict."""
        self.daily_dd = risk_cfg.get("daily_dd", self.daily_dd)
        self.weekly_dd = risk_cfg.get("weekly_dd", self.weekly_dd)
        self.trade_stop_loss = risk_cfg.get("trade_stop_loss", self.trade_stop_loss)
        self.max_concurrent = risk_cfg.get("max_concurrent", self.max_concurrent)
        self.halt_on_dd = risk_cfg.get("halt_on_dd", self.halt_on_dd)
