import time

class Executor:
    def __init__(self, broker, logger, config):
        self.broker = broker
        self.logger = logger
        self.config = config
        self.open_trades = {}  # symbol -> list of trades

    def open_trade(self, worker, side, qty, price, risk_amount):
        side = side.upper()
        if side not in ("BUY", "SELL"):
            print(f"[EXECUTOR] Invalid side {side} for {worker.name}")
            return None

        # Risk settings
        sl_pct = self.config["risk"]["stop_loss_pct"]
        rr = self.config["risk"]["rr_ratio"]
        hold = self.config["risk"]["max_hold_minutes"] * 60

        if side == "BUY":
            stop_loss = price * (1 - sl_pct)
            take_profit = price * (1 + sl_pct * rr)
        else:  # SELL (short)
            stop_loss = price * (1 + sl_pct)
            take_profit = price * (1 - sl_pct * rr)

        trade = {
            "worker": worker.name,
            "symbol": worker.symbol,
            "side": side,
            "qty": float(qty),
            "entry_price": float(price),
            "timestamp": time.time(),
            "risk": risk_amount,
            "stop_loss": stop_loss,       # renamed key
            "take_profit": take_profit,   # renamed key
            "max_hold": hold,
        }

        # Track trade in-memory
        self.open_trades.setdefault(worker.symbol, []).append(trade)

        # Log entry
        self.logger.log_trade(worker, worker.symbol, side, qty, price, 0.0)
        return trade

    def get_open_trades(self, symbol):
        return self.open_trades.get(symbol, [])

    def close_trade(self, trade, exit_price, exit_reason, pnl):
        symbol = trade["symbol"]
        if symbol in self.open_trades and trade in self.open_trades[symbol]:
            self.open_trades[symbol].remove(trade)

        self.logger.log_trade_end(
            trade["worker"], trade["symbol"], exit_price, exit_reason, pnl
        )

    def check_exit(self, worker, trade, current_price):
        now = time.time()
        elapsed = now - trade["timestamp"]
        exit_reason, exit_price, pnl = None, None, 0.0

        # Stop loss / Take profit
        if trade["side"] == "BUY":
            if current_price <= trade["stop_loss"]:
                exit_reason = "stop_loss"
            elif current_price >= trade["take_profit"]:
                exit_reason = "take_profit"
        elif trade["side"] == "SELL":
            if current_price >= trade["stop_loss"]:
                exit_reason = "stop_loss"
            elif current_price <= trade["take_profit"]:
                exit_reason = "take_profit"

        # Time stop
        if elapsed > trade["max_hold"]:
            exit_reason = "time_stop"

        if exit_reason:
            exit_price = current_price
            if trade["side"] == "BUY":
                pnl = (exit_price - trade["entry_price"]) * trade["qty"]
            else:  # SELL
                pnl = (trade["entry_price"] - exit_price) * trade["qty"]

            self.close_trade(trade, exit_price, exit_reason, pnl)
            return True, pnl

        return False, 0.0






