import sqlite3

from desk.services.dashboard_recorder import DashboardRecorder
from desk.services.execution import OpenTrade


def _make_trade() -> OpenTrade:
    return OpenTrade(
        worker="alpha",
        symbol="BTC/USD",
        side="BUY",
        qty=0.5,
        entry_price=30000.0,
        stop_loss=29500.0,
        take_profit=31000.0,
        max_hold_seconds=3600,
    )


def test_dashboard_recorder_writes_sqlite(tmp_path):
    recorder = DashboardRecorder("paper", db_dir=tmp_path)
    trade = _make_trade()

    recorder.record_trade_open(trade, fee=1.23, metadata={"foo": "bar"})
    recorder.record_equity(1250.5)
    recorder.record_ml_score(
        trade.worker,
        trade.symbol,
        probability=0.62,
        features={"edge": 0.5},
        trade_id=trade.trade_id,
    )
    recorder.record_trade_close(trade, exit_price=30750.0, exit_reason="take_profit", pnl=375.0)
    recorder.update_ml_label(trade.trade_id, 1)
    recorder.close()

    conn = sqlite3.connect(recorder.db_path)
    try:
        trades = conn.execute("SELECT status, exit, pnl FROM trades").fetchall()
        assert trades and "CLOSED" in trades[0][0]
        assert trades[0][1] == 30750.0
        assert trades[0][2] == 375.0

        equity = conn.execute("SELECT balance FROM equity").fetchall()
        assert equity and equity[0][0] == 1250.5

        ml = conn.execute("SELECT proba_win, label, features_json FROM ml_scores").fetchall()
        assert ml and ml[-1][1] == 1
        assert "trade_id" in ml[0][2]
    finally:
        conn.close()
