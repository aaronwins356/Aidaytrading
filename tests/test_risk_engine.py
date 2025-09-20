from desk.services.risk import RiskEngine


def test_trapdoor_halts_on_break():
    engine = RiskEngine(
        daily_dd=0.05,
        weekly_dd=0.1,
        trade_stop_loss=0.02,
        max_concurrent=2,
        halt_on_dd=True,
        trapdoor_pct=0.05,
    )
    engine.initialise(10_000)
    # Equity makes new high -> trapdoor moves up
    engine.check_account(11_000)
    assert engine.trapdoor is not None
    floor = engine.trapdoor.floor
    # Drop below trapdoor should halt
    engine.check_account(floor - 1)
    assert engine.halted is True


def test_enforce_position_limits_blocks_when_full():
    engine = RiskEngine(
        daily_dd=None,
        weekly_dd=None,
        trade_stop_loss=0.02,
        max_concurrent=1,
        halt_on_dd=False,
        trapdoor_pct=0.05,
    )
    assert engine.enforce_position_limits([1]) is False
    assert engine.enforce_position_limits([]) is True


def test_per_trade_notional_handles_invalid_price():
    engine = RiskEngine(0.05, 0.1, 0.02, 5, False, 0.05)
    assert engine.per_trade_notional(0, 1000) == 0.0
    qty = engine.per_trade_notional(100, 1000)
    assert qty > 0
