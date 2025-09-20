from desk.services.risk import RiskEngine


def test_trapdoor_halts_on_break():
    engine = RiskEngine(
        daily_dd=0.05,
        weekly_dd=0.1,
        default_stop_pct=0.02,
        max_concurrent=2,
        halt_on_dd=True,
        trapdoor_pct=0.05,
    )
    engine.initialise(10_000)
    equities = [10_500, 11_000, 12_000]
    expected_floors = [equity * (1 - engine.trapdoor_pct) for equity in equities]

    for equity, expected_floor in zip(equities, expected_floors):
        engine.check_account(equity)
        assert engine.trapdoor is not None
        assert engine.trapdoor.floor == expected_floor

    floor = engine.trapdoor.floor
    # Drop below trapdoor should halt
    engine.check_account(floor - 1)
    assert engine.halted is True


def test_enforce_position_limits_blocks_when_full():
    engine = RiskEngine(
        daily_dd=None,
        weekly_dd=None,
        default_stop_pct=0.02,
        max_concurrent=1,
        halt_on_dd=False,
        trapdoor_pct=0.05,
    )
    assert engine.enforce_position_limits([1]) is False
    assert engine.enforce_position_limits([]) is True


def test_position_size_uses_stop_distance():
    engine = RiskEngine(0.05, 0.1, 0.02, 5, False, 0.05)
    qty = engine.position_size(100, 50, stop_loss=98, side="BUY")
    assert qty == 25
    fallback = engine.position_size(100, 50, stop_loss=None, side="BUY")
    assert fallback > 0
