from desk.services.risk import PositionSizingResult, RiskEngine


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
    engine.check_account(10_000)
    qty = engine.position_size(100, 50, stop_loss=98, side="BUY")
    assert qty == 25
    fallback = engine.position_size(100, 50, stop_loss=None, side="BUY")
    assert fallback > 0


def test_position_size_respects_max_notional():
    engine = RiskEngine(0.05, 0.1, 0.02, 5, False, 0.05, max_position_value=500)
    engine.check_account(50_000)
    qty = engine.position_size(25_000, 200, stop_loss=24_800, side="BUY")
    assert qty <= 500 / 25_000


def test_size_position_enforces_min_notional():
    engine = RiskEngine(0.05, 0.1, 0.02, 5, False, 0.05, min_notional=10)
    engine.check_account(5_000)
    sizing: PositionSizingResult = engine.size_position(
        1.0,
        stop_loss=0.99,
        side="BUY",
        risk_budget=0.01,
    )
    assert sizing.notional >= 10
    assert sizing.min_notional_applied is True


def test_risk_budget_scales_with_equity():
    engine = RiskEngine(0.05, 0.1, 0.02, 5, False, 0.05, risk_per_trade_pct=0.03)
    engine.check_account(1_000)
    low_budget = engine.risk_budget()
    engine.check_account(5_000)
    high_budget = engine.risk_budget()
    assert high_budget > low_budget


def test_enforce_position_limits_allows_multiple():
    engine = RiskEngine(0.05, 0.1, 0.02, 4, False, 0.05)
    assert engine.enforce_position_limits([1, 2, 3]) is True
    assert engine.enforce_position_limits([1, 2, 3, 4]) is False


def test_equity_floor_halts_only_when_breached():
    engine = RiskEngine(
        daily_dd=None,
        weekly_dd=None,
        default_stop_pct=0.02,
        max_concurrent=5,
        halt_on_dd=False,
        trapdoor_pct=0.0,
        equity_floor=1_000,
    )
    engine.check_account(5_000)
    assert engine.halted is False
    engine.check_account(1_500)
    assert engine.halted is False
    engine.check_account(900)
    assert engine.halted is True
