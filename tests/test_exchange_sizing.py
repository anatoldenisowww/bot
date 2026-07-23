from exchange import MarketLimits, clamp_qty_to_market, round_down_to_step


def test_round_down_to_step():
    assert round_down_to_step(1.2345, 0.01) == 1.23
    assert round_down_to_step(1.2399, 0.01) == 1.23
    assert round_down_to_step(5.0, 1.0) == 5.0
    assert round_down_to_step(5.9, 1.0) == 5.0
    assert round_down_to_step(0.5, 0.0) == 0.5  # step 0 -> no rounding


def test_clamp_qty_rounds_to_step():
    limits = MarketLimits(amount_step=0.01, amount_min=0.01, cost_min=5.0)
    # 0.12345 units at price 100 -> notional way above min, just round to step
    assert clamp_qty_to_market(0.12345, price=100.0, limits=limits) == 0.12


def test_clamp_qty_rejects_below_amount_min():
    limits = MarketLimits(amount_step=1.0, amount_min=1.0, cost_min=5.0)
    # 0.6 units rounds down to 0, which is below amount_min -> not tradable
    assert clamp_qty_to_market(0.6, price=100.0, limits=limits) == 0.0


def test_clamp_qty_rejects_below_min_notional():
    limits = MarketLimits(amount_step=0.001, amount_min=0.001, cost_min=5.0)
    # 0.002 units at price 100 = $0.20 notional, below the $5 minimum -> rejected.
    # This is the exact small-account failure mode the clamp exists to catch.
    assert clamp_qty_to_market(0.002, price=100.0, limits=limits) == 0.0


def test_clamp_qty_accepts_when_all_minimums_met():
    limits = MarketLimits(amount_step=0.001, amount_min=0.001, cost_min=5.0)
    # 0.1 units at price 100 = $10 notional, above the $5 min, on-step -> accepted
    assert clamp_qty_to_market(0.1, price=100.0, limits=limits) == 0.1
