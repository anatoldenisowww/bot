from datetime import datetime, timezone

from config import RiskConfig
from models import Position
from risk_manager import RiskManager


def make_risk(**overrides) -> RiskManager:
    defaults = dict(
        starting_equity=10000.0,
        risk_per_trade_pct=1.0,
        max_leverage=5,
        atr_stop_multiplier=2.0,
        take_profit_r_multiple=2.5,
        trailing_activation_r=1.0,
        trailing_atr_multiplier=1.5,
        max_concurrent_positions=3,
        max_portfolio_risk_pct=3.0,
        max_daily_loss_pct=3.0,
        max_drawdown_pct=15.0,
    )
    defaults.update(overrides)
    return RiskManager(RiskConfig(**defaults))


def test_position_size_risks_exactly_risk_per_trade_pct():
    risk = make_risk()
    equity = 10000.0
    entry = 100.0
    stop = 98.0  # $2 risk per unit
    qty = risk.position_size(equity, entry, stop)
    risked_dollars = qty * abs(entry - stop)
    assert abs(risked_dollars - equity * 0.01) < 1e-6


def test_position_size_capped_by_leverage():
    risk = make_risk(max_leverage=2, risk_per_trade_pct=50.0)  # absurd risk pct to force the cap
    equity = 1000.0
    entry = 100.0
    stop = 99.0
    qty = risk.position_size(equity, entry, stop)
    notional = qty * entry
    assert notional <= equity * 2 + 1e-6


def test_stop_and_take_long():
    risk = make_risk()
    stop, take = risk.compute_stop_and_take("LONG", entry_price=100.0, atr=2.0)
    assert stop == 96.0  # 100 - 2*2
    assert take == 110.0  # 100 + 2*2*2.5


def test_stop_and_take_short():
    risk = make_risk()
    stop, take = risk.compute_stop_and_take("SHORT", entry_price=100.0, atr=2.0)
    assert stop == 104.0
    assert take == 90.0


def test_max_concurrent_positions_blocks_new_trade():
    risk = make_risk(max_concurrent_positions=1)
    existing = Position(
        symbol="ETH/USDT:USDT", side="LONG", entry_price=2000, qty=1, stop_price=1900,
        take_profit_price=2200, initial_risk_per_unit=100, opened_at=datetime.now(timezone.utc),
    )
    decision = risk.can_open_new_trade(
        "BTC/USDT:USDT", equity=10000, peak_equity=10000, daily_pnl=0,
        starting_daily_equity=10000, open_positions={"ETH/USDT:USDT": existing}, proposed_risk_dollars=100,
    )
    assert not decision.allowed


def test_drawdown_halt_blocks_new_trade():
    risk = make_risk(max_drawdown_pct=10.0)
    decision = risk.can_open_new_trade(
        "BTC/USDT:USDT", equity=8900, peak_equity=10000, daily_pnl=0,
        starting_daily_equity=8900, open_positions={}, proposed_risk_dollars=50,
    )
    assert not decision.allowed
    assert "drawdown" in decision.reason


def test_daily_loss_halt_blocks_new_trade():
    risk = make_risk(max_daily_loss_pct=3.0)
    decision = risk.can_open_new_trade(
        "BTC/USDT:USDT", equity=9600, peak_equity=10000, daily_pnl=-400,
        starting_daily_equity=10000, open_positions={}, proposed_risk_dollars=50,
    )
    assert not decision.allowed
    assert "daily_loss" in decision.reason


def test_portfolio_risk_cap_blocks_new_trade():
    risk = make_risk(max_portfolio_risk_pct=1.0)
    existing = Position(
        symbol="ETH/USDT:USDT", side="LONG", entry_price=2000, qty=1, stop_price=1900,
        take_profit_price=2200, initial_risk_per_unit=100, opened_at=datetime.now(timezone.utc),
    )
    # existing position already risks 100 / 10000 = 1% of equity, so any more risk breaches the 1% cap
    decision = risk.can_open_new_trade(
        "BTC/USDT:USDT", equity=10000, peak_equity=10000, daily_pnl=0,
        starting_daily_equity=10000, open_positions={"ETH/USDT:USDT": existing}, proposed_risk_dollars=50,
    )
    assert not decision.allowed


def test_trailing_stop_only_moves_in_favorable_direction():
    risk = make_risk(trailing_activation_r=1.0, trailing_atr_multiplier=1.0)
    position = Position(
        symbol="BTC/USDT:USDT", side="LONG", entry_price=100.0, qty=1.0, stop_price=96.0,
        take_profit_price=110.0, initial_risk_per_unit=4.0, opened_at=datetime.now(timezone.utc),
    )
    # price at 105 -> R = (105-100)/4 = 1.25 >= activation, trail = 105 - 1*atr(=2) = 103 > 96 -> should move
    new_stop = risk.update_trailing_stop(position, current_price=105.0, atr=2.0)
    assert new_stop == 103.0

    position.stop_price = 103.0
    # price pulls back to 104 -> trail target 102, which is worse than current stop -> should NOT move
    new_stop_2 = risk.update_trailing_stop(position, current_price=104.0, atr=2.0)
    assert new_stop_2 is None
