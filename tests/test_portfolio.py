from datetime import datetime, timedelta, timezone

from models import Position
from portfolio import PortfolioManager


def test_daily_window_rolls_over_on_simulated_clock_not_wall_clock():
    """Regression test: the backtester feeds PortfolioManager historical
    timestamps via advance_clock(). Before this was wired up, daily_pnl used
    real wall-clock time, so a backtest replaying months of history in a few
    seconds would never see a new "day" and would permanently trip the
    daily-loss circuit breaker after a single bad day near the start."""
    pm = PortfolioManager(starting_equity=10000.0)

    day1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    pm.advance_clock(day1)
    assert pm.daily_pnl == 0.0

    pm.equity -= 400.0  # simulate a losing day
    assert pm.daily_pnl == -400.0
    assert pm.daily_start_equity == 10000.0

    # still day 1 a few hours later -> window must not reset yet
    pm.advance_clock(day1 + timedelta(hours=5))
    assert pm.daily_pnl == -400.0

    # next simulated day -> window resets relative to the new starting equity
    day2 = day1 + timedelta(days=1)
    pm.advance_clock(day2)
    assert pm.daily_pnl == 0.0
    assert pm.daily_start_equity == 9600.0


def test_open_and_close_position_updates_equity_and_peak():
    pm = PortfolioManager(starting_equity=10000.0)
    pos = Position(
        symbol="BTC/USDT:USDT", side="LONG", entry_price=100.0, qty=1.0,
        stop_price=98.0, take_profit_price=105.0, initial_risk_per_unit=2.0,
        opened_at=datetime.now(timezone.utc),
    )
    pm.open_position(pos)
    assert "BTC/USDT:USDT" in pm.open_positions

    trade = pm.close_position("BTC/USDT:USDT", exit_price=105.0, fees=1.0, reason="take_profit")
    assert trade.pnl == 5.0 - 1.0  # (105-100)*1 - fee
    assert pm.equity == 10000.0 + trade.pnl
    assert pm.peak_equity == pm.equity
    assert "BTC/USDT:USDT" not in pm.open_positions
