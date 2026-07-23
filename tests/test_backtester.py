import numpy as np
import pandas as pd

from backtester import align_on_common_timestamps, run_backtest
from config import load_config


def _synthetic_ohlcv(n=500, seed=1, drift=0.15, start=100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1.0, n)
    trend = np.linspace(0, drift * n, n)
    close = start + trend + noise.cumsum() * 0.3
    close = np.maximum(close, 1.0)

    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    low = np.minimum(low, close - 0.01)
    open_ = close + rng.normal(0, 0.3, n)
    volume = rng.uniform(1000, 5000, n)

    timestamps = pd.date_range("2023-01-01", periods=n, freq="4h", tz="utc")
    return pd.DataFrame({
        "timestamp": timestamps, "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    })


def test_backtest_runs_end_to_end_and_produces_sane_stats():
    cfg = load_config()
    price_data = {
        "BTC/USDT:USDT": _synthetic_ohlcv(seed=1),
        "ETH/USDT:USDT": _synthetic_ohlcv(seed=2, start=50.0),
        "SOL/USDT:USDT": _synthetic_ohlcv(seed=3, start=20.0),
    }

    result = run_backtest(cfg, price_data)

    assert result.stats.final_equity > 0
    assert isinstance(result.stats.total_trades, int)
    assert 0.0 <= result.stats.win_rate <= 100.0
    assert len(result.equity_curve) > 0
    # every closed trade should have paid fees on both legs
    for trade in result.trades:
        assert trade.fees >= 0


def test_backtest_raises_on_too_little_data():
    cfg = load_config()
    price_data = {"BTC/USDT:USDT": _synthetic_ohlcv(n=50)}
    try:
        run_backtest(cfg, price_data)
        assert False, "expected ValueError for insufficient candles"
    except ValueError:
        pass


def test_backtest_runs_cross_sectional_mode_end_to_end():
    cfg = load_config()
    cfg.strategy.mode = "cross_sectional_momentum"
    cfg.strategy.top_k = 2
    cfg.strategy.bottom_k = 2
    cfg.strategy.min_abs_momentum_score = 0.0

    # A basket with a spread of drifts so the ranking has something to sort.
    price_data = {
        f"C{i}/USDT:USDT": _synthetic_ohlcv(seed=i, drift=0.05 * (i - 4), start=20.0 + i)
        for i in range(8)
    }

    result = run_backtest(cfg, price_data)

    assert result.stats.final_equity > 0
    assert isinstance(result.stats.total_trades, int)
    assert len(result.equity_curve) > 0
    # cross-sectional exits show up as signal_exit, stop_loss, take_profit, or end_of_backtest
    for trade in result.trades:
        assert trade.reason in {"signal_exit", "stop_loss", "take_profit", "end_of_backtest"}


def test_align_on_common_timestamps_trims_to_intersection():
    # BTC has the full window; NEW starts 100 bars later (a newer listing).
    btc = _synthetic_ohlcv(n=300, seed=1)
    new = _synthetic_ohlcv(n=300, seed=2).iloc[100:].reset_index(drop=True)

    aligned = align_on_common_timestamps({"BTC/USDT:USDT": btc, "NEW/USDT:USDT": new})

    # both come out the same length...
    assert len(aligned["BTC/USDT:USDT"]) == len(aligned["NEW/USDT:USDT"])
    # ...and crucially, row i is the SAME timestamp for every symbol - the
    # property the cross-sectional strategy relies on for cross-symbol ranking.
    ts_btc = aligned["BTC/USDT:USDT"]["timestamp"].reset_index(drop=True)
    ts_new = aligned["NEW/USDT:USDT"]["timestamp"].reset_index(drop=True)
    assert (ts_btc == ts_new).all()


def test_align_on_common_timestamps_is_noop_for_identical_windows():
    # Equal-length, same-timestamp series (BTC/ETH/SOL case) must be unchanged,
    # so the ensemble strategy's already-validated numbers don't shift.
    a = _synthetic_ohlcv(n=300, seed=1)
    b = _synthetic_ohlcv(n=300, seed=2)
    aligned = align_on_common_timestamps({"A": a, "B": b})
    assert len(aligned["A"]) == 300 and len(aligned["B"]) == 300


def test_backtest_respects_market_limits_skipping_tiny_notional():
    from exchange import MarketLimits
    cfg = load_config()
    cfg.risk.starting_equity = 200.0

    price_data = {
        "BTC/USDT:USDT": _synthetic_ohlcv(seed=1),
        "ETH/USDT:USDT": _synthetic_ohlcv(seed=2, start=50.0),
    }
    # An absurd $1e9 minimum notional makes every risk-sized order too small
    # to place -> zero trades, proving the clamp is actually wired in.
    impossible_limits = {
        sym: MarketLimits(amount_step=0.0001, amount_min=0.0001, cost_min=1e9)
        for sym in price_data
    }
    result = run_backtest(cfg, price_data, market_limits=impossible_limits)
    assert result.stats.total_trades == 0
