import numpy as np
import pandas as pd

import indicators
from config import load_config
from strategies import CrossSectionalMomentumStrategy


def _trend_df(n, slope, start=100.0, atr_noise=0.3, seed=0):
    """A symbol with a controllable linear drift plus small noise, so its
    momentum score is predictable relative to other symbols."""
    rng = np.random.default_rng(seed)
    close = start + np.arange(n) * slope + rng.normal(0, atr_noise, n)
    close = np.maximum(close, 1.0)
    high = close + rng.uniform(0.05, 0.5, n)
    low = close - rng.uniform(0.05, 0.5, n)
    low = np.minimum(low, close - 0.01)
    open_ = close + rng.normal(0, 0.1, n)
    volume = rng.uniform(1000, 5000, n)
    timestamps = pd.date_range("2023-01-01", periods=n, freq="4h", tz="utc")
    return pd.DataFrame({
        "timestamp": timestamps, "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    })


def _prepare(cfg, raw_dfs):
    return {sym: indicators.add_all_indicators(df, cfg.indicators) for sym, df in raw_dfs.items()}


def test_longs_strongest_shorts_weakest():
    cfg = load_config()
    cfg.strategy.mode = "cross_sectional_momentum"
    cfg.strategy.top_k = 1
    cfg.strategy.bottom_k = 1
    cfg.strategy.min_abs_momentum_score = 0.0
    cfg.strategy.momentum_lookback_bars = 20

    strat = CrossSectionalMomentumStrategy(cfg.strategy)

    dfs = _prepare(cfg, {
        "UP/USDT:USDT": _trend_df(300, slope=0.5, seed=1),     # strong uptrend
        "FLAT/USDT:USDT": _trend_df(300, slope=0.0, seed=2),   # no drift
        "DOWN/USDT:USDT": _trend_df(300, slope=-0.5, seed=3),  # strong downtrend
    })

    signals = strat.generate_signals(dfs, i=299)
    assert signals["UP/USDT:USDT"].side == "LONG"
    assert signals["DOWN/USDT:USDT"].side == "SHORT"
    assert signals["FLAT/USDT:USDT"].side == "FLAT"  # middle of the pack, not in top/bottom-1


def test_min_abs_score_filters_weak_signals():
    cfg = load_config()
    cfg.strategy.mode = "cross_sectional_momentum"
    cfg.strategy.top_k = 1
    cfg.strategy.bottom_k = 1
    cfg.strategy.min_abs_momentum_score = 1000.0  # absurdly high -> nothing qualifies
    cfg.strategy.momentum_lookback_bars = 20

    strat = CrossSectionalMomentumStrategy(cfg.strategy)
    dfs = _prepare(cfg, {
        "UP/USDT:USDT": _trend_df(300, slope=0.5, seed=1),
        "DOWN/USDT:USDT": _trend_df(300, slope=-0.5, seed=3),
    })
    signals = strat.generate_signals(dfs, i=299)
    assert all(sig.side == "FLAT" for sig in signals.values())


def test_exit_policy_closes_when_leaving_bucket():
    cfg = load_config()
    cfg.strategy.mode = "cross_sectional_momentum"
    strat = CrossSectionalMomentumStrategy(cfg.strategy)

    from strategies import Signal
    # A long position should be held while still ranked LONG...
    assert not strat.should_exit_on_signal("LONG", Signal("LONG", 0.5, "x"))
    # ...and exited the moment it fades to FLAT (dropped out of the bucket),
    # not only when it flips to SHORT - this is the key difference from the
    # per-symbol strategies' exit policy.
    assert strat.should_exit_on_signal("LONG", Signal("FLAT", 0.0, "x"))
    assert strat.should_exit_on_signal("LONG", Signal("SHORT", 0.5, "x"))


def test_generate_signal_single_raises():
    cfg = load_config()
    strat = CrossSectionalMomentumStrategy(cfg.strategy)
    df = _trend_df(50, slope=0.1)
    import pytest
    with pytest.raises(NotImplementedError):
        strat.generate_signal(df)
