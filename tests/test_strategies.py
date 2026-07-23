import numpy as np
import pandas as pd
import pytest

import indicators
from config import load_config
from strategies import CrossSectionalMomentumStrategy, EnsembleStrategy, MeanReversionScalpStrategy, build_strategy


def _make_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 1, n).cumsum()
    close = 100 + steps
    high = close + rng.uniform(0, 1, n)
    low = close - rng.uniform(0, 1, n)
    open_ = close + rng.normal(0, 0.2, n)
    volume = rng.uniform(100, 1000, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def test_build_strategy_selects_correct_class():
    cfg = load_config()
    cfg.strategy.mode = "ensemble"
    assert isinstance(build_strategy(cfg), EnsembleStrategy)
    cfg.strategy.mode = "mean_reversion_scalp"
    assert isinstance(build_strategy(cfg), MeanReversionScalpStrategy)
    cfg.strategy.mode = "cross_sectional_momentum"
    assert isinstance(build_strategy(cfg), CrossSectionalMomentumStrategy)
    cfg.strategy.mode = "not_a_real_mode"
    with pytest.raises(ValueError):
        build_strategy(cfg)


def test_mean_reversion_scalp_returns_flat_when_warming_up():
    cfg = load_config()
    strategy = MeanReversionScalpStrategy(cfg.indicators)
    df = indicators.add_all_indicators(_make_df(n=10), cfg.indicators)
    signal = strategy.generate_signal(df)
    assert signal.side == "FLAT"
    assert signal.regime == "insufficient_data"


def test_mean_reversion_scalp_fades_extreme_dip():
    cfg = load_config()
    strategy = MeanReversionScalpStrategy(cfg.indicators)

    # Build a mildly noisy, low-volatility series then crash the last candle
    # far below the recent range - a textbook RSI(2)+Bollinger extreme. A
    # perfectly flat series (zero noise) makes ADX degenerate (0/0 -> NaN
    # forever, since directional movement is identically zero), which is a
    # backtester-fixture artifact, not something real market data does.
    n = 250
    rng = np.random.default_rng(42)
    close = np.concatenate([100.0 + rng.normal(0, 0.05, n - 1), [80.0]])
    df = pd.DataFrame({
        "open": close, "high": close + 0.5, "low": close - 0.5, "close": close,
        "volume": np.full(n, 1000.0),
    })
    df = indicators.add_all_indicators(df, cfg.indicators)

    signal = strategy.generate_signal(df)
    assert signal.side == "LONG"
    assert signal.regime == "mr_scalp"


def test_mean_reversion_scalp_skips_violent_trend_fade():
    cfg = load_config()
    strategy = MeanReversionScalpStrategy(cfg.indicators)

    # A relentless, accelerating downtrend: every bar makes a new low, which
    # drives ADX very high and keeps price below the long EMA (SHORT bias).
    # The scalp strategy's RSI(2)/BB condition for a LONG fade will trigger,
    # but the strong-trend filter should veto it rather than buy a falling knife.
    n = 250
    close = 100 - np.arange(n) * 1.5
    df = pd.DataFrame({
        "open": close, "high": close + 0.2, "low": close - 0.2, "close": close,
        "volume": np.full(n, 1000.0),
    })
    df = indicators.add_all_indicators(df, cfg.indicators)

    row = df.iloc[-1]
    assert row["adx"] >= strategy.STRONG_TREND_ADX  # sanity check the fixture actually hits the filter

    signal = strategy.generate_signal(df)
    assert signal.side == "FLAT"


def test_ensemble_still_returns_flat_signal_type_by_default():
    cfg = load_config()
    strategy = EnsembleStrategy(cfg.indicators)
    df = indicators.add_all_indicators(_make_df(n=5), cfg.indicators)
    signal = strategy.generate_signal(df)
    assert signal.side == "FLAT"
