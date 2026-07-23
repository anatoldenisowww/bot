import numpy as np
import pandas as pd

import indicators


def _make_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 1, n).cumsum()
    close = 100 + steps
    high = close + rng.uniform(0, 1, n)
    low = close - rng.uniform(0, 1, n)
    open_ = close + rng.normal(0, 0.2, n)
    volume = rng.uniform(100, 1000, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def test_sma_matches_manual_mean():
    df = _make_df()
    out = indicators.sma(df["close"], 10)
    assert np.isclose(out.iloc[50], df["close"].iloc[41:51].mean())


def test_rsi_bounded_0_100():
    df = _make_df()
    out = indicators.rsi(df["close"], 14)
    valid = out.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_rsi_all_gains_is_100():
    close = pd.Series(np.arange(1, 50, dtype=float))
    out = indicators.rsi(close, 14)
    assert np.isclose(out.iloc[-1], 100.0)


def test_atr_non_negative():
    df = _make_df()
    out = indicators.atr(df, 14)
    assert (out.dropna() >= 0).all()


def test_bollinger_upper_gte_lower():
    df = _make_df()
    bb = indicators.bollinger_bands(df["close"], 20, 2.0)
    valid = bb.dropna()
    assert (valid["bb_upper"] >= valid["bb_lower"]).all()


def test_adx_bounded_0_100():
    df = _make_df()
    out = indicators.adx(df, 14)
    valid = out["adx"].dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_add_all_indicators_has_expected_columns():
    from config import load_config
    cfg = load_config()
    df = _make_df()
    out = indicators.add_all_indicators(df, cfg.indicators)
    for col in ["ema_fast", "ema_slow", "ema_trend", "rsi", "rsi_fast", "atr", "bb_upper", "bb_lower", "adx", "donchian_upper", "macd_hist"]:
        assert col in out.columns
