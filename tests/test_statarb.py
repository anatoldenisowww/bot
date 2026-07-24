import numpy as np
import pandas as pd

from statarb import (PairConfig, all_pairs, analyze_pair, half_life, hedge_ratio)
from statarb_report import render_statarb_html


def _cointegrated_pair(n=600, seed=0, beta=0.5):
    """B is a random walk; A = beta*B + a fast-mean-reverting spread. This is
    a textbook tradeable pair - the screener should find a SHORT half-life."""
    rng = np.random.default_rng(seed)
    log_b = np.cumsum(rng.normal(0, 0.01, n)) + np.log(2000)
    # Ornstein-Uhlenbeck spread with strong mean reversion (short half-life)
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.7 * spread[t - 1] + rng.normal(0, 0.02)
    log_a = beta * log_b + spread + np.log(3)
    ts = pd.date_range("2023-01-01", periods=n, freq="4h", tz="utc")
    df_a = pd.DataFrame({"timestamp": ts, "close": np.exp(log_a),
                         "open": np.exp(log_a), "high": np.exp(log_a) * 1.001,
                         "low": np.exp(log_a) * 0.999, "volume": 1.0})
    df_b = pd.DataFrame({"timestamp": ts, "close": np.exp(log_b),
                         "open": np.exp(log_b), "high": np.exp(log_b) * 1.001,
                         "low": np.exp(log_b) * 0.999, "volume": 1.0})
    return df_a, df_b


def test_hedge_ratio_recovers_beta():
    df_a, df_b = _cointegrated_pair(beta=0.5)
    beta = hedge_ratio(np.log(df_a["close"].to_numpy()), np.log(df_b["close"].to_numpy()))
    assert 0.3 < beta < 0.7  # recovers the true 0.5 within noise


def test_half_life_short_for_mean_reverting_spread():
    df_a, df_b = _cointegrated_pair()
    report = analyze_pair(df_a, df_b, "A/USDT:USDT", "B/USDT:USDT", PairConfig())
    # a strongly mean-reverting synthetic spread should have a short half-life
    assert report.stats.half_life_bars < 30


def test_half_life_infinite_for_random_walk():
    rng = np.random.default_rng(1)
    rw = np.cumsum(rng.normal(0, 1, 500))  # pure random walk does not mean-revert
    assert half_life(rw) == float("inf") or half_life(rw) > 100


def test_analyze_pair_produces_signal_and_backtest():
    df_a, df_b = _cointegrated_pair()
    report = analyze_pair(df_a, df_b, "A/USDT:USDT", "B/USDT:USDT", PairConfig())
    assert report.stats.current_signal in {"LONG_SPREAD", "SHORT_SPREAD", "FLAT"}
    assert report.backtest is not None
    assert 0.0 <= report.backtest.win_rate <= 100.0
    assert -1.0 <= report.stats.correlation <= 1.0


def test_all_pairs_enumerates_combinations():
    pairs = all_pairs(["BTC", "ETH", "SOL"])
    assert pairs == [("BTC", "ETH"), ("BTC", "SOL"), ("ETH", "SOL")]


def test_statarb_html_is_self_contained():
    df_a, df_b = _cointegrated_pair()
    report = analyze_pair(df_a, df_b, "A/USDT:USDT", "B/USDT:USDT", PairConfig())
    doc = render_statarb_html([report], PairConfig(), "4h")
    assert "<svg" in doc and "pairs-trading" in doc.lower()
    assert "http://" not in doc and "https://" not in doc and "<script" not in doc
    assert 'data-theme="dark"' in doc and doc.strip().startswith("<!doctype html>")
