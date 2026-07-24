import numpy as np
import pandas as pd

from regime_hmm import fit_regimes
from regime_report import render_regime_body, render_regime_html


def _two_regime_series(seed=0):
    """A synthetic series with two obvious regimes: a calm uptrend followed by
    a volatile downtrend. A working HMM should separate them and order them
    bearish-first."""
    rng = np.random.default_rng(seed)
    # regime A: drift up, low vol
    a = 100 * np.cumprod(1 + rng.normal(0.002, 0.005, 300))
    # regime B: drift down, high vol
    b = a[-1] * np.cumprod(1 + rng.normal(-0.003, 0.02, 300))
    close = np.concatenate([a, b])
    n = len(close)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="4h", tz="utc"),
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": np.full(n, 1000.0),
    })
    return df


def test_fit_regimes_separates_and_orders_states():
    df = _two_regime_series()
    report = fit_regimes(df, n_states=2, timeframe="4h", symbol="TEST/USDT:USDT")

    assert report.n_states == 2
    # states are ordered most-bearish first: state 0 mean return < state 1
    assert report.states[0].mean_return_pct <= report.states[1].mean_return_pct
    # transition matrix rows are probability distributions
    for i in range(report.n_states):
        assert abs(report.transition_matrix[i].sum() - 1.0) < 1e-6
    # current-bar posterior is a probability distribution
    assert abs(report.current_state_probs.sum() - 1.0) < 1e-6
    # the state sequence covers every bar (with -1 for the warmup region only)
    assert len(report.state_sequence) == len(df)
    assert (report.state_sequence[-1] in range(report.n_states))


def test_fit_regimes_frequencies_sum_to_100():
    df = _two_regime_series(seed=3)
    report = fit_regimes(df, n_states=3, timeframe="4h", symbol="TEST/USDT:USDT")
    total_freq = sum(s.frequency_pct for s in report.states)
    assert abs(total_freq - 100.0) < 1.0  # rounding tolerance


def test_render_regime_html_is_self_contained():
    df = _two_regime_series()
    report = fit_regimes(df, n_states=3, timeframe="4h", symbol="TEST/USDT:USDT")
    doc = render_regime_html(report)
    body = render_regime_body(report)
    assert "<svg" in body and "Transition probabilities" in body
    # self-contained: no external resource references, no scripts, both themes styled
    assert "http://" not in doc and "https://" not in doc
    assert "<script" not in doc
    assert 'data-theme="dark"' in body and "prefers-color-scheme: dark" in body
    # full document is well-formed enough to open standalone
    assert doc.strip().startswith("<!doctype html>") and doc.strip().endswith("</html>")
