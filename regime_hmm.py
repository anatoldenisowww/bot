"""Hidden Markov Model market-regime detector.

What this is: a Gaussian HMM fit to observable market features (log returns
and rolling volatility) that infers a small number of hidden "regimes" the
market moves between - typically something like calm-uptrend, choppy-range,
and volatile-selloff. It then tells you which regime today most likely
belongs to, how the regimes behave, and how they transition.

What this is NOT: a predictor of future prices or a profit guarantee. An HMM
describes the *current* statistical state and the historical dynamics between
states; it cannot tell you tomorrow's return. It's an information tool - use
it to understand context (e.g. "we're in a high-volatility regime, size
down") not as a crystal ball.

Design notes:
- Features are [log_return, rolling_std_of_log_return]. Return captures
  direction, volatility captures how violent the state is - the two axes that
  separate market regimes most cleanly.
- States are relabeled by mean return after fitting (HMM state indices are
  arbitrary), so "state 0" is always the most bearish and the last state the
  most bullish - stable, human-readable ordering across runs.
- We fix the random seed for reproducibility; HMM fitting is a non-convex EM
  optimization, so different seeds can land on slightly different fits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError as exc:  # pragma: no cover - only hit if dependency missing
    raise ImportError("regime_hmm requires hmmlearn - run `pip install -r requirements.txt`") from exc


@dataclass
class RegimeState:
    index: int                 # 0 = most bearish ... n-1 = most bullish (after relabel)
    label: str                 # human-readable, e.g. "volatile selloff"
    mean_return_pct: float     # per-bar mean log return, %
    volatility_pct: float      # per-bar return std dev, %
    frequency_pct: float       # % of history spent in this state
    expected_duration_bars: float  # 1 / (1 - self-transition prob)


@dataclass
class RegimeReport:
    symbol: str
    timeframe: str
    n_states: int
    states: List[RegimeState]
    transition_matrix: np.ndarray          # [i][j] = P(next=j | current=i), relabeled order
    current_state: int
    current_state_probs: np.ndarray        # posterior prob of each state on the last bar
    state_sequence: np.ndarray             # most-likely state per bar (Viterbi), relabeled
    timestamps: pd.Series
    closes: pd.Series
    log_likelihood: float
    feature_window: int = field(default=0)

    @property
    def current(self) -> RegimeState:
        return self.states[self.current_state]

    def expected_next_state(self) -> RegimeState:
        """The single most likely regime on the next bar given the current one -
        NOT a price forecast, just the highest-probability transition."""
        probs = self.transition_matrix[self.current_state]
        return self.states[int(np.argmax(probs))]


def _label_state(mean_return_pct: float, volatility_pct: float, vol_median: float) -> str:
    hi_vol = volatility_pct >= vol_median
    if mean_return_pct > 0.05:
        return "volatile bull" if hi_vol else "calm bull"
    if mean_return_pct < -0.05:
        return "volatile selloff" if hi_vol else "grinding decline"
    return "choppy range (high vol)" if hi_vol else "quiet range"


def fit_regimes(
    df: pd.DataFrame,
    n_states: int = 3,
    vol_window: int = 10,
    timeframe: str = "",
    symbol: str = "",
    seed: int = 42,
) -> RegimeReport:
    """Fit a Gaussian HMM and return a structured report.

    df needs at least ['timestamp', 'close']. Returns regimes ordered from most
    bearish (index 0) to most bullish.
    """
    if len(df) < max(50, vol_window + 20):
        raise ValueError(f"need more history to fit an HMM (have {len(df)} bars)")

    closes = df["close"].reset_index(drop=True)
    log_ret = np.log(closes / closes.shift(1))
    roll_vol = log_ret.rolling(vol_window, min_periods=vol_window).std()

    feat = pd.DataFrame({"r": log_ret, "v": roll_vol}).dropna()
    valid_index = feat.index
    X_raw = feat.to_numpy()

    # Standardize features (z-score). A Gaussian HMM with diagonal covariance
    # is very sensitive to feature scale, and raw log-returns (~1e-3) vs
    # rolling vol (~7e-3) are both tiny and unequal - unscaled, EM collapses
    # to spurious non-persistent states. Standardizing lets it find real,
    # persistent regimes. We keep X_raw around for reporting in real units.
    mu = X_raw.mean(axis=0)
    sigma = X_raw.std(axis=0)
    sigma[sigma == 0] = 1.0
    X = (X_raw - mu) / sigma

    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=500,
                        random_state=seed, tol=1e-4)
    model.fit(X)
    log_likelihood = float(model.score(X))
    raw_states = model.predict(X)
    posteriors = model.predict_proba(X)

    # Relabel states by real-unit mean return so ordering is stable and
    # meaningful (index 0 = most bearish). Reporting stats all come from X_raw
    # so they're in actual return units, not z-scores.
    state_mean_return = np.array([X_raw[raw_states == s, 0].mean() if (raw_states == s).any() else 0.0
                                  for s in range(n_states)])
    order = np.argsort(state_mean_return)  # ascending: most bearish first
    old_to_new = {int(old): new for new, old in enumerate(order)}

    relabeled_seq = np.array([old_to_new[int(s)] for s in raw_states])
    relabeled_post = posteriors[:, order]

    # Transition matrix in the new ordering.
    trans = model.transmat_[np.ix_(order, order)]

    vols_pct = []
    means_pct = []
    for new_s in range(n_states):
        mask = relabeled_seq == new_s
        means_pct.append(float(X_raw[mask, 0].mean() * 100) if mask.any() else 0.0)
        vols_pct.append(float(X_raw[mask, 0].std() * 100) if mask.any() else 0.0)
    vol_median = float(np.median(vols_pct))

    states: List[RegimeState] = []
    for new_s in range(n_states):
        mask = relabeled_seq == new_s
        freq = float(mask.mean() * 100)
        self_p = float(trans[new_s, new_s])
        expected_dur = 1.0 / (1.0 - self_p) if self_p < 1.0 else float("inf")
        states.append(RegimeState(
            index=new_s,
            label=_label_state(means_pct[new_s], vols_pct[new_s], vol_median),
            mean_return_pct=means_pct[new_s],
            volatility_pct=vols_pct[new_s],
            frequency_pct=freq,
            expected_duration_bars=expected_dur,
        ))

    current_state = int(relabeled_seq[-1])

    # Align the state sequence back onto the full-length timestamp/close series
    # (the first vol_window bars have no features and get state -1 = unknown).
    full_seq = np.full(len(closes), -1, dtype=int)
    full_seq[valid_index] = relabeled_seq

    return RegimeReport(
        symbol=symbol,
        timeframe=timeframe,
        n_states=n_states,
        states=states,
        transition_matrix=trans,
        current_state=current_state,
        current_state_probs=relabeled_post[-1],
        state_sequence=full_seq,
        timestamps=df["timestamp"].reset_index(drop=True),
        closes=closes,
        log_likelihood=log_likelihood,
        feature_window=vol_window,
    )
