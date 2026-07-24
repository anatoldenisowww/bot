"""Statistical arbitrage: pairs trading for crypto perpetuals.

The idea (Strategy 1 - the most retail-and-crypto-appropriate stat-arb):
BTC, ETH and SOL move together. Their *spread* - one priced against the
other via a hedge ratio - tends to wobble around a stable level. When the
spread stretches unusually far (a big z-score), it has historically tended
to snap back. So we short the rich leg / long the cheap leg and wait for
convergence, market-neutral to the overall crypto direction.

Honesty, up front (this whole family lives or dies on it):
- This is NOT pure arbitrage. There is no locked-in profit and no guaranteed
  convergence on any single trade. The edge is statistical - it works in
  expectation across many trades, and individual trades can and do lose.
- The relationship can *break* (a coin has its own news, a regime shifts).
  When that happens the spread keeps diverging instead of reverting - which
  is why there's a hard stop on z-score, not just a convergence target.
- Costs matter enormously. A pair trade touches TWO legs on entry and TWO on
  exit, so fees + slippage are charged four times per round trip. Thin edges
  get eaten. The backtest here charges all of it.
- Everything must be validated out-of-sample before it's trusted.

Dependency-light on purpose: hedge ratio via OLS (numpy), mean-reversion
speed via an AR(1) half-life, z-score via a rolling window. No statsmodels
required.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

Signal = str  # "LONG_SPREAD" (buy A / sell B) | "SHORT_SPREAD" (sell A / buy B) | "FLAT"


@dataclass
class PairConfig:
    zscore_window: int = 60      # rolling window for the spread's mean/std
    entry_z: float = 2.0         # open when |z| exceeds this
    exit_z: float = 0.5          # close when |z| falls back inside this
    stop_z: float = 4.0          # bail if |z| blows out past this (relationship likely broke)
    taker_fee_bps: float = 6.0   # per leg, per fill
    slippage_bps: float = 5.0    # per leg, per fill


@dataclass
class PairStats:
    symbol_a: str
    symbol_b: str
    beta: float                  # hedge ratio: spread = ln(A) - beta*ln(B)
    correlation: float           # correlation of the two log-return series
    half_life_bars: float        # AR(1) mean-reversion half-life (lower = reverts faster)
    current_z: float
    current_signal: Signal


@dataclass
class PairBacktestStats:
    total_return_pct: float
    final_equity: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int


@dataclass
class PairReport:
    stats: PairStats
    timestamps: pd.Series
    spread: pd.Series
    zscore: pd.Series
    backtest: Optional[PairBacktestStats] = None
    equity_curve: Optional[List[Tuple[pd.Timestamp, float]]] = None


def hedge_ratio(log_a: np.ndarray, log_b: np.ndarray) -> float:
    """OLS slope of log_a on log_b (spread = log_a - beta*log_b). Uses an
    intercept so the spread is mean-centered by construction."""
    X = np.vstack([log_b, np.ones_like(log_b)]).T
    beta, _intercept = np.linalg.lstsq(X, log_a, rcond=None)[0]
    return float(beta)


def half_life(spread: np.ndarray) -> float:
    """Ornstein-Uhlenbeck half-life via AR(1): regress dS_t on S_{t-1}.
    half_life = -ln(2)/b where b is the mean-reversion coefficient. Returns
    inf if the spread isn't mean-reverting (b >= 0)."""
    s = spread[~np.isnan(spread)]
    if len(s) < 10:
        return float("inf")
    s_lag = s[:-1]
    ds = np.diff(s)
    X = np.vstack([s_lag, np.ones_like(s_lag)]).T
    b, _c = np.linalg.lstsq(X, ds, rcond=None)[0]
    if b >= 0:
        return float("inf")
    return float(-np.log(2) / b)


def _signal_from_z(z: float, cfg: PairConfig) -> Signal:
    if z >= cfg.entry_z:
        return "SHORT_SPREAD"   # spread rich -> sell A, buy B
    if z <= -cfg.entry_z:
        return "LONG_SPREAD"    # spread cheap -> buy A, sell B
    return "FLAT"


def analyze_pair(df_a: pd.DataFrame, df_b: pd.DataFrame, symbol_a: str, symbol_b: str,
                 cfg: PairConfig) -> PairReport:
    """Align two OHLCV frames on timestamp, fit the spread, and compute the
    current z-score / signal plus an honest backtest of the pair."""
    merged = pd.merge(
        df_a[["timestamp", "close"]].rename(columns={"close": "a"}),
        df_b[["timestamp", "close"]].rename(columns={"close": "b"}),
        on="timestamp", how="inner",
    ).reset_index(drop=True)
    if len(merged) < cfg.zscore_window + 20:
        raise ValueError(f"not enough overlapping candles for {symbol_a}/{symbol_b} (have {len(merged)})")

    log_a = np.log(merged["a"].to_numpy())
    log_b = np.log(merged["b"].to_numpy())
    beta = hedge_ratio(log_a, log_b)
    spread = pd.Series(log_a - beta * log_b, name="spread")

    ret_a = np.diff(log_a)
    ret_b = np.diff(log_b)
    correlation = float(np.corrcoef(ret_a, ret_b)[0, 1])
    hl = half_life(spread.to_numpy())

    roll_mean = spread.rolling(cfg.zscore_window, min_periods=cfg.zscore_window).mean()
    roll_std = spread.rolling(cfg.zscore_window, min_periods=cfg.zscore_window).std(ddof=0)
    zscore = (spread - roll_mean) / roll_std.replace(0, np.nan)

    current_z = float(zscore.iloc[-1]) if not np.isnan(zscore.iloc[-1]) else 0.0
    stats = PairStats(
        symbol_a=symbol_a, symbol_b=symbol_b, beta=beta, correlation=correlation,
        half_life_bars=hl, current_z=current_z, current_signal=_signal_from_z(current_z, cfg),
    )

    bt_stats, equity_curve = _backtest_pair(merged["timestamp"], log_a, log_b, beta, zscore.to_numpy(), cfg)

    return PairReport(
        stats=stats,
        timestamps=merged["timestamp"],
        spread=spread,
        zscore=zscore,
        backtest=bt_stats,
        equity_curve=equity_curve,
    )


def _backtest_pair(timestamps: pd.Series, log_a: np.ndarray, log_b: np.ndarray, beta: float,
                   zscore: np.ndarray, cfg: PairConfig, starting_equity: float = 200.0):
    """Trade the spread market-neutral. Each bar the position earns
    dS = ret_a - beta*ret_b (long spread) or its negative (short spread).
    Gross exposure per trade is (1 + |beta|) units of equity, so costs are
    charged on that on entry and again on exit - four leg-fills round trip."""
    fee = cfg.taker_fee_bps / 10000.0
    slip = cfg.slippage_bps / 10000.0
    leg_cost = fee + slip
    gross = 1.0 + abs(beta)
    roundtrip_side_cost = gross * leg_cost  # one side (both legs); charged on entry and on exit

    equity = starting_equity
    peak = starting_equity
    position = 0  # +1 long spread, -1 short spread, 0 flat
    trade_open_equity = 0.0
    trades = []
    curve: List[Tuple[pd.Timestamp, float]] = []

    for i in range(1, len(zscore)):
        z = zscore[i]
        if np.isnan(z):
            curve.append((timestamps.iloc[i], equity))
            continue

        # Mark-to-market the open position on this bar's move.
        if position != 0:
            d_spread = (log_a[i] - log_a[i - 1]) - beta * (log_b[i] - log_b[i - 1])
            equity += position * d_spread * equity  # compounding, market-neutral leg move

        # Manage an open position: exit on reversion, or hard-stop on blow-out.
        if position != 0:
            reverted = abs(z) <= cfg.exit_z
            stopped = abs(z) >= cfg.stop_z
            if reverted or stopped:
                equity -= equity * roundtrip_side_cost  # exit costs (both legs)
                trades.append(equity - trade_open_equity)
                position = 0

        # Look for a new entry when flat.
        if position == 0:
            sig = _signal_from_z(z, cfg)
            if sig != "FLAT":
                position = 1 if sig == "LONG_SPREAD" else -1
                equity -= equity * roundtrip_side_cost  # entry costs (both legs)
                trade_open_equity = equity

        peak = max(peak, equity)
        curve.append((timestamps.iloc[i], equity))

    stats = _pair_backtest_stats(starting_equity, equity, trades, curve)
    return stats, curve


def _pair_backtest_stats(starting_equity, final_equity, trades, curve) -> PairBacktestStats:
    total_return_pct = (final_equity - starting_equity) / starting_equity * 100.0
    if curve:
        eq = np.array([e for _, e in curve])
        running_max = np.maximum.accumulate(eq)
        dd = (running_max - eq) / running_max
        max_dd = float(dd.max() * 100.0)
        rets = np.diff(eq) / eq[:-1]
        sharpe = float(rets.mean() / rets.std() * np.sqrt(365 * 6)) if len(rets) > 1 and rets.std() > 0 else 0.0
    else:
        max_dd, sharpe = 0.0, 0.0
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    win_rate = (len(wins) / len(trades) * 100.0) if trades else 0.0
    gp = sum(wins)
    gl = abs(sum(losses))
    pf = (gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)
    return PairBacktestStats(
        total_return_pct=total_return_pct, final_equity=final_equity, max_drawdown_pct=max_dd,
        sharpe_ratio=sharpe, win_rate=win_rate, profit_factor=pf, total_trades=len(trades),
    )


def all_pairs(symbols: List[str]) -> List[Tuple[str, str]]:
    out = []
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            out.append((symbols[i], symbols[j]))
    return out
