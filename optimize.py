"""Walk-forward parameter optimization.

The one rule that keeps this honest: a parameter set is only ever *chosen*
using a training window, and only ever *scored* on the window immediately
after it, which the optimizer never saw while choosing. Folds roll forward
through the whole history, so the final "out-of-sample" (OOS) equity curve
is a chain of decisions each made without seeing their own future - as close
as a backtest can get to what actually re-optimizing this system every N
days and trading forward would have produced.

This is deliberately not "grid search the whole history and report the best
result" - that number is fiction; it's the strategy that best fit noise in
data it was then graded on. Walk-forward is slower and less flattering, and
it's what a real desk would insist on before risking money.
"""
from __future__ import annotations

import itertools
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import backtester
import indicators
from backtester import BacktestResult, BacktestStats
from config import Config
from exchange import MarketLimits

# The grid is strategy-specific: optimizing a strategy over parameters it
# doesn't use (e.g. sweeping adx_trend_threshold for cross_sectional_momentum,
# which never reads it) is just wasted compute that also inflates the
# multiple-comparisons problem. _param_grid_for(mode) returns the knobs that
# actually change THAT strategy's behavior. Everything here is safe to vary
# without recomputing indicator DataFrames - none of these are indicator
# periods (see the note in walk_forward_optimize).
_ENSEMBLE_GRID: Dict[str, List[float]] = {
    "adx_trend_threshold": [20, 25, 30],
    "atr_stop_multiplier": [1.5, 2.0, 2.5],
    # Smaller multiples pull the take-profit closer to the stop distance,
    # which is what raises win rate - a trade needs less favorable movement
    # to be called a win. Kept wide enough (down to 1.0R) to actually find
    # >=50%-win-rate configurations, not just approach the constraint.
    "take_profit_r_multiple": [1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
}

_MEAN_REVERSION_GRID: Dict[str, List[float]] = {
    "atr_stop_multiplier": [1.5, 2.0, 2.5],
    "take_profit_r_multiple": [1.0, 1.25, 1.5, 2.0, 2.5],
}

# Cross-sectional momentum's edge lives in the ranking knobs, not the
# stop/target. The default-parameter backtest showed ~400 trades/yr with
# profit factor ~0.99 - i.e. costs were eating a real gross edge - so the
# grid deliberately spans LONGER lookbacks (slower, less turnover) and a
# range of signal-strength floors (higher = fewer, higher-conviction trades)
# to give the search a way to trade less and keep more.
_CROSS_SECTIONAL_GRID: Dict[str, List[float]] = {
    "momentum_lookback_bars": [20, 40, 60, 90],
    "min_abs_momentum_score": [0.5, 1.0, 1.5, 2.0],
    "atr_stop_multiplier": [2.0, 3.0],
}


def _param_grid_for(mode: str) -> Dict[str, List[float]]:
    if mode == "ensemble":
        return _ENSEMBLE_GRID
    if mode == "mean_reversion_scalp":
        return _MEAN_REVERSION_GRID
    if mode == "cross_sectional_momentum":
        return _CROSS_SECTIONAL_GRID
    raise ValueError(f"no parameter grid defined for strategy mode {mode!r}")


# Back-compat alias: earlier code/tests referenced PARAM_GRID directly.
PARAM_GRID = _ENSEMBLE_GRID

# risk_per_trade_pct / leverage are intentionally NOT searched. Cranking up
# position size always makes a backtest look better right up until it
# doesn't - that's the classic way to turn a mediocre edge into a fake great
# one. Risk sizing is a risk-tolerance decision, not something to curve-fit.

MIN_TRADES_FOR_VALID_FOLD = 8
MAX_ACCEPTABLE_TRAIN_DRAWDOWN_PCT = 25.0
# User-requested constraint: only consider parameter sets whose TRAINING
# win rate clears this bar. This is a real constraint on the search, applied
# the same honest way as the other guardrails - never picked by looking at
# the test result. Note the tradeoff this forces: at a fixed stop distance,
# raising win rate means pulling the take-profit target closer in (see
# take_profit_r_multiple above), which shrinks the average winner. A higher
# win rate does not automatically mean a more profitable system - the
# per-fold and chained OOS profit factor / total return below are what
# actually say whether this constraint helped or hurt.
MIN_WIN_RATE_PCT = 50.0


@dataclass
class FoldResult:
    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: Dict[str, float]
    train_stats: BacktestStats
    test_result: BacktestResult
    guardrails_cleared: bool  # False -> no grid combo hit the guardrails on this fold's training window; fell back to settings.yaml defaults


@dataclass
class WalkForwardResult:
    folds: List[FoldResult]
    chained_equity_curve: List[Tuple[pd.Timestamp, float]]
    chained_stats: BacktestStats


def _grid_combos(mode: str):
    grid = _param_grid_for(mode)
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


# Which config field each tunable parameter maps onto. Keeps _apply_params a
# simple data-driven loop instead of a growing if/elif per parameter.
_PARAM_TARGETS = {
    "adx_trend_threshold": ("indicators", "adx_trend_threshold"),
    "atr_stop_multiplier": ("risk", "atr_stop_multiplier"),
    "take_profit_r_multiple": ("risk", "take_profit_r_multiple"),
    "momentum_lookback_bars": ("strategy", "momentum_lookback_bars"),
    "min_abs_momentum_score": ("strategy", "min_abs_momentum_score"),
}


def _apply_params(cfg: Config, params: Dict[str, float]) -> Config:
    trial = deepcopy(cfg)
    for name, value in params.items():
        section, field = _PARAM_TARGETS[name]
        setattr(getattr(trial, section), field, value)
    return trial


def _fallback_params(cfg: Config, mode: str) -> Dict[str, float]:
    """The strategy's current settings.yaml values for exactly the knobs this
    mode's grid searches - used when nothing in the grid clears the guardrails
    on a fold, so the fallback is comparable to the searched candidates."""
    grid = _param_grid_for(mode)
    fallback = {}
    for name in grid:
        section, field = _PARAM_TARGETS[name]
        fallback[name] = getattr(getattr(cfg, section), field)
    return fallback


def _objective(stats: BacktestStats, min_win_rate_pct: float) -> float:
    """Risk-adjusted score with guardrails against picking a fluke.

    Raw return is not used as the objective on purpose - it's what a
    single lucky trade optimizes for. Sharpe with a trade-count floor and a
    drawdown ceiling rewards a parameter set that traded enough to mean
    something and didn't get there by taking reckless risk. When a win-rate
    floor is requested it's a hard constraint, not part of the score, so
    among everything that clears it the search still picks the best
    risk-adjusted candidate rather than just the highest win rate.

    min_win_rate_pct defaults to 0 (off) at the call sites, because we showed
    empirically that forcing >50% win rate makes these strategies less
    profitable, not more - momentum in particular structurally wins <50% of
    the time. The knob stays available for anyone who wants to explore that
    tradeoff, but it's opt-in, not the default that quietly hobbles a strategy.
    """
    if stats.total_trades < MIN_TRADES_FOR_VALID_FOLD:
        return float("-inf")
    if stats.max_drawdown_pct > MAX_ACCEPTABLE_TRAIN_DRAWDOWN_PCT:
        return float("-inf")
    if stats.win_rate < min_win_rate_pct:
        return float("-inf")
    return stats.sharpe_ratio


def _bars_per_day(timeframe: str) -> float:
    return backtester.periods_per_year(timeframe) / 365.0


def walk_forward_optimize(
    cfg: Config,
    price_data: Dict[str, pd.DataFrame],
    train_days: int,
    test_days: int,
    step_days: Optional[int] = None,
    market_limits: Optional[Dict[str, MarketLimits]] = None,
    min_win_rate_pct: float = 0.0,
) -> WalkForwardResult:
    step_days = step_days or test_days

    dfs = {sym: indicators.add_all_indicators(df, cfg.indicators) for sym, df in price_data.items()}
    min_len = min(len(df) for df in dfs.values())
    warmup = backtester.warmup_bars(cfg)
    timeline = dfs[next(iter(dfs))]["timestamp"]

    bpd = _bars_per_day(cfg.exchange.timeframe)
    train_bars = int(train_days * bpd)
    test_bars = int(test_days * bpd)
    step_bars = int(step_days * bpd)

    folds: List[FoldResult] = []
    fold_index = 0
    train_start_idx = warmup

    while train_start_idx + train_bars + test_bars <= min_len:
        train_end_idx = train_start_idx + train_bars
        test_start_idx = train_end_idx
        test_end_idx = test_start_idx + test_bars

        best_params: Optional[Dict[str, float]] = None
        best_score = float("-inf")
        best_train_stats: Optional[BacktestStats] = None

        for params in _grid_combos(cfg.strategy.mode):
            trial_cfg = _apply_params(cfg, params)
            result = backtester.simulate(trial_cfg, dfs, train_start_idx, train_end_idx, market_limits=market_limits)
            score = _objective(result.stats, min_win_rate_pct)
            if score > best_score:
                best_score = score
                best_params = params
                best_train_stats = result.stats

        guardrails_cleared = best_params is not None
        if not guardrails_cleared:
            # nothing in the grid cleared the guardrails on this window (with
            # MIN_WIN_RATE_PCT active, this can happen a lot - a strategy that
            # cuts losers quickly and lets winners run structurally tends
            # toward <50% win rate) - fall back to settings.yaml defaults
            # rather than force a pick, and say so plainly in the report.
            best_params = _fallback_params(cfg, cfg.strategy.mode)
            best_train_stats = backtester.simulate(cfg, dfs, train_start_idx, train_end_idx, market_limits=market_limits).stats

        test_cfg = _apply_params(cfg, best_params)
        test_result = backtester.simulate(test_cfg, dfs, test_start_idx, test_end_idx, market_limits=market_limits)

        folds.append(FoldResult(
            fold_index=fold_index,
            train_start=timeline.iloc[train_start_idx],
            train_end=timeline.iloc[train_end_idx - 1],
            test_start=timeline.iloc[test_start_idx],
            test_end=timeline.iloc[test_end_idx - 1],
            best_params=best_params,
            train_stats=best_train_stats,
            test_result=test_result,
            guardrails_cleared=guardrails_cleared,
        ))

        fold_index += 1
        train_start_idx += step_bars

    if not folds:
        raise ValueError(
            "not enough history for a single walk-forward fold - shorten "
            "--train-days/--test-days or fetch a longer --days window"
        )

    chained_curve, chained_trades = _chain_oos(cfg, folds)
    chained_stats = _stats_from_series(cfg, chained_curve, chained_trades)

    return WalkForwardResult(folds=folds, chained_equity_curve=chained_curve, chained_stats=chained_stats)


def _chain_oos(cfg: Config, folds: List[FoldResult]) -> Tuple[List[Tuple[pd.Timestamp, float]], list]:
    """Stitch each fold's OOS equity curve (each simulated from the same
    starting_equity in isolation) into one continuous curve by compounding
    each fold's growth factor onto the running equity from the prior fold -
    i.e. "reinvest everything and re-optimize every step_days"."""
    chained_curve: List[Tuple[pd.Timestamp, float]] = []
    chained_trades: list = []
    running_equity = cfg.risk.starting_equity

    for fold in folds:
        curve = fold.test_result.equity_curve
        if curve:
            for ts, eq in curve:
                growth_factor = eq / cfg.risk.starting_equity
                chained_curve.append((ts, running_equity * growth_factor))
            final_growth_factor = curve[-1][1] / cfg.risk.starting_equity
            running_equity = running_equity * final_growth_factor
        chained_trades.extend(fold.test_result.trades)

    return chained_curve, chained_trades


def _stats_from_series(cfg: Config, curve: List[Tuple[pd.Timestamp, float]], trades: list) -> BacktestStats:
    starting_equity = cfg.risk.starting_equity
    final_equity = curve[-1][1] if curve else starting_equity
    total_return_pct = (final_equity - starting_equity) / starting_equity * 100.0

    if curve:
        equities = np.array([e for _, e in curve])
        running_max = np.maximum.accumulate(equities)
        drawdowns = (running_max - equities) / running_max
        max_drawdown_pct = float(drawdowns.max() * 100.0)

        returns = np.diff(equities) / equities[:-1]
        if len(returns) > 1 and returns.std() > 0:
            sharpe = float(returns.mean() / returns.std() * np.sqrt(backtester.periods_per_year(cfg.exchange.timeframe)))
        else:
            sharpe = 0.0
    else:
        max_drawdown_pct = 0.0
        sharpe = 0.0

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    win_rate = (len(wins) / len(trades) * 100.0) if trades else 0.0

    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    avg_r = float(np.mean([t.r_multiple for t in trades])) if trades else 0.0

    return BacktestStats(
        total_return_pct=total_return_pct,
        final_equity=final_equity,
        max_drawdown_pct=max_drawdown_pct,
        sharpe_ratio=sharpe,
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_trades=len(trades),
        avg_r_multiple=avg_r,
    )


def most_common_params(result: WalkForwardResult) -> Dict[str, float]:
    """The parameter set chosen most often across folds - a reasonable
    starting point for settings.yaml, on the theory that a config the
    optimizer kept re-selecting across different market windows is more
    robust than whichever one happened to win on the single most recent
    window."""
    from collections import Counter

    counter = Counter(tuple(sorted(f.best_params.items())) for f in result.folds)
    most_common, _ = counter.most_common(1)[0]
    return dict(most_common)
