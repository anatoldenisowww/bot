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

# Parameters searched here deliberately exclude anything that changes the
# *indicator* DataFrames (EMA/RSI/BB/ATR/ADX periods) so those only need to
# be computed once per symbol for the whole optimization run, not once per
# candidate. adx_trend_threshold only affects how strategies.py *interprets*
# the ADX column, not how it's computed - safe to vary here. If you extend
# this grid with an indicator-period parameter, computing dfs must move
# inside the fold/combo loop.
PARAM_GRID: Dict[str, List[float]] = {
    "adx_trend_threshold": [20, 25, 30],
    "atr_stop_multiplier": [1.5, 2.0, 2.5],
    "take_profit_r_multiple": [1.5, 2.0, 2.5, 3.0],
}

# risk_per_trade_pct / leverage are intentionally NOT searched. Cranking up
# position size always makes a backtest look better right up until it
# doesn't - that's the classic way to turn a mediocre edge into a fake great
# one. Risk sizing is a risk-tolerance decision, not something to curve-fit.

MIN_TRADES_FOR_VALID_FOLD = 8
MAX_ACCEPTABLE_TRAIN_DRAWDOWN_PCT = 25.0


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


@dataclass
class WalkForwardResult:
    folds: List[FoldResult]
    chained_equity_curve: List[Tuple[pd.Timestamp, float]]
    chained_stats: BacktestStats


def _grid_combos():
    keys = list(PARAM_GRID)
    for values in itertools.product(*(PARAM_GRID[k] for k in keys)):
        yield dict(zip(keys, values))


def _apply_params(cfg: Config, params: Dict[str, float]) -> Config:
    trial = deepcopy(cfg)
    trial.indicators.adx_trend_threshold = params["adx_trend_threshold"]
    trial.risk.atr_stop_multiplier = params["atr_stop_multiplier"]
    trial.risk.take_profit_r_multiple = params["take_profit_r_multiple"]
    return trial


def _objective(stats: BacktestStats) -> float:
    """Risk-adjusted score with guardrails against picking a fluke.

    Raw return is not used as the objective on purpose - it's what a
    single lucky trade optimizes for. Sharpe with a trade-count floor and a
    drawdown ceiling rewards a parameter set that traded enough to mean
    something and didn't get there by taking reckless risk.
    """
    if stats.total_trades < MIN_TRADES_FOR_VALID_FOLD:
        return float("-inf")
    if stats.max_drawdown_pct > MAX_ACCEPTABLE_TRAIN_DRAWDOWN_PCT:
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

        for params in _grid_combos():
            trial_cfg = _apply_params(cfg, params)
            result = backtester.simulate(trial_cfg, dfs, train_start_idx, train_end_idx)
            score = _objective(result.stats)
            if score > best_score:
                best_score = score
                best_params = params
                best_train_stats = result.stats

        if best_params is None:
            # nothing in the grid cleared the guardrails on this window -
            # fall back to the settings.yaml defaults rather than force a pick.
            best_params = {
                "adx_trend_threshold": cfg.indicators.adx_trend_threshold,
                "atr_stop_multiplier": cfg.risk.atr_stop_multiplier,
                "take_profit_r_multiple": cfg.risk.take_profit_r_multiple,
            }
            best_train_stats = backtester.simulate(cfg, dfs, train_start_idx, train_end_idx).stats

        test_cfg = _apply_params(cfg, best_params)
        test_result = backtester.simulate(test_cfg, dfs, test_start_idx, test_end_idx)

        folds.append(FoldResult(
            fold_index=fold_index,
            train_start=timeline.iloc[train_start_idx],
            train_end=timeline.iloc[train_end_idx - 1],
            test_start=timeline.iloc[test_start_idx],
            test_end=timeline.iloc[test_end_idx - 1],
            best_params=best_params,
            train_stats=best_train_stats,
            test_result=test_result,
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
