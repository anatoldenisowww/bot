"""Event-driven backtest: replays historical candles bar-by-bar across the
whole symbol universe, using the exact same RiskManager and strategy
(selected via cfg.strategy.mode, see strategies.build_strategy) the live bot
uses. Fees and slippage are charged on every fill.

Simplifying assumption: symbols are iterated by row position, not merged on
timestamp. This holds as long as all OHLCV series were fetched with the same
timeframe and the same `since`, which `main.py backtest` always does.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import indicators
from config import Config
from models import ClosedTrade, Position, utcnow
from portfolio import PortfolioManager
from risk_manager import RiskManager
from strategies import Signal, build_strategy


@dataclass
class BacktestStats:
    total_return_pct: float
    final_equity: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_r_multiple: float


@dataclass
class BacktestResult:
    equity_curve: List[Tuple[pd.Timestamp, float]]
    trades: List[ClosedTrade]
    stats: BacktestStats


def _fee_rate(cfg: Config) -> float:
    return cfg.exchange.taker_fee_bps / 10000.0


def _slip(cfg: Config) -> float:
    return cfg.exchange.slippage_bps / 10000.0


def _check_exit(position: Position, bar: pd.Series, cfg: Config) -> Tuple[float, str] | Tuple[None, None]:
    slip = _slip(cfg)
    if position.side == "LONG":
        if bar["low"] <= position.stop_price:
            return position.stop_price * (1 - slip), "stop_loss"
        if bar["high"] >= position.take_profit_price:
            return position.take_profit_price * (1 - slip), "take_profit"
    else:
        if bar["high"] >= position.stop_price:
            return position.stop_price * (1 + slip), "stop_loss"
        if bar["low"] <= position.take_profit_price:
            return position.take_profit_price * (1 + slip), "take_profit"
    return None, None


def warmup_bars(cfg: Config) -> int:
    return max(cfg.indicators.ema_trend_filter, cfg.indicators.adx_period, cfg.indicators.bb_period) + 5


def run_backtest(cfg: Config, price_data: Dict[str, pd.DataFrame]) -> BacktestResult:
    """Compute indicators fresh and backtest the full available range.

    For repeated backtests over the same price history with different risk/
    regime parameters (e.g. walk-forward optimization), prefer computing the
    indicator DataFrames once with `indicators.add_all_indicators` and calling
    `simulate()` directly - it's the same event loop without redundant
    indicator recomputation, which dominates the cost of a grid search.
    """
    dfs = {sym: indicators.add_all_indicators(df, cfg.indicators) for sym, df in price_data.items()}
    min_len = min(len(df) for df in dfs.values())
    warmup = warmup_bars(cfg)
    if min_len <= warmup:
        raise ValueError(f"not enough candles ({min_len}) to warm up indicators (need > {warmup})")
    return simulate(cfg, dfs, warmup, min_len)


def simulate(cfg: Config, dfs: Dict[str, pd.DataFrame], start_index: int, end_index: int) -> BacktestResult:
    """Core event loop over precomputed indicator DataFrames, trading only
    bars in [start_index, end_index). Bars before start_index are still
    visible to the strategy (needed for indicator lookback / regime context)
    but no entries or exits happen there - this is what lets a walk-forward
    optimizer hand in the same full-history DataFrame for every fold and just
    move the window, without recomputing indicators per fold or per candidate
    parameter set.
    """
    strategy = build_strategy(cfg.strategy.mode, cfg.indicators)
    risk = RiskManager(cfg.risk)
    portfolio = PortfolioManager(starting_equity=cfg.risk.starting_equity)

    equity_curve: List[Tuple[pd.Timestamp, float]] = []
    fee_rate = _fee_rate(cfg)
    slip = _slip(cfg)

    timeline = dfs[next(iter(dfs))]["timestamp"]

    for i in range(start_index, end_index):
        ts = timeline.iloc[i]
        portfolio.advance_clock(ts.to_pydatetime())
        current_prices = {sym: dfs[sym].iloc[i]["close"] for sym in dfs}

        # 1. manage existing positions: stop/take-profit, then trailing stop.
        for sym in list(portfolio.open_positions.keys()):
            bar = dfs[sym].iloc[i]
            position = portfolio.open_positions[sym]
            exit_price, reason = _check_exit(position, bar, cfg)
            if exit_price is not None:
                fee = exit_price * position.qty * fee_rate
                portfolio.close_position(sym, exit_price, fee, reason, closed_at=ts.to_pydatetime())
                continue
            new_stop = risk.update_trailing_stop(position, bar["close"], bar["atr"])
            if new_stop is not None:
                position.stop_price = new_stop
                position.trailing_active = True

        portfolio.update_peak_equity(current_prices)

        # 2. look for new entries on symbols that are flat.
        for sym, df in dfs.items():
            if sym in portfolio.open_positions:
                continue
            window = df.iloc[: i + 1]
            signal: Signal = strategy.generate_signal(window)
            if not signal:
                continue

            row = df.iloc[i]
            price = row["close"]
            fill_price = price * (1 + slip) if signal.side == "LONG" else price * (1 - slip)
            stop, take = risk.compute_stop_and_take(signal.side, fill_price, row["atr"])
            qty = risk.position_size(portfolio.equity, fill_price, stop)
            if qty <= 0:
                continue

            proposed_risk = abs(fill_price - stop) * qty
            decision = risk.can_open_new_trade(
                sym, portfolio.equity, portfolio.peak_equity, portfolio.daily_pnl,
                portfolio.daily_start_equity, portfolio.open_positions, proposed_risk,
            )
            if not decision.allowed:
                continue

            fee = fill_price * qty * fee_rate
            position = Position(
                symbol=sym, side=signal.side, entry_price=fill_price, qty=qty,
                stop_price=stop, take_profit_price=take,
                initial_risk_per_unit=abs(fill_price - stop), opened_at=ts.to_pydatetime(),
            )
            portfolio.equity -= fee
            portfolio.open_position(position)

        equity_curve.append((ts, portfolio.mark_to_market_equity(current_prices)))

    # close anything still open at the end, mark-to-market, for a clean final number
    final_prices = {sym: dfs[sym].iloc[end_index - 1]["close"] for sym in dfs}
    for sym in list(portfolio.open_positions.keys()):
        price = final_prices[sym]
        fee = price * portfolio.open_positions[sym].qty * fee_rate
        portfolio.close_position(sym, price, fee, "end_of_backtest", closed_at=utcnow())

    stats = _compute_stats(cfg, portfolio, equity_curve)
    return BacktestResult(equity_curve=equity_curve, trades=portfolio.closed_trades, stats=stats)


def _compute_stats(cfg: Config, portfolio: PortfolioManager, equity_curve: List[Tuple[pd.Timestamp, float]]) -> BacktestStats:
    trades = portfolio.closed_trades
    starting_equity = cfg.risk.starting_equity
    final_equity = portfolio.equity

    total_return_pct = (final_equity - starting_equity) / starting_equity * 100.0

    if equity_curve:
        equities = np.array([e for _, e in equity_curve])
        running_max = np.maximum.accumulate(equities)
        drawdowns = (running_max - equities) / running_max
        max_drawdown_pct = float(drawdowns.max() * 100.0)

        returns = np.diff(equities) / equities[:-1]
        if len(returns) > 1 and returns.std() > 0:
            sharpe = float(returns.mean() / returns.std() * np.sqrt(periods_per_year(cfg.exchange.timeframe)))
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


def periods_per_year(timeframe: str) -> float:
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    minutes_per_period = {"m": value, "h": value * 60, "d": value * 60 * 24}.get(unit)
    if not minutes_per_period:
        return 365.0
    return (365.0 * 24 * 60) / minutes_per_period
