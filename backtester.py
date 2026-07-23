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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import indicators
from config import Config
from exchange import MarketLimits, clamp_qty_to_market
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
    """Bars to skip before trading so every indicator the ACTIVE strategy uses
    is warmed up. Strategy-aware on purpose: the ensemble leans on a 200-bar
    trend EMA, but cross_sectional_momentum never touches it and only needs
    its momentum lookback + ATR. Charging every strategy the ensemble's
    200-bar warmup wasted ~200 of only ~700 bars on the daily timeframe."""
    ind = cfg.indicators
    if cfg.strategy.mode == "cross_sectional_momentum":
        needed = max(cfg.strategy.momentum_lookback_bars, ind.atr_period, ind.bb_period)
        # allow for the whole optimizer lookback grid, so warmup is stable
        # regardless of which lookback a given fold selects
        needed = max(needed, 90)
        return needed + 5
    if cfg.strategy.mode == "mean_reversion_scalp":
        return max(ind.bb_period, ind.rsi_fast_period, ind.atr_period, ind.adx_period) + 5
    return max(ind.ema_trend_filter, ind.adx_period, ind.bb_period) + 5


def align_on_common_timestamps(dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Restrict every symbol's DataFrame to the timestamps ALL symbols share,
    so a shared positional index `i` means the same calendar bar for every
    symbol.

    The event loop indexes each symbol by row position (dfs[sym].iloc[i]).
    That's only valid if position i is the same date across symbols. When
    every series has full, identical history (e.g. BTC/ETH/SOL fetched from
    the same `since`) that already holds and this is a no-op. But the
    cross-sectional strategy ranks symbols against each other at the same i,
    and a newer coin with a shorter history would otherwise have its bar i sit
    at a different date than BTC's bar i - silently comparing mismatched
    dates. Inner-joining on timestamp fixes that; indicators are computed
    BEFORE this trim so each symbol's lookback windows stay intact.
    """
    if len(dfs) <= 1:
        return dfs
    common = None
    for df in dfs.values():
        ts = set(df["timestamp"])
        common = ts if common is None else (common & ts)
    if not common:
        raise ValueError("symbols share no common timestamps - cannot align them for a joint backtest")
    aligned = {}
    for sym, df in dfs.items():
        trimmed = df[df["timestamp"].isin(common)].sort_values("timestamp").reset_index(drop=True)
        aligned[sym] = trimmed
    return aligned


def run_backtest(
    cfg: Config, price_data: Dict[str, pd.DataFrame], market_limits: Optional[Dict[str, MarketLimits]] = None,
) -> BacktestResult:
    """Compute indicators fresh and backtest the full available range.

    For repeated backtests over the same price history with different risk/
    regime parameters (e.g. walk-forward optimization), prefer computing the
    indicator DataFrames once with `indicators.add_all_indicators`, aligning
    them with `align_on_common_timestamps`, and calling `simulate()` directly
    - it's the same event loop without redundant indicator recomputation,
    which dominates the cost of a grid search.
    """
    dfs = {sym: indicators.add_all_indicators(df, cfg.indicators) for sym, df in price_data.items()}
    dfs = align_on_common_timestamps(dfs)
    min_len = min(len(df) for df in dfs.values())
    warmup = warmup_bars(cfg)
    if min_len <= warmup:
        raise ValueError(f"not enough candles ({min_len}) to warm up indicators (need > {warmup})")
    return simulate(cfg, dfs, warmup, min_len, market_limits=market_limits)


def simulate(
    cfg: Config, dfs: Dict[str, pd.DataFrame], start_index: int, end_index: int,
    market_limits: Optional[Dict[str, MarketLimits]] = None,
) -> BacktestResult:
    """Core event loop over precomputed indicator DataFrames, trading only
    bars in [start_index, end_index). Bars before start_index are still
    visible to the strategy (needed for indicator lookback / regime context)
    but no entries or exits happen there - this is what lets a walk-forward
    optimizer hand in the same full-history DataFrame for every fold and just
    move the window, without recomputing indicators per fold or per candidate
    parameter set.

    market_limits, when provided, rounds every entry's computed quantity down
    to what the exchange would actually accept and skips the trade if it
    can't clear the exchange's minimum order size - the honest thing to do
    for a small account, where a 1% risk trade on a volatile pick can size
    out below Bitget's $5 minimum notional. Omit it (as tests do) to size
    purely off the risk model, uncapped by exchange mechanics.
    """
    strategy = build_strategy(cfg)
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

        signals = strategy.generate_signals(dfs, i)

        # 1. manage existing positions: stop/take-profit, trailing stop, then
        #    the strategy's own signal-based exit policy (e.g. dropped out of
        #    the top/bottom-K bucket, or flipped to the opposite side).
        for sym in list(portfolio.open_positions.keys()):
            bar = dfs[sym].iloc[i]
            position = portfolio.open_positions[sym]
            exit_price, reason = _check_exit(position, bar, cfg)
            if exit_price is not None:
                fee = exit_price * position.qty * fee_rate
                portfolio.close_position(sym, exit_price, fee, reason, closed_at=ts.to_pydatetime())
                continue

            signal = signals.get(sym)
            if signal is not None and strategy.should_exit_on_signal(position.side, signal):
                exit_price = bar["close"] * (1 - slip) if position.side == "LONG" else bar["close"] * (1 + slip)
                fee = exit_price * position.qty * fee_rate
                portfolio.close_position(sym, exit_price, fee, "signal_exit", closed_at=ts.to_pydatetime())
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
            signal: Signal = signals.get(sym)
            if not signal:
                continue

            row = df.iloc[i]
            price = row["close"]
            fill_price = price * (1 + slip) if signal.side == "LONG" else price * (1 - slip)
            stop, take = risk.compute_stop_and_take(signal.side, fill_price, row["atr"])
            qty = risk.position_size(portfolio.equity, fill_price, stop)
            if qty <= 0:
                continue

            if market_limits is not None and sym in market_limits:
                qty = clamp_qty_to_market(qty, fill_price, market_limits[sym])
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
