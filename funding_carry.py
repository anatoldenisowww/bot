"""Funding-rate carry (cash-and-carry) - the one structurally-grounded,
retail-accessible edge in this repo.

The idea, plainly:
Perpetual futures pay a "funding rate" every 8 hours. When funding is
POSITIVE, longs pay shorts. Historically it's positive most of the time
(over-eager longs paying to keep their leverage), so you can collect it by:

    long the coin on SPOT  +  short the same coin on the PERP  (equal size)

The two legs cancel out price direction (delta-neutral), so you don't care if
the coin goes up or down - you just harvest the funding the shorts receive.
This is CARRY, not prediction: you're paid a risk premium for supplying
leverage to the crowd. It is the closest thing to a real, persistent,
small-account-friendly edge here.

Why it's an edge YOU can capture (unlike prediction):
- It's structural, not a pattern that gets arbitraged away.
- The biggest, most persistent funding lives in coins too small for large
  funds to deploy into - being small is an advantage here, not a handicap.

Honest limits (this is rent, not riches):
- Returns are modest: liquid names pay ~3-8% annualized gross; after fees
  and the spot/perp basis, net is lower.
- Funding can turn negative (you'd pay) - it's positive ~80% of the time,
  not always. The backtest counts the negative periods.
- Basis risk: spot and perp prices can diverge; the hedge is delta-neutral
  in direction but not perfectly in basis. Modeled here as a drag.
- It needs BOTH a spot and a perp leg. At 200 EUR the exchange minimums
  ($5/leg) mean you can only run 1-2 positions, so little diversification -
  the strategy gets much better as the account grows.
- The eye-popping alt funding rates (hundreds of % annualized) are a TRAP:
  extreme, non-persistent, illiquid, and dangerous to hedge. The screener
  deliberately favors persistence and liquidity over headline rate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class CarryConfig:
    intervals_per_day: float = 3.0     # Bitget funding is every 8h -> 3/day
    taker_fee_bps: float = 6.0         # per leg, per fill
    slippage_bps: float = 5.0          # per leg, per fill
    # Ongoing spot/perp basis friction per interval held. CENTRAL ESTIMATE 0:
    # for a liquid coin held across funding intervals the perp is tugged back
    # toward spot, so expected basis PnL over a holding period is ~0. It is a
    # RISK (variance), not a guaranteed drag. Raise this to stress-test how a
    # persistent adverse basis would erode the carry.
    basis_drag_bps_per_interval: float = 0.0
    hold: str = "always"               # "always" (hold continuously - the realistic mode) | "positive_only"
    min_persistence_pct: float = 70.0  # screener: require funding positive at least this often
    min_net_annual_pct: float = 3.0    # screener: require this sustainable net carry to flag "worth it"


@dataclass
class CarryStats:
    symbol: str
    n_intervals: int
    days_covered: float
    mean_rate_pct: float          # mean funding per interval, %
    persistence_pct: float        # % of intervals with positive funding
    sustainable_annual_pct: float # annualized carry from the real funding series (incl. negative intervals)
    one_time_cost_pct: float      # one round-trip cost to open+close the paired position (NOT annualized)
    net_annual_pct: float         # sustainable_annual minus modeled ongoing basis drag
    worth_it: bool


@dataclass
class CarryReport:
    stats: CarryStats
    timestamps: pd.Series
    funding_rate: pd.Series
    equity_curve: List[Tuple[pd.Timestamp, float]]  # cumulative harvested carry (ex the one-time cost)


def _roundtrip_cost_fraction(cfg: CarryConfig) -> float:
    """Cash-and-carry touches 2 legs on entry and 2 on exit. Each leg pays
    fee + slippage -> one full open+close ~ 2 legs * 2 sides * (fee+slip).
    This is a ONE-TIME cost; held for months it amortizes to ~nothing, which
    is why it's reported separately rather than annualized over a short sample."""
    leg = (cfg.taker_fee_bps + cfg.slippage_bps) / 10000.0
    return 2 * 2 * leg


def backtest_carry(funding: pd.DataFrame, cfg: CarryConfig, symbol: str = "") -> CarryReport:
    """Harvest funding via a continuously-held delta-neutral cash-and-carry
    (long spot / short perp). Each interval you receive the funding rate
    (positive funding pays the short-perp leg); direction cancels. The real
    funding series - including the ~20% of intervals that are negative - drives
    the sustainable rate. The one-time round-trip cost is reported separately,
    because the intended holding period is months, not the sample length.

    "positive_only" mode stands aside during negative-funding intervals; it
    usually does WORSE because each re-entry pays a fresh round-trip - a lesson
    worth seeing, so it's kept as an option.
    """
    rates = funding["funding_rate"].to_numpy()
    ts = funding["timestamp"]
    n = len(rates)
    if n < 10:
        raise ValueError(f"not enough funding history for {symbol} ({n} intervals)")

    basis = cfg.basis_drag_bps_per_interval / 10000.0
    intervals_per_year = cfg.intervals_per_day * 365.0
    rt_cost = _roundtrip_cost_fraction(cfg)

    # Cumulative harvested carry (excluding the one-time entry/exit cost).
    equity = 1.0
    curve: List[Tuple[pd.Timestamp, float]] = []
    n_held = 0
    flip_costs = 0.0
    in_position = False
    for i in range(n):
        r = rates[i]
        want = True if cfg.hold == "always" else (r >= 0)
        if want != in_position:
            flip_costs += rt_cost / 2  # a side (both legs) each flip
            in_position = want
        if in_position:
            equity *= (1 + r - basis)
            n_held += 1
        curve.append((ts.iloc[i], equity))

    days_covered = n / cfg.intervals_per_day
    persistence = float((rates > 0).mean() * 100.0)

    # Sustainable annualized carry: compound the per-interval harvest over the
    # held intervals, then annualize by the held rate. This is the rate you'd
    # keep earning holding long-term - it is NOT charged the one-time cost.
    if n_held > 0 and equity > 0:
        per_interval_growth = equity ** (1 / n_held)
        sustainable_annual = (per_interval_growth ** intervals_per_year - 1) * 100.0
    else:
        sustainable_annual = 0.0

    # For "positive_only", the flip costs ARE a recurring drag, so fold them in.
    flip_drag_annual = 0.0
    if cfg.hold != "always" and days_covered > 0:
        flip_drag_annual = (flip_costs / days_covered) * 365.0 * 100.0

    net_annual = sustainable_annual - flip_drag_annual

    stats = CarryStats(
        symbol=symbol,
        n_intervals=n,
        days_covered=days_covered,
        mean_rate_pct=float(rates.mean()) * 100.0,
        persistence_pct=persistence,
        sustainable_annual_pct=sustainable_annual,
        one_time_cost_pct=rt_cost * 100.0,
        net_annual_pct=net_annual,
        worth_it=(persistence >= cfg.min_persistence_pct and net_annual >= cfg.min_net_annual_pct),
    )
    return CarryReport(stats=stats, timestamps=ts, funding_rate=funding["funding_rate"], equity_curve=curve)


def screen(reports: List[CarryReport]) -> List[CarryReport]:
    """Rank candidates: worth-it first, then by net annualized carry."""
    return sorted(reports, key=lambda r: (r.stats.worth_it, r.stats.net_annual_pct), reverse=True)
