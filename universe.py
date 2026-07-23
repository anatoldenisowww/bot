"""Liquidity-based asset selection for the cross-sectional strategy.

Trading a fixed 3-symbol universe (BTC/ETH/SOL) means one time-series signal
on three highly-correlated assets - a concentrated bet, not a diversified
one. This module ranks Bitget's full USDT-M perpetual universe by 24h quote
volume and keeps the selection logic as pure functions (no network calls),
so it's trivially testable with synthetic ticker data. The actual API call
lives on MarketDataFeed (exchange.py); this module only knows how to filter
and rank whatever ticker/market dicts it's handed.

Small-account note: a low-cap, illiquid perpetual is exactly where a small
account's nominal advantage (able to trade sizes too small for a real fund
to bother with) turns into a real disadvantage (wide spreads, thin books,
easy to move the market against yourself on entry/exit). The min_quote_volume_24h
floor exists to keep the universe to names where the assumed slippage_bps in
settings.yaml is still a reasonable estimate, not a fantasy.
"""
from __future__ import annotations

from typing import Dict, List


def is_eligible_perp_market(market: dict) -> bool:
    """USDT-margined linear perpetual swap, actively trading, no expiry.

    Bitget lists tokenized real-world-asset perpetuals (gold/XAU, silver/XAG,
    oil/CL, and tokenized single stocks like SOXL, MU, SNDK) under the exact
    same USDT-FUTURES product type as genuine crypto perpetuals - same
    swap/linear/USDT-quote shape, so the usual filters don't separate them.
    They're tagged `isRwa: "YES"` in the raw market info; excluded here
    because they track assets with real-world trading-hours gap risk and
    corporate-action risk that this strategy wasn't designed for, and mixing
    them in without dedicated handling would just be sloppy.
    """
    if str(market.get("info", {}).get("isRwa", "NO")).upper() == "YES":
        return False
    return bool(
        market.get("swap")
        and market.get("linear")
        and market.get("active", False)
        and market.get("quote") == "USDT"
        and market.get("settle") == "USDT"
        and not market.get("expiry")
    )


def rank_by_liquidity(tickers: Dict[str, dict], markets: Dict[str, dict]) -> List[tuple]:
    """Returns [(symbol, quote_volume_24h), ...] sorted descending, restricted
    to eligible perpetual markets with a usable quoteVolume figure."""
    ranked = []
    for symbol, ticker in tickers.items():
        market = markets.get(symbol)
        if not market or not is_eligible_perp_market(market):
            continue
        quote_volume = ticker.get("quoteVolume")
        if quote_volume is None:
            continue
        ranked.append((symbol, float(quote_volume)))
    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked


def select_universe(
    tickers: Dict[str, dict],
    markets: Dict[str, dict],
    top_n: int,
    min_quote_volume_24h: float,
    exclude: List[str] | None = None,
) -> List[str]:
    """Top-N most liquid eligible perpetuals clearing the volume floor.

    `exclude` is for symbols the strategy shouldn't trade even if liquid
    (e.g. stablecoin-pegged perpetuals, if any slip through the market
    filters - there are none on Bitget's USDT-M board today, but this keeps
    a deliberate override point rather than a silent assumption).
    """
    exclude_set = set(exclude or [])
    ranked = rank_by_liquidity(tickers, markets)
    selected = [
        symbol for symbol, volume in ranked
        if volume >= min_quote_volume_24h and symbol not in exclude_set
    ]
    return selected[:top_n]
