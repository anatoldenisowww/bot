"""Signal generation.

Design: detect the market regime first (trending vs ranging, via ADX on the
execution timeframe), then let the sub-strategy suited to that regime drive
the decision. A higher-timeframe EMA filter has veto power over direction -
we never take a countertrend trade against the higher timeframe.

Every strategy is a pure function of a DataFrame (already carrying indicator
columns from indicators.add_all_indicators) -> Signal. No I/O, no state,
easy to unit test and to run inside the backtester and the live bot alike.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

import pandas as pd

Side = Literal["LONG", "SHORT", "FLAT"]


@dataclass
class Signal:
    side: Side
    confidence: float  # 0..1
    regime: str
    reasons: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.side != "FLAT"


def _trend_bias(row: pd.Series) -> Optional[Side]:
    """Higher-timeframe-style veto using the long EMA on the same frame."""
    if row["close"] > row["ema_trend"]:
        return "LONG"
    if row["close"] < row["ema_trend"]:
        return "SHORT"
    return None


def trend_following_signal(df: pd.DataFrame) -> Signal:
    """EMA fast/slow crossover, confirmed by MACD histogram direction."""
    row = df.iloc[-1]
    prev = df.iloc[-2]

    bias = _trend_bias(row)
    reasons: List[str] = []

    crossed_up = prev["ema_fast"] <= prev["ema_slow"] and row["ema_fast"] > row["ema_slow"]
    crossed_down = prev["ema_fast"] >= prev["ema_slow"] and row["ema_fast"] < row["ema_slow"]

    if crossed_up and bias == "LONG" and row["macd_hist"] > 0:
        reasons = ["ema_fast crossed above ema_slow", "price above long-term EMA", "MACD histogram positive"]
        return Signal("LONG", 0.65, "trend", reasons)
    if crossed_down and bias == "SHORT" and row["macd_hist"] < 0:
        reasons = ["ema_fast crossed below ema_slow", "price below long-term EMA", "MACD histogram negative"]
        return Signal("SHORT", 0.65, "trend", reasons)

    return Signal("FLAT", 0.0, "trend")


def mean_reversion_signal(df: pd.DataFrame, rsi_oversold: float, rsi_overbought: float) -> Signal:
    """Bollinger Band extremes + RSI confirmation, only meant to run in ranging regimes."""
    row = df.iloc[-1]
    bias = _trend_bias(row)

    if row["close"] <= row["bb_lower"] and row["rsi"] <= rsi_oversold and bias != "SHORT":
        return Signal(
            "LONG", 0.55, "range",
            ["price at/below lower Bollinger band", f"RSI {row['rsi']:.1f} oversold"],
        )
    if row["close"] >= row["bb_upper"] and row["rsi"] >= rsi_overbought and bias != "LONG":
        return Signal(
            "SHORT", 0.55, "range",
            ["price at/above upper Bollinger band", f"RSI {row['rsi']:.1f} overbought"],
        )
    return Signal("FLAT", 0.0, "range")


def breakout_signal(df: pd.DataFrame) -> Signal:
    """Donchian channel breakout, confirmed by rising ADX (regime turning trendy)."""
    row = df.iloc[-1]
    prev = df.iloc[-2]

    if row["close"] > prev["donchian_upper"] and row["adx"] > prev["adx"]:
        return Signal("LONG", 0.6, "breakout", [f"close broke {prev['donchian_upper']:.4f} Donchian high", "ADX rising"])
    if row["close"] < prev["donchian_lower"] and row["adx"] > prev["adx"]:
        return Signal("SHORT", 0.6, "breakout", [f"close broke {prev['donchian_lower']:.4f} Donchian low", "ADX rising"])
    return Signal("FLAT", 0.0, "breakout")


class EnsembleStrategy:
    """Regime router: picks which sub-strategy gets to speak, based on ADX.

    ADX >= threshold  -> trending regime  -> trend-following + breakout vote
    ADX <  threshold  -> ranging regime   -> mean-reversion only
    """

    def __init__(self, indicator_cfg):
        self.cfg = indicator_cfg

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        if len(df) < max(self.cfg.ema_trend_filter, self.cfg.adx_period, self.cfg.bb_period) + 5:
            return Signal("FLAT", 0.0, "insufficient_data", ["not enough candles for stable indicators"])

        row = df.iloc[-1]
        if pd.isna(row[["ema_trend", "adx", "atr", "bb_upper"]]).any():
            return Signal("FLAT", 0.0, "warming_up", ["indicators still warming up"])

        trending = row["adx"] >= self.cfg.adx_trend_threshold

        if trending:
            trend_sig = trend_following_signal(df)
            if trend_sig:
                return trend_sig
            breakout_sig = breakout_signal(df)
            if breakout_sig:
                return breakout_sig
            return Signal("FLAT", 0.0, "trend", ["trending regime, no confirmed entry"])

        range_sig = mean_reversion_signal(df, self.cfg.rsi_oversold, self.cfg.rsi_overbought)
        if range_sig:
            return range_sig
        return Signal("FLAT", 0.0, "range", ["ranging regime, no extreme reached"])


class MeanReversionScalpStrategy:
    """Fades short-term statistical extremes: a very short RSI (period 2 by
    default) at a deep oversold/overbought reading, confirmed by the price
    also touching the Bollinger Band. Structurally different from
    EnsembleStrategy - it trades in any regime, far more often, on a
    smaller edge per trade. That combination tends toward a higher win rate
    (frequent small reversions) at the cost of a lower average win, which is
    exactly the tradeoff the walk-forward optimizer showed hurts profitability
    when forced onto the trend-following ensemble. Whether it's actually more
    profitable *as a strategy in its own right* is an empirical question -
    validate with `main.py optimize`, don't assume.

    The one safety filter kept: don't fade a violently strong trend (ADX very
    high, price already on the trend side of the long EMA) - that's the
    classic way mean-reversion systems take a catastrophic loss, buying every
    dip on the way to zero.
    """

    STRONG_TREND_ADX = 40.0

    def __init__(self, indicator_cfg):
        self.cfg = indicator_cfg

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        min_len = max(self.cfg.bb_period, self.cfg.rsi_fast_period, self.cfg.atr_period, self.cfg.adx_period) + 5
        if len(df) < min_len:
            return Signal("FLAT", 0.0, "insufficient_data", ["not enough candles for stable indicators"])

        row = df.iloc[-1]
        if pd.isna(row[["rsi_fast", "bb_upper", "bb_lower", "atr", "adx"]]).any():
            return Signal("FLAT", 0.0, "warming_up", ["indicators still warming up"])

        violent_trend = row["adx"] >= self.STRONG_TREND_ADX
        bias = _trend_bias(row)

        if row["close"] <= row["bb_lower"] and row["rsi_fast"] <= self.cfg.rsi_fast_oversold:
            if violent_trend and bias == "SHORT":
                return Signal("FLAT", 0.0, "mr_scalp", ["skipped: would be fading a violently strong downtrend"])
            return Signal(
                "LONG", 0.6, "mr_scalp",
                [f"RSI({self.cfg.rsi_fast_period}) {row['rsi_fast']:.1f} extreme oversold", "price at/below lower Bollinger band"],
            )

        if row["close"] >= row["bb_upper"] and row["rsi_fast"] >= self.cfg.rsi_fast_overbought:
            if violent_trend and bias == "LONG":
                return Signal("FLAT", 0.0, "mr_scalp", ["skipped: would be fading a violently strong uptrend"])
            return Signal(
                "SHORT", 0.6, "mr_scalp",
                [f"RSI({self.cfg.rsi_fast_period}) {row['rsi_fast']:.1f} extreme overbought", "price at/above upper Bollinger band"],
            )

        return Signal("FLAT", 0.0, "mr_scalp")


def build_strategy(mode: str, indicator_cfg):
    if mode == "ensemble":
        return EnsembleStrategy(indicator_cfg)
    if mode == "mean_reversion_scalp":
        return MeanReversionScalpStrategy(indicator_cfg)
    raise ValueError(f"unknown strategy mode: {mode!r}")
