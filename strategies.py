"""Signal generation.

Two families of strategy, both driven off indicator columns already computed
by indicators.add_all_indicators - no I/O, no hidden state, easy to unit
test and to run inside the backtester and the live bot alike.

- PerSymbolStrategy subclasses (EnsembleStrategy, MeanReversionScalpStrategy)
  decide each symbol independently from its own history.
- CrossSectionalMomentumStrategy decides jointly across the whole tradable
  universe at once (rank everyone, trade the extremes) - it structurally
  cannot answer "what's the signal for this one symbol" without seeing
  every other symbol's current data too.

Both shapes expose the same batch interface the backtester and bot actually
call: generate_signals(dfs, i) -> {symbol: Signal}, plus
should_exit_on_signal(position_side, signal) -> bool, which is a genuine
policy difference between the two families (see CrossSectionalMomentumStrategy
docstring) and not just an implementation detail.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

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


class PerSymbolStrategy:
    """Base for strategies whose decision only needs one symbol's own
    history. Subclasses implement generate_signal(df); this provides the
    batch generate_signals(dfs, i) interface the backtester/bot actually
    call, and the default exit policy: exit only on a true flip to the
    opposite side, not merely fading to FLAT. That's a deliberate choice,
    not an oversight - these strategies were walk-forward validated with
    that exact exit rule, and loosening it would change validated behavior.
    """

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        raise NotImplementedError

    def generate_signals(self, dfs: Dict[str, pd.DataFrame], i: Optional[int] = None) -> Dict[str, Signal]:
        signals = {}
        for sym, df in dfs.items():
            idx = i if i is not None else len(df) - 1
            signals[sym] = self.generate_signal(df.iloc[: idx + 1])
        return signals

    def should_exit_on_signal(self, position_side: Side, signal: Signal) -> bool:
        opposite = "SHORT" if position_side == "LONG" else "LONG"
        return bool(signal) and signal.side == opposite


class EnsembleStrategy(PerSymbolStrategy):
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


class MeanReversionScalpStrategy(PerSymbolStrategy):
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


class CrossSectionalMomentumStrategy:
    """Ranks the whole tradable universe by risk-adjusted momentum each bar
    and takes positions in the strongest and weakest names, instead of one
    time-series signal on one symbol at a time.

    Score per symbol = (price_now - price_lookback_bars_ago) / (ATR *
    sqrt(lookback_bars)) - a Sharpe-like measure of trend strength relative
    to that symbol's own typical volatility. Normalizing by ATR matters: a
    micro-cap that moved 8% and a major that moved 1% aren't comparable on
    raw return alone, since the micro-cap might just be more volatile in
    general, not stronger relative to its own noise. Ranking on raw return
    would systematically bias every pick toward whatever is most volatile.

    The exit policy genuinely differs from PerSymbolStrategy: a position's
    reason for existing is "currently ranked in the top/bottom K," so it
    should close as soon as that stops being true - including merely fading
    to FLAT, not only flipping to the opposite side.
    """

    def __init__(self, strategy_cfg):
        self.cfg = strategy_cfg

    def _score(self, df: pd.DataFrame, idx: int) -> Optional[float]:
        lookback = self.cfg.momentum_lookback_bars
        if idx < lookback:
            return None
        row = df.iloc[idx]
        if pd.isna(row[["close", "atr"]]).any():
            return None
        atr = row["atr"]
        if not atr or atr <= 0:
            return None
        price_then = df["close"].iloc[idx - lookback]
        if pd.isna(price_then):
            return None
        return (row["close"] - price_then) / (atr * (lookback ** 0.5))

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        raise NotImplementedError(
            "CrossSectionalMomentumStrategy needs the whole universe at once - "
            "call generate_signals(dfs, i), not generate_signal(df)."
        )

    def generate_signals(self, dfs: Dict[str, pd.DataFrame], i: Optional[int] = None) -> Dict[str, Signal]:
        scores: Dict[str, float] = {}
        for sym, df in dfs.items():
            idx = i if i is not None else len(df) - 1
            score = self._score(df, idx)
            if score is not None:
                scores[sym] = score

        signals: Dict[str, Signal] = {sym: Signal("FLAT", 0.0, "cross_sectional_momentum") for sym in dfs}
        if not scores:
            return signals

        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        top = [sym for sym, score in ranked[: self.cfg.top_k] if score >= self.cfg.min_abs_momentum_score]
        bottom = []
        if self.cfg.bottom_k > 0:
            bottom = [sym for sym, score in ranked[-self.cfg.bottom_k:] if score <= -self.cfg.min_abs_momentum_score]
            bottom = [sym for sym in bottom if sym not in top]  # guard tiny universes where slices overlap

        for sym in top:
            confidence = min(0.9, 0.4 + abs(scores[sym]) * 0.1)
            signals[sym] = Signal(
                "LONG", confidence, "cross_sectional_momentum",
                [f"momentum score {scores[sym]:+.2f}, ranked in top {self.cfg.top_k}"],
            )
        for sym in bottom:
            confidence = min(0.9, 0.4 + abs(scores[sym]) * 0.1)
            signals[sym] = Signal(
                "SHORT", confidence, "cross_sectional_momentum",
                [f"momentum score {scores[sym]:+.2f}, ranked in bottom {self.cfg.bottom_k}"],
            )
        return signals

    def should_exit_on_signal(self, position_side: Side, signal: Signal) -> bool:
        return signal.side != position_side


def build_strategy(cfg):
    mode = cfg.strategy.mode
    if mode == "ensemble":
        return EnsembleStrategy(cfg.indicators)
    if mode == "mean_reversion_scalp":
        return MeanReversionScalpStrategy(cfg.indicators)
    if mode == "cross_sectional_momentum":
        return CrossSectionalMomentumStrategy(cfg.strategy)
    raise ValueError(f"unknown strategy mode: {mode!r}")
