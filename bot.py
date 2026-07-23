"""Live/paper trading loop.

Every iteration: refresh candles for the tradable symbol set, manage exits on
open positions (stop-loss / take-profit / trailing stop / the strategy's own
signal-exit policy), update drawdown tracking, and - unless the drawdown
circuit breaker has tripped - look for new entries. Paper mode uses real
market data with simulated fills; live mode places real orders through
LiveBroker.

Two things only apply in "cross_sectional_momentum" mode:
- the tradable symbol set is refreshed periodically from Bitget's actual
  liquid perpetual board (see universe.py) instead of a fixed list.
- in live mode, equity is synced from the real exchange balance every cycle
  rather than tracked purely as an internal ledger, so a manual deposit (the
  plan being to top up monthly) is picked up automatically instead of being
  invisible to position sizing until someone edits the state file by hand.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional

import pandas as pd

import indicators
from config import Config
from exchange import LiveBroker, MarketDataFeed, PaperBroker, clamp_qty_to_market
from models import Position, utcnow
from portfolio import PortfolioManager
from risk_manager import RiskManager
from strategies import Signal, build_strategy

logger = logging.getLogger(__name__)

Mode = Literal["paper", "live"]


class QuantTradingBot:
    def __init__(self, cfg: Config, mode: Mode):
        self.cfg = cfg
        self.mode = mode
        self.market_data = MarketDataFeed(cfg)
        self.broker = LiveBroker(cfg, self.market_data) if mode == "live" else PaperBroker(cfg, self.market_data)
        self.strategy = build_strategy(cfg)
        self.risk = RiskManager(cfg.risk)

        self.symbols: List[str] = list(cfg.exchange.symbols)
        self._last_universe_refresh: Optional[float] = None

        state_dir = Path(cfg.runtime.state_dir) / mode
        self.portfolio = PortfolioManager.load(
            starting_equity=cfg.risk.starting_equity,
            state_file=state_dir / "state.json",
            trade_log_file=state_dir / "trades.jsonl",
        )
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run_forever(self) -> None:
        logger.info("starting bot mode=%s strategy=%s timeframe=%s", self.mode, self.cfg.strategy.mode, self.cfg.exchange.timeframe)
        while self._running:
            try:
                self.run_once()
            except Exception:
                logger.exception("error during trading loop iteration, will retry next cycle")
            time.sleep(self.cfg.runtime.poll_interval_seconds)

    def run_once(self) -> None:
        self._sync_equity_if_live()
        self._refresh_universe()

        dfs: Dict[str, pd.DataFrame] = {}
        for symbol in self.symbols:
            try:
                raw = self.market_data.fetch_ohlcv(symbol, self.cfg.exchange.timeframe, limit=300)
            except Exception:
                logger.exception("failed to fetch candles for %s, skipping it this cycle", symbol)
                continue
            if len(raw) < 5:
                continue
            dfs[symbol] = indicators.add_all_indicators(raw, self.cfg.indicators)

        if not dfs:
            logger.warning("no market data fetched this cycle, skipping")
            return

        current_prices = {sym: df.iloc[-1]["close"] for sym, df in dfs.items()}
        signals = self.strategy.generate_signals(dfs)

        for symbol in list(self.portfolio.open_positions.keys()):
            if symbol in dfs:
                self._manage_exit(symbol, dfs[symbol], signals.get(symbol))

        self.portfolio.update_peak_equity(current_prices)

        if self.risk.is_drawdown_halted(self.portfolio.equity, self.portfolio.peak_equity):
            logger.error(
                "DRAWDOWN HALT: equity=%.2f peak=%.2f - no new entries until manually reset",
                self.portfolio.equity, self.portfolio.peak_equity,
            )
        else:
            for symbol, df in dfs.items():
                if symbol in self.portfolio.open_positions:
                    continue
                signal = signals.get(symbol)
                if signal:
                    self._maybe_enter(symbol, df, signal)

        logger.info(
            "cycle done | equity=%.2f peak=%.2f daily_pnl=%.2f open=%d universe=%d",
            self.portfolio.equity, self.portfolio.peak_equity, self.portfolio.daily_pnl,
            len(self.portfolio.open_positions), len(self.symbols),
        )

    # ------------------------------------------------------------------

    def _sync_equity_if_live(self) -> None:
        """Source equity from the real exchange balance in live mode, not
        just an internal running ledger of trade P&L. This is what makes a
        manual deposit show up as more buying power on the next cycle
        instead of requiring a state-file edit; the tradeoff is that a
        mid-day deposit also shows up as that day's "P&L" in the daily-loss
        window, a cosmetic inaccuracy that's a reasonable price for not
        silently trading on a stale balance."""
        if self.mode != "live":
            return
        try:
            live_equity = self.broker.get_account_equity()
        except Exception:
            logger.exception("failed to fetch live account equity this cycle, keeping last known value")
            return
        if live_equity <= 0:
            return
        if abs(live_equity - self.portfolio.equity) > 0.01:
            logger.info("syncing equity from exchange balance: %.2f -> %.2f", self.portfolio.equity, live_equity)
        self.portfolio.equity = live_equity
        self.portfolio.peak_equity = max(self.portfolio.peak_equity, live_equity)
        self.portfolio.persist()

    def _refresh_universe(self) -> None:
        if self.cfg.strategy.mode != "cross_sectional_momentum":
            return

        now = time.monotonic()
        refresh_interval = self.cfg.strategy.universe_refresh_hours * 3600
        if self._last_universe_refresh is not None and (now - self._last_universe_refresh) < refresh_interval:
            return

        try:
            selected = self.market_data.fetch_liquid_universe(
                top_n=self.cfg.strategy.universe_top_n,
                min_quote_volume_24h=self.cfg.strategy.universe_min_quote_volume_24h,
            )
        except Exception:
            logger.exception("failed to refresh trading universe, keeping previous selection (%d symbols)", len(self.symbols))
            self._last_universe_refresh = now
            return

        if selected:
            logger.info("universe refreshed: %d symbols: %s", len(selected), selected)
            self.symbols = selected
        self._last_universe_refresh = now

    def _manage_exit(self, symbol: str, df: pd.DataFrame, signal: Optional[Signal]) -> None:
        position = self.portfolio.open_positions[symbol]
        bar = df.iloc[-1]

        hit_reason = None
        if position.side == "LONG":
            if bar["low"] <= position.stop_price:
                hit_reason = "stop_loss"
            elif bar["high"] >= position.take_profit_price:
                hit_reason = "take_profit"
        else:
            if bar["high"] >= position.stop_price:
                hit_reason = "stop_loss"
            elif bar["low"] <= position.take_profit_price:
                hit_reason = "take_profit"

        if hit_reason is None and signal is not None and self.strategy.should_exit_on_signal(position.side, signal):
            hit_reason = "signal_exit"

        if hit_reason:
            side = "sell" if position.side == "LONG" else "buy"
            fill = self.broker.execute_market_order(symbol, side, position.qty, reduce_only=True)
            trade = self.portfolio.close_position(symbol, fill.price, fill.fee, hit_reason)
            logger.info(
                "CLOSED %s %s qty=%.6f price=%.4f pnl=%.2f reason=%s R=%.2f",
                symbol, position.side, position.qty, fill.price, trade.pnl, hit_reason, trade.r_multiple,
            )
            return

        new_stop = self.risk.update_trailing_stop(position, bar["close"], bar["atr"])
        if new_stop is not None:
            position.stop_price = new_stop
            position.trailing_active = True
            self.portfolio.persist()
            logger.info("trailing stop moved for %s -> %.4f", symbol, new_stop)

    def _maybe_enter(self, symbol: str, df: pd.DataFrame, signal: Signal) -> None:
        row = df.iloc[-1]
        reference_price = self.broker.get_price(symbol)
        stop, _ = self.risk.compute_stop_and_take(signal.side, reference_price, row["atr"])
        qty = self.risk.position_size(self.portfolio.equity, reference_price, stop)
        if qty <= 0:
            return

        try:
            limits = self.market_data.get_market_limits(symbol)
            qty = clamp_qty_to_market(qty, reference_price, limits)
        except Exception:
            logger.exception("could not fetch market limits for %s, skipping entry this cycle", symbol)
            return
        if qty <= 0:
            logger.info("skip entry %s %s: risk-sized quantity is below the exchange's minimum order size", symbol, signal.side)
            return

        proposed_risk = abs(reference_price - stop) * qty
        decision = self.risk.can_open_new_trade(
            symbol, self.portfolio.equity, self.portfolio.peak_equity, self.portfolio.daily_pnl,
            self.portfolio.daily_start_equity, self.portfolio.open_positions, proposed_risk,
        )
        if not decision.allowed:
            logger.info("skip entry %s %s: %s", symbol, signal.side, decision.reason)
            return

        side = "buy" if signal.side == "LONG" else "sell"
        fill = self.broker.execute_market_order(symbol, side, qty)

        # recompute stop/take against the actual fill price, not the pre-trade estimate
        stop, take = self.risk.compute_stop_and_take(signal.side, fill.price, row["atr"])
        position = Position(
            symbol=symbol, side=signal.side, entry_price=fill.price, qty=fill.qty,
            stop_price=stop, take_profit_price=take,
            initial_risk_per_unit=abs(fill.price - stop), opened_at=utcnow(),
        )
        self.portfolio.equity -= fill.fee
        self.portfolio.open_position(position)
        logger.info(
            "OPENED %s %s qty=%.6f entry=%.4f stop=%.4f take=%.4f confidence=%.2f reasons=%s",
            symbol, signal.side, position.qty, position.entry_price, stop, take, signal.confidence, signal.reasons,
        )
