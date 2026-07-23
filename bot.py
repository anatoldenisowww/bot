"""Live/paper trading loop.

Every iteration: refresh candles for all symbols, manage exits on open
positions (stop-loss / take-profit / trailing stop / opposing signal), update
drawdown tracking, and - unless the drawdown circuit breaker has tripped -
look for new entries on symbols that are flat. Paper mode uses real market
data with simulated fills; live mode places real orders through LiveBroker.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Literal

import pandas as pd

import indicators
from config import Config
from exchange import LiveBroker, MarketDataFeed, PaperBroker
from models import Position, utcnow
from portfolio import PortfolioManager
from risk_manager import RiskManager
from strategies import build_strategy

logger = logging.getLogger(__name__)

Mode = Literal["paper", "live"]


class QuantTradingBot:
    def __init__(self, cfg: Config, mode: Mode):
        self.cfg = cfg
        self.mode = mode
        self.market_data = MarketDataFeed(cfg)
        self.broker = LiveBroker(cfg, self.market_data) if mode == "live" else PaperBroker(cfg, self.market_data)
        self.strategy = build_strategy(cfg.strategy.mode, cfg.indicators)
        self.risk = RiskManager(cfg.risk)

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
        logger.info("starting bot mode=%s symbols=%s timeframe=%s", self.mode, self.cfg.exchange.symbols, self.cfg.exchange.timeframe)
        while self._running:
            try:
                self.run_once()
            except Exception:
                logger.exception("error during trading loop iteration, will retry next cycle")
            time.sleep(self.cfg.runtime.poll_interval_seconds)

    def run_once(self) -> None:
        dfs: Dict[str, pd.DataFrame] = {}
        for symbol in self.cfg.exchange.symbols:
            raw = self.market_data.fetch_ohlcv(symbol, self.cfg.exchange.timeframe, limit=300)
            dfs[symbol] = indicators.add_all_indicators(raw, self.cfg.indicators)

        current_prices = {sym: df.iloc[-1]["close"] for sym, df in dfs.items()}

        for symbol in list(self.portfolio.open_positions.keys()):
            self._manage_exit(symbol, dfs[symbol])

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
                self._maybe_enter(symbol, df)

        logger.info(
            "cycle done | equity=%.2f peak=%.2f daily_pnl=%.2f open=%d",
            self.portfolio.equity, self.portfolio.peak_equity, self.portfolio.daily_pnl, len(self.portfolio.open_positions),
        )

    # ------------------------------------------------------------------

    def _manage_exit(self, symbol: str, df: pd.DataFrame) -> None:
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

        if hit_reason is None:
            signal = self.strategy.generate_signal(df)
            opposite = "SHORT" if position.side == "LONG" else "LONG"
            if signal and signal.side == opposite:
                hit_reason = "signal_flip"

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

    def _maybe_enter(self, symbol: str, df: pd.DataFrame) -> None:
        signal = self.strategy.generate_signal(df)
        if not signal:
            return

        row = df.iloc[-1]
        reference_price = self.broker.get_price(symbol)
        stop, _ = self.risk.compute_stop_and_take(signal.side, reference_price, row["atr"])
        qty = self.risk.position_size(self.portfolio.equity, reference_price, stop)
        if qty <= 0:
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
