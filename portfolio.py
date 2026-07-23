"""Tracks equity, open positions, drawdown and the daily-loss window.

Used by both the paper/live bot and the backtester so the same drawdown and
daily-loss accounting logic governs simulated and real money alike.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from models import ClosedTrade, Position, utcnow


class PortfolioManager:
    def __init__(self, starting_equity: float, state_file: Optional[Path] = None, trade_log_file: Optional[Path] = None):
        self.equity = starting_equity
        self.peak_equity = starting_equity
        self.open_positions: Dict[str, Position] = {}
        self.closed_trades: List[ClosedTrade] = []

        self._daily_start_equity = starting_equity
        self._daily_start_date = utcnow().date()

        self.state_file = state_file
        self.trade_log_file = trade_log_file

    # ---- daily window ---------------------------------------------------

    def _roll_daily_window_if_needed(self) -> None:
        today = utcnow().date()
        if today != self._daily_start_date:
            self._daily_start_date = today
            self._daily_start_equity = self.equity

    @property
    def daily_pnl(self) -> float:
        self._roll_daily_window_if_needed()
        return self.equity - self._daily_start_equity

    @property
    def daily_start_equity(self) -> float:
        self._roll_daily_window_if_needed()
        return self._daily_start_equity

    # ---- position lifecycle ----------------------------------------------

    def open_position(self, position: Position) -> None:
        self.open_positions[position.symbol] = position
        self._persist()

    def persist(self) -> None:
        """Public hook to flush state after mutating an open Position in place
        (e.g. a trailing stop update) without opening/closing a position."""
        self._persist()

    def close_position(self, symbol: str, exit_price: float, fees: float, reason: str, closed_at: Optional[datetime] = None) -> ClosedTrade:
        position = self.open_positions.pop(symbol)
        pnl = position.unrealized_pnl(exit_price) - fees

        trade = ClosedTrade(
            symbol=symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            qty=position.qty,
            pnl=pnl,
            fees=fees,
            initial_risk_per_unit=position.initial_risk_per_unit,
            opened_at=position.opened_at,
            closed_at=closed_at or utcnow(),
            reason=reason,
        )

        self.equity += pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        self.closed_trades.append(trade)
        self._append_trade_log(trade)
        self._persist()
        return trade

    def mark_to_market_equity(self, prices: Dict[str, float]) -> float:
        """Equity including unrealized PnL of open positions, for reporting/drawdown checks."""
        unrealized = sum(
            pos.unrealized_pnl(prices[sym]) for sym, pos in self.open_positions.items() if sym in prices
        )
        return self.equity + unrealized

    def update_peak_equity(self, prices: Dict[str, float]) -> None:
        mtm = self.mark_to_market_equity(prices)
        self.peak_equity = max(self.peak_equity, mtm)

    # ---- persistence ----------------------------------------------------

    def _persist(self) -> None:
        if not self.state_file:
            return
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "daily_start_equity": self._daily_start_equity,
            "daily_start_date": self._daily_start_date.isoformat(),
            "open_positions": {
                sym: {**asdict(pos), "opened_at": pos.opened_at.isoformat()}
                for sym, pos in self.open_positions.items()
            },
        }
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(self.state_file)

    def _append_trade_log(self, trade: ClosedTrade) -> None:
        if not self.trade_log_file:
            return
        self.trade_log_file.parent.mkdir(parents=True, exist_ok=True)
        record = asdict(trade)
        record["opened_at"] = trade.opened_at.isoformat()
        record["closed_at"] = trade.closed_at.isoformat()
        record["r_multiple"] = trade.r_multiple
        with open(self.trade_log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    @classmethod
    def load(cls, starting_equity: float, state_file: Path, trade_log_file: Optional[Path] = None) -> "PortfolioManager":
        pm = cls(starting_equity=starting_equity, state_file=state_file, trade_log_file=trade_log_file)
        if state_file.exists():
            state = json.loads(state_file.read_text())
            pm.equity = state["equity"]
            pm.peak_equity = state["peak_equity"]
            pm._daily_start_equity = state["daily_start_equity"]
            pm._daily_start_date = datetime.fromisoformat(state["daily_start_date"]).date()
            for sym, pdata in state.get("open_positions", {}).items():
                pdata = dict(pdata)
                pdata["opened_at"] = datetime.fromisoformat(pdata["opened_at"])
                pm.open_positions[sym] = Position(**pdata)
        return pm
