"""Shared data types used across risk, portfolio, exchange, backtester and bot."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Optional

Side = Literal["LONG", "SHORT"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Position:
    symbol: str
    side: Side
    entry_price: float
    qty: float
    stop_price: float
    take_profit_price: float
    initial_risk_per_unit: float  # |entry_price - stop_price| at open, used for R multiples
    opened_at: datetime
    trailing_active: bool = False

    def risk_amount(self) -> float:
        return self.initial_risk_per_unit * self.qty

    def unrealized_pnl(self, current_price: float) -> float:
        if self.side == "LONG":
            return (current_price - self.entry_price) * self.qty
        return (self.entry_price - current_price) * self.qty

    def r_multiple(self, current_price: float) -> float:
        risk = self.risk_amount()
        if risk <= 0:
            return 0.0
        return self.unrealized_pnl(current_price) / risk


@dataclass
class ClosedTrade:
    symbol: str
    side: Side
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    fees: float
    initial_risk_per_unit: float
    opened_at: datetime
    closed_at: datetime
    reason: str  # "stop_loss" | "take_profit" | "signal_flip" | "manual" | "trailing_stop"

    @property
    def r_multiple(self) -> float:
        risk = self.initial_risk_per_unit * self.qty
        if risk <= 0:
            return 0.0
        return self.pnl / risk
