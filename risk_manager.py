"""Position sizing and the circuit breakers that keep a bad day from becoming
a blown account. This is the part of a "professional" bot that actually
matters more than the entry signal - it is deliberately conservative.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from config import RiskConfig
from models import Position, Side


@dataclass
class RiskDecision:
    allowed: bool
    reason: str


class RiskManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg

    # ---- sizing -----------------------------------------------------

    def compute_stop_and_take(self, side: Side, entry_price: float, atr: float) -> Tuple[float, float]:
        stop_distance = self.cfg.atr_stop_multiplier * atr
        if side == "LONG":
            stop = entry_price - stop_distance
            take = entry_price + stop_distance * self.cfg.take_profit_r_multiple
        else:
            stop = entry_price + stop_distance
            take = entry_price - stop_distance * self.cfg.take_profit_r_multiple
        return stop, take

    def position_size(self, equity: float, entry_price: float, stop_price: float) -> float:
        """Qty such that a stop-out loses exactly risk_per_trade_pct of equity."""
        risk_dollars = equity * (self.cfg.risk_per_trade_pct / 100.0)
        risk_per_unit = abs(entry_price - stop_price)
        if risk_per_unit <= 0:
            return 0.0
        qty = risk_dollars / risk_per_unit

        # Cap by max leverage: notional can't exceed equity * max_leverage.
        max_notional = equity * self.cfg.max_leverage
        max_qty_by_leverage = max_notional / entry_price
        return min(qty, max_qty_by_leverage)

    def update_trailing_stop(self, position: Position, current_price: float, atr: float) -> Optional[float]:
        """Return a new stop price if the trailing stop should move, else None."""
        r = position.r_multiple(current_price)
        if r < self.cfg.trailing_activation_r:
            return None

        trail_distance = self.cfg.trailing_atr_multiplier * atr
        if position.side == "LONG":
            new_stop = current_price - trail_distance
            if new_stop > position.stop_price:
                return new_stop
        else:
            new_stop = current_price + trail_distance
            if new_stop < position.stop_price:
                return new_stop
        return None

    # ---- gating -------------------------------------------------------

    def can_open_new_trade(
        self,
        symbol: str,
        equity: float,
        peak_equity: float,
        daily_pnl: float,
        starting_daily_equity: float,
        open_positions: Dict[str, Position],
        proposed_risk_dollars: float,
    ) -> RiskDecision:
        if symbol in open_positions:
            return RiskDecision(False, f"already have an open position on {symbol}")

        if len(open_positions) >= self.cfg.max_concurrent_positions:
            return RiskDecision(False, "max_concurrent_positions reached")

        if peak_equity > 0:
            drawdown_pct = (peak_equity - equity) / peak_equity * 100.0
            if drawdown_pct >= self.cfg.max_drawdown_pct:
                return RiskDecision(False, f"max_drawdown_pct breached ({drawdown_pct:.2f}%) - bot halted, needs manual reset")

        if starting_daily_equity > 0:
            daily_loss_pct = -daily_pnl / starting_daily_equity * 100.0
            if daily_loss_pct >= self.cfg.max_daily_loss_pct:
                return RiskDecision(False, f"max_daily_loss_pct breached ({daily_loss_pct:.2f}%) - no new trades until next UTC day")

        current_open_risk = sum(p.risk_amount() for p in open_positions.values())
        if equity > 0:
            portfolio_risk_pct = (current_open_risk + proposed_risk_dollars) / equity * 100.0
            if portfolio_risk_pct > self.cfg.max_portfolio_risk_pct:
                return RiskDecision(False, f"max_portfolio_risk_pct would be exceeded ({portfolio_risk_pct:.2f}%)")

        return RiskDecision(True, "ok")

    def is_drawdown_halted(self, equity: float, peak_equity: float) -> bool:
        if peak_equity <= 0:
            return False
        drawdown_pct = (peak_equity - equity) / peak_equity * 100.0
        return drawdown_pct >= self.cfg.max_drawdown_pct
