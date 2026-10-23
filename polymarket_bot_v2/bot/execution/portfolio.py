"""
Positions, marking and exits.

Positions are marked at the BID (liquidation value). Exits are checked every tick, in this
priority order (the first rule that fires closes the position, the reason is logged):

  1. cutoff       less than `cutoff_minutes` before expiry, forced close, no settlement risk
  2. stop_loss    position marked down by `stop_loss_pct` or more
  3. convergence  bid >= CURRENT theo - band: the market has caught up with the model
                  (profit taking) or the model has moved against us (edge gone).
                  Uses the current fair value, not the fair value at entry.

Fees: `fee_on_profit` is charged on positive realized PnL only.
"""

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..config import ExitCfg
from ..data.types import Market


@dataclass
class Position:
    id: int
    market: str
    side: str
    tokens: float
    cost: float                      # USD paid
    entry_price: float
    entry_ts: float
    theo_at_entry: float
    edge_at_entry: float
    mark: float = 0.0                # current bid
    exit_reason: Optional[str] = None
    exit_ts: Optional[float] = None
    pnl: float = 0.0

    @property
    def value(self) -> float:
        return self.tokens * self.mark


@dataclass
class Portfolio:
    cash: float
    exits: ExitCfg
    fee: float
    trades_csv: Optional[str] = None
    open: Dict[int, Position] = field(default_factory=dict)
    closed: List[Position] = field(default_factory=list)
    _next_id: int = 1

    # ---------------------------------------------------------------- state
    @property
    def equity(self) -> float:
        return self.cash + sum(p.value for p in self.open.values())

    @property
    def total_exposure(self) -> float:
        return sum(p.cost for p in self.open.values())

    def outcome_exposure(self, market: str, side: str) -> float:
        return sum(p.cost for p in self.open.values() if p.market == market and p.side == side)

    def holds_opposite(self, market: str, side: str) -> bool:
        other = "NO" if side == "YES" else "YES"
        return self.outcome_exposure(market, other) > 0

    # ---------------------------------------------------------------- lifecycle
    def open_position(self, market: str, side: str, tokens: float, cost: float, price: float,
                      theo: float, edge: float, now: Optional[float] = None) -> Position:
        p = Position(self._next_id, market, side, tokens, cost, price, now or time.time(), theo, edge, mark=price)
        self._next_id += 1
        self.cash -= cost
        self.open[p.id] = p
        return p

    def mark(self, market: Market) -> None:
        for p in self.open.values():
            if p.market == market.slug:
                book = market.book(p.side)
                if book and book.best_bid is not None:
                    p.mark = book.best_bid

    def exit_reason(self, p: Position, market: Market, theo_now: float, now: float) -> Optional[str]:
        e = self.exits
        if market.minutes_to_expiry(now) <= e.cutoff_minutes:
            return "cutoff"
        if p.value <= p.cost * (1 - e.stop_loss_pct):
            return "stop_loss"
        if p.mark >= theo_now - e.convergence_band:
            return "convergence"
        return None

    def check_exits(self, market: Market, theo_yes: float, now: Optional[float] = None) -> List[Position]:
        now = now or time.time()
        done = []
        for p in list(self.open.values()):
            if p.market != market.slug:
                continue
            theo = theo_yes if p.side == "YES" else 1 - theo_yes
            reason = self.exit_reason(p, market, theo, now)
            if reason:
                done.append(self._close(p, market, reason, now))
        return done

    def _close(self, p: Position, market: Market, reason: str, now: float) -> Position:
        proceeds, _ = market.book(p.side).sell_fill(p.tokens)
        gross = proceeds - p.cost
        p.pnl = gross - (self.fee * gross if gross > 0 else 0.0)
        self.cash += p.cost + p.pnl
        p.exit_reason, p.exit_ts = reason, now
        del self.open[p.id]
        self.closed.append(p)
        self._journal(p)
        return p

    def _journal(self, p: Position) -> None:
        if not self.trades_csv:
            return
        path = Path(self.trades_csv)
        new = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["id", "market", "side", "entry_ts", "exit_ts", "entry_price", "tokens",
                            "cost", "theo_at_entry", "edge_at_entry", "exit_reason", "pnl", "equity_after"])
            w.writerow([p.id, p.market, p.side, round(p.entry_ts, 1), round(p.exit_ts, 1),
                        round(p.entry_price, 4), round(p.tokens, 4), round(p.cost, 2),
                        round(p.theo_at_entry, 4), round(p.edge_at_entry, 4), p.exit_reason,
                        round(p.pnl, 4), round(self.equity, 2)])
