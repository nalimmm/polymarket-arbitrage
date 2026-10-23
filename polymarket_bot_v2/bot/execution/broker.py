"""
Paper execution with realistic frictions.

A decision is queued and filled `latency_seconds` later against the order book AT FILL TIME:
  - the edge is re-checked on the new ask (cancelled if it no longer clears min_edge),
  - size is recomputed with the latest equity, exposure and depth,
  - the fill walks the asks (VWAP), so slippage beyond the best level is paid.
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

from ..data.types import Market
from ..model.pricing import Quote
from ..strategy.signal import Opportunity
from ..strategy.sizing import Sizer
from .portfolio import Portfolio

log = logging.getLogger(__name__)


@dataclass
class PendingOrder:
    opp: Opportunity
    market: str
    fill_at: float


class PaperBroker:
    def __init__(self, portfolio: Portfolio, sizer: Sizer, latency: float, min_edge: float, depth_tol: float):
        self.pf, self.sizer = portfolio, sizer
        self.latency, self.min_edge, self.depth_tol = latency, min_edge, depth_tol
        self.pending: List[PendingOrder] = []

    def submit(self, opp: Opportunity, market: Market, now: Optional[float] = None) -> None:
        self.pending.append(PendingOrder(opp, market.slug, (now or time.time()) + self.latency))

    def process(self, market: Market, q: Quote, now: Optional[float] = None) -> None:
        now = now or time.time()
        due = [o for o in self.pending if o.fill_at <= now]
        self.pending = [o for o in self.pending if o.fill_at > now]
        for o in due:
            self._fill(o, market, q, now)

    def _fill(self, o: PendingOrder, market: Market, q: Quote, now: float) -> None:
        side = o.opp.side
        book = market.book(side)
        if o.market != market.slug or book is None or book.best_ask is None:
            return log.info("cancel %s: market changed or empty book", side)
        if self.pf.holds_opposite(market.slug, side):
            return log.info("cancel %s: opposite side already held", side)

        ask = book.best_ask
        edge = q.prudent(side) - ask
        if edge < self.min_edge:
            return log.info("cancel %s: edge %.3f at fill time", side, edge)

        d = self.sizer.size(q_prudent=q.prudent(side), ask=ask, equity=self.pf.equity,
                            outcome_exposure=self.pf.outcome_exposure(market.slug, side),
                            total_exposure=self.pf.total_exposure,
                            depth_usd=book.ask_depth_usd(self.depth_tol))
        if d.usd <= 0:
            return log.info("skip %s: size 0 (%s)", side, d.binding)

        fill = book.buy_fill(d.usd)
        if fill is None:
            return log.info("cancel %s: not enough liquidity", side)
        tokens, avg_price = fill
        self.pf.open_position(market.slug, side, tokens, d.usd, avg_price, q.theo(side), edge, now)
        log.info("FILL %s %.2f$ @ %.3f | edge %.3f | kelly %.3f | binding %s",
                 side, d.usd, avg_price, edge, d.kelly_full, d.binding)
