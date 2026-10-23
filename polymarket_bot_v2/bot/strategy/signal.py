"""
Signal: buy a token when the model's PRUDENT probability exceeds its ASK by at least min_edge.

    edge(side) = prudent_probability(side) - best_ask(side)

- Measured against the ask (the price we actually pay), not the mid.
- Prudent probability = least favorable value across sigma x (1 +/- 20%): an edge that
  only exists under one volatility assumption is not traded. This replaces the old z-score,
  which compared the mispricing to its own recent mean and ignored persistent edges.
- Both sides are evaluated and only the best one is kept (YES and NO are mirror bets).
"""

import time
from dataclasses import dataclass
from typing import Optional

from ..config import SignalCfg
from ..data.types import Market
from ..model.pricing import Quote


@dataclass(frozen=True)
class Opportunity:
    side: str            # "YES" or "NO"
    ask: float
    theo: float          # central model probability
    prudent: float       # prudent model probability
    edge: float          # prudent - ask
    minutes_left: float


class SignalGenerator:
    def __init__(self, cfg: SignalCfg):
        self.cfg = cfg
        self.last_signal_ts = 0.0

    def evaluate(self, market: Market, q: Quote, now: Optional[float] = None) -> Optional[Opportunity]:
        c, now = self.cfg, now or time.time()
        minutes_left = market.minutes_to_expiry(now)
        if minutes_left <= c.no_entry_minutes_before_expiry:
            return None
        if now - self.last_signal_ts < c.cooldown_seconds:
            return None

        best = None
        for side in ("YES", "NO"):
            book = market.book(side)
            if book is None or book.best_ask is None:
                continue
            ask = book.best_ask
            if not c.min_token_price <= ask <= c.max_token_price:
                continue
            edge = q.prudent(side) - ask
            if c.min_edge <= edge <= c.max_edge and (best is None or edge > best.edge):
                best = Opportunity(side, ask, q.theo(side), q.prudent(side), edge, minutes_left)

        if best:
            self.last_signal_ts = now
        return best
