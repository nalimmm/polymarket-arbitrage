"""Market data containers shared by every module."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import time


@dataclass
class Level:
    price: float
    size: float          # number of tokens


@dataclass
class OrderBook:
    bids: List[Level]    # sorted best (highest) first
    asks: List[Level]    # sorted best (lowest) first
    timestamp: float = field(default_factory=time.time)

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> Optional[float]:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2

    def ask_depth_usd(self, tolerance: float) -> float:
        """USD available on the ask side within `tolerance` of the best ask."""
        if not self.asks:
            return 0.0
        limit = self.asks[0].price + tolerance
        return sum(l.price * l.size for l in self.asks if l.price <= limit + 1e-12)

    def buy_fill(self, usd: float) -> Optional[Tuple[float, float]]:
        """Walk the asks for `usd` of notional. Returns (tokens, average price) or None."""
        remaining, tokens = usd, 0.0
        for l in self.asks:
            take = min(remaining, l.price * l.size)
            tokens += take / l.price
            remaining -= take
            if remaining <= 1e-9:
                return tokens, usd / tokens
        return None

    def sell_fill(self, tokens: float) -> Tuple[float, float]:
        """Walk the bids for `tokens`. Returns (proceeds, average price). Unfilled part is valued at 0."""
        remaining, proceeds = tokens, 0.0
        for l in self.bids:
            take = min(remaining, l.size)
            proceeds += take * l.price
            remaining -= take
            if remaining <= 1e-9:
                break
        filled = tokens - max(remaining, 0.0)
        return proceeds, (proceeds / filled if filled > 0 else 0.0)


@dataclass
class Market:
    """One hourly 'Bitcoin Up or Down' market."""
    slug: str
    question: str
    token_yes: str
    token_no: str
    expiry_ts: float                 # unix seconds, end of the hourly candle
    strike: Optional[float] = None   # Binance 1h candle open ("price to beat")
    book_yes: Optional[OrderBook] = None
    book_no: Optional[OrderBook] = None

    def minutes_to_expiry(self, now: Optional[float] = None) -> float:
        return (self.expiry_ts - (now or time.time())) / 60.0

    def book(self, side: str) -> Optional[OrderBook]:
        return self.book_yes if side == "YES" else self.book_no
