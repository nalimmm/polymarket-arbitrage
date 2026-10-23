"""Polymarket feed: hourly market discovery (Gamma API) and order books (CLOB)."""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import aiohttp

from .types import Level, Market, OrderBook

log = logging.getLogger(__name__)
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
NY = ZoneInfo("America/New_York")          # handles EST/EDT switches


def hourly_slug(hour_start_utc: datetime) -> str:
    """bitcoin-up-or-down-october-23-10am-et for the candle starting at 10:00 New York time."""
    et = hour_start_utc.astimezone(NY)
    h = et.hour % 12 or 12
    ampm = "am" if et.hour < 12 else "pm"
    return f"bitcoin-up-or-down-{et.strftime('%B').lower()}-{et.day}-{h}{ampm}-et"


def _token_ids(market: dict) -> list:
    raw = market.get("clobTokenIds", [])
    return json.loads(raw) if isinstance(raw, str) else list(raw)


def parse_book(data: dict) -> OrderBook:
    bids = sorted((Level(float(b["price"]), float(b["size"])) for b in data.get("bids", [])),
                  key=lambda l: -l.price)
    asks = sorted((Level(float(a["price"]), float(a["size"])) for a in data.get("asks", [])),
                  key=lambda l: l.price)
    return OrderBook(bids=bids, asks=asks)


class PolymarketFeed:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.market: Optional[Market] = None

    async def _get(self, url: str, params: dict):
        async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as r:
            r.raise_for_status()
            return await r.json()

    async def discover(self, now: Optional[datetime] = None) -> Optional[Market]:
        """Find the market of the current hourly candle."""
        now = now or datetime.now(timezone.utc)
        hour_start = now.replace(minute=0, second=0, microsecond=0)
        slug = hourly_slug(hour_start)
        events = await self._get(f"{GAMMA}/events", {"slug": slug})
        if not events:
            log.warning("No event for slug %s", slug)
            return None
        event = events[0]
        m = event["markets"][0]
        yes, no = _token_ids(m)[:2]                 # outcome order: ["Up", "Down"]
        self.market = Market(
            slug=slug,
            question=m.get("question", event.get("title", "")),
            token_yes=str(yes),
            token_no=str(no),
            expiry_ts=(hour_start + timedelta(hours=1)).timestamp(),
        )
        log.info("Market %s, expiry %s UTC", slug, hour_start + timedelta(hours=1))
        return self.market

    async def refresh_books(self) -> None:
        if not self.market:
            return
        self.market.book_yes = parse_book(await self._get(f"{CLOB}/book", {"token_id": self.market.token_yes}))
        self.market.book_no = parse_book(await self._get(f"{CLOB}/book", {"token_id": self.market.token_no}))
