"""Binance REST feed: spot price, closed 1-minute klines, hourly open (strike)."""

import logging
import time
from typing import List, Optional

import aiohttp

log = logging.getLogger(__name__)
URL = "https://api.binance.com/api/v3"
SYMBOL = "BTCUSDT"


class BinanceFeed:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.price: float = 0.0
        self.price_ts: float = 0.0
        self.last_closed_open_ms: int = 0     # open time of the last closed kline already consumed

    async def _get(self, path: str, params: dict):
        async with self.session.get(f"{URL}{path}", params=params,
                                    timeout=aiohttp.ClientTimeout(total=10)) as r:
            r.raise_for_status()
            return await r.json()

    async def poll_price(self) -> float:
        data = await self._get("/ticker/price", {"symbol": SYMBOL})
        self.price, self.price_ts = float(data["price"]), time.time()
        return self.price

    async def history(self, n: int) -> List[list]:
        """Last n *closed* 1-minute klines, oldest first. Paginated (Binance caps a call at 1000)."""
        rows, end_ms = [], int(time.time() * 1000)
        while len(rows) < n + 1:
            batch = await self._get("/klines", {"symbol": SYMBOL, "interval": "1m",
                                                "limit": 1000, "endTime": end_ms})
            if not batch:
                break
            rows = batch + rows
            end_ms = batch[0][0] - 1
        now_ms = time.time() * 1000
        closed = [k for k in rows if k[6] < now_ms][-n:]   # k[6] = close time
        if closed:
            self.last_closed_open_ms = closed[-1][0]
        return closed

    async def new_closed_klines(self) -> List[list]:
        """Closed klines not yet consumed. Called every minute; returns usually one kline."""
        rows = await self._get("/klines", {"symbol": SYMBOL, "interval": "1m", "limit": 5})
        now_ms = time.time() * 1000
        fresh = [k for k in rows if k[6] < now_ms and k[0] > self.last_closed_open_ms]
        if fresh:
            self.last_closed_open_ms = fresh[-1][0]
        return fresh

    async def hourly_open(self, hour_start_ts: float) -> Optional[float]:
        """Open of the 1h candle starting at hour_start_ts: the market's strike."""
        rows = await self._get("/klines", {"symbol": SYMBOL, "interval": "1h",
                                           "startTime": int(hour_start_ts * 1000), "limit": 1})
        if rows and abs(rows[0][0] / 1000 - hour_start_ts) < 1:
            return float(rows[0][1])
        return None

    def is_fresh(self, max_age: float) -> bool:
        return self.price > 0 and time.time() - self.price_ts < max_age
