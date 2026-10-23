"""
Historical data for backtests.

Historical mode
  - Binance 1-minute klines (complete history, paginated by 1000).
  - Polymarket hourly markets: token ids and resolution from the Gamma API, minute-by-minute
    token prices from the CLOB `prices-history` endpoint. No order book history exists, so the
    simulator wraps each price in an assumed spread and depth (see config: backtest).

Replay mode
  - The bot's own `signals.csv`: real best bid/ask and sizes observed tick by tick.

Everything downloaded is cached as CSV in data_dir: re-running a backtest needs no network.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import aiohttp
import pandas as pd

from bot.data.polymarket import CLOB, GAMMA, hourly_slug

log = logging.getLogger(__name__)
BINANCE = "https://api.binance.com/api/v3/klines"
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume"]


# ============================================================================ download
async def _get(session, url, params, retries=3):
    for i in range(retries):
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as r:
                if r.status == 429:
                    await asyncio.sleep(2 ** i)
                    continue
                r.raise_for_status()
                return await r.json()
        except Exception as e:
            if i == retries - 1:
                log.warning("GET %s %s failed: %s", url, params, e)
                return None
            await asyncio.sleep(1 + i)


async def download_klines(session, start: datetime, end: datetime) -> pd.DataFrame:
    rows, cursor, end_ms = [], int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    while cursor < end_ms:
        batch = await _get(session, BINANCE, {"symbol": "BTCUSDT", "interval": "1m",
                                              "startTime": cursor, "endTime": end_ms, "limit": 1000})
        if not batch:
            break
        rows += [k[:6] for k in batch]
        cursor = batch[-1][0] + 60_000
    df = pd.DataFrame(rows, columns=KLINE_COLS).drop_duplicates("open_time")
    df[KLINE_COLS[1:]] = df[KLINE_COLS[1:]].astype(float)
    return df


def _outcome_up(market: dict) -> Optional[bool]:
    prices = market.get("outcomePrices")
    prices = json.loads(prices) if isinstance(prices, str) else prices
    if not prices or not market.get("closed"):
        return None
    return float(prices[0]) > 0.5               # outcomes are ["Up", "Down"]


async def _one_market(session, sem, hour: datetime):
    async with sem:
        slug = hourly_slug(hour)
        events = await _get(session, f"{GAMMA}/events", {"slug": slug})
        if not events:
            return None, []
        m = events[0]["markets"][0]
        ids = m.get("clobTokenIds")
        ids = json.loads(ids) if isinstance(ids, str) else ids
        meta = {"slug": slug, "hour_start": int(hour.timestamp()), "token_yes": ids[0], "token_no": ids[1],
                "outcome_up_polymarket": _outcome_up(m)}
        prices = []
        for side, tok in (("YES", ids[0]), ("NO", ids[1])):
            h = await _get(session, f"{CLOB}/prices-history",
                           {"market": tok, "startTs": int(hour.timestamp()),
                            "endTs": int(hour.timestamp()) + 3600, "fidelity": 1})
            for pt in (h or {}).get("history", []):
                prices.append({"slug": slug, "side": side, "t": int(pt["t"]), "p": float(pt["p"])})
        return meta, prices


async def download(start: datetime, end: datetime, data_dir: str, calibration_minutes: int,
                   concurrency: int = 8) -> None:
    out = Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession() as s:
        log.info("Binance klines %s -> %s", start - timedelta(minutes=calibration_minutes), end)
        k = await download_klines(s, start - timedelta(minutes=calibration_minutes), end)
        k.to_csv(out / "klines_1m.csv", index=False)
        log.info("%d klines saved", len(k))

        hours, h = [], start.replace(minute=0, second=0, microsecond=0)
        while h < end:
            hours.append(h)
            h += timedelta(hours=1)
        sem = asyncio.Semaphore(concurrency)
        results = await asyncio.gather(*[_one_market(s, sem, h) for h in hours])

    metas = [m for m, _ in results if m]
    prices = [p for _, ps in results for p in ps]
    pd.DataFrame(metas).to_csv(out / "markets.csv", index=False)
    pd.DataFrame(prices, columns=["slug", "side", "t", "p"]).to_csv(out / "prices.csv", index=False)
    log.info("%d / %d markets found, %d price points saved", len(metas), len(hours), len(prices))


# ============================================================================ load
def load_klines(data_dir: str) -> pd.DataFrame:
    df = pd.read_csv(Path(data_dir) / "klines_1m.csv")
    df["t"] = (df["open_time"] // 1000).astype(int)
    return df.sort_values("t").reset_index(drop=True)


def load_markets(data_dir: str) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "markets.csv").sort_values("hour_start").reset_index(drop=True)


def load_prices(data_dir: str) -> Dict[str, Dict[str, tuple]]:
    """slug -> {"YES": (timestamps, prices), "NO": (...)} sorted by time, for as-of lookups."""
    df = pd.read_csv(Path(data_dir) / "prices.csv").sort_values("t")
    out: Dict[str, Dict[str, tuple]] = {}
    for (slug, side), g in df.groupby(["slug", "side"]):
        out.setdefault(slug, {})[side] = (g["t"].to_numpy(), g["p"].to_numpy())
    return out


def load_replay_log(path: str) -> pd.DataFrame:
    return pd.read_csv(path).sort_values("ts").reset_index(drop=True)


def parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
