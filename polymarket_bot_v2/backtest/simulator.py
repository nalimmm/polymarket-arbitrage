"""
Backtest simulator.

The simulator does NOT reimplement the strategy. It builds the live `Engine` and replaces its
two data feeds by objects fed from history, then calls the same `Engine.tick(now)` the live bot
calls. Volatility, pricing, signal, sizing, latency, fills and exits are therefore the exact
production code.

No look-ahead:
  - at time T the volatility model has only seen klines closed at or before T,
  - the spot is the close of the kline that ended at T,
  - the Polymarket price used at T is the last price printed at or before T.

Historical mode ticks once per minute (the resolution of the price history), plus a second tick
`latency_seconds` later on the same data so that queued orders can fill.
Replay mode ticks on every logged row (every ~2 seconds), so latency is handled naturally.
"""

import dataclasses
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from bot.config import Config
from bot.data.types import Level, Market, OrderBook
from bot.engine import Engine

log = logging.getLogger(__name__)


class _SimBinance:
    def __init__(self):
        self.price = 0.0

    def is_fresh(self, max_age: float) -> bool:
        return self.price > 0


class _SimPoly:
    def __init__(self):
        self.market: Optional[Market] = None


def _book(bid: float, ask: float, bid_size: float, ask_size: float) -> OrderBook:
    bids = [Level(bid, bid_size)] if bid is not None and bid > 0 and bid_size > 0 else []
    asks = [Level(ask, ask_size)] if ask is not None and ask < 1 and ask_size > 0 else []
    return OrderBook(bids=bids, asks=asks)


class Backtester:
    def __init__(self, cfg: Config, results_dir: str):
        self.out = Path(results_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        trades_path = self.out / "trades.csv"
        trades_path.unlink(missing_ok=True)
        cfg = dataclasses.replace(cfg, logging=dataclasses.replace(
            cfg.logging, trades_csv=str(trades_path), signals_csv=""))
        self.cfg = cfg
        self.engine = Engine(cfg)
        self.engine.binance, self.engine.poly = _SimBinance(), _SimPoly()
        self.equity: List[dict] = []
        self.ticks: List[dict] = []            # one row per minute: model vs market vs outcome
        self.outcome_mismatch = 0

    # ------------------------------------------------------------------ helpers
    def _calibrate(self, klines: pd.DataFrame, before_ts: float) -> None:
        hist = klines[klines["t"] + 60 <= before_ts].tail(self.cfg.data.calibration_klines)
        self.engine.vol.calibrate(hist["close"].to_numpy())
        log.info("Calibrated on %d klines: lambda %.4f (half-life %.1f min), sigma %.1f%%",
                 len(hist), self.engine.vol.lam, self.engine.vol.half_life_minutes,
                 100 * self.engine.vol.sigma_annual)

    def _record(self, now: float, m: Market, q, mid_yes: Optional[float]) -> None:
        if q is None:
            return
        self.ticks.append({"ts": now, "slug": m.slug, "minutes_left": m.minutes_to_expiry(now),
                           "spot": self.engine.binance.price, "strike": m.strike,
                           "sigma_minute": self.engine.vol.sigma_minute,
                           "theo_yes": q.theo_yes, "prudent_yes": q.prudent_yes,
                           "prudent_no": q.prudent_no, "market_yes": mid_yes})
        self.equity.append({"ts": now, "equity": self.engine.portfolio.equity,
                            "exposure": self.engine.portfolio.total_exposure})

    def _finish_market(self, m: Optional[Market], up: Optional[bool], now: float) -> None:
        if m is None or up is None:
            return
        self.engine.portfolio.settle(m, up, now)
        for row in reversed(self.ticks):
            if row["slug"] != m.slug:
                break
            row["outcome_up"] = int(up)

    # ------------------------------------------------------------------ historical mode
    def run_historical(self, klines: pd.DataFrame, markets: pd.DataFrame,
                       prices: Dict[str, Dict[str, tuple]], start: float, end: float) -> None:
        c = self.cfg.backtest
        latency = self.cfg.execution.latency_seconds
        self._calibrate(klines, start)
        close_at = dict(zip(klines["t"] + 60, klines["close"]))      # close of kline ending at T
        open_at = dict(zip(klines["t"], klines["open"]))
        meta = {int(r.hour_start): r for r in markets.itertuples()}
        eng, m, m_up = self.engine, None, None

        for T in range(int(start), int(end), 60):
            if T not in close_at:
                continue
            eng.vol.on_closed_kline(close_at[T])
            eng.binance.price = close_at[T]

            hour = T - T % 3600
            if m is None or m.expiry_ts <= T:
                self._finish_market(m, m_up, T)
                m, m_up = None, None
                r = meta.get(hour)
                if r is not None and hour in open_at and r.slug in prices:
                    strike = open_at[hour]
                    end_close = close_at.get(hour + 3600)
                    m_up = None if end_close is None else bool(end_close >= strike)
                    pm = r.outcome_up_polymarket
                    if m_up is not None and isinstance(pm, (bool, np.bool_)) and bool(pm) != m_up:
                        self.outcome_mismatch += 1
                    m = Market(r.slug, r.slug, str(r.token_yes), str(r.token_no),
                               expiry_ts=hour + 3600, strike=strike)
                eng.poly.market = m
            if m is None:
                continue

            p_yes = _asof(prices[m.slug].get("YES"), T)
            p_no = _asof(prices[m.slug].get("NO"), T)
            if p_yes is None and p_no is None:
                continue
            p_yes = p_yes if p_yes is not None else 1 - p_no
            p_no = p_no if p_no is not None else 1 - p_yes
            hs, d = c.assumed_spread / 2, c.assumed_depth_tokens
            m.book_yes = _book(_clip(p_yes - hs), _clip(p_yes + hs), d, d)
            m.book_no = _book(_clip(p_no - hs), _clip(p_no + hs), d, d)

            q = eng.tick(T)
            eng.tick(T + latency)                                  # let queued orders fill
            self._record(T, m, q, p_yes)
        self._finish_market(m, m_up, end)

    # ------------------------------------------------------------------ replay mode
    def run_replay(self, logdf: pd.DataFrame, klines: pd.DataFrame) -> None:
        dflt = self.cfg.backtest.replay_default_depth
        self._calibrate(klines, logdf["ts"].iloc[0])
        k_t = klines["t"].to_numpy() + 60
        k_close, k_open = klines["close"].to_numpy(), dict(zip(klines["t"], klines["open"]))
        close_at = dict(zip(k_t, k_close))
        eng, m, j, last_minute = self.engine, None, int(np.searchsorted(k_t, logdf["ts"].iloc[0], "right")), -1

        for r in logdf.itertuples():
            while j < len(k_t) and k_t[j] <= r.ts:                 # feed klines closed by now
                eng.vol.on_closed_kline(k_close[j])
                j += 1
            if m is None or r.market != m.slug:
                if m is not None:
                    end_close = close_at.get(int(m.expiry_ts))
                    self._finish_market(m, None if end_close is None else bool(end_close >= m.strike), r.ts)
                expiry = round((r.ts + r.minutes_left * 60) / 3600) * 3600
                strike = r.strike if pd.notna(r.strike) else k_open.get(expiry - 3600)
                m = Market(r.market, r.market, "", "", expiry_ts=expiry, strike=strike)
                eng.poly.market = m
            eng.binance.price = r.spot
            sz = lambda name: getattr(r, name, dflt) if pd.notna(getattr(r, name, np.nan)) else dflt
            m.book_yes = _book(r.bid_yes, r.ask_yes, sz("bid_size_yes"), sz("ask_size_yes"))
            m.book_no = _book(r.bid_no, r.ask_no, sz("bid_size_no"), sz("ask_size_no"))
            q = eng.tick(r.ts)
            if int(r.ts // 60) != last_minute:                    # record once per minute
                last_minute = int(r.ts // 60)
                self._record(r.ts, m, q, m.book_yes.mid)
        if m is not None:
            end_close = close_at.get(int(m.expiry_ts))
            self._finish_market(m, None if end_close is None else bool(end_close >= m.strike), m.expiry_ts)

    # ------------------------------------------------------------------ outputs
    def trades(self) -> pd.DataFrame:
        rows = [dataclasses.asdict(p) for p in self.engine.portfolio.closed]
        return pd.DataFrame(rows)

    def equity_curve(self) -> pd.DataFrame:
        return pd.DataFrame(self.equity)

    def tick_table(self) -> pd.DataFrame:
        return pd.DataFrame(self.ticks)


def _clip(x: float) -> float:
    return float(min(max(round(x, 3), 0.001), 0.999))


def _asof(series: Optional[tuple], T: float) -> Optional[float]:
    """Last price printed at or before T (no look-ahead)."""
    if series is None:
        return None
    t, p = series
    i = int(np.searchsorted(t, T, side="right")) - 1
    return float(p[i]) if i >= 0 else None
