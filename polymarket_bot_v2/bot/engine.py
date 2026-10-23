"""
Orchestration. Three independent polling loops feed shared state; the strategy runs after
every order-book refresh:

    price loop  (1s)   Binance spot
    kline loop  (60s)  closed 1-min klines -> volatility update
    book loop   (2s)   Polymarket books -> tick():
                          1. mark positions at bid
                          2. price the digital (closed form + vol scenarios)
                          3. fill pending orders (latency elapsed)
                          4. check exits
                          5. look for a new entry, size it, queue it
                          6. log the tick (this log is the dataset for the real backtest)
"""

import asyncio
import csv
import logging
import time
from pathlib import Path

import aiohttp

from .config import Config
from .data.binance import BinanceFeed
from .data.polymarket import PolymarketFeed
from .execution.broker import PaperBroker
from .execution.portfolio import Portfolio
from .model.pricing import quote
from .model.volatility import VolatilityModel
from .strategy.signal import SignalGenerator
from .strategy.sizing import Sizer

log = logging.getLogger(__name__)

TICK_FIELDS = ["ts", "market", "spot", "strike", "minutes_left", "sigma_annual", "ewma_lambda", "jump_intensity",
               "theo_yes", "prudent_yes", "prudent_no", "bid_yes", "ask_yes", "bid_no", "ask_no",
               "signal_side", "signal_edge", "equity", "exposure", "open_positions"]


class Engine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.vol = VolatilityModel(cfg.model.ewma_lambda, cfg.model.jump_threshold_sigma,
                                  cfg.model.fit_lambda, cfg.model.lambda_bounds)
        self.signals = SignalGenerator(cfg.signal)
        self.portfolio = Portfolio(cash=cfg.capital_usd, exits=cfg.exits,
                                   fee=cfg.execution.fee_on_profit, trades_csv=cfg.logging.trades_csv)
        self.broker = PaperBroker(self.portfolio, Sizer(cfg.sizing, cfg.execution.fee_on_profit),
                                  cfg.execution.latency_seconds, cfg.signal.min_edge, cfg.sizing.depth_tolerance)
        self.running = False

    # ------------------------------------------------------------------ loops
    async def run(self) -> None:
        async with aiohttp.ClientSession() as s:
            self.binance, self.poly = BinanceFeed(s), PolymarketFeed(s)
            klines = await self.binance.history(self.cfg.data.calibration_klines)
            self.vol.calibrate([float(k[4]) for k in klines])
            log.info("Calibrated: lambda %.4f (half-life %.1f min), sigma %.1f%% annual, %d jumps / %d returns",
                     self.vol.lam, self.vol.half_life_minutes, 100 * self.vol.sigma_annual,
                     len(self.vol.jumps), self.vol.n_returns)
            if self.vol.fit_report:
                log.info("Lambda fit: %s", self.vol.fit_report)
            self.running = True
            await asyncio.gather(self._loop(self._price, self.cfg.data.price_poll_seconds),
                                 self._loop(self._klines, self.cfg.data.kline_poll_seconds),
                                 self._loop(self._books, self.cfg.data.book_poll_seconds))

    async def _loop(self, fn, every: float) -> None:
        while self.running:
            try:
                await fn()
            except Exception as e:                      # a failed poll never kills the bot
                log.warning("%s failed: %s", fn.__name__, e)
            await asyncio.sleep(every)

    async def _price(self) -> None:
        await self.binance.poll_price()

    async def _klines(self) -> None:
        for k in await self.binance.new_closed_klines():
            self.vol.on_closed_kline(float(k[4]))

    async def _books(self) -> None:
        m = self.poly.market
        if m is None or m.minutes_to_expiry() <= 0:
            m = await self.poly.discover()
        if m is None:
            return
        if m.strike is None:
            m.strike = await self.binance.hourly_open(m.expiry_ts - 3600)
        await self.poly.refresh_books()
        self.tick()

    # ------------------------------------------------------------------ strategy
    def tick(self, now: float = None) -> None:
        now = now or time.time()
        m = self.poly.market
        if m.strike is None or not self.binance.is_fresh(self.cfg.data.max_data_age_seconds):
            return
        self.portfolio.mark(m)
        q = quote(self.binance.price, m.strike, m.minutes_to_expiry(now),
                  self.vol.sigma_minute, self.vol.jump_params(), self.cfg.model.vol_stress)

        self.broker.process(m, q, now)
        self.portfolio.check_exits(m, q.theo_yes, now)

        opp = self.signals.evaluate(m, q, now)
        if opp:
            self.broker.submit(opp, m, now)
        self._log_tick(now, m, q, opp)

    def _log_tick(self, now, m, q, opp) -> None:
        path = Path(self.cfg.logging.signals_csv)
        new = not path.exists()
        by, bn = m.book_yes, m.book_no
        row = [round(now, 1), m.slug, self.binance.price, m.strike, round(m.minutes_to_expiry(now), 2),
               round(self.vol.sigma_annual, 4), round(self.vol.lam, 4), round(self.vol.jump_params().intensity, 5),
               round(q.theo_yes, 4), round(q.prudent_yes, 4), round(q.prudent_no, 4),
               by.best_bid, by.best_ask, bn.best_bid, bn.best_ask,
               opp.side if opp else "", round(opp.edge, 4) if opp else "",
               round(self.portfolio.equity, 2), round(self.portfolio.total_exposure, 2), len(self.portfolio.open)]
        with path.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(TICK_FIELDS)
            w.writerow(row)
