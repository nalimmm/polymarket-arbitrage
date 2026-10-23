"""End-to-end backtest tests on synthetic data (no network)."""

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest import analysis, data
from backtest.simulator import Backtester
from bot.config import load_config
from bot.model.pricing import prob_up
from bot.model.volatility import JumpParams

CFG = load_config(str(Path(__file__).resolve().parents[1] / "config.yaml"))
START = 1_790_000_000 - 1_790_000_000 % 86400         # a UTC midnight
DAYS = 2


def make_history(tmp: Path, seed: int = 0, noise: float = 0.04):
    """GBM with slowly varying vol; Polymarket price = true probability + AR(1) noise."""
    rng = np.random.default_rng(seed)
    n_cal = CFG.data.calibration_klines
    n = n_cal + DAYS * 1440 + 60
    t0 = START - n_cal * 60
    sig = 8e-4 * np.exp(0.3 * np.sin(np.arange(n) / 600))
    r = sig * rng.standard_normal(n)
    close = 100_000 * np.exp(np.cumsum(r))
    open_ = np.concatenate([[100_000], close[:-1]])
    kl = pd.DataFrame({"open_time": (t0 + 60 * np.arange(n)) * 1000, "open": open_, "high": close,
                       "low": close, "close": close, "volume": 1.0})
    kl.to_csv(tmp / "klines_1m.csv", index=False)

    metas, prices, close_at = [], [], dict(zip(t0 + 60 * np.arange(n) + 60, close))
    open_at = dict(zip(t0 + 60 * np.arange(n), open_))
    for h in range(START, START + DAYS * 86400, 3600):
        slug = f"m{h}"
        strike = open_at[h]
        metas.append({"slug": slug, "hour_start": h, "token_yes": "y", "token_no": "n",
                      "outcome_up_polymarket": close_at[h + 3600] >= strike})
        e = 0.0
        for T in range(h, h + 3600, 60):
            i = (T - t0) // 60
            true_p = prob_up(close_at[T], strike, (h + 3600 - T) / 60, sig[i], JumpParams())
            e = 0.8 * e + noise * rng.standard_normal()
            p = float(np.clip(true_p + e, 0.01, 0.99))
            prices += [{"slug": slug, "side": "YES", "t": T, "p": p},
                       {"slug": slug, "side": "NO", "t": T, "p": 1 - p}]
    pd.DataFrame(metas).to_csv(tmp / "markets.csv", index=False)
    pd.DataFrame(prices).to_csv(tmp / "prices.csv", index=False)
    return kl


@pytest.fixture(scope="module")
def hist(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    make_history(d)
    return d


def run_hist(d, out, cfg=CFG):
    bt = Backtester(cfg, str(out))
    bt.run_historical(data.load_klines(str(d)), data.load_markets(str(d)), data.load_prices(str(d)),
                      START, START + DAYS * 86400)
    return bt


def test_historical_end_to_end(hist, tmp_path):
    bt = run_hist(hist, tmp_path)
    trades, eq, ticks = bt.trades(), bt.equity_curve(), bt.tick_table()
    assert len(trades) > 10
    assert bt.outcome_mismatch == 0
    assert set(trades["exit_reason"]) <= {"cutoff", "stop_loss", "convergence", "settlement"}
    assert (trades["cost"] <= 0.025 * eq["equity"].max() + 0.01).all()        # 2.5% cap per outcome
    assert eq["exposure"].max() <= 0.10 * eq["equity"].max() + 1e-6       # 10% exposure cap
    assert ticks["outcome_up"].notna().mean() > 0.95                        # outcomes attached
    perf = analysis.performance(trades, eq, CFG.capital_usd, START, START + DAYS * 86400)
    assert perf["n_days"] == DAYS and 0 <= perf["win_rate"] <= 1
    assert (tmp_path / "trades.csv").exists()


def test_model_beats_noisy_market_on_calibration(hist, tmp_path):
    ticks = run_hist(hist, tmp_path).tick_table()
    stats, table = analysis.calibration(ticks)
    assert stats["markets"] == DAYS * 24
    assert stats["brier_model"] < stats["brier_market"]       # the market was built with noise


def test_vol_stress_is_measured(hist, tmp_path):
    ticks = run_hist(hist, tmp_path).tick_table()
    s = analysis.vol_stress(ticks, data.load_klines(str(hist)))
    assert 0.3 < s["ratio_quantiles"]["q50"] < 2.0
    assert 0 < s["suggested_vol_stress"] < 1


def test_no_edge_no_trades(tmp_path):
    """If Polymarket prices are exactly the true probability, the prudent filter must keep us out."""
    d = tmp_path / "fair"
    d.mkdir()
    make_history(d, seed=1, noise=0.0)
    bt = run_hist(d, tmp_path / "out")
    assert len(bt.trades()) <= 2


def test_replay_mode(hist, tmp_path):
    """Build a signals.csv-like log from the synthetic history and replay it."""
    kl = data.load_klines(str(hist))
    prices = data.load_prices(str(hist))
    close_at = dict(zip(kl["t"] + 60, kl["close"]))
    rows = []
    for h in range(START, START + 6 * 3600, 3600):
        slug = f"m{h}"
        t, p = prices[slug]["YES"]
        for T in range(h, h + 3600, 20):                          # a tick every 20 seconds
            i = int(np.searchsorted(t, T, side="right")) - 1
            if i < 0:
                continue
            py = p[i]
            rows.append({"ts": T, "market": slug, "spot": close_at[T - T % 60],
                         "strike": kl.set_index("t").loc[h, "open"], "minutes_left": (h + 3600 - T) / 60,
                         "bid_yes": round(py - 0.01, 3), "ask_yes": round(py + 0.01, 3),
                         "bid_no": round(1 - py - 0.01, 3), "ask_no": round(1 - py + 0.01, 3),
                         "bid_size_yes": 150, "ask_size_yes": 150, "bid_size_no": 150, "ask_size_no": 150})
    logdf = pd.DataFrame(rows)
    bt = Backtester(CFG, str(tmp_path))
    bt.run_replay(logdf, kl)
    assert len(bt.trades()) > 0
    assert bt.tick_table()["outcome_up"].notna().all()
