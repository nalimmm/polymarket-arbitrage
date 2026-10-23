"""
Backtest CLI.

  # 1. download history once (cached in data/)
  python run_backtest.py download --start 2026-09-01 --end 2026-10-01

  # 2a. historical mode: Polymarket price history + assumed spread/depth
  python run_backtest.py historical --start 2026-09-01 --end 2026-10-01

  # 2b. replay mode: the bot's own signals.csv (real books). Needs klines for the same period:
  python run_backtest.py download --start 2026-10-20 --end 2026-10-23 --klines-only
  python run_backtest.py replay --log signals.csv

Results (trades, equity curve, calibration, charts, report.md) are written to results/<mode>/.
"""

import argparse
import asyncio
import logging
from datetime import timedelta
from pathlib import Path

import aiohttp

from backtest import analysis, data
from backtest.simulator import Backtester
from bot.config import load_config


def _analyse(bt: Backtester, cfg, klines, mode: str, start: float, end: float, extra: dict) -> None:
    out = bt.out
    trades, equity, ticks = bt.trades(), bt.equity_curve(), bt.tick_table()
    trades.to_csv(out / "trades_detailed.csv", index=False)
    equity.to_csv(out / "equity.csv", index=False)
    ticks.to_csv(out / "ticks.csv", index=False)

    perf = analysis.performance(trades, equity, cfg.capital_usd, start, end)
    calib, table = analysis.calibration(ticks)
    stress = analysis.vol_stress(ticks, klines) if not ticks.empty else {}
    if not equity.empty:
        analysis.plot_equity(equity, cfg.capital_usd, out / "equity.png", f"Backtest ({mode})")
    if not table.empty:
        table.to_csv(out / "calibration_table.csv", index=False)
        analysis.plot_calibration(table, out / "calibration.png")
    extra = {**extra, "lambda": round(bt.engine.vol.lam, 4)}
    analysis.write_report(out, mode, perf, calib, stress, extra)
    print((out / "report.md").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["download", "historical", "replay"])
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--log", default="signals.csv")
    ap.add_argument("--klines-only", action="store_true")
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.getLogger("bot.execution.broker").setLevel(logging.WARNING)
    cfg = load_config(a.config)
    d = cfg.backtest.data_dir

    if a.command == "download":
        start, end = data.parse_date(a.start), data.parse_date(a.end)
        if a.klines_only:
            async def k():
                async with aiohttp.ClientSession() as s:
                    df = await data.download_klines(s, start - timedelta(minutes=cfg.data.calibration_klines), end)
                    Path(d).mkdir(parents=True, exist_ok=True)
                    df.to_csv(Path(d) / "klines_1m.csv", index=False)
                    print(f"{len(df)} klines saved")
            asyncio.run(k())
        else:
            asyncio.run(data.download(start, end, d, cfg.data.calibration_klines))
        return

    klines = data.load_klines(d)
    if a.command == "historical":
        start, end = data.parse_date(a.start).timestamp(), data.parse_date(a.end).timestamp()
        bt = Backtester(cfg, str(Path(cfg.backtest.results_dir) / "historical"))
        bt.run_historical(klines, data.load_markets(d), data.load_prices(d), start, end)
        _analyse(bt, cfg, klines, "historical", start, end,
                 {"outcome_mismatch_binance_vs_polymarket": bt.outcome_mismatch,
                  "assumed_spread": cfg.backtest.assumed_spread,
                  "assumed_depth_tokens": cfg.backtest.assumed_depth_tokens})
    else:
        logdf = data.load_replay_log(a.log)
        bt = Backtester(cfg, str(Path(cfg.backtest.results_dir) / "replay"))
        bt.run_replay(logdf, klines)
        _analyse(bt, cfg, klines, "replay", float(logdf["ts"].iloc[0]), float(logdf["ts"].iloc[-1]),
                 {"log_rows": len(logdf)})


if __name__ == "__main__":
    main()
