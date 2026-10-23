"""
Backtest analysis.

1. Performance: trade statistics, daily Sharpe, max drawdown, exit breakdown.
2. Model calibration: when the model says 60%, does BTC finish up 60% of the time?
   Brier score of the model vs the Polymarket price on the same minutes. This is the central
   question of the strategy: the bot can only make money if the model is better calibrated
   than the market.
3. Volatility stress: ratio between the realized vol over the next 60 minutes and the EWMA
   forecast. Its dispersion is the empirical answer to "how wrong can sigma be?", i.e. the
   value `model.vol_stress` should take.

Caveat printed in the report: consecutive minutes of the same market are strongly dependent,
so the effective sample size for calibration is the number of MARKETS, not the number of rows.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
import pandas as pd

BLUE, ORANGE, GREY = "#2F5FD0", "#ED7D31", "#404040"


# ============================================================================ performance
def performance(trades: pd.DataFrame, equity: pd.DataFrame, capital: float, start: float, end: float) -> dict:
    if trades.empty:
        return {"n_trades": 0}
    pnl = trades["pnl"]
    days = pd.to_datetime(trades["exit_ts"], unit="s", utc=True).dt.floor("D")
    span = pd.date_range(pd.to_datetime(start, unit="s", utc=True).floor("D"),
                         pd.to_datetime(end - 1, unit="s", utc=True).floor("D"), freq="D")
    daily = pnl.groupby(days).sum().reindex(span, fill_value=0.0) / capital
    eq = equity["equity"]
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    return {
        "n_trades": int(len(trades)),
        "n_markets_traded": int(trades["market"].nunique()),
        "win_rate": float((pnl > 0).mean()),
        "avg_win_usd": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_usd": float(losses.mean()) if len(losses) else 0.0,
        "win_loss_ratio": float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else None,
        "total_pnl_usd": float(pnl.sum()),
        "total_return": float(pnl.sum() / capital),
        "sharpe_daily_annualized": float(daily.mean() / daily.std(ddof=1) * np.sqrt(365))
        if daily.std(ddof=1) > 0 else None,
        "n_days": int(len(daily)),
        "max_drawdown": float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0,
        "avg_holding_minutes": float(((trades["exit_ts"] - trades["entry_ts"]) / 60).mean()),
        "avg_stake_usd": float(trades["cost"].mean()),
        "exit_reasons": trades["exit_reason"].value_counts().to_dict(),
    }


# ============================================================================ calibration
def calibration(ticks: pd.DataFrame, bins: int = 10):
    t = ticks.dropna(subset=["outcome_up", "market_yes"])
    if t.empty:
        return {}, pd.DataFrame()
    y = t["outcome_up"].astype(float)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for name, col in (("model", "theo_yes"), ("market", "market_yes")):
        b = pd.cut(t[col], edges, include_lowest=True)
        g = t.assign(y=y).groupby(b, observed=True).agg(forecast=(col, "mean"), observed=("y", "mean"),
                                                         n=("y", "size"))
        rows.append(g.reset_index(drop=True).assign(source=name))

    def brier(p):
        return float(((p - y) ** 2).mean())

    def logloss(p):
        p = p.clip(1e-4, 1 - 1e-4)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    stats = {
        "rows": int(len(t)), "markets": int(t["slug"].nunique()),
        "brier_model": brier(t["theo_yes"]), "brier_market": brier(t["market_yes"]),
        "logloss_model": logloss(t["theo_yes"]), "logloss_market": logloss(t["market_yes"]),
    }
    stats["brier_skill_vs_market"] = 1 - stats["brier_model"] / stats["brier_market"]
    return stats, pd.concat(rows, ignore_index=True)


# ============================================================================ vol stress
def vol_stress(ticks: pd.DataFrame, klines: pd.DataFrame, horizon: int = 60) -> dict:
    close = klines.set_index(klines["t"] + 60)["close"]
    r = np.log(close).diff()
    r2 = (r ** 2)
    fwd = r2[::-1].rolling(horizon, min_periods=horizon).mean()[::-1].shift(-1)   # next `horizon` minutes
    t = ticks.drop_duplicates("ts")
    minute = (t["ts"] // 60 * 60).astype(int)
    realized = np.sqrt(fwd.reindex(minute).to_numpy())
    ratio = pd.Series(realized / t["sigma_minute"].to_numpy()).replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        return {}
    q = ratio.quantile([0.1, 0.2, 0.5, 0.8, 0.9])
    return {
        "n": int(len(ratio)),
        "ratio_quantiles": {f"q{int(k * 100)}": float(v) for k, v in q.items()},
        "suggested_vol_stress": float(max(1 - q[0.2], q[0.8] - 1)),
        "note": "realized vol over the next hour / EWMA forecast; stress chosen to cover the central 60%",
    }


# ============================================================================ charts
def _style(ax, title, xlabel, ylabel):
    plt.rcParams["font.family"] = "DejaVu Sans"
    ax.set_title(title, fontsize=13, color=GREY, loc="left", pad=12)
    ax.set_xlabel(xlabel, color=GREY)
    ax.set_ylabel(ylabel, color=GREY)
    ax.grid(axis="y", color="#E0E0E0", lw=0.75)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#BFBFBF")
    ax.tick_params(colors=GREY, length=0)


def plot_equity(equity: pd.DataFrame, capital: float, path: Path, label: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    x = pd.to_datetime(equity["ts"], unit="s", utc=True)
    ax.plot(x, 100 * (equity["equity"] / capital - 1), color=BLUE, lw=1.0, label=label)
    ax.axhline(0, color="black", lw=1)
    _style(ax, "Cumulative PnL", "", "% of capital")
    ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def plot_calibration(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], color="#9CA3AF", lw=1, ls="--", label="Perfect calibration")
    for src, color in (("model", BLUE), ("market", ORANGE)):
        g = table[table["source"] == src]
        ax.plot(g["forecast"], g["observed"], marker="o", color=color, lw=1.5,
                label="Model" if src == "model" else "Polymarket price")
    _style(ax, "Calibration: forecast vs realized frequency", "Forecast probability of Up", "Observed frequency of Up")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


# ============================================================================ report
def write_report(out: Path, mode: str, perf: dict, calib: dict, stress: dict, extra: dict) -> None:
    (out / "metrics.json").write_text(json.dumps(
        {"mode": mode, "performance": perf, "calibration": calib, "vol_stress": stress, **extra}, indent=2, default=str))

    def f(x, pct=False, nd=2):
        if x is None:
            return "n/a"
        return f"{100 * x:.{nd}f}%" if pct else f"{x:.{nd}f}"

    lines = [f"# Backtest report ({mode} mode)", ""]
    if mode == "historical":
        lines += ["> Historical mode uses Polymarket price history wrapped in an ASSUMED spread and depth "
                  "(no order book history exists). Execution is therefore approximate and likely optimistic.", ""]
    lines += ["## Performance", "", "| Metric | Value |", "|---|---|",
              f"| Trades | {perf.get('n_trades', 0)} |",
              f"| Markets traded | {perf.get('n_markets_traded', 0)} |",
              f"| Win rate | {f(perf.get('win_rate'), True, 1)} |",
              f"| Avg win / avg loss | {f(perf.get('win_loss_ratio'))}x |",
              f"| Total return | {f(perf.get('total_return'), True)} |",
              f"| Sharpe (daily, annualized) | {f(perf.get('sharpe_daily_annualized'))} |",
              f"| Max drawdown | {f(perf.get('max_drawdown'), True)} |",
              f"| Days | {perf.get('n_days', 0)} |",
              f"| Avg holding (min) | {f(perf.get('avg_holding_minutes'), nd=1)} |",
              f"| Exits | {perf.get('exit_reasons', {})} |", ""]
    if calib:
        lines += ["## Model calibration", "",
                  f"{calib['markets']} markets, {calib['rows']} minute observations. "
                  "Observations within a market are dependent: the effective sample size is the number of markets.", "",
                  "| | Model | Polymarket |", "|---|---|---|",
                  f"| Brier score (lower is better) | {calib['brier_model']:.4f} | {calib['brier_market']:.4f} |",
                  f"| Log loss | {calib['logloss_model']:.4f} | {calib['logloss_market']:.4f} |", "",
                  f"Brier skill vs market: {100 * calib['brier_skill_vs_market']:+.1f}% "
                  "(positive = model better calibrated than the market).", ""]
    if stress:
        q = stress["ratio_quantiles"]
        lines += ["## Volatility stress", "",
                  "Realized vol over the next hour divided by the EWMA forecast:", "",
                  " | ".join(f"{k}: {v:.2f}" for k, v in q.items()), "",
                  f"Suggested `model.vol_stress`: **{stress['suggested_vol_stress']:.2f}** "
                  "(covers the central 60% of forecast errors).", ""]
    for k, v in extra.items():
        lines.append(f"- {k}: {v}")
    (out / "report.md").write_text("\n".join(lines) + "\n")
