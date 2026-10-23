# Polymarket BTC Up/Down Bot (v2)

Paper trading bot for Polymarket's hourly "Bitcoin Up or Down" markets. The YES token pays $1 if BTC closes the hour above its opening price, which makes it a digital option. The bot compares its price to a model probability and takes a position when the gap is large enough and robust.

## Quick start

```bash
pip install -r requirements.txt
python main.py            # paper trading
python -m pytest tests    # 26 tests (unit + end-to-end backtest)
```

All parameters live in `config.yaml`. A missing key makes startup fail: there are no hidden defaults in the code.

## Structure

```
bot/
  config.py              typed loading of config.yaml
  data/
    binance.py           spot price (1s), closed 1-min klines (60s), hourly open = strike
    polymarket.py        hourly market discovery (Gamma API), order books (CLOB)
    types.py             order book: best prices, depth, VWAP fills
  model/
    volatility.py        EWMA on 1-min returns, MLE-fitted lambda, jumps measured separately
    pricing.py           closed-form probability (Merton), volatility scenarios
  strategy/
    signal.py            edge = prudent probability - ask
    sizing.py            fractional Kelly for binary payoffs
  execution/
    broker.py            paper execution: latency, re-check at fill, VWAP
    portfolio.py         bid-side marking, exits, trade journal
  engine.py              data loops and strategy tick
backtest/
  data.py                history download (Binance, Polymarket) with CSV cache
  simulator.py           drives the live Engine with a historical clock
  analysis.py            performance, model calibration, volatility stress, charts
run_backtest.py          backtest CLI
tests/
  test_bot.py            unit tests
  test_backtest.py       end-to-end backtests on synthetic data
```

## 1. Data

Binance: spot price via REST every second; 1-minute klines polled every minute, only closed and not-yet-seen candles are consumed. 2,880 klines (2 days, fetched in pages of 1,000) are used for calibration at startup. The strike is the open of the Binance 1h candle, which is the market's settlement reference.

Polymarket: the hourly market slug is rebuilt in New York time with daylight saving handled (`zoneinfo`), then YES and NO order books are read from the CLOB every 2 seconds.

No trading if the Binance price is more than 10 seconds old. Every tick is logged to `signals.csv` (spot, strike, vol, probabilities, books, signal): this file is the dataset for an honest backtest.

## 2. Signal

Volatility: EWMA on 1-minute log returns, σ²(t) = λ × σ²(t−1) + (1 − λ) × r²(t). The variance is the state variable updated every minute; λ sets the model's memory (half-life = ln 0.5 / ln λ).

λ is not hard-coded at 0.94: that RiskMetrics value was calibrated on daily returns. It is estimated by maximum likelihood over the calibration window, i.e. the value whose one-minute variance forecasts best explain the observed returns. Validation on simulated data: the estimator recovers the true λ with a standard deviation of 0.004 on 2,880 returns (0.016 on 500, hence the 2-day window). The fitted value, its half-life and the likelihood gain versus 0.94 are logged at startup.

Jumps: a return larger than 3 local sigmas is classified as a jump, excluded from the diffusive variance and from the likelihood, and used to estimate jump frequency and size. Since detection depends on λ, calibration runs in three passes: detect jumps with the initial λ, fit λ on the remaining returns, rebuild the variance path with the fitted λ.

Fair value: P(S_T ≥ K) under a Merton jump-diffusion, computed in closed form (a Poisson-weighted sum of normal CDFs). Exact and noise-free, validated against a 400,000-path Monte Carlo in the tests.

Robustness: the probability is recomputed with volatility at −20% and +20%. For each side, the least favorable value is kept: the prudent probability.

Entry: edge = prudent probability − ask. A signal requires an edge between 3 and 30 points, an ask between 5 and 95 cents, more than 5 minutes to expiry, and 15 seconds since the last signal. Both YES and NO are evaluated, only the best one is kept.

## 3. Sizing

Fractional Kelly for binary payoffs: `f = (q × b − (1 − q)) / b`, where q is the prudent probability and b the net gain per dollar staked, `b = (1 − ask) × (1 − fee) / ask`.

The stake is a quarter of Kelly, capped at 2.5% of current equity, so stakes shrink automatically in drawdown. This amount is a target per outcome (market × side): only the gap with the exposure already open is added. The size is then limited by the remaining total exposure budget (10%) and by 50% of the depth available within 2 cents of the best ask. Orders below $1 are not sent.

| Ask | Prudent probability | Full Kelly | Stake ($100) |
|---|---|---|---|
| 0.45 | 0.48 | 4.6% | $1.15 |
| 0.42 | 0.46 | 6.1% | $1.52 |
| 0.58 | 0.66 | 18.1% | $2.50 (cap) |

## 4. Position taking and management

Orders are filled 2 seconds after the decision, against the book at that moment: the edge is re-checked on the new ask (order cancelled if it fell below 3 points), the size is recomputed, and the fill walks the book levels (volume-weighted price). The opposite side of a market already held is never opened.

Positions are marked at the bid. Exits, in order of priority:

1. less than 5 minutes to expiry: forced close;
2. loss of 80% or more: stop;
3. bid ≥ current fair value − 1 cent: convergence. Covers both profit taking (the market caught up with the model) and edge decay (the model moved against us).

A 2% fee is charged on profits. Every closed position is written to `trades.csv`.

## 5. Backtest

The backtest does not reimplement the strategy. It builds the live `Engine`, replaces its two data feeds with historical ones, and calls the same `Engine.tick()` the live bot calls. Volatility, pricing, signal, sizing, latency, fills and exits are the production code, so the backtest and live trading cannot diverge because of the code.

No look-ahead: at time T, the volatility model has only seen klines closed at or before T, the spot is the close of the kline ending at T, and the Polymarket price is the last one printed at or before T. Positions still open at expiry are settled on the Binance hourly close vs open, cross-checked against Polymarket's resolution.

Two modes:

| | Historical | Replay |
|---|---|---|
| Polymarket data | Price history (`prices-history`, 1-min) | The bot's own `signals.csv` |
| Order book | Assumed spread (2 cents) and depth | Real best bid/ask and sizes |
| Sample | Months of past markets, available now | Only what the bot has logged |
| Fidelity | Approximate, likely optimistic on execution | High |

```bash
python run_backtest.py download --start 2026-09-01 --end 2026-10-01
python run_backtest.py historical --start 2026-09-01 --end 2026-10-01

python run_backtest.py download --start 2026-10-20 --end 2026-10-23 --klines-only
python run_backtest.py replay --log signals.csv
```

Each run writes to `results/<mode>/`: trades, equity curve, `report.md`, `metrics.json` and two charts. Beyond PnL, the report answers two questions:

- **Model calibration.** When the model says 60%, does BTC finish up 60% of the time? Brier score and log loss of the model vs the Polymarket price on the same minutes. The strategy can only make money if the model is better calibrated than the market. Minutes of the same market are dependent, so the effective sample size is the number of markets.
- **Volatility stress.** Distribution of realized vol over the next hour divided by the EWMA forecast. Its central 60% gives an empirical value for `model.vol_stress` instead of an arbitrary one.

Tested end to end on synthetic markets: the model beats a deliberately noisy market on calibration, and when the market price equals the true probability the prudent filter keeps the bot out (no edge, no trades).

## Changes from v1

| v1 | v2 | Why |
|---|---|---|
| λ = 0.94 fixed (RiskMetrics, daily data) | λ fitted by MLE on 2 days of 1-min returns | Model memory matched to the minute horizon |
| 10,000-path Monte Carlo | Closed form | Removes ~0.5 pt of noise that was triggering exits |
| Jumps calibrated in a normalized space | Jumps in log-return units | Interpretable parameters, consistent with the model |
| Z-score of mispricing over 100 ticks | Edge on prudent probability | The z-score ignored persistent edges and relied on autocorrelated data |
| Mispricing on mid minus half spread | Edge measured on the ask | The price actually paid |
| Linear sizing, 2% to 2.5% | Fractional Kelly, target per outcome | Size proportional to edge, no stacking |
| Initial capital | Current equity | Automatic reduction in drawdown |
| Cap on total market liquidity | Cap on depth near the best price | Controlled slippage |
| Exit on fair value at entry | Exit on current fair value | The entry target goes stale when BTC moves |
| New York timezone hard-coded as UTC−5 | `zoneinfo` America/New_York | Slug was wrong part of the year |
| Parameters read from the wrong config sections | Typed config, no hidden defaults | Config values are actually used |


