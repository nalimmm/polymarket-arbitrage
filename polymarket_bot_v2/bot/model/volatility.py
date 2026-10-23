"""
Volatility estimation on closed 1-minute klines.

Diffusive variance follows an EWMA recursion, updated at each closed kline:

    var(t) = lambda * var(t-1) + (1 - lambda) * r(t)^2

lambda sets the memory of the model (half-life = ln 0.5 / ln lambda minutes). It is NOT taken
from RiskMetrics (0.94 was calibrated on DAILY returns): it is estimated by maximum likelihood
on the calibration window, i.e. the lambda whose one-step-ahead variance forecasts best explain
the observed 1-minute returns:

    max_lambda  sum_t  -0.5 * [ log var(t|t-1) + r(t)^2 / var(t|t-1) ]

Jumps (|r| > k local sigmas) are excluded from the recursion and from the likelihood, and their
frequency / size are measured separately for the pricing model. Because jump detection depends
on lambda, calibration runs in two passes: detect jumps with the configured lambda, fit lambda on
the remaining returns, then rebuild the variance path with the fitted lambda.
All quantities are per minute.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from scipy.optimize import minimize_scalar

MIN_PER_YEAR = 525_960
BURN_IN = 30                     # returns used to seed the recursion, excluded from the likelihood


@dataclass
class JumpParams:
    intensity: float = 0.0      # jumps per minute
    mean: float = 0.0           # mean jump log-return
    std: float = 0.0            # std of jump log-return


def _seed_variance(r: np.ndarray) -> float:
    return float(np.median(r[:BURN_IN] ** 2) / 0.4549)    # robust: median(x^2) / median(chi2_1)


def _jump_mask(r: np.ndarray, lam: float, k: float) -> np.ndarray:
    """Flag returns larger than k sigmas of the EWMA forecast available just before them."""
    var, mask = _seed_variance(r), np.zeros(len(r), bool)
    for i, x in enumerate(r):
        if abs(x) > k * np.sqrt(var):
            mask[i] = True
        else:
            var = lam * var + (1 - lam) * x * x
    return mask


def ewma_neg_loglik(lam: float, r: np.ndarray) -> float:
    """Gaussian negative log-likelihood of returns r under EWMA(lam) one-step forecasts."""
    var, nll = _seed_variance(r), 0.0
    for i, x in enumerate(r):
        if i >= BURN_IN:
            nll += 0.5 * (np.log(var) + x * x / var)
        var = lam * var + (1 - lam) * x * x
    return nll


def fit_lambda(r: np.ndarray, bounds=(0.80, 0.999)) -> float:
    res = minimize_scalar(ewma_neg_loglik, bounds=bounds, args=(r,), method="bounded",
                          options={"xatol": 1e-4})
    return float(res.x)


class VolatilityModel:
    def __init__(self, ewma_lambda: float, jump_threshold_sigma: float,
                 fit_lambda: bool = True, lambda_bounds=(0.80, 0.999)):
        self.lam = ewma_lambda          # initial value, replaced by the MLE if fit_lambda
        self.k = jump_threshold_sigma
        self.fit = fit_lambda
        self.bounds = tuple(lambda_bounds)
        self.var = 0.0
        self.last_close = 0.0
        self.jumps: List[float] = []
        self.n_returns = 0
        self.calibrated = False
        self.fit_report: Optional[dict] = None

    # ---------------------------------------------------------------- updates
    def _update(self, r: float) -> None:
        self.n_returns += 1
        if self.var > 0 and abs(r) > self.k * np.sqrt(self.var):
            self.jumps.append(r)        # jump: recorded, kept out of diffusive variance
            return
        self.var = self.lam * self.var + (1 - self.lam) * r * r

    def calibrate(self, closes: Sequence[float]) -> None:
        """Calibration on historical closes (oldest first): fit lambda, then build the variance path."""
        r = np.diff(np.log(np.asarray(closes, dtype=float)))
        if len(r) < 200:
            raise ValueError("need at least 200 one-minute returns to calibrate")

        if self.fit:
            clean = r[~_jump_mask(r, self.lam, self.k)]          # pass 1: jumps with initial lambda
            lam_hat = fit_lambda(clean, self.bounds)             # pass 2: MLE on diffusive returns
            self.fit_report = {
                "lambda_initial": self.lam,
                "lambda_fitted": lam_hat,
                "half_life_minutes": float(np.log(0.5) / np.log(lam_hat)),
                "loglik_gain_vs_initial": ewma_neg_loglik(self.lam, clean) - ewma_neg_loglik(lam_hat, clean),
                "n_returns": int(len(clean)),
            }
            self.lam = lam_hat

        self.var, self.jumps, self.n_returns = _seed_variance(r), [], 0   # pass 3: rebuild state
        for x in r:
            self._update(float(x))
        self.last_close = float(closes[-1])
        self.calibrated = True

    def on_closed_kline(self, close: float) -> None:
        if self.last_close > 0:
            self._update(float(np.log(close / self.last_close)))
        self.last_close = close

    # ---------------------------------------------------------------- outputs
    @property
    def half_life_minutes(self) -> float:
        return float(np.log(0.5) / np.log(self.lam))

    @property
    def sigma_minute(self) -> float:
        return float(np.sqrt(self.var))

    @property
    def sigma_annual(self) -> float:
        return self.sigma_minute * np.sqrt(MIN_PER_YEAR)

    def jump_params(self) -> JumpParams:
        if self.n_returns == 0 or len(self.jumps) < 2:
            return JumpParams()
        j = np.asarray(self.jumps)
        return JumpParams(intensity=len(j) / self.n_returns, mean=float(j.mean()), std=float(j.std(ddof=1)))
