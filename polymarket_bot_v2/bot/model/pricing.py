"""
Digital option pricing for 'Bitcoin Up or Down' hourly markets.

YES pays 1 if S_T >= K (K = open of the hourly candle), so its fair value is P(S_T >= K).
The log-price follows a Merton jump-diffusion. Conditional on n jumps the log-return is
Gaussian, so the probability is an exact Poisson-weighted sum of normal CDFs:

    P = sum_n  Pois(n; lambda*tau) * Phi( (ln(S/K) + drift + n*mu_J) / sqrt(sigma^2*tau + n*s_J^2) )

Closed form: exact, instantaneous and noise-free (the previous Monte Carlo had ~0.5 pt of noise).
Model uncertainty is handled by repricing with sigma x (1 +/- stress) and keeping, for each side,
the least favorable probability ("prudent probability") for entry and sizing decisions.
"""

from dataclasses import dataclass
from math import exp, lgamma, log, sqrt

import numpy as np
from scipy.stats import norm

from .volatility import JumpParams

P_FLOOR, P_CAP = 0.001, 0.999


def prob_up(spot: float, strike: float, tau_min: float, sigma_min: float, jp: JumpParams) -> float:
    """P(S_T >= K) under zero-drift (martingale) jump-diffusion. Units: minutes."""
    if tau_min <= 0 or sigma_min <= 0:
        return 1.0 if spot >= strike else 0.0
    lam_t = jp.intensity * tau_min
    kappa = exp(jp.mean + 0.5 * jp.std ** 2) - 1 if lam_t > 0 else 0.0
    base = log(spot / strike) - 0.5 * sigma_min ** 2 * tau_min - lam_t * kappa
    if lam_t == 0:
        p = norm.cdf(base / (sigma_min * sqrt(tau_min)))
    else:
        p, mass, n = 0.0, 0.0, 0
        while mass < 1 - 1e-12 and n < 60:
            w = exp(-lam_t + n * log(lam_t) - lgamma(n + 1))
            v = sigma_min ** 2 * tau_min + n * jp.std ** 2
            p += w * norm.cdf((base + n * jp.mean) / sqrt(v))
            mass += w
            n += 1
    return float(min(max(p, P_FLOOR), P_CAP))


@dataclass(frozen=True)
class Quote:
    theo_yes: float            # central fair value of YES
    prudent_yes: float         # lowest P(up) across vol scenarios
    prudent_no: float          # lowest P(down) across vol scenarios

    @property
    def theo_no(self) -> float:
        return 1.0 - self.theo_yes

    def theo(self, side: str) -> float:
        return self.theo_yes if side == "YES" else self.theo_no

    def prudent(self, side: str) -> float:
        return self.prudent_yes if side == "YES" else self.prudent_no


def quote(spot: float, strike: float, tau_min: float, sigma_min: float,
          jp: JumpParams, stress: float) -> Quote:
    scen = [prob_up(spot, strike, tau_min, sigma_min * f, jp) for f in (1 - stress, 1.0, 1 + stress)]
    return Quote(theo_yes=scen[1], prudent_yes=min(scen), prudent_no=1 - max(scen))


def prob_up_mc(spot, strike, tau_min, sigma_min, jp: JumpParams, n_paths=400_000, seed=0) -> float:
    """Monte Carlo reference, used only in tests to validate the closed form."""
    rng = np.random.default_rng(seed)
    lam_t = jp.intensity * tau_min
    kappa = exp(jp.mean + 0.5 * jp.std ** 2) - 1 if lam_t > 0 else 0.0
    x = (-0.5 * sigma_min ** 2 * tau_min - lam_t * kappa) + sigma_min * sqrt(tau_min) * rng.standard_normal(n_paths)
    if lam_t > 0:
        n = rng.poisson(lam_t, n_paths)
        x += n * jp.mean + np.sqrt(n) * jp.std * rng.standard_normal(n_paths)
    return float(np.mean(spot * np.exp(x) >= strike))
