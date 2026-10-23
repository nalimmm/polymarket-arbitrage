"""
Position sizing: fractional Kelly for binary tokens.

Buying a token at price `ask` that pays 1 with probability q:
    b = (1 - ask) * (1 - fee) / ask        net gain per dollar staked if the token pays out
    f = (q * b - (1 - q)) / b              full Kelly fraction of equity

Rules on top of Kelly:
  1. q is the *prudent* probability (worst case of the vol scenarios), not the central one.
  2. Stake = kelly_fraction x f, capped at max_pct_per_outcome, on CURRENT equity
     (stakes shrink automatically in drawdown).
  3. The result is a TARGET exposure per outcome (market x side): we only add the difference
     with what is already open, so repeated signals never stack.
  4. Capped by remaining total-exposure budget and by 50% of ask depth near the best price.
"""

from dataclasses import dataclass

from ..config import SizingCfg


def kelly_binary(q: float, ask: float, fee: float) -> float:
    if not 0 < ask < 1:
        return 0.0
    b = (1 - ask) * (1 - fee) / ask
    return max(0.0, (q * b - (1 - q)) / b)


@dataclass(frozen=True)
class SizeDecision:
    usd: float
    kelly_full: float
    target_usd: float
    binding: str               # which constraint set the final size


class Sizer:
    def __init__(self, cfg: SizingCfg, fee: float):
        self.cfg, self.fee = cfg, fee

    def size(self, q_prudent: float, ask: float, equity: float, outcome_exposure: float,
             total_exposure: float, depth_usd: float) -> SizeDecision:
        c = self.cfg
        f = kelly_binary(q_prudent, ask, self.fee)
        target = equity * min(c.kelly_fraction * f, c.max_pct_per_outcome)

        candidates = {
            "kelly" if c.kelly_fraction * f < c.max_pct_per_outcome else "cap_per_outcome":
                target - outcome_exposure,
            "total_exposure": equity * c.max_total_exposure_pct - total_exposure,
            "depth": c.max_depth_share * depth_usd,
        }
        binding = min(candidates, key=candidates.get)
        usd = max(0.0, candidates[binding])
        if usd < c.min_order_usd:
            usd, binding = 0.0, binding + "|below_min_order"
        return SizeDecision(usd=round(usd, 2), kelly_full=f, target_usd=target, binding=binding)
