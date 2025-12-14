"""Small statistical helpers used by ecology loop gating."""

from __future__ import annotations

import math


def compute_mean_ci(series: list[float], alpha: float = 0.05) -> tuple[float, float, float, float]:
    """Compute a normal-approximation CI for the mean of a series.

    Returns (ci_low, ci_high, mean, se). Uses sample std (Bessel corrected) when n>1.
    """
    n = len(series)
    if n == 0:
        return (0.0, 0.0, 0.0, float("inf"))
    mu = sum(series) / n
    if n > 1:
        var = sum((x - mu) ** 2 for x in series) / (n - 1)
    else:
        var = 0.0
    se = math.sqrt(max(var, 1e-12)) / math.sqrt(n)
    z = 1.96 if abs(alpha - 0.05) < 1e-6 else 1.0
    return (mu - z * se, mu + z * se, mu, se)


def power_proxy(mu: float, baseline: float, margin: float, se: float, alpha: float = 0.05) -> float:
    """Approximate one-sided power using a normal approximation.

    Computes z_eff = (mu - (baseline + margin)) / se and returns Phi(z_eff - z_alpha).
    This is a heuristic proxy in lieu of an exact test; bounded to [0,1].
    """
    if se <= 0.0:
        return 1.0 if mu > (baseline + margin) else 0.0
    z_alpha = 1.645 if abs(alpha - 0.05) < 1e-6 else 1.0
    z_eff = (mu - (baseline + margin)) / max(se, 1e-6)
    # standard normal CDF via erf
    cdf = 0.5 * (1.0 + math.erf((z_eff - z_alpha) / math.sqrt(2.0)))
    return max(0.0, min(1.0, float(cdf)))


def team_accept(ci_low: float, baseline: float, margin: float, n: int, min_tasks: int) -> bool:
    if n < min_tasks:
        return False
    return ci_low > (baseline + margin)


__all__ = ["compute_mean_ci", "power_proxy", "team_accept"]
