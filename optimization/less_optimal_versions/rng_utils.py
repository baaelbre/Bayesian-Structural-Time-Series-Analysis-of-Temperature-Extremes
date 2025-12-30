# optimization/rng_utils.py
from __future__ import annotations

import numpy as np
from scipy.stats import invgauss  # type: ignore


def rand_invgauss(mu: float, lam: float, rng: np.random.Generator) -> float:
    """
    Draw X ~ IG(mu, lam) with density proportional to:
        sqrt(lam/(2π x^3)) exp(-lam (x-mu)^2 / (2 mu^2 x))

    SciPy's invgauss(mu=...) corresponds to IG(mu, lam=1).
    Using scaling: if Y ~ IG(mu/lam, 1) then X = lam * Y ~ IG(mu, lam).
    """
    if mu <= 0.0 or lam <= 0.0:
        raise ValueError("Inverse-Gaussian requires mu>0 and lam>0")

    y = invgauss.rvs(mu=mu / lam, random_state=rng)
    return float(lam * y)
