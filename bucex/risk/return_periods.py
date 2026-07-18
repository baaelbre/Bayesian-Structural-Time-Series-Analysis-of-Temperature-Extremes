from __future__ import annotations

import numpy as np

from .exceedance import exceedance_probability_trajectory


def return_period_trajectory(
    threshold: float | np.ndarray,
    eta: np.ndarray,
    obs,
    params_obs: dict[str, float],
    *,
    min_probability: float = 1e-12,
) -> np.ndarray:
    """Return 1 / P(Y_t > u) for each time point."""
    p = exceedance_probability_trajectory(threshold=threshold, eta=eta, obs=obs, params_obs=params_obs)
    return 1.0 / np.maximum(p, min_probability)
