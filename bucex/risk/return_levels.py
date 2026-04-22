from __future__ import annotations

import numpy as np

from ._base import ppf_from_obs


def return_level_trajectory(
    exceedance_probability: float | np.ndarray,
    eta: np.ndarray,
    obs,
    params_obs: dict[str, float],
) -> np.ndarray:
    """Return the level exceeded with probability p at each time point."""
    p = np.asarray(exceedance_probability, dtype=float)
    q = 1.0 - p
    return np.asarray(ppf_from_obs(q, eta, obs, params_obs), dtype=float)
