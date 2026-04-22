from __future__ import annotations

from typing import Any

import numpy as np

from ._base import cdf_from_obs


def exceedance_probability_trajectory(
    threshold: float | np.ndarray,
    eta: np.ndarray,
    obs: Any,
    params_obs: dict[str, Any],
) -> np.ndarray:
    """Return P(Y_t > threshold) along a predictor trajectory."""
    thr = np.asarray(threshold, dtype=float)
    eta = np.asarray(eta, dtype=float)
    return 1.0 - np.asarray(cdf_from_obs(thr, eta, obs, params_obs), dtype=float)
