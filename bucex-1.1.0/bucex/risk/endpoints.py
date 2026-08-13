from __future__ import annotations

import numpy as np


def endpoint_trajectory(eta: np.ndarray, params_obs: dict[str, float]) -> np.ndarray:
    """Endpoint trajectory for models with finite upper endpoint.

    For a GEV with xi < 0 the upper endpoint is mu - sigma / xi.
    For xi >= 0 the endpoint is infinite.
    """
    eta = np.asarray(eta, dtype=float)
    sigma = float(params_obs["sigma"])
    xi = float(params_obs["xi"])
    if xi >= 0.0:
        return np.full_like(eta, np.inf, dtype=float)
    return eta - sigma / xi
