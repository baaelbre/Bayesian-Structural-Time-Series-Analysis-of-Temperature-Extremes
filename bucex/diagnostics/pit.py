from __future__ import annotations

import numpy as np

from ..risk._base import cdf_from_obs


def probability_integral_transform(y, eta, obs, params_obs):
    """Compute PIT values F_t(y_t) under the fitted observation model."""
    return np.asarray(cdf_from_obs(y=y, eta=eta, obs=obs, params=params_obs), dtype=float)
