from __future__ import annotations

from typing import Any

import numpy as np

from ..api.forecast import forecast_state_path
from ..models.base import StateSpaceModel
from .exceedance import exceedance_probability_trajectory


def forecast_exceedance_from_states(
    model: StateSpaceModel,
    x_last: np.ndarray,
    params_state: dict[str, Any],
    params_obs: dict[str, Any],
    threshold: float,
    horizon: int,
    *,
    exog_future: np.ndarray | None = None,
) -> np.ndarray:
    """Forecast exceedance probabilities from the current latent state mean."""
    fc = forecast_state_path(
        model=model,
        x_last=x_last,
        params_state=params_state,
        horizon=horizon,
        exog_future=exog_future,
    )
    return exceedance_probability_trajectory(
        threshold=threshold,
        eta=fc.eta_future,
        obs=model.obs,
        params_obs=params_obs,
    )
