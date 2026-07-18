from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ..models.base import StateSpaceModel

Array = np.ndarray


@dataclass
class StateForecast:
    x_future: Array
    eta_future: Array


def forecast_state_path(
    model: StateSpaceModel,
    x_last: Array,
    params_state: Dict[str, Any],
    horizon: int,
    *,
    exog_future: Optional[Array] = None,
) -> StateForecast:
    """Deterministic state mean forecast under the latent transition law.

    This low-level helper intentionally remains simple: it propagates transition means and
    predictor means only, without simulating future observation noise.
    """
    x_last = np.asarray(x_last, dtype=float).reshape(model.state_dim)
    x_future = np.zeros((horizon, model.state_dim), dtype=float)
    eta_future = np.zeros(horizon, dtype=float)
    x_prev = x_last.copy()

    for h in range(1, horizon + 1):
        sys = model.system(t=h, params_state=params_state)
        x_now = sys.T @ x_prev + sys.c
        x_future[h - 1] = x_now
        exog_t = None if exog_future is None else np.asarray(exog_future[h - 1], dtype=float)
        des = model.design(t=h, params_state=params_state, exog_t=exog_t)
        eta_future[h - 1] = float(np.atleast_1d(des.Z @ x_now + des.d)[0])
        x_prev = x_now

    return StateForecast(x_future=x_future, eta_future=eta_future)
