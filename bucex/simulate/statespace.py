# src/sts_extremes/simulate/statespace.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..models.base import LinearDesign, LinearGaussianSystem, StateSpaceModel

Array = np.ndarray


@dataclass(frozen=True)
class SimResult:
    x: Array                  # (T+1, m) including x0
    mu: Array                 # (T,)    eta (scalar) for t=1..T
    y: Array                  # (T,)
    state_names: Tuple[str, ...]

def simulate_statespace(
    T: int,
    model: StateSpaceModel,
    params_state: Dict[str, Any],
    params_obs: Dict[str, Any],
    rng: Optional[np.random.Generator] = None,
    exog: Optional[Array] = None,   # (T, k) optional
    x0: Optional[Array] = None,
) -> SimResult:
    """
    Generic simulator:
      x_t = T_t x_{t-1} + c_t + R_t eps_t, eps_t ~ N(0,Q_t)
      eta_t = Z_t x_t + d_t
      y_t ~ p(· | obs_params(t, x_t, eta_t))

    Notes:
      - We assume linear-Gaussian state evolution.
      - Observation distribution must be specified (currently: Gaussian, GEV) via model.obs.sample(...)
    """
    rng = rng if rng is not None else np.random.default_rng()

    m = model.state_dim
    x = np.zeros((T + 1, m), dtype=float)

    if x0 is not None:
        x[0] = np.asarray(x0, dtype=float).reshape(m)
    else:
        m0, P0 = model.initial_state(params_state)
        m0 = np.asarray(m0, dtype=float).reshape(m)
        P0 = np.asarray(P0, dtype=float).reshape(m, m)
        if m > 0:
            x[0] = rng.multivariate_normal(mean=m0, cov=P0)

    mu = np.zeros(T, dtype=float)
    y = np.zeros(T, dtype=float)

    for t in range(1, T + 1):
        sys: LinearGaussianSystem = model.system(t=t, params_state=params_state)

        # draw eps ~ N(0,Q)
        r = sys.Q.shape[0]
        eps = rng.multivariate_normal(mean=np.zeros(r), cov=sys.Q)
        x[t] = sys.T @ x[t - 1] + sys.c + sys.R @ eps

        exog_t = None if exog is None else np.asarray(exog[t - 1], dtype=float)
        des: LinearDesign = model.design(t=t, params_state=params_state, exog_t=exog_t)

        eta_t = des.Z @ x[t] + des.d
        mu[t - 1] = float(np.atleast_1d(eta_t)[0])

        op = model.obs_params(t=t, x_t=x[t], eta_t=eta_t, params_obs=params_obs, exog_t=exog_t)
        y[t - 1] = float(model.obs.sample(eta=float(np.atleast_1d(eta_t)[0]),params=op,rng=rng,))

    return SimResult(x=x, mu=mu, y=y, state_names=model.state_names)