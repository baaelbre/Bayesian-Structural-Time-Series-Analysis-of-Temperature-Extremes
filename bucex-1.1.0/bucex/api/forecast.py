from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ..models.base import StateSpaceModel
from .._general.forecast import Forecast, _future_dates
from .._general.observations import GEV, Gaussian
from ..inference.fit.noncentered_utils import _sample_gaussian

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


def posterior_predict_fs(
    fit,
    horizon: int,
    *,
    exog_future: Optional[Array] = None,
    draws: Optional[int] = None,
    seed: Optional[int] = None,
    dates: Optional[Array] = None,
) -> Forecast:
    """Simulate the full posterior predictive distribution from an FS fit."""
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be positive.")
    if fit.draws_states is None or fit.n_draws < 1:
        raise ValueError("The fit contains no retained state trajectories.")
    rng = np.random.default_rng(seed)
    requested = fit.n_draws if draws is None else int(draws)
    if requested < 1:
        raise ValueError("draws must be positive.")
    indices = (
        np.arange(fit.n_draws)
        if requested == fit.n_draws
        else rng.choice(fit.n_draws, size=requested, replace=requested > fit.n_draws)
    )
    if exog_future is not None:
        exog_future = np.asarray(exog_future, dtype=float)
        if exog_future.shape[0] != horizon:
            raise ValueError("exog_future must have horizon rows.")

    states = np.zeros((requested, horizon, fit.model.state_dim), dtype=float)
    eta_model = np.zeros((requested, horizon), dtype=float)
    observations_model = np.zeros((requested, horizon), dtype=float)
    parameter_values = {
        name: np.asarray(fit.draws_static[name])[indices]
        for name in ("sigma", "xi")
        if name in fit.draws_static
    }
    initial_state_params = dict(
        getattr(fit, "initial_values", {}).get("params_state", {})
    )
    start_time = int(fit.n_time or 0)

    for output_index, posterior_index in enumerate(indices):
        state = np.asarray(fit.draws_states[posterior_index, -1], dtype=float).copy()
        params_state = dict(initial_state_params)
        for name, values in fit.draws_static.items():
            array = np.asarray(values)
            if array.shape[0] == fit.n_draws:
                value = array[posterior_index]
                params_state[name] = (
                    float(value) if np.asarray(value).ndim == 0 else np.asarray(value).copy()
                )
        params_obs = {
            name: float(values[output_index]) for name, values in parameter_values.items()
        }
        for step in range(horizon):
            absolute_time = start_time + step + 1
            system = fit.model.system(t=absolute_time, params_state=params_state)
            disturbance = _sample_gaussian(
                np.zeros(system.Q.shape[0], dtype=float),
                np.asarray(system.Q, dtype=float),
                rng,
            )
            state = system.T @ state + system.c + system.R @ disturbance
            states[output_index, step] = state
            exog_t = None if exog_future is None else exog_future[step]
            design = fit.model.design(
                t=absolute_time, params_state=params_state, exog_t=exog_t
            )
            eta = float(np.atleast_1d(design.Z @ state + design.d)[0])
            eta_model[output_index, step] = eta
            observations_model[output_index, step] = float(
                fit.model.obs.sample(eta=eta, params=params_obs, rng=rng)
            )

    resolved_dates = _future_dates(fit.dates, horizon) if dates is None else np.asarray(dates)
    if np.asarray(resolved_dates).reshape(-1).size != horizon:
        raise ValueError("dates must have length equal to horizon.")
    sign = float(fit.transform_sign)
    observation_model = GEV() if fit.obs_name == "gev" else Gaussian()
    return Forecast(
        observations=sign * observations_model,
        eta=sign * eta_model,
        states=states,
        parameters=parameter_values,
        dates=resolved_dates,
        family=str(fit.obs_name),
        tail="lower" if sign < 0.0 else "upper",
        observation_model=observation_model,
        transform_sign=sign,
    )
