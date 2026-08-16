"""Simulation from a compiled structural model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models.compiler import compile_model
from ..models.factor import FactorModel
from ..models.multiseries import MultiSeriesModel
from ..models.structural import Model


@dataclass
class Simulation:
    y: np.ndarray
    eta: np.ndarray
    states: np.ndarray
    params: dict[str, float]
    model: Model | MultiSeriesModel | FactorModel
    exog: object = None

    @property
    def channel_names(self) -> tuple[str, ...]:
        return (
            self.model.channel_names
            if isinstance(self.model, (MultiSeriesModel, FactorModel))
            else ()
        )


def simulate(
    model: Model | MultiSeriesModel | FactorModel,
    n_time: int,
    params: dict[str, float],
    *,
    exog=None,
    initial_state=None,
    seed: int | None = None,
) -> Simulation:
    n_time = int(n_time)
    if n_time < 1:
        raise ValueError("n_time must be positive.")
    rng = np.random.default_rng(seed)
    placeholder = (
        np.zeros((n_time, len(model.channel_names)))
        if isinstance(model, (MultiSeriesModel, FactorModel))
        else np.zeros(n_time)
    )
    compiled = compile_model(model, placeholder, exog=exog)
    missing = [f"sd.{name}" for name in compiled.noise_names if f"sd.{name}" not in params]
    if isinstance(model, (MultiSeriesModel, FactorModel)):
        required = list(compiled.observation_parameter_names) + list(
            compiled.estimated_loading_names
        )
    else:
        required = ["sigma"] + (["xi"] if model.family == "gev" else [])
    missing.extend(name for name in required if name not in params)
    if missing:
        raise ValueError(f"Missing simulation parameters: {missing}")
    states = np.zeros((n_time + 1, compiled.state_dim))
    if initial_state is None:
        states[0] = np.zeros(compiled.state_dim)
    else:
        states[0] = np.asarray(initial_state, dtype=float).reshape(compiled.state_dim)
    process_sd = compiled.process_vector(params)
    design = compiled.design(params=params)
    if isinstance(model, (MultiSeriesModel, FactorModel)):
        eta = np.zeros((n_time, len(model.channel_names)))
        y = np.zeros_like(eta)
    else:
        eta = np.zeros(n_time)
        y = np.zeros(n_time)
    for t in range(1, n_time + 1):
        states[t] = (
            compiled.transition @ states[t - 1]
            + compiled.loading @ (process_sd * rng.normal(size=compiled.noise_dim))
        )
        if isinstance(model, (MultiSeriesModel, FactorModel)):
            eta[t - 1] = design[t - 1] @ states[t]
            y[t - 1] = compiled.sample_observation(eta[t - 1], params, rng)
        else:
            eta[t - 1] = float(design[t - 1] @ states[t])
            y[t - 1] = float(
                model.observation.sample(
                    eta=eta[t - 1],
                    sigma=float(params["sigma"]),
                    xi=params.get("xi"),
                    rng=rng,
                )
            )
    if isinstance(model, (MultiSeriesModel, FactorModel)):
        signs = model.transform_signs[None, :]
        y = signs * y
        eta = signs * eta
    return Simulation(y=y, eta=eta, states=states, params=dict(params), model=model, exog=compiled.exog)
