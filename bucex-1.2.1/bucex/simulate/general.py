"""Simulation from a compiled structural model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models.compiler import compile_model
from ..models.structural import Model


@dataclass
class Simulation:
    y: np.ndarray
    eta: np.ndarray
    states: np.ndarray
    params: dict[str, float]
    model: Model
    exog: np.ndarray | None = None


def simulate(
    model: Model,
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
    compiled = compile_model(model, np.zeros(n_time), exog=exog)
    missing = [f"sd.{name}" for name in compiled.noise_names if f"sd.{name}" not in params]
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
    design = compiled.design()
    eta = np.zeros(n_time)
    y = np.zeros(n_time)
    for t in range(1, n_time + 1):
        states[t] = (
            compiled.transition @ states[t - 1]
            + compiled.loading @ (process_sd * rng.normal(size=compiled.noise_dim))
        )
        eta[t - 1] = float(design[t - 1] @ states[t])
        y[t - 1] = float(
            model.observation.sample(
                eta=eta[t - 1],
                sigma=float(params["sigma"]),
                xi=params.get("xi"),
                rng=rng,
            )
        )
    return Simulation(y=y, eta=eta, states=states, params=dict(params), model=model, exog=compiled.exog)
