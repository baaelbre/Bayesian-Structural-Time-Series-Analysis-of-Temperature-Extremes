from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

Array = np.ndarray


def rmse(a: Array, b: Array) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def state_index_map(state_names: tuple[str, ...]) -> dict[str, int]:
    return {name: i for i, name in enumerate(state_names)}


def reconstruct_eta_path(
    x_path: Array,
    model,
    params_state: Mapping[str, object],
    exog: Array | None = None,
) -> Array:
    x_path = np.asarray(x_path, dtype=float)
    T = x_path.shape[0] - 1
    eta = np.zeros(T, dtype=float)

    if exog is not None:
        exog = np.asarray(exog, dtype=float)
        if exog.shape[0] != T:
            raise ValueError("exog must have shape (T, k).")

    for t in range(1, T + 1):
        exog_t = None if exog is None else exog[t - 1]
        design = model.design(t=t, params_state=dict(params_state), exog_t=exog_t)
        eta_t = design.Z @ x_path[t] + design.d
        eta[t - 1] = float(np.atleast_1d(eta_t)[0])

    return eta


def reconstruct_eta_draws(
    draws_states: Array,
    model,
    params_state: Mapping[str, object],
    exog: Array | None = None,
) -> Array:
    draws_states = np.asarray(draws_states, dtype=float)
    M = draws_states.shape[0]
    T = draws_states.shape[1] - 1
    eta_draws = np.zeros((M, T), dtype=float)

    for m in range(M):
        eta_draws[m] = reconstruct_eta_path(
            x_path=draws_states[m],
            model=model,
            params_state=params_state,
            exog=exog,
        )

    return eta_draws


def pointwise_band(
    draws: Array,
    lower: float = 0.05,
    upper: float = 0.95,
) -> tuple[Array, Array, Array]:
    draws = np.asarray(draws, dtype=float)
    lo = np.quantile(draws, lower, axis=0)
    med = np.quantile(draws, 0.5, axis=0)
    hi = np.quantile(draws, upper, axis=0)
    return lo, med, hi


def ensure_results_dir(dirname: str = "results") -> Path:
    outdir = Path(dirname)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir
