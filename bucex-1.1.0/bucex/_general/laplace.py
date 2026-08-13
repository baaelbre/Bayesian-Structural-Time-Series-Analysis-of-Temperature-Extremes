"""One convergence-checked iterated Laplace backend for GEV models."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .compiler import CompiledModel
from .kalman import KalmanResult, ffbs, kalman_filter, kalman_smoother
from .numerics import (
    factored_singular_normal_logpdf,
    gaussian_support,
    singular_normal_logpdf,
)


Array = np.ndarray


@dataclass
class LaplaceDraw:
    path: Array
    mode_path: Array
    pseudo_y: Array
    pseudo_variance: Array
    converged: bool
    iterations: int
    relative_change: float
    objective: float
    support_rejections: int
    filter: KalmanResult


def observation_log_likelihood(
    y: Array,
    eta: Array,
    compiled: CompiledModel,
    params: dict[str, float],
) -> float:
    mask = np.isfinite(y)
    if not np.any(mask):
        return 0.0
    values = compiled.model.observation.logpdf(
        np.asarray(y)[mask],
        np.asarray(eta)[mask],
        sigma=float(params["sigma"]),
        xi=params.get("xi"),
    )
    if not np.all(np.isfinite(values)):
        return -np.inf
    return float(np.sum(values))


def state_log_density(path: Array, compiled: CompiledModel, params: dict[str, float]) -> float:
    path = np.asarray(path, dtype=float)
    value = singular_normal_logpdf(path[0], compiled.initial_mean, compiled.initial_cov)
    covariance = compiled.transition_cov(params)
    means = path[:-1] @ compiled.transition.T
    transition_values = factored_singular_normal_logpdf(
        path[1:], means, gaussian_support(covariance)
    )
    if not np.all(np.isfinite(transition_values)):
        return -np.inf
    return float(value + np.sum(transition_values))


def joint_state_log_density(
    y: Array,
    path: Array,
    compiled: CompiledModel,
    params: dict[str, float],
) -> float:
    eta = compiled.eta(path)
    return state_log_density(path, compiled, params) + observation_log_likelihood(
        y, eta, compiled, params
    )


def _pseudo_data(
    y: Array,
    eta: Array,
    compiled: CompiledModel,
    params: dict[str, float],
    *,
    curvature_floor: float,
    maximum_variance: float,
    shift_limit: float,
) -> tuple[Array, Array]:
    y = np.asarray(y, dtype=float)
    pseudo_y = np.full(y.size, np.nan)
    pseudo_variance = np.ones(y.size)
    mask = np.isfinite(y)
    if not np.any(mask):
        return pseudo_y, pseudo_variance
    obs = compiled.model.observation
    grad = np.asarray(
        obs.grad_eta(y[mask], eta[mask], sigma=params["sigma"], xi=params.get("xi")),
        dtype=float,
    )
    hess = np.asarray(
        obs.hess_eta(y[mask], eta[mask], sigma=params["sigma"], xi=params.get("xi")),
        dtype=float,
    )
    if np.any(~np.isfinite(grad)) or np.any(~np.isfinite(hess)):
        raise FloatingPointError("Invalid GEV derivatives at the Laplace expansion point.")
    information = np.maximum(-hess, float(curvature_floor))
    variance = np.minimum(1.0 / information, float(maximum_variance))
    shift = np.clip(grad / information, -float(shift_limit), float(shift_limit))
    pseudo_y[mask] = eta[mask] + shift
    pseudo_variance[mask] = variance
    return pseudo_y, pseudo_variance


def iterated_laplace(
    y: Array,
    compiled: CompiledModel,
    params: dict[str, float],
    rng: np.random.Generator,
    *,
    initial_path: Array | None = None,
    max_iterations: int = 30,
    tolerance: float = 1e-5,
    curvature_floor: float = 1e-6,
    maximum_variance: float = 1e8,
    shift_limit: float | None = None,
    draw_attempts: int = 30,
) -> LaplaceDraw:
    if compiled.family != "gev":
        raise ValueError("The iterated Laplace backend is intended for GEV observations.")
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.size != compiled.n_time:
        raise ValueError("y and compiled model have different lengths.")
    if max_iterations < 1 or tolerance <= 0.0:
        raise ValueError("max_iterations and tolerance must be positive.")
    shift_limit = float(10.0 * params["sigma"] if shift_limit is None else shift_limit)

    if initial_path is None:
        initial_filter = kalman_filter(
            y,
            compiled,
            params,
            observation_variance=max(float(params["sigma"]) ** 2, 1e-8),
        )
        mode_path = compiled.project_path(
            kalman_smoother(initial_filter, compiled).mean, params
        )
    else:
        mode_path = np.asarray(initial_path, dtype=float).copy()
        if mode_path.shape != (y.size + 1, compiled.state_dim):
            raise ValueError("initial_path has the wrong shape.")

    current_objective = joint_state_log_density(y, mode_path, compiled, params)
    if not np.isfinite(current_objective):
        # A Gaussian warm start can cross a finite GEV endpoint. Use the data as
        # pseudo observations once to obtain a support-compatible location.
        fallback = kalman_filter(
            y,
            compiled,
            params,
            observation_variance=max(float(params["sigma"]) ** 2, 1e-6),
        )
        mode_path = compiled.project_path(kalman_smoother(fallback, compiled).mean, params)
        current_objective = joint_state_log_density(y, mode_path, compiled, params)
    if not np.isfinite(current_objective):
        raise FloatingPointError("Could not initialize the GEV Laplace mode inside the support.")

    converged = False
    relative_change = np.inf
    pseudo_y = y.copy()
    pseudo_variance = np.full(y.size, float(params["sigma"]) ** 2)
    filter_result: KalmanResult | None = None

    for iteration in range(1, int(max_iterations) + 1):
        eta = compiled.eta(mode_path)
        pseudo_y, pseudo_variance = _pseudo_data(
            y,
            eta,
            compiled,
            params,
            curvature_floor=curvature_floor,
            maximum_variance=maximum_variance,
            shift_limit=shift_limit,
        )
        filter_result = kalman_filter(
            pseudo_y,
            compiled,
            params,
            observation_variance=pseudo_variance,
        )
        candidate = compiled.project_path(kalman_smoother(filter_result, compiled).mean, params)

        accepted = False
        step = 1.0
        candidate_objective = -np.inf
        proposal = mode_path
        while step >= 2.0**-12:
            proposal = mode_path + step * (candidate - mode_path)
            candidate_objective = joint_state_log_density(y, proposal, compiled, params)
            if np.isfinite(candidate_objective) and candidate_objective >= current_objective - 1e-8:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            relative_change = 0.0
            converged = True
            break

        old_eta = eta
        mode_path = proposal
        current_objective = candidate_objective
        new_eta = compiled.eta(mode_path)
        relative_change = float(
            np.max(np.abs(new_eta - old_eta)) / (1.0 + np.max(np.abs(old_eta)))
        )
        if relative_change < tolerance:
            converged = True
            break

    # Refresh the quadratic approximation at the final accepted mode.
    pseudo_y, pseudo_variance = _pseudo_data(
        y,
        compiled.eta(mode_path),
        compiled,
        params,
        curvature_floor=curvature_floor,
        maximum_variance=maximum_variance,
        shift_limit=shift_limit,
    )
    filter_result = kalman_filter(
        pseudo_y,
        compiled,
        params,
        observation_variance=pseudo_variance,
    )
    support_rejections = 0
    path = mode_path.copy()
    for _ in range(int(draw_attempts)):
        proposal, _ = ffbs(
            pseudo_y,
            compiled,
            params,
            rng,
            observation_variance=pseudo_variance,
            filter_result=filter_result,
        )
        if np.isfinite(observation_log_likelihood(y, compiled.eta(proposal), compiled, params)):
            path = proposal
            break
        support_rejections += 1

    return LaplaceDraw(
        path=path,
        mode_path=mode_path,
        pseudo_y=pseudo_y,
        pseudo_variance=pseudo_variance,
        converged=converged,
        iterations=iteration,
        relative_change=relative_change,
        objective=current_objective,
        support_rejections=support_rejections,
        filter=filter_result,
    )
