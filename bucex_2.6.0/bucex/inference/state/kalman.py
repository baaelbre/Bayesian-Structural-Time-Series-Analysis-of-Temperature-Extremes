"""Univariate/multichannel Kalman filtering, smoothing, and FFBS."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.numerics import sample_mvn, symmetrize
from ...models.compiler import CompiledModel


Array = np.ndarray


@dataclass
class KalmanResult:
    log_likelihood: float
    predicted_mean: Array
    predicted_cov: Array
    filtered_mean: Array
    filtered_cov: Array
    design: Array
    observation_variance: Array
    innovations: Array
    innovation_variance: Array
    missing: Array


@dataclass
class SmootherResult:
    mean: Array
    covariance: Array
    filter: KalmanResult


def _observation_variance(value: float | Array, n_time: int, n_channels: int) -> Array:
    out = np.asarray(value, dtype=float)
    if out.ndim == 0:
        out = np.full((n_time, n_channels), float(out))
    elif out.ndim == 1:
        if n_channels == 1 and out.size == n_time:
            out = out[:, None]
        elif out.size == n_channels:
            out = np.broadcast_to(out, (n_time, n_channels)).copy()
        elif out.size == n_time and n_channels > 1 and n_time != n_channels:
            out = np.broadcast_to(out[:, None], (n_time, n_channels)).copy()
        else:
            raise ValueError(
                "observation_variance must be scalar, length channels, length T "
                "for a common time-varying variance, or shape (T, channels)."
            )
    elif out.ndim == 2 and out.shape == (n_time, n_channels):
        out = out.copy()
    else:
        raise ValueError(
            "observation_variance must be scalar, a vector, or shape (T, channels)."
        )
    if np.any(out <= 0.0) or not np.all(np.isfinite(out)):
        raise ValueError("All observation variances must be positive and finite.")
    return out


def kalman_filter(
    y: Array,
    compiled: CompiledModel,
    params: dict[str, float],
    *,
    observation_variance: float | Array | None = None,
    exog=None,
) -> KalmanResult:
    raw_y = np.asarray(y, dtype=float)
    scalar_observation = raw_y.ndim <= 1
    y_matrix = raw_y.reshape(-1, 1) if scalar_observation else raw_y
    if y_matrix.ndim != 2:
        raise ValueError("y must be one- or two-dimensional.")
    n, n_channels = y_matrix.shape
    if n != compiled.n_time and exog is None:
        raise ValueError("Use a compiled model matching y, or supply matching exog.")
    h = compiled.design(n, exog=exog, params=params)
    if h.ndim == 2:
        h_matrix = h[:, None, :]
    elif h.ndim == 3:
        h_matrix = h
    else:
        raise ValueError("compiled.design() must have shape (T, m) or (T, p, m).")
    if h_matrix.shape[:2] != (n, n_channels):
        raise ValueError("The observation and compiled design channel dimensions differ.")
    if observation_variance is None:
        default_variance = (
            compiled.observation_variance(params, n)
            if hasattr(compiled, "observation_variance")
            else float(params["sigma"]) ** 2
        )
    else:
        default_variance = observation_variance
    variance = _observation_variance(
        default_variance if observation_variance is None else observation_variance,
        n,
        n_channels,
    )
    f = compiled.transition
    w = compiled.transition_cov(params)
    m = compiled.state_dim

    predicted_mean = np.zeros((n + 1, m))
    predicted_cov = np.zeros((n + 1, m, m))
    filtered_mean = np.zeros((n + 1, m))
    filtered_cov = np.zeros((n + 1, m, m))
    innovations = np.full((n, n_channels), np.nan)
    innovation_variance = np.full((n, n_channels, n_channels), np.nan)
    missing = ~np.isfinite(y_matrix)
    filtered_mean[0] = compiled.initial_mean
    filtered_cov[0] = compiled.initial_cov
    predicted_mean[0] = compiled.initial_mean
    predicted_cov[0] = compiled.initial_cov
    log_likelihood = 0.0
    identity = np.eye(m)

    for t in range(1, n + 1):
        a = f @ filtered_mean[t - 1]
        p = symmetrize(f @ filtered_cov[t - 1] @ f.T + w)
        predicted_mean[t] = a
        predicted_cov[t] = p
        observed = ~missing[t - 1]
        if not np.any(observed):
            filtered_mean[t] = a
            filtered_cov[t] = p
            continue
        ht = h_matrix[t - 1, observed]
        v = y_matrix[t - 1, observed] - ht @ a
        observation_cov = np.diag(variance[t - 1, observed])
        s = symmetrize(ht @ p @ ht.T + observation_cov)
        try:
            chol = np.linalg.cholesky(s)
        except np.linalg.LinAlgError as error:
            raise np.linalg.LinAlgError(
                "Non-positive Kalman innovation covariance."
            ) from error
        gain = np.linalg.solve(s, (p @ ht.T).T).T
        filtered_mean[t] = a + gain @ v
        ikh = identity - gain @ ht
        filtered_cov[t] = symmetrize(
            ikh @ p @ ikh.T + gain @ observation_cov @ gain.T
        )
        innovations[t - 1, observed] = v
        indices = np.flatnonzero(observed)
        innovation_variance[t - 1][np.ix_(indices, indices)] = s
        solved = np.linalg.solve(chol, v)
        logdet = 2.0 * float(np.sum(np.log(np.diag(chol))))
        log_likelihood += -0.5 * (
            observed.sum() * np.log(2.0 * np.pi) + logdet + float(solved @ solved)
        )

    stored_design = h_matrix[:, 0] if scalar_observation else h_matrix
    stored_variance = variance[:, 0] if scalar_observation else variance
    stored_innovations = innovations[:, 0] if scalar_observation else innovations
    stored_innovation_variance = (
        innovation_variance[:, 0, 0] if scalar_observation else innovation_variance
    )
    stored_missing = missing[:, 0] if scalar_observation else missing

    return KalmanResult(
        log_likelihood=float(log_likelihood),
        predicted_mean=predicted_mean,
        predicted_cov=predicted_cov,
        filtered_mean=filtered_mean,
        filtered_cov=filtered_cov,
        design=stored_design,
        observation_variance=stored_variance,
        innovations=stored_innovations,
        innovation_variance=stored_innovation_variance,
        missing=stored_missing,
    )


def kalman_smoother(filter_result: KalmanResult, compiled: CompiledModel) -> SmootherResult:
    n = filter_result.filtered_mean.shape[0] - 1
    mean = filter_result.filtered_mean.copy()
    covariance = filter_result.filtered_cov.copy()
    f = compiled.transition
    for t in range(n - 1, -1, -1):
        predicted = filter_result.predicted_cov[t + 1]
        gain = filter_result.filtered_cov[t] @ f.T @ np.linalg.pinv(predicted, hermitian=True)
        mean[t] = filter_result.filtered_mean[t] + gain @ (
            mean[t + 1] - filter_result.predicted_mean[t + 1]
        )
        covariance[t] = symmetrize(
            filter_result.filtered_cov[t]
            + gain @ (covariance[t + 1] - predicted) @ gain.T
        )
    return SmootherResult(mean=mean, covariance=covariance, filter=filter_result)


def ffbs(
    y: Array,
    compiled: CompiledModel,
    params: dict[str, float],
    rng: np.random.Generator,
    *,
    observation_variance: float | Array | None = None,
    exog=None,
    filter_result: KalmanResult | None = None,
) -> tuple[Array, KalmanResult]:
    result = filter_result or kalman_filter(
        y,
        compiled,
        params,
        observation_variance=observation_variance,
        exog=exog,
    )
    n = result.filtered_mean.shape[0] - 1
    path = np.zeros_like(result.filtered_mean)
    path[n] = sample_mvn(result.filtered_mean[n], result.filtered_cov[n], rng)
    f = compiled.transition
    for t in range(n - 1, -1, -1):
        predicted = result.predicted_cov[t + 1]
        gain = result.filtered_cov[t] @ f.T @ np.linalg.pinv(predicted, hermitian=True)
        conditional_mean = result.filtered_mean[t] + gain @ (
            path[t + 1] - result.predicted_mean[t + 1]
        )
        conditional_cov = symmetrize(
            result.filtered_cov[t] - gain @ predicted @ gain.T
        )
        path[t] = sample_mvn(conditional_mean, conditional_cov, rng)
    return compiled.project_path(path, params), result
