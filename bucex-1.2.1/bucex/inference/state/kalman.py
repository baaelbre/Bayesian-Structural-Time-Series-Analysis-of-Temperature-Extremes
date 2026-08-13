"""Scalar-observation Kalman filtering, smoothing, and FFBS."""
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


def _observation_variance(value: float | Array, n_time: int) -> Array:
    out = np.asarray(value, dtype=float)
    if out.ndim == 0:
        out = np.full(n_time, float(out))
    out = out.reshape(-1)
    if out.size != n_time:
        raise ValueError("observation_variance must be scalar or length T.")
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
    y = np.asarray(y, dtype=float).reshape(-1)
    n = y.size
    if n != compiled.n_time and exog is None:
        raise ValueError("Use a compiled model matching y, or supply matching exog.")
    h = compiled.design(n, exog=exog)
    variance = _observation_variance(
        float(params["sigma"]) ** 2 if observation_variance is None else observation_variance,
        n,
    )
    f = compiled.transition
    w = compiled.transition_cov(params)
    m = compiled.state_dim

    predicted_mean = np.zeros((n + 1, m))
    predicted_cov = np.zeros((n + 1, m, m))
    filtered_mean = np.zeros((n + 1, m))
    filtered_cov = np.zeros((n + 1, m, m))
    innovations = np.full(n, np.nan)
    innovation_variance = np.full(n, np.nan)
    missing = ~np.isfinite(y)
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
        if missing[t - 1]:
            filtered_mean[t] = a
            filtered_cov[t] = p
            continue
        ht = h[t - 1]
        v = float(y[t - 1] - ht @ a)
        s = float(ht @ p @ ht + variance[t - 1])
        if not np.isfinite(s) or s <= 0.0:
            raise np.linalg.LinAlgError("Non-positive Kalman innovation variance.")
        gain = p @ ht / s
        filtered_mean[t] = a + gain * v
        ikh = identity - np.outer(gain, ht)
        filtered_cov[t] = symmetrize(
            ikh @ p @ ikh.T + variance[t - 1] * np.outer(gain, gain)
        )
        innovations[t - 1] = v
        innovation_variance[t - 1] = s
        log_likelihood += -0.5 * (np.log(2.0 * np.pi * s) + v * v / s)

    return KalmanResult(
        log_likelihood=float(log_likelihood),
        predicted_mean=predicted_mean,
        predicted_cov=predicted_cov,
        filtered_mean=filtered_mean,
        filtered_cov=filtered_cov,
        design=h,
        observation_variance=variance,
        innovations=innovations,
        innovation_variance=innovation_variance,
        missing=missing,
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
