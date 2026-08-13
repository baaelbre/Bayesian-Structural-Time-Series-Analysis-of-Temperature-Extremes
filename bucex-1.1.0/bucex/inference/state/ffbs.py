from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ...core.results import FilterResult, StateSample
from ...models.base import StateSpaceModel
from .kalman import kalman_filter

Array = np.ndarray
ParamDict = Dict[str, Any]


def _symmetrize(A: Array) -> Array:
    return 0.5 * (A + A.T)


def _sample_gaussian(
    mean: Array,
    cov: Array,
    rng: np.random.Generator,
    jitter: float = 1e-10,
    max_tries: int = 6,
) -> Array:
    mean = np.asarray(mean, dtype=float)
    cov = _symmetrize(np.asarray(cov, dtype=float))
    m = mean.shape[0]

    for k in range(max_tries):
        try:
            return rng.multivariate_normal(mean=mean, cov=cov, check_valid="raise")
        except Exception:
            cov = _symmetrize(cov + (10.0**k) * jitter * np.eye(m))
    raise np.linalg.LinAlgError("Failed to sample from Gaussian with jittered covariance.")


def ffbs_sample(
    y: Array,
    model: StateSpaceModel,
    params_state: ParamDict,
    params_obs: ParamDict,
    exog: Optional[Array] = None,
    rng: Optional[np.random.Generator] = None,
    filter_result: Optional[FilterResult] = None,
    check_gaussian: bool = True,
) -> StateSample:
    """
    Exact forward-filtering backward-sampling for linear-Gaussian models.
    """
    rng = rng if rng is not None else np.random.default_rng()

    if filter_result is None:
        filter_result = kalman_filter(
            y=y,
            model=model,
            params_state=params_state,
            params_obs=params_obs,
            exog=exog,
            check_gaussian=check_gaussian,
        )

    fr = filter_result
    Tn = fr.n_time
    m = fr.state_dim

    x = np.zeros((Tn + 1, m), dtype=float)

    x[Tn] = _sample_gaussian(fr.m_filt[Tn], fr.P_filt[Tn], rng)

    for t in range(Tn - 1, -1, -1):
        T_next = fr.T_seq[t + 1]
        P_pred_next = fr.P_pred[t + 1]

        J_t = np.linalg.solve(P_pred_next, T_next @ fr.P_filt[t].T).T

        mean_t = fr.m_filt[t] + J_t @ (x[t + 1] - fr.m_pred[t + 1])
        cov_t = _symmetrize(fr.P_filt[t] - J_t @ P_pred_next @ J_t.T)

        x[t] = _sample_gaussian(mean_t, cov_t, rng)

    return StateSample(
        x=x,
        filter_result=fr,
        smoother_result=None,
        meta={"backend": "ffbs", "exact": True},
    )


def ffbs_samples(
    n_samples: int,
    y: Array,
    model: StateSpaceModel,
    params_state: ParamDict,
    params_obs: ParamDict,
    exog: Optional[Array] = None,
    rng: Optional[np.random.Generator] = None,
    filter_result: Optional[FilterResult] = None,
    check_gaussian: bool = True,
) -> Array:
    """
    Draw multiple FFBS trajectories. Returns shape (n_samples, T+1, m).
    """
    rng = rng if rng is not None else np.random.default_rng()

    if filter_result is None:
        filter_result = kalman_filter(
            y=y,
            model=model,
            params_state=params_state,
            params_obs=params_obs,
            exog=exog,
            check_gaussian=check_gaussian,
        )

    fr = filter_result
    out = np.zeros((n_samples, fr.n_time + 1, fr.state_dim), dtype=float)

    for i in range(n_samples):
        out[i] = ffbs_sample(
            y=y,
            model=model,
            params_state=params_state,
            params_obs=params_obs,
            exog=exog,
            rng=rng,
            filter_result=fr,
            check_gaussian=check_gaussian,
        ).x

    return out