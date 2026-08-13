from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ...core.results import FilterResult, SmootherResult
from ...models.base import LinearDesign, LinearGaussianSystem, StateSpaceModel

Array = np.ndarray
ParamDict = Dict[str, Any]


def _as_2d_y(y: Array) -> Array:
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        return y[:, None]
    if y.ndim == 2:
        return y
    raise ValueError("y must be 1D or 2D.")


def _symmetrize(A: Array) -> Array:
    return 0.5 * (A + A.T)


def _gaussian_loglik(v: Array, S: Array) -> float:
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0:
        raise np.linalg.LinAlgError("Innovation covariance is not positive definite.")
    quad = float(v.T @ np.linalg.solve(S, v))
    p = v.shape[0]
    return float(-0.5 * (p * np.log(2.0 * np.pi) + logdet + quad))


def _resolve_initial_state(
    model: StateSpaceModel,
    params_state: ParamDict,
) -> Tuple[Array, Array]:
    m0, P0 = model.initial_state(params_state)
    m0 = np.asarray(m0, dtype=float).reshape(model.state_dim)
    P0 = np.asarray(P0, dtype=float).reshape(model.state_dim, model.state_dim)
    return m0, _symmetrize(P0)


def _validate_gaussian_model(model: StateSpaceModel) -> None:
    spec = getattr(getattr(model, "obs", None), "spec", None)
    if spec is not None:
        name = getattr(spec, "name", None)
        if name is not None and str(name).lower() != "gaussian":
            raise ValueError(
                f"kalman_filter is exact only for Gaussian observations, got obs='{name}'."
            )


def _resolve_obs_cov(
    t: int,
    p: int,
    params_obs: ParamDict,
    exog_t: Optional[Array] = None,
) -> Array:
    """
    Resolve Gaussian observation covariance H_t from params_obs.

    Supported entries:
      - H
      - sigma2
      - sigma

    Each may be scalar, vector, matrix, or callable returning one of those.
    """
    if "H" in params_obs:
        val = params_obs["H"]
        if callable(val):
            val = val(t=t, exog_t=exog_t, params_obs=params_obs)
        H = np.asarray(val, dtype=float)
        if H.ndim == 0:
            return float(H) * np.eye(p)
        if H.ndim == 1:
            if H.shape[0] != p:
                raise ValueError("1D H must have length p.")
            return np.diag(H)
        if H.ndim == 2:
            return _symmetrize(H)
        raise ValueError("Unsupported H specification.")

    if "sigma2" in params_obs:
        val = params_obs["sigma2"]
        if callable(val):
            val = val(t=t, exog_t=exog_t, params_obs=params_obs)
        sigma2 = np.asarray(val, dtype=float)
        if sigma2.ndim == 0:
            return float(sigma2) * np.eye(p)
        if sigma2.ndim == 1:
            if sigma2.shape[0] != p:
                raise ValueError("1D sigma2 must have length p.")
            return np.diag(sigma2)
        if sigma2.ndim == 2:
            return _symmetrize(sigma2)
        raise ValueError("Unsupported sigma2 specification.")

    if "sigma" in params_obs:
        val = params_obs["sigma"]
        if callable(val):
            val = val(t=t, exog_t=exog_t, params_obs=params_obs)
        sigma = np.asarray(val, dtype=float)
        if sigma.ndim == 0:
            return float(sigma) ** 2 * np.eye(p)
        if sigma.ndim == 1:
            if sigma.shape[0] != p:
                raise ValueError("1D sigma must have length p.")
            return np.diag(np.square(sigma))
        if sigma.ndim == 2:
            return _symmetrize(sigma)
        raise ValueError("Unsupported sigma specification.")

    raise KeyError("params_obs must contain one of: 'H', 'sigma2', or 'sigma'.")


def kalman_filter(
    y: Array,
    model: StateSpaceModel,
    params_state: ParamDict,
    params_obs: ParamDict,
    exog: Optional[Array] = None,
    check_gaussian: bool = True,
) -> FilterResult:
    """
    Exact Kalman filter for linear-Gaussian state-space models.
    """
    if check_gaussian:
        _validate_gaussian_model(model)

    y2 = _as_2d_y(y)
    Tn, p = y2.shape
    m = model.state_dim

    if exog is not None:
        exog = np.asarray(exog, dtype=float)
        if exog.shape[0] != Tn:
            raise ValueError("exog must have the same number of rows as y.")

    m0, P0 = _resolve_initial_state(model, params_state)

    m_pred = np.zeros((Tn + 1, m), dtype=float)
    P_pred = np.zeros((Tn + 1, m, m), dtype=float)
    m_filt = np.zeros((Tn + 1, m), dtype=float)
    P_filt = np.zeros((Tn + 1, m, m), dtype=float)

    innovations = np.full((Tn, p), np.nan, dtype=float)
    innovation_cov = np.full((Tn, p, p), np.nan, dtype=float)
    kalman_gain = np.zeros((Tn, m, p), dtype=float)
    missing = np.zeros(Tn, dtype=bool)

    T_seq = np.zeros((Tn + 1, m, m), dtype=float)
    c_seq = np.zeros((Tn + 1, m), dtype=float)
    Z_seq = np.zeros((Tn + 1, p, m), dtype=float)
    d_seq = np.zeros((Tn + 1, p), dtype=float)
    H_seq = np.zeros((Tn + 1, p, p), dtype=float)

    m_pred[0] = m0
    P_pred[0] = P0
    m_filt[0] = m0
    P_filt[0] = P0

    loglik = 0.0
    I_m = np.eye(m, dtype=float)

    for t in range(1, Tn + 1):
        exog_t = None if exog is None else np.asarray(exog[t - 1], dtype=float)

        sys: LinearGaussianSystem = model.system(t=t, params_state=params_state)
        des: LinearDesign = model.design(t=t, params_state=params_state, exog_t=exog_t)

        T_t = np.asarray(sys.T, dtype=float).reshape(m, m)
        R_t = np.asarray(sys.R, dtype=float)
        Q_t = np.asarray(sys.Q, dtype=float)
        c_t = np.asarray(sys.c, dtype=float).reshape(m)

        Z_t = np.asarray(des.Z, dtype=float).reshape(p, m)
        d_t = np.asarray(des.d, dtype=float).reshape(p)
        H_t = _resolve_obs_cov(t=t, p=p, params_obs=params_obs, exog_t=exog_t)

        T_seq[t] = T_t
        c_seq[t] = c_t
        Z_seq[t] = Z_t
        d_seq[t] = d_t
        H_seq[t] = H_t

        W_t = R_t @ Q_t @ R_t.T
        a_t = T_t @ m_filt[t - 1] + c_t
        P_t = _symmetrize(T_t @ P_filt[t - 1] @ T_t.T + W_t)

        m_pred[t] = a_t
        P_pred[t] = P_t

        y_t = y2[t - 1]
        if np.any(np.isnan(y_t)):
            missing[t - 1] = True
            m_filt[t] = a_t
            P_filt[t] = P_t
            continue

        f_t = Z_t @ a_t + d_t
        v_t = y_t - f_t
        S_t = _symmetrize(Z_t @ P_t @ Z_t.T + H_t)

        K_t = np.linalg.solve(S_t, Z_t @ P_t.T).T

        m_t = a_t + K_t @ v_t

        IKZ = I_m - K_t @ Z_t
        P_t_up = IKZ @ P_t @ IKZ.T + K_t @ H_t @ K_t.T
        P_t_up = _symmetrize(P_t_up)

        innovations[t - 1] = v_t
        innovation_cov[t - 1] = S_t
        kalman_gain[t - 1] = K_t

        m_filt[t] = m_t
        P_filt[t] = P_t_up

        loglik += _gaussian_loglik(v_t, S_t)

    return FilterResult(
        y=y2,
        m0=m0,
        P0=P0,
        m_pred=m_pred,
        P_pred=P_pred,
        m_filt=m_filt,
        P_filt=P_filt,
        loglik=float(loglik),
        innovations=innovations,
        innovation_cov=innovation_cov,
        kalman_gain=kalman_gain,
        missing=missing,
        T_seq=T_seq,
        c_seq=c_seq,
        Z_seq=Z_seq,
        d_seq=d_seq,
        H_seq=H_seq,
        meta={"backend": "kalman", "exact": True},
    )


def kalman_smoother(filter_result: FilterResult) -> SmootherResult:
    """
    Rauch-Tung-Striebel smoother.
    """
    fr = filter_result
    Tn = fr.n_time
    m = fr.state_dim

    m_smooth = np.zeros_like(fr.m_filt)
    P_smooth = np.zeros_like(fr.P_filt)
    J = np.zeros((Tn, m, m), dtype=float)

    m_smooth[Tn] = fr.m_filt[Tn]
    P_smooth[Tn] = fr.P_filt[Tn]

    for t in range(Tn - 1, -1, -1):
        T_next = fr.T_seq[t + 1]
        P_pred_next = fr.P_pred[t + 1]

        J_t = np.linalg.solve(P_pred_next, T_next @ fr.P_filt[t].T).T

        m_smooth[t] = fr.m_filt[t] + J_t @ (m_smooth[t + 1] - fr.m_pred[t + 1])
        P_smooth[t] = _symmetrize(
            fr.P_filt[t] + J_t @ (P_smooth[t + 1] - P_pred_next) @ J_t.T
        )

        J[t] = J_t

    return SmootherResult(
        m_smooth=m_smooth,
        P_smooth=P_smooth,
        smoother_gain=J,
        lag_cov=None,
        filter_result=fr,
        meta={"backend": "kalman_rts", "exact": True},
    )


def filter_and_smooth(
    y: Array,
    model: StateSpaceModel,
    params_state: ParamDict,
    params_obs: ParamDict,
    exog: Optional[Array] = None,
    check_gaussian: bool = True,
) -> Tuple[FilterResult, SmootherResult]:
    fr = kalman_filter(
        y=y,
        model=model,
        params_state=params_state,
        params_obs=params_obs,
        exog=exog,
        check_gaussian=check_gaussian,
    )
    sr = kalman_smoother(fr)
    return fr, sr