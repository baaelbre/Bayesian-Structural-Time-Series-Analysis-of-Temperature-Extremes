from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ...core.results import FilterResult, SmootherResult, StateSample
from ...models.base import StateSpaceModel
from .ffbs import ffbs_sample
from .kalman import filter_and_smooth

Array = np.ndarray
ParamDict = Dict[str, Any]


def _as_1d_y(y: Array) -> Array:
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        return y
    if y.ndim == 2 and y.shape[1] == 1:
        return y[:, 0]
    raise ValueError("laplace.py currently supports only univariate observations.")


def _reconstruct_eta_path(
    x_path: Array,
    model: StateSpaceModel,
    params_state: ParamDict,
    exog: Optional[Array] = None,
) -> Array:
    """
    Recompute eta_t = Z_t x_t + d_t for a state path x_0:T.

    Returns
    -------
    eta : array
        Shape (T,).
    """
    x_path = np.asarray(x_path, dtype=float)
    Tn = x_path.shape[0] - 1

    if exog is not None:
        exog = np.asarray(exog, dtype=float)
        if exog.shape[0] != Tn:
            raise ValueError("exog must have shape (T, k).")

    eta = np.zeros(Tn, dtype=float)

    for t in range(1, Tn + 1):
        exog_t = None if exog is None else exog[t - 1]
        des = model.design(t=t, params_state=params_state, exog_t=exog_t)
        eta_t = des.Z @ x_path[t] + des.d
        eta[t - 1] = float(np.atleast_1d(eta_t)[0])

    return eta


def _obs_param_dict_from_eta(
    t: int,
    eta: float,
    model: StateSpaceModel,
    params_obs: ParamDict,
    exog_t: Optional[Array] = None,
) -> ParamDict:
    """
    Build the observation-parameter dictionary for a given scalar eta_t.

    Notes
    -----
    This uses a dummy x_t because in the current design `obs_params(...)`
    is expected to depend on eta_t and static observation parameters.
    """
    x_dummy = np.zeros(model.state_dim, dtype=float)
    eta_arr = np.asarray([eta], dtype=float)
    return model.obs_params(
        t=t,
        x_t=x_dummy,
        eta_t=eta_arr,
        params_obs=params_obs,
        exog_t=exog_t,
    )


def _numeric_grad_hess_eta(
    y_t: float,
    eta_t: float,
    t: int,
    model: StateSpaceModel,
    params_obs: ParamDict,
    exog_t: Optional[Array],
    fd_eps: float,
) -> Tuple[float, float]:
    """
    Finite-difference gradient and Hessian of log p(y_t | eta_t, params).

    Central differences:
      f'(x)  ≈ [f(x+h) - f(x-h)] / (2h)
      f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h^2
    """
    h = fd_eps * (1.0 + abs(float(eta_t)))

    def f(eta_val: float) -> float:
        op = _obs_param_dict_from_eta(
            t=t,
            eta=eta_val,
            model=model,
            params_obs=params_obs,
            exog_t=exog_t,
        )
        return float(model.obs.logpdf(y=float(y_t), eta=float(eta_val), params=op))

    f0 = f(eta_t)
    fp = f(eta_t + h)
    fm = f(eta_t - h)

    grad = (fp - fm) / (2.0 * h)
    hess = (fp - 2.0 * f0 + fm) / (h * h)

    return float(grad), float(hess)


def _grad_hess_eta(
    y_t: float,
    eta_t: float,
    t: int,
    model: StateSpaceModel,
    params_obs: ParamDict,
    exog_t: Optional[Array],
    fd_eps: float,
) -> Tuple[float, float]:
    """
    Use analytic grad/hess if available; otherwise fall back to finite differences.
    """
    op = _obs_param_dict_from_eta(
        t=t,
        eta=eta_t,
        model=model,
        params_obs=params_obs,
        exog_t=exog_t,
    )

    # Try analytic derivatives first
    try:
        grad = float(model.obs.grad_eta(y=float(y_t), eta=float(eta_t), params=op))
        hess = float(model.obs.hess_eta(y=float(y_t), eta=float(eta_t), params=op))
        if np.isfinite(grad) and np.isfinite(hess):
            return grad, hess
    except Exception:
        pass

    # Fall back to finite differences
    return _numeric_grad_hess_eta(
        y_t=y_t,
        eta_t=eta_t,
        t=t,
        model=model,
        params_obs=params_obs,
        exog_t=exog_t,
        fd_eps=fd_eps,
    )


def _pseudo_observations(
    y: Array,
    eta_star: Array,
    model: StateSpaceModel,
    params_obs: ParamDict,
    exog: Optional[Array] = None,
    fd_eps: float = 1e-5,
    curvature_floor: float = 1e-8,
) -> Tuple[Array, Array]:
    """
    Build Laplace pseudo-observations z_t and pseudo-variances R_t.

    For log-likelihood l_t(eta), expanded at eta*:
      l_t(eta) ≈ const + g_t (eta - eta*) - 0.5 * I_t (eta - eta*)^2

    with:
      g_t = l'_t(eta*)
      I_t = -l''_t(eta*)   (observed information)

    Match to a Gaussian pseudo-observation:
      z_t | eta_t ~ N(eta_t, R_t)

    giving:
      R_t = 1 / I_t
      z_t = eta*_t + g_t / I_t
    """
    y = np.asarray(y, dtype=float)
    eta_star = np.asarray(eta_star, dtype=float)

    Tn = y.size
    if eta_star.shape != (Tn,):
        raise ValueError("eta_star must have shape (T,).")

    if exog is not None:
        exog = np.asarray(exog, dtype=float)
        if exog.shape[0] != Tn:
            raise ValueError("exog must have shape (T, k).")

    z = np.zeros(Tn, dtype=float)
    R = np.zeros(Tn, dtype=float)

    for t in range(1, Tn + 1):
        exog_t = None if exog is None else exog[t - 1]

        grad, hess = _grad_hess_eta(
            y_t=float(y[t - 1]),
            eta_t=float(eta_star[t - 1]),
            t=t,
            model=model,
            params_obs=params_obs,
            exog_t=exog_t,
            fd_eps=fd_eps,
        )

        # We need positive observed information
        info = max(-float(hess), float(curvature_floor))
        R_t = 1.0 / info
        z_t = float(eta_star[t - 1]) + float(grad) / info

        z[t - 1] = z_t
        R[t - 1] = R_t

    return z, R


def _run_laplace_iterations(
    y: Array,
    model: StateSpaceModel,
    params_state: ParamDict,
    params_obs: ParamDict,
    exog: Optional[Array] = None,
    init_eta: Optional[Array] = None,
    max_iter: int = 20,
    tol: float = 1e-4,
    fd_eps: float = 1e-5,
    curvature_floor: float = 1e-8,
) -> Tuple[FilterResult, SmootherResult, Array, Array, bool, int]:
    """
    Core Laplace iteration:

      1. start from eta*
      2. build pseudo-observations (z, R)
      3. run Gaussian filter/smoother on pseudo model
      4. update eta* using smoothed eta
      5. iterate to convergence

    Returns
    -------
    fr, sr, z, R, converged, n_iter
    """
    y1 = _as_1d_y(y)

    if np.any(np.isnan(y1)):
        raise NotImplementedError("laplace.py currently does not support missing observations.")

    Tn = y1.size

    if exog is not None:
        exog = np.asarray(exog, dtype=float)
        if exog.shape[0] != Tn:
            raise ValueError("exog must have shape (T, k).")

    # Initial expansion point
    if init_eta is None:
        y_mean = float(np.mean(y1))
        eta_star = np.where(np.isfinite(y1), y1, y_mean).astype(float)
    else:
        eta_star = np.asarray(init_eta, dtype=float).reshape(Tn)

    fr_last: Optional[FilterResult] = None
    sr_last: Optional[SmootherResult] = None
    z_last = np.zeros(Tn, dtype=float)
    R_last = np.ones(Tn, dtype=float)

    converged = False

    for it in range(1, max_iter + 1):
        z_t, R_t = _pseudo_observations(
            y=y1,
            eta_star=eta_star,
            model=model,
            params_obs=params_obs,
            exog=exog,
            fd_eps=fd_eps,
            curvature_floor=curvature_floor,
        )

        # Gaussian pseudo model: z_t = eta_t + eps_t, eps_t ~ N(0, R_t)
        pseudo_obs = {
            "H": (lambda R_arr: (lambda t, exog_t=None, params_obs=None: float(R_arr[t - 1])))(R_t)
        }

        fr, sr = filter_and_smooth(
            y=z_t,
            model=model,
            params_state=params_state,
            params_obs=pseudo_obs,
            exog=exog,
            check_gaussian=False,
        )

        eta_new = _reconstruct_eta_path(
            x_path=sr.m_smooth,
            model=model,
            params_state=params_state,
            exog=exog,
        )

        delta = float(np.max(np.abs(eta_new - eta_star)))

        fr_last = fr
        sr_last = sr
        z_last = z_t
        R_last = R_t

        eta_star = eta_new

        if delta < tol:
            converged = True
            break

    assert fr_last is not None
    assert sr_last is not None

    # Enrich metadata
    fr_last.meta = {
        **fr_last.meta,
        "backend": "laplace_filter",
        "exact": False,
        "pseudo_y": z_last,
        "pseudo_var": R_last,
        "mode_eta": eta_star,
        "converged": converged,
        "n_iter": it,
        "tol": tol,
    }

    sr_last.meta = {
        **sr_last.meta,
        "backend": "laplace_smoother",
        "exact": False,
        "pseudo_y": z_last,
        "pseudo_var": R_last,
        "mode_eta": eta_star,
        "converged": converged,
        "n_iter": it,
        "tol": tol,
    }

    return fr_last, sr_last, z_last, R_last, converged, it


def laplace_filter(
    y: Array,
    model: StateSpaceModel,
    params_state: ParamDict,
    params_obs: ParamDict,
    exog: Optional[Array] = None,
    init_eta: Optional[Array] = None,
    max_iter: int = 20,
    tol: float = 1e-4,
    fd_eps: float = 1e-5,
    curvature_floor: float = 1e-8,
) -> FilterResult:
    """
    Laplace-based approximate filter for non-Gaussian observation models.

    Current assumptions
    -------------------
    - univariate observations
    - scalar eta_t
    - no missing observations
    """
    fr, _, _, _, _, _ = _run_laplace_iterations(
        y=y,
        model=model,
        params_state=params_state,
        params_obs=params_obs,
        exog=exog,
        init_eta=init_eta,
        max_iter=max_iter,
        tol=tol,
        fd_eps=fd_eps,
        curvature_floor=curvature_floor,
    )
    return fr


def laplace_smoother(
    y: Array,
    model: StateSpaceModel,
    params_state: ParamDict,
    params_obs: ParamDict,
    exog: Optional[Array] = None,
    init_eta: Optional[Array] = None,
    max_iter: int = 20,
    tol: float = 1e-4,
    fd_eps: float = 1e-5,
    curvature_floor: float = 1e-8,
) -> SmootherResult:
    """
    Laplace-based approximate smoother for non-Gaussian observation models.
    """
    _, sr, _, _, _, _ = _run_laplace_iterations(
        y=y,
        model=model,
        params_state=params_state,
        params_obs=params_obs,
        exog=exog,
        init_eta=init_eta,
        max_iter=max_iter,
        tol=tol,
        fd_eps=fd_eps,
        curvature_floor=curvature_floor,
    )
    return sr


def laplace_ffbs(
    y: Array,
    model: StateSpaceModel,
    params_state: ParamDict,
    params_obs: ParamDict,
    exog: Optional[Array] = None,
    rng: Optional[np.random.Generator] = None,
    init_eta: Optional[Array] = None,
    max_iter: int = 20,
    tol: float = 1e-4,
    fd_eps: float = 1e-5,
    curvature_floor: float = 1e-8,
) -> StateSample:
    """
    Laplace-based approximate state draw.

    Procedure
    ---------
    1. run Laplace iterations to obtain a pseudo-Gaussian approximation
    2. run exact Gaussian FFBS on that pseudo model
    """
    fr, _, z_t, R_t, converged, n_iter = _run_laplace_iterations(
        y=y,
        model=model,
        params_state=params_state,
        params_obs=params_obs,
        exog=exog,
        init_eta=init_eta,
        max_iter=max_iter,
        tol=tol,
        fd_eps=fd_eps,
        curvature_floor=curvature_floor,
    )

    # ffbs_sample ignores y if filter_result is supplied, but we pass z_t for clarity.
    draw = ffbs_sample(
        y=z_t,
        model=model,
        params_state=params_state,
        params_obs={"H": (lambda R_arr: (lambda t, exog_t=None, params_obs=None: float(R_arr[t - 1])))(R_t)},
        exog=exog,
        rng=rng,
        filter_result=fr,
        check_gaussian=False,
    )

    draw.meta = {
        **draw.meta,
        "backend": "laplace_ffbs",
        "exact": False,
        "converged": converged,
        "n_iter": n_iter,
        "pseudo_y": z_t,
        "pseudo_var": R_t,
        "mode_eta": fr.meta.get("mode_eta", None),
    }

    return draw