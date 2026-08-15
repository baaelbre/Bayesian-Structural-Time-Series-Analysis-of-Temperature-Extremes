from __future__ import annotations

from dataclasses import dataclass
from math import lgamma
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ...components.regression import RegressionComponent
from ...components.seasonal import DummySeasonal
from ...components.trend import LocalLinearTrend
from ...models.base import StateSpaceModel
from ...models.structural import StructuralSSM
from ...core.numerics import symmetrize as _symmetrize

Array = np.ndarray
ParamDict = Dict[str, Any]




@dataclass(frozen=True)
class NCPLayout:
    centered_state_names: tuple[str, ...]
    centered_state_dim: int
    has_alpha: bool
    has_beta: bool
    season_dim: int
    idx_alpha: Optional[int]
    idx_beta: Optional[int]
    season_slice: slice
    idx_tilde_alpha: Optional[int]
    idx_tilde_beta: Optional[int]
    idx_A: Optional[int]
    season_ncp_slice: slice
    ncp_state_names: tuple[str, ...]
    ncp_state_dim: int


def _project_psd(A: Array, floor: float = 1e-12) -> Array:
    """Return a symmetric positive-semidefinite numerical projection.

    Kalman and smoothing covariance updates can acquire tiny negative
    eigenvalues from floating-point cancellation. We repair only the numerical
    part: eigenvalues below a scale-aware floor are clipped before sampling.
    """

    A = _symmetrize(np.asarray(A, dtype=float))
    if A.size == 0:
        return A
    values, vectors = np.linalg.eigh(A)
    scale = max(1.0, float(np.max(np.abs(values))))
    min_value = max(float(floor), 100.0 * np.finfo(float).eps * scale)
    values = np.maximum(values, min_value)
    out = (vectors * values) @ vectors.T
    return _symmetrize(out)


def spd_solve(A: Array, B: Array, jitter: float = 1e-12) -> Array:
    A = _symmetrize(np.asarray(A, dtype=float))
    B = np.asarray(B, dtype=float)
    n = A.shape[0]
    try:
        L = np.linalg.cholesky(A)
        Y = np.linalg.solve(L, B)
        return np.linalg.solve(L.T, Y)
    except np.linalg.LinAlgError:
        pass
    values = np.linalg.eigvalsh(A)
    scale = max(float(np.max(np.abs(values))), 1.0)
    if np.any(values < -1e-6 * scale):
        raise np.linalg.LinAlgError("Matrix is materially indefinite.")
    return np.linalg.pinv(A, rcond=1e-12, hermitian=True) @ B


def _sample_gaussian(
    mean: Array,
    cov: Array,
    rng: np.random.Generator,
    jitter: float = 1e-12,
) -> Array:
    """Sample from a Gaussian after deterministic PSD repair.

    This avoids NumPy's warning-based fallback, which can silently alter an
    indefinite covariance matrix.
    """

    mean = np.asarray(mean, dtype=float)
    covariance = _symmetrize(np.asarray(cov, dtype=float))
    values, vectors = np.linalg.eigh(covariance)
    scale = max(float(np.max(np.abs(values))) if values.size else 0.0, 1.0)
    if np.any(values < -1e-6 * scale):
        raise np.linalg.LinAlgError("Gaussian covariance is materially indefinite.")
    values = np.clip(values, 0.0, None)
    return mean + vectors @ (np.sqrt(values) * rng.normal(size=mean.size))


def _ess(w: Array) -> float:
    s2 = float(np.sum(np.square(np.asarray(w, dtype=float))))
    if not np.isfinite(s2) or s2 <= 0.0:
        return 0.0
    return float(1.0 / s2)


def validate_ncp_model(model: StateSpaceModel) -> None:
    """Validate that the model matches the specialized non-centred sampler scope.

    Supported structure:
      - one LocalLinearTrend with dynamic level and trend in {dynamic, off}
      - at most one DummySeasonal with mode in {dynamic, off}
      - no regression component
    """
    if not isinstance(model, StructuralSSM):
        raise TypeError("Non-centred fitters require a StructuralSSM model.")

    trend_blocks = [c for c in model.components if isinstance(c, LocalLinearTrend)]
    season_blocks = [c for c in model.components if isinstance(c, DummySeasonal)]
    reg_blocks = [c for c in model.components if isinstance(c, RegressionComponent)]
    other = [c for c in model.components if not isinstance(c, (LocalLinearTrend, DummySeasonal, RegressionComponent))]

    if len(trend_blocks) != 1:
        raise NotImplementedError("Non-centred fitters currently require exactly one LocalLinearTrend component.")
    if len(season_blocks) > 1:
        raise NotImplementedError("Non-centred fitters support at most one DummySeasonal component.")
    if reg_blocks:
        raise NotImplementedError("Non-centred fitters do not support RegressionComponent yet.")
    if other:
        raise NotImplementedError("Non-centred fitters support only LocalLinearTrend plus optional DummySeasonal in the current implementation.")

    trend = trend_blocks[0]
    if trend.level_mode != "dynamic":
        raise NotImplementedError("Non-centred fitters require a dynamic level state.")
    if trend.trend_mode not in {"dynamic", "off"}:
        raise NotImplementedError("Non-centred fitters support trend_mode in {'dynamic', 'off'} only.")

    if season_blocks and season_blocks[0].mode not in {"dynamic", "off"}:
        raise NotImplementedError("Non-centred fitters support DummySeasonal mode in {'dynamic', 'off'} only.")


def infer_ncp_layout(model: StateSpaceModel) -> NCPLayout:
    validate_ncp_model(model)

    names = tuple(str(nm) for nm in model.state_names)
    trend = next(c for c in model.components if isinstance(c, LocalLinearTrend))
    seasonal = next(
        (c for c in model.components if isinstance(c, DummySeasonal) and c.mode != "off"),
        None,
    )
    idx_alpha = names.index(trend.level_name)
    idx_beta = (
        names.index(trend.slope_name)
        if trend.trend_mode != "off" and trend.slope_name in names
        else None
    )
    seasonal_prefix = None if seasonal is None else f"{seasonal.name}["
    g_idx = [
        i
        for i, name in enumerate(names)
        if seasonal_prefix is not None and name.startswith(seasonal_prefix)
    ]
    season_slice = slice(min(g_idx), max(g_idx) + 1) if g_idx else slice(0, 0)
    season_dim = len(g_idx)

    z_names: list[str] = []
    idx_tilde_alpha: Optional[int] = None
    idx_tilde_beta: Optional[int] = None
    idx_A: Optional[int] = None

    if idx_alpha is not None:
        idx_tilde_alpha = len(z_names)
        z_names.append("tilde_alpha")
    if idx_beta is not None:
        idx_tilde_beta = len(z_names)
        z_names.append("tilde_beta")
        idx_A = len(z_names)
        z_names.append("A")

    season_ncp_start = len(z_names)
    for k in range(season_dim):
        z_names.append(f"tilde_g{k+1}")
    season_ncp_slice = slice(season_ncp_start, season_ncp_start + season_dim)

    if idx_alpha is None:
        raise NotImplementedError("Non-centred samplers require a dynamic level state.")

    return NCPLayout(
        centered_state_names=names,
        centered_state_dim=len(names),
        has_alpha=idx_alpha is not None,
        has_beta=idx_beta is not None,
        season_dim=season_dim,
        idx_alpha=idx_alpha,
        idx_beta=idx_beta,
        season_slice=season_slice,
        idx_tilde_alpha=idx_tilde_alpha,
        idx_tilde_beta=idx_tilde_beta,
        idx_A=idx_A,
        season_ncp_slice=season_ncp_slice,
        ncp_state_names=tuple(z_names),
        ncp_state_dim=len(z_names),
    )


def _get_scalar(params_state: ParamDict, key: str, fallback: Optional[str] = None, default: float = 0.0) -> float:
    if key in params_state:
        return float(params_state[key])
    if fallback is not None and fallback in params_state:
        return float(params_state[fallback])
    return float(default)


def canonicalize_ncp_params(params_state: ParamDict, layout: NCPLayout) -> ParamDict:
    out = dict(params_state)
    out.setdefault("alpha0", _get_scalar(out, "alpha0", fallback="m0_level", default=0.0))
    out.setdefault("beta0", _get_scalar(out, "beta0", fallback="m0_trend", default=0.0))

    if layout.season_dim > 0:
        if "gamma0_season" in out:
            g0 = np.asarray(out["gamma0_season"], dtype=float).reshape(-1)
        elif "m0_season" in out:
            g0 = np.asarray(out["m0_season"], dtype=float).reshape(-1)
        else:
            g0 = np.zeros(layout.season_dim, dtype=float)
        if g0.size != layout.season_dim:
            raise ValueError(f"gamma0_season / m0_season must have length {layout.season_dim}.")
        out["gamma0_season"] = g0.copy()

    if "s_level" not in out:
        out["s_level"] = float(np.sqrt(max(_get_scalar(out, "q_level", default=0.0), 0.0)))
    if layout.has_beta and "s_trend" not in out:
        out["s_trend"] = float(np.sqrt(max(_get_scalar(out, "q_trend", default=0.0), 0.0)))
    if layout.season_dim > 0 and "s_season" not in out:
        out["s_season"] = float(np.sqrt(max(_get_scalar(out, "q_season", default=0.0), 0.0)))

    if not layout.has_beta:
        out.setdefault("s_trend", 0.0)
    if layout.season_dim == 0:
        out.setdefault("s_season", 0.0)

    out["q_level"] = float(out["s_level"]) ** 2
    out["q_trend"] = float(out.get("s_trend", 0.0)) ** 2
    out["q_season"] = float(out.get("s_season", 0.0)) ** 2
    return out


def seasonal_rotation_matrix(K: int) -> Array:
    if K <= 0:
        return np.zeros((0, 0), dtype=float)
    S = np.zeros((K, K), dtype=float)
    S[0, :] = -1.0
    if K > 1:
        S[1:, :-1] = np.eye(K - 1)
    return S


def build_ncp_system(layout: NCPLayout) -> tuple[Array, Array]:
    d = layout.ncp_state_dim
    G = np.zeros((d, d), dtype=float)
    Q = np.zeros((d, d), dtype=float)

    if layout.idx_tilde_alpha is not None:
        i = layout.idx_tilde_alpha
        G[i, i] = 1.0
        Q[i, i] = 1.0

    if layout.idx_tilde_beta is not None:
        ib = layout.idx_tilde_beta
        iA = int(layout.idx_A)
        G[ib, ib] = 1.0
        G[iA, iA] = 1.0
        G[iA, ib] = 1.0
        Q[ib, ib] = 1.0

    if layout.season_dim > 0:
        gs = layout.season_ncp_slice.start
        ge = layout.season_ncp_slice.stop
        G[gs:ge, gs:ge] = seasonal_rotation_matrix(layout.season_dim)
        Q[gs, gs] = 1.0

    return G, Q


def baseline_mu_path(Tn: int, params_state: ParamDict, layout: NCPLayout) -> Array:
    t1 = np.arange(1, Tn + 1, dtype=float)
    mu = np.full(Tn, float(params_state.get("alpha0", 0.0)), dtype=float)
    if layout.has_beta:
        mu += float(params_state.get("beta0", 0.0)) * t1

    if layout.season_dim > 0:
        g1 = np.asarray(params_state["gamma0_season"], dtype=float).reshape(layout.season_dim)
        S = seasonal_rotation_matrix(layout.season_dim)
        g = g1.copy()
        for t in range(1, Tn + 1):
            if t > 1:
                g = S @ g
            mu[t - 1] += float(g[0])
    return mu


def baseline_centered_season_path(Tn: int, params_state: ParamDict, layout: NCPLayout) -> Array:
    if layout.season_dim == 0:
        return np.zeros((Tn + 1, 0), dtype=float)

    g1 = np.asarray(params_state["gamma0_season"], dtype=float).reshape(layout.season_dim)
    S = seasonal_rotation_matrix(layout.season_dim)
    out = np.zeros((Tn + 1, layout.season_dim), dtype=float)

    if Tn >= 1:
        out[1] = g1
    if Tn >= 0:
        out[0] = np.linalg.solve(S, g1)
    for t in range(2, Tn + 1):
        out[t] = S @ out[t - 1]
    return out


def measurement_vector(params_state: ParamDict, layout: NCPLayout) -> Array:
    H = np.zeros(layout.ncp_state_dim, dtype=float)
    if layout.idx_tilde_alpha is not None:
        H[layout.idx_tilde_alpha] = float(params_state.get("s_level", 0.0))
    if layout.idx_A is not None:
        H[layout.idx_A] = float(params_state.get("s_trend", 0.0))
    if layout.season_dim > 0:
        H[layout.season_ncp_slice.start] = float(params_state.get("s_season", 0.0))
    return H


def map_ncp_to_centered(z_path: Array, params_state: ParamDict, layout: NCPLayout) -> Array:
    z_path = np.asarray(z_path, dtype=float)
    Tn = z_path.shape[0] - 1
    x = np.zeros((Tn + 1, layout.centered_state_dim), dtype=float)

    t0 = np.arange(0, Tn + 1, dtype=float)
    alpha0 = float(params_state.get("alpha0", 0.0))
    beta0 = float(params_state.get("beta0", 0.0))
    s_level = float(params_state.get("s_level", 0.0))
    s_trend = float(params_state.get("s_trend", 0.0))
    s_season = float(params_state.get("s_season", 0.0))

    alpha = alpha0 + beta0 * t0 + s_level * z_path[:, int(layout.idx_tilde_alpha)]
    if layout.has_beta:
        alpha = alpha + s_trend * z_path[:, int(layout.idx_A)]
        beta = beta0 + s_trend * z_path[:, int(layout.idx_tilde_beta)]
        x[:, int(layout.idx_beta)] = beta

    x[:, int(layout.idx_alpha)] = alpha

    if layout.season_dim > 0:
        g_base = baseline_centered_season_path(Tn, params_state, layout)
        g = g_base + s_season * z_path[:, layout.season_ncp_slice]
        x[:, layout.season_slice] = g

    return x


def map_centered_to_ncp(
    x_path: Array,
    params_state: ParamDict,
    layout: NCPLayout,
    *,
    tolerance: float = 1e-7,
) -> Array:
    """Inverse of :func:`map_ncp_to_centered` for ASIS interweaving."""
    x = np.asarray(x_path, dtype=float)
    if x.ndim != 2 or x.shape[1] != layout.centered_state_dim:
        raise ValueError("x_path has the wrong centred-state shape.")
    Tn = x.shape[0] - 1
    z = np.zeros((Tn + 1, layout.ncp_state_dim), dtype=float)
    time = np.arange(Tn + 1, dtype=float)
    alpha0 = float(params_state.get("alpha0", 0.0))
    beta0 = float(params_state.get("beta0", 0.0))

    if layout.has_beta:
        s_trend = float(params_state.get("s_trend", 0.0))
        beta_deviation = x[:, int(layout.idx_beta)] - beta0
        if abs(s_trend) <= 1e-14:
            if np.max(np.abs(beta_deviation)) > tolerance:
                raise ValueError("A dynamic slope path cannot be represented with s_trend=0.")
            tilde_beta = np.zeros(Tn + 1, dtype=float)
        else:
            tilde_beta = beta_deviation / s_trend
        z[:, int(layout.idx_tilde_beta)] = tilde_beta
        integrated = np.zeros(Tn + 1, dtype=float)
        for t in range(1, Tn + 1):
            integrated[t] = integrated[t - 1] + tilde_beta[t - 1]
        z[:, int(layout.idx_A)] = integrated
    else:
        integrated = np.zeros(Tn + 1, dtype=float)

    alpha_deviation = (
        x[:, int(layout.idx_alpha)]
        - alpha0
        - beta0 * time
        - float(params_state.get("s_trend", 0.0)) * integrated
    )
    s_level = float(params_state.get("s_level", 0.0))
    if abs(s_level) <= 1e-14:
        if np.max(np.abs(alpha_deviation)) > tolerance:
            raise ValueError("A dynamic level path cannot be represented with s_level=0.")
    else:
        z[:, int(layout.idx_tilde_alpha)] = alpha_deviation / s_level

    if layout.season_dim > 0:
        baseline = baseline_centered_season_path(Tn, params_state, layout)
        seasonal_deviation = x[:, layout.season_slice] - baseline
        s_season = float(params_state.get("s_season", 0.0))
        if abs(s_season) <= 1e-14:
            if np.max(np.abs(seasonal_deviation)) > tolerance:
                raise ValueError("A dynamic seasonal path cannot be represented with s_season=0.")
        else:
            z[:, layout.season_ncp_slice] = seasonal_deviation / s_season

    reconstruction = map_ncp_to_centered(z, params_state, layout)
    error = np.max(np.abs(reconstruction - x))
    if error > tolerance * (1.0 + np.max(np.abs(x))):
        raise FloatingPointError("Centred/FS round-trip failed during ASIS.")
    return z


def _centered_innovations_by_component(
    x_path: Array,
    layout: NCPLayout,
) -> dict[str, Array]:
    x = np.asarray(x_path, dtype=float)
    innovations: dict[str, Array] = {}
    alpha = x[:, int(layout.idx_alpha)]
    if layout.has_beta:
        beta = x[:, int(layout.idx_beta)]
        innovations["level"] = alpha[1:] - alpha[:-1] - beta[:-1]
        innovations["trend"] = beta[1:] - beta[:-1]
    else:
        innovations["level"] = alpha[1:] - alpha[:-1]
    if layout.season_dim > 0:
        seasonal = x[:, layout.season_slice]
        rotation = seasonal_rotation_matrix(layout.season_dim)
        residual = seasonal[1:] - seasonal[:-1] @ rotation.T
        innovations["season"] = residual[:, 0]
    return innovations


def _signed_scale_prior_logpdf(
    value: float,
    component: str,
    priors: Any,
    *,
    tau: Optional[dict[str, float]],
    lasso_variance_scale: float,
    horseshoe_state: Optional[dict[str, Any]],
    triple_gamma_state: Optional[dict[str, Any]],
) -> float:
    if getattr(priors, "lasso", None) is not None:
        if tau is None or component not in tau:
            raise ValueError(f"Missing Bayesian-lasso tau for {component}.")
        coefficient_scale = lasso_coefficient_scale(priors.lasso, component)
        variance = (
            float(lasso_variance_scale)
            * coefficient_scale**2
            * float(tau[component])
        )
        return _normal_zero_logpdf(value, variance)
    if getattr(priors, "pc", None) is not None:
        if tau is None or component not in tau:
            raise ValueError(f"Missing PC mixture scale for {component}.")
        coefficient_scale = priors.pc.coefficient_scale_for(component)
        return _normal_zero_logpdf(
            value, coefficient_scale**2 * float(tau[component])
        )
    if getattr(priors, "horseshoe", None) is not None:
        if horseshoe_state is None:
            raise ValueError("Missing regularized-horseshoe state.")
        return _normal_zero_logpdf(
            value,
            horseshoe_conditional_variance(priors, horseshoe_state, component),
        )
    if getattr(priors, "triple_gamma", None) is not None:
        if triple_gamma_state is None:
            raise ValueError("Missing triple-gamma state.")
        return _normal_zero_logpdf(
            value,
            triple_gamma_conditional_variance(
                priors, triple_gamma_state, component
            ),
        )
    key = {"level": "s_level", "trend": "s_trend", "season": "s_season"}[component]
    prior = getattr(priors, key)
    if prior is None:
        raise ValueError(f"Missing signed Normal prior for {key}.")
    z_value = (float(value) - float(prior.mean)) / float(prior.sd)
    return float(
        -0.5 * z_value**2 - np.log(float(prior.sd)) - 0.5 * np.log(2.0 * np.pi)
    )


def asis_centered_scale_update(
    z_path: Array,
    params_state: ParamDict,
    priors: Any,
    layout: NCPLayout,
    rng: np.random.Generator,
    *,
    tau: Optional[dict[str, float]] = None,
    lasso_variance_scale: float = 1.0,
    horseshoe_state: Optional[dict[str, Any]] = None,
    triple_gamma_state: Optional[dict[str, Any]] = None,
    proposal_step: float = 0.20,
) -> tuple[Array, ParamDict, dict[str, bool]]:
    """ASIS update of signed scales while holding the centred path fixed."""
    if getattr(priors, "ssvs", None) is not None:
        raise ValueError("ASIS is not combined with exact structural SSVS.")
    centered = map_ncp_to_centered(z_path, params_state, layout)
    innovations = _centered_innovations_by_component(centered, layout)
    out = dict(params_state)
    accepted: dict[str, bool] = {}
    key_map = {"level": "s_level", "trend": "s_trend", "season": "s_season"}
    for component, values in innovations.items():
        key = key_map[component]
        current_signed = float(out.get(key, 0.0))
        current = max(abs(current_signed), 1e-14)
        sign = -1.0 if current_signed < 0.0 else 1.0
        proposal = float(np.exp(np.log(current) + proposal_step * rng.normal()))
        sum_squares = float(np.asarray(values) @ np.asarray(values))
        n_values = int(np.asarray(values).size)

        def target(magnitude: float) -> float:
            if magnitude <= 0.0:
                return -np.inf
            transition = -n_values * np.log(magnitude) - 0.5 * sum_squares / magnitude**2
            prior_value = _signed_scale_prior_logpdf(
                sign * magnitude,
                component,
                priors,
                tau=tau,
                lasso_variance_scale=lasso_variance_scale,
                horseshoe_state=horseshoe_state,
                triple_gamma_state=triple_gamma_state,
            )
            return float(transition + prior_value + np.log(magnitude))

        take = bool(np.log(rng.random()) < target(proposal) - target(current))
        if take:
            out[key] = sign * proposal
        accepted[f"asis_{key}"] = take
    out["q_level"] = float(out.get("s_level", 0.0)) ** 2
    out["q_trend"] = float(out.get("s_trend", 0.0)) ** 2
    out["q_season"] = float(out.get("s_season", 0.0)) ** 2
    transformed = map_centered_to_ncp(centered, out, layout)
    return transformed, out, accepted


def mu_from_ncp(z_path: Array, params_state: ParamDict, layout: NCPLayout) -> Array:
    z_path = np.asarray(z_path, dtype=float)
    Tn = z_path.shape[0] - 1
    mu = baseline_mu_path(Tn, params_state, layout)
    if layout.idx_tilde_alpha is not None:
        mu += float(params_state.get("s_level", 0.0)) * z_path[1:, int(layout.idx_tilde_alpha)]
    if layout.idx_A is not None:
        mu += float(params_state.get("s_trend", 0.0)) * z_path[1:, int(layout.idx_A)]
    if layout.season_dim > 0:
        mu += float(params_state.get("s_season", 0.0)) * z_path[1:, layout.season_ncp_slice.start]
    return mu


def ffbs_gaussian_1d(
    y: Array,
    G: Array,
    Q: Array,
    H: Array,
    R: float,
    *,
    m0: Optional[Array] = None,
    C0: Optional[Array] = None,
    rng: Optional[np.random.Generator] = None,
    jitter: float = 1e-12,
) -> Array:
    y = np.asarray(y, dtype=float).reshape(-1)
    Tn = int(y.size)
    d = int(G.shape[0])
    H = np.asarray(H, dtype=float).reshape(1, d)

    if rng is None:
        rng = np.random.default_rng()
    if m0 is None:
        m0 = np.zeros(d, dtype=float)
    if C0 is None:
        C0 = 1e-6 * np.eye(d)

    m = np.zeros((Tn + 1, d), dtype=float)
    C = np.zeros((Tn + 1, d, d), dtype=float)
    a = np.zeros((Tn + 1, d), dtype=float)
    Rm = np.zeros((Tn + 1, d, d), dtype=float)

    m[0] = m0
    C[0] = _symmetrize(C0)

    for t in range(1, Tn + 1):
        a[t] = G @ m[t - 1]
        Rm[t] = _symmetrize(G @ C[t - 1] @ G.T + Q)
        F = float((H @ Rm[t] @ H.T).item() + float(R))
        if not np.isfinite(F) or F <= 0.0:
            F = float((H @ (Rm[t] + 1e-10 * np.eye(d)) @ H.T).item() + float(R))
        K = (Rm[t] @ H.T) / F
        v = float(y[t - 1] - (H @ a[t]).item())
        m[t] = a[t] + K[:, 0] * v
        I_KH = np.eye(d) - K @ H
        C[t] = _symmetrize(
            I_KH @ Rm[t] @ I_KH.T + float(R) * (K @ K.T)
        )

    z = np.zeros((Tn + 1, d), dtype=float)
    z[Tn] = _sample_gaussian(m[Tn], C[Tn], rng, jitter=jitter)

    I = np.eye(d)
    for t in range(Tn - 1, -1, -1):
        Rinv = spd_solve(Rm[t + 1], I, jitter=jitter)
        J = C[t] @ G.T @ Rinv
        mean = m[t] + J @ (z[t + 1] - (G @ m[t]))
        cov = _symmetrize(C[t] - J @ Rm[t + 1] @ J.T)
        z[t] = _sample_gaussian(mean, cov, rng, jitter=jitter)

    return z


def ffbs_gaussian_1d_tvR(
    y: Array,
    G: Array,
    Q: Array,
    H: Array,
    R_t: Array,
    *,
    m0: Optional[Array] = None,
    C0: Optional[Array] = None,
    rng: Optional[np.random.Generator] = None,
    jitter: float = 1e-12,
    R_floor: float = 1e-12,
) -> Array:
    y = np.asarray(y, dtype=float).reshape(-1)
    R_t = np.asarray(R_t, dtype=float).reshape(-1)
    Tn = int(y.size)
    d = int(G.shape[0])
    H = np.asarray(H, dtype=float).reshape(1, d)

    if R_t.size != Tn:
        raise ValueError("R_t must have length T")
    if rng is None:
        rng = np.random.default_rng()
    if m0 is None:
        m0 = np.zeros(d, dtype=float)
    if C0 is None:
        C0 = 1e-6 * np.eye(d)

    m = np.zeros((Tn + 1, d), dtype=float)
    C = np.zeros((Tn + 1, d, d), dtype=float)
    a = np.zeros((Tn + 1, d), dtype=float)
    Rm = np.zeros((Tn + 1, d, d), dtype=float)

    m[0] = m0
    C[0] = _symmetrize(C0)

    for t in range(1, Tn + 1):
        Robs = float(max(R_floor, R_t[t - 1]))
        a[t] = G @ m[t - 1]
        Rm[t] = _symmetrize(G @ C[t - 1] @ G.T + Q)
        F = float((H @ Rm[t] @ H.T).item() + Robs)
        if not np.isfinite(F) or F <= 0.0:
            F = float((H @ (Rm[t] + 1e-10 * np.eye(d)) @ H.T).item() + Robs)
        K = (Rm[t] @ H.T) / F
        v = float(y[t - 1] - (H @ a[t]).item())
        m[t] = a[t] + K[:, 0] * v
        I_KH = np.eye(d) - K @ H
        C[t] = _symmetrize(
            I_KH @ Rm[t] @ I_KH.T + Robs * (K @ K.T)
        )

    z = np.zeros((Tn + 1, d), dtype=float)
    z[Tn] = _sample_gaussian(m[Tn], C[Tn], rng, jitter=jitter)

    I = np.eye(d)
    for t in range(Tn - 1, -1, -1):
        Rinv = spd_solve(Rm[t + 1], I, jitter=jitter)
        J = C[t] @ G.T @ Rinv
        mean = m[t] + J @ (z[t + 1] - (G @ m[t]))
        cov = _symmetrize(C[t] - J @ Rm[t + 1] @ J.T)
        z[t] = _sample_gaussian(mean, cov, rng, jitter=jitter)

    return z


def gaussian_smoother_mean_1d_tvR(
    y: Array,
    G: Array,
    Q: Array,
    H: Array,
    R_t: Array,
    *,
    m0: Optional[Array] = None,
    C0: Optional[Array] = None,
    jitter: float = 1e-12,
    R_floor: float = 1e-12,
) -> Array:
    """Kalman/RTS posterior mean for a scalar, time-varying Gaussian layer."""
    y = np.asarray(y, dtype=float).reshape(-1)
    R_t = np.asarray(R_t, dtype=float).reshape(-1)
    Tn = int(y.size)
    d = int(G.shape[0])
    H = np.asarray(H, dtype=float).reshape(1, d)
    if R_t.size != Tn:
        raise ValueError("R_t must have length T")
    if m0 is None:
        m0 = np.zeros(d, dtype=float)
    if C0 is None:
        C0 = np.zeros((d, d), dtype=float)

    m = np.zeros((Tn + 1, d), dtype=float)
    C = np.zeros((Tn + 1, d, d), dtype=float)
    a = np.zeros((Tn + 1, d), dtype=float)
    Rm = np.zeros((Tn + 1, d, d), dtype=float)
    m[0] = np.asarray(m0, dtype=float)
    C[0] = _symmetrize(np.asarray(C0, dtype=float))

    for t in range(1, Tn + 1):
        observation_variance = float(max(R_floor, R_t[t - 1]))
        a[t] = G @ m[t - 1]
        Rm[t] = _symmetrize(G @ C[t - 1] @ G.T + Q)
        innovation_variance = float(
            (H @ Rm[t] @ H.T).item() + observation_variance
        )
        if not np.isfinite(innovation_variance) or innovation_variance <= 0.0:
            innovation_variance = float(
                (H @ (Rm[t] + jitter * np.eye(d)) @ H.T).item()
                + observation_variance
            )
        gain = (Rm[t] @ H.T) / innovation_variance
        innovation = float(y[t - 1] - (H @ a[t]).item())
        m[t] = a[t] + gain[:, 0] * innovation
        identity_minus = np.eye(d) - gain @ H
        C[t] = _symmetrize(
            identity_minus @ Rm[t] @ identity_minus.T
            + observation_variance * (gain @ gain.T)
        )

    smooth = m.copy()
    for t in range(Tn - 1, -1, -1):
        if np.linalg.norm(Rm[t + 1]) <= np.finfo(float).tiny:
            continue
        inverse = np.linalg.pinv(_symmetrize(Rm[t + 1]), rcond=1e-12)
        gain = C[t] @ G.T @ inverse
        smooth[t] = m[t] + gain @ (smooth[t + 1] - a[t + 1])
    return smooth


def design_matrix_ncp(
    z_path: Array,
    layout: NCPLayout,
    *,
    center_time: bool = True,
) -> tuple[Array, list[str], float]:
    """Build the Fruehwirth-Schnatter regression design.

    Time is centred by default. This is important for long records: independent
    priors on ``alpha0`` and ``beta0`` induce a correlated prior on the centred
    intercept ``alpha_c = alpha0 + tbar * beta0``.
    """
    z_path = np.asarray(z_path, dtype=float)
    Tn = z_path.shape[0] - 1
    cols: list[Array] = []
    names: list[str] = []

    t1 = np.arange(1, Tn + 1, dtype=float)
    tbar = float(t1.mean()) if center_time else 0.0
    cols.append(np.ones(Tn, dtype=float))
    names.append("alpha_c" if center_time and layout.has_beta else "alpha0")
    if layout.has_beta:
        cols.append(t1 - tbar)
        names.append("beta0")

    if layout.season_dim > 0:
        K = layout.season_dim
        S = np.zeros((Tn, K), dtype=float)
        for i in range(Tn):
            season = i % (K + 1)
            if season < K:
                S[i, season] = 1.0
            else:
                S[i, :] = -1.0
        for j in range(K):
            cols.append(S[:, j])
            names.append(f"gamma0_season_{j+1}")

    cols.append(z_path[1:, int(layout.idx_tilde_alpha)])
    names.append("s_level")

    if layout.has_beta:
        cols.append(z_path[1:, int(layout.idx_A)])
        names.append("s_trend")

    if layout.season_dim > 0:
        cols.append(z_path[1:, layout.season_ncp_slice.start])
        names.append("s_season")

    X = np.column_stack(cols) if cols else np.zeros((Tn, 0), dtype=float)
    return X, names, tbar


def _posterior_gaussian_precision(
    X: Array,
    y: Array,
    obs_var: Array | float,
    prior_mean: Array,
    prior_precision: Array,
    rng: np.random.Generator,
) -> Array:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    prior_mean = np.asarray(prior_mean, dtype=float).reshape(-1)
    prior_precision = np.asarray(prior_precision, dtype=float)
    if np.ndim(obs_var) == 0:
        w = np.full(y.size, 1.0 / max(float(obs_var), 1e-12), dtype=float)
    else:
        v = np.asarray(obs_var, dtype=float).reshape(-1)
        if v.size != y.size:
            raise ValueError("obs_var must be scalar or have length T.")
        w = 1.0 / np.maximum(v, 1e-12)

    WX = X * w[:, None]
    precision = _symmetrize(prior_precision + X.T @ WX) + 1e-12 * np.eye(X.shape[1])
    covariance = _symmetrize(spd_solve(precision, np.eye(X.shape[1])))
    mean = covariance @ (prior_precision @ prior_mean + X.T @ (w * y))
    return _sample_gaussian(mean, covariance, rng)


def _active_scale_names(layout: NCPLayout) -> list[str]:
    names = ["level"]
    if layout.has_beta:
        names.append("trend")
    if layout.season_dim > 0:
        names.append("season")
    return names


def initialise_lasso(priors: Any, layout: NCPLayout) -> tuple[dict[str, float], Any]:
    """Initialise normal-mixture scales for lasso or PC shrinkage."""
    if (
        getattr(priors, "lasso", None) is None
        and getattr(priors, "pc", None) is None
    ):
        return {}, np.nan
    if getattr(priors, "lasso", None) is None:
        pc = priors.pc
        return {
            name: float(pc.initial_tau) for name in _active_scale_names(layout)
        }, np.nan
    lp = priors.lasso
    names = _active_scale_names(layout)
    if bool(getattr(lp, "componentwise", False)):
        tau = {name: float(lp.initial_tau_for(name)) for name in names}
        lambda2 = {name: float(lp.initial_lambda2_for(name)) for name in names}
        return tau, lambda2
    tau = {name: float(lp.initial_tau) for name in names}
    return tau, float(lp.initial_lambda2)


def copy_lasso_lambda2(lambda2: Any) -> Any:
    return dict(lambda2) if isinstance(lambda2, dict) else float(lambda2)


def lasso_coefficient_scale(lp: Any, component: str) -> float:
    method = getattr(lp, "coefficient_scale_for", None)
    return float(method(component)) if callable(method) else 1.0


def initialise_horseshoe(priors: Any, layout: NCPLayout) -> dict[str, Any]:
    """Initialise the regularized-horseshoe hierarchy.

    The returned object is deliberately a plain dictionary so it can be stored
    without pickle and copied safely when a DGEV iteration is restored.
    """
    hp = getattr(priors, "horseshoe", None)
    if hp is None:
        return {}
    names = _active_scale_names(layout)
    return {
        "local": {name: float(hp.initial_local) for name in names},
        "global": float(hp.initial_global_value()),
        "slab2": float(hp.initial_slab2_value()),
    }


def copy_horseshoe_state(state: dict[str, Any]) -> dict[str, Any]:
    if not state:
        return {}
    return {
        "local": {key: float(value) for key, value in state["local"].items()},
        "global": float(state["global"]),
        "slab2": float(state["slab2"]),
    }


def initialise_triple_gamma(priors: Any, layout: NCPLayout) -> dict[str, Any]:
    """Initialise the paper's normal-gamma-gamma hierarchy."""

    prior = getattr(priors, "triple_gamma", None)
    if prior is None:
        return {}
    names = _active_scale_names(layout)
    return {
        "numerator": {
            name: float(prior.initial_numerator) for name in names
        },
        "denominator": {
            name: float(prior.initial_denominator) for name in names
        },
        "global": float(prior.global_scale),
        "a": float(prior.spike_shape),
        "c": float(prior.tail_shape),
        "slab2": (
            float(prior.initial_slab2_value()) if prior.regularized else None
        ),
    }


def copy_triple_gamma_state(state: dict[str, Any]) -> dict[str, Any]:
    if not state:
        return {}
    return {
        "numerator": {
            key: float(value) for key, value in state["numerator"].items()
        },
        "denominator": {
            key: float(value) for key, value in state["denominator"].items()
        },
        "global": float(state["global"]),
        "a": float(state["a"]),
        "c": float(state["c"]),
        "slab2": (
            None if state.get("slab2") is None else float(state["slab2"])
        ),
    }


def triple_gamma_conditional_variance(
    priors: Any,
    state: dict[str, Any],
    component: str,
) -> float:
    prior = getattr(priors, "triple_gamma", None)
    if prior is None:
        raise ValueError("A triple-gamma prior is not active.")
    if not state or component not in state.get("numerator", {}):
        raise ValueError(f"Missing triple-gamma state for {component}.")
    return float(
        prior.conditional_variance(
            component,
            numerator=float(state["numerator"][component]),
            denominator=float(state["denominator"][component]),
            global_scale=float(state["global"]),
            slab2=state.get("slab2"),
        )
    )


def horseshoe_conditional_variance(
    priors: Any,
    state: dict[str, Any],
    component: str,
) -> float:
    hp = getattr(priors, "horseshoe", None)
    if hp is None:
        raise ValueError("A regularized-horseshoe prior is not active.")
    if not state or component not in state.get("local", {}):
        raise ValueError(f"Missing regularized-horseshoe state for {component}.")
    return float(
        hp.conditional_variance(
            component,
            local=float(state["local"][component]),
            global_scale=float(state["global"]),
            slab2=float(state["slab2"]),
        )
    )


def _normal_zero_logpdf(value: float, variance: float) -> float:
    variance = max(float(variance), 1e-300)
    return float(
        -0.5 * (np.log(2.0 * np.pi * variance) + float(value) ** 2 / variance)
    )


def _half_cauchy_logpdf(value: float, scale: float) -> float:
    value = float(value)
    scale = float(scale)
    if value <= 0.0 or scale <= 0.0:
        return -np.inf
    return float(np.log(2.0 / (np.pi * scale)) - np.log1p((value / scale) ** 2))


def _inverse_gamma_logpdf(value: float, shape: float, scale: float) -> float:
    value = float(value)
    if value <= 0.0:
        return -np.inf
    return float(
        shape * np.log(scale)
        - lgamma(shape)
        - (shape + 1.0) * np.log(value)
        - scale / value
    )


def _gamma_logpdf(value: float, shape: float, rate: float = 1.0) -> float:
    value = float(value)
    shape = float(shape)
    rate = float(rate)
    if value <= 0.0 or shape <= 0.0 or rate <= 0.0:
        return -np.inf
    return float(
        shape * np.log(rate)
        - lgamma(shape)
        + (shape - 1.0) * np.log(value)
        - rate * value
    )


def _beta_prime_logpdf(value: float, first: float, second: float) -> float:
    value = float(value)
    first = float(first)
    second = float(second)
    if value <= 0.0 or first <= 0.0 or second <= 0.0:
        return -np.inf
    log_beta = lgamma(first) + lgamma(second) - lgamma(first + second)
    return float(
        (first - 1.0) * np.log(value)
        - (first + second) * np.log1p(value)
        - log_beta
    )


def _beta_logpdf(value: float, first: float, second: float) -> float:
    value = float(value)
    if not 0.0 < value < 1.0 or first <= 0.0 or second <= 0.0:
        return -np.inf
    log_beta = lgamma(first) + lgamma(second) - lgamma(first + second)
    return float(
        (first - 1.0) * np.log(value)
        + (second - 1.0) * np.log1p(-value)
        - log_beta
    )


def _slice_sample_real(
    current: float,
    log_density,
    rng: np.random.Generator,
    *,
    width: float = 1.0,
    max_steps_out: int = 50,
    max_shrink: int = 500,
) -> tuple[float, int]:
    """Univariate stepping-out slice sampler on the real line.

    The routine is used on log-scales (and on logit shape coordinates), so all
    positivity constraints are respected without rejected random-walk moves.
    It leaves the exact conditional invariant; the returned count is the
    number of target evaluations, useful for future performance diagnostics.
    """

    current = float(current)
    width = max(float(width), 1e-6)
    current_density = float(log_density(current))
    if not np.isfinite(current_density):
        raise FloatingPointError("Slice sampler started outside finite support.")
    log_height = current_density + np.log(max(float(rng.random()), 1e-300))
    left = current - width * float(rng.random())
    right = left + width
    evaluations = 1
    left_steps = int(rng.integers(max_steps_out + 1))
    right_steps = max_steps_out - left_steps
    while left_steps > 0:
        value = float(log_density(left))
        evaluations += 1
        if not np.isfinite(value) or value <= log_height:
            break
        left -= width
        left_steps -= 1
    while right_steps > 0:
        value = float(log_density(right))
        evaluations += 1
        if not np.isfinite(value) or value <= log_height:
            break
        right += width
        right_steps -= 1
    for _ in range(max_shrink):
        proposal = float(rng.uniform(left, right))
        value = float(log_density(proposal))
        evaluations += 1
        if np.isfinite(value) and value >= log_height:
            return proposal, evaluations
        if proposal < current:
            left = proposal
        else:
            right = proposal
    raise RuntimeError("Slice sampler failed to find a point on the slice.")


def update_triple_gamma_scales(
    params_state: ParamDict,
    state: dict[str, Any],
    priors: Any,
    layout: NCPLayout,
    *,
    rng: np.random.Generator,
    width_local: float = 1.0,
    width_global: float = 1.0,
    width_shape: float = 0.8,
    width_slab: float = 0.8,
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Exact slice updates for a (regularized) triple-gamma hierarchy."""

    prior = getattr(priors, "triple_gamma", None)
    if prior is None:
        return state, {}
    out = copy_triple_gamma_state(state)
    names = _active_scale_names(layout)
    scale_keys = {"level": "s_level", "trend": "s_trend", "season": "s_season"}
    moved: dict[str, bool] = {}

    def coefficient_log_density(candidate: dict[str, Any], selected=None) -> float:
        total = 0.0
        for component in names if selected is None else selected:
            variance = prior.conditional_variance(
                component,
                numerator=candidate["numerator"][component],
                denominator=candidate["denominator"][component],
                global_scale=candidate["global"],
                slab2=candidate.get("slab2"),
            )
            total += _normal_zero_logpdf(
                float(params_state.get(scale_keys[component], 0.0)), variance
            )
        return float(total)

    for component in names:
        for field_name, shape_name in (
            ("numerator", "a"),
            ("denominator", "c"),
        ):
            current_log = np.log(max(float(out[field_name][component]), 1e-300))

            def target(log_value, *, field_name=field_name, shape_name=shape_name):
                candidate = copy_triple_gamma_state(out)
                value = float(np.exp(np.clip(log_value, -700.0, 700.0)))
                candidate[field_name][component] = value
                return (
                    coefficient_log_density(candidate, [component])
                    + _gamma_logpdf(value, candidate[shape_name], 1.0)
                    + log_value
                )

            proposed_log, _ = _slice_sample_real(
                current_log, target, rng, width=width_local
            )
            out[field_name][component] = float(np.exp(proposed_log))
            key = f"triple_gamma_{field_name}_{component}"
            moved[key] = not np.isclose(proposed_log, current_log)

    if prior.learn_global:
        current_log = np.log(max(float(out["global"]), 1e-300))

        def global_target(log_value):
            candidate = copy_triple_gamma_state(out)
            value = float(np.exp(np.clip(log_value, -700.0, 700.0)))
            candidate["global"] = value
            return (
                coefficient_log_density(candidate)
                + _beta_prime_logpdf(value, candidate["c"], candidate["a"])
                + log_value
            )

        proposed_log, _ = _slice_sample_real(
            current_log, global_target, rng, width=width_global
        )
        out["global"] = float(np.exp(proposed_log))
        moved["triple_gamma_global"] = not np.isclose(proposed_log, current_log)

    if prior.learn_shapes:
        for shape_name, hyperprior, local_field in (
            ("a", prior.spike_shape_prior, "numerator"),
            ("c", prior.tail_shape_prior, "denominator"),
        ):
            current_shape = float(out[shape_name])
            current_logit = np.log(current_shape / (0.5 - current_shape))

            def shape_target(logit_value, *, shape_name=shape_name,
                             hyperprior=hyperprior, local_field=local_field):
                probability = 1.0 / (1.0 + np.exp(-np.clip(logit_value, -700.0, 700.0)))
                shape = 0.5 * probability
                candidate = copy_triple_gamma_state(out)
                candidate[shape_name] = shape
                total = sum(
                    _gamma_logpdf(candidate[local_field][component], shape, 1.0)
                    for component in names
                )
                if prior.learn_global:
                    total += _beta_prime_logpdf(
                        candidate["global"], candidate["c"], candidate["a"]
                    )
                total += _beta_logpdf(probability, *hyperprior)
                total += np.log(max(probability * (1.0 - probability), 1e-300))
                return float(total)

            proposed_logit, _ = _slice_sample_real(
                current_logit, shape_target, rng, width=width_shape
            )
            probability = 1.0 / (1.0 + np.exp(-proposed_logit))
            out[shape_name] = float(0.5 * probability)
            moved[f"triple_gamma_{shape_name}"] = not np.isclose(
                proposed_logit, current_logit
            )

    if prior.regularized:
        current_log = np.log(max(float(out["slab2"]), 1e-300))
        shape = 0.5 * float(prior.slab_df)
        scale = 0.5 * float(prior.slab_df) * float(prior.slab_scale) ** 2

        def slab_target(log_value):
            candidate = copy_triple_gamma_state(out)
            value = float(np.exp(np.clip(log_value, -700.0, 700.0)))
            candidate["slab2"] = value
            return (
                coefficient_log_density(candidate)
                + _inverse_gamma_logpdf(value, shape, scale)
                + log_value
            )

        proposed_log, _ = _slice_sample_real(
            current_log, slab_target, rng, width=width_slab
        )
        out["slab2"] = float(np.exp(proposed_log))
        moved["triple_gamma_slab2"] = not np.isclose(proposed_log, current_log)

    return out, moved


def update_horseshoe_scales(
    params_state: ParamDict,
    state: dict[str, Any],
    priors: Any,
    layout: NCPLayout,
    *,
    rng: np.random.Generator,
    step_local: float = 0.35,
    step_global: float = 0.25,
    step_slab: float = 0.20,
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Exact slice updates for regularized-horseshoe hyperparameters.

    v2.1.4 used one-step log random walks for the local, global and slab
    scales.  In the three-component structural comparison that hierarchy forms
    a pronounced ridge, producing low ESS even when the signed innovation
    coefficients mix.  Stepping-out slice updates target the same exact
    conditional without tuning an acceptance rate and traverse that ridge much
    more reliably.  The ``step_*`` arguments now act as slice widths.
    """
    hp = getattr(priors, "horseshoe", None)
    if hp is None:
        return state, {}
    out = copy_horseshoe_state(state)
    names = _active_scale_names(layout)
    scale_keys = {"level": "s_level", "trend": "s_trend", "season": "s_season"}
    moved: dict[str, bool] = {}

    def q_log_density(candidate: dict[str, Any], subset: Optional[list[str]] = None) -> float:
        total = 0.0
        for component in names if subset is None else subset:
            variance = hp.conditional_variance(
                component,
                local=float(candidate["local"][component]),
                global_scale=float(candidate["global"]),
                slab2=float(candidate["slab2"]),
            )
            total += _normal_zero_logpdf(
                float(params_state.get(scale_keys[component], 0.0)), variance
            )
        return float(total)

    for component in names:
        current = float(out["local"][component])
        current_log = np.log(current)

        def local_target(log_value):
            candidate = copy_horseshoe_state(out)
            value = float(np.exp(np.clip(log_value, -700.0, 700.0)))
            candidate["local"][component] = value
            return (
                q_log_density(candidate, [component])
                + _half_cauchy_logpdf(value, 1.0)
                + log_value
            )

        proposed_log, _ = _slice_sample_real(
            current_log, local_target, rng, width=step_local
        )
        out["local"][component] = float(np.exp(proposed_log))
        moved[f"horseshoe_local_{component}"] = not np.isclose(
            proposed_log, current_log
        )

    current = float(out["global"])
    current_log = np.log(current)

    def global_target(log_value):
        candidate = copy_horseshoe_state(out)
        value = float(np.exp(np.clip(log_value, -700.0, 700.0)))
        candidate["global"] = value
        return (
            q_log_density(candidate)
            + _half_cauchy_logpdf(value, float(hp.global_scale))
            + log_value
        )

    proposed_log, _ = _slice_sample_real(
        current_log, global_target, rng, width=step_global
    )
    out["global"] = float(np.exp(proposed_log))
    moved["horseshoe_global"] = not np.isclose(proposed_log, current_log)

    current = float(out["slab2"])
    current_log = np.log(current)
    shape = 0.5 * float(hp.slab_df)
    scale = 0.5 * float(hp.slab_df) * float(hp.slab_scale) ** 2

    def slab_target(log_value):
        candidate = copy_horseshoe_state(out)
        value = float(np.exp(np.clip(log_value, -700.0, 700.0)))
        candidate["slab2"] = value
        return (
            q_log_density(candidate)
            + _inverse_gamma_logpdf(value, shape, scale)
            + log_value
        )

    proposed_log, _ = _slice_sample_real(
        current_log, slab_target, rng, width=step_slab
    )
    out["slab2"] = float(np.exp(proposed_log))
    moved["horseshoe_slab2"] = not np.isclose(proposed_log, current_log)
    return out, moved


def _rand_invgauss(mu: float, lam: float, rng: np.random.Generator) -> float:
    """Numerically guarded Michael-Schucany-Haas inverse-Gaussian draw."""
    mu = float(np.clip(mu, 1e-10, 1e10))
    lam = float(np.clip(lam, 1e-10, 1e10))
    v = float(rng.normal())
    y = v * v
    root = np.sqrt(max(0.0, 4.0 * mu * lam * y + mu * mu * y * y))
    x = mu + (mu * mu * y) / (2.0 * lam) - (mu / (2.0 * lam)) * root
    x = max(x, 1e-12)
    if rng.random() > mu / (mu + x):
        x = (mu * mu) / x
    return float(np.clip(x, 1e-12, 1e12))


def update_lasso_scales(
    params_state: ParamDict,
    tau: dict[str, float],
    lambda2: Any,
    priors: Any,
    layout: NCPLayout,
    *,
    variance_scale: float,
    rng: np.random.Generator,
) -> tuple[dict[str, float], Any]:
    """Park-Casella local-scale update with optional component-wise lambdas."""
    lp = getattr(priors, "lasso", None)
    if lp is None:
        return tau, lambda2

    sig2 = max(float(variance_scale), 1e-12)
    out: dict[str, float] = {}
    key_map = {"level": "s_level", "trend": "s_trend", "season": "s_season"}
    names = _active_scale_names(layout)
    componentwise = bool(getattr(lp, "componentwise", False))

    for name in names:
        lam2 = max(
            float(lambda2[name]) if componentwise else float(lambda2),
            1e-12,
        )
        coefficient_scale = max(lasso_coefficient_scale(lp, name), 1e-16)
        sk = float(params_state.get(key_map[name], 0.0))
        standardized_s2 = (sk / coefficient_scale) ** 2
        if standardized_s2 < 1e-16:
            if componentwise:
                initial_tau = float(lp.initial_tau_for(name))
            else:
                initial_tau = float(lp.initial_tau)
            out[name] = max(float(tau.get(name, initial_tau)), 1e-8)
        else:
            mu_u = np.sqrt(lam2 * sig2 / standardized_s2)
            u_k = _rand_invgauss(mu_u, lam2, rng)
            out[name] = float(np.clip(1.0 / u_k, 1e-12, 1e12))

    if componentwise:
        new_lambda2 = {}
        for name in names:
            shape = float(lp.a_for(name) + 1.0)
            rate = float(lp.b_for(name) + 0.5 * out[name])
            new_lambda2[name] = float(
                rng.gamma(shape=shape, scale=1.0 / max(rate, 1e-12))
            )
        return out, new_lambda2

    shape = float(lp.a_lambda + len(out))
    rate = float(lp.b_lambda + 0.5 * sum(out.values()))
    new_lambda2 = float(rng.gamma(shape=shape, scale=1.0 / max(rate, 1e-12)))
    return out, new_lambda2


def update_pc_scales(
    params_state: ParamDict,
    tau: dict[str, float],
    priors: Any,
    layout: NCPLayout,
    *,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Exact local-mixture update for the symmetric-exponential PC prior."""
    pc = getattr(priors, "pc", None)
    if pc is None:
        return tau
    output: dict[str, float] = {}
    key_map = {"level": "s_level", "trend": "s_trend", "season": "s_season"}
    for component in _active_scale_names(layout):
        coefficient_scale = float(pc.coefficient_scale_for(component))
        standardized = float(params_state.get(key_map[component], 0.0)) / coefficient_scale
        lambda2 = float(pc.standardized_rate_for(component)) ** 2
        if standardized * standardized < 1e-16:
            output[component] = max(
                float(tau.get(component, pc.initial_tau)), 1e-8
            )
            continue
        inverse_mean = np.sqrt(lambda2 / (standardized * standardized))
        inverse_tau = _rand_invgauss(inverse_mean, lambda2, rng)
        output[component] = float(np.clip(1.0 / inverse_tau, 1e-12, 1e12))
    return output

def _theta_prior_mean_precision(
    priors: Any,
    layout: NCPLayout,
    theta_names: list[str],
    *,
    tbar: float,
    tau: Optional[dict[str, float]],
    lasso_variance_scale: float,
    horseshoe_state: Optional[dict[str, Any]] = None,
    triple_gamma_state: Optional[dict[str, Any]] = None,
) -> tuple[Array, Array]:
    d = len(theta_names)
    mean = np.zeros(d, dtype=float)
    precision = np.zeros((d, d), dtype=float)
    index = {name: i for i, name in enumerate(theta_names)}

    # Independent priors on alpha0 and beta0 induce a correlated prior on
    # (alpha_c, beta0), where alpha_c = alpha0 + tbar * beta0.
    ia = index["alpha_c"] if "alpha_c" in index else index["alpha0"]
    va = float(priors.alpha0.sd) ** 2
    ma = float(priors.alpha0.mean)
    if layout.has_beta:
        ib = index["beta0"]
        vb = float(priors.beta0.sd) ** 2
        mb = float(priors.beta0.mean)
        mean[ia] = ma + tbar * mb
        mean[ib] = mb
        V = np.array(
            [[va + (tbar * tbar) * vb, tbar * vb], [tbar * vb, vb]],
            dtype=float,
        )
        precision[np.ix_([ia, ib], [ia, ib])] = _symmetrize(spd_solve(V, np.eye(2)))
    else:
        mean[ia] = ma
        precision[ia, ia] = 1.0 / max(va, 1e-12)

    if layout.season_dim > 0:
        gp = priors.gamma0_season
        if gp is None:
            raise ValueError("gamma0_season prior is required for a dynamic seasonal block.")
        gm, gs = gp.mean_array(), gp.sd_array()
        if gm.size != layout.season_dim:
            raise ValueError("gamma0_season prior dimension mismatch.")
        for j in range(layout.season_dim):
            i = index[f"gamma0_season_{j+1}"]
            mean[i] = float(gm[j])
            precision[i, i] = 1.0 / max(float(gs[j]) ** 2, 1e-12)

    scale_map = {"s_level": "level", "s_trend": "trend", "s_season": "season"}
    for theta_name, block_name in scale_map.items():
        if theta_name not in index:
            continue
        i = index[theta_name]
        if getattr(priors, "lasso", None) is not None:
            if tau is None or block_name not in tau:
                raise ValueError(f"Missing tau scale for {block_name}.")
            coefficient_scale = lasso_coefficient_scale(priors.lasso, block_name)
            prior_variance = (
                lasso_variance_scale
                * coefficient_scale**2
                * float(tau[block_name])
            )
            precision[i, i] = 1.0 / max(prior_variance, 1e-16)
        elif getattr(priors, "pc", None) is not None:
            if tau is None or block_name not in tau:
                raise ValueError(f"Missing PC mixture scale for {block_name}.")
            coefficient_scale = priors.pc.coefficient_scale_for(block_name)
            prior_variance = coefficient_scale**2 * float(tau[block_name])
            precision[i, i] = 1.0 / max(prior_variance, 1e-16)
        elif getattr(priors, "horseshoe", None) is not None:
            if horseshoe_state is None:
                raise ValueError("Missing regularized-horseshoe latent scales.")
            prior_variance = horseshoe_conditional_variance(
                priors, horseshoe_state, block_name
            )
            precision[i, i] = 1.0 / max(prior_variance, 1e-16)
        elif getattr(priors, "triple_gamma", None) is not None:
            if triple_gamma_state is None:
                raise ValueError("Missing triple-gamma latent scales.")
            prior_variance = triple_gamma_conditional_variance(
                priors, triple_gamma_state, block_name
            )
            precision[i, i] = 1.0 / max(prior_variance, 1e-16)
        else:
            prior = getattr(priors, theta_name)
            if prior is None:
                raise ValueError(f"Missing fixed NormalPrior for {theta_name}.")
            mean[i] = float(prior.mean)
            precision[i, i] = 1.0 / max(float(prior.sd) ** 2, 1e-12)

    return mean, precision


def apply_theta_draw(
    params_state: ParamDict,
    theta: Array,
    theta_names: list[str],
    layout: NCPLayout,
    *,
    tbar: float = 0.0,
) -> ParamDict:
    out = dict(params_state)
    season_vals = np.zeros(layout.season_dim, dtype=float)
    alpha_c: Optional[float] = None
    beta0 = float(out.get("beta0", 0.0))
    for val, name in zip(theta, theta_names):
        if name == "alpha_c":
            alpha_c = float(val)
        elif name == "beta0":
            beta0 = float(val)
            out[name] = beta0
        elif name.startswith("gamma0_season_"):
            j = int(name.rsplit("_", 1)[1]) - 1
            season_vals[j] = float(val)
        else:
            out[name] = float(val)
    if alpha_c is not None:
        out["alpha0"] = alpha_c - float(tbar) * beta0
    if layout.season_dim > 0:
        out["gamma0_season"] = season_vals
    out["q_level"] = float(out.get("s_level", 0.0)) ** 2
    out["q_trend"] = float(out.get("s_trend", 0.0)) ** 2
    out["q_season"] = float(out.get("s_season", 0.0)) ** 2
    return out


def gaussian_theta_update(
    y: Array,
    z_path: Array,
    sigma2: float,
    priors: Any,
    layout: NCPLayout,
    rng: np.random.Generator,
    *,
    tau: Optional[dict[str, float]] = None,
    lasso_variance_scale: Optional[float] = None,
    horseshoe_state: Optional[dict[str, Any]] = None,
    triple_gamma_state: Optional[dict[str, Any]] = None,
) -> ParamDict:
    X, names, tbar = design_matrix_ncp(z_path, layout, center_time=True)
    if lasso_variance_scale is None:
        lasso_variance_scale = float(sigma2)
    pm, pp = _theta_prior_mean_precision(
        priors,
        layout,
        names,
        tbar=tbar,
        tau=tau,
        lasso_variance_scale=float(lasso_variance_scale),
        horseshoe_state=horseshoe_state,
        triple_gamma_state=triple_gamma_state,
    )
    theta = _posterior_gaussian_precision(X, y, float(sigma2), pm, pp, rng)
    return apply_theta_draw({}, theta, names, layout, tbar=tbar)


def gev_theta_update(
    z_pseudo: Array,
    R_t: Array,
    z_path: Array,
    priors: Any,
    layout: NCPLayout,
    rng: np.random.Generator,
    *,
    tau: Optional[dict[str, float]] = None,
    lasso_variance_scale: float = 1.0,
    horseshoe_state: Optional[dict[str, Any]] = None,
    triple_gamma_state: Optional[dict[str, Any]] = None,
) -> ParamDict:
    X, names, tbar = design_matrix_ncp(z_path, layout, center_time=True)
    pm, pp = _theta_prior_mean_precision(
        priors,
        layout,
        names,
        tbar=tbar,
        tau=tau,
        lasso_variance_scale=float(lasso_variance_scale),
        horseshoe_state=horseshoe_state,
        triple_gamma_state=triple_gamma_state,
    )
    theta = _posterior_gaussian_precision(X, z_pseudo, R_t, pm, pp, rng)
    return apply_theta_draw({}, theta, names, layout, tbar=tbar)


def theta_vector_from_params(
    params_state: ParamDict,
    theta_names: list[str],
    layout: NCPLayout,
    *,
    tbar: float,
) -> Array:
    """Extract the FS regression vector in the same coordinates as its design."""
    values = np.zeros(len(theta_names), dtype=float)
    gamma = np.asarray(
        params_state.get("gamma0_season", np.zeros(layout.season_dim)), dtype=float
    ).reshape(-1)
    beta0 = float(params_state.get("beta0", 0.0))
    for index, name in enumerate(theta_names):
        if name == "alpha_c":
            values[index] = float(params_state.get("alpha0", 0.0)) + tbar * beta0
        elif name == "alpha0":
            values[index] = float(params_state.get("alpha0", 0.0))
        elif name == "beta0":
            values[index] = beta0
        elif name.startswith("gamma0_season_"):
            component = int(name.rsplit("_", 1)[1]) - 1
            values[index] = float(gamma[component])
        else:
            values[index] = float(params_state.get(name, 0.0))
    return values


def fs_theta_prior(
    priors: Any,
    layout: NCPLayout,
    theta_names: list[str],
    *,
    tbar: float,
    tau: Optional[dict[str, float]] = None,
    lasso_variance_scale: float = 1.0,
    horseshoe_state: Optional[dict[str, Any]] = None,
    triple_gamma_state: Optional[dict[str, Any]] = None,
) -> tuple[Array, Array]:
    """Return the conditional Gaussian mean/covariance of the FS block."""
    mean, precision = _theta_prior_mean_precision(
        priors,
        layout,
        theta_names,
        tbar=tbar,
        tau=tau,
        lasso_variance_scale=lasso_variance_scale,
        horseshoe_state=horseshoe_state,
        triple_gamma_state=triple_gamma_state,
    )
    covariance = _symmetrize(spd_solve(precision, np.eye(precision.shape[0])))
    return mean, covariance


def elliptical_slice_gaussian_prior(
    current: Array,
    prior_mean: Array,
    prior_covariance: Array,
    log_likelihood,
    rng: np.random.Generator,
    *,
    max_steps: int = 10_000,
) -> tuple[Array, int]:
    """Elliptical-slice update for a Gaussian-prior regression block.

    ``log_likelihood`` must contain only the likelihood contribution.  The
    Gaussian prior is represented by the ellipse and is therefore not included
    in the slice threshold.
    """
    current = np.asarray(current, dtype=float)
    prior_mean = np.asarray(prior_mean, dtype=float)
    root = np.linalg.cholesky(_project_psd(prior_covariance, floor=1e-14))
    direction = root @ rng.normal(size=current.size)
    centered = current - prior_mean
    current_loglik = float(log_likelihood(current))
    if not np.isfinite(current_loglik):
        raise FloatingPointError("The current FS regression vector has invalid likelihood.")
    threshold = current_loglik + float(np.log(rng.random()))
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    lower = angle - 2.0 * np.pi
    upper = angle
    for step in range(1, int(max_steps) + 1):
        proposal = prior_mean + centered * np.cos(angle) + direction * np.sin(angle)
        proposal_loglik = float(log_likelihood(proposal))
        if np.isfinite(proposal_loglik) and proposal_loglik >= threshold:
            return proposal, step
        if angle < 0.0:
            lower = angle
        else:
            upper = angle
        angle = float(rng.uniform(lower, upper))
    raise FloatingPointError("Elliptical-slice sampling did not find an acceptable point.")


def random_sign_switches(
    z_path: Array,
    params_state: ParamDict,
    layout: NCPLayout,
    rng: np.random.Generator,
) -> tuple[Array, ParamDict]:
    z = np.asarray(z_path, dtype=float).copy()
    out = dict(params_state)

    if layout.idx_tilde_alpha is not None and rng.random() < 0.5:
        out["s_level"] = -float(out.get("s_level", 0.0))
        z[:, int(layout.idx_tilde_alpha)] *= -1.0

    if layout.has_beta and rng.random() < 0.5:
        out["s_trend"] = -float(out.get("s_trend", 0.0))
        z[:, int(layout.idx_tilde_beta)] *= -1.0
        z[:, int(layout.idx_A)] *= -1.0

    if layout.season_dim > 0 and rng.random() < 0.5:
        out["s_season"] = -float(out.get("s_season", 0.0))
        z[:, layout.season_ncp_slice] *= -1.0

    out["q_level"] = float(out.get("s_level", 0.0)) ** 2
    out["q_trend"] = float(out.get("s_trend", 0.0)) ** 2
    out["q_season"] = float(out.get("s_season", 0.0)) ** 2
    return z, out


# ---------- genuine conditional SMC / PGAS in the FS parameterization ----------

@dataclass(frozen=True)
class NCPPGASResult:
    z_path: Array
    ess: Array
    unique_ancestors: Array
    reference_ancestors: Array
    path_changed: bool
    changed_fraction: float
    exact_invariant: bool = True


@dataclass(frozen=True)
class NCPLaplaceResult:
    z_path: Array
    mode_path: Array
    pseudo_y: Array
    pseudo_variance: Array
    converged: bool
    iterations: int
    relative_change: float
    objective: float
    support_rejections: int


def _strict_normalize_logweights(log_weights: Array) -> tuple[Array, Array, float]:
    """Normalize without discarding finite log probabilities through underflow."""
    log_weights = np.asarray(log_weights, dtype=float)
    finite = np.isfinite(log_weights)
    if not np.any(finite):
        raise FloatingPointError("All particle weights are zero.")
    maximum = float(np.max(log_weights[finite]))
    shifted = np.full_like(log_weights, -np.inf)
    shifted[finite] = log_weights[finite] - maximum
    total = float(np.sum(np.exp(shifted[finite])))
    log_normalizer = maximum + np.log(total)
    normalized_log = log_weights - log_normalizer
    weights = np.zeros_like(log_weights)
    weights[finite] = np.exp(normalized_log[finite])
    weights /= float(np.sum(weights))
    return weights, normalized_log, float(log_normalizer)


def _ncp_transition_structure(Q: Array) -> tuple[Array, Array, Array]:
    diagonal = np.diag(np.asarray(Q, dtype=float))
    active = np.flatnonzero(diagonal > 1e-12)
    deterministic = np.flatnonzero(diagonal <= 1e-12)
    loading = np.eye(Q.shape[0], dtype=float)[:, active]
    return active, deterministic, loading


def _ncp_transition_logpdf(
    value: Array,
    means: Array,
    active: Array,
    deterministic: Array,
    *,
    tolerance: float = 1e-6,
) -> Array:
    means = np.asarray(means, dtype=float)
    value = np.asarray(value, dtype=float)
    residual = value - means
    one = residual.ndim == 1
    if one:
        residual = residual[None, :]
        means = means[None, :]
    output = np.full(residual.shape[0], -np.inf, dtype=float)
    on_support = np.ones(residual.shape[0], dtype=bool)
    if deterministic.size:
        deterministic_error = np.max(np.abs(residual[:, deterministic]), axis=1)
        reference_scale = 1.0 + np.max(np.abs(means[:, deterministic]), axis=1)
        on_support &= deterministic_error <= tolerance * reference_scale
    if np.any(on_support):
        stochastic = residual[on_support][:, active]
        output[on_support] = -0.5 * (
            active.size * np.log(2.0 * np.pi)
            + np.sum(stochastic * stochastic, axis=1)
        )
    return output[0] if one else output


def _ncp_observation_logweights(
    y_t: float,
    states: Array,
    offset_t: float,
    H: Array,
    model: StateSpaceModel,
    params_obs: ParamDict,
) -> Array:
    eta = float(offset_t) + np.asarray(states, dtype=float) @ np.asarray(H, dtype=float)
    values = np.full(eta.size, -np.inf, dtype=float)
    for index, eta_value in enumerate(eta):
        try:
            value = float(
                model.obs.logpdf(
                    y=float(y_t), eta=float(eta_value), params=params_obs
                )
            )
        except Exception:
            value = -np.inf
        if np.isfinite(value):
            values[index] = value
    return values


def _ncp_guided_statistics(
    y_t: float,
    means: Array,
    offset_t: float,
    H: Array,
    loading: Array,
    model: StateSpaceModel,
    params_obs: ParamDict,
) -> tuple[Array, Array, Array, float]:
    noise_dim = loading.shape[1]
    identity = np.eye(noise_dim)
    if noise_dim == 0:
        return np.zeros((means.shape[0], 0)), identity, identity, 0.0
    direction_in_state = loading.T @ np.asarray(H, dtype=float)
    if np.linalg.norm(direction_in_state) <= np.finfo(float).tiny:
        return np.zeros((means.shape[0], noise_dim)), identity, identity, 0.0
    try:
        gradient = float(
            model.obs.grad_eta(float(y_t), float(y_t), params_obs)
        )
        hessian = float(
            model.obs.hess_eta(float(y_t), float(y_t), params_obs)
        )
    except Exception:
        return np.zeros((means.shape[0], noise_dim)), identity, identity, 0.0
    if not np.isfinite(gradient) or not np.isfinite(hessian):
        return np.zeros((means.shape[0], noise_dim)), identity, identity, 0.0
    information = max(-hessian, 1e-10)
    precision = identity + information * np.outer(
        direction_in_state, direction_in_state
    )
    covariance = np.linalg.inv(precision)
    root = np.linalg.cholesky(_project_psd(covariance, floor=1e-14))
    predicted_eta = float(offset_t) + means @ np.asarray(H, dtype=float)
    coefficient = gradient + information * (float(y_t) - predicted_eta)
    direction = covariance @ direction_in_state
    proposal_mean = coefficient[:, None] * direction[None, :]
    logdet = float(np.linalg.slogdet(covariance)[1])
    return proposal_mean, root, precision, logdet


def _ncp_proposal_draw(
    y_t: float,
    means: Array,
    offset_t: float,
    H: Array,
    loading: Array,
    model: StateSpaceModel,
    params_obs: ParamDict,
    proposal: str,
    rng: np.random.Generator,
) -> tuple[Array, Array]:
    n_particles = means.shape[0]
    noise_dim = loading.shape[1]
    if proposal == "bootstrap":
        disturbance = rng.normal(size=(n_particles, noise_dim))
        return means + disturbance @ loading.T, np.zeros(n_particles)
    proposal_mean, root, precision, logdet = _ncp_guided_statistics(
        y_t, means, offset_t, H, loading, model, params_obs
    )
    centered = rng.normal(size=(n_particles, noise_dim)) @ root.T
    disturbance = proposal_mean + centered
    prior_quadratic = np.sum(disturbance * disturbance, axis=1)
    proposal_quadratic = np.einsum(
        "ni,ij,nj->n", centered, precision, centered
    )
    correction = 0.5 * (logdet + proposal_quadratic - prior_quadratic)
    return means + disturbance @ loading.T, correction


def _ncp_reference_proposal_correction(
    y_t: float,
    reference_t: Array,
    parent: Array,
    offset_t: float,
    H: Array,
    G: Array,
    loading: Array,
    active: Array,
    deterministic: Array,
    model: StateSpaceModel,
    params_obs: ParamDict,
    proposal: str,
) -> float:
    if proposal == "bootstrap" or loading.shape[1] == 0:
        return 0.0
    mean = G @ parent
    transition_logpdf = _ncp_transition_logpdf(
        reference_t, mean, active, deterministic
    )
    if not np.isfinite(transition_logpdf):
        return -np.inf
    residual = np.asarray(reference_t, dtype=float) - mean
    disturbance = residual[active]
    proposal_mean, _, precision, logdet = _ncp_guided_statistics(
        y_t,
        mean[None, :],
        offset_t,
        H,
        loading,
        model,
        params_obs,
    )
    centered = disturbance - proposal_mean[0]
    prior_quadratic = float(disturbance @ disturbance)
    proposal_quadratic = float(centered @ precision @ centered)
    return float(0.5 * (logdet + proposal_quadratic - prior_quadratic))


def ncp_pgas(
    y: Array,
    model: StateSpaceModel,
    params_state: ParamDict,
    params_obs: ParamDict,
    layout: NCPLayout,
    reference: Array,
    *,
    n_particles: int = 256,
    proposal: str = "guided",
    rng: Optional[np.random.Generator] = None,
) -> NCPPGASResult:
    """One exact-invariant PGAS update for the FS non-centred trajectory."""
    if int(n_particles) < 2:
        raise ValueError("n_particles must be at least 2.")
    if proposal not in {"bootstrap", "guided"}:
        raise ValueError("proposal must be 'bootstrap' or 'guided'.")
    rng = np.random.default_rng() if rng is None else rng
    y = np.asarray(y, dtype=float).reshape(-1)
    reference = np.asarray(reference, dtype=float)
    Tn = y.size
    d = layout.ncp_state_dim
    if reference.shape != (Tn + 1, d):
        raise ValueError("reference must have shape (T+1, ncp_state_dim).")
    G, Q = build_ncp_system(layout)
    active, deterministic, loading = _ncp_transition_structure(Q)
    if np.linalg.norm(reference[0]) > 1e-7:
        raise ValueError("The FS non-centred initial state must be zero.")
    transition_means = reference[:-1] @ G.T
    transition_values = _ncp_transition_logpdf(
        reference[1:], transition_means, active, deterministic
    )
    if not np.all(np.isfinite(transition_values)):
        first = int(np.flatnonzero(~np.isfinite(transition_values))[0]) + 1
        raise ValueError(f"Reference path is outside FS transition support at time {first}.")

    offset = baseline_mu_path(Tn, params_state, layout)
    H = measurement_vector(params_state, layout)
    n_particles = int(n_particles)
    conditioned = n_particles - 1
    cloud = np.zeros((Tn + 1, n_particles, d), dtype=float)
    weights = np.full((Tn + 1, n_particles), 1.0 / n_particles, dtype=float)
    log_weights = np.full((Tn + 1, n_particles), -np.log(n_particles), dtype=float)
    ancestors = np.zeros((Tn + 1, n_particles), dtype=int)
    ess = np.zeros(Tn + 1, dtype=float)
    unique = np.zeros(Tn + 1, dtype=int)
    reference_ancestors = np.zeros(Tn + 1, dtype=int)
    ancestors[0] = np.arange(n_particles)
    ess[0] = float(n_particles)
    unique[0] = n_particles
    reference_ancestors[0] = conditioned

    for t in range(1, Tn + 1):
        previous_weights = weights[t - 1]
        parents = rng.choice(
            n_particles, size=conditioned, replace=True, p=previous_weights
        )
        ancestors[t, :conditioned] = parents
        means = cloud[t - 1, parents] @ G.T
        cloud[t, :conditioned], correction = _ncp_proposal_draw(
            float(y[t - 1]),
            means,
            float(offset[t - 1]),
            H,
            loading,
            model,
            params_obs,
            proposal,
            rng,
        )

        previous_means = cloud[t - 1] @ G.T
        log_as = log_weights[t - 1] + _ncp_transition_logpdf(
            np.broadcast_to(reference[t], previous_means.shape),
            previous_means,
            active,
            deterministic,
        )
        ancestor_weights, _, _ = _strict_normalize_logweights(log_as)
        reference_parent = int(rng.choice(n_particles, p=ancestor_weights))
        ancestors[t, conditioned] = reference_parent
        reference_ancestors[t] = reference_parent
        cloud[t, conditioned] = reference[t]
        reference_correction = _ncp_reference_proposal_correction(
            float(y[t - 1]),
            reference[t],
            cloud[t - 1, reference_parent],
            float(offset[t - 1]),
            H,
            G,
            loading,
            active,
            deterministic,
            model,
            params_obs,
            proposal,
        )
        unique[t] = int(np.unique(ancestors[t]).size)
        incremental = _ncp_observation_logweights(
            float(y[t - 1]),
            cloud[t],
            float(offset[t - 1]),
            H,
            model,
            params_obs,
        )
        incremental[:conditioned] += correction
        incremental[conditioned] += reference_correction
        weights[t], log_weights[t], _ = _strict_normalize_logweights(incremental)
        ess[t] = _ess(weights[t])

    index_path = np.zeros(Tn + 1, dtype=int)
    index_path[Tn] = int(rng.choice(n_particles, p=weights[Tn]))
    path = np.zeros_like(reference)
    path[Tn] = cloud[Tn, index_path[Tn]]
    for t in range(Tn, 0, -1):
        index_path[t - 1] = ancestors[t, index_path[t]]
        path[t - 1] = cloud[t - 1, index_path[t - 1]]
    difference = np.linalg.norm(path - reference, axis=1)
    changed = difference > 1e-10 * (
        1.0 + np.linalg.norm(reference, axis=1)
    )
    return NCPPGASResult(
        z_path=path,
        ess=ess,
        unique_ancestors=unique,
        reference_ancestors=reference_ancestors,
        path_changed=bool(np.any(changed)),
        changed_fraction=float(np.mean(changed)),
    )


def _ncp_exact_observation_loglik(
    y: Array,
    z_path: Array,
    params_state: ParamDict,
    params_obs: ParamDict,
    model: StateSpaceModel,
    layout: NCPLayout,
) -> float:
    mu = mu_from_ncp(z_path, params_state, layout)
    total = 0.0
    for observed, location in zip(np.asarray(y, dtype=float), mu):
        try:
            value = float(
                model.obs.logpdf(
                    y=float(observed), eta=float(location), params=params_obs
                )
            )
        except Exception:
            return -np.inf
        if not np.isfinite(value):
            return -np.inf
        total += value
    return float(total)


def _ncp_state_log_density(z_path: Array, G: Array, Q: Array) -> float:
    z_path = np.asarray(z_path, dtype=float)
    if np.linalg.norm(z_path[0]) > 1e-7:
        return -np.inf
    active, deterministic, _ = _ncp_transition_structure(Q)
    values = _ncp_transition_logpdf(
        z_path[1:], z_path[:-1] @ G.T, active, deterministic
    )
    if not np.all(np.isfinite(values)):
        return -np.inf
    return float(np.sum(values))


def _ncp_laplace_pseudo_data(
    y: Array,
    eta: Array,
    model: StateSpaceModel,
    params_obs: ParamDict,
    *,
    curvature_floor: float,
    maximum_variance: float,
    shift_limit: float,
) -> tuple[Array, Array]:
    y = np.asarray(y, dtype=float).reshape(-1)
    eta = np.asarray(eta, dtype=float).reshape(-1)
    pseudo_y = np.zeros(y.size, dtype=float)
    pseudo_variance = np.zeros(y.size, dtype=float)
    for index, (observed, location) in enumerate(zip(y, eta)):
        gradient = float(
            model.obs.grad_eta(float(observed), float(location), params_obs)
        )
        hessian = float(
            model.obs.hess_eta(float(observed), float(location), params_obs)
        )
        if not np.isfinite(gradient) or not np.isfinite(hessian):
            raise FloatingPointError("Invalid GEV derivatives at the Laplace mode.")
        information = max(-hessian, float(curvature_floor))
        pseudo_variance[index] = min(
            1.0 / information, float(maximum_variance)
        )
        pseudo_y[index] = float(location) + float(
            np.clip(gradient / information, -shift_limit, shift_limit)
        )
    return pseudo_y, pseudo_variance


def iterated_laplace_ncp(
    y: Array,
    model: StateSpaceModel,
    params_state: ParamDict,
    params_obs: ParamDict,
    layout: NCPLayout,
    *,
    initial_path: Optional[Array] = None,
    rng: Optional[np.random.Generator] = None,
    max_iterations: int = 30,
    tolerance: float = 1e-5,
    curvature_floor: float = 1e-6,
    maximum_variance: float = 1e8,
    shift_limit: Optional[float] = None,
    draw_attempts: int = 30,
) -> NCPLaplaceResult:
    """Convergence-checked iterated Laplace draw for the FS state block."""
    rng = np.random.default_rng() if rng is None else rng
    y = np.asarray(y, dtype=float).reshape(-1)
    if int(max_iterations) < 1 or float(tolerance) <= 0.0:
        raise ValueError("max_iterations and tolerance must be positive.")
    Tn = y.size
    G, Q = build_ncp_system(layout)
    H = measurement_vector(params_state, layout)
    offset = baseline_mu_path(Tn, params_state, layout)
    if initial_path is None:
        mode = np.zeros((Tn + 1, layout.ncp_state_dim), dtype=float)
    else:
        mode = np.asarray(initial_path, dtype=float).copy()
    if mode.shape != (Tn + 1, layout.ncp_state_dim):
        raise ValueError("initial_path has the wrong shape.")
    mode[0] = 0.0
    shift_limit = float(
        10.0 * float(params_obs["sigma"])
        if shift_limit is None
        else shift_limit
    )

    def objective(path: Array) -> float:
        return _ncp_state_log_density(path, G, Q) + _ncp_exact_observation_loglik(
            y, path, params_state, params_obs, model, layout
        )

    current_objective = objective(mode)
    if not np.isfinite(current_objective):
        raise FloatingPointError(
            "Could not initialize the FS GEV Laplace mode inside the support."
        )
    converged = False
    relative_change = np.inf
    pseudo_y = np.asarray(y, dtype=float).copy()
    pseudo_variance = np.full(Tn, float(params_obs["sigma"]) ** 2)
    iteration = 0

    for iteration in range(1, int(max_iterations) + 1):
        eta = mu_from_ncp(mode, params_state, layout)
        pseudo_y, pseudo_variance = _ncp_laplace_pseudo_data(
            y,
            eta,
            model,
            params_obs,
            curvature_floor=curvature_floor,
            maximum_variance=maximum_variance,
            shift_limit=shift_limit,
        )
        candidate = gaussian_smoother_mean_1d_tvR(
            pseudo_y - offset,
            G,
            Q,
            H,
            pseudo_variance,
            m0=np.zeros(layout.ncp_state_dim),
            C0=np.zeros((layout.ncp_state_dim, layout.ncp_state_dim)),
        )
        candidate[0] = 0.0
        accepted = False
        step_size = 1.0
        proposal = mode
        candidate_objective = -np.inf
        while step_size >= 2.0**-12:
            proposal = mode + step_size * (candidate - mode)
            proposal[0] = 0.0
            candidate_objective = objective(proposal)
            if (
                np.isfinite(candidate_objective)
                and candidate_objective >= current_objective - 1e-8
            ):
                accepted = True
                break
            step_size *= 0.5
        if not accepted:
            relative_change = 0.0
            converged = True
            break
        old_eta = eta
        mode = proposal
        current_objective = candidate_objective
        new_eta = mu_from_ncp(mode, params_state, layout)
        relative_change = float(
            np.max(np.abs(new_eta - old_eta))
            / (1.0 + np.max(np.abs(old_eta)))
        )
        if relative_change < tolerance:
            converged = True
            break

    pseudo_y, pseudo_variance = _ncp_laplace_pseudo_data(
        y,
        mu_from_ncp(mode, params_state, layout),
        model,
        params_obs,
        curvature_floor=curvature_floor,
        maximum_variance=maximum_variance,
        shift_limit=shift_limit,
    )
    path = mode.copy()
    support_rejections = 0
    for _ in range(int(draw_attempts)):
        candidate = ffbs_gaussian_1d_tvR(
            pseudo_y - offset,
            G,
            Q,
            H,
            pseudo_variance,
            m0=np.zeros(layout.ncp_state_dim),
            C0=np.zeros((layout.ncp_state_dim, layout.ncp_state_dim)),
            rng=rng,
        )
        candidate[0] = 0.0
        if np.isfinite(
            _ncp_exact_observation_loglik(
                y, candidate, params_state, params_obs, model, layout
            )
        ):
            path = candidate
            break
        support_rejections += 1
    return NCPLaplaceResult(
        z_path=path,
        mode_path=mode,
        pseudo_y=pseudo_y,
        pseudo_variance=pseudo_variance,
        converged=converged,
        iterations=int(iteration),
        relative_change=float(relative_change),
        objective=float(current_objective),
        support_rejections=int(support_rejections),
    )
