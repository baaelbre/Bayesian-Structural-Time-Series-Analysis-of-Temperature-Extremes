from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ...components.regression import RegressionComponent
from ...components.seasonal import DummySeasonal
from ...components.trend import LocalLinearTrend
from ...core.results import FilterResult, SmootherResult, StateSample
from ...models.base import StateSpaceModel
from ...models.structural import StructuralSSM
from ..state.particle import ParticleConfig

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


def _symmetrize(A: Array) -> Array:
    A = np.asarray(A, dtype=float)
    return 0.5 * (A + A.T)


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
    eye = np.eye(n)
    for k in range(8):
        try:
            L = np.linalg.cholesky(A + (10.0**k) * jitter * eye)
            Y = np.linalg.solve(L, B)
            return np.linalg.solve(L.T, Y)
        except np.linalg.LinAlgError:
            continue
    repaired = _project_psd(A, floor=max(jitter, 1e-12))
    L = np.linalg.cholesky(repaired)
    Y = np.linalg.solve(L, B)
    return np.linalg.solve(L.T, Y)


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
    eye = np.eye(mean.size)
    for power in range(8):
        try:
            L = np.linalg.cholesky(
                covariance + (10.0**power) * max(jitter, 1e-12) * eye
            )
            return mean + L @ rng.normal(size=mean.size)
        except np.linalg.LinAlgError:
            continue
    repaired = _project_psd(covariance, floor=max(jitter, 1e-12))
    L = np.linalg.cholesky(repaired)
    return mean + L @ rng.normal(size=mean.size)


def _weighted_mean_cov(x: Array, w: Array) -> Tuple[Array, Array]:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.sum(w[:, None] * x, axis=0)
    xc = x - m
    C = (xc * w[:, None]).T @ xc
    return m, _symmetrize(C)


def _logsumexp(x: Array) -> float:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return -np.inf
    xf = x[finite]
    m = np.max(xf)
    return float(m + np.log(np.sum(np.exp(xf - m))))


def _normalize_logweights(logw: Array) -> Tuple[Array, float]:
    logw = np.asarray(logw, dtype=float)
    finite = np.isfinite(logw)
    if not np.any(finite):
        N = logw.size
        return np.full(N, 1.0 / N, dtype=float), -np.inf
    xf = logw[finite]
    m = np.max(xf)
    log_norm = float(m + np.log(np.sum(np.exp(xf - m))))
    w = np.zeros_like(logw, dtype=float)
    w[finite] = np.exp(xf - log_norm)
    s = np.sum(w)
    if not np.isfinite(s) or s <= 0.0:
        N = logw.size
        return np.full(N, 1.0 / N, dtype=float), -np.inf
    w /= s
    return w, log_norm


def _ess(w: Array) -> float:
    s2 = float(np.sum(np.square(np.asarray(w, dtype=float))))
    if not np.isfinite(s2) or s2 <= 0.0:
        return 0.0
    return float(1.0 / s2)


def _systematic_resample(w: Array, rng: np.random.Generator) -> Array:
    N = w.size
    positions = (rng.random() + np.arange(N)) / N
    cdf = np.cumsum(w)
    idx = np.searchsorted(cdf, positions, side="right")
    return idx.astype(int)


def _multinomial_resample(w: Array, rng: np.random.Generator) -> Array:
    N = w.size
    return rng.choice(np.arange(N), size=N, replace=True, p=w).astype(int)


def _resample_indices(w: Array, rng: np.random.Generator, method: str = "systematic") -> Array:
    if method == "systematic":
        return _systematic_resample(w, rng)
    if method == "multinomial":
        return _multinomial_resample(w, rng)
    raise ValueError(f"Unknown resampling method '{method}'.")


def _resolve_initial_ncp(d: int) -> Tuple[Array, Array]:
    return np.zeros(d, dtype=float), 1e-6 * np.eye(d)


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
    idx_alpha = names.index("alpha") if "alpha" in names else None
    idx_beta = names.index("beta") if "beta" in names else None
    g_idx = [i for i, nm in enumerate(names) if nm.startswith("g")]
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
        raise NotImplementedError("Non-centred samplers currently require a dynamic level state 'alpha'.")

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
    C[0] = _symmetrize(C0) + jitter * np.eye(d)

    for t in range(1, Tn + 1):
        a[t] = G @ m[t - 1]
        Rm[t] = _symmetrize(G @ C[t - 1] @ G.T + Q) + jitter * np.eye(d)
        F = float((H @ Rm[t] @ H.T).item() + float(R))
        if not np.isfinite(F) or F <= 0.0:
            F = float((H @ (Rm[t] + 1e-10 * np.eye(d)) @ H.T).item() + float(R))
        K = (Rm[t] @ H.T) / F
        v = float(y[t - 1] - (H @ a[t]).item())
        m[t] = a[t] + K[:, 0] * v
        I_KH = np.eye(d) - K @ H
        C[t] = _symmetrize(
            I_KH @ Rm[t] @ I_KH.T + float(R) * (K @ K.T)
        ) + jitter * np.eye(d)

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
    C[0] = _symmetrize(C0) + jitter * np.eye(d)

    for t in range(1, Tn + 1):
        Robs = float(max(R_floor, R_t[t - 1]))
        a[t] = G @ m[t - 1]
        Rm[t] = _symmetrize(G @ C[t - 1] @ G.T + Q) + jitter * np.eye(d)
        F = float((H @ Rm[t] @ H.T).item() + Robs)
        if not np.isfinite(F) or F <= 0.0:
            F = float((H @ (Rm[t] + 1e-10 * np.eye(d)) @ H.T).item() + Robs)
        K = (Rm[t] @ H.T) / F
        v = float(y[t - 1] - (H @ a[t]).item())
        m[t] = a[t] + K[:, 0] * v
        I_KH = np.eye(d) - K @ H
        C[t] = _symmetrize(
            I_KH @ Rm[t] @ I_KH.T + Robs * (K @ K.T)
        ) + jitter * np.eye(d)

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
    """Initialise local scales and global or component-wise shrinkage."""
    if getattr(priors, "lasso", None) is None:
        return {}, np.nan
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

def _prior_mean_precision(
    priors: Any,
    layout: NCPLayout,
    theta_names: list[str],
    *,
    tbar: float,
    tau: Optional[dict[str, float]],
    lasso_variance_scale: float,
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
) -> ParamDict:
    X, names, tbar = design_matrix_ncp(z_path, layout, center_time=True)
    if lasso_variance_scale is None:
        lasso_variance_scale = float(sigma2)
    pm, pp = _prior_mean_precision(
        priors,
        layout,
        names,
        tbar=tbar,
        tau=tau,
        lasso_variance_scale=float(lasso_variance_scale),
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
) -> ParamDict:
    X, names, tbar = design_matrix_ncp(z_path, layout, center_time=True)
    pm, pp = _prior_mean_precision(
        priors,
        layout,
        names,
        tbar=tbar,
        tau=tau,
        lasso_variance_scale=float(lasso_variance_scale),
    )
    theta = _posterior_gaussian_precision(X, z_pseudo, R_t, pm, pp, rng)
    return apply_theta_draw({}, theta, names, layout, tbar=tbar)


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


# ---------- NCP particle machinery for exact non-Gaussian state updates ----------

def _obs_loglik_ncp(y_t: float, z_t: Array, params_state: ParamDict, params_obs: ParamDict, model: StateSpaceModel, layout: NCPLayout, t: int) -> float:
    mu_t = float(baseline_mu_path(1, params_state | {"_tmp_t_offset": None}, layout)[0])  # unused fallback
    # explicit form avoids repeated 1-step baseline path creation
    mu_t = float(params_state.get("alpha0", 0.0)) + float(params_state.get("beta0", 0.0)) * float(t)
    if layout.season_dim > 0:
        # compute baseline seasonal contribution for time t
        g0 = np.asarray(params_state["gamma0_season"], dtype=float).reshape(layout.season_dim)
        S = seasonal_rotation_matrix(layout.season_dim)
        g = g0.copy()
        for _ in range(1, t):
            g = S @ g
        mu_t += float(g[0])
    if layout.idx_tilde_alpha is not None:
        mu_t += float(params_state.get("s_level", 0.0)) * float(z_t[int(layout.idx_tilde_alpha)])
    if layout.idx_A is not None:
        mu_t += float(params_state.get("s_trend", 0.0)) * float(z_t[int(layout.idx_A)])
    if layout.season_dim > 0:
        mu_t += float(params_state.get("s_season", 0.0)) * float(z_t[layout.season_ncp_slice.start])
    try:
        return float(model.obs.logpdf(y=float(y_t), eta=mu_t, params=params_obs))
    except Exception:
        return -np.inf


def _transition_mean_cov_ncp(z_prev: Array, G: Array, Q: Array) -> Tuple[Array, Array]:
    a_t = G @ z_prev
    W_t = _symmetrize(Q)
    return a_t, W_t


def _sample_bootstrap_transition_ncp(z_prev: Array, G: Array, Q: Array, rng: np.random.Generator) -> Array:
    a_t, W_t = _transition_mean_cov_ncp(z_prev, G, Q)
    if W_t.size == 0:
        return a_t.copy()
    return _sample_gaussian(a_t, W_t, rng)


def ncp_particle_filter(
    y: Array,
    model: StateSpaceModel,
    params_state: ParamDict,
    params_obs: ParamDict,
    layout: NCPLayout,
    *,
    n_particles: int = 1000,
    method: str = "bootstrap",
    rng: Optional[np.random.Generator] = None,
    config: Optional[ParticleConfig] = None,
) -> FilterResult:
    y1 = np.asarray(y, dtype=float).reshape(-1)
    Tn = y1.size
    m = layout.centered_state_dim
    d = layout.ncp_state_dim

    cfg = config if config is not None else ParticleConfig(n_particles=n_particles, method=method)
    cfg = ParticleConfig(**{**asdict(cfg), "n_particles": int(n_particles), "method": cfg.method if config is not None else method})
    method = cfg.method
    if method not in {"bootstrap", "auxiliary"}:
        raise ValueError("ncp_particle_filter currently supports only 'bootstrap' and 'auxiliary'.")
    rng = rng if rng is not None else np.random.default_rng()

    G, Q = build_ncp_system(layout)
    z0_mean, z0_cov = _resolve_initial_ncp(d)

    z0_particles = np.zeros((cfg.n_particles, d), dtype=float)
    for i in range(cfg.n_particles):
        z0_particles[i] = _sample_gaussian(z0_mean, z0_cov, rng)
    x0_particles = np.stack([map_ncp_to_centered(z0_particles[i:i+1].repeat(1, axis=0), params_state, layout)[0] for i in range(cfg.n_particles)], axis=0)

    w0 = np.full(cfg.n_particles, 1.0 / cfg.n_particles, dtype=float)

    z_particles = np.zeros((Tn + 1, cfg.n_particles, d), dtype=float)
    x_particles = np.zeros((Tn + 1, cfg.n_particles, m), dtype=float)
    weights = np.zeros((Tn + 1, cfg.n_particles), dtype=float)
    ancestors = -np.ones((Tn + 1, cfg.n_particles), dtype=int)
    ess_hist = np.zeros(Tn + 1, dtype=float)
    resampled_hist = np.zeros(Tn + 1, dtype=bool)
    collapsed_hist = np.zeros(Tn + 1, dtype=bool)

    z_particles[0] = z0_particles
    x_particles[0] = np.stack([map_ncp_to_centered(z_particles[0, i:i+1], params_state, layout)[0] for i in range(cfg.n_particles)], axis=0)
    weights[0] = w0
    ess_hist[0] = _ess(w0)

    m_pred = np.zeros((Tn + 1, m), dtype=float)
    P_pred = np.zeros((Tn + 1, m, m), dtype=float)
    m_filt = np.zeros((Tn + 1, m), dtype=float)
    P_filt = np.zeros((Tn + 1, m, m), dtype=float)
    m_pred[0], P_pred[0] = _weighted_mean_cov(x_particles[0], w0)
    m_filt[0], P_filt[0] = m_pred[0], P_pred[0]

    loglik = 0.0
    for t in range(1, Tn + 1):
        y_t = float(y1[t - 1])
        z_prev = z_particles[t - 1]
        w_prev = weights[t - 1]

        do_resample = cfg.resample_every_step or (_ess(w_prev) < cfg.ess_threshold * cfg.n_particles)
        if method == "auxiliary":
            do_resample = True

        if method == "bootstrap":
            if do_resample:
                anc_idx = _resample_indices(w_prev, rng, method=cfg.resample_method)
                base_particles = z_prev[anc_idx]
                base_logw = np.full(cfg.n_particles, -np.log(cfg.n_particles), dtype=float)
                pred_w = np.full(cfg.n_particles, 1.0 / cfg.n_particles, dtype=float)
                resampled_hist[t] = True
            else:
                anc_idx = np.arange(cfg.n_particles, dtype=int)
                base_particles = z_prev
                with np.errstate(divide="ignore"):
                    base_logw = np.where(w_prev > 0.0, np.log(w_prev), -np.inf)
                pred_w = w_prev
                resampled_hist[t] = False

            z_prop = np.zeros((cfg.n_particles, d), dtype=float)
            x_prop = np.zeros((cfg.n_particles, m), dtype=float)
            logw = np.zeros(cfg.n_particles, dtype=float)
            for i in range(cfg.n_particles):
                z_i = _sample_bootstrap_transition_ncp(base_particles[i], G, Q, rng)
                z_prop[i] = z_i
                x_prop[i] = map_ncp_to_centered(z_i[None, :], params_state, layout)[0]
                ll_i = _obs_loglik_ncp(y_t, z_i, params_state, params_obs, model, layout, t)
                logw[i] = base_logw[i] + ll_i

            m_pred[t], P_pred[t] = _weighted_mean_cov(x_prop, pred_w)
            w_t, log_norm = _normalize_logweights(logw)
            if not np.isfinite(log_norm):
                collapsed_hist[t] = True
            else:
                if do_resample:
                    loglik += float(log_norm - np.log(cfg.n_particles))
                else:
                    loglik += float(log_norm)

            z_particles[t] = z_prop
            x_particles[t] = x_prop
            weights[t] = w_t
            ancestors[t] = anc_idx

        else:  # auxiliary
            log_aux = np.zeros(cfg.n_particles, dtype=float)
            for i in range(cfg.n_particles):
                a_i, _ = _transition_mean_cov_ncp(z_prev[i], G, Q)
                ll_look = _obs_loglik_ncp(y_t, a_i, params_state, params_obs, model, layout, t)
                with np.errstate(divide="ignore"):
                    lw_prev_i = np.log(w_prev[i]) if w_prev[i] > 0.0 else -np.inf
                log_aux[i] = lw_prev_i + ll_look
            aux_w, log_c = _normalize_logweights(log_aux)
            if not np.isfinite(log_c):
                collapsed_hist[t] = True
                aux_w = np.full(cfg.n_particles, 1.0 / cfg.n_particles, dtype=float)
            anc_idx = _resample_indices(aux_w, rng, method=cfg.resample_method)
            base_particles = z_prev[anc_idx]

            z_prop = np.zeros((cfg.n_particles, d), dtype=float)
            x_prop = np.zeros((cfg.n_particles, m), dtype=float)
            logw_corr = np.zeros(cfg.n_particles, dtype=float)
            for i in range(cfg.n_particles):
                z_i = _sample_bootstrap_transition_ncp(base_particles[i], G, Q, rng)
                z_prop[i] = z_i
                x_prop[i] = map_ncp_to_centered(z_i[None, :], params_state, layout)[0]
                ll_i = _obs_loglik_ncp(y_t, z_i, params_state, params_obs, model, layout, t)
                with np.errstate(divide="ignore"):
                    lw_prev_anc = np.log(w_prev[anc_idx[i]]) if w_prev[anc_idx[i]] > 0.0 else -np.inf
                ll_look = float(log_aux[anc_idx[i]] - lw_prev_anc)
                logw_corr[i] = ll_i - ll_look

            m_pred[t], P_pred[t] = _weighted_mean_cov(x_prop, np.full(cfg.n_particles, 1.0 / cfg.n_particles, dtype=float))
            w_t, log_norm = _normalize_logweights(logw_corr)
            if not np.isfinite(log_norm):
                collapsed_hist[t] = True
            else:
                if np.isfinite(log_c):
                    loglik += float(log_c + log_norm - np.log(cfg.n_particles))

            z_particles[t] = z_prop
            x_particles[t] = x_prop
            weights[t] = w_t
            ancestors[t] = anc_idx
            resampled_hist[t] = True

        m_filt[t], P_filt[t] = _weighted_mean_cov(x_particles[t], weights[t])
        ess_hist[t] = _ess(weights[t])

    return FilterResult(
        y=y1[:, None],
        m0=m_filt[0],
        P0=P_filt[0],
        m_pred=m_pred,
        P_pred=P_pred,
        m_filt=m_filt,
        P_filt=P_filt,
        loglik=float(loglik),
        missing=np.zeros(Tn, dtype=bool),
        meta={
            "backend": "ncp_particle_filter",
            "method": method,
            "exact": False,
            "parameterization": "noncentered",
            "n_particles": cfg.n_particles,
            "ess": ess_hist,
            "resampled": resampled_hist,
            "collapsed": collapsed_hist,
            "z_particles": z_particles,
            "x_particles": x_particles,
            "weights": weights,
            "ancestors": ancestors,
            "config": asdict(cfg),
        },
    )


def _trajectory_from_genealogy_ncp(z_particles: Array, x_particles: Array, ancestors: Array, final_weights: Array, rng: np.random.Generator) -> Tuple[Array, Array, Array]:
    Tn = z_particles.shape[0] - 1
    d = z_particles.shape[2]
    m = x_particles.shape[2]
    idx_path = np.zeros(Tn + 1, dtype=int)
    z_path = np.zeros((Tn + 1, d), dtype=float)
    x_path = np.zeros((Tn + 1, m), dtype=float)
    idx_path[Tn] = int(rng.choice(np.arange(final_weights.size), p=final_weights))
    z_path[Tn] = z_particles[Tn, idx_path[Tn]]
    x_path[Tn] = x_particles[Tn, idx_path[Tn]]
    for t in range(Tn, 0, -1):
        idx_path[t - 1] = ancestors[t, idx_path[t]]
        z_path[t - 1] = z_particles[t - 1, idx_path[t - 1]]
        x_path[t - 1] = x_particles[t - 1, idx_path[t - 1]]
    return z_path, x_path, idx_path


def ncp_particle_state_sample(
    y: Array,
    model: StateSpaceModel,
    params_state: ParamDict,
    params_obs: ParamDict,
    layout: NCPLayout,
    *,
    rng: Optional[np.random.Generator] = None,
    n_particles: int = 1000,
    method: str = "bootstrap",
    config: Optional[ParticleConfig] = None,
    filter_result: Optional[FilterResult] = None,
) -> StateSample:
    rng = rng if rng is not None else np.random.default_rng()
    if filter_result is None:
        filter_result = ncp_particle_filter(
            y=y,
            model=model,
            params_state=params_state,
            params_obs=params_obs,
            layout=layout,
            n_particles=n_particles,
            method=method,
            rng=rng,
            config=config,
        )
    meta = filter_result.meta
    z_particles = np.asarray(meta["z_particles"], dtype=float)
    x_particles = np.asarray(meta["x_particles"], dtype=float)
    weights = np.asarray(meta["weights"], dtype=float)
    ancestors = np.asarray(meta["ancestors"], dtype=int)
    z_path, x_path, idx_path = _trajectory_from_genealogy_ncp(z_particles, x_particles, ancestors, weights[-1], rng)
    return StateSample(
        x=x_path,
        filter_result=filter_result,
        smoother_result=None,
        meta={
            "backend": "ncp_particle_state_sample",
            "method": meta.get("method", method),
            "parameterization": "noncentered",
            "particle_index_path": idx_path,
            "z_path": z_path,
        },
    )


def ncp_particle_smoother(
    y: Array,
    model: StateSpaceModel,
    params_state: ParamDict,
    params_obs: ParamDict,
    layout: NCPLayout,
    *,
    n_particles: int = 1000,
    method: str = "bootstrap",
    rng: Optional[np.random.Generator] = None,
    n_smoother_draws: int = 100,
    config: Optional[ParticleConfig] = None,
    filter_result: Optional[FilterResult] = None,
) -> SmootherResult:
    rng = rng if rng is not None else np.random.default_rng()
    if filter_result is None:
        filter_result = ncp_particle_filter(
            y=y,
            model=model,
            params_state=params_state,
            params_obs=params_obs,
            layout=layout,
            n_particles=n_particles,
            method=method,
            rng=rng,
            config=config,
        )
    meta = filter_result.meta
    z_particles = np.asarray(meta["z_particles"], dtype=float)
    x_particles = np.asarray(meta["x_particles"], dtype=float)
    weights = np.asarray(meta["weights"], dtype=float)
    ancestors = np.asarray(meta["ancestors"], dtype=int)
    Tn = x_particles.shape[0] - 1
    m = x_particles.shape[2]
    draws = np.zeros((n_smoother_draws, Tn + 1, m), dtype=float)
    for j in range(n_smoother_draws):
        _, x_path, _ = _trajectory_from_genealogy_ncp(z_particles, x_particles, ancestors, weights[-1], rng)
        draws[j] = x_path
    m_smooth = np.mean(draws, axis=0)
    P_smooth = np.zeros((Tn + 1, m, m), dtype=float)
    for t in range(Tn + 1):
        xc = draws[:, t, :] - m_smooth[t]
        P_smooth[t] = _symmetrize((xc.T @ xc) / max(n_smoother_draws - 1, 1))
    return SmootherResult(
        m_smooth=m_smooth,
        P_smooth=P_smooth,
        filter_result=filter_result,
        meta={
            "backend": "ncp_particle_smoother",
            "method": meta.get("method", method),
            "parameterization": "noncentered",
            "n_smoother_draws": n_smoother_draws,
        },
    )
