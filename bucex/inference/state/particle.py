from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ...core.results import FilterResult, SmootherResult, StateSample
from ...models.base import LinearDesign, LinearGaussianSystem, StateSpaceModel

Array = np.ndarray
ParamDict = Dict[str, Any]


# ============================================================================
# Config
# ============================================================================

@dataclass
class ParticleConfig:
    """
    Simple particle filter configuration.

    method
    ------
    - "bootstrap"
    - "auxiliary"
    """
    n_particles: int = 1000
    method: str = "bootstrap"

    ess_threshold: float = 0.5
    resample_every_step: bool = False
    resample_method: str = "systematic"

    # particle smoother
    n_smoother_draws: int = 100

    # progress
    verbose: bool = True
    use_tqdm: bool = True
    progress_desc: str = "particle"


# ============================================================================
# Basic helpers
# ============================================================================

def _as_2d_y(y: Array) -> Array:
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        return y[:, None]
    if y.ndim == 2:
        return y
    raise ValueError("y must be 1D or 2D.")


def _symmetrize(A: Array) -> Array:
    return 0.5 * (A + A.T)


def _logsumexp(x: Array) -> float:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return -np.inf
    xf = x[finite]
    m = np.max(xf)
    return float(m + np.log(np.sum(np.exp(xf - m))))


def _normalize_logweights(logw: Array) -> Tuple[Array, float]:
    """
    Normalize log-weights robustly.

    Returns
    -------
    w : normalized weights
    log_norm : log(sum(exp(logw)))

    Notes
    -----
    If all log-weights are -inf (complete collapse), return uniform weights
    and log_norm = -inf.
    """
    logw = np.asarray(logw, dtype=float)
    finite = np.isfinite(logw)

    if not np.any(finite):
        N = logw.size
        return np.full(N, 1.0 / N, dtype=float), -np.inf

    logw_f = logw[finite]
    m = np.max(logw_f)
    log_norm = float(m + np.log(np.sum(np.exp(logw_f - m))))

    w = np.zeros_like(logw, dtype=float)
    w[finite] = np.exp(logw_f - log_norm)

    s = np.sum(w)
    if not np.isfinite(s) or s <= 0.0:
        N = logw.size
        return np.full(N, 1.0 / N, dtype=float), -np.inf

    w /= s
    return w, log_norm


def _ess(w: Array) -> float:
    w = np.asarray(w, dtype=float)
    s2 = np.sum(np.square(w))
    if not np.isfinite(s2) or s2 <= 0.0:
        return 0.0
    return float(1.0 / s2)


def _weighted_mean_cov(x: Array, w: Array) -> Tuple[Array, Array]:
    """
    Weighted mean and covariance of particles x of shape (N, m).
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.sum(w[:, None] * x, axis=0)
    xc = x - m
    C = (xc * w[:, None]).T @ xc
    return m, _symmetrize(C)


def _systematic_resample(w: Array, rng: np.random.Generator) -> Array:
    N = w.size
    positions = (rng.random() + np.arange(N)) / N
    cdf = np.cumsum(w)
    idx = np.searchsorted(cdf, positions, side="right")
    return idx.astype(int)


def _multinomial_resample(w: Array, rng: np.random.Generator) -> Array:
    N = w.size
    return rng.choice(np.arange(N), size=N, replace=True, p=w).astype(int)


def _resample_indices(
    w: Array,
    rng: np.random.Generator,
    method: str = "systematic",
) -> Array:
    if method == "systematic":
        return _systematic_resample(w, rng)
    if method == "multinomial":
        return _multinomial_resample(w, rng)
    raise ValueError(f"Unknown resampling method '{method}'.")


def _sample_mvn(
    mean: Array,
    cov: Array,
    rng: np.random.Generator,
    jitter: float = 1e-10,
    max_tries: int = 6,
) -> Array:
    mean = np.asarray(mean, dtype=float)
    cov = _symmetrize(np.asarray(cov, dtype=float))
    d = mean.size

    for k in range(max_tries):
        try:
            cov_try = _symmetrize(cov + (10.0**k) * jitter * np.eye(d))
            return rng.multivariate_normal(mean=mean, cov=cov_try, check_valid="raise")
        except Exception:
            continue

    raise np.linalg.LinAlgError("Failed to sample from multivariate normal.")


def _progress_iter(iterable, *, enabled: bool, use_tqdm: bool, desc: str):
    """
    Wrap iterable in tqdm if requested and available.
    """
    if not enabled:
        return iterable

    if use_tqdm:
        try:
            from tqdm.auto import tqdm
            return tqdm(iterable, desc=desc, leave=False)
        except Exception:
            pass

    return iterable


# ============================================================================
# Model helpers
# ============================================================================

def _resolve_initial_state(
    model: StateSpaceModel,
    params_state: ParamDict,
) -> Tuple[Array, Array]:
    m0, P0 = model.initial_state(params_state)
    m0 = np.asarray(m0, dtype=float).reshape(model.state_dim)
    P0 = np.asarray(P0, dtype=float).reshape(model.state_dim, model.state_dim)
    return m0, _symmetrize(P0)


def _obs_loglik_from_state(
    y_t: Array,
    x_t: Array,
    model: StateSpaceModel,
    t: int,
    params_state: ParamDict,
    params_obs: ParamDict,
    exog_t: Optional[Array] = None,
) -> float:
    """
    Observation log-likelihood at time t for one latent state x_t.
    """
    des: LinearDesign = model.design(t=t, params_state=params_state, exog_t=exog_t)
    eta_t = des.Z @ x_t + des.d

    op = model.obs_params(
        t=t,
        x_t=x_t,
        eta_t=eta_t,
        params_obs=params_obs,
        exog_t=exog_t,
    )

    y_arr = np.asarray(y_t, dtype=float).reshape(-1)
    eta_arr = np.asarray(eta_t, dtype=float).reshape(-1)

    if y_arr.size == 1:
        return float(model.obs.logpdf(y=float(y_arr[0]), eta=float(eta_arr[0]), params=op))

    raise NotImplementedError("Current particle backend supports only scalar observation models.")


def _transition_mean_cov(
    x_prev: Array,
    model: StateSpaceModel,
    t: int,
    params_state: ParamDict,
) -> Tuple[Array, Array, LinearGaussianSystem]:
    """
    Return prior mean a_t and covariance W_t for x_t | x_{t-1}.
    """
    sys: LinearGaussianSystem = model.system(t=t, params_state=params_state)
    T_t = np.asarray(sys.T, dtype=float)
    R_t = np.asarray(sys.R, dtype=float)
    Q_t = np.asarray(sys.Q, dtype=float)
    c_t = np.asarray(sys.c, dtype=float)

    a_t = T_t @ x_prev + c_t
    W_t = _symmetrize(R_t @ Q_t @ R_t.T)

    return a_t, W_t, sys


def _sample_bootstrap_transition(
    x_prev: Array,
    model: StateSpaceModel,
    t: int,
    params_state: ParamDict,
    rng: np.random.Generator,
) -> Tuple[Array, LinearGaussianSystem]:
    """
    Sample x_t from the latent transition:
      x_t = T x_{t-1} + c + R eps, eps ~ N(0, Q)
    """
    a_t, _, sys = _transition_mean_cov(x_prev, model, t, params_state)

    R_t = np.asarray(sys.R, dtype=float)
    Q_t = np.asarray(sys.Q, dtype=float)

    r = Q_t.shape[0]
    if r == 0:
        return a_t.copy(), sys

    eps = _sample_mvn(
        mean=np.zeros(r, dtype=float),
        cov=Q_t,
        rng=rng,
    )
    x_t = a_t + R_t @ eps
    return x_t, sys


# ============================================================================
# Future extension hooks
# ============================================================================

def _sample_local_laplace_transition(*args, **kwargs):
    raise NotImplementedError("Local Laplace particle proposal not implemented yet.")


def _liu_west_parameter_step(*args, **kwargs):
    raise NotImplementedError("Liu-West parameter step not implemented yet.")


# ============================================================================
# Main particle filter
# ============================================================================

def particle_filter(
    y: Array,
    model: StateSpaceModel,
    params_state: ParamDict,
    params_obs: ParamDict,
    exog: Optional[Array] = None,
    n_particles: int = 1000,
    *,
    method: str = "bootstrap",
    rng: Optional[np.random.Generator] = None,
    config: Optional[ParticleConfig] = None,
) -> FilterResult:
    """
    Particle filter.

    Supported methods
    -----------------
    - "bootstrap"
    - "auxiliary"
    """
    y2 = _as_2d_y(y)
    Tn, _ = y2.shape
    m = model.state_dim

    if exog is not None:
        exog = np.asarray(exog, dtype=float)
        if exog.shape[0] != Tn:
            raise ValueError("exog must have the same number of rows as y.")

    cfg = config if config is not None else ParticleConfig(n_particles=n_particles, method=method)
    cfg = ParticleConfig(
        **{
            **asdict(cfg),
            "n_particles": int(n_particles),
            "method": cfg.method if config is not None else method,
        }
    )
    method = cfg.method

    if method not in {"bootstrap", "auxiliary"}:
        raise ValueError("particle_filter currently supports only 'bootstrap' and 'auxiliary'.")

    rng = rng if rng is not None else np.random.default_rng()

    m0, P0 = _resolve_initial_state(model, params_state)

    x0_particles = np.zeros((cfg.n_particles, m), dtype=float)
    for i in range(cfg.n_particles):
        x0_particles[i] = _sample_mvn(m0, P0, rng)

    w0 = np.full(cfg.n_particles, 1.0 / cfg.n_particles, dtype=float)

    particles = np.zeros((Tn + 1, cfg.n_particles, m), dtype=float)
    weights = np.zeros((Tn + 1, cfg.n_particles), dtype=float)
    ancestors = -np.ones((Tn + 1, cfg.n_particles), dtype=int)
    ess_hist = np.zeros(Tn + 1, dtype=float)
    resampled_hist = np.zeros(Tn + 1, dtype=bool)
    collapsed_hist = np.zeros(Tn + 1, dtype=bool)

    particles[0] = x0_particles
    weights[0] = w0
    ess_hist[0] = _ess(w0)

    m_pred = np.zeros((Tn + 1, m), dtype=float)
    P_pred = np.zeros((Tn + 1, m, m), dtype=float)
    m_filt = np.zeros((Tn + 1, m), dtype=float)
    P_filt = np.zeros((Tn + 1, m, m), dtype=float)

    m_pred[0] = m0
    P_pred[0] = P0
    m_filt[0], P_filt[0] = _weighted_mean_cov(x0_particles, w0)

    loglik = 0.0

    time_iter = _progress_iter(
        range(1, Tn + 1),
        enabled=cfg.verbose,
        use_tqdm=cfg.use_tqdm,
        desc=f"{cfg.progress_desc}:{method}:filter",
    )

    for t in time_iter:
        y_t = y2[t - 1]
        exog_t = None if exog is None else exog[t - 1]

        x_prev = particles[t - 1]
        w_prev = weights[t - 1]

        do_resample = (
            cfg.resample_every_step
            or (_ess(w_prev) < cfg.ess_threshold * cfg.n_particles)
        )
        if method == "auxiliary":
            do_resample = True

        if method == "bootstrap":
            if do_resample:
                anc_idx = _resample_indices(w_prev, rng, method=cfg.resample_method)
                base_particles = x_prev[anc_idx]
                base_logw = np.full(cfg.n_particles, -np.log(cfg.n_particles), dtype=float)
                pred_w = np.full(cfg.n_particles, 1.0 / cfg.n_particles, dtype=float)
                resampled_hist[t] = True
            else:
                anc_idx = np.arange(cfg.n_particles, dtype=int)
                base_particles = x_prev
                with np.errstate(divide="ignore"):
                    base_logw = np.where(w_prev > 0.0, np.log(w_prev), -np.inf)
                pred_w = w_prev
                resampled_hist[t] = False

            x_prop = np.zeros((cfg.n_particles, m), dtype=float)
            logw = np.zeros(cfg.n_particles, dtype=float)

            for i in range(cfg.n_particles):
                x_i, _ = _sample_bootstrap_transition(
                    x_prev=base_particles[i],
                    model=model,
                    t=t,
                    params_state=params_state,
                    rng=rng,
                )
                x_prop[i] = x_i

                ll_i = _obs_loglik_from_state(
                    y_t=y_t,
                    x_t=x_i,
                    model=model,
                    t=t,
                    params_state=params_state,
                    params_obs=params_obs,
                    exog_t=exog_t,
                )
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

            particles[t] = x_prop
            weights[t] = w_t
            ancestors[t] = anc_idx

        elif method == "auxiliary":
            log_aux = np.zeros(cfg.n_particles, dtype=float)

            for i in range(cfg.n_particles):
                a_i, _, _ = _transition_mean_cov(
                    x_prev=x_prev[i],
                    model=model,
                    t=t,
                    params_state=params_state,
                )
                ll_look = _obs_loglik_from_state(
                    y_t=y_t,
                    x_t=a_i,
                    model=model,
                    t=t,
                    params_state=params_state,
                    params_obs=params_obs,
                    exog_t=exog_t,
                )
                with np.errstate(divide="ignore"):
                    lw_prev_i = np.log(w_prev[i]) if w_prev[i] > 0.0 else -np.inf
                log_aux[i] = lw_prev_i + ll_look

            aux_w, log_c = _normalize_logweights(log_aux)
            if not np.isfinite(log_c):
                collapsed_hist[t] = True
                aux_w = np.full(cfg.n_particles, 1.0 / cfg.n_particles, dtype=float)

            anc_idx = _resample_indices(aux_w, rng, method=cfg.resample_method)
            base_particles = x_prev[anc_idx]

            x_prop = np.zeros((cfg.n_particles, m), dtype=float)
            logw_corr = np.zeros(cfg.n_particles, dtype=float)

            for i in range(cfg.n_particles):
                x_i, _ = _sample_bootstrap_transition(
                    x_prev=base_particles[i],
                    model=model,
                    t=t,
                    params_state=params_state,
                    rng=rng,
                )
                x_prop[i] = x_i

                ll_i = _obs_loglik_from_state(
                    y_t=y_t,
                    x_t=x_i,
                    model=model,
                    t=t,
                    params_state=params_state,
                    params_obs=params_obs,
                    exog_t=exog_t,
                )

                with np.errstate(divide="ignore"):
                    lw_prev_anc = np.log(w_prev[anc_idx[i]]) if w_prev[anc_idx[i]] > 0.0 else -np.inf
                ll_look = float(log_aux[anc_idx[i]] - lw_prev_anc)
                logw_corr[i] = ll_i - ll_look

            m_pred[t], P_pred[t] = _weighted_mean_cov(
                x_prop,
                np.full(cfg.n_particles, 1.0 / cfg.n_particles, dtype=float),
            )

            w_t, log_norm = _normalize_logweights(logw_corr)
            if not np.isfinite(log_norm):
                collapsed_hist[t] = True
            else:
                if np.isfinite(log_c):
                    loglik += float(log_c + log_norm - np.log(cfg.n_particles))

            particles[t] = x_prop
            weights[t] = w_t
            ancestors[t] = anc_idx
            resampled_hist[t] = True

        m_filt[t], P_filt[t] = _weighted_mean_cov(particles[t], weights[t])
        ess_hist[t] = _ess(weights[t])

        if cfg.verbose and not cfg.use_tqdm:
            every = max(1, Tn // 20)
            if (t % every == 0) or (t == Tn):
                print(
                    f"[particle filter {method}] "
                    f"t={t}/{Tn} "
                    f"ESS={ess_hist[t]:.1f} "
                    f"collapsed={bool(collapsed_hist[t])}"
                )

    meta: Dict[str, Any] = {
        "backend": "particle_filter",
        "method": method,
        "exact": False,
        "n_particles": cfg.n_particles,
        "ess": ess_hist,
        "resampled": resampled_hist,
        "collapsed": collapsed_hist,
        "particles": particles,
        "weights": weights,
        "ancestors": ancestors,
        "config": asdict(cfg),
    }

    return FilterResult(
        y=y2,
        m0=m0,
        P0=P0,
        m_pred=m_pred,
        P_pred=P_pred,
        m_filt=m_filt,
        P_filt=P_filt,
        loglik=float(loglik),
        innovations=None,
        innovation_cov=None,
        kalman_gain=None,
        missing=np.zeros(Tn, dtype=bool),
        T_seq=None,
        c_seq=None,
        Z_seq=None,
        d_seq=None,
        H_seq=None,
        meta=meta,
    )


# ============================================================================
# Ancestor tracing smoother / sampler
# ============================================================================

def _trajectory_from_genealogy(
    particles: Array,
    ancestors: Array,
    final_weights: Array,
    rng: np.random.Generator,
) -> Tuple[Array, Array]:
    """
    Sample one trajectory by drawing a final index from final_weights and
    tracing ancestors backwards.

    Returns
    -------
    x_path : (T+1, m)
    idx_path : (T+1,)
    """
    Tn = particles.shape[0] - 1
    m = particles.shape[2]

    idx_path = np.zeros(Tn + 1, dtype=int)
    x_path = np.zeros((Tn + 1, m), dtype=float)

    idx_path[Tn] = int(rng.choice(np.arange(final_weights.size), p=final_weights))
    x_path[Tn] = particles[Tn, idx_path[Tn]]

    for t in range(Tn, 0, -1):
        idx_path[t - 1] = ancestors[t, idx_path[t]]
        x_path[t - 1] = particles[t - 1, idx_path[t - 1]]

    return x_path, idx_path


def particle_state_sample(
    y: Array,
    model: StateSpaceModel,
    params_state: ParamDict,
    params_obs: ParamDict,
    exog: Optional[Array] = None,
    rng: Optional[np.random.Generator] = None,
    n_particles: int = 1000,
    *,
    method: str = "bootstrap",
    config: Optional[ParticleConfig] = None,
    filter_result: Optional[FilterResult] = None,
) -> StateSample:
    """
    Sample one state trajectory from the particle genealogy by ancestor tracing.
    """
    rng = rng if rng is not None else np.random.default_rng()

    if filter_result is None:
        filter_result = particle_filter(
            y=y,
            model=model,
            params_state=params_state,
            params_obs=params_obs,
            exog=exog,
            n_particles=n_particles,
            method=method,
            rng=rng,
            config=config,
        )

    meta = filter_result.meta
    particles = np.asarray(meta["particles"], dtype=float)
    weights = np.asarray(meta["weights"], dtype=float)
    ancestors = np.asarray(meta["ancestors"], dtype=int)

    x_path, idx_path = _trajectory_from_genealogy(
        particles=particles,
        ancestors=ancestors,
        final_weights=weights[-1],
        rng=rng,
    )

    return StateSample(
        x=x_path,
        filter_result=filter_result,
        smoother_result=None,
        meta={
            "backend": "particle_state_sample",
            "method": meta.get("method", method),
            "particle_index_path": idx_path,
        },
    )


def particle_smoother(
    y: Array,
    model: StateSpaceModel,
    params_state: ParamDict,
    params_obs: ParamDict,
    exog: Optional[Array] = None,
    n_particles: int = 1000,
    *,
    method: str = "bootstrap",
    rng: Optional[np.random.Generator] = None,
    n_smoother_draws: int = 100,
    config: Optional[ParticleConfig] = None,
    filter_result: Optional[FilterResult] = None,
) -> SmootherResult:
    """
    Approximate particle smoother via repeated ancestor-trace trajectory draws.
    """
    rng = rng if rng is not None else np.random.default_rng()

    if filter_result is None:
        filter_result = particle_filter(
            y=y,
            model=model,
            params_state=params_state,
            params_obs=params_obs,
            exog=exog,
            n_particles=n_particles,
            method=method,
            rng=rng,
            config=config,
        )

    meta = filter_result.meta
    particles = np.asarray(meta["particles"], dtype=float)
    weights = np.asarray(meta["weights"], dtype=float)
    ancestors = np.asarray(meta["ancestors"], dtype=int)

    Tn = particles.shape[0] - 1
    m = particles.shape[2]

    if config is not None:
        verbose = config.verbose
        use_tqdm = config.use_tqdm
        progress_desc = config.progress_desc
    else:
        verbose = False
        use_tqdm = True
        progress_desc = "particle"

    draws = np.zeros((n_smoother_draws, Tn + 1, m), dtype=float)

    draw_iter = _progress_iter(
        range(n_smoother_draws),
        enabled=verbose,
        use_tqdm=use_tqdm,
        desc=f"{progress_desc}:{method}:smoother",
    )

    for s in draw_iter:
        draws[s], _ = _trajectory_from_genealogy(
            particles=particles,
            ancestors=ancestors,
            final_weights=weights[-1],
            rng=rng,
        )

        if verbose and not use_tqdm:
            every = max(1, n_smoother_draws // 20)
            if ((s + 1) % every == 0) or (s + 1 == n_smoother_draws):
                print(f"[particle smoother {method}] draw={s+1}/{n_smoother_draws}")

    m_smooth = np.mean(draws, axis=0)
    P_smooth = np.zeros((Tn + 1, m, m), dtype=float)

    for t in range(Tn + 1):
        xc = draws[:, t, :] - m_smooth[t]
        P_smooth[t] = _symmetrize((xc.T @ xc) / max(n_smoother_draws - 1, 1))

    return SmootherResult(
        m_smooth=m_smooth,
        P_smooth=P_smooth,
        smoother_gain=None,
        lag_cov=None,
        filter_result=filter_result,
        meta={
            "backend": "particle_smoother",
            "method": meta.get("method", method),
            "n_smoother_draws": int(n_smoother_draws),
        },
    )