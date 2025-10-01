from __future__ import annotations

import os, math, json, time
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, List, Sequence

import numpy as np
from numpy.linalg import inv, cholesky
from scipy.stats import invgamma
from tqdm import tqdm
from datetime import datetime

# =============================================================================
# Utilities
# =============================================================================

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_seasonal(period: int) -> np.ndarray:
    """
    Default smooth seasonal for first (p-1) entries (last is implied by sum-to-zero).
    """
    g = np.cos(2 * np.pi * np.arange(period) / period)
    g -= np.mean(g)
    return g[: period - 1].astype(float)


# =============================================================================
# Priors & Config (Gaussian DLM)
# =============================================================================

@dataclass
class Priors:
    # Observation variance: sigma^2 ~ IG(a_sigma, b_sigma)
    a_sigma: float = 1.0
    b_sigma: float = 1.0

    # Process noises for dynamic coords: Q_k ~ IG(a_q, b_q) (shape, scale)
    # E[Q]=b/(a-1) for a>1 ; Var[Q]=b^2/((a-1)^2 (a-2)) for a>2
    a_q_alpha: float = 1.1
    b_q_alpha: float = 1e-3
    a_q_beta:  float = 1.1
    b_q_beta:  float = 1e-6
    a_q_gamma: float = 1.5
    b_q_gamma: float = 5e-4

    # Deterministic components’ Gaussian priors
    m_level: float = 0.0
    s2_level: float = 100.0
    m_slope: float = 0.0
    s2_slope: float = 100.0

    # Deterministic season prior (first p-1 means; last implied)
    m_season: Optional[Sequence[float]] = None
    s2_season: float = 25.0


@dataclass
class SamplerConfig:
    n_iter: int = 4000
    burn: int = 1000
    thin: int = 2

    random_seed: Optional[int] = 123
    progress: bool = True
    progress_every: int = 0  # 0 => auto (~2% of n_iter)


# =============================================================================
# Core Kalman FFBS helpers
# =============================================================================

class _LinearGaussianSystem:
    """Builds state-space (F, R) and time-varying design Z_t and offset u_t.

    State order mirrors the dynamic layout used throughout:
        [alpha] [beta] [g1, ..., g_{p-1}]
    Only included if corresponding mode is dynamic.
    """

    def __init__(self,
                 T: int,
                 period: int,
                 level_mode: str,
                 trend_mode: str,
                 seasonal_mode: str,
                 slope_value: float,
                 season_vec: Optional[np.ndarray]):
        self.T = int(T)
        self.period = int(period)
        self.level_mode = level_mode
        self.trend_mode = trend_mode
        self.seasonal_mode = seasonal_mode
        self.slope_value = float(slope_value)
        self.season_vec = None if season_vec is None else np.asarray(season_vec, float)

        # layout
        layout: List[str] = []
        if level_mode == "dynamic":
            layout.append("alpha")
        if trend_mode == "dynamic":
            layout.append("beta")
        if seasonal_mode == "dynamic":
            layout.extend([f"gamma_{k}" for k in range(1, period)])
        self.layout = layout
        self.D = len(layout)

        self.idx_alpha = layout.index("alpha") if "alpha" in layout else None
        self.idx_beta = layout.index("beta") if "beta" in layout else None
        if seasonal_mode == "dynamic":
            self.idx_g0 = layout.index("gamma_1") if "gamma_1" in layout else None
            self.idx_gL = self.idx_g0 + (period - 2) if self.idx_g0 is not None else None
        else:
            self.idx_g0 = None
            self.idx_gL = None

        # Build constant transition F and process-noise selector G (implicitly identity per coord).
        self.F = np.eye(self.D)
        if self.idx_alpha is not None and self.idx_beta is not None:
            # local level + local trend:
            self.F[self.idx_alpha, self.idx_beta] = 1.0  # alpha_t = alpha_{t-1} + beta_{t-1} + noise
        elif self.idx_alpha is not None and trend_mode == "deterministic":
            # alpha_t = alpha_{t-1} + slope + noise -> handled via d_t on alpha coord
            pass

        if seasonal_mode == "dynamic":
            g0, gL = self.idx_g0, self.idx_gL
            # shift: g1<-g2, ..., g_{p-2}<-g_{p-1}; last becomes -sum(prev)
            if g0 is not None:
                # zero seasonal block then fill shift
                self.F[g0:gL, g0+1:gL+1] = np.eye((period - 2))
                # last row: all -1 over previous seasonal coords
                self.F[gL, g0:gL+1] = -1.0

        # time-varying deterministic drift d_t for slope into alpha when slope is deterministic
        self.d_t = np.zeros((self.T, self.D))
        if self.idx_alpha is not None and trend_mode == "deterministic":
            self.d_t[:, self.idx_alpha] = self.slope_value

        # Build time-varying observation design Z_t and offsets u_t (deterministic pieces)
        self.Z_t = np.zeros((self.T, self.D))
        self.u_t = np.zeros(self.T)
        for t in range(self.T):
            # dynamic alpha always contributes with coefficient 1
            if self.idx_alpha is not None:
                self.Z_t[t, self.idx_alpha] = 1.0
            # dynamic seasonal contributes with coeff 1 on last seasonal coord
            if self.idx_gL is not None:
                self.Z_t[t, self.idx_gL] = 1.0
            # special case: level deterministic & trend dynamic -> observation has t * beta_t
            if (level_mode == "deterministic") and (self.idx_beta is not None):
                self.Z_t[t, self.idx_beta] += float(t)
            # deterministic seasonal offset
            if seasonal_mode == "deterministic" and self.season_vec is not None:
                self.u_t[t] += float(self.season_vec[t % period])
            # deterministic level+trend offset
            if level_mode == "deterministic":
                if trend_mode == "deterministic":
                    self.u_t[t] += 0.0  # accounted below when sampling level/slope; keep 0 here
                else:
                    self.u_t[t] += 0.0  # level enters via regression step as offset too


# ---- Kalman filter + backward sampling ---- #

def _kalman_ffbs(y: np.ndarray,
                 sys: _LinearGaussianSystem,
                 Q: np.ndarray,
                 sigma2: float,
                 m0: np.ndarray,
                 P0: np.ndarray) -> np.ndarray:
    """
    Forward-filter backward-sample a full trajectory x_{1:T}.
    Returns array of shape (T, D).
    """
    y = np.asarray(y, float)
    T, D = sys.T, sys.D
    if D == 0:
        return np.zeros((T, 0), float)

    F = sys.F
    Z_t = sys.Z_t
    d_t = sys.d_t
    u_t = sys.u_t

    Qmat = np.diag(Q)  # process noise covariance (diagonal by construction)

    m_tt = np.zeros((T, D), float)  # filtered means
    P_tt = np.zeros((T, D, D), float)
    a_t  = np.zeros((T, D), float)  # one-step-ahead means
    R_t  = np.zeros((T, D, D), float)

    m_prev = m0.copy()
    P_prev = P0.copy()

    # Forward pass
    for t in range(T):
        # Predict
        a = F @ m_prev + d_t[t]
        R = F @ P_prev @ F.T + Qmat

        # Update with observation
        z = Z_t[t]
        y_t = y[t] - u_t[t]
        S = float(z @ R @ z.T + sigma2)
        K = (R @ z.T) / S  # (D,)
        m = a + K * (y_t - float(z @ a))
        P = R - np.outer(K, z) @ R

        # store
        a_t[t] = a
        R_t[t] = R
        m_tt[t] = m
        P_tt[t] = P

        m_prev, P_prev = m, P

    # Backward sampling (FFBS)
    x = np.zeros((T, D), float)
    # draw x_T ~ N(m_T, P_T)
    x[T - 1] = np.random.multivariate_normal(mean=m_tt[T - 1], cov=P_tt[T - 1])
    for t in range(T - 2, -1, -1):
        P = P_tt[t]
        R = R_t[t + 1]
        a = a_t[t + 1]
        J = P @ F.T @ inv(R)  # smoother gain
        mean = m_tt[t] + J @ (x[t + 1] - a)
        cov = P - J @ R @ J.T
        # numerical stabilization
        cov = 0.5 * (cov + cov.T)
        x[t] = np.random.multivariate_normal(mean=mean, cov=cov)

    return x


# =============================================================================
# Gibbs Sampler (Gaussian DLM)
# =============================================================================

class GaussianDLmGibbs:
    # --------------------------- Construction --------------------------- #
    def __init__(
        self,
        y: np.ndarray,
        period: int,
        level_mode: str = "dynamic",
        trend_mode: str = "dynamic",
        seasonal_mode: str = "dynamic",
        # Initial state priors for dynamic coords
        m0_level: float = 0.0, v0_level: float = 1.0,
        m0_trend: float = 0.0, v0_trend: float = 1.0,
        m0_season: Sequence[float] | None = None,
        v0_season: Sequence[float] | None = None,
        # Priors and config
        priors: Priors = Priors(),
        cfg: SamplerConfig = SamplerConfig(),
        # Deterministic initial values
        level_value_init: float = 0.0,
        slope_value_init: float = 0.0,
        seasonal_vector_init: Optional[Sequence[float]] = None,  # p-1 entries
    ):
        # Data
        self.y = np.asarray(y, float)
        self.T = int(self.y.size)
        self.period = int(period)

        # Modes
        assert level_mode in {"dynamic", "deterministic"}
        assert trend_mode in {"dynamic", "deterministic", "none"}
        assert seasonal_mode in {"dynamic", "deterministic", "none"}
        self.level_mode = level_mode
        self.trend_mode = trend_mode
        self.seasonal_mode = seasonal_mode

        # Priors & config
        self.priors = priors
        self.cfg = cfg
        if cfg.random_seed is not None:
            np.random.seed(cfg.random_seed)

        # ---- Latent state layout
        layout: List[str] = []
        if self.level_mode == "dynamic":
            layout.append("alpha")
        if self.trend_mode == "dynamic":
            layout.append("beta")
        if self.seasonal_mode == "dynamic":
            layout.extend([f"gamma_{k}" for k in range(1, self.period)])
        self._layout = layout
        self.dim = len(layout)

        if self.dim == 0 and (self.seasonal_mode != "deterministic") and (self.level_mode != "deterministic"):
            raise ValueError("At least one contribution to mu_t must exist (dynamic or deterministic).")

        # Indices
        self.idx_alpha = layout.index("alpha") if "alpha" in layout else None
        self.idx_beta = layout.index("beta") if "beta" in layout else None
        if self.seasonal_mode == "dynamic":
            self.idx_gamma_start = layout.index("gamma_1") if "gamma_1" in layout else None
            self.idx_gamma_end = self.idx_gamma_start + (self.period - 2) if self.idx_gamma_start is not None else None
        else:
            self.idx_gamma_start = None
            self.idx_gamma_end = None

        # ---- Deterministic parameters
        self.level_value = float(level_value_init)
        self.slope_value = float(slope_value_init)

        if self.seasonal_mode == "deterministic":
            if seasonal_vector_init is not None:
                g_first = np.asarray(seasonal_vector_init, float)
                if g_first.size != self.period - 1:
                    raise ValueError("seasonal_vector_init must have length = period-1.")
            else:
                if self.priors.m_season is not None:
                    g_first = np.asarray(self.priors.m_season, float)
                    if g_first.size != self.period - 1:
                        raise ValueError("priors.m_season must have length = period-1.")
                else:
                    g_first = build_seasonal(self.period)
            g_last = -np.sum(g_first)
            self.season_vec = np.concatenate([g_first, [g_last]]).astype(float)
        else:
            self.season_vec = None

        # ---- Innovation variances Q (for dynamics): initialize at prior means
        self.Q = np.zeros(self.dim, float)
        if self.idx_alpha is not None:
            self.Q[self.idx_alpha] = self.priors.b_q_alpha / (self.priors.a_q_alpha - 1.0)
        if self.idx_beta is not None:
            self.Q[self.idx_beta]  = self.priors.b_q_beta  / (self.priors.a_q_beta  - 1.0)
        if self.seasonal_mode == "dynamic":
            self.Q[self.idx_gamma_end] = self.priors.b_q_gamma / (self.priors.a_q_gamma - 1.0)

        # ---- Observation variance sigma^2
        self.sigma2 = self.priors.b_sigma / (self.priors.a_sigma - 1.0) if self.priors.a_sigma > 1.0 else 1.0

        # ---- Initial latent path x_{1:T}
        self.x = np.zeros((self.T, self.dim), float)
        if self.seasonal_mode == "dynamic":
            m0_season_arr = np.zeros(self.period - 1) if m0_season is None else np.asarray(m0_season, float)
            v0_season_arr = np.ones(self.period - 1)  if v0_season is None else np.asarray(v0_season, float)
            if m0_season_arr.size != self.period - 1 or v0_season_arr.size != self.period - 1:
                raise ValueError("m0_season and v0_season must have length p-1 in dynamic mode.")

        m0_list, v0_list = [], []
        for tag in layout:
            if tag == "alpha":
                m0_list.append(float(m0_level)); v0_list.append(float(v0_level))
            elif tag == "beta":
                m0_list.append(float(m0_trend)); v0_list.append(float(v0_trend))
            else:
                k = int(tag.split("_")[1]) - 1
                m0_list.append(float(m0_season_arr[k]))
                v0_list.append(float(v0_season_arr[k]))
        if self.dim > 0:
            m0 = np.array(m0_list, float)
            P0 = np.diag(np.array(v0_list, float))
            # one FFBS draw with tiny Q to seed a reasonable path
            sys = _LinearGaussianSystem(T=self.T, period=self.period,
                                        level_mode=self.level_mode, trend_mode=self.trend_mode,
                                        seasonal_mode=self.seasonal_mode,
                                        slope_value=self.slope_value,
                                        season_vec=self.season_vec)
            self.x = _kalman_ffbs(self.y, sys, Q=np.maximum(self.Q, 1e-8), sigma2=max(self.sigma2, 1e-3), m0=m0, P0=P0)

        # ---- Storage (filled after knowing n_kept in run())
        self.keep: Dict[str, np.ndarray] = {}

        # ---- Truth overlays (optional)
        self.true_sigma2: Optional[float] = None
        self.true_Q: Optional[np.ndarray] = None
        self.true_mu_t: Optional[np.ndarray] = None
        self.true_alpha_t: Optional[np.ndarray] = None
        self.true_beta_t: Optional[np.ndarray] = None
        self.true_gamma_t: Optional[np.ndarray] = None

    # --------------------- Truth registration (optional) --------------------- #
    def set_truth(self, sigma2: Optional[float] = None, Q: Optional[np.ndarray] = None) -> None:
        self.true_sigma2 = sigma2
        self.true_Q = None if Q is None else np.asarray(Q, float)

    def set_truth_paths(self, mu: Optional[np.ndarray] = None,
                        alpha: Optional[np.ndarray] = None,
                        beta: Optional[np.ndarray] = None,
                        gamma: Optional[np.ndarray] = None) -> None:
        self.true_mu_t = None if mu is None else np.asarray(mu, float)
        self.true_alpha_t = None if alpha is None else np.asarray(alpha, float)
        self.true_beta_t = None if beta is None else np.asarray(beta, float)
        self.true_gamma_t = None if gamma is None else np.asarray(gamma, float)

    # ----------------------------- Helpers --------------------------------- #
    def _mu_vec_from_states(self, x: np.ndarray) -> np.ndarray:
        """Compute mu_t given a full state path x_{1:T}."""
        T = self.T
        mu = np.zeros(T, float)
        for t in range(T):
            m = 0.0
            if self.idx_alpha is not None:
                m += float(x[t, self.idx_alpha])
            if self.seasonal_mode == "dynamic" and self.idx_gamma_end is not None:
                m += float(x[t, self.idx_gamma_end])
            if (self.level_mode == "deterministic") and (self.idx_beta is not None):
                m += float(t) * float(x[t, self.idx_beta])
            if self.seasonal_mode == "deterministic":
                m += float(self.season_vec[t % self.period])
            if self.level_mode == "deterministic" and self.trend_mode == "deterministic":
                m += self.level_value + self.slope_value * t
            elif self.level_mode == "deterministic" and self.trend_mode != "deterministic":
                m += self.level_value
            return_val = m
            mu[t] = return_val
        return mu

    # ----------------------- Parameter updates ----------------------------- #
    def _sample_states(self) -> None:
        if self.dim == 0:
            return
        # Build system for current deterministic pieces
        sys = _LinearGaussianSystem(T=self.T, period=self.period,
                                    level_mode=self.level_mode, trend_mode=self.trend_mode,
                                    seasonal_mode=self.seasonal_mode,
                                    slope_value=self.slope_value,
                                    season_vec=self.season_vec)
        # initial prior (diffuse-ish via last draw's endpoints)
        m0 = np.zeros(self.dim)
        P0 = np.eye(self.dim) * 10.0
        self.x = _kalman_ffbs(self.y, sys, Q=np.maximum(self.Q, 1e-12), sigma2=max(self.sigma2, 1e-8), m0=m0, P0=P0)

    def _sample_Q(self) -> None:
        pr = self.priors
        T = self.T
        if self.idx_alpha is not None:
            resid = []
            for t in range(1, T):
                drift = 0.0
                if self.idx_beta is not None:
                    drift = self.x[t - 1, self.idx_beta]
                elif self.trend_mode == "deterministic":
                    drift = self.slope_value
                mean = self.x[t - 1, self.idx_alpha] + drift
                resid.append(self.x[t, self.idx_alpha] - mean)
            rss = float(np.sum(np.square(resid)))
            a = pr.a_q_alpha + 0.5 * (T - 1)
            b = pr.b_q_alpha + 0.5 * rss
            self.Q[self.idx_alpha] = invgamma.rvs(a=a, scale=b)

        if self.idx_beta is not None:
            resid = self.x[1:, self.idx_beta] - self.x[:-1, self.idx_beta]
            rss = float(np.sum(np.square(resid)))
            a = pr.a_q_beta + 0.5 * (T - 1)
            b = pr.b_q_beta + 0.5 * rss
            self.Q[self.idx_beta] = invgamma.rvs(a=a, scale=b)

        if self.seasonal_mode == "dynamic":
            g0, gL = self.idx_gamma_start, self.idx_gamma_end
            resid = []
            for t in range(1, T):
                prev_gamma = self.x[t - 1, g0 : gL + 1]
                mean_new = -np.sum(prev_gamma)
                resid.append(self.x[t, gL] - mean_new)
            rss = float(np.sum(np.square(resid)))
            a = pr.a_q_gamma + 0.5 * (T - 1)
            b = pr.b_q_gamma + 0.5 * rss
            self.Q[gL] = invgamma.rvs(a=a, scale=b)

    def _sample_sigma2(self) -> None:
        mu = self._mu_vec_from_states(self.x)
        resid = self.y - mu
        a = self.priors.a_sigma + 0.5 * self.T
        b = self.priors.b_sigma + 0.5 * float(np.sum(resid * resid))
        self.sigma2 = invgamma.rvs(a=a, scale=b)

    def _sample_level_slope_if_needed(self) -> None:
        pr = self.priors
        T = self.T
        t_idx = np.arange(T, dtype=float)

        # Case A: level deterministic, trend deterministic -> joint Gaussian regression
        if self.level_mode == "deterministic" and self.trend_mode == "deterministic":
            # y = level + slope * t + (dynamic seasonal contribution if any) + eps
            dyn = np.zeros(T)
            if self.seasonal_mode == "dynamic" and self.idx_gamma_end is not None:
                dyn += self.x[:, self.idx_gamma_end]
            if self.idx_alpha is not None:
                dyn += self.x[:, self.idx_alpha]
            if self.seasonal_mode == "deterministic":
                dyn += self.season_vec[np.mod(np.arange(T), self.period)]
            y_til = self.y - dyn
            X = np.vstack([np.ones(T), t_idx]).T
            # prior N(m, S)
            m = np.array([pr.m_level, pr.m_slope])
            S = np.diag([pr.s2_level, pr.s2_slope])
            Sigma_inv = (X.T @ X) / self.sigma2 + inv(S)
            Sigma = inv(Sigma_inv)
            mu = Sigma @ ((X.T @ y_til) / self.sigma2 + inv(S) @ m)
            draw = np.random.multivariate_normal(mu, Sigma)
            self.level_value = float(draw[0])
            self.slope_value = float(draw[1])
            return

        # Case B: level deterministic, trend dynamic -> sample level by regression
        if self.level_mode == "deterministic" and self.idx_beta is not None:
            dyn = np.zeros(T)
            if self.idx_alpha is not None:
                dyn += self.x[:, self.idx_alpha]
            if self.seasonal_mode == "dynamic" and self.idx_gamma_end is not None:
                dyn += self.x[:, self.idx_gamma_end]
            if self.seasonal_mode == "deterministic":
                dyn += self.season_vec[np.mod(np.arange(T), self.period)]
            dyn += t_idx * self.x[:, self.idx_beta]
            y_til = self.y - dyn
            # y_til = level + eps
            s2_post = 1.0 / (T / self.sigma2 + 1.0 / pr.s2_level)
            m_post = s2_post * (np.sum(y_til) / self.sigma2 + pr.m_level / pr.s2_level)
            self.level_value = float(np.random.normal(m_post, np.sqrt(s2_post)))
            return

        # Case C: level dynamic (alpha present), trend deterministic -> sample slope from alpha transitions
        if (self.idx_alpha is not None) and (self.trend_mode == "deterministic"):
            # alpha_t - alpha_{t-1} = slope + noise_alpha
            d = self.x[1:, self.idx_alpha] - self.x[:-1, self.idx_alpha]
            Tm1 = d.size
            s2 = self.Q[self.idx_alpha]
            # regression with prior N(m_slope, s2_slope)
            s2_post = 1.0 / (Tm1 / s2 + 1.0 / pr.s2_slope)
            m_post = s2_post * (np.sum(d) / s2 + pr.m_slope / pr.s2_slope)
            self.slope_value = float(np.random.normal(m_post, np.sqrt(s2_post)))
            return

        # Other cases: nothing to update here
        return

    # --------------------------------- MCMC --------------------------------- #
    def run(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg

        # kept draws
        save_iters = list(range(cfg.burn, cfg.n_iter, cfg.thin))
        n_kept = max(0, len(save_iters))

        # storage
        keep_idx = 0
        self.keep = {
            "sigma2": np.zeros(n_kept, float),
            "mu":     np.zeros((n_kept, self.T), float),
        }
        if self.dim > 0:
            self.keep["Q"] = np.zeros((n_kept, self.dim), float)
            if self.idx_alpha is not None:
                self.keep["alpha_t"] = np.zeros((n_kept, self.T), float)
            if self.idx_beta is not None:
                self.keep["beta_t"] = np.zeros((n_kept, self.T), float)
            if self.seasonal_mode == "dynamic":
                self.keep["gamma_t"] = np.zeros((n_kept, self.T), float)
        if self.level_mode == "deterministic":
            self.keep["level_value"] = np.zeros(n_kept, float)
        if self.trend_mode == "deterministic":
            self.keep["slope_value"] = np.zeros(n_kept, float)
        if self.seasonal_mode == "deterministic":
            self.keep["season_vector"] = np.zeros((n_kept, self.period), float)

        print_every = cfg.progress_every if cfg.progress_every > 0 else max(1, cfg.n_iter // 50)

        for it in range(cfg.n_iter):
            if cfg.progress:
                print(f"Iteration {it + 1}/{cfg.n_iter}")

            # 1) latent states via FFBS
            if self.dim > 0:
                self._sample_states()

            # 2) dynamic Q
            if self.dim > 0:
                self._sample_Q()

            # 3) deterministic structural params (conjugate where applicable)
            self._sample_level_slope_if_needed()

            # 4) observation variance
            self._sample_sigma2()

            # 5) compact progress line
            if cfg.progress and ((it + 1) % print_every == 0 or it == cfg.n_iter - 1):
                mu_vec = self._mu_vec_from_states(self.x)
                info = [
                    f"sigma={np.sqrt(self.sigma2):.3f}",
                ]
                if self.idx_alpha is not None:
                    info.append(f"Q_alpha={self.Q[self.idx_alpha]:.3e}")
                if self.idx_beta is not None:
                    info.append(f"Q_beta={self.Q[self.idx_beta]:.3e}")
                if self.seasonal_mode == "dynamic" and self.idx_gamma_end is not None:
                    info.append(f"Q_gamma={self.Q[self.idx_gamma_end]:.3e}")
                if self.level_mode == "deterministic":
                    info.append(f"level={self.level_value:.3f}")
                if self.trend_mode == "deterministic":
                    info.append(f"slope={self.slope_value:.5f}")
                print("  [status] " + " | ".join(info))

            # 6) store
            if it in save_iters and keep_idx < n_kept:
                mu_vec = self._mu_vec_from_states(self.x)
                self.keep["mu"][keep_idx, :] = mu_vec
                self.keep["sigma2"][keep_idx] = float(self.sigma2)
                if self.dim > 0:
                    self.keep["Q"][keep_idx, :] = self.Q
                if self.idx_alpha is not None:
                    self.keep["alpha_t"][keep_idx, :] = self.x[:, self.idx_alpha]
                if self.idx_beta is not None:
                    self.keep["beta_t"][keep_idx, :] = self.x[:, self.idx_beta]
                if self.seasonal_mode == "dynamic":
                    self.keep["gamma_t"][keep_idx, :] = self.x[:, self.idx_gamma_end]
                if self.level_mode == "deterministic":
                    self.keep["level_value"][keep_idx] = self.level_value
                if self.trend_mode == "deterministic":
                    self.keep["slope_value"][keep_idx] = self.slope_value
                if self.seasonal_mode == "deterministic":
                    self.keep["season_vector"][keep_idx, :] = self.season_vec
                keep_idx += 1

        return self.keep

    # ------------------------------- Persistence ---------------------------- #
    def save_posterior(self, out_npz_path: str, extra_meta: Optional[dict] = None) -> None:
        _ensure_dir(os.path.dirname(out_npz_path))

        arrays = dict(self.keep)
        arrays["y"] = self.y.copy()
        arrays["x_last"] = self.x.copy() if self.dim > 0 else np.zeros((self.T, 0))

        if self.true_mu_t is not None: arrays["true_mu_t"] = np.asarray(self.true_mu_t, float)
        if self.true_alpha_t is not None: arrays["true_alpha_t"] = np.asarray(self.true_alpha_t, float)
        if self.true_beta_t is not None: arrays["true_beta_t"] = np.asarray(self.true_beta_t, float)
        if self.true_gamma_t is not None: arrays["true_gamma_t"] = np.asarray(self.true_gamma_t, float)

        np.savez_compressed(out_npz_path, **arrays)

        meta = {
            "T": int(self.T),
            "dim": int(self.dim),
            "period": int(self.period),
            "modes": {
                "level_mode": self.level_mode,
                "trend_mode": self.trend_mode,
                "seasonal_mode": self.seasonal_mode,
            },
            "layout": list(self._layout),
            "idx_alpha": self.idx_alpha,
            "idx_beta": self.idx_beta,
            "idx_gamma_start": self.idx_gamma_start,
            "idx_gamma_end": self.idx_gamma_end,
            "cfg": asdict(self.cfg),
            "priors": asdict(self.priors),
            "true_sigma2": self.true_sigma2,
            "true_Q": (None if self.true_Q is None else np.asarray(self.true_Q, float).tolist()),
        }
        if extra_meta:
            meta.update(extra_meta)

        meta_path = out_npz_path.replace(".npz", ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[save] Posterior -> {out_npz_path}")
        print(f"[save] Metadata  -> {meta_path}")


# ------------------------- CLI / Example run & plots ------------------------ #
if __name__ == "__main__":
    import sys, argparse
    import matplotlib.pyplot as plt
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from simulator.mean_time_series import Mean_Time_Series  # the Gaussian simulator provided by the user

    parser = argparse.ArgumentParser(description="Gaussian DLM Gibbs Sampler")
    # Modes
    parser.add_argument("--level-mode", choices=["dynamic", "deterministic"], default="dynamic")
    parser.add_argument("--trend-mode", choices=["dynamic", "deterministic", "none"], default="dynamic")
    parser.add_argument("--season-mode", choices=["dynamic", "deterministic", "none"], default="none")
    # Basics
    parser.add_argument("--period", type=int, default=12)
    parser.add_argument("--T", type=int, default=200)
    # Initial values (shared)
    parser.add_argument("--level-init", type=float, default=5.0)
    parser.add_argument("--slope-init", type=float, default=0.02)

    # True params for simulator
    parser.add_argument("--true-sigma", type=float, default=2.0)
    parser.add_argument("--q-alpha", type=float, default=1e-1)
    parser.add_argument("--q-beta",  type=float, default=1e-5)
    parser.add_argument("--q-gamma", type=float, default=1e-2)

    # Priors (observation/process variances)
    parser.add_argument("--prior-a-sigma", type=float, default=2.0)
    parser.add_argument("--prior-b-sigma", type=float, default=2.0)
    parser.add_argument("--prior-aq-alpha", type=float, default=1.1)
    parser.add_argument("--prior-aq-beta", type=float, default=1.1)
    parser.add_argument("--prior-aq-gamma", type=float, default=1.1)
    parser.add_argument("--prior-bq-alpha", type=float, default=1)
    parser.add_argument("--prior-bq-beta", type=float, default=1)
    parser.add_argument("--prior-bq-gamma", type=float, default=1)

    # Deterministic priors
    parser.add_argument("--prior-m-level", type=float, default=0.0)
    parser.add_argument("--prior-s2-level", type=float, default=100.0)
    parser.add_argument("--prior-m-slope", type=float, default=0.0)
    parser.add_argument("--prior-s2-slope", type=float, default=100.0)
    parser.add_argument("--prior-m-season", type=str, default=None,
                        help="Comma-separated first (p-1) means for deterministic seasonal prior (e.g. '0,0,0').")
    parser.add_argument("--prior-s2-season", type=float, default=25.0)

    # Sampler config
    parser.add_argument("--n-iter", type=int, default=10000)
    parser.add_argument("--burn", type=int, default=1000)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--progress", default=True)
    parser.add_argument("--progress-every", type=int, default=10,
                        help="print compact summary every k iterations (0=auto)")

    # Output & plotting
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--show-plots", action="store_true")

    args = parser.parse_args()
    np.random.seed(args.seed)

    sim_level_mode  = args.level_mode
    sim_trend_mode  = args.trend_mode
    sim_season_mode = args.season_mode

    # Simulate data via user's Gaussian simulator
    mts = Mean_Time_Series(
        sigma=args.true_sigma,
        level_mode=sim_level_mode,
        trend_mode=sim_trend_mode,
        seasonal_mode=sim_season_mode,
        period=args.period,
        q_level=args.q_alpha,
        q_trend=args.q_beta,
        q_season=args.q_gamma,
        m0_level=args.level_init, v0_level=0.2,
        m0_trend=(args.slope_init if sim_trend_mode != "none" else 0.0), v0_trend=0.05,
        m0_season=[0.0]*(args.period-1), v0_season=[0.5]*(args.period-1),
        start_date=datetime(1980, 1, 1),
    )

    y = []
    for _ in range(args.T):
        mts.move()
        y.append(mts.measure())
    y = np.asarray(y, float)

    truths = mts.get_truth_paths(as_numpy=True)
    mu_T    = np.asarray(truths["mu"][1:1 + args.T], float)
    alpha_T = np.asarray(truths["alpha"][1:1 + args.T], float) if sim_level_mode == "dynamic" else None
    beta_T  = np.asarray(truths["beta"][1:1 + args.T], float)  if sim_trend_mode == "dynamic" else None
    gamma_T = np.asarray(truths["gamma_last"][1:1 + args.T], float) if sim_season_mode == "dynamic" else None

    # Priors & config
    def _parse_csv(s: Optional[str]) -> Optional[List[float]]:
        if s is None: return None
        s = s.strip()
        if not s: return None
        return [float(tok) for tok in s.split(",")]

    m_season_prior = _parse_csv(args.prior_m_season)
    if m_season_prior is not None and len(m_season_prior) != args.period - 1:
        raise ValueError(f"--prior-m-season must have length {args.period - 1} (got {len(m_season_prior)}).")

    priors = Priors(
        a_sigma=float(args.prior_a_sigma), b_sigma=float(args.prior_b_sigma),
        a_q_alpha=float(args.prior_aq_alpha), b_q_alpha=float(args.prior_bq_alpha),
        a_q_beta=float(args.prior_aq_beta), b_q_beta=float(args.prior_bq_beta),
        a_q_gamma=float(args.prior_aq_gamma), b_q_gamma=float(args.prior_bq_gamma),
        m_level=float(args.prior_m_level), s2_level=float(args.prior_s2_level),
        m_slope=float(args.prior_m_slope), s2_slope=float(args.prior_s2_slope),
        m_season=m_season_prior, s2_season=float(args.prior_s2_season),
    )

    cfg = SamplerConfig(
        n_iter=args.n_iter, burn=args.burn, thin=args.thin,
        random_seed=args.seed, progress=bool(args.progress),
        progress_every=int(args.progress_every),
    )

    seasonal_init_pminus1 = (
        np.asarray(m_season_prior, float) if (sim_season_mode == "deterministic" and m_season_prior is not None)
        else (build_seasonal(args.period) if sim_season_mode == "deterministic" else None)
    )

    sampler = GaussianDLmGibbs(
        y=y, period=args.period,
        level_mode=sim_level_mode, trend_mode=sim_trend_mode, seasonal_mode=sim_season_mode,
        m0_level=args.level_init, v0_level=0.2,
        m0_trend=(args.slope_init if sim_trend_mode != "none" else 0.0), v0_trend=0.05,
        m0_season=(None if sim_season_mode != "dynamic" else [0.0]*(args.period-1)),
        v0_season=(None if sim_season_mode != "dynamic" else [0.5]*(args.period-1)),
        priors=priors, cfg=cfg,
        seasonal_vector_init=seasonal_init_pminus1,
    )

    true_Q = []
    if sim_level_mode == "dynamic": true_Q.append(args.q_alpha)
    if sim_trend_mode == "dynamic": true_Q.append(args.q_beta)
    if sim_season_mode == "dynamic": true_Q += [args.q_gamma] + [0.0] * (args.period - 2)
    sampler.set_truth(sigma2=args.true_sigma**2,
                      Q=(np.asarray(true_Q, float) if len(true_Q) else None))
    sampler.set_truth_paths(mu=mu_T, alpha=alpha_T, beta=beta_T, gamma=gamma_T)

    tag = f"{sim_level_mode}-{sim_trend_mode}-{sim_season_mode}"
    out_dir = args.out_dir or os.path.join("results", "simulations", "DLM",
                                           f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    fig_dir = os.path.join(out_dir, "figures")
    _ensure_dir(out_dir); _ensure_dir(fig_dir)

    t0 = time.time()
    posterior = sampler.run()
    elapsed = time.time() - t0
    print(f"Sampler run time: {elapsed:.2f}s")

    sampler.save_posterior(
        out_npz_path=os.path.join(out_dir, "posterior.npz"),
        extra_meta={"modes": tag, "elapsed_seconds": float(elapsed)},
    )

    # ---- Summaries ----
    print(f"Posterior mean sigma: {np.mean(np.sqrt(posterior['sigma2'])):.3f} (true {args.true_sigma})")

    if sampler.true_Q is not None:
        tq = np.asarray(sampler.true_Q, float)
        if sampler.idx_alpha is not None and tq.size > sampler.idx_alpha:
            print(f"True Q_alpha:         {tq[sampler.idx_alpha]:.6g}")
        if sampler.idx_beta is not None and tq.size > sampler.idx_beta:
            print(f"True Q_beta:          {tq[sampler.idx_beta]:.6g}")
        if sampler.seasonal_mode == "dynamic" and sampler.idx_gamma_end is not None and tq.size > sampler.idx_gamma_end:
            print(f"True Q_gamma(last):   {tq[sampler.idx_gamma_end]:.6g}")

    if "Q" in posterior and sampler.dim > 0 and posterior["Q"].size > 0:
        if sampler.idx_alpha is not None:
            print(f"Posterior mean Q_alpha: {np.mean(posterior['Q'][:, sampler.idx_alpha]):.6g}")
        if sampler.idx_beta is not None:
            print(f"Posterior mean Q_beta:  {np.mean(posterior['Q'][:, sampler.idx_beta]):.6g}")
        if sampler.seasonal_mode == "dynamic":
            print(f"Posterior mean Q_gamma: {np.mean(posterior['Q'][:, sampler.idx_gamma_end]):.6g}")

    # Quick diagnostic plot
    if not args.no_plots:
        mu_mean = posterior["mu"].mean(axis=0)
        lo = np.percentile(posterior["mu"], 5, axis=0)
        hi = np.percentile(posterior["mu"], 95, axis=0)
        plt.figure(figsize=(11, 4))
        plt.plot(y, label="y", linewidth=1.0)
        plt.plot(mu_mean, "--", label="E[mu|y]", linewidth=1.2)
        plt.fill_between(np.arange(len(mu_mean)), lo, hi, alpha=0.15, label="90% band")
        plt.title(f"Gaussian DLM Gibbs :: {tag}")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "mu_fit.png"), dpi=150)
        if args.show_plots:
            plt.show()
        plt.close()
