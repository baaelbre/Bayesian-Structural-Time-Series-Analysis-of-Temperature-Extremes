"""
Kalman filter and Gibbs sampling for state-space models. Fully conjugate updates using
inverse-gamma (IG) priors for all variances and Normal priors for deterministic parameters.
"""

# %% optimization/dlmgibbs.py
from __future__ import annotations

import os, math, json, time
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, List, Sequence

import numpy as np
from numpy.linalg import inv
from datetime import datetime

import warnings
warnings.filterwarnings(
    "ignore",
    message="Conversion of an array with ndim > 0 to a scalar is deprecated",
    category=DeprecationWarning,
)



# =============================================================================
# Utilities
# =============================================================================

def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

def _mad(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    med = np.median(v)
    return float(np.median(np.abs(v - med)))

def _robust_sd(v: np.ndarray) -> float:
    return _mad(np.asarray(v, float)) / 1.4826 if v.size else 0.0


# =============================================================================
# Priors & Config (conjugate: IG for variances; Normal for deterministic params)
# =============================================================================

@dataclass
class Priors:
    # Observation variance: sigma^2 ~ IG(a_sigma, b_sigma)
    a_sigma: float = 1.0
    b_sigma: float = 1.0

    # Process variances (only used if the component is dynamic)
    a_alpha: float = 1.0
    b_alpha: float = 1.0
    a_beta:  float = 1.0
    b_beta:  float = 1.0
    a_gamma: float = 1.0
    b_gamma: float = 1.0

    # Deterministic parameters (Normal)
    m_level: float = 0.0
    s_level: float = 10.0
    m_slope: float = 0.0
    s_slope: float = 10.0

    # Deterministic seasonal prior (first p-1 entries; last implied to enforce sum-zero)
    m_season: Optional[Sequence[float]] = None  # length p-1
    s_season: float = 5.0


@dataclass
class SamplerConfig:
    n_iter: int = 2000
    burn: int = 200
    thin: int = 1
    random_seed: Optional[int] = 7
    progress: bool = True
    progress_every: int = 10  # 0 => auto (~2% of n_iter)
    ema_decay: float = 0.9    # smoothing for progress EMA of logZ


# =============================================================================
# DLM (pure Gibbs) with structural switches and seasonal sum-zero (p-1 states)
# =============================================================================

class DLMGibbs:
    """
    Structural Gaussian DLM:

        y_t = mu_t + eps_t,   eps_t ~ N(0, sigma^2)

    Switchable components:
      level_mode    ∈ {"dynamic", "deterministic"}
      trend_mode    ∈ {"dynamic", "deterministic", "none"}
      seasonal_mode ∈ {"dynamic", "deterministic", "none"}

    Dynamic state layout (if present): [alpha] [beta] [g1 ... g_{p-1}]

    Seasonal dynamics (dynamic mode):
      g_{1..p-2,t+1} = g_{2..p-1,t}
      g_{p-1,t+1}    = -sum(g_{1..p-1,t}) + eta_t,   eta_t ~ N(0, q_gamma)

    Deterministic season uses first p-1 free entries; last is implied to enforce sum-zero.
    """

    # --------------------------- Construction --------------------------- #
    def __init__(
        self,
        y: np.ndarray,
        period: int,
        level_mode: str = "dynamic",
        trend_mode: str = "dynamic",
        seasonal_mode: str = "dynamic",
        # Initial state priors (for dynamic coords)
        m0_level: float = 0.0, v0_level: float = 1.0,
        m0_trend: float = 0.0, v0_trend: float = 1.0,
        m0_season: Optional[Sequence[float]] = None,  # length p-1
        v0_season: Optional[Sequence[float]] = None,  # length p-1
        # Initial variances (starting values; will be sampled)
        sigma2_init: float = 1.0,
        q_alpha_init: float = 1e-3,
        q_beta_init:  float = 1e-6,
        q_gamma_init: float = 1e-4,
        # Deterministic params initial values
        level_value_init: float = 0.0,
        slope_value_init: float = 0.0,
        seasonal_vector_init: Optional[Sequence[float]] = None,  # first p-1 entries
        # Priors & config
        priors: Priors = Priors(),
        cfg: SamplerConfig = SamplerConfig(),
    ):
        # Data
        self.y = np.asarray(y, float)
        self.T = int(self.y.size)
        self.period = int(period)
        assert self.period >= 2

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

        # ---- State layout
        layout: List[str] = []
        if self.level_mode == "dynamic":
            layout.append("alpha")
        if self.trend_mode == "dynamic":
            layout.append("beta")
        if self.seasonal_mode == "dynamic":
            layout.extend([f"g{k}" for k in range(1, self.period)])
        self._layout = layout
        self.dim = len(layout)

        self.idx_alpha = layout.index("alpha") if "alpha" in layout else None
        self.idx_beta  = layout.index("beta")  if "beta"  in layout else None
        if self.seasonal_mode == "dynamic":
            self.idx_g_start = layout.index("g1")
            self.idx_g_end   = self.idx_g_start + (self.period - 2)
        else:
            self.idx_g_start = None
            self.idx_g_end = None

        # ---- Deterministic components
        self.level_value = float(level_value_init)
        self.slope_value = float(slope_value_init)

        if self.seasonal_mode == "deterministic":
            if seasonal_vector_init is not None:
                g_first = np.asarray(seasonal_vector_init, float).reshape(-1)
                if g_first.size != self.period - 1:
                    raise ValueError("seasonal_vector_init must have length p-1.")
            else:
                if self.priors.m_season is not None:
                    g_first = np.asarray(self.priors.m_season, float).reshape(-1)
                    if g_first.size != self.period - 1:
                        raise ValueError("priors.m_season must have length p-1.")
                else:
                    g_first = np.zeros(self.period - 1, float)
            g_last = -np.sum(g_first)
            self.season_vec_full = np.concatenate([g_first, [g_last]]).astype(float)
        else:
            self.season_vec_full = None

        # ---- Variances (to be updated in Gibbs)
        self.sigma2  = float(max(1e-12, sigma2_init))
        self.q_alpha = float(max(1e-18, q_alpha_init))
        self.q_beta  = float(max(1e-18, q_beta_init))
        self.q_gamma = float(max(1e-18, q_gamma_init))

        # ---- Initial latent path x_{0:T}
        self.x = np.zeros((self.T + 1, self.dim), float) if self.dim > 0 else np.zeros((self.T + 1, 0), float)
        if self.dim > 0:
            # m0, v0 per coordinate following layout
            m0_list, v0_list = [], []
            if self.idx_alpha is not None:
                m0_list.append(float(m0_level)); v0_list.append(float(v0_level))
            if self.idx_beta is not None:
                m0_list.append(float(m0_trend)); v0_list.append(float(v0_trend))
            if self.seasonal_mode == "dynamic":
                m0_season = np.zeros(self.period - 1) if m0_season is None else np.asarray(m0_season, float)
                v0_season = np.ones(self.period - 1)  if v0_season is None else np.asarray(v0_season, float)
                if m0_season.size != self.period - 1 or v0_season.size != self.period - 1:
                    raise ValueError("m0_season and v0_season must be length p-1 in dynamic mode.")
                for k in range(self.period - 1):
                    m0_list.append(float(m0_season[k])); v0_list.append(float(v0_season[k]))
            self.m0 = np.array(m0_list, float)
            self.C0 = np.diag(np.maximum(1e-10, np.array(v0_list, float)))
            self.x[0] = np.random.multivariate_normal(self.m0, self.C0)
        else:
            self.m0 = np.array([], float)
            self.C0 = np.zeros((0, 0), float)

        # ---- Storage (filled after knowing n_kept in run())
        self.keep: Dict[str, np.ndarray] = {}

        # ---- Truth overlays (optional)
        self.true_sigma: Optional[float] = None
        self.true_Q: Optional[np.ndarray] = None
        self.true_mu_t: Optional[np.ndarray] = None
        self.true_alpha_t: Optional[np.ndarray] = None
        self.true_beta_t: Optional[np.ndarray] = None
        self.true_gamma_t: Optional[np.ndarray] = None

    # --------------------- Truth registration (optional) --------------------- #
    def set_truth(self, sigma: Optional[float] = None, Q: Optional[np.ndarray] = None) -> None:
        self.true_sigma = sigma
        self.true_Q = None if Q is None else np.asarray(Q, float)

    def set_truth_paths(self, mu: Optional[np.ndarray] = None,
                        alpha: Optional[np.ndarray] = None,
                        beta: Optional[np.ndarray] = None,
                        gamma: Optional[np.ndarray] = None) -> None:
        self.true_mu_t = None if mu is None else np.asarray(mu, float)
        self.true_alpha_t = None if alpha is None else np.asarray(alpha, float)
        self.true_beta_t = None if beta is None else np.asarray(beta, float)
        self.true_gamma_t = None if gamma is None else np.asarray(gamma, float)

    # ----------------------------- Helpers ---------------------------------- #
    def _H_t(self) -> np.ndarray:
        """Observation selector for dynamic state at time t (time-invariant here)."""
        if self.dim == 0:
            return np.zeros((1, 0))
        h = np.zeros(self.dim, float)
        if self.idx_alpha is not None:
            h[self.idx_alpha] = 1.0
        if self.seasonal_mode == "dynamic":
            h[self.idx_g_end] = 1.0  # last seasonal coord contributes to mu_t
        return h.reshape(1, -1)

    def _G_t(self) -> np.ndarray:
        """State transition matrix for dynamic part (time-invariant)."""
        if self.dim == 0:
            return np.zeros((0, 0))
        G = np.eye(self.dim, dtype=float)

        # alpha with drift from beta in G
        if (self.idx_alpha is not None) and (self.idx_beta is not None):
            G[self.idx_alpha, self.idx_beta] = 1.0  # alpha_{t+1} = alpha_t + beta_t + noise_alpha

        # seasonal companion (shift)
        if self.seasonal_mode == "dynamic":
            gs, ge = self.idx_g_start, self.idx_g_end
            # shift g1..g_{p-2} <- g2..g_{p-1}
            for k in range(ge - gs):
                G[gs + k, gs + k] = 0.0
                G[gs + k, gs + k + 1] = 1.0
            # last row handled via input vector u_t; keep zeros here
            G[ge, gs:ge + 1] = 0.0
        return G

    def _u_t(self, x_prev: np.ndarray) -> np.ndarray:
        """Deterministic part of transition: handles deterministic slope and seasonal closure."""
        if self.dim == 0:
            return np.zeros(0)
        u = np.zeros(self.dim, float)

        # alpha drift when trend is deterministic
        if (self.idx_alpha is not None) and (self.idx_beta is None) and (self.trend_mode == "deterministic"):
            u[self.idx_alpha] = self.slope_value

        # seasonal closure for last coord: g_{p-1,t+1} = -sum(g_{1..p-1,t}) + noise
        if self.seasonal_mode == "dynamic":
            gs, ge = self.idx_g_start, self.idx_g_end
            prev = x_prev[gs:ge + 1]
            u[ge] = -float(np.sum(prev))
        return u

    def _Q_mat(self) -> np.ndarray:
        """Process noise covariance for dynamic part."""
        if self.dim == 0:
            return np.zeros((0, 0))
        Q = np.zeros((self.dim, self.dim), float)
        if self.idx_alpha is not None and self.q_alpha > 0.0:
            Q[self.idx_alpha, self.idx_alpha] = self.q_alpha
        if self.idx_beta is not None and self.q_beta > 0.0:
            Q[self.idx_beta, self.idx_beta] = self.q_beta
        if self.seasonal_mode == "dynamic" and self.q_gamma > 0.0:
            Q[self.idx_g_end, self.idx_g_end] = self.q_gamma
        return Q

    def _deterministic_mu_t(self, t: int) -> float:
        """Contribution to mu_t from deterministic components at time index t (0-based)."""
        out = 0.0
        # deterministic level + trend
        if self.level_mode == "deterministic":
            out += self.level_value
            if self.trend_mode == "deterministic":
                out += self.slope_value * t
        # deterministic season
        if self.seasonal_mode == "deterministic":
            out += float(self.season_vec_full[t % self.period])
        return out

    def _mu_vec_current(self) -> np.ndarray:
        """Compute mu_t for t=1..T under current state/params."""
        if self.T == 0:
            return np.zeros(0)
        H = self._H_t()
        mu = np.zeros(self.T, float)
        for t in range(1, self.T + 1):
            mu_dyn = float(H @ self.x[t]) if self.dim > 0 else 0.0
            mu[t - 1] = self._deterministic_mu_t(t - 1) + mu_dyn
        return mu

    def _gaussian_loglike(self, mu_vec: np.ndarray) -> float:
        """Log marginal of y given mu and sigma^2 (up to constants irrelevant to params)."""
        e = self.y - mu_vec
        s2 = self.sigma2
        return float(-0.5 * self.T * (math.log(2.0 * math.pi * s2)) - 0.5 * np.sum(e * e) / s2)

    # ------------------------- FFBS: sample x_{0:T} ------------------------- #
    def _ffbs(self) -> np.ndarray:
        """Sample full dynamic state trajectory given params (sigma2, q's, deterministic parts)."""
        if self.dim == 0:
            return self.x.copy()

        H = self._H_t()                 # (1 x D)
        G = self._G_t()                 # (D x D)
        Q = self._Q_mat()               # (D x D)
        R = float(self.sigma2)          # scalar observation variance

        # Forward Kalman filter
        a = np.zeros((self.T + 1, self.dim))              # prior means
        Rm = np.zeros((self.T + 1, self.dim, self.dim))   # prior covs
        m  = np.zeros((self.T + 1, self.dim))             # post means
        C  = np.zeros((self.T + 1, self.dim, self.dim))   # post covs

        m[0] = self.m0
        C[0] = self.C0

        for t in range(1, self.T + 1):
            # Predict
            u = self._u_t(m[t - 1])
            a[t]  = G @ m[t - 1] + u
            Rm[t] = G @ C[t - 1] @ G.T + Q

            # Observe
            det_mu = self._deterministic_mu_t(t - 1)
            y_t = self.y[t - 1] - det_mu

            S = float(H @ Rm[t] @ H.T + R)             # scalar
            K = (Rm[t] @ H.T) / S                      # (D x 1)
            v = y_t - float(H @ a[t])                  # scalar innovation

            m[t] = a[t] + (K.flatten() * v)
            C[t] = Rm[t] - K @ (H @ Rm[t])

        # Backward sampling (Carter–Kohn; correct conditional covariance)
        x = np.zeros_like(self.x)
        x[self.T] = np.random.multivariate_normal(m[self.T], C[self.T])
        for t in range(self.T - 1, -1, -1):
            u = self._u_t(m[t])
            J = C[t] @ G.T @ np.linalg.inv(Rm[t + 1])
            mean = m[t] + J @ (x[t + 1] - (G @ m[t] + u))
            cov  = C[t] - J @ Rm[t + 1] @ J.T
            cov  = 0.5 * (cov + cov.T)                   # symmetrize
            eig  = np.linalg.eigvalsh(cov)
            if eig.min() <= 0:
                cov += (1e-12 - eig.min()) * np.eye(cov.shape[0])
            x[t] = np.random.multivariate_normal(mean, cov)

        return x

    # ------------------- Conjugate variance updates (IG) -------------------- #
    def _sample_sigma2(self) -> None:
        # residuals: y_t - (deterministic + dynamic contribution H x_t)
        ss = 0.0
        H = self._H_t()
        for t in range(1, self.T + 1):
            mu_dyn = float(H @ self.x[t])
            mu = self._deterministic_mu_t(t - 1) + mu_dyn
            e = self.y[t - 1] - mu
            ss += e * e
        a_post = self.priors.a_sigma + 0.5 * self.T
        b_post = self.priors.b_sigma + 0.5 * ss
        # IG(a,b): sample sigma2 via Gamma on precision
        self.sigma2 = float(b_post / np.random.gamma(shape=a_post, scale=1.0))

    def _sample_q_alpha(self) -> None:
        if self.idx_alpha is None:
            return
        ss = 0.0
        for t in range(1, self.T + 1):
            drift = 0.0
            if self.idx_beta is not None:
                drift = self.x[t - 1, self.idx_beta]
            elif self.trend_mode == "deterministic":
                drift = self.slope_value
            diff = self.x[t, self.idx_alpha] - (self.x[t - 1, self.idx_alpha] + drift)
            ss += diff * diff
        a_post = self.priors.a_alpha + 0.5 * self.T
        b_post = self.priors.b_alpha + 0.5 * ss
        self.q_alpha = float(b_post / np.random.gamma(shape=a_post, scale=1.0))

    def _sample_q_beta(self) -> None:
        if self.idx_beta is None:
            return
        diffs = self.x[1:, self.idx_beta] - self.x[:-1, self.idx_beta]
        ss = float(np.sum(diffs * diffs))
        a_post = self.priors.a_beta + 0.5 * self.T
        b_post = self.priors.b_beta + 0.5 * ss
        self.q_beta = float(b_post / np.random.gamma(shape=a_post, scale=1.0))

    def _sample_q_gamma(self) -> None:
        if self.seasonal_mode != "dynamic":
            return
        gs, ge = self.idx_g_start, self.idx_g_end
        ss = 0.0
        for t in range(1, self.T + 1):
            prev = self.x[t - 1, gs:ge + 1]
            mean_new = -float(np.sum(prev))
            diff = self.x[t, ge] - mean_new
            ss += diff * diff
        a_post = self.priors.a_gamma + 0.5 * self.T
        b_post = self.priors.b_gamma + 0.5 * ss
        self.q_gamma = float(b_post / np.random.gamma(shape=a_post, scale=1.0))

    # -------- Deterministic params (Gaussian conjugate regressions) --------- #
    def _sample_level_trend_deterministic(self) -> None:
        """
        If level is deterministic (and possibly slope deterministic), sample
        [level, slope]^T jointly via conjugate Normal regression:
           y = X theta + residual,  residual ~ N(0, sigma2)
        after removing dynamic and deterministic seasonal parts.
        """
        det_level = (self.level_mode == "deterministic")
        det_trend = (self.trend_mode == "deterministic")

        if not det_level and not det_trend:
            return

        # Build response after removing dynamic part and deterministic seasonal part
        H = self._H_t()
        y_star = np.zeros(self.T, float)
        for t in range(1, self.T + 1):
            mu_dyn = float(H @ self.x[t])
            det_seas = self.season_vec_full[(t - 1) % self.period] if self.seasonal_mode == "deterministic" else 0.0
            y_star[t - 1] = self.y[t - 1] - (mu_dyn + det_seas)

        # Design for [level, slope]: include column(s) only if parameter present
        cols = []
        prior_mean = []
        prior_prec = []

        if det_level:
            cols.append(np.ones(self.T, float))
            prior_mean.append(self.priors.m_level)
            prior_prec.append(1.0 / (self.priors.s_level ** 2))
        if det_trend:
            tvec = np.arange(self.T, dtype=float)
            cols.append(tvec)
            prior_mean.append(self.priors.m_slope)
            prior_prec.append(1.0 / (self.priors.s_slope ** 2))

        X = np.vstack(cols).T  # (T x k)
        m0 = np.array(prior_mean, float)
        P0 = np.diag(np.array(prior_prec, float))     # precision matrix (k x k)

        # Posterior
        XtX = (X.T @ X) / self.sigma2
        XtY = (X.T @ y_star) / self.sigma2
        Pn = P0 + XtX
        Vn = inv(Pn)
        mn = Vn @ (P0 @ m0 + XtY)

        theta = np.random.multivariate_normal(mn, Vn)
        idx = 0
        if det_level:
            self.level_value = float(theta[idx]); idx += 1
        if det_trend:
            self.slope_value = float(theta[idx])

    def _sample_deterministic_season(self) -> None:
        """
        For deterministic season, sample the first p-1 entries (last implied).
        Design Z is 1-of-(p-1) seasonal dummy; observation is after removing
        dynamic part and deterministic (level+trend) part.
        Prior: theta ~ N(m_season, s_season^2 I)
        """
        if self.seasonal_mode != "deterministic":
            return

        p = self.period
        k = p - 1
        m0 = self.priors.m_season
        if m0 is None:
            m0 = np.zeros(k, float)
        else:
            m0 = np.asarray(m0, float).reshape(-1)
            if m0.size != k:
                raise ValueError("priors.m_season must have length p-1.")

        s2 = float(self.priors.s_season ** 2)
        P0 = (1.0 / s2) * np.eye(k)

        # response after removing dynamic + deterministic (level+trend)
        H = self._H_t()
        y_star = np.zeros(self.T, float)
        for t in range(1, self.T + 1):
            mu_dyn = float(H @ self.x[t])
            det_lt = 0.0
            if self.level_mode == "deterministic":
                det_lt += self.level_value
                if self.trend_mode == "deterministic":
                    det_lt += self.slope_value * (t - 1)
            y_star[t - 1] = self.y[t - 1] - (mu_dyn + det_lt)

        # Seasonal design: one-hot on first p-1; last is implied -> no column for it
        Z = np.zeros((self.T, k), float)
        for t in range(self.T):
            idx = t % p
            if idx < k:
                Z[t, idx] = 1.0
        # Implement implied last by augmenting rows where idx==k with -1 across all columns
        for t in range(self.T):
            if (t % p) == k:
                Z[t, :] += -1.0

        # Posterior
        XtX = (Z.T @ Z) / self.sigma2
        XtY = (Z.T @ y_star) / self.sigma2
        Pn = P0 + XtX
        Vn = inv(Pn)
        mn = Vn @ (P0 @ m0 + XtY)

        theta = np.random.multivariate_normal(mn, Vn)  # length k
        last = -float(np.sum(theta))
        self.season_vec_full = np.concatenate([theta, [last]])

    # -------------------------- Progress line format ------------------------ #
    def _progress_line(self, it: int, logZ: float, ema_logZ: float) -> str:
        left = f"[it {it+1}/{self.cfg.n_iter}] logZ={logZ:.3f} ema={ema_logZ:.3f}"
        mid  = f" | σ={math.sqrt(self.sigma2):.3f}"

        # Group 1: Q_α, Q_β, Q_γ ("/" where absent)
        q_a = f"Q_α = {self.q_alpha:.3e}" if self.idx_alpha is not None else "/"
        q_b = f"Q_β = {self.q_beta:.3e}"  if self.idx_beta  is not None else "/"
        q_g = f"Q_γ = {self.q_gamma:.3e}" if self.seasonal_mode == "dynamic" else "/"
        grp1 = f" | {q_a}, {q_b}, {q_g}"

        # Group 2: α, β, γ (deterministic params) with "/" placeholders
        a_det = f"α = {self.level_value:.3f}" if self.level_mode == "deterministic" else "/"
        b_det = f"β = {self.slope_value:.5f}" if self.trend_mode == "deterministic" else "/"
        if self.seasonal_mode == "deterministic":
            prev = np.array2string(self.season_vec_full[:min(3, self.period)], precision=3, separator=",")
            g_det = f"γ = {prev} ..."
        else:
            g_det = "/"
        grp2 = f" | {a_det}, {b_det}, {g_det}"

        return left + mid + grp1 + grp2

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
            self.keep["x"] = np.zeros((n_kept, self.T, self.dim), float)
            if self.idx_alpha is not None:
                self.keep["q_alpha"] = np.zeros(n_kept, float)
            if self.idx_beta is not None:
                self.keep["q_beta"]  = np.zeros(n_kept, float)
            if self.seasonal_mode == "dynamic":
                self.keep["q_gamma"] = np.zeros(n_kept, float)
        if self.level_mode == "deterministic":
            self.keep["level_value"] = np.zeros(n_kept, float)
        if self.trend_mode == "deterministic":
            self.keep["slope_value"] = np.zeros(n_kept, float)
        if self.seasonal_mode == "deterministic":
            self.keep["season_vector"] = np.zeros((n_kept, self.period), float)

        print_every = cfg.progress_every if cfg.progress_every > 0 else max(1, cfg.n_iter // 50)
        ema = None

        for it in range(cfg.n_iter):
            # 1) Sample dynamic states via FFBS
            if self.dim > 0:
                self.x = self._ffbs()

            # 2) Sample deterministic parameters (Gaussian conjugate)
            self._sample_level_trend_deterministic()
            self._sample_deterministic_season()

            # 3) Sample variances (IG conjugate)
            self._sample_sigma2()
            if self.dim > 0:
                self._sample_q_alpha()
                self._sample_q_beta()
                self._sample_q_gamma()

            # --- progress line (DGEV style) ---
            if cfg.progress and ((it + 1) % print_every == 0 or it == cfg.n_iter - 1):
                mu_now = self._mu_vec_current()
                logZ = self._gaussian_loglike(mu_now)
                ema = (cfg.ema_decay * ema + (1.0 - cfg.ema_decay) * logZ) if (ema is not None) else logZ
                print(self._progress_line(it, logZ, ema))

            # 4) Save
            if it in save_iters and keep_idx < n_kept:
                mu = self._mu_vec_current()
                self.keep["mu"][keep_idx, :] = mu
                self.keep["sigma2"][keep_idx] = self.sigma2
                if self.dim > 0:
                    self.keep["x"][keep_idx, :, :] = self.x[1:self.T + 1, :]
                    if "q_alpha" in self.keep: self.keep["q_alpha"][keep_idx] = self.q_alpha
                    if "q_beta"  in self.keep: self.keep["q_beta"][keep_idx]  = self.q_beta
                    if "q_gamma" in self.keep: self.keep["q_gamma"][keep_idx] = self.q_gamma
                if "level_value" in self.keep:
                    self.keep["level_value"][keep_idx] = self.level_value
                if "slope_value" in self.keep:
                    self.keep["slope_value"][keep_idx] = self.slope_value
                if "season_vector" in self.keep:
                    self.keep["season_vector"][keep_idx, :] = self.season_vec_full
                keep_idx += 1

        return self.keep

    # ------------------------------- Persistence ---------------------------- #
    def save_posterior(self, out_npz_path: str, extra_meta: Optional[dict] = None) -> None:
        _ensure_dir(os.path.dirname(out_npz_path))

        arrays = dict(self.keep)
        arrays["y"] = self.y.copy()
        arrays["x_last"] = self.x[1 : self.T + 1].copy() if self.dim > 0 else np.zeros((self.T, 0))

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
            "idx_g_start": self.idx_g_start,
            "idx_g_end": self.idx_g_end,
            "cfg": asdict(self.cfg),
            "priors": asdict(self.priors),
            "true_sigma": self.true_sigma,
            "true_Q": (None if self.true_Q is None else np.asarray(self.true_Q, float).tolist()),
        }
        if extra_meta:
            meta.update(extra_meta)

        meta_path = out_npz_path.replace(".npz", ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[save] Posterior -> {out_npz_path}")
        print(f"[save] Metadata  -> {meta_path}")


# ------------------------- CLI / Example run ------------------------------- #
if __name__ == "__main__":
    import sys, argparse
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    # Reuse your simulator for ground truth
    from simulator.mean_time_series import Mean_Time_Series  # adapt path if needed

    parser = argparse.ArgumentParser(description="Gaussian DLM Gibbs (FFBS) with conjugate IG priors")
    # Modes
    parser.add_argument("--level-mode", choices=["dynamic", "deterministic"], default="dynamic")
    parser.add_argument("--trend-mode", choices=["dynamic", "deterministic", "none"], default="deterministic")
    parser.add_argument("--season-mode", choices=["dynamic", "deterministic", "none"], default="none")
    # Basics
    parser.add_argument("--period", type=int, default=12)
    parser.add_argument("--T", type=int, default=500)
    # Initial values / truth for simulator
    parser.add_argument("--level-init", type=float, default=5.0)
    parser.add_argument("--slope-init", type=float, default=0.02)

    parser.add_argument("--true-sigma", type=float, default=2.0)
    parser.add_argument("--q-alpha", type=float, default=0.05)
    parser.add_argument("--q-beta",  type=float, default=0.02)
    parser.add_argument("--q-gamma", type=float, default=0.15)

    # Priors (conjugate IG for variances; Normal for deterministic params)
    parser.add_argument("--a-sigma", type=float, default=1.0)
    parser.add_argument("--b-sigma", type=float, default=1.0)
    parser.add_argument("--a-alpha", type=float, default=1.0)
    parser.add_argument("--b-alpha", type=float, default=1.0)
    parser.add_argument("--a-beta",  type=float, default=1.0)
    parser.add_argument("--b-beta",  type=float, default=1.0)
    parser.add_argument("--a-gamma", type=float, default=1.0)
    parser.add_argument("--b-gamma", type=float, default=1.0)

    parser.add_argument("--m-level", type=float, default=0.0)
    parser.add_argument("--s-level", type=float, default=10.0)
    parser.add_argument("--m-slope", type=float, default=0.0)
    parser.add_argument("--s-slope", type=float, default=10.0)
    parser.add_argument("--m-season", type=str, default=None,
                        help="Comma-separated first (p-1) means for deterministic seasonal prior.")
    parser.add_argument("--s-season", type=float, default=5.0)

    # Sampler config
    parser.add_argument("--n-iter", type=int, default=2000)
    parser.add_argument("--burn", type=int, default=200)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--progress", default=True)
    parser.add_argument("--progress-every", type=int, default=10)

    # Output
    parser.add_argument("--out-dir", type=str, default=None)

    args = parser.parse_args()
    np.random.seed(args.seed)

    # Build seasonal priors for simulator (length p-1)
    p = int(args.period)
    m0_season_dyn = [0.0] * (p - 1)
    v0_season_dyn = [0.5] * (p - 1)
    m0_season_det = [0.0] * (p - 1)  # can be overwritten via --m-season
    v0_season_det = [0.0] * (p - 1)
    m0_season_none = [0.0] * (p - 1)
    v0_season_none = [1.0] * (p - 1)

    sim_level_mode  = args.level_mode
    sim_trend_mode  = args.trend_mode
    sim_season_mode = args.season_mode

    if sim_season_mode == "deterministic":
        m0_season = m0_season_det
        v0_season = v0_season_det
        q_season  = args.q_gamma  # ignored
    elif sim_season_mode == "dynamic":
        m0_season = m0_season_dyn
        v0_season = v0_season_dyn
        q_season  = args.q_gamma
    else:
        m0_season = m0_season_none
        v0_season = v0_season_none
        q_season  = args.q_gamma  # ignored

    # Simulate data
    mts = Mean_Time_Series(
        sigma=args.true_sigma,
        level_mode=sim_level_mode,
        trend_mode=sim_trend_mode,
        seasonal_mode=sim_season_mode,
        period=p,
        q_level=args.q_alpha,
        q_trend=args.q_beta,
        q_season=q_season,
        m0_level=args.level_init, v0_level=0.25,
        m0_trend=(args.slope_init if sim_trend_mode != "none" else 0.0), v0_trend=0.05,
        m0_season=m0_season, v0_season=v0_season,
        start_date=datetime(2000, 1, 1),
    )

    y = []
    for _ in range(args.T):
        mts.move()
        y.append(mts.measure())
    y = np.asarray(y, float)

    truths = mts.get_truth_paths(as_numpy=True)
    mu_T    = truths["mu"][1:1 + args.T]
    alpha_T = truths["alpha"][1:1 + args.T] if sim_level_mode == "dynamic" else None
    beta_T  = truths["beta"][1:1 + args.T]  if sim_trend_mode == "dynamic" else None
    gamma_T = truths["gamma_last"][1:1 + args.T] if sim_season_mode == "dynamic" else None

    # Priors
    m_season_prior = None
    if args.m_season is not None:
        toks = [t.strip() for t in args.m_season.split(",") if t.strip() != ""]
        m_season_prior = np.array([float(z) for z in toks], float)
        if m_season_prior.size != p - 1:
            raise ValueError(f"--m-season must have length {p-1} (got {m_season_prior.size}).")

    priors = Priors(
        a_sigma=float(args.a_sigma), b_sigma=float(args.b_sigma),
        a_alpha=float(args.a_alpha), b_alpha=float(args.b_alpha),
        a_beta=float(args.a_beta),   b_beta=float(args.b_beta),
        a_gamma=float(args.a_gamma), b_gamma=float(args.b_gamma),
        m_level=float(args.m_level), s_level=float(args.s_level),
        m_slope=float(args.m_slope), s_slope=float(args.s_slope),
        m_season=m_season_prior, s_season=float(args.s_season),
    )

    cfg = SamplerConfig(
        n_iter=args.n_iter, burn=args.burn, thin=args.thin,
        random_seed=args.seed, progress=bool(args.progress),
        progress_every=int(args.progress_every),
    )

    # Initial deterministic seasonal vector if needed
    seasonal_init_pminus1 = (m_season_prior if (sim_season_mode == "deterministic" and m_season_prior is not None)
                             else (np.zeros(p - 1, float) if sim_season_mode == "deterministic" else None))

    sampler = DLMGibbs(
        y=y, period=p,
        level_mode=sim_level_mode, trend_mode=sim_trend_mode, seasonal_mode=sim_season_mode,
        m0_level=args.level_init, v0_level=0.25,
        m0_trend=(args.slope_init if sim_trend_mode != "none" else 0.0), v0_trend=0.05,
        m0_season=(m0_season if sim_season_mode == "dynamic" else None),
        v0_season=(v0_season if sim_season_mode == "dynamic" else None),
        sigma2_init=args.true_sigma**2,
        q_alpha_init=args.q_alpha, q_beta_init=args.q_beta, q_gamma_init=args.q_gamma,
        level_value_init=args.level_init,
        slope_value_init=(args.slope_init if sim_trend_mode == "deterministic" else 0.0),
        seasonal_vector_init=seasonal_init_pminus1,
        priors=priors, cfg=cfg,
    )

    true_Q = []
    if sim_level_mode == "dynamic": true_Q.append(args.q_alpha)
    if sim_trend_mode == "dynamic": true_Q.append(args.q_beta)
    if sim_season_mode == "dynamic": true_Q += [args.q_gamma] + [0.0] * (p - 2)
    sampler.set_truth(sigma=args.true_sigma, Q=(np.asarray(true_Q, float) if len(true_Q) else None))
    sampler.set_truth_paths(mu=mu_T, alpha=alpha_T, beta=beta_T, gamma=gamma_T)

    tag = f"{sim_level_mode}-{sim_trend_mode}-{sim_season_mode}"
    out_dir = args.out_dir or os.path.join("results", "simulations", "DLM",
                                           f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    _ensure_dir(out_dir)

    t0 = time.time()
    posterior = sampler.run()
    elapsed = time.time() - t0
    print(f"Sampler run time: {elapsed:.2f}s")

    sampler.save_posterior(
        out_npz_path=os.path.join(out_dir, "posterior.npz"),
        extra_meta={"modes": tag, "elapsed_seconds": float(elapsed)},
    )

    # ---- Summaries (robust; only print if arrays exist and are non-empty) ----
    print(f"Posterior mean sigma: {np.sqrt(np.mean(posterior['sigma2'])):.3f} (true {args.true_sigma})")
    if "q_alpha" in posterior and posterior["q_alpha"].size:
        print(f"Posterior mean q_alpha: {np.mean(posterior['q_alpha']):.6g}")
    if "q_beta" in posterior and posterior["q_beta"].size:
        print(f"Posterior mean q_beta:  {np.mean(posterior['q_beta']):.6g}")
    if "q_gamma" in posterior and posterior["q_gamma"].size:
        print(f"Posterior mean q_gamma: {np.mean(posterior['q_gamma']):.6g}")
