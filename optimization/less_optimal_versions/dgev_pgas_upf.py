# %% optimization/dgev_pgas.py
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm


# =============================================================================
# Small utilities
# =============================================================================

def _ensure_dir(path: str) -> None:
    """Create directory if non-empty path is provided."""
    if path:
        os.makedirs(path, exist_ok=True)


def _np_array(x: Iterable[float] | np.ndarray, dtype=float) -> np.ndarray:
    return np.asarray(x, dtype=dtype)


def _safe_logsumexp(v: np.ndarray) -> float:
    vmax = np.max(v)
    return float(vmax + math.log(np.mean(np.exp(v - vmax)) + 1e-300))


def build_seasonal(period: int) -> np.ndarray:
    """
    Default smooth seasonal template for first (p-1) entries. The last one is
    implied so the whole vector sums to zero.
    """
    g = np.cos(2.0 * np.pi * np.arange(period) / period)
    g -= np.mean(g)
    return g[: period - 1].astype(float)


def parse_csv_floats(s: Optional[str]) -> Optional[List[float]]:
    """Parse '0,0.2,-0.1' -> [0.0, 0.2, -0.1]. Returns None on empty."""
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return [float(tok) for tok in s.split(",")]


# =============================================================================
# GEV likelihood helpers
# =============================================================================

def _gev_logpdf_scalar(y: float, mu: float, sigma: float, xi: float) -> float:
    """log f(y|mu, sigma>0, xi) for the Generalized Extreme Value distribution."""
    if not np.isfinite(mu) or sigma <= 0.0:
        return -np.inf
    z = (y - mu) / sigma
    u = 1.0 + xi * z
    if u <= 0.0:
        return -np.inf
    if abs(xi) < 1e-8:  # Gumbel limit
        return -math.log(sigma) - math.exp(-z) - z
    inv_xi = 1.0 / xi
    return -math.log(sigma) - (1.0 + inv_xi) * math.log(u) - u ** (-inv_xi)


def gev_loglike_sum(y: np.ndarray, mu_vec: np.ndarray, sigma: float, xi: float) -> float:
    """Sum_t log f(y_t | mu_t, sigma, xi). Vectorized and safe."""
    if sigma <= 0.0 or np.any(np.isnan(mu_vec)):
        return -np.inf
    z = (y - mu_vec) / sigma
    u = 1.0 + xi * z
    if np.any(u <= 0.0):
        return -np.inf
    if abs(xi) < 1e-8:
        return float(np.sum(-np.log(sigma) - np.exp(-z) - z))
    inv_xi = 1.0 / xi
    return float(np.sum(-np.log(sigma) - (1.0 + inv_xi) * np.log(u) - u ** (-inv_xi)))


def gev_d2loglik_dmu2(y: float, mu: float, sigma: float, xi: float) -> float:
    """
    ∂^2/∂μ^2 log f(y | μ, σ, ξ). Used to approximate a measurement variance
    for the UKF (observed Fisher ≈ -curvature).
    """
    if sigma <= 0.0 or not np.isfinite(mu):
        return float("nan")
    z = (y - mu) / sigma
    u = 1.0 + xi * z
    if u <= 0.0:
        return float("nan")
    s2 = sigma * sigma
    if abs(xi) < 1e-8:  # Gumbel
        return -math.exp(-z) / s2
    term1 = xi * (xi + 1.0) / (u * u)
    term2 = (xi + 1.0) * u ** (-1.0 / xi - 2.0)
    return (term1 - term2) / s2


# =============================================================================
# Linear algebra + UKF helpers
# =============================================================================

def _chol_psd(M: np.ndarray, jitter: float = 1e-12) -> np.ndarray:
    """
    Cholesky for PSD matrices with exponential backoff jitter.
    Raises if it ultimately fails.
    """
    d = M.shape[0]
    eps = jitter
    for _ in range(8):
        try:
            return np.linalg.cholesky(M + eps * np.eye(d))
        except np.linalg.LinAlgError:
            eps *= 10.0
    # final attempt – let this raise if it fails
    return np.linalg.cholesky(M + eps * np.eye(d))


def _mvnorm_logpdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    L = _chol_psd(cov, 1e-12)
    diff = x - mean
    sol = np.linalg.solve(L, diff)
    quad = sol @ sol
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    d = mean.size
    return -0.5 * (d * math.log(2.0 * math.pi) + logdet + quad)


def _log_ratio_normals(x: np.ndarray,
                       m_num: np.ndarray, S_num: np.ndarray,
                       m_den: np.ndarray, S_den: np.ndarray) -> float:
    """log N(x; m_num,S_num) - log N(x; m_den,S_den)."""
    return _mvnorm_logpdf(x, m_num, S_num) - _mvnorm_logpdf(x, m_den, S_den)


def _unscented_sigma_points(
    m: np.ndarray,
    P: np.ndarray,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sigma points and weights (2d+1, d), (2d+1), (2d+1)."""
    d = m.size
    lam = alpha * alpha * (d + kappa) - d
    S = _chol_psd((d + lam) * P)
    X = np.empty((2 * d + 1, d), dtype=float)
    X[0] = m
    for i in range(d):
        col = S[:, i]
        X[i + 1] = m + col
        X[i + 1 + d] = m - col
    Wm = np.full(2 * d + 1, 1.0 / (2.0 * (d + lam)), dtype=float)
    Wc = Wm.copy()
    Wm[0] = lam / (d + lam)
    Wc[0] = lam / (d + lam) + (1.0 - alpha * alpha + beta)
    return X, Wm, Wc


# =============================================================================
# Priors & Configs
# =============================================================================

@dataclass
class Priors:
    # NOTE: m_sigma is the prior mean for log(sigma)
    m_sigma: float = 0.0
    s_sigma: float = 10.0
    m_xi: float = 0.0
    s_xi: float = 1.0

    # Process noise IG(a, b) (shape a, scale b)
    a_q_alpha: float = 1.1
    b_q_alpha: float = 1e-4
    a_q_beta: float = 1.1
    b_q_beta: float = 1e-12
    a_q_gamma: float = 1.5
    b_q_gamma: float = 5e-6

    # Deterministic components N(m, s^2)
    m_level: float = 0.0
    s_level: float = 10.0
    m_slope: float = 0.0
    s_slope: float = 10.0

    # Deterministic seasonal prior over first p-1 entries
    m_season: Optional[Sequence[float]] = None
    s_season: float = 5.0


@dataclass
class SamplerConfig:
    # MCMC
    n_iter: int = 5000
    burn: int = 1000
    thin: int = 5

    # RW–MH step sizes
    step_logsigma: float = 0.05
    step_xi: float = 0.05
    step_level: float = 0.05
    step_slope: float = 0.02
    step_season: float = 0.05

    # PF
    n_particles: int = 200
    trans_eps: float = 1e-8
    random_seed: Optional[int] = 123
    progress: bool = True
    progress_every: int = 0  # 0 => auto (~2% of n_iter)

    # Adaptive RW–MH (Robbins–Monro, windowed)
    adapt_steps: bool = False
    adapt_every: int = 25
    adapt_until: str = "burn"  # "burn" or "all"
    adapt_target_1d: float = 0.44
    adapt_eta0: float = 0.05
    adapt_eta_decay: float = 0.75
    step_min: float = 1e-5
    step_max: float = 1.0

    # UKF scaling
    ukf_alpha: float = 1e-3
    ukf_beta: float = 2.0
    ukf_kappa: float = 0.0

    # Curvature-based R caps
    R_min: float = 1e-6
    R_max: float = 1e6


# =============================================================================
# DGEV Particle Gibbs with Ancestor Sampling (UKF proposal)
# =============================================================================

class DGEVParticleGibbs:
    """
    Particle Gibbs with Ancestor Sampling (PGAS) for a DGEV observation model in
    which the GEV location μ_t is driven by a structural state:
        - local level (alpha) – dynamic or deterministic constant
        - local trend (beta) – dynamic, deterministic slope, or none
        - seasonal component – dynamic (sum-to-zero) or deterministic vector or none

    State transitions are linear in state with Gaussian innovations (diag Q).
    The UKF provides a one-step Gaussian proposal q(x_t | x_{t-1}, y_t).
    """

    # --------------------------- Construction --------------------------- #
    def __init__(
        self,
        y: np.ndarray,
        period: int,
        level_mode: str = "dynamic",
        trend_mode: str = "dynamic",
        seasonal_mode: str = "dynamic",
        # initial priors for dynamic coords
        m0_level: float = 0.0,
        v0_level: float = 1.0,
        m0_trend: float = 0.0,
        v0_trend: float = 1.0,
        m0_season: Optional[Sequence[float]] = None,
        v0_season: Optional[Sequence[float]] = None,
        # priors + config
        priors: Priors = Priors(),
        cfg: SamplerConfig = SamplerConfig(),
        # deterministic initial values
        level_value_init: float = 0.0,
        slope_value_init: float = 0.0,
        seasonal_vector_init: Optional[Sequence[float]] = None,  # p-1 entries
    ):
        # data
        self.y = _np_array(y, float)
        self.T = int(self.y.size)
        self.period = int(period)

        # modes
        assert level_mode in {"dynamic", "deterministic"}
        assert trend_mode in {"dynamic", "deterministic", "none"}
        assert seasonal_mode in {"dynamic", "deterministic", "none"}
        self.level_mode = level_mode
        self.trend_mode = trend_mode
        self.seasonal_mode = seasonal_mode

        # priors/config
        self.priors = priors
        self.cfg = cfg
        if cfg.random_seed is not None:
            np.random.seed(cfg.random_seed)

        # state layout
        layout: List[str] = []
        if self.level_mode == "dynamic":
            layout.append("alpha")
        if self.trend_mode == "dynamic":
            layout.append("beta")
        if self.seasonal_mode == "dynamic":
            layout.extend([f"gamma_{k}" for k in range(1, self.period)])
        self._layout = layout
        self.dim = len(layout)

        if self.dim == 0 and self.level_mode != "deterministic" and self.seasonal_mode != "deterministic":
            raise ValueError("Model must include at least one contribution to μ_t.")

        # indices
        self.idx_alpha = layout.index("alpha") if "alpha" in layout else None
        self.idx_beta = layout.index("beta") if "beta" in layout else None
        if self.seasonal_mode == "dynamic":
            self.idx_gamma_start = layout.index("gamma_1")
            self.idx_gamma_end = self.idx_gamma_start + (self.period - 2)
        else:
            self.idx_gamma_start = None
            self.idx_gamma_end = None

        # deterministic params
        self.level_value = float(level_value_init)
        self.slope_value = float(slope_value_init)
        if self.seasonal_mode == "deterministic":
            if seasonal_vector_init is not None:
                g_first = _np_array(seasonal_vector_init, float)
                if g_first.size != self.period - 1:
                    raise ValueError("seasonal_vector_init must have length p-1.")
            else:
                if self.priors.m_season is not None:
                    g_first = _np_array(self.priors.m_season, float)
                    if g_first.size != self.period - 1:
                        raise ValueError("priors.m_season must have length p-1.")
                else:
                    g_first = build_seasonal(self.period)
            self.season_vec = np.concatenate([g_first, [-np.sum(g_first)]])
        else:
            self.season_vec = None

        # observation params
        self.logsigma = float(self.priors.m_sigma)  # we work on log-scale
        self.sigma = float(np.exp(self.logsigma))
        self.xi = float(self.priors.m_xi)

        # state noise variances Q (diag)
        self.Q = np.zeros(self.dim, float)
        if self.idx_alpha is not None:
            self.Q[self.idx_alpha] = self.priors.b_q_alpha / (self.priors.a_q_alpha - 1.0)
        if self.idx_beta is not None:
            self.Q[self.idx_beta] = self.priors.b_q_beta / (self.priors.a_q_beta - 1.0)
        if self.seasonal_mode == "dynamic":
            self.Q[self.idx_gamma_end] = self.priors.b_q_gamma / (self.priors.a_q_gamma - 1.0)

        # initial latent path x_{0:T}
        self.x = np.zeros((self.T + 1, self.dim), float)
        if self.seasonal_mode == "dynamic":
            m0_season_arr = np.zeros(self.period - 1) if m0_season is None else _np_array(m0_season, float)
            v0_season_arr = np.ones(self.period - 1) if v0_season is None else _np_array(v0_season, float)
            if m0_season_arr.size != self.period - 1 or v0_season_arr.size != self.period - 1:
                raise ValueError("m0_season and v0_season must each have length p-1 in dynamic mode.")
        m0_list: List[float] = []
        v0_list: List[float] = []
        for tag in layout:
            if tag == "alpha":
                m0_list += [float(m0_level)]
                v0_list += [float(v0_level)]
            elif tag == "beta":
                m0_list += [float(m0_trend)]
                v0_list += [float(v0_trend)]
            else:
                k = int(tag.split("_")[1]) - 1
                m0_list += [float(m0_season_arr[k])]
                v0_list += [float(v0_season_arr[k])]
        if self.dim > 0:
            self.x[0] = np.random.normal(_np_array(m0_list), np.sqrt(_np_array(v0_list)))
            self._propagate_initial_path(Q_init=np.full(self.dim, 1e-6, float))

        # storage
        self.keep: Dict[str, np.ndarray] = {}

        # MH bookkeeping
        self.accept = {"logsigma": 0, "xi": 0, "level": 0, "slope": 0, "season": 0}
        self.proposals = {"logsigma": 0, "xi": 0, "level": 0, "slope": 0, "season": 0}
        self._mh_prev_acc = dict(self.accept)
        self._mh_prev_prop = dict(self.proposals)
        self._adapt_round = 0

        # truth overlays (optional)
        self.true_sigma: Optional[float] = None
        self.true_xi: Optional[float] = None
        self.true_Q: Optional[np.ndarray] = None
        self.true_mu_t: Optional[np.ndarray] = None
        self.true_alpha_t: Optional[np.ndarray] = None
        self.true_beta_t: Optional[np.ndarray] = None
        self.true_gamma_t: Optional[np.ndarray] = None

        # PF diagnostics
        self.last_log_evidence: float = float("nan")
        self.last_pf_diag: Dict[str, float] = {}

    # ----------------------- Truth registration (optional) ----------------------- #
    def set_truth(self, sigma: Optional[float] = None, xi: Optional[float] = None, Q: Optional[np.ndarray] = None) -> None:
        self.true_sigma = sigma
        self.true_xi = xi
        self.true_Q = None if Q is None else _np_array(Q, float)

    def set_truth_paths(
        self,
        mu: Optional[np.ndarray] = None,
        alpha: Optional[np.ndarray] = None,
        beta: Optional[np.ndarray] = None,
        gamma: Optional[np.ndarray] = None,
    ) -> None:
        self.true_mu_t = None if mu is None else _np_array(mu, float)
        self.true_alpha_t = None if alpha is None else _np_array(alpha, float)
        self.true_beta_t = None if beta is None else _np_array(beta, float)
        self.true_gamma_t = None if gamma is None else _np_array(gamma, float)

    # ----------------------- Helpers: norm / ESS / EMA ----------------------- #
    @staticmethod
    def _safe_normalize(p: np.ndarray) -> np.ndarray:
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0).astype(float)
        p[p < 0] = 0.0
        s = float(np.sum(p))
        if not np.isfinite(s) or s <= 0.0:
            return np.full_like(p, 1.0 / p.size)
        p /= s
        # second pass to enforce exact normalization when near machine noise
        s2 = float(np.sum(p))
        if not np.isclose(s2, 1.0, atol=1e-12):
            p /= s2
        return p

    @staticmethod
    def _ess(w: np.ndarray) -> float:
        s2 = float(np.sum(w * w))
        return (1.0 / s2) if s2 > 0.0 else 0.0

    @staticmethod
    def _ema(old: Optional[float], new: float, alpha: float = 0.1) -> float:
        return alpha * new + (1.0 - alpha) * (0.0 if old is None else old)

    def _mh_accept(self, logacc: float) -> bool:
        return (np.log(np.random.rand()) < min(0.0, logacc))

    # ----------------------------- State model ------------------------------ #
    def _state_mean(self, x_prev: np.ndarray, t: int) -> np.ndarray:
        """Deterministic structural evolution m_t^- = A x_{t-1}."""
        m = np.zeros_like(x_prev)

        # alpha (local level) with drift from beta or deterministic slope
        if self.idx_alpha is not None:
            drift = 0.0
            if self.idx_beta is not None:
                drift = x_prev[self.idx_beta]
            elif self.trend_mode == "deterministic":
                drift = self.slope_value
            m[self.idx_alpha] = x_prev[self.idx_alpha] + drift

        # beta (local trend)
        if self.idx_beta is not None:
            m[self.idx_beta] = x_prev[self.idx_beta]

        # dynamic seasonal (shift + enforce sum-zero at the last slot)
        if self.seasonal_mode == "dynamic":
            g0, gL = self.idx_gamma_start, self.idx_gamma_end
            if g0 <= gL - 1:
                m[g0:gL] = x_prev[g0 + 1 : gL + 1]
            prev_gamma = x_prev[g0 : gL + 1]
            m[gL] = -np.sum(prev_gamma)
        return m

    def _alpha_contribution(self, x_t: np.ndarray, t: int) -> float:
        if self.level_mode == "dynamic":
            return 0.0 if self.idx_alpha is None else float(x_t[self.idx_alpha])
        base = self.level_value
        if self.idx_beta is not None:  # dynamic trend present
            return float(base + x_t[self.idx_beta] * t)
        if self.trend_mode == "deterministic":
            return float(base + self.slope_value * t)
        return float(base)

    def _season_contribution(self, x_t: np.ndarray, t: int) -> float:
        if self.seasonal_mode == "dynamic":
            return 0.0 if self.idx_gamma_end is None else float(x_t[self.idx_gamma_end])
        if self.seasonal_mode == "deterministic":
            return float(self.season_vec[t % self.period])
        return 0.0

    def mu_from_state(self, x_t: np.ndarray, t: int) -> float:
        return self._alpha_contribution(x_t, t) + self._season_contribution(x_t, t)

    def _transition_logpdf(self, x_prev: np.ndarray, x_cur: np.ndarray, t: int) -> float:
        mean = self._state_mean(x_prev, t)
        eps = self.cfg.trans_eps
        out = 0.0
        for k in range(self.dim):
            var = self.Q[k] if self.Q[k] > 0.0 else eps
            diff = x_cur[k] - mean[k]
            out += -0.5 * (math.log(2.0 * math.pi * var) + (diff * diff) / var)
        return float(out)

    def _propagate_initial_path(self, Q_init: np.ndarray) -> None:
        if self.dim == 0:
            return
        for t in range(1, self.T + 1):
            mean = self._state_mean(self.x[t - 1], t)
            self.x[t] = mean + np.random.normal(0.0, np.sqrt(Q_init), size=self.dim)

    # ----------------------- Static parameter updates ----------------------- #
    def _mu_vec_current(self) -> np.ndarray:
        return np.array([self.mu_from_state(self.x[t], t - 1) for t in range(1, self.T + 1)], float)

    def update_Q(self) -> None:
        # alpha
        if self.idx_alpha is not None:
            a0, b0 = self.priors.a_q_alpha, self.priors.b_q_alpha
            resid: List[float] = []
            for t in range(1, self.T + 1):
                drift = 0.0
                if self.idx_beta is not None:
                    drift = self.x[t - 1, self.idx_beta]
                elif self.trend_mode == "deterministic":
                    drift = self.slope_value
                mean = self.x[t - 1, self.idx_alpha] + drift
                resid.append(self.x[t, self.idx_alpha] - mean)
            rss = float(np.sum(np.square(resid)))
            a = a0 + 0.5 * self.T
            b = b0 + 0.5 * rss
            self.Q[self.idx_alpha] = 1.0 / np.random.gamma(a, 1.0 / b)

        # beta
        if self.idx_beta is not None:
            a0, b0 = self.priors.a_q_beta, self.priors.b_q_beta
            resid = self.x[1:, self.idx_beta] - self.x[:-1, self.idx_beta]
            rss = float(np.sum(np.square(resid)))
            a = a0 + 0.5 * self.T
            b = b0 + 0.5 * rss
            self.Q[self.idx_beta] = 1.0 / np.random.gamma(a, 1.0 / b)

        # seasonal last coord (closure-enforcing innovation)
        if self.seasonal_mode == "dynamic":
            a0, b0 = self.priors.a_q_gamma, self.priors.b_q_gamma
            g0, gL = self.idx_gamma_start, self.idx_gamma_end
            resid: List[float] = []
            for t in range(1, self.T + 1):
                prev_gamma = self.x[t - 1, g0 : gL + 1]
                mean_new_last = -np.sum(prev_gamma)
                resid.append(self.x[t, gL] - mean_new_last)
            rss = float(np.sum(np.square(resid)))
            a = a0 + 0.5 * self.T
            b = b0 + 0.5 * rss
            self.Q[gL] = 1.0 / np.random.gamma(a, 1.0 / b)

    def update_logsigma(self) -> None:
        step = self.cfg.step_logsigma
        cur = self.logsigma
        prop = cur + np.random.normal(0.0, step)
        sigma_cur, sigma_prop = float(np.exp(cur)), float(np.exp(prop))

        mu = self._mu_vec_current()
        ll_old = gev_loglike_sum(self.y, mu, sigma_cur, self.xi)
        ll_new = gev_loglike_sum(self.y, mu, sigma_prop, self.xi)
        self.proposals["logsigma"] += 1
        if ll_new == -np.inf:
            return

        lp_old = -0.5 * ((cur - self.priors.m_sigma) ** 2) / (self.priors.s_sigma ** 2)
        lp_new = -0.5 * ((prop - self.priors.m_sigma) ** 2) / (self.priors.s_sigma ** 2)
        if self._mh_accept((ll_new + lp_new) - (ll_old + lp_old)):
            self.logsigma = prop
            self.sigma = sigma_prop
            self.accept["logsigma"] += 1

    def update_xi(self) -> None:
        step = self.cfg.step_xi
        cur = self.xi
        prop = cur + np.random.normal(0.0, step)

        mu = self._mu_vec_current()
        ll_old = gev_loglike_sum(self.y, mu, self.sigma, cur)
        ll_new = gev_loglike_sum(self.y, mu, self.sigma, prop)
        self.proposals["xi"] += 1
        if ll_new == -np.inf:
            return

        lp_old = -0.5 * ((cur - self.priors.m_xi) ** 2) / (self.priors.s_xi ** 2)
        lp_new = -0.5 * ((prop - self.priors.m_xi) ** 2) / (self.priors.s_xi ** 2)
        if self._mh_accept((ll_new + lp_new) - (ll_old + lp_old)):
            self.xi = prop
            self.accept["xi"] += 1

    def update_level_value(self) -> None:
        if self.level_mode != "deterministic":
            return
        step = self.cfg.step_level
        cur = self.level_value
        prop = cur + np.random.normal(0.0, step)

        old = self.level_value
        self.level_value = prop
        mu_prop = self._mu_vec_current()
        self.level_value = old
        mu_old = self._mu_vec_current()

        ll_old = gev_loglike_sum(self.y, mu_old, self.sigma, self.xi)
        ll_new = gev_loglike_sum(self.y, mu_prop, self.sigma, self.xi)
        self.proposals["level"] += 1
        if ll_new == -np.inf:
            return

        lp_old = -0.5 * ((cur - self.priors.m_level) ** 2) / (self.priors.s_level ** 2)
        lp_new = -0.5 * ((prop - self.priors.m_level) ** 2) / (self.priors.s_level ** 2)
        if self._mh_accept((ll_new + lp_new) - (ll_old + lp_old)):
            self.level_value = prop
            self.accept["level"] += 1

    def _alpha_transition_loglike_given_slope(self, slope: float) -> float:
        """Transition ll for alpha when trend is deterministic (0 if alpha absent)."""
        if self.idx_alpha is None:
            return 0.0
        var = self.Q[self.idx_alpha] if self.Q[self.idx_alpha] > 0.0 else self.cfg.trans_eps
        inv_var = 1.0 / var
        cst = -0.5 * math.log(2.0 * math.pi * var)
        ll = 0.0
        for t in range(1, self.T + 1):
            drift = self.x[t - 1, self.idx_beta] if self.idx_beta is not None else slope
            mean = self.x[t - 1, self.idx_alpha] + drift
            diff = self.x[t, self.idx_alpha] - mean
            ll += cst - 0.5 * diff * diff * inv_var
        return float(ll)

    def update_slope(self) -> None:
        if self.trend_mode != "deterministic":
            return
        step = self.cfg.step_slope
        cur = self.slope_value
        prop = cur + np.random.normal(0.0, step)

        # observation contribution if level is deterministic (otherwise μ uses beta directly)
        if self.level_mode == "deterministic":
            mu_old = self._mu_vec_current()
            self.slope_value = prop
            mu_new = self._mu_vec_current()
            self.slope_value = cur
            ll_obs_old = gev_loglike_sum(self.y, mu_old, self.sigma, self.xi)
            ll_obs_new = gev_loglike_sum(self.y, mu_new, self.sigma, self.xi)
            if ll_obs_new == -np.inf:
                self.proposals["slope"] += 1
                return
        else:
            ll_obs_old = 0.0
            ll_obs_new = 0.0

        # transition contribution when alpha is dynamic
        ll_trans_old = self._alpha_transition_loglike_given_slope(cur) if self.idx_alpha is not None else 0.0
        ll_trans_new = self._alpha_transition_loglike_given_slope(prop) if self.idx_alpha is not None else 0.0

        lp_old = -0.5 * ((cur - self.priors.m_slope) ** 2) / (self.priors.s_slope ** 2)
        lp_new = -0.5 * ((prop - self.priors.m_slope) ** 2) / (self.priors.s_slope ** 2)

        self.proposals["slope"] += 1
        if self._mh_accept((ll_obs_new + ll_trans_new + lp_new) - (ll_obs_old + ll_trans_old + lp_old)):
            self.slope_value = prop
            self.accept["slope"] += 1

    def update_season_vec(self) -> None:
        if self.seasonal_mode != "deterministic":
            return
        step = self.cfg.step_season
        v_cur = self.season_vec.copy()

        prop = v_cur.copy()
        prop[:-1] = v_cur[:-1] + np.random.normal(0.0, step, size=self.period - 1)
        prop[-1] = -np.sum(prop[:-1])

        old = self.season_vec
        self.season_vec = prop
        mu_prop = self._mu_vec_current()
        self.season_vec = old
        mu_old = self._mu_vec_current()

        ll_old = gev_loglike_sum(self.y, mu_old, self.sigma, self.xi)
        ll_new = gev_loglike_sum(self.y, mu_prop, self.sigma, self.xi)
        self.proposals["season"] += 1
        if ll_new == -np.inf:
            return

        m_first = self.priors.m_season
        if m_first is None:
            m_first = np.zeros(self.period - 1, float)
        else:
            m_first = _np_array(m_first, float)
            if m_first.size != self.period - 1:
                raise ValueError("priors.m_season must have length p-1.")
        s = float(self.priors.s_season)

        lp_old = -0.5 * np.sum(((v_cur[:-1] - m_first) / s) ** 2)
        lp_new = -0.5 * np.sum(((prop[:-1] - m_first) / s) ** 2)
        if self._mh_accept((ll_new + lp_new) - (ll_old + lp_old)):
            self.season_vec = prop
            self.accept["season"] += 1

    # ------------------- Adaptive step-size ------------------ #
    def _get_step(self, key: str) -> float:
        return {
            "logsigma": self.cfg.step_logsigma,
            "xi": self.cfg.step_xi,
            "level": self.cfg.step_level,
            "slope": self.cfg.step_slope,
            "season": self.cfg.step_season,
        }[key]

    def _set_step(self, key: str, val: float) -> None:
        v = float(np.clip(val, self.cfg.step_min, self.cfg.step_max))
        if key == "logsigma":
            self.cfg.step_logsigma = v
        elif key == "xi":
            self.cfg.step_xi = v
        elif key == "level":
            self.cfg.step_level = v
        elif key == "slope":
            self.cfg.step_slope = v
        elif key == "season":
            self.cfg.step_season = v
        else:
            raise KeyError(key)

    def _adapt_steps(self, it: int) -> None:
        cfg = self.cfg
        if not cfg.adapt_steps:
            return
        in_window = (cfg.adapt_until == "all") or (it < cfg.burn)
        if not in_window or (it + 1) % max(1, cfg.adapt_every) != 0:
            return
        k = self._adapt_round
        eta = cfg.adapt_eta0 / ((1.0 + k) ** cfg.adapt_eta_decay)

        keys: List[str] = ["logsigma", "xi"]
        if self.level_mode == "deterministic":
            keys.append("level")
        if self.trend_mode == "deterministic":
            keys.append("slope")
        if self.seasonal_mode == "deterministic":
            keys.append("season")

        for key in keys:
            acc_now = self.accept[key]
            prop_now = self.proposals[key]
            acc_win = acc_now - self._mh_prev_acc[key]
            prop_win = prop_now - self._mh_prev_prop[key]
            if prop_win <= 0:
                continue
            rate = acc_win / max(1, prop_win)
            s_new = self._get_step(key) * math.exp(eta * (rate - cfg.adapt_target_1d))
            self._set_step(key, s_new)
            self._mh_prev_acc[key] = acc_now
            self._mh_prev_prop[key] = prop_now
        self._adapt_round += 1

    # ---------------- UKF proposal for one step (given x_{t-1}) ------------- #
    def _ukf_proposal_params(
        self, x_prev: np.ndarray, t: int, y_t: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (m_plus, P_plus, m_pred, Qdiag) where:
          q(x_t|x_{t-1}, y_t) = N(m_plus, P_plus)  (UKF update)
          f(x_t|x_{t-1})      = N(m_pred, diag(Qdiag)) (transition prior)
        """
        d = self.dim
        m_pred = self._state_mean(x_prev, t)
        Qdiag = np.where(self.Q > 0.0, self.Q, self.cfg.trans_eps)
        P_pred = np.diag(Qdiag)

        if d == 0:
            return m_pred, P_pred, m_pred, Qdiag

        # sigma points
        X, Wm, Wc = _unscented_sigma_points(
            m_pred, P_pred, self.cfg.ukf_alpha, self.cfg.ukf_beta, self.cfg.ukf_kappa
        )
        # pass through h(x)=μ(x,t-1)
        z_pts = np.array([self.mu_from_state(xi, t - 1) for xi in X], float)
        z_mean = float(np.sum(Wm * z_pts))

        # curvature-based measurement variance R
        curv = gev_d2loglik_dmu2(y_t, z_mean, self.sigma, self.xi)  # may be negative
        if (not np.isfinite(curv)) or curv >= -1e-12:
            R = self.cfg.R_max
        else:
            R = 1.0 / max(-curv, 1.0 / self.cfg.R_max)
            R = float(np.clip(R, self.cfg.R_min, self.cfg.R_max))

        # innovation variance S and cross-covariance C
        S = float(np.sum(Wc * (z_pts - z_mean) * (z_pts - z_mean))) + R
        C = np.zeros(d, float)
        for i in range(X.shape[0]):
            dx = X[i] - m_pred
            dz = z_pts[i] - z_mean
            C += Wc[i] * dx * dz

        # Kalman-style update
        if not np.isfinite(S) or S <= 0.0:
            return m_pred, P_pred, m_pred, Qdiag
        K = C / S  # [d]
        m_plus = m_pred + K * (y_t - z_mean)
        P_plus = P_pred - np.outer(K, K) * S

        # ensure PSD
        P_plus = 0.5 * (P_plus + P_plus.T)
        try:
            _ = _chol_psd(P_plus, 1e-12)
        except Exception:
            P_plus += 1e-10 * np.eye(d)

        return m_plus, P_plus, m_pred, Qdiag

    # --------------- Conditional SMC with Ancestor Sampling (UKF-PGAS) ------ #
    def _conditional_pgas(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
        """
        Conditional SMC + Ancestor Sampling (PGAS) using the UKF proposal.
        Returns (parts, w, a, logZ, pf_diag).
        The reference path is the current self.x[1:T], enforced as particle N.
        """
        N, T, D = self.cfg.n_particles, self.T, self.dim
        parts = np.zeros((T + 1, N, D), float) if D > 0 else np.zeros((T + 1, N, 0), float)
        w = np.zeros((T + 1, N), float)
        a = np.zeros((T + 1, N), int)
        logZ = 0.0

        ess_list: List[float] = []
        maxw_list: List[float] = []

        if D > 0:
            parts[0, :, :] = self.x[0]

        # ---- t=1: propose N-1 particles; set reference particle = x[1]
        for n in range(N - 1):
            if D > 0:
                m_plus, P_plus, _, _ = self._ukf_proposal_params(parts[0, n, :], t=1, y_t=self.y[0])
                parts[1, n, :] = np.random.multivariate_normal(m_plus, P_plus)
            a[1, n] = n
        if D > 0:
            parts[1, N - 1, :] = self.x[1].copy()

        # weights at t=1
        lw = np.zeros(N, float)
        for n in range(N):
            mu = self.mu_from_state(parts[1, n, :] if D > 0 else np.zeros(0), t=0)
            if D > 0 and n < N - 1:
                m_plus, P_plus, m_pred, Qdiag = self._ukf_proposal_params(parts[0, n, :], t=1, y_t=self.y[0])
                lw[n] = _gev_logpdf_scalar(self.y[0], mu, self.sigma, self.xi) + _log_ratio_normals(
                    parts[1, n, :], m_pred, np.diag(Qdiag), m_plus, P_plus
                )
            else:
                if D > 0:
                    m_pred = self._state_mean(parts[0, N - 1, :], t=1)
                    Qdiag = np.where(self.Q > 0.0, self.Q, self.cfg.trans_eps)
                    lw[n] = _gev_logpdf_scalar(self.y[0], mu, self.sigma, self.xi) + _mvnorm_logpdf(
                        parts[1, n, :], m_pred, np.diag(Qdiag)
                    )
                else:
                    lw[n] = _gev_logpdf_scalar(self.y[0], mu, self.sigma, self.xi)

        # ancestor sampling for the reference particle at t=1
        if D > 0:
            logf = np.array([self._transition_logpdf(parts[0, j, :], parts[1, N - 1, :], t=1) for j in range(N)])
            post = self._safe_normalize(np.exp(logf - np.max(logf)))
            a[1, N - 1] = np.random.choice(N, p=post)
        else:
            a[1, N - 1] = N - 1

        # normalize weights + evidence contribution
        logZ += _safe_logsumexp(lw)
        w[1, :] = self._safe_normalize(np.exp(lw - np.max(lw)))
        ess_list.append(self._ess(w[1, :]))
        maxw_list.append(float(np.max(w[1, :])))

        if self.cfg.progress:
            print("  Running conditional PGAS (UKF proposal)...")

        rng = tqdm(range(2, T + 1)) if self.cfg.progress else range(2, T + 1)
        for t in rng:
            # resample ancestors for non-reference particles
            res_p = self._safe_normalize(w[t - 1, :])
            anc = np.random.choice(N, size=N - 1, p=res_p, replace=True)

            # propagate non-reference particles
            for n in range(N - 1):
                a[t, n] = anc[n]
                if D > 0:
                    x_prev = parts[t - 1, a[t, n], :]
                    m_plus, P_plus, _, _ = self._ukf_proposal_params(x_prev, t=t, y_t=self.y[t - 1])
                    parts[t, n, :] = np.random.multivariate_normal(m_plus, P_plus)

            # fix the reference particle and do ancestor sampling for its parent
            if D > 0:
                x_ref_t = self.x[t]
                parts[t, N - 1, :] = x_ref_t.copy()
                logw_prev = np.log(np.clip(w[t - 1, :], 1e-300, None))
                logf = np.array([self._transition_logpdf(parts[t - 1, j, :], x_ref_t, t=t) for j in range(N)], float)
                post = self._safe_normalize(np.exp((logw_prev + logf) - np.max(logw_prev + logf)))
                a[t, N - 1] = np.random.choice(N, p=post)
            else:
                a[t, N - 1] = N - 1

            # weights at t
            y_idx = t - 1
            lw = np.zeros(N, float)
            for n in range(N):
                mu = self.mu_from_state(parts[t, n, :] if D > 0 else np.zeros(0), t=y_idx)
                if D > 0 and n < N - 1:
                    x_prev = parts[t - 1, a[t, n], :]
                    m_plus, P_plus, m_pred, Qdiag = self._ukf_proposal_params(x_prev, t=t, y_t=self.y[y_idx])
                    lw[n] = _gev_logpdf_scalar(self.y[y_idx], mu, self.sigma, self.xi) + _log_ratio_normals(
                        parts[t, n, :], m_pred, np.diag(Qdiag), m_plus, P_plus
                    )
                else:
                    if D > 0:
                        x_prev = parts[t - 1, a[t, n], :]
                        m_pred = self._state_mean(x_prev, t)
                        Qdiag = np.where(self.Q > 0.0, self.Q, self.cfg.trans_eps)
                        lw[n] = _gev_logpdf_scalar(self.y[y_idx], mu, self.sigma, self.xi) + _mvnorm_logpdf(
                            parts[t, n, :], m_pred, np.diag(Qdiag)
                        )
                    else:
                        lw[n] = _gev_logpdf_scalar(self.y[y_idx], mu, self.sigma, self.xi)

            logZ += _safe_logsumexp(lw)
            w[t, :] = self._safe_normalize(np.exp(lw - np.max(lw)))

            ess_t = self._ess(w[t, :])
            maxw_t = float(np.max(w[t, :]))
            ess_list.append(ess_t)
            maxw_list.append(maxw_t)
            if self.cfg.progress and hasattr(rng, "set_postfix"):
                rng.set_postfix(ESS=f"{ess_t:6.1f}", MaxW=f"{maxw_t:7.4f}")

        pf_diag = {
            "ess_mean": float(np.mean(ess_list)),
            "ess_min": float(np.min(ess_list)),
            "maxw_mean": float(np.mean(maxw_list)),
            "maxw_max": float(np.max(maxw_list)),
        }
        return parts, w, a, float(logZ), pf_diag

    def _trace_single_trajectory(self, parts: np.ndarray, a: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Trace one trajectory by sampling k_T ~ w[T] and following ancestors."""
        if self.dim == 0:
            return self.x
        N, T = self.cfg.n_particles, self.T
        idx = np.zeros(T + 1, dtype=int)
        idx[T] = np.random.choice(N, p=w[T, :])
        for t in range(T, 1, -1):
            idx[t - 1] = a[t, idx[t]]
        x_new = self.x.copy()
        x_new[0, :] = parts[0, 0, :]  # common x0
        for t in range(1, T + 1):
            x_new[t, :] = parts[t, idx[t], :]
        return x_new

    def update_states_pgas(self) -> None:
        parts, w, a, logZ, pf_diag = self._conditional_pgas()
        self.x = self._trace_single_trajectory(parts, a, w)
        self.last_log_evidence = float(logZ)
        self.last_pf_diag = pf_diag

    # --------------------------- Progress helpers --------------------------- #
    def _fmt_acc(self, key: str) -> str:
        a, p = self.accept[key], self.proposals[key]
        pct = (100.0 * a / p) if p > 0 else 0.0
        return f"{a}/{p} ({pct:4.1f}%)"

    def _q_snapshot(self, ema_Q: Optional[np.ndarray] = None) -> str:
        rows: List[str] = []

        def add(label: str, idx: Optional[int]) -> None:
            if idx is None:
                return
            q = float(self.Q[idx])
            logq = math.log10(max(q, 1e-20))
            if ema_Q is not None:
                ema = float(ema_Q[idx])
                logema = math.log10(max(ema, 1e-20))
                rows.append(f"{label}: cur={logq:6.2f} ema={logema:6.2f}")
            else:
                rows.append(f"{label}: cur={logq:6.2f}")

        add("Q_alpha [log10]", self.idx_alpha)
        add("Q_beta [log10]", self.idx_beta)
        if self.seasonal_mode == "dynamic":
            add("Q_gamma [log10]", self.idx_gamma_end)
        return (" | " + " | ".join(rows)) if rows else ""

    # --------------------------------- MCMC --------------------------------- #
    def run(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg

        save_iters = list(range(cfg.burn, cfg.n_iter, cfg.thin))
        n_kept = max(0, len(save_iters))

        # storage
        keep_idx = 0
        self.keep = {
            "sigma": np.zeros(n_kept, float),
            "xi": np.zeros(n_kept, float),
            "mu": np.zeros((n_kept, self.T), float),
            "log_evidence": np.zeros(n_kept, float),
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

        ema_logZ: Optional[float] = None
        ema_Q = np.zeros(self.dim, float) if self.dim > 0 else None
        print_every = cfg.progress_every if cfg.progress_every > 0 else max(1, cfg.n_iter // 50)

        for it in range(cfg.n_iter):
            if cfg.progress:
                print(f"Iteration {it + 1}/{cfg.n_iter}")

            # 1) latent states via PGAS
            if self.dim > 0:
                self.update_states_pgas()
                current_log_ev = float(self.last_log_evidence)
            else:
                mu_now = self._mu_vec_current()
                current_log_ev = float(gev_loglike_sum(self.y, mu_now, self.sigma, self.xi))

            ema_logZ = self._ema(ema_logZ, current_log_ev, alpha=0.1)
            if cfg.progress:
                print(f"  log p(y | theta) [PF states-marginalized] = {current_log_ev:.6f}")

            # 2) update Q
            if self.dim > 0:
                if cfg.progress:
                    print("  Updating Q...")
                self.update_Q()
                if ema_Q is not None:
                    ema_Q = 0.9 * ema_Q + 0.1 * self.Q

            # 3) deterministic structural params
            if self.level_mode == "deterministic":
                if cfg.progress:
                    print("  Updating deterministic level...")
                self.update_level_value()
            if self.trend_mode == "deterministic":
                if cfg.progress:
                    print("  Updating deterministic slope...")
                self.update_slope()
            if self.seasonal_mode == "deterministic":
                if cfg.progress:
                    print("  Updating deterministic seasonality...")
                self.update_season_vec()

            # 4) observation params
            if cfg.progress:
                print("  Updating logsigma / xi ...")
            self.update_logsigma()
            self.update_xi()

            # 5) adapt proposal scales
            self._adapt_steps(it)

            # 6) progress line
            if cfg.progress and ((it + 1) % print_every == 0 or it == cfg.n_iter - 1):
                det_info = ""
                if self.level_mode == "deterministic":
                    det_info += f" | acc(level)={self._fmt_acc('level')} value={self.level_value:.4f}"
                if self.trend_mode == "deterministic":
                    det_info += f" | acc(slope)={self._fmt_acc('slope')} value={self.slope_value:.4f}"
                if self.seasonal_mode == "deterministic":
                    preview = np.array2string(self.season_vec[:min(3, self.period)], precision=3, separator=",")
                    det_info += f" | acc(season)={self._fmt_acc('season')} γ[:3]={preview} ..."

                q_info = self._q_snapshot(ema_Q) if self.dim > 0 else ""
                pf_info = ""
                if self.last_pf_diag:
                    d = self.last_pf_diag
                    pf_info = f" | PF: ESS(mean/min)={d['ess_mean']:.1f}/{d['ess_min']:.1f} MaxW(max)={d['maxw_max']:.4f}"

                print(
                    f"[it {it+1}/{cfg.n_iter}] "
                    f"logZ={current_log_ev:.3f} ema={ema_logZ:.3f} "
                    f"σ={np.exp(self.logsigma):.3f} ξ={self.xi:.3f} "
                    f"acc(logσ)={self._fmt_acc('logsigma')} acc(ξ)={self._fmt_acc('xi')}"
                    f"{det_info}{q_info}{pf_info}"
                )

                # early warnings
                if self.dim > 0 and np.any(self.Q < 1e-10):
                    print(
                        f"  [warn] Q near-zero at idx {np.where(self.Q < 1e-10)[0].tolist()} "
                        f"(min Q={float(np.min(self.Q)):.3e}). Consider larger b_q or a_q→1+."
                    )
                if self.last_pf_diag and self.last_pf_diag["ess_min"] < 0.2 * self.cfg.n_particles:
                    print("  [warn] PF degeneracy (ESS_min < 0.2*N). Consider more particles/regularization.)")

            # 7) store
            if it in save_iters and keep_idx < n_kept:
                mu_vec = self._mu_vec_current()
                self.keep["mu"][keep_idx, :] = mu_vec
                self.keep["sigma"][keep_idx] = float(np.exp(self.logsigma))
                self.keep["xi"][keep_idx] = float(self.xi)
                self.keep["log_evidence"][keep_idx] = current_log_ev
                if self.dim > 0:
                    self.keep["Q"][keep_idx, :] = self.Q
                if self.idx_alpha is not None:
                    self.keep["alpha_t"][keep_idx, :] = self.x[1 : self.T + 1, self.idx_alpha]
                if self.idx_beta is not None:
                    self.keep["beta_t"][keep_idx, :] = self.x[1 : self.T + 1, self.idx_beta]
                if self.seasonal_mode == "dynamic":
                    self.keep["gamma_t"][keep_idx, :] = self.x[1 : self.T + 1, self.idx_gamma_end]
                if self.level_mode == "deterministic":
                    self.keep["level_value"][keep_idx] = self.level_value
                if self.trend_mode == "deterministic":
                    self.keep["slope_value"][keep_idx] = self.slope_value
                if self.seasonal_mode == "deterministic":
                    self.keep["season_vector"][keep_idx, :] = self.season_vec
                keep_idx += 1

        if cfg.progress:
            print(
                f"[{it + 1}/{cfg.n_iter}] "
                f"acc(logsigma)={self._fmt_acc('logsigma')} "
                f"acc(xi)={self._fmt_acc('xi')} "
                f"acc(level)={self._fmt_acc('level')} "
                f"acc(slope)={self._fmt_acc('slope')} "
                f"acc(season)={self._fmt_acc('season')}"
            )
        return self.keep

    # ------------------------------- Persistence ---------------------------- #
    def save_posterior(self, out_npz_path: str, extra_meta: Optional[dict] = None) -> None:
        _ensure_dir(os.path.dirname(out_npz_path))

        arrays = dict(self.keep)
        arrays["y"] = self.y.copy()
        arrays["x_last"] = self.x[1 : self.T + 1].copy() if self.dim > 0 else np.zeros((self.T, 0))

        if self.true_mu_t is not None:
            arrays["true_mu_t"] = _np_array(self.true_mu_t, float)
        if self.true_alpha_t is not None:
            arrays["true_alpha_t"] = _np_array(self.true_alpha_t, float)
        if self.true_beta_t is not None:
            arrays["true_beta_t"] = _np_array(self.true_beta_t, float)
        if self.true_gamma_t is not None:
            arrays["true_gamma_t"] = _np_array(self.true_gamma_t, float)

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
            "true_sigma": self.true_sigma,
            "true_xi": self.true_xi,
            "true_Q": (None if self.true_Q is None else _np_array(self.true_Q, float).tolist()),
            "accept": self.accept,
            "proposals": self.proposals,
        }
        if extra_meta:
            meta.update(extra_meta)

        meta_path = out_npz_path.replace(".npz", ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[save] Posterior -> {out_npz_path}")
        print(f"[save] Metadata  -> {meta_path}")


# ------------------------- CLI / Example run & summaries -------------------- #
if __name__ == "__main__":
    import argparse
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from simulator.extremal_time_series import Extremal_Time_Series
    # from simulator.dgev_plotter import DGEVPlotter  # optional

    parser = argparse.ArgumentParser(description="DGEV PGAS Sampler (UKF proposal)")
    # Modes
    parser.add_argument("--level-mode", choices=["dynamic", "deterministic"], default="dynamic")
    parser.add_argument("--trend-mode", choices=["dynamic", "deterministic", "none"], default="none")
    parser.add_argument("--season-mode", choices=["dynamic", "deterministic", "none"], default="none")
    # Basics
    parser.add_argument("--period", type=int, default=4)
    parser.add_argument("--T", type=int, default=100)
    # Initial values (shared)
    parser.add_argument("--level-init", type=float, default=5.0)
    parser.add_argument("--slope-init", type=float, default=0.02)

    parser.add_argument("--true-sigma", type=float, default=2.0)
    parser.add_argument("--true-xi", type=float, default=0.1)
    parser.add_argument("--q-alpha", type=float, default=1e-3)
    parser.add_argument("--q-beta", type=float, default=1e-9)
    parser.add_argument("--q-gamma", type=float, default=1e-7)

    parser.add_argument("--prior-m-sigma", type=float, default=1.0)
    parser.add_argument("--prior-s-sigma", type=float, default=1.0)
    parser.add_argument("--prior-m-xi", type=float, default=0.0)
    parser.add_argument("--prior-s-xi", type=float, default=0.2)
    parser.add_argument("--prior-aq-alpha", type=float, default=1.1)
    parser.add_argument("--prior-aq-beta", type=float, default=1.1)
    parser.add_argument("--prior-aq-gamma", type=float, default=1.1)
    parser.add_argument("--prior-bq-alpha", type=float, default=1.0)
    parser.add_argument("--prior-bq-beta", type=float, default=1.0)
    parser.add_argument("--prior-bq-gamma", type=float, default=1.0)
    parser.add_argument("--prior-m-level", type=float, default=0.0)
    parser.add_argument("--prior-s-level", type=float, default=10.0)
    parser.add_argument("--prior-m-slope", type=float, default=0.0)
    parser.add_argument("--prior-s-slope", type=float, default=10.0)
    parser.add_argument("--prior-m-season", type=str, default=None,
                        help="Comma-separated first (p-1) means for deterministic seasonal prior (e.g. '0,0,0').")
    parser.add_argument("--prior-s-season", type=float, default=5.0)

    # Sampler config
    parser.add_argument("--n-iter", type=int, default=2000)
    parser.add_argument("--burn", type=int, default=100)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--step-logsigma", type=float, default=0.2)
    parser.add_argument("--step-xi", type=float, default=0.2)
    parser.add_argument("--step-level", type=float, default=0.02)
    parser.add_argument("--step-slope", type=float, default=0.0005)
    parser.add_argument("--step-season", type=float, default=0.02)
    parser.add_argument("--particles", type=int, default=250)
    parser.add_argument("--trans-eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--progress", default=True)
    parser.add_argument("--progress-every", type=int, default=10,
                        help="print compact summary every k iterations (0=auto)")

    # Adaptive RW–MH CLI
    parser.add_argument("--adapt-steps", action="store_true")
    parser.add_argument("--adapt-every", type=int, default=25)
    parser.add_argument("--adapt-until", choices=["burn", "all"], default="burn")
    parser.add_argument("--adapt-eta0", type=float, default=0.05)
    parser.add_argument("--adapt-decay", type=float, default=0.75)
    parser.add_argument("--adapt-target-1d", type=float, default=0.44)
    parser.add_argument("--step-min", type=float, default=1e-5)
    parser.add_argument("--step-max", type=float, default=1.0)

    # UKF params
    parser.add_argument("--ukf-alpha", type=float, default=1e-3)
    parser.add_argument("--ukf-beta", type=float, default=2.0)
    parser.add_argument("--ukf-kappa", type=float, default=0.0)
    parser.add_argument("--R-min", type=float, default=1e-6)
    parser.add_argument("--R-max", type=float, default=1e6)

    # Output
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--show-plots", action="store_true")

    args = parser.parse_args()
    np.random.seed(args.seed)

    sim_level_mode = args.level_mode
    sim_trend_mode = args.trend_mode
    sim_season_mode = args.season_mode

    # Seasonal priors for simulator
    m0_season_first = np.zeros(args.period - 1, float)
    v0_season_first = np.full(args.period - 1, 0.5, float)
    if sim_season_mode == "deterministic":
        m0_season_first = build_seasonal(args.period)
    elif sim_season_mode == "none":
        m0_season_first = None
        v0_season_first = None

    # Simulate data
    ts = Extremal_Time_Series(
        parameters=(args.true_sigma, args.true_xi),
        level_mode=sim_level_mode,
        trend_mode=sim_trend_mode,
        seasonal_mode=sim_season_mode,
        period=args.period,
        q_level=args.q_alpha,
        q_trend=args.q_beta,
        q_season=args.q_gamma,
        m0_level=args.level_init,
        v0_level=0.2,
        m0_trend=(args.slope_init if sim_trend_mode != "none" else 0.0),
        v0_trend=0.05,
        m0_season=m0_season_first,
        v0_season=v0_season_first,
        start_date=datetime(1980, 1, 1),
    )

    y = []
    for _ in range(args.T):
        ts.move()
        y.append(ts.measure())
    y = _np_array(y, float)

    truths = ts.get_truth_paths(as_numpy=False)
    mu_T = _np_array(truths["mu"][1 : 1 + args.T], float)
    alpha_T = _np_array(truths["alpha"][1 : 1 + args.T], float) if sim_level_mode == "dynamic" else None
    beta_T = _np_array(truths["beta"][1 : 1 + args.T], float) if sim_trend_mode == "dynamic" else None
    gamma_T = _np_array(truths["gamma_last"][1 : 1 + args.T], float) if sim_season_mode == "dynamic" else None

    # Priors & config
    m_season_prior = parse_csv_floats(args.prior_m_season)
    if m_season_prior is not None and len(m_season_prior) != args.period - 1:
        raise ValueError(f"--prior-m-season must have length {args.period - 1} (got {len(m_season_prior)}).")

    priors = Priors(
        m_sigma=float(args.prior_m_sigma),
        s_sigma=float(args.prior_s_sigma),
        m_xi=float(args.prior_m_xi),
        s_xi=float(args.prior_s_xi),
        a_q_alpha=float(args.prior_aq_alpha),
        b_q_alpha=float(args.prior_bq_alpha),
        a_q_beta=float(args.prior_aq_beta),
        b_q_beta=float(args.prior_bq_beta),
        a_q_gamma=float(args.prior_aq_gamma),
        b_q_gamma=float(args.prior_bq_gamma),
        m_level=float(args.prior_m_level),
        s_level=float(args.prior_s_level),
        m_slope=float(args.prior_m_slope),
        s_slope=float(args.prior_s_slope),
        m_season=m_season_prior,
        s_season=float(args.prior_s_season),
    )

    cfg = SamplerConfig(
        n_iter=args.n_iter,
        burn=args.burn,
        thin=args.thin,
        step_logsigma=args.step_logsigma,
        step_xi=args.step_xi,
        step_level=args.step_level,
        step_slope=args.step_slope,
        step_season=args.step_season,
        n_particles=args.particles,
        trans_eps=args.trans_eps,
        random_seed=args.seed,
        progress=bool(args.progress),
        progress_every=int(args.progress_every),
        adapt_steps=bool(args.adapt_steps),
        adapt_every=int(args.adapt_every),
        adapt_until=str(args.adapt_until),
        adapt_eta0=float(args.adapt_eta0),
        adapt_eta_decay=float(args.adapt_decay),
        adapt_target_1d=float(args.adapt_target_1d),
        step_min=float(args.step_min),
        step_max=float(args.step_max),
        ukf_alpha=float(args.ukf_alpha),
        ukf_beta=float(args.ukf_beta),
        ukf_kappa=float(args.ukf_kappa),
        R_min=float(args.R_min),
        R_max=float(args.R_max),
    )

    seasonal_init_pminus1 = (
        _np_array(m_season_prior, float)
        if (sim_season_mode == "deterministic" and m_season_prior is not None)
        else (build_seasonal(args.period) if sim_season_mode == "deterministic" else None)
    )

    sampler = DGEVParticleGibbs(
        y=y,
        period=args.period,
        level_mode=sim_level_mode,
        trend_mode=sim_trend_mode,
        seasonal_mode=sim_season_mode,
        m0_level=args.level_init,
        v0_level=0.2,
        m0_trend=(args.slope_init if sim_trend_mode != "none" else 0.0),
        v0_trend=0.05,
        m0_season=(m0_season_first if sim_season_mode == "dynamic" else None),
        v0_season=(v0_season_first if sim_season_mode == "dynamic" else None),
        priors=priors,
        cfg=cfg,
        level_value_init=args.level_init,
        slope_value_init=args.slope_init,
        seasonal_vector_init=seasonal_init_pminus1,
    )

    true_Q: List[float] = []
    if sim_level_mode == "dynamic":
        true_Q.append(args.q_alpha)
    if sim_trend_mode == "dynamic":
        true_Q.append(args.q_beta)
    if sim_season_mode == "dynamic":
        true_Q += [args.q_gamma] + [0.0] * (args.period - 2)

    sampler.set_truth(
        sigma=args.true_sigma,
        xi=args.true_xi,
        Q=(np.asarray(true_Q, float) if len(true_Q) else None),
    )
    sampler.set_truth_paths(mu=mu_T, alpha=alpha_T, beta=beta_T, gamma=gamma_T)

    tag = f"{sim_level_mode}-{sim_trend_mode}-{sim_season_mode}"
    out_dir = args.out_dir or os.path.join(
        "results", "simulations", "DGEV", f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    _ensure_dir(out_dir)

    t0 = time.time()
    posterior = sampler.run()
    elapsed = time.time() - t0
    print(f"Sampler run time: {elapsed:.2f}s")

    sampler.save_posterior(
        out_npz_path=os.path.join(out_dir, "posterior.npz"),
        extra_meta={"modes": tag, "elapsed_seconds": float(elapsed)},
    )

    # ---- Summaries ----
    print(f"Posterior mean sigma: {np.mean(posterior['sigma']):.3f} (true {args.true_sigma})")
    print(f"Posterior mean xi:    {np.mean(posterior['xi']):.3f} (true {args.true_xi})")

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

    if "log_evidence" in posterior and posterior["log_evidence"].size > 0:
        le = posterior["log_evidence"]
        print(f"Mean log p(y|theta): {np.nanmean(le):.3f} | Median: {np.nanmedian(le):.3f} | Best: {np.nanmax(le):.3f}")
