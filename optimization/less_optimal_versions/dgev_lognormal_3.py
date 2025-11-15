# %% optimization/dgev_pgas_ln.py
from __future__ import annotations

import os, sys, math, json, time
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, Dict, List, Sequence

import numpy as np
from tqdm import tqdm
from datetime import datetime

# =============================================================================
# Utilities
# =============================================================================

def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

def build_seasonal(period: int) -> np.ndarray:
    """Default smooth seasonal for first (p-1) entries (last implied by sum-to-zero)."""
    g = np.cos(2 * np.pi * np.arange(period) / period)
    g -= np.mean(g)
    return g[: period - 1].astype(float)

def parse_csv_floats(s: Optional[str]) -> Optional[List[float]]:
    """Parse comma-separated floats like '0,0.2,-0.1' -> [0.0, 0.2, -0.1]."""
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return [float(tok) for tok in s.split(",")]

def _mad(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    if v.size == 0:
        return 0.0
    med = np.median(v)
    return float(np.median(np.abs(v - med)))

def _robust_sd(v: np.ndarray) -> float:
    return _mad(np.asarray(v, float)) / 1.4826 if v.size else 0.0

def _spd_solve(M: np.ndarray, B: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Solve M X = B for SPD (or near-SPD) M using Cholesky with small jitter.
    B can be a vector or matrix.
    """
    n = M.shape[0]
    I = np.eye(n)
    for k in range(3):
        try:
            L = np.linalg.cholesky(M + (eps * (10**k)) * I)
            Y = np.linalg.solve(L, B)
            return np.linalg.solve(L.T, Y)
        except np.linalg.LinAlgError:
            continue
    return np.linalg.pinv(M) @ B

# =============================================================================
# Univariate slice sampler (stepping-out + shrinkage) on R
# =============================================================================

class Slice1D:
    """
    Univariate slice sampler for log-densities h(z) (up to a constant).
    - Stepping-out width w and max steps m
    - Classic shrinkage until acceptance
    """
    def __init__(self, h, w: float = 1.0, m: int = 20, rng: Optional[np.random.Generator] = None):
        self.h = h
        self.w = float(w)
        self.m = int(m)
        self.rng = rng if rng is not None else np.random.default_rng()

    def sample(self, z0: float, n: int = 1) -> np.ndarray:
        out = np.empty(n, float)
        z = float(z0)
        for i in range(n):
            hz = self.h(z)
            # draw vertical level
            u = self.rng.random()
            y = hz + math.log(u)
            # stepping-out
            w = self.w
            L = z - w * self.rng.random()
            R = L + w
            J = int(self.rng.integers(0, self.m))
            K = self.m - 1 - J
            while J > 0 and self.h(L) > y:
                L -= w
                J -= 1
            while K > 0 and self.h(R) > y:
                R += w
                K -= 1
            # shrinkage
            while True:
                z_new = self.rng.uniform(L, R)
                if self.h(z_new) >= y:
                    z = z_new
                    break
                elif z_new < z:
                    L = z_new
                else:
                    R = z_new
            out[i] = z
        return out

# =============================================================================
# GEV helpers
# =============================================================================

def gev_logpdf(y: float, mu: float, sigma: float, xi: float) -> float:
    """log f(y | mu, sigma>0, xi) under GEV parameterization (mu, sigma, xi)."""
    if sigma <= 0.0 or np.isnan(mu):
        return -np.inf
    z = (y - mu) / sigma
    u = 1.0 + xi * z
    if u <= 0.0:
        return -np.inf
    if abs(xi) < 1e-8:  # Gumbel limit
        return -np.log(sigma) - np.exp(-z) - z
    return -np.log(sigma) - (1.0 + 1.0 / xi) * np.log(u) - u ** (-1.0 / xi)

def gev_loglike_sum(y: np.ndarray, mu_vec: np.ndarray, sigma: float, xi: float) -> float:
    """Sum_t log f(y_t | mu_t, sigma, xi)."""
    if sigma <= 0.0 or np.any(np.isnan(mu_vec)):
        return -np.inf
    z = (y - mu_vec) / sigma
    u = 1.0 + xi * z
    if np.any(u <= 0.0):
        return -np.inf
    if abs(xi) < 1e-8:
        return float(np.sum(-np.log(sigma) - np.exp(-z) - z))
    return float(np.sum(-np.log(sigma) - (1.0 + 1.0 / xi) * np.log(u) - u ** (-1.0 / xi)))

# =============================================================================
# Priors & Config
#   - Observation: σ² ~ InvGamma(a_sigma, b_sigma)
#   - ξ ~ Uniform[xi_lower, xi_upper]  (bounded uniform prior)
#   - Process SDs s_* still have log-normal priors (slice sampling on ln s)
# =============================================================================

@dataclass
class Priors:
    # Observation: σ² ~ InvGamma(a_sigma, b_sigma)
    # (shape a_sigma, scale b_sigma, on the variance v = σ²)
    a_sigma: float = 2.0
    b_sigma: float = 2.0

    # Bounded uniform prior for ξ
    # ξ ~ Uniform[xi_lower, xi_upper]
    xi_lower: float = -0.5
    xi_upper: float = 0.5

    # m0 priors (used both for dynamic x0 means and for deterministic components)
    m_m0_alpha: float = 0.0
    s_m0_alpha: float = 10.0
    m_m0_beta: float = 0.0
    s_m0_beta: float = 10.0
    m_m0_gamma: Optional[Sequence[float]] = None  # length p-1 if provided (NEWEST-FIRST)
    s_m0_gamma: float = 5.0

    # Initial-state variances P0 ~ InvGamma(a_P0_*, b_P0_*)
    a_P0_alpha: float = 2.0
    b_P0_alpha: float = 1.0
    a_P0_beta: float = 2.0
    b_P0_beta: float = 1.0
    a_P0_gamma: float = 2.0
    b_P0_gamma: float = 1.0   # shared across p-1 seasonal coords

    # Deterministic components’ Gaussian priors (used when mode='deterministic')
    m_level: float = 0.0
    s_level: float = 10.0
    m_slope: float = 0.0
    s_slope: float = 10.0
    m_season: Optional[Sequence[float]] = None  # first p-1 means (NEWEST-FIRST)
    s_season: float = 5.0

    # Log-Normal priors for process SDs: s_k ~ LogNormal(mu, sd^2)
    mu_log_s_alpha: float = -2.3
    sd_log_s_alpha: float = 0.7
    mu_log_s_beta:  float = -3.5
    sd_log_s_beta:  float = 0.7
    mu_log_s_gamma: float = -3.0
    sd_log_s_gamma: float = 0.7


@dataclass
class SamplerConfig:
    n_iter: int = 5000
    burn: int = 1000
    thin: int = 5
    random_seed: Optional[int] = 123
    progress: bool = True
    progress_every: int = 0  # 0 => auto (~2% of n_iter)

    # PF
    n_particles: int = 200
    trans_eps: float = 1e-8
    ess_threshold_frac: float = 0.50   # resample when ESS < frac * N

    # RW–MH step sizes (initial)
    step_logsigma: float = 0.05
    step_xi: float = 0.05
    step_level: float = 0.05
    step_slope: float = 0.02
    step_season: float = 0.05

    # Adaptive RW–MH (Robbins–Monro; windowed)
    adapt_steps: bool = True
    adapt_every: int = 25
    adapt_until: str = "burn"         # "burn" or "all"
    adapt_target_1d: float = 0.44
    adapt_eta0: float = 0.05
    adapt_eta_decay: float = 0.75
    step_min: float = 1e-5
    step_max: float = 1.0

    # Slice sampler knobs for log s updates
    slice_w: float = 1.0
    slice_m: int = 20

# =============================================================================
# Particle Gibbs with Ancestor Sampling (PGAS) for structural DGEV
#
# State layout (unchanged):
#   - Dynamic state layout: [alpha] [beta] [g1 ... g_{p-1}]
#   - Seasonal vector is NEWEST-FIRST; observation loads FIRST seasonal coord.
#   - Transition for season:
#         g1(t) = -sum(g1..g_{p-1})(t-1) + ε_{γ,t}
#         gk(t) = g_{k-1}(t-1),  k=2..p-1
#     ⇒ Q has s_γ^2 on the FIRST seasonal coord only (others 0).
# =============================================================================

class DGEVParticleGibbs:
    # --------------------------- Construction --------------------------- #
    def __init__(
        self,
        y: np.ndarray,
        period: int,
        level_mode: str = "dynamic",
        trend_mode: str = "dynamic",
        seasonal_mode: str = "dynamic",
        # initial values for dynamic x0 priors (variance scale)
        m0_level_init: float = 0.0, P0_level_init: float = 1.0,
        m0_trend_init: float = 0.0, P0_trend_init: float = 1.0,
        m0_season_init: Sequence[float] | None = None,  # length p-1, NEWEST-FIRST
        P0_season_init: float = 1.0,
        # observation initial values
        sigma_init: float = 1.0,
        xi_init: float = 0.0,
        # process sd initial values
        s_alpha_init: float = 1e-2,
        s_beta_init:  float = 1e-3,
        s_gamma_init: float = 1e-3,
        # priors and config
        priors: Priors = Priors(),
        cfg: SamplerConfig = SamplerConfig(),
        # deterministic initial values (used if mode='deterministic')
        level_value_init: float = 0.0,
        slope_value_init: float = 0.0,
        seasonal_vector_init: Optional[Sequence[float]] = None,  # p-1 entries, NEWEST-FIRST
    ):
        # Data
        self.y = np.asarray(y, float)
        self.T = int(self.y.size)
        self.period = int(period)
        if self.period < 2:
            raise ValueError("period must be >= 2")

        # Modes
        ok = {"dynamic", "deterministic", "none"}
        if level_mode not in ok or trend_mode not in ok or seasonal_mode not in ok:
            raise ValueError("invalid mode")
        if trend_mode == "dynamic" and level_mode != "dynamic":
            raise ValueError("dynamic trend requires dynamic level")
        self.level_mode, self.trend_mode, self.seasonal_mode = level_mode, trend_mode, seasonal_mode

        # Priors / cfg
        self.priors, self.cfg = priors, cfg
        if cfg.random_seed is not None:
            np.random.seed(cfg.random_seed)
            self._rng = np.random.default_rng(cfg.random_seed)
        else:
            self._rng = np.random.default_rng()

        # Check bounded uniform prior for xi
        if not (self.priors.xi_lower < self.priors.xi_upper):
            raise ValueError("priors.xi_lower must be < priors.xi_upper")
        if not (self.priors.xi_lower <= xi_init <= self.priors.xi_upper):
            raise ValueError(
                f"Initial xi_init={xi_init} must lie in [xi_lower, xi_upper] = "
                f"[{self.priors.xi_lower}, {self.priors.xi_upper}]"
            )

        # Dynamic state layout
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
            self.idx_g_start = self.idx_g_end = None

        if self.dim == 0 and (self.seasonal_mode != "deterministic") and (self.level_mode != "deterministic"):
            raise ValueError("At least one contribution to μ_t must exist (dynamic or deterministic).")

        # ---- Deterministic parameters (outside state)
        self.level_value = float(level_value_init)
        self.slope_value = float(slope_value_init)
        if self.seasonal_mode == "deterministic":
            if seasonal_vector_init is not None:
                base = np.asarray(seasonal_vector_init, float)
                if base.size != self.period - 1:
                    raise ValueError("seasonal_vector_init must have length p-1 (NEWEST-FIRST).")
            else:
                if self.priors.m_season is not None:
                    base = np.asarray(self.priors.m_season, float)
                    if base.size != self.period - 1:
                        raise ValueError("priors.m_season must have length p-1 (NEWEST-FIRST).")
                else:
                    base = build_seasonal(self.period)
            self.season_vec = np.r_[base, -float(np.sum(base))].astype(float)
        else:
            self.season_vec = None

        # ---- Observation params (GEV)
        self.logsigma = float(np.log(max(1e-12, sigma_init)))
        self.sigma    = float(np.exp(self.logsigma))
        self.xi       = float(xi_init)

        # ---- Process SDs (log-normal priors)
        self.s_alpha = float(max(1e-12, s_alpha_init)) if self.idx_alpha is not None else 0.0
        self.s_beta  = float(max(1e-12, s_beta_init))  if self.idx_beta  is not None else 0.0
        self.s_gamma = float(max(1e-12, s_gamma_init)) if self.seasonal_mode == "dynamic" else 0.0

        # ---- Initial m0 and P0 for dynamic coords (variance scale)
        self.m0_alpha = float(m0_level_init) if self.idx_alpha is not None else 0.0
        self.P0_alpha = float(P0_level_init) if self.idx_alpha is not None else 0.0
        self.m0_beta  = float(m0_trend_init) if self.idx_beta  is not None else 0.0
        self.P0_beta  = float(P0_trend_init) if self.idx_beta  is not None else 0.0
        if self.seasonal_mode == "dynamic":
            if m0_season_init is None:
                self.m0_gamma = np.zeros(self.period - 1, float)
            else:
                g = np.asarray(m0_season_init, float)
                if g.size != self.period - 1:
                    raise ValueError("m0_season_init must have length p-1 (NEWEST-FIRST)")
                self.m0_gamma = g
            self.P0_gamma = float(P0_season_init)
        else:
            self.m0_gamma = None
            self.P0_gamma = 0.0

        # ---- Latent path x_{0:T}
        self.x = np.zeros((self.T + 1, self.dim), float)
        if self.dim > 0:
            m0_vec, P0_diag = self._current_m0_P0()
            self.x[0] = np.random.multivariate_normal(
                m0_vec, np.diag(P0_diag) + 1e-10 * np.eye(self.dim)
            )
            self._propagate_initial_path(Q_init=np.full(self.dim, 1e-6, float))

        # ---- Storage
        self.keep: Dict[str, np.ndarray] = {}

        # ---- MH bookkeeping
        self.accept = {
            "logsigma": 0, "xi": 0, "level": 0, "slope": 0, "season": 0,
        }
        self.proposals = dict(self.accept)

        # Adaptation bookkeeping (windowed)
        self._mh_prev_acc = dict(self.accept)
        self._mh_prev_prop = dict(self.proposals)
        self._adapt_round = 0

        # Optional truth overlays
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

        # ---- Slice samplers for z = log s
        self._slice_alpha = Slice1D(
            lambda z: self._h_lognorm(z, *self._ln_params_alpha()),
            w=self.cfg.slice_w, m=self.cfg.slice_m, rng=self._rng
        ) if self.idx_alpha is not None else None

        self._slice_beta = Slice1D(
            lambda z: self._h_lognorm(z, *self._ln_params_beta()),
            w=self.cfg.slice_w, m=self.cfg.slice_m, rng=self._rng
        ) if self.idx_beta is not None else None

        self._slice_gamma = Slice1D(
            lambda z: self._h_lognorm(z, *self._ln_params_gamma()),
            w=self.cfg.slice_w, m=self.cfg.slice_m, rng=self._rng
        ) if self.seasonal_mode == "dynamic" else None

    # --------------------- Truth registration (optional) --------------------- #
    def set_truth(self, sigma: Optional[float] = None, xi: Optional[float] = None, Q: Optional[np.ndarray] = None) -> None:
        self.true_sigma = sigma
        self.true_xi = xi
        self.true_Q = None if Q is None else np.asarray(Q, float)

    def set_truth_paths(self, mu: Optional[np.ndarray] = None,
                        alpha: Optional[np.ndarray] = None,
                        beta: Optional[np.ndarray] = None,
                        gamma: Optional[np.ndarray] = None) -> None:
        self.true_mu_t = None if mu is None else np.asarray(mu, float)
        self.true_alpha_t = None if alpha is None else np.asarray(alpha, float)
        self.true_beta_t = None if beta is None else np.asarray(beta, float)
        self.true_gamma_t = None if gamma is None else np.asarray(gamma, float)

    # ------------------ Helpers: μ and current m0/P0 ------------------ #
    def _mu_vec_current(self) -> np.ndarray:
        return np.array([self.mu_from_state(self.x[t], t - 1) for t in range(1, self.T + 1)], float)

    def _current_m0_P0(self) -> Tuple[np.ndarray, np.ndarray]:
        m0, P0 = [], []
        if self.idx_alpha is not None:
            m0.append(self.m0_alpha)
            P0.append(self.P0_alpha)
        if self.idx_beta is not None:
            m0.append(self.m0_beta)
            P0.append(self.P0_beta)
        if self.seasonal_mode == "dynamic":
            m0.extend(list(self.m0_gamma))  # NEWEST-FIRST
            P0.extend([self.P0_gamma] * (self.period - 1))
        return np.asarray(m0, float), np.asarray(P0, float)

    # ----------------------------- State model ------------------------------ #
    def _state_mean(self, x_prev: np.ndarray, t: int) -> np.ndarray:
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
        # dynamic seasonal (NEWEST-FIRST; innovation on FIRST coord)
        if self.seasonal_mode == "dynamic":
            gs, ge = self.idx_g_start, self.idx_g_end
            if gs < ge + 1:
                # shift-down: gk(t) = g_{k-1}(t-1), k=2..p-1
                m[gs + 1 : ge + 1] = x_prev[gs : ge]
                # first coord closure to sum-zero
                prev = x_prev[gs : ge + 1]
                m[gs] = -float(np.sum(prev))
        return m

    def _alpha_contribution(self, x_t: np.ndarray, t: int) -> float:
        if self.level_mode == "dynamic":
            return float(x_t[self.idx_alpha]) if self.idx_alpha is not None else 0.0
        base = self.level_value
        if self.idx_beta is not None:      # dynamic trend
            return float(base + x_t[self.idx_beta] * t)
        if self.trend_mode == "deterministic":
            return float(base + self.slope_value * t)
        return float(base)

    def _season_contribution(self, x_t: np.ndarray, t: int) -> float:
        if self.seasonal_mode == "dynamic":
            return float(x_t[self.idx_g_start]) if self.idx_g_start is not None else 0.0
        if self.seasonal_mode == "deterministic":
            return float(self.season_vec[t % self.period])
        return 0.0

    def mu_from_state(self, x_t: np.ndarray, t: int) -> float:
        return self._alpha_contribution(x_t, t) + self._season_contribution(x_t, t)

    def _transition_logpdf(self, x_prev: np.ndarray, x_cur: np.ndarray, t: int) -> float:
        mean = self._state_mean(x_prev, t)
        eps = self.cfg.trans_eps
        out = 0.0
        for tag_idx, tag in enumerate(self._layout):
            if tag == "alpha":
                var = (self.s_alpha ** 2) if self.s_alpha > 0 else eps
            elif tag == "beta":
                var = (self.s_beta ** 2) if self.s_beta > 0 else eps
            elif tag.startswith("g"):
                var = (self.s_gamma ** 2) if (tag_idx == self.idx_g_start) else 0.0
                if var <= 0.0:
                    var = eps
            else:
                var = eps
            diff = x_cur[tag_idx] - mean[tag_idx]
            out += -0.5 * (math.log(2.0 * math.pi * var) + (diff * diff) / var)
        return float(out)

    def _transition_sample(self, x_prev: np.ndarray, t: int) -> np.ndarray:
        mean = self._state_mean(x_prev, t)
        var = np.zeros(self.dim, float)
        for tag_idx, tag in enumerate(self._layout):
            if tag == "alpha":
                var[tag_idx] = (self.s_alpha ** 2) if self.s_alpha > 0 else self.cfg.trans_eps
            elif tag == "beta":
                var[tag_idx] = (self.s_beta ** 2) if self.s_beta > 0 else self.cfg.trans_eps
            elif tag.startswith("g"):
                var[tag_idx] = (self.s_gamma ** 2) if (tag_idx == self.idx_g_start) else 0.0
                if var[tag_idx] <= 0.0:
                    var[tag_idx] = self.cfg.trans_eps
            else:
                var[tag_idx] = self.cfg.trans_eps
        return mean + np.random.normal(0.0, np.sqrt(var), size=self.dim)

    def _propagate_initial_path(self, Q_init: np.ndarray) -> None:
        for t in range(1, self.T + 1):
            if self.dim == 0:
                break
            mean = self._state_mean(self.x[t - 1], t)
            self.x[t] = mean + np.random.normal(0.0, np.sqrt(Q_init), size=self.dim)

    # --------------------- Innovation sums-of-squares --------------------- #
    def _innovation_ss_alpha(self) -> Tuple[float, int]:
        if self.idx_alpha is None:
            return 0.0, 0
        ss = 0.0
        for t in range(1, self.T + 1):
            drift = 0.0
            if self.idx_beta is not None:
                drift = self.x[t - 1, self.idx_beta]
            elif self.trend_mode == "deterministic":
                drift = float(self.slope_value)
            mean = self.x[t - 1, self.idx_alpha] + drift
            ss += (self.x[t, self.idx_alpha] - mean) ** 2
        return float(ss), self.T

    def _innovation_ss_beta(self) -> Tuple[float, int]:
        if self.idx_beta is None:
            return 0.0, 0
        d = self.x[1:, self.idx_beta] - self.x[:-1, self.idx_beta]
        return float(np.sum(d * d)), self.T

    def _innovation_ss_gamma(self) -> Tuple[float, int]:
        """
        Seasonal innovation on FIRST coord:
            g1(t) ~ N(-sum(prev seasonal coords), s_gamma^2)
        """
        if self.seasonal_mode != "dynamic":
            return 0.0, 0
        gs, ge = self.idx_g_start, self.idx_g_end
        ss = 0.0
        for t in range(1, self.T + 1):
            prev = self.x[t - 1, gs : ge + 1]
            mean_new_first = -float(np.sum(prev))
            ss += (self.x[t, gs] - mean_new_first) ** 2
        return float(ss), self.T

    # ----------------- Log-Normal prior on s: log-posterior h(z) ----------- #
    # h(z) = -T z - 0.5 * SS * exp(-2z) - (z - mu)^2 / (2 v) + const
    def _h_lognorm(self, z: float, SS: float, T_eff: int, mu: float, var: float) -> float:
        SS = max(float(SS), 1e-300)
        T_eff = max(int(T_eff), 0)
        var = max(float(var), 1e-16)
        return -(T_eff) * z - 0.5 * SS * math.exp(-2.0 * z) - 0.5 * ((z - mu) ** 2) / var

    def _ln_params_alpha(self) -> Tuple[float, int, float, float]:
        SS, T_eff = self._innovation_ss_alpha()
        mu0 = float(self.priors.mu_log_s_alpha)
        v0  = float(self.priors.sd_log_s_alpha) ** 2
        return SS, T_eff, mu0, v0

    def _ln_params_beta(self) -> Tuple[float, int, float, float]:
        SS, T_eff = self._innovation_ss_beta()
        mu0 = float(self.priors.mu_log_s_beta)
        v0  = float(self.priors.sd_log_s_beta) ** 2
        return SS, T_eff, mu0, v0

    def _ln_params_gamma(self) -> Tuple[float, int, float, float]:
        SS, T_eff = self._innovation_ss_gamma()
        mu0 = float(self.priors.mu_log_s_gamma)
        v0  = float(self.priors.sd_log_s_gamma) ** 2
        return SS, T_eff, mu0, v0

    def update_process_s_lognormal_slice(self) -> None:
        # α
        if self.idx_alpha is not None:
            SS, T_eff, mu0, v0 = self._ln_params_alpha()  # recomputed for completeness
            z0 = math.log(max(self.s_alpha, 1e-12))
            z  = float(self._slice_alpha.sample(z0, 1)[0])
            self.s_alpha = float(np.exp(z))
        # β
        if self.idx_beta is not None:
            SS, T_eff, mu0, v0 = self._ln_params_beta()
            z0 = math.log(max(self.s_beta, 1e-12))
            z  = float(self._slice_beta.sample(z0, 1)[0])
            self.s_beta = float(np.exp(z))
        # γ
        if self.seasonal_mode == "dynamic":
            SS, T_eff, mu0, v0 = self._ln_params_gamma()
            z0 = math.log(max(self.s_gamma, 1e-12))
            z  = float(self._slice_gamma.sample(z0, 1)[0])
            self.s_gamma = float(np.exp(z))

    # ----------------------- Observation parameter updates ------------------ #
    def _mh_accept(self, logacc: float) -> bool:
        return (np.log(np.random.rand()) < min(0.0, logacc))

    def _get_step(self, key: str) -> float:
        if   key == "logsigma": return self.cfg.step_logsigma
        elif key == "xi":       return self.cfg.step_xi
        elif key == "level":    return self.cfg.step_level
        elif key == "slope":    return self.cfg.step_slope
        elif key == "season":   return self.cfg.step_season
        else: raise KeyError(key)

    def _set_step(self, key: str, val: float) -> None:
        v = float(np.clip(val, self.cfg.step_min, self.cfg.step_max))
        if   key == "logsigma": self.cfg.step_logsigma = v
        elif key == "xi":       self.cfg.step_xi = v
        elif key == "level":    self.cfg.step_level = v
        elif key == "slope":    self.cfg.step_slope = v
        elif key == "season":   self.cfg.step_season = v
        else: raise KeyError(key)

    # Inverse-gamma prior on σ²:
    # v = σ² ~ IG(a_sigma, b_sigma) (shape a, scale b)
    # p(v) ∝ v^{-(a+1)} exp(-b / v)
    # For l = ln σ, v = exp(2l), dv/dl = 2 exp(2l) = 2 v
    # log p(l) = -a * log v - b / v + const = -2a l - b * exp(-2l) + const
    def _log_prior_logsigma(self, logsigma: float) -> float:
        a = float(self.priors.a_sigma)
        b = float(self.priors.b_sigma)
        return -2.0 * a * logsigma - b * math.exp(-2.0 * logsigma)

    def update_logsigma(self) -> None:
        step = self.cfg.step_logsigma
        cur = self.logsigma
        prop = cur + np.random.normal(0.0, step)
        sigma_cur, sigma_prop = float(np.exp(cur)), float(np.exp(prop))
        mu_vec = self._mu_vec_current()
        ll_old = gev_loglike_sum(self.y, mu_vec, sigma_cur, self.xi)
        ll_new = gev_loglike_sum(self.y, mu_vec, sigma_prop, self.xi)
        self.proposals["logsigma"] += 1
        if ll_new == -np.inf:
            return
        lp_old = self._log_prior_logsigma(cur)
        lp_new = self._log_prior_logsigma(prop)
        if self._mh_accept((ll_new + lp_new) - (ll_old + lp_old)):
            self.logsigma = prop
            self.sigma = sigma_prop
            self.accept["logsigma"] += 1

    def update_xi(self) -> None:
        """
        Random-walk MH update for ξ with **bounded uniform prior** on [xi_lower, xi_upper].
        Proposals outside the interval are immediately rejected.
        Inside the interval, the prior is constant, so the MH ratio depends on the likelihood only.
        """
        step = self.cfg.step_xi
        cur = self.xi
        prop = cur + np.random.normal(0.0, step)

        self.proposals["xi"] += 1

        # Enforce bounded uniform prior support
        lb = float(self.priors.xi_lower)
        ub = float(self.priors.xi_upper)
        if not (lb <= prop <= ub):
            return

        mu_vec = self._mu_vec_current()
        ll_old = gev_loglike_sum(self.y, mu_vec, self.sigma, cur)
        ll_new = gev_loglike_sum(self.y, mu_vec, self.sigma, prop)
        if ll_new == -np.inf:
            return

        # Uniform prior over [lb, ub] ⇒ constant log prior, cancels in ratio
        if self._mh_accept(ll_new - ll_old):
            self.xi = prop
            self.accept["xi"] += 1

    # ---------------- Deterministic structural parameter updates ------------- #
    def update_level_value(self) -> None:
        if self.level_mode != "deterministic":
            return
        step = self.cfg.step_level
        cur = self.level_value
        prop = cur + np.random.normal(0.0, step)

        old = self.level_value
        self.level_value = prop
        mu_vec_prop = self._mu_vec_current()
        self.level_value = old
        mu_vec_old = self._mu_vec_current()

        ll_old = gev_loglike_sum(self.y, mu_vec_old, self.sigma, self.xi)
        ll_new = gev_loglike_sum(self.y, mu_vec_prop, self.sigma, self.xi)
        self.proposals["level"] += 1
        if ll_new == -np.inf:
            return
        lp_old = -0.5 * ((cur - self.priors.m_level) ** 2) / (self.priors.s_level ** 2)
        lp_new = -0.5 * ((prop - self.priors.m_level) ** 2) / (self.priors.s_level ** 2)
        if self._mh_accept((ll_new + lp_new) - (ll_old + lp_old)):
            self.level_value = prop
            self.accept["level"] += 1

    def _alpha_transition_loglike_given_slope(self, slope: float) -> float:
        if self.idx_alpha is None:
            return 0.0
        var = (self.s_alpha ** 2) if self.s_alpha > 0 else self.cfg.trans_eps
        inv_var = 1.0 / var
        cst = -0.5 * np.log(2.0 * np.pi * var)
        ll = 0.0
        for t in range(1, self.T + 1):
            if self.idx_beta is not None:
                drift = self.x[t - 1, self.idx_beta]
            else:
                drift = slope
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

        if self.idx_alpha is not None:
            ll_trans_old = self._alpha_transition_loglike_given_slope(cur)
            ll_trans_new = self._alpha_transition_loglike_given_slope(prop)
        else:
            ll_trans_old = 0.0
            ll_trans_new = 0.0

        lp_old = -0.5 * ((cur - self.priors.m_slope) ** 2) / (self.priors.s_slope ** 2)
        lp_new = -0.5 * ((prop - self.priors.m_slope) ** 2) / (self.priors.s_slope ** 2)

        logacc = (ll_obs_new + ll_trans_new + lp_new) - (ll_obs_old + ll_trans_old + lp_old)
        self.proposals["slope"] += 1
        if self._mh_accept(logacc):
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
        mu_vec_prop = self._mu_vec_current()
        self.season_vec = old
        mu_vec_old = self._mu_vec_current()

        ll_old = gev_loglike_sum(self.y, mu_vec_old, self.sigma, self.xi)
        ll_new = gev_loglike_sum(self.y, mu_vec_prop, self.sigma, self.xi)
        self.proposals["season"] += 1
        if ll_new == -np.inf:
            return

        m_first = self.priors.m_season
        if m_first is None:
            m_first = np.zeros(self.period - 1, float)
        else:
            m_first = np.asarray(m_first, float)
            if m_first.size != self.period - 1:
                raise ValueError("priors.m_season must have length = period-1 (NEWEST-FIRST).")
        s = float(self.priors.s_season)

        lp_old = -0.5 * np.sum(((v_cur[:-1] - m_first) / s) ** 2)
        lp_new = -0.5 * np.sum(((prop[:-1] - m_first) / s) ** 2)
        if self._mh_accept((ll_new + lp_new) - (ll_old + lp_old)):
            self.season_vec = prop
            self.accept["season"] += 1

    # ------------------- Adaptive step-size ------------------ #
    def _adapt_steps(self, it: int) -> None:
        cfg = self.cfg
        if not cfg.adapt_steps:
            return
        in_window = (cfg.adapt_until == "all") or (it < cfg.burn)
        if (it + 1) % max(1, cfg.adapt_every) != 0 or (not in_window):
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

        target = cfg.adapt_target_1d
        for key in keys:
            acc_now = self.accept[key]
            prop_now = self.proposals[key]
            acc_win = acc_now - self._mh_prev_acc[key]
            prop_win = prop_now - self._mh_prev_prop[key]
            if prop_win <= 0:
                continue
            rate = acc_win / max(1, prop_win)
            s = self._get_step(key)
            s_new = s * np.exp(eta * (rate - target))
            self._set_step(key, s_new)
            self._mh_prev_acc[key] = acc_now
            self._mh_prev_prop[key] = prop_now
        self._adapt_round += 1
        
        if self.cfg.progress:
            step_info = ", ".join([f"{key}={self._get_step(key):.2g}" for key in keys])
            print(f"  [adapt] it={it+1}, η={eta:.2g}, steps: {step_info}")

    # ---------------- Conditional SMC (Bootstrap) + ESS-triggered resampling ------- #
    @staticmethod
    def _safe_normalize(p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, float)
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        p[p < 0.0] = 0.0
        s = float(np.sum(p))
        if not np.isfinite(s) or s <= 0.0:
            return np.full_like(p, 1.0 / p.size)
        p /= s
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

    def _conditional_pgas(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
        """
        Bootstrap conditional SMC with ESS-triggered resampling.

        Returns (parts, w, a, logZ, pf_diag):
        - parts[t, n, :] is particle n at time t
        - w[t, n] are normalized weights
        - a[t, n] is ancestor index of particle n at time t
        - logZ is log p(y|theta) estimate
        """
        N, T, D = self.cfg.n_particles, self.T, self.dim
        parts = np.zeros((T + 1, N, D), float) if D > 0 else np.zeros((T + 1, N, 0), float)
        w = np.zeros((T + 1, N), float)
        a = np.zeros((T + 1, N), int)
        logZ = 0.0

        ess_list: List[float] = []
        maxw_list: List[float] = []
        resample_count: int = 0

        if D > 0:
            parts[0, :, :] = self.x[0]  # common x0

        # ----- t = 1
        for n in range(N - 1):
            if D > 0:
                parts[1, n, :] = self._transition_sample(parts[0, n, :], t=1)
            a[1, n] = n

        if D > 0:
            parts[1, N - 1, :] = self.x[1].copy()

        lw = np.zeros(N, float)
        for n in range(N):
            mu = self.mu_from_state(parts[1, n, :] if D > 0 else np.zeros(0), t=0)
            lw[n] = gev_logpdf(self.y[0], mu, self.sigma, self.xi)

        # AS for reference particle at t=1
        if D > 0:
            logf = np.array([self._transition_logpdf(parts[0, j, :], parts[1, N - 1, :], t=1) for j in range(N)], float)
            post = self._safe_normalize(np.exp(logf - np.max(logf)))
            a[1, N - 1] = np.random.choice(N, p=post)
        else:
            a[1, N - 1] = N - 1

        lw_max = np.max(lw)
        logZ += lw_max + math.log(np.mean(np.exp(lw - lw_max)) + 1e-300)
        w[1, :] = self._safe_normalize(np.exp(lw - lw_max))
        ess_list.append(self._ess(w[1, :]))
        maxw_list.append(float(np.max(w[1, :])))

        if self.cfg.progress:
            print(f"  Running conditional PGAS (Bootstrap proposal; ESS threshold={self.cfg.ess_threshold_frac:.2f}, N={N})")

        # ----- t = 2..T
        it = tqdm(range(2, T + 1)) if self.cfg.progress else range(2, T + 1)
        for t in it:
            # Resample decision using previous weights
            res_p = self._safe_normalize(w[t - 1, :])
            ess_prev = self._ess(res_p)
            thresh = self.cfg.ess_threshold_frac * N
            do_resample = (ess_prev < thresh)

            if do_resample:
                resample_count += 1
                anc = np.random.choice(N, size=N - 1, p=res_p, replace=True)
            else:
                anc = np.arange(N - 1, dtype=int)

            # propagate N-1
            for n in range(N - 1):
                a[t, n] = anc[n]
                if D > 0:
                    prev = parts[t - 1, a[t, n], :]
                    parts[t, n, :] = self._transition_sample(prev, t=t)

            # reference path and AS
            if D > 0:
                x_ref_t = self.x[t]
                parts[t, N - 1, :] = x_ref_t.copy()
                logw_prev = np.log(np.clip(w[t - 1, :], 1e-300, None))
                logf = np.array([self._transition_logpdf(parts[t - 1, j, :], x_ref_t, t=t) for j in range(N)], float)
                log_post = (logw_prev + logf)
                post = self._safe_normalize(np.exp(log_post - np.max(log_post)))
                a[t, N - 1] = np.random.choice(N, p=post)
            else:
                a[t, N - 1] = N - 1

            # weights at time t
            y_idx = t - 1
            lw = np.zeros(N, float)
            for n in range(N):
                mu = self.mu_from_state(parts[t, n, :] if D > 0 else np.zeros(0), t=y_idx)
                lw[n] = gev_logpdf(self.y[y_idx], mu, self.sigma, self.xi)

            if do_resample:
                lw_eff = lw
            else:
                log_w_prev = np.log(np.clip(w[t - 1, :], 1e-300, None))
                lw_eff = lw + log_w_prev

            lw_eff_max = np.max(lw_eff)
            logZ += lw_eff_max + math.log(np.mean(np.exp(lw_eff - lw_eff_max)) + 1e-300)
            w[t, :] = self._safe_normalize(np.exp(lw_eff - lw_eff_max))

            ess_t = self._ess(w[t, :])
            maxw_t = float(np.max(w[t, :]))
            ess_list.append(ess_t)
            maxw_list.append(maxw_t)

            if self.cfg.progress and hasattr(it, "set_postfix_str"):
                decision = "resamp" if do_resample else "skip"
                it.set_postfix_str(
                    f"ESS_prev={ess_prev:6.1f} ({ess_prev / N:4.2f}N) -> {decision} | "
                    f"ESS={ess_t:6.1f} MaxW={maxw_t:7.4f}"
                )

        steps_considered = max(1, T - 1)
        pf_diag = {
            "ess_mean": float(np.mean(ess_list)),
            "ess_min": float(np.min(ess_list)),
            "maxw_mean": float(np.mean(maxw_list)),
            "maxw_max": float(np.max(maxw_list)),
            "resample_count": int(resample_count),
            "resample_rate": float(resample_count / steps_considered),
        }
        return parts, w, a, float(logZ), pf_diag

    def _trace_single_trajectory(self, parts: np.ndarray, a: np.ndarray, w: np.ndarray) -> np.ndarray:
        N, T, D = self.cfg.n_particles, self.T, self.dim
        if D == 0:
            return self.x
        idx = np.zeros(T + 1, dtype=int)
        idx[T] = np.random.choice(N, p=w[T, :])
        for t in range(T, 1, -1):
            idx[t - 1] = a[t, idx[t]]
        x_new = self.x.copy()
        x_new[0, :] = parts[0, 0, :] if D > 0 else self.x[0, :]
        for t in range(1, T + 1):
            x_new[t, :] = parts[t, idx[t], :]
        return x_new

    def update_states_pgas(self) -> None:
        parts, w, a, logZ, pf_diag = self._conditional_pgas()
        self.x = self._trace_single_trajectory(parts, a, w)
        self.last_log_evidence = float(logZ)
        self.last_pf_diag = pf_diag

    # ------------------- m0 | P0, x0  &  P0 | m0, x0  (Gibbs) ------------------- #
    @staticmethod
    def _gibbs_m0_scalar(x0: float, m_prior: float, s_prior: float, P0: float) -> float:
        prec = 1.0 / (s_prior**2) + 1.0 / max(1e-18, P0)
        var = 1.0 / prec
        mean = var * (m_prior / (s_prior**2) + x0 / max(1e-18, P0))
        return float(np.random.normal(mean, math.sqrt(var)))

    def update_m0(self) -> None:
        if self.dim == 0:
            return
        pos = 0
        if self.idx_alpha is not None:
            self.m0_alpha = self._gibbs_m0_scalar(
                float(self.x[0, pos]), self.priors.m_m0_alpha, self.priors.s_m0_alpha, self.P0_alpha
            )
            pos += 1
        if self.idx_beta is not None:
            self.m0_beta = self._gibbs_m0_scalar(
                float(self.x[0, pos]), self.priors.m_m0_beta, self.priors.s_m0_beta, self.P0_beta
            )
            pos += 1
        if self.seasonal_mode == "dynamic":
            m_prior = (
                np.zeros(self.period - 1, float)
                if self.priors.m_m0_gamma is None
                else np.asarray(self.priors.m_m0_gamma, float)
            )
            if m_prior.size != self.period - 1:
                raise ValueError("priors.m_m0_gamma must be length p-1 (NEWEST-FIRST)")
            s = float(self.priors.s_m0_gamma)
            for k in range(self.period - 1):
                self.m0_gamma[k] = self._gibbs_m0_scalar(float(self.x[0, pos + k]), float(m_prior[k]), s, self.P0_gamma)

    def update_P0(self) -> None:
        if self.dim == 0:
            return
        pos = 0
        if self.idx_alpha is not None:
            a = self.priors.a_P0_alpha + 0.5
            b = self.priors.b_P0_alpha + 0.5 * (float(self.x[0, pos]) - self.m0_alpha) ** 2
            self.P0_alpha = 1.0 / np.random.gamma(shape=a, scale=1.0 / b)
            pos += 1
        if self.idx_beta is not None:
            a = self.priors.a_P0_beta + 0.5
            b = self.priors.b_P0_beta + 0.5 * (float(self.x[0, pos]) - self.m0_beta) ** 2
            self.P0_beta = 1.0 / np.random.gamma(shape=a, scale=1.0 / b)
            pos += 1
        if self.seasonal_mode == "dynamic":
            diffsq = 0.0
            for k in range(self.period - 1):
                diffsq += (float(self.x[0, pos + k]) - float(self.m0_gamma[k])) ** 2
            a = self.priors.a_P0_gamma + 0.5 * (self.period - 1)
            b = self.priors.b_P0_gamma + 0.5 * diffsq
            self.P0_gamma = 1.0 / np.random.gamma(shape=a, scale=1.0 / b)

    # ---------------- Wrapper for deterministic params ------------------ #
    def update_deterministic_params(self) -> None:
        if self.level_mode == "deterministic":
            self.update_level_value()
        if self.trend_mode == "deterministic":
            self.update_slope()
        if self.seasonal_mode == "deterministic":
            self.update_season_vec()

    # --------------------------- Progress helpers --------------------------- #
    def _fmt_list(self, vals, max_elems: int = 6, fmt: str = ".4g") -> str:
        if vals is None:
            return "-"
        v = np.asarray(vals, float).ravel()
        if v.size == 0:
            return "[]"
        if v.size <= max_elems:
            return "[" + ", ".join(f"{x:{fmt}}" for x in v) + "]"
        head = ", ".join(f"{x:{fmt}}" for x in v[:max_elems])
        return f"[{head}, …]"

    def _progress_line(self, it: int) -> str:
        def pct(key: str) -> str:
            p = self.proposals.get(key, 0)
            a = self.accept.get(key, 0)
            return "0.0%" if p <= 0 else f"{100.0 * a / p:.1f}%"

        parts = [f"[it {it + 1}/{self.cfg.n_iter}]"]
        parts.append(f"logZ={self.last_log_evidence:.3f}")
        parts.append(f"σ={math.exp(self.logsigma):.3f} ({pct('logsigma')})")
        parts.append(f"ξ={self.xi:.3f} ({pct('xi')})")

        # Q block now shows variances s_*^2
        if self.idx_alpha is not None:
            parts.append(f"Qα={self.s_alpha**2:.4g}")
        if self.idx_beta is not None:
            parts.append(f"Qβ={self.s_beta**2:.4g}")
        if self.seasonal_mode == "dynamic":
            parts.append(f"Qγ={self.s_gamma**2:.4g}")

        # m0/P0 or slope
        if self.level_mode == "dynamic":
            if hasattr(self, "m0_alpha"):
                parts.append(f"m0α={self.m0_alpha:.4g} P0α={self.P0_alpha:.4g}")
            else:
                parts.append("m0α=/ P0α=/")
        elif self.level_mode == "deterministic":
            parts.append(f"m0α={self.level_value:.4g} P0α=0 ({pct('level')})")

        if self.trend_mode == "dynamic":
            if hasattr(self, "m0_beta"):
                parts.append(f"m0β={self.m0_beta:.4g} P0β={self.P0_beta:.4g}")
            else:
                parts.append("m0β=/ P0β=/")
        elif self.trend_mode == "deterministic":
            parts.append(f"m0β={self.slope_value:.4g} P0β=0 ({pct('slope')})")

        if self.seasonal_mode != "none":
            if self.seasonal_mode == "dynamic" and hasattr(self, "m0_gamma"):
                head = self.m0_gamma
                head_show = head[:6] if len(head) > 6 else head
                head_str = "[" + ", ".join(f"{float(x):.4g}" for x in head_show) + (", …]" if len(head) > 6 else "]")
                parts.append(f"m0γ={head_str} P0γ={self.P0_gamma:.4g}")
            elif self.seasonal_mode == "deterministic" and self.season_vec is not None:
                head = self.season_vec[:-1]
                head_show = head[:6] if len(head) > 6 else head
                head_str = "[" + ", ".join(f"{float(x):.4g}" for x in head_show) + (", …]" if len(head) > 6 else "]")
                parts.append(f"m0γ={head_str} P0γ=0 ({pct('season')})")

        return " | ".join(parts)

    # --------------------------------- MCMC --------------------------------- #
    def run(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        save_iters = list(range(cfg.burn, cfg.n_iter, cfg.thin))
        n_kept, keep_idx = len(save_iters), 0

        # allocate storage
        self.keep = {"sigma": np.zeros(n_kept, float),
                     "xi": np.zeros(n_kept, float),
                     "mu": np.zeros((n_kept, self.T), float),
                     "log_evidence": np.zeros(n_kept, float)}
        if self.idx_alpha is not None:
            self.keep.update({"Q_alpha": np.zeros(n_kept),
                              "m0_alpha": np.zeros(n_kept), "P0_alpha": np.zeros(n_kept)})
        if self.idx_beta is not None:
            self.keep.update({"Q_beta": np.zeros(n_kept),
                              "m0_beta": np.zeros(n_kept), "P0_beta": np.zeros(n_kept)})
        if self.seasonal_mode == "dynamic":
            self.keep.update({"Q_gamma": np.zeros(n_kept),
                              "m0_gamma": np.zeros((n_kept, self.period - 1)),
                              "P0_gamma": np.zeros(n_kept),
                              "x": np.zeros((n_kept, self.T, self.dim))})
        elif self.dim > 0:
            self.keep["x"] = np.zeros((n_kept, self.T, self.dim))
        if self.level_mode == "deterministic":
            self.keep["level_value"] = np.zeros(n_kept)
        if self.trend_mode == "deterministic":
            self.keep["slope_value"] = np.zeros(n_kept)
        if self.seasonal_mode == "deterministic":
            self.keep["season_vector"] = np.zeros((n_kept, self.period), float)

        print_every = cfg.progress_every if cfg.progress_every > 0 else max(1, cfg.n_iter // 50) or 1

        for it in range(cfg.n_iter):

            if cfg.progress:
                print(f"Iteration {it + 1}/{cfg.n_iter}")

            # 1) States via PGAS (or pure likelihood if dim=0)
            if self.dim > 0:
                self.update_states_pgas()
                current_log_ev = float(self.last_log_evidence)
            else:
                mu_vec_now = self._mu_vec_current()
                current_log_ev = float(gev_loglike_sum(self.y, mu_vec_now, self.sigma, self.xi))

            # 2) Process SDs via slice sampling on log s
            if self.dim > 0:
                self.update_process_s_lognormal_slice()

            # 3) m0 (Gibbs) and 4) P0 (Gibbs Inv-Gamma)
            if self.dim > 0:
                self.update_m0()
                self.update_P0()

            # 5) Deterministic params (MH)
            self.update_deterministic_params()

            # 6) Observation params (MH)
            self.update_logsigma()
            self.update_xi()

            # 7) Adapt RW–MH step sizes (Robbins–Monro)
            self._adapt_steps(it)

            # 8) progress
            if cfg.progress and ((it + 1) % print_every == 0 or it == cfg.n_iter - 1):
                print(self._progress_line(it))

            # 9) save
            if it in save_iters:
                mu = self._mu_vec_current()
                self.keep["mu"][keep_idx, :] = mu
                self.keep["sigma"][keep_idx] = float(np.exp(self.logsigma))
                self.keep["xi"][keep_idx] = float(self.xi)
                self.keep["log_evidence"][keep_idx] = current_log_ev
                if self.idx_alpha is not None:
                    self.keep["Q_alpha"][keep_idx] = self.s_alpha**2
                    self.keep["m0_alpha"][keep_idx] = self.m0_alpha
                    self.keep["P0_alpha"][keep_idx] = self.P0_alpha
                if self.idx_beta is not None:
                    self.keep["Q_beta"][keep_idx] = self.s_beta**2
                    self.keep["m0_beta"][keep_idx] = self.m0_beta
                    self.keep["P0_beta"][keep_idx] = self.P0_beta
                if self.seasonal_mode == "dynamic":
                    self.keep["Q_gamma"][keep_idx] = self.s_gamma**2
                    self.keep["m0_gamma"][keep_idx, :] = self.m0_gamma
                    self.keep["P0_gamma"][keep_idx] = self.P0_gamma
                if "x" in self.keep and self.dim > 0:
                    self.keep["x"][keep_idx, :, :] = self.x[1 : self.T + 1, :]
                if self.level_mode == "deterministic":
                    self.keep["level_value"][keep_idx] = self.level_value
                if self.trend_mode == "deterministic":
                    self.keep["slope_value"][keep_idx] = self.slope_value
                if self.seasonal_mode == "deterministic":
                    self.keep["season_vector"][keep_idx, :] = self.season_vec
                keep_idx += 1

        if cfg.progress:
            print(self._progress_line(cfg.n_iter - 1))
        return self.keep

    # ------------------------------- Persistence ---------------------------- #
    def save_posterior(self, out_npz_path: str, extra_meta: Optional[dict] = None) -> None:
        _ensure_dir(os.path.dirname(out_npz_path))
        arrays = dict(self.keep)
        arrays["y"] = self.y.copy()
        if "x" not in arrays:
            arrays["x"] = np.zeros((0, 0, 0))
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
            "cfg": asdict(self.cfg),
            "priors": asdict(self.priors),
            "lognormal_priors": {
                "alpha": {"mu": self.priors.mu_log_s_alpha, "sd": self.priors.sd_log_s_alpha},
                "beta" : {"mu": self.priors.mu_log_s_beta , "sd": self.priors.sd_log_s_beta },
                "gamma": {"mu": self.priors.mu_log_s_gamma, "sd": self.priors.sd_log_s_gamma},
            },
            "inv_gamma_prior_sigma": {
                "a_sigma": self.priors.a_sigma,
                "b_sigma": self.priors.b_sigma,
            },
            "true_sigma": self.true_sigma,
            "true_xi": self.true_xi,
            "true_Q": (None if self.true_Q is None else np.asarray(self.true_Q, float).tolist()),
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

# ------------------------- CLI / Example run & plots ------------------------
if __name__ == "__main__":
    import sys, argparse, math, time, os
    from datetime import datetime
    import numpy as np
    import matplotlib.pyplot as plt

    # make simulator importable
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from simulator.extremal_time_series import Extremal_Time_Series

    # --- helpers -------------------------------------------------------------
    def _parse_date(s: str | None):
        if not s:
            return datetime.today()
        parts = [int(p) for p in s.split("-")]
        if   len(parts) == 1: return datetime(parts[0], 1, 1)
        elif len(parts) == 2: return datetime(parts[0], parts[1], 1)
        elif len(parts) == 3: return datetime(parts[0], parts[1], parts[2])
        raise ValueError("start-date must be YYYY, YYYY-MM, or YYYY-MM-DD")

    def _csv_floats_or_none(s: str | None):
        if s is None: return None
        s = s.strip()
        if s == "": return None
        return [float(z) for z in s.split(",") if z.strip() != ""]

    # --- CLI ----------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description=(
            "DGEV PGAS Sampler with log-normal priors on process standard deviations "
            "and an inverse-gamma prior on the observation variance σ². "
            "Shape parameter ξ has a bounded uniform prior on [xi_lower, xi_upper]."
        )
    )

    # Simulation controls
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--period", type=int, default=12)
    parser.add_argument("--start-date", type=str, default="2000-01-01")

    parser.add_argument("--level-mode",   choices=["dynamic", "deterministic"],            default="dynamic")
    parser.add_argument("--trend-mode",   choices=["dynamic", "deterministic", "none"],    default="dynamic")
    parser.add_argument("--seasonal-mode", choices=["dynamic", "deterministic", "none"],   default="dynamic")

    # Truth / simulator params
    parser.add_argument("--sigma",     type=float, default=4.0)
    parser.add_argument("--xi",        type=float, default=-0.1)
    parser.add_argument("--q-level",   type=float, default=1e-1)
    parser.add_argument("--q-trend",   type=float, default=1e-3)
    parser.add_argument("--q-season",  type=float, default=5e-2)

    parser.add_argument("--m0-level",  type=float, default=5.0)
    parser.add_argument("--v0-level",  type=float, default=0.2)
    parser.add_argument("--m0-trend",  type=float, default=0.1)
    parser.add_argument("--v0-trend",  type=float, default=0.05)
    parser.add_argument("--m0-season", type=str,   default=None, help="comma-separated (length p-1)")
    parser.add_argument("--v0-season", type=str,   default=None, help="comma-separated (length p-1)")

    # Sampler initialization (m0, P0)
    parser.add_argument("--init-m0-level",  type=float, default=None,
                        help="Sampler x0 prior mean for level (default: use --m0-level).")
    parser.add_argument("--init-p0-level",  type=float, default=None,
                        help="Sampler x0 prior variance for level (default: use --v0-level).")
    parser.add_argument("--init-m0-trend",  type=float, default=None,
                        help="Sampler x0 prior mean for trend (default: use --m0-trend).")
    parser.add_argument("--init-p0-trend",  type=float, default=None,
                        help="Sampler x0 prior variance for trend (default: use --v0-trend).")
    parser.add_argument("--init-m0-season", type=str,   default=None,
                        help="Sampler x0 prior mean for seasonal first p-1 coords (comma-separated). "
                             "Default: use --m0-season or zeros if None.")
    parser.add_argument("--init-p0-season", type=float, default=None,
                        help="Sampler x0 prior VARIANCE for each seasonal coord (scalar). "
                             "Default: mean of --v0-season or 0.5 if None.")

    # Inference priors (observation + deterministic components)
    parser.add_argument("--prior-a-sigma",  type=float, default=2.0,
                        help="Shape parameter a for InvGamma prior on σ².")
    parser.add_argument("--prior-b-sigma",  type=float, default=2.0,
                        help="Scale parameter b for InvGamma prior on σ².")

    # Bounded uniform prior for xi
    parser.add_argument("--prior-xi-lower", type=float, default=-0.5,
                        help="Lower bound for bounded uniform prior on ξ.")
    parser.add_argument("--prior-xi-upper", type=float, default=0.5,
                        help="Upper bound for bounded uniform prior on ξ.")

    parser.add_argument("--prior-m-level",  type=float, default=0.0)
    parser.add_argument("--prior-s-level",  type=float, default=10.0)
    parser.add_argument("--prior-m-slope",  type=float, default=0.0)
    parser.add_argument("--prior-s-slope",  type=float, default=10.0)
    parser.add_argument("--prior-m-season", type=str,   default=None, help="comma-separated (length p-1)")
    parser.add_argument("--prior-s-season", type=float, default=5.0)

    # (Optional) log-normal hyperparameters for process sds ln s_*
    parser.add_argument("--prior-ln-s-alpha-m", type=float, default=-1,
                        help="Mean of ln s_alpha prior (if used)")
    parser.add_argument("--prior-ln-s-alpha-sd", type=float, default=2.0,
                        help="SD of ln s_alpha prior (if used)")
    parser.add_argument("--prior-ln-s-beta-m", type=float, default=-1,
                        help="Mean of ln s_beta prior (if used)")
    parser.add_argument("--prior-ln-s-beta-sd", type=float, default=2.0,
                        help="SD of ln s_beta prior (if used)")
    parser.add_argument("--prior-ln-s-gamma-m", type=float, default=-1,
                        help="Mean of ln s_gamma prior (if used)")
    parser.add_argument("--prior-ln-s-gamma-sd", type=float, default=2.0,
                        help="SD of ln s_gamma prior (if used)")

    # Sampler config
    parser.add_argument("--n-iter", type=int, default=4000)
    parser.add_argument("--burn",   type=int, default=1000)
    parser.add_argument("--thin",   type=int, default=1)

    # RW–MH steps (observation + deterministic params)
    parser.add_argument("--step-logsigma", type=float, default=0.2)
    parser.add_argument("--step-xi",       type=float, default=0.1)
    parser.add_argument("--step-level",    type=float, default=0.2)
    parser.add_argument("--step-slope",    type=float, default=0.001)
    parser.add_argument("--step-season",   type=float, default=0.02)

    # PGAS / PF
    parser.add_argument("--particles",   type=int,   default=500)
    parser.add_argument("--trans-eps",   type=float, default=1e-8)
    parser.add_argument("--ess-frac",    type=float, default=0.5, help="Resample when ESS < ess_frac * N")

    # Adaptation
    parser.add_argument("--adapt-steps",        default=True)
    parser.add_argument("--adapt-every",        type=int,    default=20)
    parser.add_argument("--adapt-until",        choices=["burn","all"], default="burn")
    parser.add_argument("--adapt-eta0",         type=float,  default=0.2)
    parser.add_argument("--adapt-decay",        type=float,  default=0.75)
    parser.add_argument("--adapt-target-1d",    type=float,  default=0.44)
    parser.add_argument("--step-min",           type=float,  default=1e-5)
    parser.add_argument("--step-max",           type=float,  default=1.0)

    # I/O & misc
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--progress-every", type=int, default=1,
                        help="Print compact progress every k iterations (default 1 = every line).")
    parser.add_argument("--out-dir", type=str, default="results/simulations/DGEV")

    parser.add_argument("--plot", default=True)
    parser.add_argument("--print-summary", default=True)
    parser.add_argument("--progress", default=True)

    args = parser.parse_args()
    np.random.seed(args.seed)

    # --- simulate data ------------------------------------------------------
    start_date = _parse_date(args.start_date)
    args.v0_season = _csv_floats_or_none(args.v0_season)
    if args.m0_season is None:
        args.m0_season = [2] * (args.period - 1)  # 2, 2, 2, -6 for p=4
    if args.v0_season is None:
        args.v0_season = [0.01] * (args.period - 1)

    ts = Extremal_Time_Series(
        parameters=(args.sigma, args.xi),
        level_mode=args.level_mode,
        trend_mode=args.trend_mode,
        seasonal_mode=args.seasonal_mode,
        period=args.period,
        q_level=(args.q_level  if args.level_mode  == "dynamic" else 0.0),
        q_trend=(args.q_trend  if args.trend_mode  == "dynamic" else 0.0),
        q_season=(args.q_season if args.seasonal_mode == "dynamic" else 0.0),
        m0_level=args.m0_level, v0_level=args.v0_level,
        m0_trend=(args.m0_trend if args.trend_mode != "none" else 0.0), v0_trend=args.v0_trend,
        m0_season=(args.m0_season if args.seasonal_mode != "none" else None),
        v0_season=(args.v0_season if args.seasonal_mode != "none" else None),
        start_date=start_date,
    )

    y = []
    for _ in range(args.T):
        ts.move()
        y.append(ts.measure())
    y = np.asarray(y, float)

    truths = ts.get_truth_paths(as_numpy=False)
    mu_T    = np.asarray(truths["mu"][1 : 1 + args.T], float)
    alpha_T = (np.asarray(truths["alpha"][1 : 1 + args.T], float) if args.level_mode == "dynamic" else None)
    beta_T  = (np.asarray(truths["beta"][1  : 1 + args.T], float) if args.trend_mode == "dynamic" else None)
    gamma_T = (np.asarray(truths["gamma"][1 : 1 + args.T], float) if args.seasonal_mode == "dynamic" else None)
    dates_T = truths.get("index", np.arange(args.T))

    # --- priors & config ----------------------------------------------------
    pri_season_first = _csv_floats_or_none(args.prior_m_season)
    priors = Priors(
        a_sigma=float(args.prior_a_sigma),
        b_sigma=float(args.prior_b_sigma),
        xi_lower=float(args.prior_xi_lower),
        xi_upper=float(args.prior_xi_upper),
        m_level=float(args.prior_m_level), s_level=float(args.prior_s_level),
        m_slope=float(args.prior_m_slope), s_slope=float(args.prior_s_slope),
        m_season=None if pri_season_first is None else pri_season_first,
        s_season=float(args.prior_s_season),
        # ln-s hyperparameters wired below
    )

    if args.level_mode == "deterministic":
        if args.init_m0_level is None:
            # robust init for level (median)
            args.init_m0_level = float(np.median(y))
            # also center the prior on the level at the same place (helps mixing)
            priors.m_level = args.init_m0_level
            priors.s_level = max(2.0, 0.5 * y.std(ddof=1))  # not too tight, but informative

    # (optional) override process ln-s priors from CLI if desired
    priors.mu_log_s_alpha = float(args.prior_ln_s_alpha_m)
    priors.sd_log_s_alpha = float(args.prior_ln_s_alpha_sd)
    priors.mu_log_s_beta  = float(args.prior_ln_s_beta_m)
    priors.sd_log_s_beta  = float(args.prior_ln_s_beta_sd)
    priors.mu_log_s_gamma = float(args.prior_ln_s_gamma_m)
    priors.sd_log_s_gamma = float(args.prior_ln_s_gamma_sd)

    cfg = SamplerConfig(
        n_iter=int(args.n_iter),
        burn=int(args.burn),
        thin=int(args.thin),

        # RW–MH steps (obs + deterministic)
        step_logsigma=float(args.step_logsigma),
        step_xi=float(args.step_xi),
        step_level=float(args.step_level),
        step_slope=float(args.step_slope),
        step_season=float(args.step_season),

        # PF / PGAS
        n_particles=int(args.particles),
        trans_eps=float(args.trans_eps),
        random_seed=int(args.seed),
        progress=bool(args.progress),
        progress_every=int(args.progress_every),
        ess_threshold_frac=float(args.ess_frac),

        # Adaptation
        adapt_steps=bool(args.adapt_steps),
        adapt_every=int(args.adapt_every),
        adapt_until=str(args.adapt_until),
        adapt_eta0=float(args.adapt_eta0),
        adapt_eta_decay=float(args.adapt_decay),
        adapt_target_1d=float(args.adapt_target_1d),
        step_min=float(args.step_min),
        step_max=float(args.step_max),
    )

    # deterministic-seasonal initializer (first p-1; last implied)
    seasonal_init_pminus1 = (
        np.asarray(pri_season_first, float) if (args.seasonal_mode == "deterministic" and pri_season_first is not None)
        else (build_seasonal(args.period) if args.seasonal_mode == "deterministic" else None)
    )

    # --- sampler x0 prior (init), decoupled from simulator truth ------------
    init_m0_level = args.init_m0_level if args.init_m0_level is not None else args.m0_level
    init_p0_level = args.init_p0_level if args.init_p0_level is not None else args.v0_level

    init_m0_trend = (args.init_m0_trend if args.init_m0_trend is not None
                     else (args.m0_trend if args.trend_mode != "none" else 0.0))
    init_p0_trend = args.init_p0_trend if args.init_p0_trend is not None else args.v0_trend

    init_m0_season = _csv_floats_or_none(args.init_m0_season)
    if init_m0_season is None:
        init_m0_season = (args.m0_season if args.seasonal_mode == "dynamic" else None)

    if args.init_p0_season is not None:
        init_p0_season_scalar = float(args.init_p0_season)
    else:
        if isinstance(args.v0_season, (list, tuple, np.ndarray)) and len(args.v0_season) > 0:
            init_p0_season_scalar = float(np.mean(args.v0_season))
        else:
            init_p0_season_scalar = 0.5

    # --- build sampler ------------------------------------------------------
    sampler = DGEVParticleGibbs(
        y=y,
        period=int(args.period),
        level_mode=args.level_mode,
        trend_mode=args.trend_mode,
        seasonal_mode=args.seasonal_mode,

        # x0 prior ...
        m0_level_init=float(init_m0_level),
        P0_level_init=float(init_p0_level),
        m0_trend_init=float(init_m0_trend),
        P0_trend_init=float(init_p0_trend),
        m0_season_init=(init_m0_season if args.seasonal_mode == "dynamic" else None),
        P0_season_init=(init_p0_season_scalar if args.seasonal_mode == "dynamic" else 0.0),
        s_alpha_init=1e-1,
        s_beta_init=1e-2,
        s_gamma_init=1e-1,

        # deterministic level warm start
        level_value_init=(float(args.init_m0_level) if args.level_mode == "deterministic" else 0.0),

        # priors & config
        priors=priors,
        cfg=cfg,

        # deterministic seasonal initializer
        seasonal_vector_init=seasonal_init_pminus1,
        # observation initial values
        sigma_init=args.sigma,
        xi_init=args.xi,
    )

    # truths for diagnostics/saving
    true_Q = []
    if args.level_mode   == "dynamic": true_Q.append(args.q_level)
    if args.trend_mode   == "dynamic": true_Q.append(args.q_trend)
    if args.seasonal_mode == "dynamic": true_Q += [args.q_season] + [0.0] * (args.period - 2)
    sampler.set_truth(sigma=args.sigma, xi=args.xi,
                      Q=(np.asarray(true_Q, float) if len(true_Q) else None))
    sampler.set_truth_paths(mu=mu_T, alpha=alpha_T, beta=beta_T, gamma=gamma_T)

    # --- summaries (show sim truth vs sampler init) -------------------------
    if args.print_summary:
        with np.printoptions(suppress=True, precision=4):
            print("\n--- Summary (simulation truth) ---")
            print(f"level={args.level_mode}, trend={args.trend_mode}, season={args.seasonal_mode}")
            print(f"sigma={args.sigma}, xi={args.xi}, "
                  f"q_level={args.q_level if args.level_mode=='dynamic' else 0.0}, "
                  f"q_trend={args.q_trend if args.trend_mode=='dynamic' else 0.0}, "
                  f"q_season={args.q_season if args.seasonal_mode=='dynamic' else 0.0}")
            print(f"m0_level={args.m0_level}, v0_level={args.v0_level}")
            print(f"m0_trend={args.m0_trend}, v0_trend={args.v0_trend}")
            if args.seasonal_mode == "none":
                print(f"m0_season={[0.0]*(args.period-1)} (none), v0_season=0.0 (none)")
            else:
                print(f"m0_season(sim)={np.array(args.m0_season)}")
                print(f"v0_season(sim)={np.array(args.v0_season)}")
            print("\n--- Sampler x0 prior (init) ---")
            print(f"m0_level_init={init_m0_level}, P0_level_init={init_p0_level}")
            print(f"m0_trend_init={init_m0_trend}, P0_trend_init={init_p0_trend}")
            if args.seasonal_mode == "dynamic":
                print(f"m0_season_init={np.array(init_m0_season)}")
                print(f"P0_season_init={init_p0_season_scalar}")
            print(f"period={args.period}, start={dates_T[0]}, end={dates_T[-1]}")
            print(f"y mean={y.mean():.3f}, sd={y.std(ddof=1):.3f}")
            print(f"InvGamma prior on σ²: a={priors.a_sigma:.3g}, b={priors.b_sigma:.3g}")
            print(f"ξ ~ Uniform[{priors.xi_lower}, {priors.xi_upper}]")

    # --- run sampler --------------------------------------------------------
    t0 = time.time()
    post = sampler.run()
    elapsed = time.time() - t0
    print(f"Run time: {elapsed:.2f}s")

    # --- save ---------------------------------------------------------------
    out_dir = os.path.join(
        args.out_dir,
        f"{args.level_mode}-{args.trend_mode}-{args.seasonal_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(out_dir, exist_ok=True)
    sampler.save_posterior(
        out_npz_path=os.path.join(out_dir, "posterior.npz"),
        extra_meta={"elapsed_seconds": float(elapsed)}
    )

    # --- posterior summaries ------------------------------------------------
    if args.print_summary:
        def _fmt_vec(v: np.ndarray, k: int = 6) -> str:
            v = np.asarray(v, float).ravel()
            if v.size == 0:
                return "[]"
            if v.size <= k:
                return "[" + ", ".join(f"{x:.4g}" for x in v) + "]"
            return "[" + ", ".join(f"{x:.4g}" for x in v[:k]) + ", …]"

        with np.printoptions(suppress=True, precision=4):
            print("\n--- Summary (posterior means) ---")
            # Observation params
            sig_mean = float(np.mean(post["sigma"])) if "sigma" in post else float("nan")
            xi_mean  = float(np.mean(post["xi"]))    if "xi"    in post else float("nan")
            print(f"σ: {sig_mean:.4g}")
            print(f"ξ: {xi_mean:.4g}  (prior Uniform[{priors.xi_lower}, {priors.xi_upper}])")

            # Process variances (Q)
            if sampler.idx_alpha is not None and "Q_alpha" in post:
                m = float(np.mean(post["Q_alpha"]))
                print(f"Q_alpha: {m:.4g}  (√Q_alpha ≈ {math.sqrt(max(m,0.0)):.4g})")
            else:
                print("Q_alpha: n/a (level deterministic)")
            if sampler.idx_beta is not None and "Q_beta" in post:
                m = float(np.mean(post["Q_beta"]))
                print(f"Q_beta:  {m:.4g}  (√Q_beta  ≈ {math.sqrt(max(m,0.0)):.4g})")
            else:
                print("Q_beta: n/a (trend deterministic/none)")
            if args.seasonal_mode == "dynamic" and "Q_gamma" in post:
                m = float(np.mean(post["Q_gamma"]))
                print(f"Q_gamma: {m:.4g}  (√Q_gamma ≈ {math.sqrt(max(m,0.0)):.4g})")
            else:
                print("Q_gamma: n/a (season deterministic/none)")

            # Evidence (optional)
            if "log_evidence" in post and post["log_evidence"].size > 0:
                le = post["log_evidence"]
                print(f"log p(y|θ): mean={np.nanmean(le):.3f}, median={np.nanmedian(le):.3f}, best={np.nanmax(le):.3f}")

            # ---- m0 / P0 block ----
            print("\n--- m0 / P0 (posterior means) ---")

            # Level
            if args.level_mode == "dynamic":
                m0a = float(np.mean(post["m0_alpha"])) if "m0_alpha" in post else float("nan")
                P0a = float(np.mean(post["P0_alpha"])) if "P0_alpha" in post else float("nan")
                print(f"m0α: {m0a:.4g} | P0α: {P0a:.4g}")
            else:
                lvl_samples = post.get("level_value", None)
                m0a_det = (float(np.mean(lvl_samples)) if isinstance(lvl_samples, np.ndarray) and lvl_samples.size
                           else float(sampler.level_value))
                print(f"m0α (deterministic level): {m0a_det:.4g} | P0α: 0")

            # Trend
            if args.trend_mode == "dynamic":
                m0b = float(np.mean(post["m0_beta"])) if "m0_beta" in post else float("nan")
                P0b = float(np.mean(post["P0_beta"])) if "P0_beta" in post else float("nan")
                print(f"m0β: {m0b:.4g} | P0β: {P0b:.4g}")
            elif args.trend_mode == "deterministic":
                slp_samples = post.get("slope_value", None)
                m0b_det = (float(np.mean(slp_samples)) if isinstance(slp_samples, np.ndarray) and slp_samples.size
                           else float(sampler.slope_value))
                print(f"m0β (deterministic slope): {m0b_det:.4g} | P0β: 0")
            else:
                print("m0β: n/a (trend none) | P0β: n/a")

            # Seasonal
            if args.seasonal_mode == "dynamic":
                if "m0_gamma" in post and post["m0_gamma"].size:
                    m0g_vec = np.mean(post["m0_gamma"], axis=0)  # first p-1 entries
                    P0g = float(np.mean(post["P0_gamma"])) if "P0_gamma" in post else float("nan")
                    print(f"m0γ (first p−1): {_fmt_vec(m0g_vec)} | P0γ: {P0g:.4g}")
                else:
                    print("m0γ: n/a | P0γ: n/a")
            elif args.seasonal_mode == "deterministic":
                sv_samples = post.get("season_vector", None)  # shape [n_kept, period]
                if isinstance(sv_samples, np.ndarray) and sv_samples.ndim == 2 and sv_samples.size:
                    sv_mean = sv_samples.mean(axis=0)
                else:
                    sv_mean = np.asarray(sampler.season_vec, float) if sampler.season_vec is not None else np.zeros(args.period)
                print(f"season vector (p): {_fmt_vec(sv_mean)}")
                print(f"m0γ (first p−1): {_fmt_vec(sv_mean[:-1])} | P0γ: 0")
            else:
                print("m0γ: n/a (season none) | P0γ: n/a")
