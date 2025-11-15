# %% optimization/dgev_pgas_ln.py
from __future__ import annotations

import os, sys, math, json, time
from dataclasses import dataclass, asdict
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
    if s is None: return None
    s = s.strip()
    if not s: return None
    return [float(tok) for tok in s.split(",")]

def _mad(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    if v.size == 0: return 0.0
    med = np.median(v)
    return float(np.median(np.abs(v - med)))

def _robust_sd(v: np.ndarray) -> float:
    return _mad(np.asarray(v, float)) / 1.4826 if v.size else 0.0

# =============================================================================
# Univariate slice sampler (stepping-out + shrinkage) on R
# =============================================================================

class Slice1D:
    def __init__(self, h, w: float = 1.0, m: int = 20, rng: Optional[np.random.Generator] = None):
        self.h = h; self.w = float(w); self.m = int(m)
        self.rng = rng if rng is not None else np.random.default_rng()

    def sample(self, z0: float, n: int = 1) -> np.ndarray:
        out = np.empty(n, float)
        z = float(z0)
        for i in range(n):
            hz = self.h(z)
            y = hz + math.log(self.rng.random())
            w = self.w
            L = z - w * self.rng.random()
            R = L + w
            J = int(self.rng.integers(0, self.m))
            K = self.m - 1 - J
            while J > 0 and self.h(L) > y:
                L -= w; J -= 1
            while K > 0 and self.h(R) > y:
                R += w; K -= 1
            while True:
                z_new = self.rng.uniform(L, R)
                if self.h(z_new) >= y:
                    z = z_new; break
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
    if sigma <= 0.0 or np.isnan(mu): return -np.inf
    z = (y - mu) / sigma
    u = 1.0 + xi * z
    if u <= 0.0: return -np.inf
    if abs(xi) < 1e-8:
        return -np.log(sigma) - np.exp(-z) - z
    return -np.log(sigma) - (1.0 + 1.0 / xi) * np.log(u) - u ** (-1.0 / xi)

def gev_loglike_sum(y: np.ndarray, mu_vec: np.ndarray, sigma: float, xi: float) -> float:
    if sigma <= 0.0 or np.any(np.isnan(mu_vec)): return -np.inf
    z = (y - mu_vec) / sigma
    u = 1.0 + xi * z
    if np.any(u <= 0.0): return -np.inf
    if abs(xi) < 1e-8:
        return float(np.sum(-np.log(sigma) - np.exp(-z) - z))
    return float(np.sum(-np.log(sigma) - (1.0 + 1.0 / xi) * np.log(u) - u ** (-1.0 / xi)))

# =============================================================================
# Priors & Config
#   • Log-Normal on process SDs (slice on ln s)
#   • Uniform prior on ξ in [xi_lower, xi_upper]
# =============================================================================

@dataclass
class Priors:
    # Observation priors
    m_sigma: float = 0.0
    s_sigma: float = 10.0

    # Uniform prior for xi on [xi_lower, xi_upper]
    xi_lower: float = -0.5
    xi_upper: float = 0.5

    # m0 priors (dynamic x0 means); also used when block is deterministic
    m_m0_alpha: float = 0.0
    s_m0_alpha: float = 10.0
    m_m0_beta: float = 0.0
    s_m0_beta: float = 10.0
    m_m0_gamma: Optional[Sequence[float]] = None  # length p-1 (NEWEST-FIRST)
    s_m0_gamma: float = 5.0

    # Initial-state variances P0 ~ InvGamma(a,b)
    a_P0_alpha: float = 2.0
    b_P0_alpha: float = 1.0
    a_P0_beta:  float = 2.0
    b_P0_beta:  float = 1.0
    a_P0_gamma: float = 2.0
    b_P0_gamma: float = 1.0

    # Deterministic components’ priors
    m_level: float = 0.0
    s_level: float = 10.0
    m_slope: float = 0.0
    s_slope: float = 10.0
    m_season: Optional[Sequence[float]] = None
    s_season: float = 5.0

    # Log-Normal priors for process SDs
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
# PGAS in disturbance-space (NCP): particles are innovations w_t, not states x_t
#
# Layout:
#   - Dynamic coords: [alpha] [beta] [g1 ... g_{p-1}]
#   - Seasonal vector is NEWEST-FIRST; obs loads FIRST seasonal coord.
# Dynamics (deterministic map + disturbances):
#   alpha_t   = alpha_{t-1} + drift_t + w_alpha_t,     w_alpha_t ~ N(0, s_alpha^2) if dynamic
#   beta_t    = beta_{t-1}              + w_beta_t,    w_beta_t  ~ N(0, s_beta^2)  if dynamic
#   g1_t      = -sum(g_{1..p-1,t-1})    + w_gamma_t,   w_gamma_t ~ N(0, s_gamma^2) if dynamic (first only)
#   gk_t      = g_{k-1,t-1}, k=2..p-1 (no disturbance)
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

        # Priors / cfg / rng
        self.priors, self.cfg = priors, cfg
        if cfg.random_seed is not None:
            np.random.seed(cfg.random_seed)
            self._rng = np.random.default_rng(cfg.random_seed)
        else:
            self._rng = np.random.default_rng()

        # Dynamic layout
        layout: List[str] = []
        if self.level_mode == "dynamic":  layout.append("alpha")
        if self.trend_mode == "dynamic":  layout.append("beta")
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

        # Deterministic parameters
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

        # Observation params (GEV)
        self.logsigma = float(np.log(max(1e-12, sigma_init)))
        self.sigma    = float(np.exp(self.logsigma))
        self.xi       = float(xi_init)

        # Ensure initial xi lies in uniform prior support
        lo, hi = float(self.priors.xi_lower), float(self.priors.xi_upper)
        if lo >= hi:
            raise ValueError("priors.xi_lower must be < priors.xi_upper")
        if not (lo <= self.xi <= hi):
            width = hi - lo
            eps = 1e-3 * width
            self.xi = float(np.clip(self.xi, lo + eps, hi - eps))

        # Process SDs (log-normal priors)
        self.s_alpha = float(max(1e-12, s_alpha_init)) if self.idx_alpha is not None else 0.0
        self.s_beta  = float(max(1e-12, s_beta_init))  if self.idx_beta  is not None else 0.0
        self.s_gamma = float(max(1e-12, s_gamma_init)) if self.seasonal_mode == "dynamic" else 0.0

        # Initial m0 and P0 for dynamic coords
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

        # Latents:
        #   x[0:T] (states; we keep x[0] too)  — maintained deterministically from innovations
        #   w[1:T] (innovations, only on active coords alpha/beta/g1)
        self.x = np.zeros((self.T + 1, self.dim), float)
        if self.dim > 0:
            m0_vec, P0_diag = self._current_m0_P0()
            self.x[0] = np.random.multivariate_normal(
                m0_vec,
                np.diag(P0_diag) + 1e-10 * np.eye(self.dim)
            )
        self.w = np.zeros((self.T + 1, self.dim), float)  # only some entries non-zero

        # initialize path with tiny innovations
        if self.dim > 0:
            self._propagate_from_innovations()

        # Storage
        self.keep: Dict[str, np.ndarray] = {}

        # MH bookkeeping
        self.accept = {"logsigma": 0, "xi": 0, "level": 0, "slope": 0, "season": 0}
        self.proposals = dict(self.accept)

        # Adapt bookkeeping
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

        # Slice samplers for z = log s (disturbance SS is used)
        self._slice_alpha = Slice1D(
            lambda z: self._h_lognorm(z, *self._ln_params_alpha()),
            w=self.cfg.slice_w, m=self.cfg.slice_m, rng=self._rng
        ) if self.idx_alpha is not None else None
        self._slice_beta  = Slice1D(
            lambda z: self._h_lognorm(z, *self._ln_params_beta()),
            w=self.cfg.slice_w, m=self.cfg.slice_m, rng=self._rng
        ) if self.idx_beta  is not None else None
        self._slice_gamma = Slice1D(
            lambda z: self._h_lognorm(z, *self._ln_params_gamma()),
            w=self.cfg.slice_w, m=self.cfg.slice_m, rng=self._rng
        ) if self.seasonal_mode == "dynamic" else None

    # --------------------- Truth registration (optional) --------------------- #
    def set_truth(self, sigma: Optional[float] = None, xi: Optional[float] = None,
                  Q: Optional[np.ndarray] = None) -> None:
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
        if self.dim == 0:
            base = (self.level_value + self.slope_value * np.arange(self.T)) \
                if self.trend_mode == "deterministic" else self.level_value
            seas = (self.season_vec[np.arange(self.T) % self.period]
                    if self.seasonal_mode == "deterministic" else 0.0)
            return np.asarray(base + seas, float)
        return np.array(
            [self.mu_from_state(self.x[t], t - 1) for t in range(1, self.T + 1)],
            float
        )

    def _current_m0_P0(self) -> Tuple[np.ndarray, np.ndarray]:
        m0, P0 = [], []
        if self.idx_alpha is not None:
            m0.append(self.m0_alpha); P0.append(self.P0_alpha)
        if self.idx_beta  is not None:
            m0.append(self.m0_beta ); P0.append(self.P0_beta )
        if self.seasonal_mode == "dynamic":
            m0.extend(list(self.m0_gamma))
            P0.extend([self.P0_gamma] * (self.period - 1))
        return np.asarray(m0, float), np.asarray(P0, float)

    # ----------------------------- Deterministic map ----------------------------- #
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
                m[gs + 1: ge + 1] = x_prev[gs: ge]           # shift down
                prev = x_prev[gs: ge + 1]
                m[gs] = -float(np.sum(prev))                  # closure
        return m

    def _apply_innovation(self, mean: np.ndarray, w_vec: np.ndarray) -> np.ndarray:
        """Add disturbance only where active: alpha, beta, g1; others are deterministic."""
        x_new = mean.copy()
        if self.idx_alpha is not None:
            x_new[self.idx_alpha] += w_vec[self.idx_alpha]
        if self.idx_beta is not None:
            x_new[self.idx_beta]  += w_vec[self.idx_beta]
        if self.seasonal_mode == "dynamic":
            x_new[self.idx_g_start] += w_vec[self.idx_g_start]  # first seasonal coord only
        return x_new

    def _propagate_from_innovations(self) -> None:
        """Given x[0] and w[1:T], build x[1:T] deterministically."""
        if self.dim == 0: return
        for t in range(1, self.T + 1):
            mean = self._state_mean(self.x[t - 1], t)
            self.x[t] = self._apply_innovation(mean, self.w[t])

    # ------------------ Observation composition ------------------ #
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

    # --------------------- Disturbance SS (for ln s slices) --------------------- #
    def _disturbance_ss_alpha(self) -> Tuple[float, int]:
        if self.idx_alpha is None: return 0.0, 0
        w = self.w[1:, self.idx_alpha]
        return float(np.dot(w, w)), self.T

    def _disturbance_ss_beta(self) -> Tuple[float, int]:
        if self.idx_beta is None: return 0.0, 0
        w = self.w[1:, self.idx_beta]
        return float(np.dot(w, w)), self.T

    def _disturbance_ss_gamma(self) -> Tuple[float, int]:
        if self.seasonal_mode != "dynamic": return 0.0, 0
        w = self.w[1:, self.idx_g_start]   # only first seasonal coord has noise
        return float(np.dot(w, w)), self.T

    # h(z) = -T z - 0.5 * SS * exp(-2z) - (z - mu)^2 / (2 v) + const
    def _h_lognorm(self, z: float, SS: float, T_eff: int, mu: float, var: float) -> float:
        SS = max(float(SS), 1e-300)
        T_eff = max(int(T_eff), 0)
        var = max(float(var), 1e-16)
        return -(T_eff) * z - 0.5 * SS * math.exp(-2.0 * z) - 0.5 * ((z - mu) ** 2) / var

    def _ln_params_alpha(self) -> Tuple[float, int, float, float]:
        SS, T_eff = self._disturbance_ss_alpha()
        mu0 = float(self.priors.mu_log_s_alpha)
        v0  = float(self.priors.sd_log_s_alpha) ** 2
        return SS, T_eff, mu0, v0

    def _ln_params_beta(self) -> Tuple[float, int, float, float]:
        SS, T_eff = self._disturbance_ss_beta()
        mu0 = float(self.priors.mu_log_s_beta)
        v0  = float(self.priors.sd_log_s_beta) ** 2
        return SS, T_eff, mu0, v0

    def _ln_params_gamma(self) -> Tuple[float, int, float, float]:
        SS, T_eff = self._disturbance_ss_gamma()
        mu0 = float(self.priors.mu_log_s_gamma)
        v0  = float(self.priors.sd_log_s_gamma) ** 2
        return SS, T_eff, mu0, v0

    def update_process_s_lognormal_slice(self) -> None:
        if self.idx_alpha is not None:
            z0 = math.log(max(self.s_alpha, 1e-12))
            z  = float(self._slice_alpha.sample(z0, 1)[0])
            self.s_alpha = float(np.exp(z))
        if self.idx_beta is not None:
            z0 = math.log(max(self.s_beta, 1e-12))
            z  = float(self._slice_beta.sample(z0, 1)[0])
            self.s_beta = float(np.exp(z))
        if self.seasonal_mode == "dynamic":
            z0 = math.log(max(self.s_gamma, 1e-12))
            z  = float(self._slice_gamma.sample(z0, 1)[0])
            self.s_gamma = float(np.exp(z))

    # ----------------------- Observation parameter updates ------------------ #
    def _mh_accept(self, logacc: float) -> bool:
        return (np.log(np.random.rand()) < min(0.0, logacc))

    def _get_step(self, key: str) -> float:
        return {"logsigma": self.cfg.step_logsigma, "xi": self.cfg.step_xi,
                "level": self.cfg.step_level, "slope": self.cfg.step_slope,
                "season": self.cfg.step_season}[key]

    def _set_step(self, key: str, val: float) -> None:
        v = float(np.clip(val, self.cfg.step_min, self.cfg.step_max))
        if key == "logsigma": self.cfg.step_logsigma = v
        elif key == "xi":     self.cfg.step_xi = v
        elif key == "level":  self.cfg.step_level = v
        elif key == "slope":  self.cfg.step_slope = v
        elif key == "season": self.cfg.step_season = v
        else: raise KeyError(key)

    def update_logsigma(self) -> None:
        step = self.cfg.step_logsigma
        cur = self.logsigma
        prop = cur + np.random.normal(0.0, step)
        sigma_cur, sigma_prop = float(np.exp(cur)), float(np.exp(prop))
        mu_vec = self._mu_vec_current()
        ll_old = gev_loglike_sum(self.y, mu_vec, sigma_cur, self.xi)
        ll_new = gev_loglike_sum(self.y, mu_vec, sigma_prop, self.xi)
        self.proposals["logsigma"] += 1
        if ll_new == -np.inf: return
        lp_old = -0.5 * ((cur - self.priors.m_sigma) ** 2) / (self.priors.s_sigma ** 2)
        lp_new = -0.5 * ((prop - self.priors.m_sigma) ** 2) / (self.priors.s_sigma ** 2)
        if self._mh_accept((ll_new + lp_new) - (ll_old + lp_old)):
            self.logsigma = prop
            self.sigma = sigma_prop
            self.accept["logsigma"] += 1

    # ---------------- Uniform prior for xi & MH update ---------------- #
    def update_xi(self) -> None:
        step = self.cfg.step_xi
        cur = self.xi
        prop = cur + np.random.normal(0.0, step)
        self.proposals["xi"] += 1

        lo = float(self.priors.xi_lower)
        hi = float(self.priors.xi_upper)

        # Uniform prior support: reject immediately if outside [lo, hi]
        if (prop < lo) or (prop > hi):
            return

        mu_vec = self._mu_vec_current()
        ll_old = gev_loglike_sum(self.y, mu_vec, self.sigma, cur)
        ll_new = gev_loglike_sum(self.y, mu_vec, self.sigma, prop)

        # Invalid likelihood (support violation, etc.) ⇒ reject
        if ll_new == -np.inf:
            return

        # Prior is constant on [lo, hi], so log prior cancels in MH ratio
        if self._mh_accept(ll_new - ll_old):
            self.xi = prop
            self.accept["xi"] += 1

    # ---------------- Deterministic structural parameter updates ------------- #
    def update_level_value(self) -> None:
        if self.level_mode != "deterministic": return
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
        if ll_new == -np.inf: return
        lp_old = -0.5 * ((cur - self.priors.m_level) ** 2) / (self.priors.s_level ** 2)
        lp_new = -0.5 * ((prop - self.priors.m_level) ** 2) / (self.priors.s_level ** 2)
        if self._mh_accept((ll_new + lp_new) - (ll_old + lp_old)):
            self.level_value = prop
            self.accept["level"] += 1

    def update_slope(self) -> None:
        if self.trend_mode != "deterministic": return
        step = self.cfg.step_slope
        cur = self.slope_value
        prop = cur + np.random.normal(0.0, step)

        # Obs contribution if level is deterministic (otherwise it's absorbed into alpha dynamics)
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
            ll_obs_old = 0.0; ll_obs_new = 0.0

        # Transition part is independent of slope in NCP
        lp_old = -0.5 * ((cur - self.priors.m_slope) ** 2) / (self.priors.s_slope ** 2)
        lp_new = -0.5 * ((prop - self.priors.m_slope) ** 2) / (self.priors.s_slope ** 2)
        logacc = (ll_obs_new + lp_new) - (ll_obs_old + lp_old)
        self.proposals["slope"] += 1
        if self._mh_accept(logacc):
            self.slope_value = prop
            self.accept["slope"] += 1

    def update_season_vec(self) -> None:
        if self.seasonal_mode != "deterministic": return
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
        if ll_new == -np.inf: return

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
        if not cfg.adapt_steps: return
        in_window = (cfg.adapt_until == "all") or (it < cfg.burn)
        if (it + 1) % max(1, cfg.adapt_every) != 0 or (not in_window): return
        k = self._adapt_round
        eta = cfg.adapt_eta0 / ((1.0 + k) ** cfg.adapt_eta_decay)

        keys: List[str] = ["logsigma", "xi"]
        if self.level_mode == "deterministic":   keys.append("level")
        if self.trend_mode == "deterministic":   keys.append("slope")
        if self.seasonal_mode == "deterministic": keys.append("season")

        target = cfg.adapt_target_1d
        for key in keys:
            acc_now = self.accept[key]; prop_now = self.proposals[key]
            acc_win = acc_now - self._mh_prev_acc[key]
            prop_win = prop_now - self._mh_prev_prop[key]
            if prop_win <= 0: continue
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

    # ---------------- Conditional SMC in disturbance space (prior proposal) ------- #
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

    def _draw_w_prior(self, size_vec: np.ndarray) -> np.ndarray:
        """Draw innovation vector ~ N(0, diag(size_vec^2)) for active coords; zeros elsewhere."""
        d = size_vec.size
        z = np.random.normal(0.0, 1.0, size=d)
        return z * size_vec

    def _diag_s_vector(self) -> np.ndarray:
        """Return a vector of per-dimension innovation SDs (zeros on deterministic coords)."""
        v = np.zeros(self.dim, float)
        if self.idx_alpha is not None: v[self.idx_alpha] = self.s_alpha
        if self.idx_beta  is not None: v[self.idx_beta]  = self.s_beta
        if self.seasonal_mode == "dynamic": v[self.idx_g_start] = self.s_gamma
        return v

    def _compute_reference_w_from_x(self) -> np.ndarray:
        """Given the current x path, compute its implied disturbances w (alpha, beta, g1)."""
        w = np.zeros_like(self.x)
        if self.dim == 0: return w
        for t in range(1, self.T + 1):
            mean = self._state_mean(self.x[t - 1], t)
            diff = self.x[t] - mean
            if self.idx_alpha is not None:
                w[t, self.idx_alpha] = diff[self.idx_alpha]
            if self.idx_beta is not None:
                w[t, self.idx_beta] = diff[self.idx_beta]
            if self.seasonal_mode == "dynamic":
                w[t, self.idx_g_start] = diff[self.idx_g_start]
        return w

    def _conditional_pgas(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
        """
        Disturbance-space conditional SMC with ancestor sampling.
        Particles live in w-space; proposal is the PRIOR => weights are GEV likelihood only.

        Returns (parts_w, wnorm, a, logZ, pf_diag):
          - parts_w[t, n, :] innovation at time t for particle n (zeros on deterministic coords)
          - wnorm[t, n] normalized weights at time t
          - a[t, n] ancestor index
          - logZ log-evidence estimate
        """
        N, T, D = self.cfg.n_particles, self.T, self.dim
        if D == 0:
            # no dynamics in state; pure deterministic mean block
            mu_vec = self._mu_vec_current()
            logZ = float(gev_loglike_sum(self.y, mu_vec, self.sigma, self.xi))
            pf_diag = {"ess_mean": float(N), "ess_min": float(N),
                       "maxw_mean": 1.0, "maxw_max": 1.0,
                       "resample_count": 0, "resample_rate": 0.0}
            parts_w = np.zeros((T + 1, N, 0), float)
            wnorm   = np.ones((T + 1, N), float) / N
            a       = np.zeros((T + 1, N), int)
            return parts_w, wnorm, a, logZ, pf_diag

        # SD vector per dimension
        svec = self._diag_s_vector()

        # buffers
        parts_w = np.zeros((T + 1, N, D), float)
        parts_x = np.zeros((T + 1, N, D), float)
        wnorm   = np.zeros((T + 1, N), float)
        aidx    = np.zeros((T + 1, N), int)
        logZ    = 0.0

        # x0 common to all particles
        parts_x[0, :, :] = self.x[0]

        # reference disturbances implied by current path
        w_ref = self._compute_reference_w_from_x()

        # --- t = 1
        for n in range(N - 1):
            w_draw = self._draw_w_prior(svec)
            parts_w[1, n, :] = w_draw
            mean = self._state_mean(parts_x[0, n, :], t=1)
            parts_x[1, n, :] = self._apply_innovation(mean, w_draw)
            aidx[1, n] = n

        # reference particle n = N-1
        parts_w[1, N - 1, :] = w_ref[1, :]
        mean = self._state_mean(parts_x[0, N - 1, :], t=1)
        parts_x[1, N - 1, :] = self._apply_innovation(mean, parts_w[1, N - 1, :])

        # weights at t=1 (likelihood only)
        lw = np.zeros(N, float)
        for n in range(N):
            mu = self.mu_from_state(parts_x[1, n, :], t=0)
            lw[n] = gev_logpdf(self.y[0], mu, self.sigma, self.xi)

        # ancestor for ref particle at t=1
        aidx[1, N - 1] = np.random.choice(N)

        lw_max = np.max(lw)
        logZ += lw_max + math.log(np.mean(np.exp(lw - lw_max)) + 1e-300)
        wnorm[1, :] = self._safe_normalize(np.exp(lw - lw_max))

        ess_list: List[float] = [self._ess(wnorm[1, :])]
        maxw_list: List[float] = [float(np.max(wnorm[1, :]))]
        resample_count: int = 0

        if self.cfg.progress:
            print(f"  Running conditional PGAS (disturbance space; ESS threshold={self.cfg.ess_threshold_frac:.2f}, N={N})")

        # --- t = 2..T
        it = tqdm(range(2, T + 1)) if self.cfg.progress else range(2, T + 1)
        for t in it:
            # Resample on ESS criterion using w_{t-1}
            res_p = self._safe_normalize(wnorm[t - 1, :])
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
                aidx[t, n] = anc[n]
                parts_x[t - 1, n, :] = parts_x[t - 1, aidx[t, n], :]
                w_draw = self._draw_w_prior(svec)
                parts_w[t, n, :] = w_draw
                mean = self._state_mean(parts_x[t - 1, n, :], t=t)
                parts_x[t, n, :] = self._apply_innovation(mean, w_draw)

            # reference path (n = N-1)
            parts_x[t - 1, N - 1, :] = parts_x[t - 1, np.random.choice(N, p=res_p), :]
            parts_w[t, N - 1, :] = w_ref[t, :]
            mean_ref = self._state_mean(parts_x[t - 1, N - 1, :], t=t)
            parts_x[t, N - 1, :] = self._apply_innovation(mean_ref, parts_w[t, N - 1, :])
            aidx[t, N - 1] = np.random.choice(N, p=res_p)

            # weights at time t (likelihood only; proposal is prior)
            y_idx = t - 1
            lw = np.zeros(N, float)
            for n in range(N):
                mu = self.mu_from_state(parts_x[t, n, :], t=y_idx)
                lw[n] = gev_logpdf(self.y[y_idx], mu, self.sigma, self.xi)

            lw_max = np.max(lw)
            logZ += lw_max + math.log(np.mean(np.exp(lw - lw_max)) + 1e-300)
            wnorm[t, :] = self._safe_normalize(np.exp(lw - lw_max))

            ess_t = self._ess(wnorm[t, :])
            maxw_t = float(np.max(wnorm[t, :]))
            ess_list.append(ess_t); maxw_list.append(maxw_t)

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
        return parts_w, wnorm, aidx, float(logZ), pf_diag

    def _trace_single_trajectory(self, parts_w: np.ndarray, a: np.ndarray,
                                 wnorm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Backward-sample one disturbance path; reconstruct x deterministically."""
        N, T, D = self.cfg.n_particles, self.T, self.dim
        idx = np.zeros(T + 1, dtype=int)
        idx[T] = np.random.choice(N, p=wnorm[T, :])
        for t in range(T, 1, -1):
            idx[t - 1] = a[t, idx[t]]
        w_new = np.zeros((T + 1, D), float)
        x_new = np.zeros((T + 1, D), float)
        if D > 0:
            x_new[0, :] = self.x[0, :]
            for t in range(1, T + 1):
                w_new[t, :] = parts_w[t, idx[t], :]
                mean = self._state_mean(x_new[t - 1, :], t=t)
                x_new[t, :] = self._apply_innovation(mean, w_new[t, :])
        return x_new, w_new

    def update_states_pgas(self) -> None:
        parts_w, wnorm, a, logZ, pf_diag = self._conditional_pgas()
        x_new, w_new = self._trace_single_trajectory(parts_w, a, wnorm)
        self.x = x_new
        self.w = w_new
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
        if self.dim == 0: return
        pos = 0
        if self.idx_alpha is not None:
            self.m0_alpha = self._gibbs_m0_scalar(
                float(self.x[0, pos]),
                self.priors.m_m0_alpha, self.priors.s_m0_alpha, self.P0_alpha
            )
            pos += 1
        if self.idx_beta is not None:
            self.m0_beta = self._gibbs_m0_scalar(
                float(self.x[0, pos]),
                self.priors.m_m0_beta, self.priors.s_m0_beta, self.P0_beta
            )
            pos += 1
        if self.seasonal_mode == "dynamic":
            m_prior = (np.zeros(self.period - 1, float)
                       if self.priors.m_m0_gamma is None
                       else np.asarray(self.priors.m_m0_gamma, float))
            if m_prior.size != self.period - 1:
                raise ValueError("priors.m_m0_gamma must be length p-1 (NEWEST-FIRST)")
            s = float(self.priors.s_m0_gamma)
            for k in range(self.period - 1):
                self.m0_gamma[k] = self._gibbs_m0_scalar(
                    float(self.x[0, pos + k]),
                    float(m_prior[k]), s, self.P0_gamma
                )

    def update_P0(self) -> None:
        if self.dim == 0: return
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
        if self.level_mode == "deterministic":  self.update_level_value()
        if self.trend_mode == "deterministic":  self.update_slope()
        if self.seasonal_mode == "deterministic": self.update_season_vec()

    # --------------------------- Progress helpers --------------------------- #
    def _fmt_list(self, vals, max_elems: int = 6, fmt: str = ".4g") -> str:
        if vals is None: return "-"
        v = np.asarray(vals, float).ravel()
        if v.size == 0: return "[]"
        if v.size <= max_elems:
            return "[" + ", ".join(f"{x:{fmt}}" for x in v) + "]"
        head = ", ".join(f"{x:{fmt}}" for x in v[:max_elems])
        return f"[{head}, …]"

    def _progress_line(self, it: int) -> str:
        def pct(key: str) -> str:
            p = self.proposals.get(key, 0); a = self.accept.get(key, 0)
            return "0.0%" if p <= 0 else f"{100.0 * a / p:.1f}%"

        parts = [f"[it {it + 1}/{self.cfg.n_iter}]"]
        parts.append(f"logZ={self.last_log_evidence:.3f}")
        parts.append(f"σ={math.exp(self.logsigma):.3f} ({pct('logsigma')})")
        parts.append(f"ξ={self.xi:.3f} ({pct('xi')})")
        # show Q = s^2
        if self.idx_alpha is not None: parts.append(f"Qα={self.s_alpha**2:.4g}")
        if self.idx_beta  is not None: parts.append(f"Qβ={self.s_beta**2:.4g}")
        if self.seasonal_mode == "dynamic": parts.append(f"Qγ={self.s_gamma**2:.4g}")
        # m0/P0 or slope
        if self.level_mode == "dynamic":
            parts.append(f"m0α={self.m0_alpha:.4g} P0α={self.P0_alpha:.4g}")
        else:
            parts.append(f"m0α={self.level_value:.4g} P0α=0 ({pct('level')})")
        if self.trend_mode == "dynamic":
            parts.append(f"m0β={self.m0_beta:.4g} P0β={self.P0_beta:.4g}")
        elif self.trend_mode == "deterministic":
            parts.append(f"m0β={self.slope_value:.4g} P0β=0 ({pct('slope')})")
        if self.seasonal_mode != "none":
            if self.seasonal_mode == "dynamic":
                head = self.m0_gamma; head_show = head[:6] if len(head) > 6 else head
                head_str = "[" + ", ".join(f"{float(x):.4g}" for x in head_show) \
                           + (", …]" if len(head) > 6 else "]")
                parts.append(f"m0γ={head_str} P0γ={self.P0_gamma:.4g}")
            elif self.seasonal_mode == "deterministic" and self.season_vec is not None:
                head = self.season_vec[:-1]; head_show = head[:6] if len(head) > 6 else head
                head_str = "[" + ", ".join(f"{float(x):.4g}" for x in head_show) \
                           + (", …]" if len(head) > 6 else "]")
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
                              "x": np.zeros((n_kept, self.T, self.dim)),
                              "w": np.zeros((n_kept, self.T, self.dim))})
        elif self.dim > 0:
            self.keep["x"] = np.zeros((n_kept, self.T, self.dim))
            self.keep["w"] = np.zeros((n_kept, self.T, self.dim))

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

            # 1) States via PGAS (disturbance space) or pure likelihood if dim=0
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

            # 7) Adapt RW–MH step sizes
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
                    self.keep["w"][keep_idx, :, :] = self.w[1 : self.T + 1, :]
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
        arrays["w_last"] = self.w[1 : self.T + 1].copy() if self.dim > 0 else np.zeros((self.T, 0))

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
            "xi_prior": {
                "type": "uniform",
                "lower": float(self.priors.xi_lower),
                "upper": float(self.priors.xi_upper),
            },
            "true_sigma": self.true_sigma,
            "true_xi": self.true_xi,
            "true_Q": (None if self.true_Q is None else np.asarray(self.true_Q, float).tolist()),
            "accept": self.accept,
            "proposals": self.proposals,
        }
        if extra_meta: meta.update(extra_meta)
        meta_path = out_npz_path.replace(".npz", ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[save] Posterior -> {out_npz_path}")
        print(f"[save] Metadata  -> {meta_path}")


# ------------------------- CLI / Example run & plots ------------------------
if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    # make simulator importable
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from simulator.extremal_time_series import Extremal_Time_Series

    def _parse_date(s: str | None):
        if not s: return datetime.today()
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

    parser = argparse.ArgumentParser(
        description=("DGEV PGAS Sampler — disturbance-space NCP with log-normal priors on process SDs "
                     "and a uniform prior for xi on [xi_lower, xi_upper].")
    )

    # Simulation controls
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--period", type=int, default=4)
    parser.add_argument("--start-date", type=str, default="2000-01-01")

    parser.add_argument("--level-mode",   choices=["dynamic", "deterministic"],            default="dynamic")
    parser.add_argument("--trend-mode",   choices=["dynamic", "deterministic", "none"],    default="dynamic")
    parser.add_argument("--seasonal-mode", choices=["dynamic", "deterministic", "none"],   default="dynamic")

    # Truth / simulator params
    parser.add_argument("--sigma",     type=float, default=4.0)
    parser.add_argument("--xi",        type=float, default=-0.1)
    parser.add_argument("--q-level",   type=float, default=1e-3)
    parser.add_argument("--q-trend",   type=float, default=7e-6)
    parser.add_argument("--q-season",  type=float, default=5e-7)

    parser.add_argument("--m0-level",  type=float, default=5.0)
    parser.add_argument("--v0-level",  type=float, default=0.2)
    parser.add_argument("--m0-trend",  type=float, default=0.1)
    parser.add_argument("--v0-trend",  type=float, default=0.05)
    parser.add_argument("--m0-season", type=str,   default=None, help="comma-separated (length p-1)")
    parser.add_argument("--v0-season", type=str,   default=None, help="comma-separated (length p-1)")

    # Sampler initialization (m0, P0)
    parser.add_argument("--init-m0-level",  type=float, default=None)
    parser.add_argument("--init-p0-level",  type=float, default=None)
    parser.add_argument("--init-m0-trend",  type=float, default=None)
    parser.add_argument("--init-p0-trend",  type=float, default=None)
    parser.add_argument("--init-m0-season", type=str,   default=None)
    parser.add_argument("--init-p0-season", type=float, default=None)

    # Inference priors (obs + deterministic)
    parser.add_argument("--prior-m-sigma",  type=float, default=0.0)
    parser.add_argument("--prior-s-sigma",  type=float, default=0.1)

    # Uniform prior bounds for xi
    parser.add_argument("--prior-xi-lower", type=float, default=-0.5)
    parser.add_argument("--prior-xi-upper", type=float, default=0.5)

    parser.add_argument("--prior-m-level",  type=float, default=0.0)
    parser.add_argument("--prior-s-level",  type=float, default=10.0)
    parser.add_argument("--prior-m-slope",  type=float, default=0.0)
    parser.add_argument("--prior-s-slope",  type=float, default=10.0)
    parser.add_argument("--prior-m-season", type=str,   default=None, help="comma-separated (length p-1)")
    parser.add_argument("--prior-s-season", type=float, default=5.0)

    # (Optional) ln s hyperparams (if you want to override dataclass defaults)
    parser.add_argument("--prior-ln-s-alpha-m", type=float, default=None)
    parser.add_argument("--prior-ln-s-alpha-sd", type=float, default=None)
    parser.add_argument("--prior-ln-s-beta-m",  type=float, default=None)
    parser.add_argument("--prior-ln-s-beta-sd", type=float, default=None)
    parser.add_argument("--prior-ln-s-gamma-m", type=float, default=None)
    parser.add_argument("--prior-ln-s-gamma-sd", type=float, default=None)

    # Sampler config
    parser.add_argument("--n-iter", type=int, default=4000)
    parser.add_argument("--burn",   type=int, default=1000)
    parser.add_argument("--thin",   type=int, default=1)

    # RW–MH steps (obs + deterministic)
    parser.add_argument("--step-logsigma", type=float, default=0.05)
    parser.add_argument("--step-xi",       type=float, default=0.02)
    parser.add_argument("--step-level",    type=float, default=0.2)
    parser.add_argument("--step-slope",    type=float, default=0.001)
    parser.add_argument("--step-season",   type=float, default=0.02)

    # PF / PGAS
    parser.add_argument("--particles",   type=int,   default=500)
    parser.add_argument("--ess-frac",    type=float, default=0.5)

    # Adaptation
    parser.add_argument("--adapt-steps",        default=True)
    parser.add_argument("--adapt-every",        type=int,    default=20)
    parser.add_argument("--adapt-until",        choices=["burn","all"], default="burn")
    parser.add_argument("--adapt-eta0",         type=float,  default=0.1)
    parser.add_argument("--adapt-decay",        type=float,  default=0.75)
    parser.add_argument("--adapt-target-1d",    type=float,  default=0.3)
    parser.add_argument("--step-min",           type=float,  default=1e-5)
    parser.add_argument("--step-max",           type=float,  default=1.0)

    # I/O & misc
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--progress-every", type=int, default=1)
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
        m_sigma=float(args.prior_m_sigma), s_sigma=float(args.prior_s_sigma),
        xi_lower=float(args.prior_xi_lower), xi_upper=float(args.prior_xi_upper),
        m_level=float(args.prior_m_level), s_level=float(args.prior_s_level),
        m_slope=float(args.prior_m_slope), s_slope=float(args.prior_s_slope),
        m_season=None if pri_season_first is None else pri_season_first,
        s_season=float(args.prior_s_season),
    )
    # optional override ln s hyperparams
    if args.prior_ln_s_alpha_m is not None: priors.mu_log_s_alpha = float(args.prior_ln_s_alpha_m)
    if args.prior_ln_s_alpha_sd is not None: priors.sd_log_s_alpha = float(args.prior_ln_s_alpha_sd)
    if args.prior_ln_s_beta_m  is not None: priors.mu_log_s_beta  = float(args.prior_ln_s_beta_m)
    if args.prior_ln_s_beta_sd is not None: priors.sd_log_s_beta  = float(args.prior_ln_s_beta_sd)
    if args.prior_ln_s_gamma_m is not None: priors.mu_log_s_gamma = float(args.prior_ln_s_gamma_m)
    if args.prior_ln_s_gamma_sd is not None: priors.sd_log_s_gamma = float(args.prior_ln_s_gamma_sd)

    if args.level_mode == "deterministic":
        if args.init_m0_level is None:
            args.init_m0_level = float(np.median(y))
            priors.m_level = args.init_m0_level
            priors.s_level = max(2.0, 0.5 * y.std(ddof=1))

    cfg = SamplerConfig(
        n_iter=int(args.n_iter),
        burn=int(args.burn),
        thin=int(args.thin),
        step_logsigma=float(args.step_logsigma),
        step_xi=float(args.step_xi),
        step_level=float(args.step_level),
        step_slope=float(args.step_slope),
        step_season=float(args.step_season),
        n_particles=int(args.particles),
        random_seed=int(args.seed),
        progress=bool(args.progress),
        progress_every=int(args.progress_every),
        ess_threshold_frac=float(args.ess_frac),
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

    # sampler init (x0 prior), decoupled from simulator truth
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

    sampler = DGEVParticleGibbs(
        y=y, period=int(args.period),
        level_mode=args.level_mode, trend_mode=args.trend_mode, seasonal_mode=args.seasonal_mode,
        m0_level_init=float(init_m0_level), P0_level_init=float(init_p0_level),
        m0_trend_init=float(init_m0_trend), P0_trend_init=float(init_p0_trend),
        m0_season_init=(init_m0_season if args.seasonal_mode == "dynamic" else None),
        P0_season_init=(init_p0_season_scalar if args.seasonal_mode == "dynamic" else 0.0),
        sigma_init=float(args.sigma), xi_init=float(args.xi),
        s_alpha_init=(math.sqrt(args.q_level)  if args.level_mode  == "dynamic" else 0.0),
        s_beta_init=(math.sqrt(args.q_trend)  if args.trend_mode  == "dynamic" else 0.0),
        s_gamma_init=(math.sqrt(args.q_season) if args.seasonal_mode == "dynamic" else 0.0),
        priors=priors, cfg=cfg,
        level_value_init=(float(args.init_m0_level) if args.level_mode == "deterministic" else 0.0),
        slope_value_init=0.0,
        seasonal_vector_init=seasonal_init_pminus1,
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
            print(f"xi prior: Uniform[{priors.xi_lower}, {priors.xi_upper}]")

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
            if v.size == 0: return "[]"
            if v.size <= k: return "[" + ", ".join(f"{x:.4g}" for x in v) + "]"
            return "[" + ", ".join(f"{x:.4g}" for x in v[:k]) + ", …]"

        with np.printoptions(suppress=True, precision=4):
            print("\n--- Summary (posterior means) ---")
            sig_mean = float(np.mean(post["sigma"])) if "sigma" in post else float("nan")
            xi_mean  = float(np.mean(post["xi"]))    if "xi"    in post else float("nan")
            print(f"σ: {sig_mean:.4g}")
            print(f"ξ: {xi_mean:.4g}")

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

            if "log_evidence" in post and post["log_evidence"].size > 0:
                le = post["log_evidence"]
                print(f"log p(y|θ): mean={np.nanmean(le):.3f}, median={np.nanmedian(le):.3f}, best={np.nanmax(le):.3f}")

            print("\n--- m0 / P0 (posterior means) ---")
            if args.level_mode == "dynamic":
                m0a = float(np.mean(post["m0_alpha"])) if "m0_alpha" in post else float("nan")
                P0a = float(np.mean(post["P0_alpha"])) if "P0_alpha" in post else float("nan")
                print(f"m0α: {m0a:.4g} | P0α: {P0a:.4g}")
            else:
                lvl_samples = post.get("level_value", None)
                m0a_det = (float(np.mean(lvl_samples)) if isinstance(lvl_samples, np.ndarray) and lvl_samples.size
                           else float(sampler.level_value))
                print(f"m0α (deterministic level): {m0a_det:.4g} | P0α: 0")

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

            if args.seasonal_mode == "dynamic":
                if "m0_gamma" in post and post["m0_gamma"].size:
                    m0g_vec = np.mean(post["m0_gamma"], axis=0)
                    P0g = float(np.mean(post["P0_gamma"])) if "P0_gamma" in post else float("nan")
                    print(f"m0γ (first p−1): {_fmt_vec(m0g_vec)} | P0γ: {P0g:.4g}")
                else:
                    print("m0γ: n/a | P0γ: n/a")
            elif args.seasonal_mode == "deterministic":
                sv_samples = post.get("season_vector", None)
                if isinstance(sv_samples, np.ndarray) and sv_samples.ndim == 2 and sv_samples.size:
                    sv_mean = sv_samples.mean(axis=0)
                else:
                    sv_mean = np.asarray(sampler.season_vec, float) if sampler.season_vec is not None else np.zeros(args.period)
                print(f"season vector (p): {_fmt_vec(sv_mean)}")
                print(f"m0γ (first p−1): {_fmt_vec(sv_mean[:-1])} | P0γ: 0")
            else:
                print("m0γ: n/a (season none) | P0γ: n/a")
