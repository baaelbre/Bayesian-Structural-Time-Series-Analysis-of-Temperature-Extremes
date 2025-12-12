from __future__ import annotations

"""
Structural DGEV with harmonic seasonality (cos/sin pairs + optional Nyquist)
— Particle Gibbs with Ancestor Sampling (PGAS) for the latent state
— RW–MH for (σ, ξ) with:
      σ² ~ Inv-Gamma(a_sigma, b_sigma)
      ξ  ~ Uniform[xi_lower, xi_upper]
— Log-normal priors on process SDs s_α, s_β, s_γ with slice sampling on ln s.

State (dynamic):
  [alpha] [beta] [c1 s1 | c2 s2 | ... | cK sK | nyq?]

Location:
  μ_t = μ_det(t) + H x_t

Measurement:
  Y_t | μ_t, σ, ξ ~ GEV(μ_t, σ, ξ)

"""

import os, sys, math, json, time
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, Dict, List, Sequence

import numpy as np
from tqdm import tqdm

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_dir)
from optimization.harmonic_helpers import (center_and_report_dummies_full, dummies_full_to_harmonics_fft,harmonics_to_dummies_full_fft)

# =============================================================================
# Utilities
# =============================================================================

def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

def _mad(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    if v.size == 0:
        return 0.0
    med = np.median(v)
    return float(np.median(np.abs(v - med)))

def _robust_sd(v: np.ndarray) -> float:
    return _mad(np.asarray(v, float)) / 1.4826 if v.size else 0.0

def _spd_solve(M: np.ndarray, B: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Solve M X = B for SPD (or near-SPD) M using Cholesky with small jitter."""
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
# Univariate slice sampler
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
            u = self.rng.random()
            y = hz + math.log(u)

            w = self.w
            L = z - w * self.rng.random()
            R = L + w
            J = int(self.rng.integers(0, self.m)) if self.m > 0 else 0
            K = (self.m - 1 - J) if self.m > 0 else 0

            # step out
            while J > 0 and self.h(L) > y:
                L -= w
                J -= 1
            while K > 0 and self.h(R) > y:
                R += w
                K -= 1

            # shrink
            while True:
                z_new = self.rng.uniform(L, R)
                if self.h(z_new) >= y:
                    z = z_new
                    break
                if z_new < z:
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
# =============================================================================

@dataclass
class Priors:
    # σ² ~ Inv-Gamma(a_sigma, b_sigma)
    a_sigma: float = 2.0
    b_sigma: float = 2.0

    # ξ ~ Uniform[ xi_lower, xi_upper ]
    xi_lower: float = -0.5
    xi_upper: float = 0.5

    # m0 priors (Normal)
    m_m0_alpha: float = 0.0
    s_m0_alpha: float = 10.0
    m_m0_beta: float = 0.0
    s_m0_beta: float = 10.0
    m_m0_nyq: float = 0.0
    s_m0_harm: float = 5.0
    # optional means for harmonics (if desired)
    m_m0_cos: Optional[Sequence[float]] = None
    m_m0_sin: Optional[Sequence[float]] = None

    # P0 priors (Inv-Gamma) for initial dynamic states
    a_P0_alpha: float = 2.0
    b_P0_alpha: float = 1.0
    a_P0_beta: float = 2.0
    b_P0_beta: float = 1.0
    a_P0_harm: float = 2.0
    b_P0_harm: float = 1.0

    # Log-normal priors on process SDs: ln s_k ~ N(mu, sd^2)
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

    # PF / PGAS
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
# DGEV with harmonic seasonality + PGAS
# =============================================================================
class DGEVParticleGibbs:
    """
    Structural DGEV with harmonic seasonality.

    Dynamic state:
      [alpha] [beta] [c1 s1 | ... | cK sK | nyq?]

    Modes:
      level_mode   ∈ {"dynamic", "deterministic"}
      trend_mode   ∈ {"dynamic", "deterministic", "none"}
      seasonal_mode∈ {"dynamic", "deterministic", "none"}
    """

    # --------------------------- Construction --------------------------- #
    def __init__(
        self,
        y: np.ndarray,
        period: int,
        harmonics: Optional[int] = None,
        use_nyquist: Optional[bool] = None,
        level_mode: str = "dynamic",
        trend_mode: str = "dynamic",
        seasonal_mode: str = "dynamic",
        # Initial means / variances for dynamic x0
        m0_alpha_init: float = 0.0,
        P0_alpha_init: float = 1.0,
        m0_beta_init: float = 0.0,
        P0_beta_init: float = 1.0,
        m0_cos_init: Optional[Sequence[float]] = None,
        m0_sin_init: Optional[Sequence[float]] = None,
        m0_nyq_init: float = 0.0,
        P0_harm_init: float = 1.0,
        # Observation parameters
        sigma_init: float = 1.0,
        xi_init: float = 0.0,
        # Process SD initial values
        s_alpha_init: float = 1e-2,
        s_beta_init:  float = 1e-3,
        s_gamma_init: float = 1e-3,
        # priors / config
        priors: Priors = Priors(),
        cfg: SamplerConfig = SamplerConfig(),
    ):
        # ----- data / period -----
        self.y = np.asarray(y, float)
        self.T = int(self.y.size)
        self.s = int(period)
        if self.s < 2:
            raise ValueError("period must be >= 2")

        # harmonic resolution
        K_full = (self.s - 1) // 2
        self.K = K_full if harmonics is None else int(harmonics)
        if not (0 <= self.K <= K_full):
            raise ValueError(f"harmonics K must be in [0, {K_full}] for s={self.s}")
        even = (self.s % 2) == 0
        if use_nyquist is None:
            self.use_nyq = bool(even and (self.K >= (self.s // 2 - 1)))
        else:
            self.use_nyq = bool(use_nyquist and even)

        # modes
        ok = {"dynamic", "deterministic", "none"}
        if level_mode not in ok or trend_mode not in ok or seasonal_mode not in ok:
            raise ValueError("invalid mode")
        if trend_mode == "dynamic" and level_mode != "dynamic":
            raise ValueError("dynamic trend requires dynamic level")
        self.level_mode, self.trend_mode, self.seasonal_mode = level_mode, trend_mode, seasonal_mode

        self.priors, self.cfg = priors, cfg
        if cfg.random_seed is not None:
            np.random.seed(cfg.random_seed)
            self._rng = np.random.default_rng(cfg.random_seed)
        else:
            self._rng = np.random.default_rng()

        # harmonic rotation caches
        self._omegas = 2.0 * np.pi * (np.arange(1, self.K + 1, dtype=float)) / float(self.s)
        self._cosw = np.cos(self._omegas)
        self._sinw = np.sin(self._omegas)

        # ----- layout for dynamic state -----
        layout: List[str] = []
        if self.level_mode == "dynamic":
            layout.append("alpha")
        if self.trend_mode == "dynamic":
            layout.append("beta")
        if self.seasonal_mode == "dynamic":
            for k in range(1, self.K + 1):
                layout += [f"c{k}", f"s{k}"]
            if self.use_nyq:
                layout.append("nyq")
        self._layout = layout
        self.dim = len(layout)

        self.idx_alpha = layout.index("alpha") if "alpha" in layout else None
        self.idx_beta  = layout.index("beta")  if "beta"  in layout else None

        def _idx_pair(k: int) -> int:
            pos = 0
            if self.idx_alpha is not None: pos += 1
            if self.idx_beta  is not None: pos += 1
            pos += 2 * (k - 1)
            return pos
        self._idx_pair = _idx_pair

        if self.seasonal_mode == "dynamic":
            self.idx_first_season = (_idx_pair(1) if self.K > 0 else None)
            self.idx_nyq = (None if not self.use_nyq else
                            ((1 if self.idx_alpha is not None else 0) +
                             (1 if self.idx_beta  is not None else 0) +
                             2*self.K))
        else:
            self.idx_first_season = None
            self.idx_nyq = None

        # ----- observation params -----
        self.logsigma = float(np.log(max(1e-12, sigma_init)))
        self.sigma = float(np.exp(self.logsigma))
        self.xi = float(xi_init)
        if not (priors.xi_lower < priors.xi_upper):
            raise ValueError("priors.xi_lower must be < priors.xi_upper")
        if not (priors.xi_lower <= self.xi <= priors.xi_upper):
            raise ValueError(f"Initial xi={xi_init} must be in [{priors.xi_lower}, {priors.xi_upper}]")

        # ----- process SDs -----
        self.s_alpha = float(max(1e-12, s_alpha_init)) if self.idx_alpha is not None else 0.0
        self.s_beta  = float(max(1e-12, s_beta_init))  if self.idx_beta  is not None else 0.0
        self.s_gamma = float(max(1e-12, s_gamma_init)) if self.seasonal_mode == "dynamic" else 0.0

        # ----- m0 / P0 -----
        self.m0_alpha = float(m0_alpha_init) if self.idx_alpha is not None else 0.0
        self.P0_alpha = float(P0_alpha_init) if self.idx_alpha is not None else 0.0
        self.m0_beta  = float(m0_beta_init)  if self.idx_beta  is not None else 0.0
        self.P0_beta  = float(P0_beta_init)  if self.idx_beta  is not None else 0.0

        if self.K > 0:
            if m0_cos_init is None: m0_cos_init = np.zeros(self.K, float)
            if m0_sin_init is None: m0_sin_init = np.zeros(self.K, float)
            if len(m0_cos_init) != self.K or len(m0_sin_init) != self.K:
                raise ValueError("m0_cos_init and m0_sin_init must have length K")
            self.m0_cos = np.asarray(m0_cos_init, float)
            self.m0_sin = np.asarray(m0_sin_init, float)
        else:
            self.m0_cos = np.zeros(0, float)
            self.m0_sin = np.zeros(0, float)
        self.m0_nyq = (float(m0_nyq_init) if self.use_nyq else None)
        self.P0_harm = float(P0_harm_init)

        # ----- latent path x_{0:T} -----
        self.x = np.zeros((self.T + 1, self.dim), float)
        if self.dim > 0:
            m0_vec, P0_diag = self._current_m0_P0()
            self.x[0] = np.random.multivariate_normal(
                m0_vec, np.diag(P0_diag) + 1e-12 * np.eye(self.dim)
            )
            self._propagate_initial_path(Q_init=np.full(self.dim, 1e-6, float))

        # storage
        self.keep: Dict[str, np.ndarray] = {}

        # MH bookkeeping
        self.accept = {"logsigma": 0, "xi": 0, "level": 0, "slope": 0, "season": 0}
        self.proposals = dict(self.accept)
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

        # slice samplers for ln s
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

        if self.cfg.progress:
            sd1 = _robust_sd(np.diff(self.y)) if self.y.size >= 2 else 0.0
            sd2 = _robust_sd(np.diff(self.y, n=2)) if self.y.size >= 3 else 0.0
            nyq_tag = f"nyq={self.use_nyq}"
            print(f"[init] sd1={sd1:.4g}, sd2={sd2:.4g} | K={self.K} {nyq_tag}")

    # --------------------- Truth registration (optional) --------------------- #
    def set_truth(self, sigma: Optional[float] = None,
                  xi: Optional[float] = None,
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

    # ----------------------------- Model matrices ----------------------------- #
    def _H(self) -> np.ndarray:
        if self.dim == 0:
            return np.zeros((1, 0))
        h = np.zeros(self.dim, float)
        if self.idx_alpha is not None:
            h[self.idx_alpha] = 1.0
        if self.seasonal_mode == "dynamic" and self.K > 0:
            for k in range(1, self.K + 1):
                i = self._idx_pair(k)
                h[i] = 1.0  # cos only
            if self.use_nyq:
                h[self.idx_nyq] = 1.0
        return h.reshape(1, -1)

    def _A(self) -> np.ndarray:
        if self.dim == 0:
            return np.zeros((0, 0))
        A = np.eye(self.dim)
        if self.idx_alpha is not None and self.idx_beta is not None:
            A[self.idx_alpha, self.idx_beta] = 1.0
        if self.seasonal_mode == "dynamic" and self.K > 0:
            for k in range(1, self.K + 1):
                i = self._idx_pair(k)
                co, si = float(self._cosw[k - 1]), float(self._sinw[k - 1])
                A[i,   i  ] =  co
                A[i,   i+1] =  si
                A[i+1, i  ] = -si
                A[i+1, i+1] =  co
            if self.use_nyq:
                A[self.idx_nyq, self.idx_nyq] = -1.0
        return A

    def _u(self) -> np.ndarray:
        if self.dim == 0:
            return np.zeros(0, float)
        u = np.zeros(self.dim, float)
        if (self.idx_alpha is not None) and (self.trend_mode == "deterministic"):
            # deterministic slope m0_beta drives alpha when trend is deterministic
            u[self.idx_alpha] = float(self.m0_beta)
        return u

    def _Q(self) -> np.ndarray:
        if self.dim == 0:
            return np.zeros((0, 0))
        Q = np.zeros((self.dim, self.dim))
        if self.idx_alpha is not None and self.s_alpha > 0:
            Q[self.idx_alpha, self.idx_alpha] = self.s_alpha**2
        if self.idx_beta is not None and self.s_beta > 0:
            Q[self.idx_beta, self.idx_beta] = self.s_beta**2
        if self.seasonal_mode == "dynamic" and self.s_gamma > 0:
            for k in range(1, self.K + 1):
                i = self._idx_pair(k)
                Q[i, i] = self.s_gamma**2
                Q[i+1, i+1] = self.s_gamma**2
            if self.use_nyq:
                Q[self.idx_nyq, self.idx_nyq] = self.s_gamma**2
        return Q

    # ---------------- deterministic mean pieces ----------------
    def _season_det(self, t: int) -> float:
        if self.seasonal_mode != "deterministic":
            return 0.0
        val = 0.0
        for k in range(1, self.K + 1):
            w = self._omegas[k - 1]
            val += self.m0_cos[k - 1] * math.cos(w * t) + self.m0_sin[k - 1] * math.sin(w * t)
        if self.use_nyq and (self.m0_nyq is not None):
            val += float(self.m0_nyq) * ((-1.0) ** t)
        return float(val)

    def _mu_det(self, t: int) -> float:
        out = 0.0
        if self.level_mode == "deterministic":
            out += self.m0_alpha
        if (self.trend_mode == "deterministic") and (self.idx_alpha is None):
            out += self.m0_beta * t
        out += self._season_det(t)
        return out

    def _current_m0_P0(self) -> Tuple[np.ndarray, np.ndarray]:
        m0, P0 = [], []
        if self.idx_alpha is not None:
            m0.append(self.m0_alpha); P0.append(self.P0_alpha)
        if self.idx_beta is not None:
            m0.append(self.m0_beta);  P0.append(self.P0_beta)
        if self.seasonal_mode == "dynamic":
            for k in range(self.K):
                m0 += [self.m0_cos[k], self.m0_sin[k]]
                P0 += [self.P0_harm,   self.P0_harm]
            if self.use_nyq:
                m0.append(float(0.0 if self.m0_nyq is None else self.m0_nyq))
                P0.append(self.P0_harm)
        return np.asarray(m0, float), np.asarray(P0, float)

    # ------------------ μ and state mean ------------------ #
    def _mu_vec_current(self) -> np.ndarray:
        H = self._H()
        mu = np.zeros(self.T, float)
        for t in range(1, self.T + 1):
            dyn = float(H @ self.x[t]) if self.dim > 0 else 0.0
            mu[t - 1] = self._mu_det(t - 1) + dyn
        return mu

    def mu_from_state(self, x_t: np.ndarray, t: int) -> float:
        if self.dim == 0:
            return self._mu_det(t)
        H = self._H()
        dyn = float(H @ x_t)
        return float(self._mu_det(t) + dyn)

    def _state_mean(self, x_prev: np.ndarray) -> np.ndarray:
        A = self._A()
        u = self._u()
        return A @ x_prev + u

    # --------------------- Transitions --------------------- #
    def _transition_logpdf(self, x_prev: np.ndarray, x_cur: np.ndarray) -> float:
        if self.dim == 0:
            return 0.0
        mean = self._state_mean(x_prev)
        Q = self._Q()
        # ensure writable copy
        var = np.array(np.diag(Q), copy=True)
        var[var <= 0.0] = self.cfg.trans_eps
        diff = x_cur - mean
        out = -0.5 * (np.log(2.0 * np.pi * var) + (diff * diff) / var).sum()
        return float(out)


    def _transition_sample(self, x_prev: np.ndarray) -> np.ndarray:
        if self.dim == 0:
            return x_prev.copy()
        mean = self._state_mean(x_prev)
        Q = self._Q()
        # ensure writable copy
        var = np.array(np.diag(Q), copy=True)
        var[var <= 0.0] = self.cfg.trans_eps
        return mean + np.random.normal(0.0, np.sqrt(var), size=self.dim)


    def _propagate_initial_path(self, Q_init: np.ndarray) -> None:
        if self.dim == 0:
            return
        A, u = self._A(), self._u()
        for t in range(1, self.T + 1):
            self.x[t] = A @ self.x[t - 1] + u + np.random.normal(
                0.0, np.sqrt(Q_init), size=self.dim
            )

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
                drift = float(self.m0_beta)
            mean = self.x[t - 1, self.idx_alpha] + drift
            ss += (self.x[t, self.idx_alpha] - mean) ** 2
        return float(ss), self.T

    def _innovation_ss_beta(self) -> Tuple[float, int]:
        if self.idx_beta is None:
            return 0.0, 0
        d = self.x[1:, self.idx_beta] - self.x[:-1, self.idx_beta]
        return float(np.sum(d * d)), self.T

    def _innovation_ss_gamma(self) -> Tuple[float, int]:
        """Harmonic seasonal innovations: rotations for each (c_k, s_k) and Nyquist."""
        if self.seasonal_mode != "dynamic":
            return 0.0, 0
        ss = 0.0
        per_step = 0
        # cos/sin pairs
        for k in range(1, self.K + 1):
            i = self._idx_pair(k)
            co, si = float(self._cosw[k - 1]), float(self._sinw[k - 1])
            R = np.array([[co, si], [-si, co]], float)
            for t in range(1, self.T + 1):
                prev = self.x[t - 1, i:i+2]
                mean = R @ prev
                err = self.x[t, i:i+2] - mean
                ss += float(err @ err)
        per_step += 2 * self.K
        # Nyquist
        if self.use_nyq:
            j = self.idx_nyq
            for t in range(1, self.T + 1):
                mean = -self.x[t - 1, j]
                err = self.x[t, j] - mean
                ss += float(err * err)
            per_step += 1
        T_eff = self.T * per_step
        return float(ss), int(T_eff)

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
            z0 = math.log(max(self.s_alpha, 1e-12))
            z  = float(self._slice_alpha.sample(z0, 1)[0])
            self.s_alpha = float(np.exp(z))
        # β
        if self.idx_beta is not None:
            z0 = math.log(max(self.s_beta, 1e-12))
            z  = float(self._slice_beta.sample(z0, 1)[0])
            self.s_beta = float(np.exp(z))
        # γ
        if self.seasonal_mode == "dynamic":
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

    # Inv-Gamma prior on v = σ²:
    # v ~ IG(a,b): p(v) ∝ v^{-(a+1)} exp(-b/v)
    # For l = ln σ, v = exp(2l) ⇒ log p(l) = -2a l - b exp(-2l) + const
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
        """RW–MH for ξ with bounded uniform prior on [xi_lower, xi_upper]."""
        step = self.cfg.step_xi
        cur = self.xi
        prop = cur + np.random.normal(0.0, step)
        self.proposals["xi"] += 1

        lb = float(self.priors.xi_lower)
        ub = float(self.priors.xi_upper)
        if not (lb <= prop <= ub):
            return

        mu_vec = self._mu_vec_current()
        ll_old = gev_loglike_sum(self.y, mu_vec, self.sigma, cur)
        ll_new = gev_loglike_sum(self.y, mu_vec, self.sigma, prop)
        if ll_new == -np.inf:
            return

        # uniform prior cancels in ratio
        if self._mh_accept(ll_new - ll_old):
            self.xi = prop
            self.accept["xi"] += 1

    # ---------------- Deterministic structural parameter updates ------------- #
    def update_level_value(self) -> None:
        """Random-walk MH on m0_alpha when level_mode='deterministic'."""
        if self.level_mode != "deterministic":
            return
        step = self.cfg.step_level
        cur = self.m0_alpha
        prop = cur + np.random.normal(0.0, step)

        # μ under old/new
        mu_old = self._mu_vec_current()
        self.m0_alpha = prop
        mu_new = self._mu_vec_current()
        self.m0_alpha = cur

        ll_old = gev_loglike_sum(self.y, mu_old, self.sigma, self.xi)
        ll_new = gev_loglike_sum(self.y, mu_new, self.sigma, self.xi)
        self.proposals["level"] += 1
        if ll_new == -np.inf:
            return
        m0, s0 = float(self.priors.m_m0_alpha), float(self.priors.s_m0_alpha)
        lp_old = -0.5 * ((cur - m0) ** 2) / (s0**2)
        lp_new = -0.5 * ((prop - m0) ** 2) / (s0**2)
        if self._mh_accept((ll_new + lp_new) - (ll_old + lp_old)):
            self.m0_alpha = prop
            self.accept["level"] += 1

    def _alpha_transition_loglike_given_slope(self, slope: float) -> float:
        """Transition loglik for dynamic alpha when trend is deterministic (m0_beta = slope)."""
        if self.idx_alpha is None:
            return 0.0
        var = (self.s_alpha ** 2) if self.s_alpha > 0 else self.cfg.trans_eps
        inv_var = 1.0 / var
        cst = -0.5 * math.log(2.0 * math.pi * var)
        ll = 0.0
        for t in range(1, self.T + 1):
            mean = self.x[t - 1, self.idx_alpha] + slope
            diff = self.x[t, self.idx_alpha] - mean
            ll += cst - 0.5 * diff * diff * inv_var
        return float(ll)

    def update_slope(self) -> None:
        """
        RW–MH on m0_beta when trend_mode='deterministic'.

        - If alpha is dynamic (idx_alpha is not None), slope enters only via
          alpha transition (u_alpha). Use Gaussian transition likelihood.
        - If alpha is not dynamic (idx_alpha is None), slope enters μ_det(t),
          so we use GEV likelihood only.
        """
        if self.trend_mode != "deterministic":
            return
        step = self.cfg.step_slope
        cur = self.m0_beta
        prop = cur + np.random.normal(0.0, step)

        if self.idx_alpha is None:
            # no dynamic alpha ⇒ slope only in μ_det
            mu_old = self._mu_vec_current()
            self.m0_beta = prop
            mu_new = self._mu_vec_current()
            self.m0_beta = cur
            ll_obs_old = gev_loglike_sum(self.y, mu_old, self.sigma, self.xi)
            ll_obs_new = gev_loglike_sum(self.y, mu_new, self.sigma, self.xi)
            if ll_obs_new == -np.inf:
                self.proposals["slope"] += 1
                return
            ll_trans_old = 0.0
            ll_trans_new = 0.0
        else:
            # dynamic alpha with deterministic slope ⇒ only transition effect
            ll_obs_old = 0.0
            ll_obs_new = 0.0
            ll_trans_old = self._alpha_transition_loglike_given_slope(cur)
            ll_trans_new = self._alpha_transition_loglike_given_slope(prop)

        m0, s0 = float(self.priors.m_m0_beta), float(self.priors.s_m0_beta)
        lp_old = -0.5 * ((cur - m0) ** 2) / (s0**2)
        lp_new = -0.5 * ((prop - m0) ** 2) / (s0**2)

        logacc = (ll_obs_new + ll_trans_new + lp_new) - (ll_obs_old + ll_trans_old + lp_old)
        self.proposals["slope"] += 1
        if self._mh_accept(logacc):
            self.m0_beta = prop
            self.accept["slope"] += 1

    def update_season_det(self) -> None:
        """RW–MH on deterministic harm coefficients (m0_cos, m0_sin, m0_nyq)."""
        if self.seasonal_mode != "deterministic":
            return
        step = self.cfg.step_season

        # current vector of parameters
        theta_cur = []
        if self.K > 0:
            theta_cur.extend(self.m0_cos.tolist())
            theta_cur.extend(self.m0_sin.tolist())
        if self.use_nyq:
            theta_cur.append(0.0 if self.m0_nyq is None else float(self.m0_nyq))
        theta_cur = np.asarray(theta_cur, float)

        # propose
        theta_prop = theta_cur + np.random.normal(0.0, step, size=theta_cur.size)

        # unpack proposal
        idx = 0
        if self.K > 0:
            m0_cos_prop = theta_prop[idx : idx + self.K]; idx += self.K
            m0_sin_prop = theta_prop[idx : idx + self.K]; idx += self.K
        else:
            m0_cos_prop = np.zeros(0); m0_sin_prop = np.zeros(0)
        if self.use_nyq:
            m0_nyq_prop = float(theta_prop[idx])
        else:
            m0_nyq_prop = self.m0_nyq

        # likelihood under old / new
        mu_old = self._mu_vec_current()
        m0_cos_old, m0_sin_old, m0_nyq_old = self.m0_cos, self.m0_sin, self.m0_nyq
        self.m0_cos, self.m0_sin, self.m0_nyq = m0_cos_prop, m0_sin_prop, m0_nyq_prop
        mu_new = self._mu_vec_current()
        self.m0_cos, self.m0_sin, self.m0_nyq = m0_cos_old, m0_sin_old, m0_nyq_old

        ll_old = gev_loglike_sum(self.y, mu_old, self.sigma, self.xi)
        ll_new = gev_loglike_sum(self.y, mu_new, self.sigma, self.xi)
        self.proposals["season"] += 1
        if ll_new == -np.inf:
            return

        # Gaussian prior on harmonic coefficients
        s0 = float(self.priors.s_m0_harm)
        # prior means
        m_cos = (np.zeros(self.K) if self.priors.m_m0_cos is None
                 else np.asarray(self.priors.m_m0_cos, float))
        m_sin = (np.zeros(self.K) if self.priors.m_m0_sin is None
                 else np.asarray(self.priors.m_m0_sin, float))
        if m_cos.size != self.K: m_cos = np.zeros(self.K)
        if m_sin.size != self.K: m_sin = np.zeros(self.K)
        # build prior mean vector
        m_theta = []
        if self.K > 0:
            m_theta.extend(m_cos.tolist())
            m_theta.extend(m_sin.tolist())
        if self.use_nyq:
            m_theta.append(float(self.priors.m_m0_nyq))
        m_theta = np.asarray(m_theta, float) if len(m_theta) else np.zeros_like(theta_cur)

        lp_old = -0.5 * np.sum(((theta_cur - m_theta) / s0) ** 2)
        lp_new = -0.5 * np.sum(((theta_prop - m_theta) / s0) ** 2)

        if self._mh_accept((ll_new + lp_new) - (ll_old + lp_old)):
            self.m0_cos, self.m0_sin, self.m0_nyq = m0_cos_prop, m0_sin_prop, m0_nyq_prop
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

    # ---------------- Conditional PGAS ---------------- #
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

    def _conditional_pgas(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
        """
        Bootstrap conditional SMC with ESS-triggered resampling and ancestor sampling.
        Returns (parts, w, a, logZ, pf_diag).
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
                parts[1, n, :] = self._transition_sample(parts[0, n, :])
            a[1, n] = n

        if D > 0:
            parts[1, N - 1, :] = self.x[1].copy()

        lw = np.zeros(N, float)
        for n in range(N):
            mu = self.mu_from_state(parts[1, n, :] if D > 0 else np.zeros(0), t=0)
            lw[n] = gev_logpdf(self.y[0], mu, self.sigma, self.xi)

        # ancestor sampling at t=1
        if D > 0:
            logf = np.array([self._transition_logpdf(parts[0, j, :], parts[1, N - 1, :])
                             for j in range(N)], float)
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
                    parts[t, n, :] = self._transition_sample(prev)

            # reference path + ancestor sampling
            if D > 0:
                x_ref_t = self.x[t]
                parts[t, N - 1, :] = x_ref_t.copy()
                logw_prev = np.log(np.clip(w[t - 1, :], 1e-300, None))
                logf = np.array([self._transition_logpdf(parts[t - 1, j, :], x_ref_t)
                                 for j in range(N)], float)
                log_post = logw_prev + logf
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
            ); pos += 1
        if self.idx_beta is not None:
            self.m0_beta = self._gibbs_m0_scalar(
                float(self.x[0, pos]), self.priors.m_m0_beta, self.priors.s_m0_beta, self.P0_beta
            ); pos += 1
        if self.seasonal_mode == "dynamic":
            s0 = float(self.priors.s_m0_harm)
            m_cos = (np.zeros(self.K) if self.priors.m_m0_cos is None
                     else np.asarray(self.priors.m_m0_cos, float))
            m_sin = (np.zeros(self.K) if self.priors.m_m0_sin is None
                     else np.asarray(self.priors.m_m0_sin, float))
            if m_cos.size != self.K: m_cos = np.zeros(self.K)
            if m_sin.size != self.K: m_sin = np.zeros(self.K)
            for k in range(self.K):
                self.m0_cos[k] = self._gibbs_m0_scalar(
                    float(self.x[0, pos + 2*k    ]), float(m_cos[k]), s0, self.P0_harm
                )
                self.m0_sin[k] = self._gibbs_m0_scalar(
                    float(self.x[0, pos + 2*k + 1]), float(m_sin[k]), s0, self.P0_harm
                )
            if self.use_nyq:
                j = pos + 2*self.K
                self.m0_nyq = self._gibbs_m0_scalar(
                    float(self.x[0, j]), float(self.priors.m_m0_nyq), s0, self.P0_harm
                )

    def update_P0(self) -> None:
        if self.dim == 0:
            return
        pos = 0
        if self.idx_alpha is not None:
            a = self.priors.a_P0_alpha + 0.5
            b = self.priors.b_P0_alpha + 0.5 * (float(self.x[0, pos]) - self.m0_alpha) ** 2
            self.P0_alpha = 1.0 / np.random.gamma(shape=a, scale=1.0 / b); pos += 1
        if self.idx_beta is not None:
            a = self.priors.a_P0_beta + 0.5
            b = self.priors.b_P0_beta + 0.5 * (float(self.x[0, pos]) - self.m0_beta) ** 2
            self.P0_beta = 1.0 / np.random.gamma(shape=a, scale=1.0 / b); pos += 1
        if self.seasonal_mode == "dynamic":
            diffsq = 0.0
            Ktot = 2*self.K + (1 if self.use_nyq else 0)
            target = [*self.m0_cos, *self.m0_sin] + ([float(self.m0_nyq)] if self.use_nyq else [])
            for k in range(Ktot):
                diffsq += (float(self.x[0, pos + k]) - float(target[k])) ** 2
            a = self.priors.a_P0_harm + 0.5 * Ktot
            b = self.priors.b_P0_harm + 0.5 * diffsq
            self.P0_harm = 1.0 / np.random.gamma(shape=a, scale=1.0 / b)

    # ---------------- Wrapper for deterministic params ------------------ #
    def update_deterministic_params(self) -> None:
        if self.level_mode == "deterministic":
            self.update_level_value()
        if self.trend_mode == "deterministic":
            self.update_slope()
        if self.seasonal_mode == "deterministic":
            self.update_season_det()

    # --------------------------- Progress helpers --------------------------- #
    @staticmethod
    def _fmt_list(vals, max_elems: int = 6, fmt: str = ".4g") -> str:
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

        parts = [f"[it {it + 1}/{self.cfg.n_iter}]",
                 f"logZ={self.last_log_evidence:.3f}",
                 f"σ={math.exp(self.logsigma):.3f} ({pct('logsigma')})",
                 f"ξ={self.xi:.3f} ({pct('xi')})"]

        if self.idx_alpha is not None:
            parts.append(f"Qα={self.s_alpha**2:.4g}")
        if self.idx_beta is not None:
            parts.append(f"Qβ={self.s_beta**2:.4g}")
        if self.seasonal_mode == "dynamic":
            parts.append(f"Qγ={self.s_gamma**2:.4g}")

        if self.level_mode == "dynamic":
            parts.append(f"m0α={self.m0_alpha:.4g} P0α={self.P0_alpha:.4g}")
        elif self.level_mode == "deterministic":
            parts.append(f"m0α={self.m0_alpha:.4g} P0α=0 ({pct('level')})")

        if self.trend_mode == "dynamic":
            parts.append(f"m0β={self.m0_beta:.4g} P0β={self.P0_beta:.4g}")
        elif self.trend_mode == "deterministic":
            parts.append(f"m0β={self.m0_beta:.4g} P0β=0 ({pct('slope')})")
        else:
            parts.append("m0β=n/a")

        if self.seasonal_mode != "none":
            parts.append(
                f"m0cos={self._fmt_list(self.m0_cos,6)} "
                f"m0sin={self._fmt_list(self.m0_sin,6)}" +
                (f" nyq={0.0 if (self.m0_nyq is None) else float(self.m0_nyq):.4g}"
                 if self.use_nyq else "")
            )
            if self.seasonal_mode == "dynamic":
                parts.append(f"P0harm={self.P0_harm:.4g}")
            else:
                parts.append(f"P0harm=0 ({pct('season')})")

        return " | ".join(parts)

    # --------------------------------- MCMC --------------------------------- #
    def run(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        save_iters = list(range(cfg.burn, cfg.n_iter, cfg.thin))
        n_kept, keep_idx = len(save_iters), 0

        self.keep = {
            "sigma": np.zeros(n_kept, float),
            "xi": np.zeros(n_kept, float),
            "mu": np.zeros((n_kept, self.T), float),
            "log_evidence": np.zeros(n_kept, float),
        }
        if self.idx_alpha is not None:
            self.keep.update({
                "Q_alpha": np.zeros(n_kept),
                "m0_alpha": np.zeros(n_kept),
                "P0_alpha": np.zeros(n_kept),
            })
        if self.idx_beta is not None:
            self.keep.update({
                "Q_beta": np.zeros(n_kept),
                "m0_beta": np.zeros(n_kept),
                "P0_beta": np.zeros(n_kept),
            })
        if self.seasonal_mode == "dynamic":
            self.keep.update({
                "Q_gamma": np.zeros(n_kept),
                "m0_cos": np.zeros((n_kept, self.K)),
                "m0_sin": np.zeros((n_kept, self.K)),
                "m0_nyq": np.zeros(n_kept) if self.use_nyq else np.zeros(0),
                "P0_harm": np.zeros(n_kept),
                "x": np.zeros((n_kept, self.T, self.dim)) if self.dim > 0 else np.zeros((0,0,0)),
            })
        else:
            self.keep.update({
                "m0_cos": np.zeros((n_kept, self.K)),
                "m0_sin": np.zeros((n_kept, self.K)),
                "m0_nyq": np.zeros(n_kept) if self.use_nyq else np.zeros(0),
            })
            if self.dim > 0:
                self.keep["x"] = np.zeros((n_kept, self.T, self.dim))
        if self.level_mode == "deterministic":
            self.keep["m0_alpha_det"] = np.zeros(n_kept)
        if self.trend_mode == "deterministic":
            self.keep["m0_beta_det"] = np.zeros(n_kept)

        print_every = cfg.progress_every if cfg.progress_every > 0 else max(1, cfg.n_iter // 50) or 1

        for it in range(cfg.n_iter):
            if cfg.progress:
                print(f"Iteration {it + 1}/{cfg.n_iter}")

            # 1) latent states via PGAS
            if self.dim > 0:
                self.update_states_pgas()
                current_log_ev = float(self.last_log_evidence)
            else:
                mu_vec_now = self._mu_vec_current()
                current_log_ev = float(gev_loglike_sum(self.y, mu_vec_now, self.sigma, self.xi))

            # 2) process SDs (slice on ln s)
            if self.dim > 0:
                self.update_process_s_lognormal_slice()

            # 3) m0 and 4) P0 for dynamic coords
            if self.dim > 0:
                self.update_m0()
                self.update_P0()

            # 5) deterministic structural parameters
            self.update_deterministic_params()

            # 6) observation parameters
            self.update_logsigma()
            self.update_xi()

            # 7) adapt MH step sizes
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
                    self.keep["m0_cos"][keep_idx, :] = self.m0_cos
                    self.keep["m0_sin"][keep_idx, :] = self.m0_sin
                    if self.use_nyq:
                        self.keep["m0_nyq"][keep_idx] = 0.0 if (self.m0_nyq is None) else float(self.m0_nyq)
                    self.keep["P0_harm"][keep_idx] = self.P0_harm
                else:
                    self.keep["m0_cos"][keep_idx, :] = self.m0_cos
                    self.keep["m0_sin"][keep_idx, :] = self.m0_sin
                    if self.use_nyq:
                        self.keep["m0_nyq"][keep_idx] = 0.0 if (self.m0_nyq is None) else float(self.m0_nyq)
                if "x" in self.keep and self.dim > 0:
                    self.keep["x"][keep_idx, :, :] = self.x[1 : self.T + 1, :]
                if self.level_mode == "deterministic":
                    self.keep["m0_alpha_det"][keep_idx] = self.m0_alpha
                if self.trend_mode == "deterministic":
                    self.keep["m0_beta_det"][keep_idx] = self.m0_beta
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

        if self.true_mu_t is not None:    arrays["true_mu_t"] = np.asarray(self.true_mu_t, float)
        if self.true_alpha_t is not None: arrays["true_alpha_t"] = np.asarray(self.true_alpha_t, float)
        if self.true_beta_t is not None:  arrays["true_beta_t"] = np.asarray(self.true_beta_t, float)
        if self.true_gamma_t is not None: arrays["true_gamma_t"] = np.asarray(self.true_gamma_t, float)

        np.savez_compressed(out_npz_path, **arrays)

        meta = {
            "T": int(self.T),
            "dim": int(self.dim),
            "period": int(self.s),
            "harmonics": int(self.K),
            "use_nyquist": bool(self.use_nyq),
            "modes": {
                "level_mode": self.level_mode,
                "trend_mode": self.trend_mode,
                "seasonal_mode": self.seasonal_mode,
            },
            "layout": list(self._layout),
            "cfg": asdict(self.cfg),
            "priors": asdict(self.priors),
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

# ------------------------- CLI / Example run ------------------------------- #
if __name__ == "__main__":
    import argparse
    from datetime import datetime

    import sys, os, time, math
    import numpy as np
    import matplotlib.pyplot as plt

    # make project root importable
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(base_dir)

    # harmonic extremal simulator
    from simulator.extremal_time_series_harmonic import Extremal_Time_Series

    # harmonic helper utilities (for projecting dummies → harmonics)
    from optimization.harmonic_helpers import (
        center_and_report_dummies_full,
        dummies_full_to_harmonics_fft,
    )

    # ---------- small helpers ------------------------------------------------
    def _parse_date(s: str | None):
        if not s:
            from datetime import datetime as _dt
            return _dt.today()
        parts = [int(p) for p in s.split("-")]
        if   len(parts) == 1: return datetime(parts[0], 1, 1)
        elif len(parts) == 2: return datetime(parts[0], parts[1], 1)
        elif len(parts) == 3: return datetime(parts[0], parts[1], parts[2])
        raise ValueError("start-date must be YYYY, YYYY-MM, or YYYY-MM-DD")

    def _parse_csv_maybe(s: Optional[str]) -> Optional[List[float]]:
        if s is None:
            return None
        s = s.strip()
        if s == "":
            return None
        return [float(z) for z in s.split(",")]

    # ---------- CLI ----------------------------------------------------------
    p = argparse.ArgumentParser(description="MCMC inference for seasonal extreme value time series using PGAS.")

    # ----- Simulation controls (truth) --------------------------------------
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--period", type=int, default=4)
    p.add_argument("--start-date", type=str, default="2000-01-01")

    p.add_argument("--sigma", type=float, default=4.0)
    p.add_argument("--xi",    type=float, default=-0.1)

    p.add_argument("--level-mode",   choices=["dynamic", "deterministic"],         default="dynamic")
    p.add_argument("--trend-mode",   choices=["dynamic", "deterministic", "none"], default="dynamic")
    p.add_argument("--seasonal-mode",choices=["dynamic", "deterministic", "none"], default="dynamic")

    p.add_argument("--q-level",  type=float, default=0.1)
    p.add_argument("--q-trend",  type=float, default=0.000002)
    p.add_argument("--q-season", type=float, default=0.05,
                   help="One scalar seasonal process variance (simulator, dynamic mode only)")

    # truth priors for level/trend
    p.add_argument("--m0-level", type=float, default=5.0)
    p.add_argument("--v0-level", type=float, default=0.25)
    p.add_argument("--m0-trend", type=float, default=0.02)
    p.add_argument("--v0-trend", type=float, default=0.05)

    # ----- Harmonic spec (simulator + sampler) ------------------------------
    p.add_argument("--harmonics", type=int, default=None,
                   help="Number of harmonics K; None = full floor((period-1)/2).")
    p.add_argument("--use-nyquist", type=int, default=1,
                   help="1/0; None/1 = allow Nyquist when period is even and K allows it.")

    # ----- Option A (simulator) : harmonic seasonal truth -------------------
    p.add_argument("--sim-m0-cos", type=str, default=None,
                   help="CSV length K of cosine coefficients for simulator (Option A).")
    p.add_argument("--sim-m0-sin", type=str, default=None,
                   help="CSV length K of sine coefficients for simulator (Option A).")
    p.add_argument("--sim-m0-nyq", type=float, default=0.0,
                   help="Nyquist coefficient for simulator (if used).")

    # ----- Option B (sim + sampler init) : full seasonal dummies ------------
    p.add_argument("--season-dummies", type=str, default="1,1,1,-3",
                   help="CSV of length = period representing a full seasonal pattern.")

    # ----- Sampler configuration (MCMC + PGAS) -----------------------------
    p.add_argument("--n-iter", type=int, default=4000)
    p.add_argument("--burn",   type=int, default=1000)
    p.add_argument("--thin",   type=int, default=1)
    p.add_argument("--seed",   type=int, default=7)

    p.add_argument("--progress",        type=int, default=1)
    p.add_argument("--progress-every",  type=int, default=1)

    # PGAS / PF options
    p.add_argument("--particles",  type=int,   default=200)
    p.add_argument("--trans-eps",  type=float, default=1e-8)
    p.add_argument("--ess-frac",   type=float, default=0.5,
                   help="Resample when ESS < ess_frac * N")

    # RW–MH steps (obs + deterministic structural params)
    p.add_argument("--step-logsigma", type=float, default=0.2)
    p.add_argument("--step-xi",       type=float, default=0.1)
    p.add_argument("--step-level",    type=float, default=0.2)
    p.add_argument("--step-slope",    type=float, default=0.001)
    p.add_argument("--step-season",   type=float, default=0.02)

    # Adaptive RW–MH
    p.add_argument("--adapt-steps",     type=int,   default=1)
    p.add_argument("--adapt-every",     type=int,   default=20)
    p.add_argument("--adapt-until",     choices=["burn", "all"], default="burn")
    p.add_argument("--adapt-eta0",      type=float, default=0.2)
    p.add_argument("--adapt-decay",     type=float, default=0.75)
    p.add_argument("--adapt-target-1d", type=float, default=0.44)
    p.add_argument("--step-min",        type=float, default=1e-5)
    p.add_argument("--step-max",        type=float, default=1.0)

    # Slice sampler (for ln s)
    p.add_argument("--slice-w", type=float, default=1.0)
    p.add_argument("--slice-m", type=int,   default=10)

    # ----- Observation priors ----------------------------------------------
    p.add_argument("--prior-a-sigma", type=float, default=2.0,
                   help="Shape a for InvGamma prior on σ².")
    p.add_argument("--prior-b-sigma", type=float, default=2.0,
                   help="Scale b for InvGamma prior on σ².")

    # ξ ~ Uniform[lower, upper]
    p.add_argument("--prior-xi-lower", type=float, default=-0.5)
    p.add_argument("--prior-xi-upper", type=float, default=0.5)

    # ----- m0 priors (level/trend + harmonic coefficients) -----------------
    p.add_argument("--prior-m-m0-alpha", type=float, default=0.0)
    p.add_argument("--prior-s-m0-alpha", type=float, default=10.0)
    p.add_argument("--prior-m-m0-beta",  type=float, default=0.0)
    p.add_argument("--prior-s-m0-beta",  type=float, default=10.0)

    p.add_argument("--prior-m-m0-cos", type=str, default=None,
                   help="CSV length K for prior mean of harmonic cos coefficients.")
    p.add_argument("--prior-m-m0-sin", type=str, default=None,
                   help="CSV length K for prior mean of harmonic sin coefficients.")
    p.add_argument("--prior-m-m0-nyq", type=float, default=0.0)
    p.add_argument("--prior-s-m0-harm", type=float, default=5.0)

    # ----- P0 priors (initial state variances) -----------------------------
    p.add_argument("--prior-a-P0-alpha", type=float, default=2.0)
    p.add_argument("--prior-b-P0-alpha", type=float, default=1.0)
    p.add_argument("--prior-a-P0-beta",  type=float, default=2.0)
    p.add_argument("--prior-b-P0-beta",  type=float, default=1.0)
    p.add_argument("--prior-a-P0-harm",  type=float, default=2.0)
    p.add_argument("--prior-b-P0-harm",  type=float, default=1.0)

    # ----- Log-normal priors on process SDs ln s_* -------------------------
    p.add_argument("--ln-s-alpha-mu", type=float, default=-1.0)
    p.add_argument("--ln-s-alpha-sd", type=float, default=2.0)
    p.add_argument("--ln-s-beta-mu",  type=float, default=-1.0)
    p.add_argument("--ln-s-beta-sd",  type=float, default=2.0)
    p.add_argument("--ln-s-gamma-mu", type=float, default=-1.0)
    p.add_argument("--ln-s-gamma-sd", type=float, default=2.0)

    # ----- Initial inference values ----------------------------------------
    p.add_argument("--sigma-init", type=float, default=1.0)
    p.add_argument("--xi-init",    type=float, default=-0.01)

    p.add_argument("--s-alpha-init", type=float, default=1e-1)
    p.add_argument("--s-beta-init",  type=float, default=1e-1)
    p.add_argument("--s-gamma-init", type=float, default=1e-1)

    p.add_argument("--P0-alpha-init", type=float, default=0.25)
    p.add_argument("--P0-beta-init",  type=float, default=0.05)
    p.add_argument("--P0-harm-init",  type=float, default=0.25)

    p.add_argument("--m0-cos-init", type=str, default=None,
                   help="CSV length K for sampler initial cos coefficients; "
                        "if None and season-dummies given, CLI projects them.")
    p.add_argument("--m0-sin-init", type=str, default=None,
                   help="CSV length K for sampler initial sin coefficients; "
                        "if None and season-dummies given, CLI projects them.")
    p.add_argument("--m0-nyq-init", type=float, default=None,
                   help="Sampler initial Nyquist coeff; "
                        "if None and dummies given (and Nyquist used), CLI projects them.")

    # ----- Output / UX ------------------------------------------------------
    p.add_argument("--out-dir", type=str, default="results/simulations/DGEV_harm")
    p.add_argument("--plot", type=int, default=1)
    p.add_argument("--print-summary", type=int, default=1)

    args = p.parse_args()
    np.random.seed(args.seed)

    # ----- resolve harmonics / Nyquist -------------------------------------
    if args.harmonics is None:
        args.harmonics = (args.period - 1) // 2
    use_nyq = bool(int(args.use_nyquist))

    # ----- parse list-type CLI inputs --------------------------------------
    sim_m0_cos  = _parse_csv_maybe(args.sim_m0_cos)
    sim_m0_sin  = _parse_csv_maybe(args.sim_m0_sin)
    pri_m_cos   = _parse_csv_maybe(args.prior_m_m0_cos)
    pri_m_sin   = _parse_csv_maybe(args.prior_m_m0_sin)
    m0_cos_init = _parse_csv_maybe(args.m0_cos_init)
    m0_sin_init = _parse_csv_maybe(args.m0_sin_init)
    season_dummies = _parse_csv_maybe(args.season_dummies)

    # ----- build simulator (Extremal_Time_Series with harmonics) ----------- #
    ts = Extremal_Time_Series(
        parameters=(args.sigma, args.xi),
        level_mode=args.level_mode,
        trend_mode=args.trend_mode,
        seasonal_mode=args.seasonal_mode,
        period=args.period,
        season_harmonics=args.harmonics,
        season_use_nyquist=use_nyq,
        q_level=(args.q_level  if args.level_mode   == "dynamic" else 0.0),
        q_trend=(args.q_trend  if args.trend_mode   == "dynamic" else 0.0),
        q_season=(args.q_season if args.seasonal_mode == "dynamic" else 0.0),
        m0_level=args.m0_level, v0_level=args.v0_level,
        m0_trend=(0.0 if args.trend_mode == "none" else args.m0_trend),
        v0_trend=args.v0_trend,
        # Option A: explicit harmonics for simulator
        m0_cos=sim_m0_cos,
        m0_sin=sim_m0_sin,
        m0_nyq=args.sim_m0_nyq,
        # Option B: dummies (simulator will center + project internally if needed)
        season_dummies=season_dummies,
        start_date=_parse_date(args.start_date),
        rng=np.random.default_rng(args.seed),
    )

    # simulate T observations
    y = np.array([ts.move() or ts.measure() for _ in range(args.T)], float)
    truth = ts.get_truth_paths(as_numpy=True)

    mu_T    = truth["mu"][1 : 1 + args.T]
    alpha_T = truth["alpha"][1 : 1 + args.T]
    beta_T  = truth["beta"][1 : 1 + args.T]
    gamma_T = truth["gamma"][1 : 1 + args.T]
    dates_T = truth["index"][:args.T]

    # ----- Priors -----------------------------------------------------------
    priors = Priors(
        a_sigma=float(args.prior_a_sigma),
        b_sigma=float(args.prior_b_sigma),
        xi_lower=float(args.prior_xi_lower),
        xi_upper=float(args.prior_xi_upper),

        m_m0_alpha=float(args.prior_m_m0_alpha),
        s_m0_alpha=float(args.prior_s_m0_alpha),
        m_m0_beta=float(args.prior_m_m0_beta),
        s_m0_beta=float(args.prior_s_m0_beta),

        m_m0_cos=None if pri_m_cos is None else pri_m_cos,
        m_m0_sin=None if pri_m_sin is None else pri_m_sin,
        m_m0_nyq=float(args.prior_m_m0_nyq),
        s_m0_harm=float(args.prior_s_m0_harm),

        a_P0_alpha=float(args.prior_a_P0_alpha),
        b_P0_alpha=float(args.prior_b_P0_alpha),
        a_P0_beta=float(args.prior_a_P0_beta),
        b_P0_beta=float(args.prior_b_P0_beta),
        a_P0_harm=float(args.prior_a_P0_harm),
        b_P0_harm=float(args.prior_b_P0_harm),

        mu_log_s_alpha=float(args.ln_s_alpha_mu),
        sd_log_s_alpha=float(args.ln_s_alpha_sd),
        mu_log_s_beta=float(args.ln_s_beta_mu),
        sd_log_s_beta=float(args.ln_s_beta_sd),
        mu_log_s_gamma=float(args.ln_s_gamma_mu),
        sd_log_s_gamma=float(args.ln_s_gamma_sd),
    )

    # For deterministic level: re-center its prior around the data (helps mixing)
    if args.level_mode == "deterministic":
        priors.m_m0_alpha = float(np.median(y))
        priors.s_m0_alpha = max(2.0, 0.5 * y.std(ddof=1))

    # ----- Sampler config ---------------------------------------------------
    cfg = SamplerConfig(
        n_iter=int(args.n_iter),
        burn=int(args.burn),
        thin=int(args.thin),
        random_seed=int(args.seed),

        progress=bool(args.progress),
        progress_every=int(args.progress_every),

        n_particles=int(args.particles),
        trans_eps=float(args.trans_eps),
        ess_threshold_frac=float(args.ess_frac),

        step_logsigma=float(args.step_logsigma),
        step_xi=float(args.step_xi),
        step_level=float(args.step_level),
        step_slope=float(args.step_slope),
        step_season=float(args.step_season),

        adapt_steps=bool(args.adapt_steps),
        adapt_every=int(args.adapt_every),
        adapt_until=str(args.adapt_until),
        adapt_eta0=float(args.adapt_eta0),
        adapt_eta_decay=float(args.adapt_decay),
        adapt_target_1d=float(args.adapt_target_1d),
        step_min=float(args.step_min),
        step_max=float(args.step_max),

        slice_w=float(args.slice_w),
        slice_m=int(args.slice_m),
    )

    # ----- derive sampler seasonal inits from dummies if needed -------------
    if (m0_cos_init is None or m0_sin_init is None) and (season_dummies is not None):
        if len(season_dummies) != args.period:
            raise ValueError(f"--season-dummies must have length period={args.period}")
        centered = center_and_report_dummies_full(season_dummies, tol=1e-12)
        K = args.harmonics
        cos_coefs, sin_coefs, nyq_val = dummies_full_to_harmonics_fft(
            centered, K=K, use_nyquist=use_nyq
        )
        if m0_cos_init is None:
            m0_cos_init = list(cos_coefs)
        if m0_sin_init is None:
            m0_sin_init = list(sin_coefs)
        if args.m0_nyq_init is None and use_nyq:
            args.m0_nyq_init = float(0.0 if nyq_val is None else nyq_val)

    # ensure we always have arrays for sampler
    if m0_cos_init is None:
        m0_cos_init = [0.0] * args.harmonics
    if m0_sin_init is None:
        m0_sin_init = [0.0] * args.harmonics
    if args.m0_nyq_init is None:
        args.m0_nyq_init = 0.0

    # ----- build sampler ----------------------------------------------------
    sampler = DGEVParticleGibbs(
        y=y,
        period=args.period,
        harmonics=args.harmonics,
        use_nyquist=use_nyq,
        level_mode=args.level_mode,
        trend_mode=args.trend_mode,
        seasonal_mode=args.seasonal_mode,
        # x0 priors
        m0_alpha_init=args.m0_level,
        P0_alpha_init=args.P0_alpha_init,
        m0_beta_init=(0.0 if args.trend_mode == "none" else args.m0_trend),
        P0_beta_init=args.P0_beta_init,
        m0_cos_init=m0_cos_init,
        m0_sin_init=m0_sin_init,
        m0_nyq_init=args.m0_nyq_init,
        P0_harm_init=args.P0_harm_init,
        # observation init
        sigma_init=args.sigma_init,
        xi_init=args.xi_init,
        # process SD init
        s_alpha_init=args.s_alpha_init,
        s_beta_init=args.s_beta_init,
        s_gamma_init=args.s_gamma_init,
        priors=priors,
        cfg=cfg,
    )

    # truth overlays
    true_Q = []
    if args.level_mode == "dynamic":
        true_Q.append(args.q_level)
    if args.trend_mode == "dynamic":
        true_Q.append(args.q_trend)
    if args.seasonal_mode == "dynamic":
        true_Q.append(args.q_season)

    sampler.set_truth(
        sigma=args.sigma,
        xi=args.xi,
        Q=(np.asarray(true_Q, float) if len(true_Q) else None),
    )
    sampler.set_truth_paths(mu=mu_T, alpha=alpha_T, beta=beta_T, gamma=gamma_T)

    # ----- summary of simulation / setup ------------------------------------
    if args.print_summary:
        with np.printoptions(suppress=True, precision=4):
            print("\n--- Simulation truth ---")
            print(f"modes: level={args.level_mode}, trend={args.trend_mode}, season={args.seasonal_mode}")
            print(f"sigma={args.sigma}, xi={args.xi}")
            print(
                f"q_level={args.q_level if args.level_mode=='dynamic' else 0.0}, "
                f"q_trend={args.q_trend if args.trend_mode=='dynamic' else 0.0}, "
                f"q_season={args.q_season if args.seasonal_mode=='dynamic' else 0.0}"
            )
            print(f"m0_level(sim)={args.m0_level}, v0_level(sim)={args.v0_level}")
            print(f"m0_trend(sim)={args.m0_trend}, v0_trend(sim)={args.v0_trend}")
            print(f"period={args.period}, start={dates_T[0]}, end={dates_T[-1]}")
            print(f"y mean={y.mean():.3f}, sd={y.std(ddof=1):.3f}")
            print(f"InvGamma prior on σ²: a={priors.a_sigma:.3g}, b={priors.b_sigma:.3g}")
            print(f"ξ ~ Uniform[{priors.xi_lower}, {priors.xi_upper}]")
            print(f"Harmonics: K={sampler.K}, Nyquist={sampler.use_nyq}")
            print(
                f"LN priors (ln s): "
                f"α~N({priors.mu_log_s_alpha:.2f},{priors.sd_log_s_alpha:.2f}²), "
                f"β~N({priors.mu_log_s_beta:.2f},{priors.sd_log_s_beta:.2f}²), "
                f"γ~N({priors.mu_log_s_gamma:.2f},{priors.sd_log_s_gamma:.2f}²)"
            )

    # ----- run sampler ------------------------------------------------------
    t0 = time.time()
    post = sampler.run()
    elapsed = time.time() - t0
    print(f"[Run completed in {elapsed:.1f}s]")

    # ----- save posterior ---------------------------------------------------
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        args.out_dir,
        f"{args.level_mode}-{args.trend_mode}-{args.seasonal_mode}"
        f"_K{sampler.K}_nyq{int(bool(sampler.use_nyq))}_{stamp}",
    )
    os.makedirs(out_dir, exist_ok=True)
    sampler.save_posterior(
        out_npz_path=os.path.join(out_dir, "posterior.npz"),
        extra_meta={
            "elapsed_seconds": float(elapsed),
            "ln_s_priors": {
                "alpha": {"mu": priors.mu_log_s_alpha, "sd": priors.sd_log_s_alpha},
                "beta":  {"mu": priors.mu_log_s_beta,  "sd": priors.sd_log_s_beta},
                "gamma": {"mu": priors.mu_log_s_gamma, "sd": priors.sd_log_s_gamma},
            },
        },
    )

    # ----- quick posterior summary / plot -----------------------------------
    if args.print_summary:
        print("\n--- Posterior means ---")
        sig_mean = float(np.mean(post["sigma"])) if "sigma" in post else float("nan")
        xi_mean  = float(np.mean(post["xi"]))    if "xi"    in post else float("nan")
        print(f"σ = {sig_mean:.4g}")
        print(f"ξ = {xi_mean:.4g}")

        for k in ["alpha", "beta", "gamma"]:
            key = f"Q_{k}"
            if key in post:
                mQ = float(np.mean(post[key]))
                print(f"Q_{k} = {mQ:.4g} (√Q ≈ {math.sqrt(max(mQ, 0.0)):.4g})")

        if "log_evidence" in post and post["log_evidence"].size > 0:
            le = post["log_evidence"]
            print(
                f"log p(y|θ): mean={np.nanmean(le):.3f}, "
                f"median={np.nanmedian(le):.3f}, best={np.nanmax(le):.3f}"
            )

    if args.plot:
        mu_hat = post["mu"].mean(axis=0)
        plt.figure(figsize=(10, 4))
        plt.plot(dates_T, y, label="y_t", lw=1)
        plt.plot(dates_T, mu_T, "--", label="μ_t (truth)")
        plt.plot(dates_T, mu_hat, "-.", label="μ̂_t (post mean)")
        plt.title(
            f"DGEV harmonic PGAS: {args.level_mode}/{args.trend_mode}/{args.seasonal_mode} "
            f"| K={sampler.K}, nyq={sampler.use_nyq}"
        )
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
