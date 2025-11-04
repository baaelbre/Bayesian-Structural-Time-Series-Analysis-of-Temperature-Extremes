from __future__ import annotations

from tqdm import tqdm

"""
Particle Gibbs with Ancestor Sampling (PGAS) for a volatility DLM.

We model observations as
    y_t ~ Normal(mu_fixed, sigma_t^2),  with  log sigma_t = eta_eff_t,
where
    eta_eff_t = eta_det(t) + (alpha_t [+ first seasonal g_{t,1}])

The latent volatility state (when dynamic) follows a DLM with components
(level / trend / seasonal newest-first block), laying out the state as:
    x_t = [alpha_t] [beta_t] [g_{t,1}, ..., g_{t,p-1}]  (dim can be 0 as well).

This file implements a *drop-in* rewrite of the PF-within-Gibbs sampler
using a PGAS kernel to update the entire latent volatility path in one step,
conditional on the previous path. Deterministic pieces and PC priors for the
volatility process standard deviations are handled as in the original.

Main class: DLMVolatilityPGASSampler

Notes
-----
* Seasonal block is newest-first; observation loads the first seasonal coord.
* Ancestor sampling weights for the conditional particle at time t are
  proportional to w_{t-1}^i * q(x_t^\*, x_{t-1}^i), where q is the transition
  density N(A x_{t-1} + u, Q). See Lindsten, Jordan & Schön (2014).
* When there are no dynamic coordinates (dim = 0), we fall back to the exact
  deterministic likelihood and path.
"""

import json, math, os, time, warnings
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

# =============================================================================
# Small utils
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


# =============================================================================
# Priors & Config (mirrors dlm_location API)
# =============================================================================

@dataclass
class PCPrior:
    lambda_s: Optional[float] = None   # fixed if given
    a_lambda: float = 1.0              # shape
    b_lambda: float = 1.0              # rate
    frac: float = 0.10                 # λ init: u = frac * scale_proxy
    alpha_prob: float = 0.05           # P(s > u) = alpha_prob


@dataclass
class Priors:
    # m0 priors (for volatility state initial means and deterministic pieces)
    m_m0_alpha: float = 0.0
    s_m0_alpha: float = 10.0
    m_m0_beta:  float = 0.0
    s_m0_beta:  float = 10.0
    m_m0_gamma: Optional[Sequence[float]] = None  # length p-1 (NEWEST-FIRST)
    s_m0_gamma: float = 5.0

    # Initial-state variances P0 ~ InvGamma(a, b) for dynamic coords
    a_P0_alpha: float = 2.0
    b_P0_alpha: float = 1.0
    a_P0_beta:  float = 2.0
    b_P0_beta:  float = 1.0
    a_P0_gamma: float = 2.0
    b_P0_gamma: float = 1.0

    # PC priors (for volatility process sds)
    pc_alpha: PCPrior = field(default_factory=PCPrior)
    pc_beta:  PCPrior = field(default_factory=PCPrior)
    pc_gamma: PCPrior = field(default_factory=PCPrior)


@dataclass
class SamplerConfig:
    n_iter: int = 4000
    burn: int = 1000
    thin: int = 1
    random_seed: Optional[int] = 7
    progress: bool = True
    progress_every: int = 0  # 0 => ~2% of n_iter

    # Slice for log s
    slice_w: float = 0.4
    slice_m: int = 40
    slice_max_shrink: int = 1000

    # PGAS / Conditional SMC settings
    n_particles: int = 512
    ess_resample_ratio: float = 0.5  # resample when ESS < ratio*N (for *non*-conditional particles)
    resample_method: str = "systematic"  # "multinomial"|"systematic"|"stratified"


# =============================================================================
# DLM building blocks for volatility state
# =============================================================================

class DLMVolatilityPGASSampler:
    """
    PGAS sampler for the volatility (log-sd) DLM.

    Observation model:
        y_t | x_t  ~  N(mu_fixed, exp(2 * eta_eff_t))
        eta_eff_t = eta_det(t) + [alpha_t + 1_{seasonal-dyn} * g_{t,1}]

    State evolution (when dynamic):
        x_t = A x_{t-1} + u + e_t,   e_t ~ N(0, Q)

    Layout (NEWEST-FIRST seasonal block):
        x_t = [alpha] [beta] [g1 ... g_{p-1}]  (dimension can be 0)
    """

    # --------------------------- Construction --------------------------- #
    def __init__(
        self,
        y: np.ndarray,
        period: int,
        mu_fixed: float = 0.0,
        # volatility state modes (level can NEVER be 'none')
        level_mode: str = "dynamic",
        trend_mode: str = "dynamic",
        seasonal_mode: str = "dynamic",
        # initial values for volatility state prior
        m0_alpha_init: float = 0.0,
        P0_alpha_init: float = 0.25,
        m0_beta_init: float  = 0.0,
        P0_beta_init: float  = 0.05,
        m0_gamma_init: Optional[Sequence[float]] = None,
        P0_gamma_init: float = 0.25,
        # process sds for volatility state
        s_alpha_sig_init: float = 1e-2,
        s_beta_sig_init:  float = 1e-3,
        s_gamma_sig_init: float = 1e-3,
        priors: Priors = Priors(),
        cfg: SamplerConfig = SamplerConfig(),
    ):
        # Data
        self.y = np.asarray(y, float)
        self.T = int(self.y.size)
        self.period = int(period)
        self.mu_fixed = float(mu_fixed)
        if self.period < 2:
            raise ValueError("period must be >= 2")

        # Modes (volatility)
        if level_mode not in {"dynamic", "deterministic"}:
            raise ValueError("sigma-level can never be 'none'; use 'dynamic' or 'deterministic'.")
        if trend_mode not in {"dynamic", "deterministic", "none"}:
            raise ValueError("invalid trend_mode")
        if seasonal_mode not in {"dynamic", "deterministic", "none"}:
            raise ValueError("invalid seasonal_mode")
        if trend_mode == "dynamic" and level_mode != "dynamic":
            raise ValueError("dynamic trend requires dynamic level")

        self.level_mode, self.trend_mode, self.seasonal_mode = level_mode, trend_mode, seasonal_mode

        # Priors / cfg
        self.priors, self.cfg = priors, cfg
        if cfg.random_seed is not None:
            np.random.seed(cfg.random_seed)

        # Numerics
        self._eta_min = -6.0
        self._eta_max =  6.0
        self._eps = 1e-12

        # State layout for eta_t
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

        # Process sds for volatility state
        self.s_alpha_sig = float(s_alpha_sig_init) if self.idx_alpha is not None else 0.0
        self.s_beta_sig  = float(s_beta_sig_init)  if self.idx_beta  is not None else 0.0
        self.s_gamma_sig = float(s_gamma_sig_init) if self.seasonal_mode == "dynamic" else 0.0
        self.lambda_alpha_sig, self.lambda_beta_sig, self.lambda_gamma_sig = self._init_pc_lambdas_for_sig()

        # Initial m0 and P0 of x_0 (for dynamic coords)
        self.m0_alpha = float(m0_alpha_init) if self.idx_alpha is not None else 0.0
        self.P0_alpha = float(P0_alpha_init) if self.idx_alpha is not None else 0.0
        self.m0_beta  = float(m0_beta_init)  if self.idx_beta  is not None else 0.0
        self.P0_beta  = float(P0_beta_init)  if self.idx_beta  is not None else 0.0
        if self.seasonal_mode == "dynamic":
            if m0_gamma_init is None:
                self.m0_gamma = np.zeros(self.period - 1, float)
            else:
                g = np.asarray(m0_gamma_init, float)
                if g.size != self.period - 1:
                    raise ValueError("m0_gamma_init must have length p-1 (newest-first)")
                self.m0_gamma = g
            self.P0_gamma = float(P0_gamma_init)
        else:
            self.m0_gamma = None
            self.P0_gamma = 0.0

        # Deterministic contributions to eta_t
        if self.level_mode == "deterministic":
            self.m0_alpha = float(self.priors.m_m0_alpha)
        if self.trend_mode == "deterministic":
            self.m0_beta = float(self.priors.m_m0_beta)
        if self.seasonal_mode == "deterministic":
            base = (
                np.zeros(self.period - 1, float)
                if self.priors.m_m0_gamma is None
                else np.asarray(self.priors.m_m0_gamma, float)
            )
            if base.size != self.period - 1:
                raise ValueError("priors.m_m0_gamma must have length p-1")
            # full length-p seasonal with sum-zero (index by t % p)
            self.m0_gamma = np.r_[base, -float(np.sum(base))].astype(float)

        # Latent path placeholder (x/eta)
        self.eta_path = np.zeros((self.T + 1, max(1, self.dim)), float)

        # Storage
        self.keep: Dict[str, np.ndarray] = {}

        # Optional truths
        self.true_mu_t: Optional[np.ndarray] = None
        self.true_sigma_t: Optional[np.ndarray] = None

    # --------------------- Truth overlays (optional) --------------------- #
    def set_truth_paths(
        self,
        mu: Optional[np.ndarray] = None,
        sigma_t: Optional[np.ndarray] = None,
    ) -> None:
        self.true_mu_t = None if mu is None else np.asarray(mu, float)
        self.true_sigma_t = None if sigma_t is not None else None if sigma_t is None else np.asarray(sigma_t, float)

    # --------------------- PC λ init (volatility block) --------------------- #
    def _init_pc_lambdas_for_sig(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        y = self.y
        sd1 = _robust_sd(np.diff(y)) if y.size >= 2 else 0.0
        sd2 = _robust_sd(np.diff(y, n=2)) if y.size >= 3 else 0.0
        sdg = sd1

        def _cal(pc: PCPrior, sd: float) -> float:
            u = max(1e-12, pc.frac * max(1e-12, sd))
            return float(-math.log(max(1e-12, pc.alpha_prob)) / u)

        la = (float(self.priors.pc_alpha.lambda_s)
              if self.priors.pc_alpha.lambda_s is not None
              else (_cal(self.priors.pc_alpha, sd1) if self.idx_alpha is not None else None))
        lb = (float(self.priors.pc_beta.lambda_s)
              if self.priors.pc_beta.lambda_s is not None
              else (_cal(self.priors.pc_beta, sd2) if self.idx_beta is not None else None))
        lg = (float(self.priors.pc_gamma.lambda_s)
              if self.priors.pc_gamma.lambda_s is not None
              else (_cal(self.priors.pc_gamma, sdg) if self.seasonal_mode == "dynamic" else None))
        if self.cfg.progress:
            f = lambda x: "n/a" if x is None else f"{x:.4g}"
            print(f"[init] PC λ(sig): α={f(la)}, β={f(lb)}, γ={f(lg)}")
        return la, lb, lg

    # ----------------------------- Model pieces (for eta) ----------------------------- #
    def _A(self) -> np.ndarray:
        if self.dim == 0:
            return np.zeros((0, 0))
        A = np.eye(self.dim)
        if self.idx_alpha is not None and self.idx_beta is not None:
            A[self.idx_alpha, self.idx_beta] = 1.0
        if self.seasonal_mode == "dynamic":
            gs, ge = self.idx_g_start, self.idx_g_end
            K = ge - gs + 1
            A[gs, gs:ge+1] = -1.0
            A[gs+1:ge+1, gs:ge] = np.eye(K-1)
            A[gs+1:ge+1, ge] = 0.0
        return A

    def _u(self) -> np.ndarray:
        if self.dim == 0:
            return np.zeros(0, float)
        u = np.zeros(self.dim, float)
        if (self.idx_alpha is not None) and (self.trend_mode == "deterministic"):
            u[self.idx_alpha] = float(self.m0_beta)
        return u

    def _Q_sig(self) -> np.ndarray:
        if self.dim == 0:
            return np.zeros((0, 0))
        Q = np.zeros((self.dim, self.dim))
        if self.idx_alpha is not None and self.s_alpha_sig > 0:
            Q[self.idx_alpha, self.idx_alpha] = self.s_alpha_sig**2
        if self.idx_beta is not None and self.s_beta_sig > 0:
            Q[self.idx_beta, self.idx_beta] = self.s_beta_sig**2
        if self.seasonal_mode == "dynamic" and self.s_gamma_sig > 0:
            Q[self.idx_g_start, self.idx_g_start] = self.s_gamma_sig**2
        # tiny jitter for numerical stability
        for k in range(self.dim):
            if Q[k, k] == 0.0:
                Q[k, k] = 1e-12
        return Q

    def _current_m0_P0(self) -> Tuple[np.ndarray, np.ndarray]:
        m0, P0 = [], []
        if self.idx_alpha is not None:
            m0.append(self.m0_alpha); P0.append(self.P0_alpha)
        if self.idx_beta is not None:
            m0.append(self.m0_beta);  P0.append(self.P0_beta)
        if self.seasonal_mode == "dynamic":
            m0.extend(list(self.m0_gamma)); P0.extend([self.P0_gamma] * (self.period - 1))
        m0 = np.asarray(m0, float)
        P0 = np.asarray(P0, float)
        P0[P0 == 0.0] = 1e-12
        return m0, P0

    def _eta_det(self, t: int) -> float:
        out = 0.0
        if self.level_mode == "deterministic":
            out += self.m0_alpha
        if (self.trend_mode == "deterministic") and (self.idx_alpha is None):
            out += self.m0_beta * t
        if self.seasonal_mode == "deterministic":
            out += float(self.m0_gamma[t % self.period])
        return out

    # ----------------------------- Deterministic fast path ----------------------------- #
    def _deterministic_loglik_and_sigma(self) -> Tuple[float, np.ndarray]:
        eta_eff = np.array([self._eta_det(t) for t in range(self.T)], dtype=float)
        eta_eff = np.clip(eta_eff, self._eta_min, self._eta_max)
        sig = np.exp(eta_eff)
        resid = self.y - self.mu_fixed
        loglik = float(np.sum(-0.5*np.log(2*np.pi) - np.log(sig) - 0.5*(resid/sig)**2))
        return loglik, sig

    # ----------------------------- Weights & densities ----------------------------- #
    def _obs_loglik_vec(self, eta_eff: np.ndarray, y_t: float) -> np.ndarray:
        resid = y_t - self.mu_fixed
        return -0.5*math.log(2.0*math.pi) - eta_eff - 0.5 * (resid * np.exp(-eta_eff))**2

    def _gaussian_trans_logpdf(self, x_next: np.ndarray, mean: np.ndarray, Q: np.ndarray) -> np.ndarray:
        # Q is (approximately) diagonal after our construction – use diag shortcut
        diag = np.diag(Q)
        inv = 1.0 / np.maximum(diag, 1e-12)
        diff = x_next - mean
        quad = np.sum((diff**2) * inv, axis=1)
        logdet = np.sum(np.log(np.maximum(diag, 1e-12)))
        const = -0.5 * (self.dim * math.log(2.0*math.pi) + logdet)
        return const - 0.5 * quad

    def _normalize_logw(self, logw: np.ndarray) -> np.ndarray:
        m = np.nanmax(logw)
        w = np.exp(logw - m)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        s = w.sum()
        if not np.isfinite(s) or s <= 0.0:
            return np.full_like(w, 1.0 / max(1, w.size))
        return w / s

    def _resample(self, w: np.ndarray, method: str) -> np.ndarray:
        N = w.size
        if method == "multinomial":
            return np.random.choice(N, size=N, p=w)
        u0 = np.random.rand() / N
        if method == "stratified":
            u = (np.arange(N) + np.random.rand(N)) / N
        else:  # systematic
            u = u0 + np.arange(N) / N
        cdf = np.cumsum(w)
        idx = np.searchsorted(cdf, u, side="right")
        idx[idx == N] = N - 1
        return idx

    # ----------------------------- PGAS kernel ----------------------------- #
    def _pgas_update_path(self, x_ref: Optional[np.ndarray]) -> Tuple[np.ndarray, float]:
        """Run one conditional SMC pass with ancestor sampling, returning a
        new path x_{0:T} and an unbiased log-likelihood estimate for monitoring.

        If x_ref is None (first iteration), we run a standard bootstrap filter
        (no conditioning) but still do a backward draw as in PG.
        """
        # Deterministic fast path
        if self.dim == 0:
            path = np.zeros((self.T + 1, 1))
            for t in range(1, self.T + 1):
                path[t, 0] = self._eta_det(t - 1)
            loglik, _ = self._deterministic_loglik_and_sigma()
            return path, float(loglik)

        N = int(self.cfg.n_particles)
        A = self._A(); u = self._u(); Q = self._Q_sig()

        m0, P0 = self._current_m0_P0()
        # precompute for efficiency
        chol0 = np.sqrt(P0)

        parts: List[np.ndarray] = []      # x_t particles (N x dim)
        weights: List[np.ndarray] = []    # normalized weights at each t (N)
        ancestors: List[np.ndarray] = []  # ancestor indices (N)
        loglik_est = 0.0

        # Designate index for conditional particle at each time
        j_star = 0  # force the conditional/reference particle to be at slot 0

        # t = 0 (initial particles)
        x0 = np.random.randn(N, self.dim) * chol0 + m0
        if x_ref is not None:
            x0[j_star] = x_ref[0]
        w_prev = np.full(N, 1.0 / N)
        prev = x0

        for t in range(1, self.T + 1):
            y_t = float(self.y[t - 1])

            # Propagate *unconditional* proposal for all particles
            eps = np.random.multivariate_normal(np.zeros(self.dim), Q, size=N)
            mean = prev @ A.T + u.reshape(1, -1)
            x_t = mean + eps

            # Overwrite the conditional/reference particle with x_ref[t]
            if x_ref is not None:
                x_t[j_star] = x_ref[t]

            # Observation weights – only alpha and possibly first seasonal enter
            eta_dyn = x_t[:, self.idx_alpha] if self.idx_alpha is not None else np.zeros(N)
            if self.seasonal_mode == "dynamic":
                eta_dyn = eta_dyn + x_t[:, self.idx_g_start]
            eta_eff = np.clip(eta_dyn + self._eta_det(t - 1), self._eta_min, self._eta_max)
            logw_t = self._obs_loglik_vec(eta_eff, y_t)

            # Normalize weights
            w_t = self._normalize_logw(logw_t)

            # Incremental likelihood estimate (standard BF estimator)
            m = float(np.nanmax(logw_t))
            inc = m + math.log(max(np.sum(np.exp(logw_t - m) * w_prev), self._eps))
            loglik_est += inc

            # Ancestor sampling for the *conditional* particle (Lindsten et al.)
            if x_ref is not None:
                # weights proportional to w_{t-1}^i * p(x_t^* | x_{t-1}^i)
                mean_ref = prev @ A.T + u.reshape(1, -1)
                log_as = np.log(np.maximum(w_prev, self._eps)) + \
                         self._gaussian_trans_logpdf(x_ref[t][None, :].repeat(N, axis=0), mean_ref, Q)
                a_idx_star = np.random.choice(N, p=self._normalize_logw(log_as))
            else:
                # if no reference, just resample the slot j_star with standard resampling
                a_idx_star = None

            # Resample the *non-conditional* particles according to ESS
            ess = 1.0 / float(np.sum(w_t**2))
            if ess < self.cfg.ess_resample_ratio * N:
                idx = self._resample(w_t, self.cfg.resample_method)
                # enforce conditional particle stays at position j_star
                if x_ref is not None and a_idx_star is not None:
                    idx[j_star] = a_idx_star
                prev = x_t[idx, :]
                w_prev = np.full(N, 1.0 / N)
                ancestors.append(idx)
            else:
                prev = x_t
                w_prev = w_t
                ancestors.append(np.arange(N))

            parts.append(prev.copy())
            weights.append(w_prev.copy())

        # Backward sampling of a trajectory
        # Choose terminal index ~ weights[T]
        wT = np.nan_to_num(weights[-1], nan=0.0, posinf=0.0, neginf=0.0)
        wT = wT / (wT.sum() if wT.sum() > 0 else 1.0)
        idx = np.random.choice(N, p=wT)

        path = np.zeros((self.T + 1, self.dim))
        path[self.T] = parts[-1][idx]
        child = idx
        for t in range(self.T, 1, -1):
            anc = ancestors[t - 1]
            parent = anc[child]
            path[t - 1] = parts[t - 2][parent]
            child = parent
        # initial draw
        path[0] = np.random.randn(self.dim) * chol0 + m0

        return path, float(loglik_est)

    # -------------------- Slice for log s -------------------- #
    def _slice(self, f: Callable[[float], float], z0: float) -> float:
        w, m, limit = float(self.cfg.slice_w), int(self.cfg.slice_m), int(self.cfg.slice_max_shrink)
        y_star = f(z0) - np.random.exponential(1.0)
        u = np.random.rand()
        L = z0 - u * w
        R = L + w
        j = int(np.floor(m * np.random.rand()))
        k = (m - 1) - j
        while j > 0 and f(L) > y_star:
            L -= w; j -= 1
        while k > 0 and f(R) > y_star:
            R += w; k -= 1
        for _ in range(limit):
            z_prop = np.random.uniform(L, R)
            if f(z_prop) >= y_star:
                return z_prop
            if z_prop < z0: L = z_prop
            else:           R = z_prop
        return z0

    # ------------- Innovation sums of squares for volatility state ------------- #
    def _innovation_ss_alpha_sig(self, eta_path: np.ndarray) -> Tuple[float, int]:
        if self.idx_alpha is None: return 0.0, 0
        ss = 0.0
        for t in range(1, self.T + 1):
            drift = 0.0
            if self.idx_beta is not None:           # dynamic trend
                drift = eta_path[t - 1, self.idx_beta]
            elif self.trend_mode == "deterministic":
                drift = float(self.m0_beta)
            mean = eta_path[t - 1, self.idx_alpha] + drift
            ss += (eta_path[t, self.idx_alpha] - mean) ** 2
        return float(ss), self.T

    def _innovation_ss_beta_sig(self, eta_path: np.ndarray) -> Tuple[float, int]:
        if self.idx_beta is None: return 0.0, 0
        d = eta_path[1:, self.idx_beta] - eta_path[:-1, self.idx_beta]
        return float(np.sum(d * d)), self.T

    def _innovation_ss_gamma_sig(self, eta_path: np.ndarray) -> Tuple[float, int]:
        if self.seasonal_mode != "dynamic": return 0.0, 0
        gs, ge = self.idx_g_start, self.idx_g_end
        ss = 0.0
        for t in range(1, self.T + 1):
            prev = eta_path[t - 1, gs:ge + 1]
            mean_new_first = -float(np.sum(prev))
            ss += (eta_path[t, gs] - mean_new_first) ** 2
        return float(ss), self.T

    def _slice_logsd(self, z0: float, SS: float, T_eff: int, lam: float) -> float:
        def f(z: float) -> float:
            return -(T_eff * z) - 0.5 * SS * math.exp(-2 * z) - lam * math.exp(z) + z
        return self._slice(f, z0)

    # --- λ | s ~ Gamma(a+1, b+s)  (volatility block)
    def _gibbs_lambda_sig_single(self, which: str) -> None:
        if which == "alpha":
            pc = self.priors.pc_alpha
            if pc.lambda_s is not None or self.idx_alpha is None: return
            self.lambda_alpha_sig = float(
                np.random.gamma(shape=pc.a_lambda + 1.0, scale=1.0 / (pc.b_lambda + max(0.0, self.s_alpha_sig)))
            )
        elif which == "beta":
            pc = self.priors.pc_beta
            if pc.lambda_s is not None or self.idx_beta is None: return
            self.lambda_beta_sig = float(
                np.random.gamma(shape=pc.a_lambda + 1.0, scale=1.0 / (pc.b_lambda + max(0.0, self.s_beta_sig)))
            )
        elif which == "gamma":
            pc = self.priors.pc_gamma
            if pc.lambda_s is not None or self.seasonal_mode != "dynamic": return
            self.lambda_gamma_sig = float(
                np.random.gamma(shape=pc.a_lambda + 1.0, scale=1.0 / (pc.b_lambda + max(0.0, self.s_gamma_sig)))
            )

    def update_pc_lambdas_sig(self) -> None:
        self._gibbs_lambda_sig_single("alpha")
        self._gibbs_lambda_sig_single("beta")
        self._gibbs_lambda_sig_single("gamma")

    # --- m0 | P0, eta0  and  P0 | m0, eta0 (dynamic coords)
    @staticmethod
    def _gibbs_m0_scalar(x0: float, m_prior: float, s_prior: float, P0: float) -> float:
        prec = 1.0 / (s_prior**2) + 1.0 / max(1e-18, P0)
        var = 1.0 / prec
        mean = var * (m_prior / (s_prior**2) + x0 / max(1e-18, P0))
        return float(np.random.normal(mean, math.sqrt(var)))

    def update_m0_from_path(self, eta_path: np.ndarray) -> None:
        if self.dim == 0: return
        pos = 0
        if self.idx_alpha is not None:
            self.m0_alpha = self._gibbs_m0_scalar(
                float(eta_path[0, pos]), self.priors.m_m0_alpha, self.priors.s_m0_alpha, self.P0_alpha
            ); pos += 1
        if self.idx_beta is not None:
            self.m0_beta = self._gibbs_m0_scalar(
                float(eta_path[0, pos]), self.priors.m_m0_beta, self.priors.s_m0_beta, self.P0_beta
            ); pos += 1
        if self.seasonal_mode == "dynamic":
            m_prior = (np.zeros(self.period - 1, float)
                       if self.priors.m_m0_gamma is None
                       else np.asarray(self.priors.m_m0_gamma, float))
            if m_prior.size != self.period - 1:
                raise ValueError("priors.m_m0_gamma must be length p-1 (newest-first)")
            s = float(self.priors.s_m0_gamma)
            for k in range(self.period - 1):
                self.m0_gamma[k] = self._gibbs_m0_scalar(float(eta_path[0, pos + k]), float(m_prior[k]), s, self.P0_gamma)

    def update_P0_from_path(self, eta_path: np.ndarray) -> None:
        if self.dim == 0: return
        pos = 0
        if self.idx_alpha is not None:
            a = self.priors.a_P0_alpha + 0.5
            b = self.priors.b_P0_alpha + 0.5 * (float(eta_path[0, pos]) - self.m0_alpha) ** 2
            self.P0_alpha = 1.0 / np.random.gamma(shape=a, scale=1.0 / b); pos += 1
        if self.idx_beta is not None:
            a = self.priors.a_P0_beta + 0.5
            b = self.priors.b_P0_beta + 0.5 * (float(eta_path[0, pos]) - self.m0_beta) ** 2
            self.P0_beta = 1.0 / np.random.gamma(shape=a, scale=1.0 / b); pos += 1
        if self.seasonal_mode == "dynamic":
            diffsq = 0.0
            for k in range(self.period - 1):
                diffsq += (float(eta_path[0, pos + k]) - float(self.m0_gamma[k])) ** 2
            a = self.priors.a_P0_gamma + 0.5 * (self.period - 1)
            b = self.priors.b_P0_gamma + 0.5 * diffsq
            self.P0_gamma = 1.0 / np.random.gamma(shape=a, scale=1.0 / b)

    # ------------------- Deterministic η-parameters (slice / conjugate) ------------------- #
    def _loglik_given_eta_det_shift(self, base_eta_eff: np.ndarray, z: np.ndarray, theta: float) -> float:
        eta = np.clip(base_eta_eff + z * theta, self._eta_min, self._eta_max)
        sig = np.exp(eta)
        resid = self.y - self.mu_fixed
        return float(np.sum(-0.5*np.log(2*np.pi) - np.log(sig) - 0.5*(resid/sig)**2))

    def _slice_det_param(self, base_eta_eff: np.ndarray, z: np.ndarray,
                         theta0: float, m_prior: float, s_prior: float) -> float:
        def f(th: float) -> float:
            lp = self._loglik_given_eta_det_shift(base_eta_eff, z, th)
            lp += -0.5*math.log(2*math.pi) - math.log(max(1e-18, s_prior)) \
                  - 0.5*((th - m_prior)/max(1e-18, s_prior))**2
            return lp
        return self._slice(f, theta0)

    def update_deterministic_params(self, eta_path: Optional[np.ndarray]) -> None:
        # Build base η_eff(t) excluding the parameter currently updated
        dyn_alpha = np.zeros(self.T, float)
        dyn_seas  = np.zeros(self.T, float)
        if self.dim > 0 and eta_path is not None:
            if self.idx_alpha is not None:
                dyn_alpha = eta_path[1:self.T+1, self.idx_alpha]
            if self.seasonal_mode == "dynamic":
                dyn_seas = eta_path[1:self.T+1, self.idx_g_start]
        det = np.zeros(self.T, float)
        if self.seasonal_mode == "deterministic":
            det += self.m0_gamma[np.arange(self.T) % self.period]
        if (self.trend_mode == "deterministic") and (self.idx_alpha is None):
            det += self.m0_beta * np.arange(self.T, dtype=float)
        if self.level_mode == "deterministic":
            det += self.m0_alpha

        # m0_alpha (deterministic level of η)
        if self.level_mode == "deterministic":
            z = np.ones(self.T, float)
            base = dyn_alpha + dyn_seas + (det - self.m0_alpha)
            self.m0_alpha = self._slice_det_param(
                base_eta_eff=base, z=z,
                theta0=float(self.m0_alpha),
                m_prior=float(self.priors.m_m0_alpha), s_prior=float(self.priors.s_m0_alpha)
            )

        # m0_beta (deterministic trend)
        if self.trend_mode == "deterministic":
            if self.idx_alpha is not None:
                d = eta_path[1:, self.idx_alpha] - eta_path[:-1, self.idx_alpha]
                s2 = float(self.s_alpha_sig**2) if self.s_alpha_sig > 0 else 1e-12
                m0, s0 = float(self.priors.m_m0_beta), float(self.priors.s_m0_beta)
                prec = (self.T / s2) + 1.0 / (s0**2)
                mean = ((float(np.sum(d)) / s2) + m0 / (s0**2)) / prec
                var = 1.0 / prec
                self.m0_beta = float(np.random.normal(mean, math.sqrt(var)))
            else:
                t = np.arange(self.T, dtype=float)
                z = t
                base = dyn_alpha + dyn_seas + (det - self.m0_beta * t)
                self.m0_beta = self._slice_det_param(
                    base_eta_eff=base, z=z,
                    theta0=float(self.m0_beta),
                    m_prior=float(self.priors.m_m0_beta), s_prior=float(self.priors.s_m0_beta)
                )

        # m0_gamma (deterministic season) via contrast-wise slice
        if self.seasonal_mode == "deterministic":
            K = self.period - 1
            midx = np.arange(self.T) % self.period
            Z = np.zeros((self.T, K))
            for k in range(K):
                Z[:, k] = (midx == k).astype(float) - (midx == K).astype(float)

            theta = self.m0_gamma[:-1].copy()
            base = dyn_alpha + dyn_seas + (det - self.m0_gamma[np.arange(self.T) % self.period])
            m_prior = (np.zeros(K) if self.priors.m_m0_gamma is None
                       else np.asarray(self.priors.m_m0_gamma, float).reshape(-1))
            s0 = float(self.priors.s_m0_gamma)
            for k in range(K):
                base_k = base + Z[:, k] * theta[k]
                theta[k] = self._slice_det_param(
                    base_eta_eff=base_k, z=Z[:, k],
                    theta0=float(theta[k]),
                    m_prior=float(m_prior[k] if m_prior.size == K else 0.0),
                    s_prior=s0
                )
                base = base_k - Z[:, k] * theta[k]
            self.m0_gamma = np.r_[theta, -float(np.sum(theta))]

    # ------------------- Progress formatting ------------------- #
    @staticmethod
    def _fmt_list(vals, max_elems: int = 6, fmt: str = ".4g") -> str:
        if vals is None: return "/"
        v = np.asarray(vals, float).ravel()
        if v.size == 0: return "[]"
        if v.size <= max_elems:
            return "[" + ", ".join(f"{x:{fmt}}" for x in v) + "]"
        head = ", ".join(f"{x:{fmt}}" for x in v[:max_elems])
        return f"[{head}, …]"

    def _fmt_m0P0(self, which: str) -> str:
        if which == "alpha":
            if self.level_mode == "dynamic":
                return f"m0α={self.m0_alpha:.4g} P0α={self.P0_alpha:.4g}"
            elif self.level_mode == "deterministic":
                return f"m0α={self.m0_alpha:.4g} P0α=0"
            else:
                return "m0α=/ P0α=/"
        if which == "beta":
            if self.trend_mode == "dynamic":
                return f"m0β={self.m0_beta:.4g} P0β={self.P0_beta:.4g}"
            elif self.trend_mode == "deterministic":
                return f"m0β={self.m0_beta:.4g} P0β=0"
            else:
                return "m0β=/ P0β=/"
        if which == "gamma":
            if self.seasonal_mode == "dynamic":
                g = self._fmt_list(self.m0_gamma, 6, ".4g")
                return f"m0γ={g} P0γ={self.P0_gamma:.4g}"
            elif self.seasonal_mode == "deterministic":
                g = self._fmt_list(self.m0_gamma[:-1], 6, ".4g")
                return f"m0γ={g} P0γ=0"
            else:
                return "m0γ=/ P0γ=/"
        return ""

    def _fmt_lambda(self, lam, mode, fixed):
        if mode != "dynamic":
            return "/"
        if fixed is not None:
            return f"{fixed:.3g}(fix)"
        return "-" if lam is None else f"{lam:.3g}"

    def _progress_line(self, it: int, loglik: float) -> str:
        parts = [f"[it {it + 1}/{self.cfg.n_iter}]",
                 f"loglik≈{loglik:.3f}"]
        if self.idx_alpha is not None: parts.append(f"Qα(sig)={self.s_alpha_sig**2:.4g}")
        if self.idx_beta  is not None: parts.append(f"Qβ(sig)={self.s_beta_sig**2:.4g}")
        if self.seasonal_mode == "dynamic": parts.append(f"Qγ(sig)={self.s_gamma_sig**2:.4g}")

        parts.append("λ(sig)=(" + ",".join([
            self._fmt_lambda(self.lambda_alpha_sig, self.level_mode, self.priors.pc_alpha.lambda_s),
            self._fmt_lambda(self.lambda_beta_sig,  self.trend_mode, self.priors.pc_beta.lambda_s),
            self._fmt_lambda(self.lambda_gamma_sig, self.seasonal_mode, self.priors.pc_gamma.lambda_s),
        ]) + ")")

        parts.append(self._fmt_m0P0("alpha"))
        parts.append(self._fmt_m0P0("beta"))
        parts.append(self._fmt_m0P0("gamma"))
        return " | ".join(parts)

    # --------------------------------- MCMC --------------------------------- #
    def run(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        save_iters = list(range(cfg.burn, cfg.n_iter, cfg.thin))
        n_kept, keep_idx = len(save_iters), 0

        # allocate storage
        self.keep = {
            "mu_fixed": np.array([self.mu_fixed]),
            "loglik": np.zeros(n_kept, float),
            "sigma_path": np.zeros((n_kept, self.T), float),
        }
        if self.dim > 0:
            self.keep["eta"] = np.zeros((n_kept, self.T, self.dim))
        # volatility sds and lambdas
        if self.idx_alpha is not None:
            self.keep.update({"Q_alpha_sig": np.zeros(n_kept), "lambda_alpha_sig": np.zeros(n_kept),
                              "m0_alpha": np.zeros(n_kept), "P0_alpha": np.zeros(n_kept)})
        if self.idx_beta is not None:
            self.keep.update({"Q_beta_sig": np.zeros(n_kept), "lambda_beta_sig": np.zeros(n_kept),
                              "m0_beta": np.zeros(n_kept), "P0_beta": np.zeros(n_kept)})
        if self.seasonal_mode == "dynamic":
            self.keep.update({"Q_gamma_sig": np.zeros(n_kept), "lambda_gamma_sig": np.zeros(n_kept),
                              "m0_gamma": np.zeros((n_kept, self.period - 1)), "P0_gamma": np.zeros(n_kept)})

        print_every = cfg.progress_every if cfg.progress_every > 0 else max(1, cfg.n_iter // 50) or 1

        ref_path = None  # initial PGAS has no reference

        for it in range(cfg.n_iter):
            # 1) PGAS path update (returns x_{0:T} and loglik estimate)
            eta, loglik = self._pgas_update_path(ref_path)
            self.eta_path = eta
            ref_path = eta  # condition on the freshly sampled path next iteration

            # 2) process sds (slice under PC) + 3) λ (Gibbs) — only if dynamic
            if self.dim > 0:
                if self.idx_alpha is not None and (self.lambda_alpha_sig is not None):
                    ss, T_eff = self._innovation_ss_alpha_sig(eta)
                    z = self._slice_logsd(math.log(max(1e-18, self.s_alpha_sig)), ss, T_eff, float(self.lambda_alpha_sig))
                    self.s_alpha_sig = float(math.exp(z))
                if self.idx_beta is not None and (self.lambda_beta_sig is not None):
                    ss, T_eff = self._innovation_ss_beta_sig(eta)
                    z = self._slice_logsd(math.log(max(1e-18, self.s_beta_sig)), ss, T_eff, float(self.lambda_beta_sig))
                    self.s_beta_sig = float(math.exp(z))
                if self.seasonal_mode == "dynamic" and (self.lambda_gamma_sig is not None):
                    ss, T_eff = self._innovation_ss_gamma_sig(eta)
                    z = self._slice_logsd(math.log(max(1e-18, self.s_gamma_sig)), ss, T_eff, float(self.lambda_gamma_sig))
                    self.s_gamma_sig = float(math.exp(z))
                self.update_pc_lambdas_sig()

            # 4) m0 and 5) P0 (Gibbs from eta_0) — only for dynamic coords
            if self.dim > 0:
                self.update_m0_from_path(eta)
                self.update_P0_from_path(eta)

            # 6) deterministic η-parameters (slice/conjugate)
            self.update_deterministic_params(eta if self.dim > 0 else None)

            # progress
            if cfg.progress and ((it + 1) % print_every == 0 or it == cfg.n_iter - 1):
                print(self._progress_line(it, loglik))

            # save
            if it in save_iters:
                if self.dim > 0:
                    H_alpha = (self.idx_alpha is not None)
                    H_seas  = (self.seasonal_mode == "dynamic")
                    eta_eff_t = np.zeros(self.T, float)
                    for t in range(1, self.T + 1):
                        e = 0.0
                        if H_alpha: e += eta[t, self.idx_alpha]
                        if H_seas:  e += eta[t, self.idx_g_start]
                        e += self._eta_det(t - 1)
                        eta_eff_t[t - 1] = np.clip(e, self._eta_min, self._eta_max)
                    self.keep["sigma_path"][keep_idx, :] = np.exp(eta_eff_t)
                    self.keep["eta"][keep_idx, :, :] = eta[1:self.T+1, :]
                else:
                    _, sig = self._deterministic_loglik_and_sigma()
                    self.keep["sigma_path"][keep_idx, :] = sig
                self.keep["loglik"][keep_idx] = float(loglik)
                if self.idx_alpha is not None:
                    self.keep["Q_alpha_sig"][keep_idx] = self.s_alpha_sig**2
                    self.keep["lambda_alpha_sig"][keep_idx] = float(self.lambda_alpha_sig or 0.0)
                    self.keep["m0_alpha"][keep_idx] = self.m0_alpha
                    self.keep["P0_alpha"][keep_idx] = self.P0_alpha if self.level_mode == "dynamic" else 0.0
                if self.idx_beta is not None:
                    self.keep["Q_beta_sig"][keep_idx] = self.s_beta_sig**2
                    self.keep["lambda_beta_sig"][keep_idx] = float(self.lambda_beta_sig or 0.0)
                    self.keep["m0_beta"][keep_idx] = self.m0_beta
                    self.keep["P0_beta"][keep_idx] = self.P0_beta if self.trend_mode == "dynamic" else 0.0
                if self.seasonal_mode == "dynamic":
                    self.keep["Q_gamma_sig"][keep_idx] = self.s_gamma_sig**2
                    self.keep["lambda_gamma_sig"][keep_idx] = float(self.lambda_gamma_sig or 0.0)
                    self.keep["m0_gamma"][keep_idx, :] = self.m0_gamma
                    self.keep["P0_gamma"][keep_idx] = self.P0_gamma if self.seasonal_mode == "dynamic" else 0.0
                keep_idx += 1

        return self.keep

    # ------------------------------- Persistence ---------------------------- #
    def save_posterior(self, out_npz_path: str, extra_meta: Optional[dict] = None) -> None:
        _ensure_dir(os.path.dirname(out_npz_path))
        arrays = dict(self.keep)
        arrays["y"] = self.y.copy()
        if self.true_mu_t is not None:
            arrays["true_mu_t"] = np.asarray(self.true_mu_t, float)
        if self.true_sigma_t is not None:
            arrays["true_sigma_t"] = np.asarray(self.true_sigma_t, float)
        np.savez_compressed(out_npz_path, **arrays)

        meta = {
            "T": int(self.T),
            "dim": int(self.dim),
            "period": int(self.period),
            "mu_fixed": float(self.mu_fixed),
            "modes": {
                "vol_level_mode": self.level_mode,
                "vol_trend_mode": self.trend_mode,
                "vol_seasonal_mode": self.seasonal_mode,
            },
            "layout": list(self._layout),
            "cfg": asdict(self.cfg),
            "priors": asdict(self.priors),
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
    import argparse, sys

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(base_dir)

    # If you have your simulator, you can import similarly to the original file.
    from simulator.mean_time_series_volatility import Mean_Time_Series

    p = argparse.ArgumentParser(
        description=(
            "PGAS for stochastic/deterministic volatility DLM "
            "(level/trend/season on log-sd). Seasonal state is newest-first; "
            "observation loads the first seasonal coord of volatility."
        )
    )

    # --- Data / simulator controls ---
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--period", type=int, default=4)

    # μ-block for simulator: deterministic level only (constant mean), no trend/season
    p.add_argument("--mu-level", choices=["deterministic"], default="deterministic")
    p.add_argument("--mu-trend", choices=["none"], default="none")
    p.add_argument("--mu-season", choices=["none"], default="none")
    p.add_argument("--m0-level", type=float, default=0.0)

    # η-block for simulator: stochastic volatility
    p.add_argument("--m0-level-sigma", type=float, default=3.0)
    p.add_argument("--m0-trend-sigma", type=float, default=0.005)
    p.add_argument("--m0-season-sigma", type=list, default=None,
                   help="comma-separated list (length p-1, newest-first)")
    p.add_argument("--v0-level-sigma", type=float, default=0.1)
    p.add_argument("--v0-trend-sigma", type=float, default=0.01)
    p.add_argument("--v0-season-sigma", type=list, default=None,
                   help="comma-separated list (length p-1, newest-first)")

    # η (volatility) modes
    p.add_argument("--level-mode", choices=["dynamic", "deterministic"], default="deterministic")
    p.add_argument("--trend-mode", choices=["dynamic", "deterministic", "none"], default="deterministic")
    p.add_argument("--seasonal-mode", choices=["dynamic", "deterministic", "none"], default="deterministic")

    # Simulator innovations (η)
    p.add_argument("--q-level-sigma", type=float, default=0.05)
    p.add_argument("--q-trend-sigma", type=float, default=0.01)
    p.add_argument("--q-season-sigma", type=float, default=0.00)

    # --- Model & MCMC controls ---
    p.add_argument("--m0-gamma-init", type=str, default=None, help="comma-separated (length p-1, newest-first)")
    p.add_argument("--P0-alpha-init", type=float, default=0.25)
    p.add_argument("--P0-beta-init",  type=float, default=0.05)
    p.add_argument("--P0-gamma-init", type=float, default=0.25)
    p.add_argument("--s-alpha-sig-init", type=float, default=1e-4)
    p.add_argument("--s-beta-sig-init",  type=float, default=1e-3)
    p.add_argument("--s-gamma-sig-init", type=float, default=1e-3)

    # Priors
    p.add_argument("--prior-m-m0-alpha", type=float, default=0.0)
    p.add_argument("--prior-s-m0-alpha", type=float, default=10.0)
    p.add_argument("--prior-m-m0-beta",  type=float, default=0.0)
    p.add_argument("--prior-s-m0-beta",  type=float, default=10.0)
    p.add_argument("--prior-m-m0-gamma", type=str, default=None, help="comma-separated (length p-1, newest-first)")
    p.add_argument("--prior-s-m0-gamma", type=float, default=5.0)

    p.add_argument("--prior-a-P0-alpha", type=float, default=2.0)
    p.add_argument("--prior-b-P0-alpha", type=float, default=1.0)
    p.add_argument("--prior-a-P0-beta",  type=float, default=2.0)
    p.add_argument("--prior-b-P0-beta",  type=float, default=1.0)
    p.add_argument("--prior-a-P0-gamma", type=float, default=2.0)
    p.add_argument("--prior-b-P0-gamma", type=float, default=1.0)

    # PC priors for volatility sds
    p.add_argument("--pc-frac-alpha",  type=float, default=0.10)
    p.add_argument("--pc-frac-beta",   type=float, default=0.10)
    p.add_argument("--pc-frac-gamma",  type=float, default=0.10)
    p.add_argument("--pc-alpha-prob",  type=float, default=0.05)
    p.add_argument("--pc-lambda-alpha", type=float, default=None)
    p.add_argument("--pc-lambda-beta",  type=float, default=None)
    p.add_argument("--pc-lambda-gamma", type=float, default=None)
    p.add_argument("--pc-a-lambda-alpha", type=float, default=1.0)
    p.add_argument("--pc-b-lambda-alpha", type=float, default=1.0)
    p.add_argument("--pc-a-lambda-beta",  type=float, default=1.0)
    p.add_argument("--pc-b-lambda-beta",  type=float, default=1.0)
    p.add_argument("--pc-a-lambda-gamma", type=float, default=1.0)
    p.add_argument("--pc-b-lambda-gamma", type=float, default=1.0)

    # Sampler config & PGAS
    p.add_argument("--n-iter", type=int, default=4000)
    p.add_argument("--burn", type=int, default=1000)
    p.add_argument("--thin", type=int, default=1)
    p.add_argument("--seed", type=int, default=40)
    p.add_argument("--progress", default=True)
    p.add_argument("--progress-every", type=float, default=0)
    p.add_argument("--slice-w", type=float, default=0.4)
    p.add_argument("--slice-m", type=int, default=40)
    p.add_argument("--slice-max-shrink", type=int, default=1000)
    p.add_argument("--n-particles", type=int, default=100)
    p.add_argument("--ess-resample-ratio", type=float, default=0.9)
    p.add_argument("--resample-method", choices=["systematic","multinomial","stratified"], default="systematic")

    # I/O
    p.add_argument("--out-dir", type=str, default="results/simulations/DLM_vol_PGAS")
    p.add_argument("--print-summary", default=True)

    args = p.parse_args()
    np.random.seed(args.seed)

    def _csv_floats_or_none(s: str | None):
        if s is None: return None
        s = s.strip()
        if s == "": return None
        return [float(z) for z in s.split(",") if z.strip() != ""]

    # Simulate data if simulator is available

    start_date = datetime(2000,1,1)
    if args.m0_season_sigma is None:
        args.m0_season_sigma = [0.0]*(args.period-1)
    if args.v0_season_sigma is None:
        args.v0_season_sigma = [0.1]*(args.period-1)
    if args.m0_gamma_init is None:
        args.m0_gamma_init = [0.0]*(args.period-1)

    # Make simple Mean Time Series with stochastic volatility (fixed location)

    mts = Mean_Time_Series(
        level_mode="deterministic", trend_mode="none", seasonal_mode="none",
        level_mode_sigma=args.level_mode,
        trend_mode_sigma=args.trend_mode,
        seasonal_mode_sigma=args.seasonal_mode,
        period=args.period,
        m0_level=args.m0_level, v0_level=0.0,
        m0_trend=0.0, v0_trend=0.0,
        m0_season=args.m0_season_sigma, v0_season=args.v0_season_sigma,
        q_level_sigma=args.q_level_sigma,
        q_trend_sigma=args.q_trend_sigma,
        q_season_sigma=args.q_season_sigma,
        m0_level_sigma=args.m0_level_sigma,
        v0_level_sigma=args.v0_level_sigma,
        m0_trend_sigma=args.m0_trend_sigma,
        v0_trend_sigma=args.v0_trend_sigma,
        m0_season_sigma=args.m0_season_sigma,
        v0_season_sigma=args.v0_season_sigma,
        start_date=start_date,
    )

    y = []
    for _ in range(args.T):
        mts.move(); y.append(mts.measure())
    y = np.asarray(y, float)

    truths = mts.get_truth_paths(as_numpy=True)
    sigma_true = truths["sigma_t"][1:1+args.T]
    mu_const = float(args.m0_level)


    pri_gamma_vec = _csv_floats_or_none(args.prior_m_m0_gamma)

    priors = Priors(
        m_m0_alpha=float(args.prior_m_m0_alpha),
        s_m0_alpha=float(args.prior_s_m0_alpha),
        m_m0_beta=float(args.prior_m_m0_beta),
        s_m0_beta=float(args.prior_s_m0_beta),
        m_m0_gamma=None if pri_gamma_vec is None else pri_gamma_vec,
        s_m0_gamma=float(args.prior_s_m0_gamma),
        a_P0_alpha=float(args.prior_a_P0_alpha),
        b_P0_alpha=float(args.prior_b_P0_alpha),
        a_P0_beta=float(args.prior_a_P0_beta),
        b_P0_beta=float(args.prior_b_P0_beta),
        a_P0_gamma=float(args.prior_a_P0_gamma),
        b_P0_gamma=float(args.prior_b_P0_gamma),
        pc_alpha=PCPrior(
            lambda_s=(None if args.pc_lambda_alpha is None else float(args.pc_lambda_alpha)),
            a_lambda=float(args.pc_a_lambda_alpha),
            b_lambda=float(args.pc_b_lambda_alpha),
            frac=float(args.pc_frac_alpha),
            alpha_prob=float(args.pc_alpha_prob),
        ),
        pc_beta=PCPrior(
            lambda_s=(None if args.pc_lambda_beta is None else float(args.pc_lambda_beta)),
            a_lambda=float(args.pc_a_lambda_beta),
            b_lambda=float(args.pc_b_lambda_beta),
            frac=float(args.pc_frac_beta),
            alpha_prob=float(args.pc_alpha_prob),
        ),
        pc_gamma=PCPrior(
            lambda_s=(None if args.pc_lambda_gamma is None else float(args.pc_lambda_gamma)),
            a_lambda=float(args.pc_a_lambda_gamma),
            b_lambda=float(args.pc_b_lambda_gamma),
            frac=float(args.pc_frac_gamma),
            alpha_prob=float(args.pc_alpha_prob),
        ),
    )

    cfg = SamplerConfig(
        n_iter=int(args.n_iter),
        burn=int(args.burn),
        thin=int(args.thin),
        random_seed=int(args.seed),
        progress=bool(args.progress),
        progress_every=float(args.progress_every),
        slice_w=float(args.slice_w),
        slice_m=int(args.slice_m),
        slice_max_shrink=int(args.slice_max_shrink),
        n_particles=int(args.n_particles),
        ess_resample_ratio=float(args.ess_resample_ratio),
        resample_method=str(args.resample_method),
    )

    sampler = DLMVolatilityPGASSampler(
        y=y,
        period=int(args.period),
        mu_fixed=mu_const,
        level_mode=args.level_mode,
        trend_mode=args.trend_mode,
        seasonal_mode=args.seasonal_mode,
        m0_gamma_init=args.m0_gamma_init,
        P0_alpha_init=float(args.P0_alpha_init),
        P0_beta_init=float(args.P0_beta_init),
        P0_gamma_init=float(args.P0_gamma_init),
        s_alpha_sig_init=float(args.s_alpha_sig_init),
        s_beta_sig_init=float(args.s_beta_sig_init),
        s_gamma_sig_init=float(args.s_gamma_sig_init),
        priors=priors,
        cfg=cfg,
    )

    if sigma_true is not None:
        sampler.set_truth_paths(mu=None, sigma_t=sigma_true)
    print("dim =", sampler.dim, "layout =", sampler._layout)
    # If dim == 0, you’re on the deterministic fast path (no PF/PGAS at all).

    t0 = time.time()
    post = sampler.run()
    elapsed = time.time() - t0
    print(f"Run time: {elapsed:.2f}s")

    out_dir = os.path.join(
        args.out_dir,
        f"vol-PGAS-{args.level_mode}-{args.trend_mode}-{args.seasonal_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(out_dir, exist_ok=True)
    sampler.save_posterior(
        out_npz_path=os.path.join(out_dir, "posterior.npz"),
        extra_meta={"elapsed_seconds": float(elapsed)}
    )

    if args.print_summary:
        import math as _m
        with np.printoptions(suppress=True, precision=4):
            print("\n--- Summary (posterior means) ---")
            print(f"loglik: {np.mean(post['loglik']):.4g}")
            if "Q_alpha_sig" in post:
                m = float(np.mean(post["Q_alpha_sig"]))
                print(f"Q_alpha(sig): {m:.4g}  (√Q ≈ {_m.sqrt(m):.4g})")
            else:
                print("Q_alpha(sig): n/a")
            if "Q_beta_sig" in post:
                m = float(np.mean(post["Q_beta_sig"]))
                print(f"Q_beta(sig):  {m:.4g}  (√Q ≈ {_m.sqrt(m):.4g})")
            else:
                print("Q_beta(sig): n/a")
            if "Q_gamma_sig" in post:
                m = float(np.mean(post["Q_gamma_sig"]))
                print(f"Q_gamma(sig): {m:.4g}  (√Q ≈ {_m.sqrt(m):.4g})")
            else:
                print("Q_gamma(sig): n/a")
