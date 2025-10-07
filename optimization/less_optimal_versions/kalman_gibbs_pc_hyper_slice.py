from __future__ import annotations

"""
Bayesian Gaussian DLM via Kalman-FFBS + Gibbs/MH (concise, clarified semantics)
-------------------------------------------------------------------------------

Key differences / highlights
• Innovation **standard deviations** are sampled (slice); we **report/store Q = s²**.
• Clean separation of: model wiring (H/G/u/Q), state simulation (FFBS),
  parameter blocks (σ, {s,λ}, {m0,P0}, deterministics), and persistence.
• PC priors on innovation s with optional Gamma hyperpriors on λ.
• Deterministic components live **outside** the latent state vector.
• CLI supports level_mode in {"dynamic","deterministic","none"}.

m0/P0 semantics per component (α level, β trend, γ season)
• dynamic:      m0_* is the prior mean of the initial latent state x0_*;
                 P0_* is the prior variance of x0_* (both inferred).
• deterministic: m0_* is a fixed (non-evolving) contribution to μ_t with Normal prior;
                 P0_* = 0 and is not sampled.
• none:         component absent. m0_* = 0 by convention; we do NOT compute or store it; priors ignored.

Author: Bastiaan Aelbrecht
"""

import json
import math
import os
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import warnings, numpy as np
warnings.filterwarnings(
    "ignore",
    message="Conversion of an array with ndim > 0 to a scalar is deprecated",
    category=DeprecationWarning,
)



# =============================================================================
# Small utilities
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
# Priors & Config
# =============================================================================

@dataclass
class PCPrior:
    """
    Penalized Complexity prior on innovation sd s>0: p(s | λ) = λ exp(-λ s).
    Optionally place λ ~ Gamma(a_lambda, b_lambda) with (shape=a, rate=b).
    If lambda_s is provided, treat it as fixed.
    We initialize λ via: λ ≈ -log(alpha_prob) / (frac * scale_proxy)
    """
    lambda_s: Optional[float] = None
    a_lambda: float = 1.0
    b_lambda: float = 1.0
    frac: float = 0.10
    alpha_prob: float = 0.05


@dataclass
class Priors:
    # Observation noise: log σ ~ N(m_sigma, s_sigma^2)
    m_sigma: float = 0.0
    s_sigma: float = 1.0

    # m0 priors (reused across modes):
    # - dynamic: prior for initial latent mean x0_* (conjugate update)
    # - deterministic: prior for the fixed (non-latent) contribution in μ_t
    # - none: ignored
    m_m0_alpha: float = 0.0
    s_m0_alpha: float = 10.0
    m_m0_beta: float = 0.0
    s_m0_beta: float = 10.0
    # first p-1 entries; last is implied)
    m_m0_gamma: Optional[Sequence[float]] = None
    s_m0_gamma: float = 5.0

    # log P0 priors (only used in dynamic mode):
    # z = log P0 ~ N(m_logP0_*, s_logP0_*^2)
    m_logP0_alpha: float = 0.0
    s_logP0_alpha: float = 1.0
    m_logP0_beta: float = 0.0
    s_logP0_beta: float = 1.0
    m_logP0_gamma: float = 0.0  # shared for the p-1 seasonal entries
    s_logP0_gamma: float = 1.0

    # PC priors for innovation sds (with optional hyperpriors)
    pc_alpha: PCPrior = field(default_factory=PCPrior)
    pc_beta: PCPrior = field(default_factory=PCPrior)
    pc_gamma: PCPrior = field(default_factory=PCPrior)


@dataclass
class SamplerConfig:
    n_iter: int = 4000
    burn: int = 1000
    thin: int = 1

    random_seed: Optional[int] = 7
    progress: bool = True
    progress_every: int = 0  # 0 => ~2% of n_iter

    # RW–MH step sizes (only logsigma + deterministic params)
    step_logsigma: float = 0.15
    step_level: float = 0.05
    step_slope: float = 0.02
    step_season: float = 0.05

    # Adaptive RW–MH (Robbins–Monro on log-scale)
    adapt_steps: bool = True
    adapt_every: int = 25
    adapt_until: str = "burn"  # "burn" or "all"
    adapt_target_1d: float = 0.44
    adapt_eta0: float = 0.05
    adapt_eta_decay: float = 0.75
    step_min: float = 1e-5
    step_max: float = 1.0

    # Slice sampler (for log s and log P0)
    slice_w: float = 0.4
    slice_m: int = 40
    slice_max_shrink: int = 1000


# =============================================================================
# DLM Sampler
# =============================================================================

class DLMGibbs:
    """
    Gaussian Dynamic Linear Model (DLM)

    State (dynamic components only):
        x0 ~ N(m0, P0)
        x_t | x_{t-1} ~ N(G x_{t-1} + u(x_{t-1}), Q)
    Observation:
        y_t | x_t ~ N(H x_t + μ_det(t), σ^2)

    Modes:
        level_mode    ∈ {"dynamic", "deterministic", "none"}
        trend_mode    ∈ {"dynamic", "deterministic", "none"}
        seasonal_mode ∈ {"dynamic", "deterministic", "none"}

    Notes:
        • Deterministic components are NOT included in x_t.
        • Dynamic seasonality uses the standard (p-1)-dimensional shift + closure.
        • If trend is dynamic then level must be dynamic.

    Inference blocks (depending on modes):
        σ, (λ_α, λ_β, λ_γ), Q_α,Q_β,Q_γ (reported as variances),
        m0 and P0 for dynamic coordinates; static “m0_*” for deterministic parts.
    """

    # --------------------------- Construction --------------------------- #
    def __init__(
        self,
        y: np.ndarray,
        period: int,
        level_mode: str = "dynamic",
        trend_mode: str = "dynamic",
        seasonal_mode: str = "dynamic",
        # initial placeholders (chains start here; they’re updated later)
        m0_alpha_init: float = 0.0,
        logP0_alpha_init: float = 0.0,
        m0_beta_init: float = 0.0,
        logP0_beta_init: float = 0.0,
        m0_gamma_init: Optional[Sequence[float]] = None,  # length p-1
        logP0_gamma_init: float = 0.0,
        # obs/process init
        logsigma_init: float = 0.0,
        s_alpha_init: float = 1e-2,  # internal sd for α
        s_beta_init: float = 1e-3,   # internal sd for β
        s_gamma_init: float = 1e-3,  # internal sd for γ
        priors: Priors = Priors(),
        cfg: SamplerConfig = SamplerConfig(),
    ):
        # Data checks
        self.y = np.asarray(y, float)
        self.T = int(self.y.size)
        self._t = np.arange(self.T, dtype=float)
        self._t_c = self._t - self._t.mean()
        self.period = int(period)
        if self.period < 2:
            raise ValueError("period must be >= 2")

        # Modes
        if level_mode not in {"dynamic", "deterministic", "none"}:
            raise ValueError("invalid level_mode")
        if trend_mode not in {"dynamic", "deterministic", "none"}:
            raise ValueError("invalid trend_mode")
        if seasonal_mode not in {"dynamic", "deterministic", "none"}:
            raise ValueError("invalid seasonal_mode")
        if trend_mode == "dynamic" and level_mode != "dynamic":
            raise ValueError("dynamic trend requires dynamic level")

        self.level_mode = level_mode
        self.trend_mode = trend_mode
        self.seasonal_mode = seasonal_mode

        # Priors / config
        self.priors = priors
        self.cfg = cfg
        if cfg.random_seed is not None:
            np.random.seed(cfg.random_seed)

        # --- dynamic state layout
        layout: List[str] = []
        if self.level_mode == "dynamic":
            layout.append("alpha")  # level
        if self.trend_mode == "dynamic":
            layout.append("beta")   # slope
        if self.seasonal_mode == "dynamic":
            layout.extend([f"g{k}" for k in range(1, self.period)])  # p-1 seasonal coords

        self._layout = layout
        self.dim = len(layout)

        self.idx_alpha = layout.index("alpha") if "alpha" in layout else None
        self.idx_beta = layout.index("beta") if "beta" in layout else None

        if self.seasonal_mode == "dynamic":
            self.idx_g_start = layout.index("g1")
            self.idx_g_end = self.idx_g_start + (self.period - 2)
        else:
            self.idx_g_start = None
            self.idx_g_end = None

        # --- parameters
        # observation noise
        self.logsigma = float(logsigma_init)
        self.sigma2 = float(np.exp(2.0 * self.logsigma))

        # process sds (we store s, but report Q = s^2)
        self.s_alpha = float(s_alpha_init) if self.idx_alpha is not None else 0.0
        self.s_beta = float(s_beta_init) if self.idx_beta is not None else 0.0
        self.s_gamma = float(s_gamma_init) if self.seasonal_mode == "dynamic" else 0.0

        # PC λ initialization
        self.lambda_alpha, self.lambda_beta, self.lambda_gamma = self._init_pc_lambdas()

        # initial m0 & logP0 for dynamic coords
        self.m0_alpha = float(m0_alpha_init) if self.idx_alpha is not None else 0.0
        self.logP0_alpha = float(logP0_alpha_init) if self.idx_alpha is not None else 0.0
        self.m0_beta = float(m0_beta_init) if self.idx_beta is not None else 0.0
        self.logP0_beta = float(logP0_beta_init) if self.idx_beta is not None else 0.0
        if self.seasonal_mode == "dynamic":
            if m0_gamma_init is None:
                self.m0_gamma = np.zeros(self.period - 1, float)
            else:
                g = np.asarray(m0_gamma_init, float)
                if g.size != self.period - 1:
                    raise ValueError("m0_gamma_init must have length p-1")
                self.m0_gamma = g
            self.logP0_gamma = float(logP0_gamma_init)
        else:
            self.m0_gamma = None
            self.logP0_gamma = 0.0

        # deterministic parameters (live only in the observation mean)
        self.m0_alpha = float(self.priors.m_m0_alpha) if self.level_mode == "deterministic" else 0.0
        self.m0_beta = float(self.priors.m_m0_beta) if self.trend_mode == "deterministic" else 0.0
        if self.seasonal_mode == "deterministic":
            if m0_gamma_init is None:
                base = np.zeros(self.period - 1, float) if self.priors.m_m0_gamma is None else np.asarray(self.priors.m_m0_gamma, float)
                if base.size != self.period - 1:
                    raise ValueError("priors.m_m0_gamma must be length p-1")
            else:
                base = np.asarray(m0_gamma_init, float)
                if base.size != self.period - 1:
                    raise ValueError("m0_gamma_init must be length p-1")
            last = -float(np.sum(base))
            self.m0_gamma = np.r_[base, last].astype(float)
        else:
            self.m0_gamma = None

        # For 'none' mode, make intent explicit (no sampling; not stored)
        if self.level_mode == "none":
            self.m0_alpha = 0.0
            self.logP0_alpha = 0.0
        if self.trend_mode == "none":
            self.m0_beta = 0.0
            self.logP0_beta = 0.0
        if self.seasonal_mode == "none":
            self.m0_gamma = None
            self.logP0_gamma = 0.0

        # latent path (x0..xT) for dynamic coords
        self.x = np.zeros((self.T + 1, self.dim), float)
        if self.dim > 0:
            m0_vec, P0_diag = self._current_m0_P0()
            self.x[0] = np.random.multivariate_normal(m0_vec, np.diag(P0_diag) + 1e-10 * np.eye(self.dim))
            self._propagate_initial_path(Q_init=np.full(self.dim, 1e-6, float))

        # storage for draws
        self.keep: Dict[str, np.ndarray] = {}

        # MH bookkeeping
        self.accept = {"logsigma": 0, "level": 0, "slope": 0, "season": 0}
        self.proposals = {"logsigma": 0, "level": 0, "slope": 0, "season": 0}
        self._mh_prev_acc = dict(self.accept)
        self._mh_prev_prop = dict(self.proposals)
        self._adapt_round = 0

        # Optional truth overlays (kept separate; never override inference params)
        self.true_sigma: Optional[float] = None
        self.true_Q: Optional[np.ndarray] = None
        self.true_mu_t: Optional[np.ndarray] = None
        self.true_alpha_t: Optional[np.ndarray] = None
        self.true_beta_t: Optional[np.ndarray] = None
        self.true_gamma_t: Optional[np.ndarray] = None

    # --------------------- Truth overlays (optional) --------------------- #
    def set_truth(
        self,
        sigma: Optional[float] = None,
        Q: Optional[np.ndarray] = None,
        m0_level: Optional[float] = None,
        m0_trend: Optional[float] = None,
        m0_season: Optional[Sequence[float]] = None,
        P0_level: Optional[float] = None,
        P0_trend: Optional[float] = None,
        P0_season: Optional[Sequence[float]] = None
    ) -> None:
        self.true_sigma = None if sigma is None else float(sigma)
        self.true_Q = None if Q is None else np.asarray(Q, float)
        self.true_m0_level = None if m0_level is None else float(m0_level)
        self.true_m0_trend = None if m0_trend is None else float(m0_trend)
        self.true_m0_season = None if m0_season is None else np.asarray(m0_season, float)
        self.true_P0_level = None if P0_level is None else float(P0_level)
        self.true_P0_trend = None if P0_trend is None else float(P0_trend)
        self.true_P0_season = None if P0_season is None else np.asarray(P0_season, float)

    def set_truth_paths(
        self,
        mu: Optional[np.ndarray] = None,
        alpha: Optional[np.ndarray] = None,
        beta: Optional[np.ndarray] = None,
        gamma: Optional[np.ndarray] = None,
    ) -> None:
        self.true_mu_t = None if mu is None else np.asarray(mu, float)
        self.true_alpha_t = None if alpha is None else np.asarray(alpha, float)
        self.true_beta_t = None if beta is None else np.asarray(beta, float)
        self.true_gamma_t = None if gamma is None else np.asarray(gamma, float)

    # --------------------- PC λ initialization --------------------- #
    def _init_pc_lambdas(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        y = self.y
        sd1 = _robust_sd(np.diff(y)) if y.size >= 2 else 0.0     # proxy for level
        sd2 = _robust_sd(np.diff(y, n=2)) if y.size >= 3 else 0.0  # proxy for slope
        sdg = sd1  # proxy for season (rough)

        def _cal(pc: PCPrior, sd: float) -> float:
            u = max(1e-12, pc.frac * max(1e-12, sd))
            return float(-math.log(max(1e-12, pc.alpha_prob)) / u)

        la = float(self.priors.pc_alpha.lambda_s) if self.priors.pc_alpha.lambda_s is not None else (_cal(self.priors.pc_alpha, sd1) if self.idx_alpha is not None else None)
        lb = float(self.priors.pc_beta.lambda_s) if self.priors.pc_beta.lambda_s is not None else (_cal(self.priors.pc_beta, sd2) if self.idx_beta is not None else None)
        lg = float(self.priors.pc_gamma.lambda_s) if self.priors.pc_gamma.lambda_s is not None else (_cal(self.priors.pc_gamma, sdg) if self.seasonal_mode == "dynamic" else None)

        if self.cfg.progress:
            def f(x): return "n/a" if x is None else f"{x:.4g}"
            print(f"[init] PC λ: α={f(la)}, β={f(lb)}, γ={f(lg)} (fixed if provided)")
        return la, lb, lg

    # ----------------------------- Model matrices ----------------------------- #
    def _H(self) -> np.ndarray:
        """1×dim observation loading for dynamic coordinates."""
        if self.dim == 0:
            return np.zeros((1, 0))
        h = np.zeros(self.dim, float)
        if self.idx_alpha is not None:
            h[self.idx_alpha] = 1.0
        if self.seasonal_mode == "dynamic":
            h[self.idx_g_end] = 1.0
        return h.reshape(1, -1)

    def _G(self) -> np.ndarray:
        """dim×dim evolution matrix."""
        if self.dim == 0:
            return np.zeros((0, 0))
        G = np.eye(self.dim)
        if self.idx_alpha is not None and self.idx_beta is not None:
            G[self.idx_alpha, self.idx_beta] = 1.0
        if self.seasonal_mode == "dynamic":
            gs, ge = self.idx_g_start, self.idx_g_end
            # shift block
            for k in range(ge - gs):
                G[gs + k, gs + k] = 0.0
                G[gs + k, gs + k + 1] = 1.0
            # last seasonal row gets zero rows (closure via u(.))
            G[ge, gs: ge + 1] = 0.0
        return G

    def _u(self, x_prev: np.ndarray) -> np.ndarray:
        """
        Mean shift in state evolution (only for dynamic seasonality closure):
            g_{t,p-1} = -sum(g_{t-1,0:(p-2)})
        """
        if self.dim == 0:
            return np.zeros(0, float)
        u = np.zeros(self.dim, float)
        if self.seasonal_mode == "dynamic":
            gs, ge = self.idx_g_start, self.idx_g_end
            prev = x_prev[gs: ge + 1]
            u[ge] = -float(np.sum(prev))
        return u

    def _Q(self) -> np.ndarray:
        """Process covariance from current innovation sds."""
        if self.dim == 0:
            return np.zeros((0, 0))
        Q = np.zeros((self.dim, self.dim))
        if self.idx_alpha is not None and self.s_alpha > 0:
            Q[self.idx_alpha, self.idx_alpha] = self.s_alpha**2
        if self.idx_beta is not None and self.s_beta > 0:
            Q[self.idx_beta, self.idx_beta] = self.s_beta**2
        if self.seasonal_mode == "dynamic" and self.s_gamma > 0:
            Q[self.idx_g_end, self.idx_g_end] = self.s_gamma**2
        return Q

    def _mu_det(self, t: int) -> float:
        """Deterministic contribution to observation mean."""
        out = 0.0
        if self.level_mode == "deterministic":
            out += self.m0_alpha
        if self.trend_mode == "deterministic":
            out += self.m0_beta * t
        if self.seasonal_mode == "deterministic":
            out += float(self.m0_gamma[t % self.period])
        return out

    def _current_m0_P0(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build m0 vector and P0 diagonal for current dynamic layout."""
        m0, P0 = [], []
        if self.idx_alpha is not None:
            m0.append(self.m0_alpha)
            P0.append(float(np.exp(self.logP0_alpha)))
        if self.idx_beta is not None:
            m0.append(self.m0_beta)
            P0.append(float(np.exp(self.logP0_beta)))
        if self.seasonal_mode == "dynamic":
            m0.extend(list(self.m0_gamma))
            P0.extend([float(np.exp(self.logP0_gamma))] * (self.period - 1))
        return np.asarray(m0, float), np.asarray(P0, float)

    # ------------------------- FFBS (Kalman + Carter–Kohn) ------------------------- #
    def _ffbs(self) -> np.ndarray:
        if self.dim == 0:
            return self.x.copy()

        H = self._H()
        G = self._G()
        Q = self._Q()
        R = float(self.sigma2)

        m0_vec, P0_diag = self._current_m0_P0()

        m = np.zeros((self.T + 1, self.dim))
        C = np.zeros((self.T + 1, self.dim, self.dim))
        a = np.zeros((self.T + 1, self.dim))
        Rm = np.zeros((self.T + 1, self.dim, self.dim))

        m[0] = m0_vec
        C[0] = np.diag(P0_diag) + 1e-12 * np.eye(self.dim)

        # Forward (filter)
        for t in range(1, self.T + 1):
            u = self._u(m[t - 1])
            a[t] = G @ m[t - 1] + u
            Rm[t] = G @ C[t - 1] @ G.T + Q

            resid_mean = float(self.y[t - 1] - self._mu_det(t - 1))
            S = float(H @ Rm[t] @ H.T + R)  # scalar
            K = (Rm[t] @ H.T) / S
            v = resid_mean - float(H @ a[t])
            m[t] = a[t] + (K.flatten() * v)
            C[t] = Rm[t] - K @ (H @ Rm[t])

        # Backward (simulate smoothing)
        x = np.zeros_like(self.x)
        # terminal draw
        x[self.T] = np.random.multivariate_normal(m[self.T], C[self.T])

        for t in range(self.T - 1, -1, -1):
            u = self._u(m[t])
            # J = C_t G' R_{t+1}^{-1}
            J = C[t] @ G.T @ np.linalg.inv(Rm[t + 1])
            mean = m[t] + J @ (x[t + 1] - (G @ m[t] + u))
            cov = C[t] - J @ Rm[t + 1] @ J.T
            cov = 0.5 * (cov + cov.T)
            min_eig = np.linalg.eigvalsh(cov).min()
            if min_eig <= 0:
                cov += (1e-10 - min_eig) * np.eye(cov.shape[0])
            x[t] = np.random.multivariate_normal(mean, cov)

        return x

    def _propagate_initial_path(self, Q_init: np.ndarray) -> None:
        if self.dim == 0:
            return
        G = self._G()
        for t in range(1, self.T + 1):
            u = self._u(self.x[t - 1])
            self.x[t] = (
                G @ self.x[t - 1]
                + u
                + np.random.normal(0.0, np.sqrt(Q_init), size=self.dim)
            )

    # -------------------- Generic slice sampler -------------------- #
    def _slice(self, f: Callable[[float], float], z0: float) -> float:
        w = float(self.cfg.slice_w)
        m = int(self.cfg.slice_m)
        limit = int(self.cfg.slice_max_shrink)

        y_star = f(z0) - np.random.exponential(1.0)
        u = np.random.rand()
        L = z0 - u * w
        R = L + w

        j = int(np.floor(m * np.random.rand()))
        k = (m - 1) - j
        while j > 0 and f(L) > y_star:
            L -= w
            j -= 1
        while k > 0 and f(R) > y_star:
            R += w
            k -= 1

        for _ in range(limit):
            z_prop = np.random.uniform(L, R)
            if f(z_prop) >= y_star:
                return z_prop
            if z_prop < z0:
                L = z_prop
            else:
                R = z_prop
        return z0  # give up (rare)

    # ------------------ Helpers: means & likelihood ------------------ #
    def _mu_vec(self) -> np.ndarray:
        H = self._H()
        mu = np.zeros(self.T, float)
        for t in range(1, self.T + 1):
            dyn = float(H @ self.x[t]) if self.dim > 0 else 0.0
            mu[t - 1] = self._mu_det(t - 1) + dyn
        return mu

    def _ll_gauss(self, mu_vec: np.ndarray, sigma2: float) -> float:
        e = self.y - mu_vec
        T = float(self.T)
        return float(-0.5 * T * math.log(2.0 * math.pi * sigma2) - 0.5 * np.sum(e * e) / sigma2)

    # ------------------ σ (RW–MH on logsigma) ------------------ #
    def update_logsigma(self) -> None:
        step = float(self.cfg.step_logsigma)
        cur = self.logsigma
        prop = cur + np.random.normal(0.0, step)
        mu = self._mu_vec()
        ll_cur = self._ll_gauss(mu, math.exp(2.0 * cur))
        ll_new = self._ll_gauss(mu, math.exp(2.0 * prop))
        lp_cur = -0.5 * ((cur - self.priors.m_sigma) / self.priors.s_sigma) ** 2
        lp_new = -0.5 * ((prop - self.priors.m_sigma) / self.priors.s_sigma) ** 2
        self.proposals["logsigma"] += 1
        if np.log(np.random.rand()) < (ll_new + lp_new) - (ll_cur + lp_cur):
            self.logsigma = prop
            self.sigma2 = float(math.exp(2.0 * prop))
            self.accept["logsigma"] += 1

    # ------------- Innovation sums of squares (for s updates) ------------- #
    def _innovation_ss_alpha(self) -> Tuple[float, int]:
        if self.idx_alpha is None:
            return 0.0, 0
        ss = 0.0
        for t in range(1, self.T + 1):
            drift = self.x[t - 1, self.idx_beta] if self.idx_beta is not None else 0.0
            mean = self.x[t - 1, self.idx_alpha] + drift
            ss += (self.x[t, self.idx_alpha] - mean) ** 2
        return float(ss), self.T

    def _innovation_ss_beta(self) -> Tuple[float, int]:
        if self.idx_beta is None:
            return 0.0, 0
        d = self.x[1:, self.idx_beta] - self.x[:-1, self.idx_beta]
        return float(np.sum(d * d)), self.T

    def _innovation_ss_gamma(self) -> Tuple[float, int]:
        if self.seasonal_mode != "dynamic":
            return 0.0, 0
        gs, ge = self.idx_g_start, self.idx_g_end
        ss = 0.0
        for t in range(1, self.T + 1):
            prev = self.x[t - 1, gs: ge + 1]
            mean_new = -float(np.sum(prev))
            ss += (self.x[t, ge] - mean_new) ** 2
        return float(ss), self.T

    # --- slice for z = log s under PC prior: p(s|λ) ∝ exp(-λ s)
    def _slice_logsd(self, z0: float, SS: float, T_eff: int, lam: float) -> float:
        # log π(z) = -T z - 0.5*SS*exp(-2z) - λ*exp(z) + z   (Jacobian for s = exp(z))
        def f(z: float) -> float:
            return -(T_eff * z) - 0.5 * SS * math.exp(-2 * z) - lam * math.exp(z) + z

        return self._slice(f, z0)

    def update_process_sds(self) -> None:
        # α
        if self.idx_alpha is not None and (self.lambda_alpha is not None):
            ss, T_eff = self._innovation_ss_alpha()
            z0 = math.log(max(1e-18, self.s_alpha))
            z = self._slice_logsd(z0, ss, T_eff, float(self.lambda_alpha))
            self.s_alpha = float(math.exp(z))
        # β
        if self.idx_beta is not None and (self.lambda_beta is not None):
            ss, T_eff = self._innovation_ss_beta()
            z0 = math.log(max(1e-18, self.s_beta))
            z = self._slice_logsd(z0, ss, T_eff, float(self.lambda_beta))
            self.s_beta = float(math.exp(z))
        # γ
        if self.seasonal_mode == "dynamic" and (self.lambda_gamma is not None):
            ss, T_eff = self._innovation_ss_gamma()
            z0 = math.log(max(1e-18, self.s_gamma))
            z = self._slice_logsd(z0, ss, T_eff, float(self.lambda_gamma))
            self.s_gamma = float(math.exp(z))

    # --- λ | s ~ Gamma(a+1, b+s) (shape–rate) ---
    def _gibbs_lambda_single(self, which: str) -> None:
        if which == "alpha":
            pc = self.priors.pc_alpha
            if pc.lambda_s is not None or self.idx_alpha is None:
                return
            a = pc.a_lambda + 1.0
            b = pc.b_lambda + max(0.0, self.s_alpha)
            self.lambda_alpha = float(np.random.gamma(shape=a, scale=1.0 / b))
        elif which == "beta":
            pc = self.priors.pc_beta
            if pc.lambda_s is not None or self.idx_beta is None:
                return
            a = pc.a_lambda + 1.0
            b = pc.b_lambda + max(0.0, self.s_beta)
            self.lambda_beta = float(np.random.gamma(shape=a, scale=1.0 / b))
        elif which == "gamma":
            pc = self.priors.pc_gamma
            if pc.lambda_s is not None or self.seasonal_mode != "dynamic":
                return
            a = pc.a_lambda + 1.0
            b = pc.b_lambda + max(0.0, self.s_gamma)
            self.lambda_gamma = float(np.random.gamma(shape=a, scale=1.0 / b))

    def update_pc_lambdas(self) -> None:
        self._gibbs_lambda_single("alpha")
        self._gibbs_lambda_single("beta")
        self._gibbs_lambda_single("gamma")

    # --- m0 (conjugate Normal) for dynamic coords ---
    @staticmethod
    def _gibbs_m0_scalar(x0: float, m_prior: float, s_prior: float, P0: float) -> float:
        # m0 ~ N(m_prior, s_prior^2), x0 | m0 ~ N(m0, P0)
        prec = 1.0 / (s_prior**2) + 1.0 / max(1e-18, P0)
        var = 1.0 / prec
        mean = var * (m_prior / (s_prior**2) + x0 / max(1e-18, P0))
        return float(np.random.normal(mean, math.sqrt(var)))

    def update_m0(self) -> None:
        if self.dim == 0:
            return
        _, P0_diag = self._current_m0_P0()
        pos = 0
        if self.idx_alpha is not None:
            P0 = float(P0_diag[pos])
            self.m0_alpha = self._gibbs_m0_scalar(float(self.x[0, pos]), self.priors.m_m0_alpha, self.priors.s_m0_alpha, P0)
            pos += 1
        if self.idx_beta is not None:
            P0 = float(P0_diag[pos])
            self.m0_beta = self._gibbs_m0_scalar(float(self.x[0, pos]), self.priors.m_m0_beta, self.priors.s_m0_beta, P0)
            pos += 1
        if self.seasonal_mode == "dynamic":
            s = float(self.priors.s_m0_gamma)
            m_prior = np.zeros(self.period - 1, float) if self.priors.m_m0_gamma is None else np.asarray(self.priors.m_m0_gamma, float)
            if m_prior.size != self.period - 1:
                raise ValueError("priors.m_m0_gamma must have length p-1")
            for k in range(self.period - 1):
                P0 = float(P0_diag[pos])
                self.m0_gamma[k] = self._gibbs_m0_scalar(float(self.x[0, pos]), float(m_prior[k]), s, P0)
                pos += 1

    # --- log P0 (slice) ---
    def _slice_logP0_scalar(self, z0: float, x0: float, m0: float, m_z: float, s_z: float) -> float:
        # x0 ~ N(m0, P0=exp(z)) ⇒ log π(z) = -0.5 z - 0.5 (x0-m0)^2 exp(-z) - 0.5 ((z-m_z)^2 / s_z^2)
        s2 = s_z**2
        d2 = (x0 - m0) ** 2

        def f(z: float) -> float:
            return -0.5 * z - 0.5 * d2 * math.exp(-z) - 0.5 * ((z - m_z) ** 2) / s2

        return self._slice(f, z0)

    def update_logP0(self) -> None:
        if self.dim == 0:
            return
        pos = 0
        if self.idx_alpha is not None:
            x0 = float(self.x[0, pos])
            self.logP0_alpha = self._slice_logP0_scalar(self.logP0_alpha, x0, self.m0_alpha, self.priors.m_logP0_alpha, self.priors.s_logP0_alpha)
            pos += 1
        if self.idx_beta is not None:
            x0 = float(self.x[0, pos])
            self.logP0_beta = self._slice_logP0_scalar(self.logP0_beta, x0, self.m0_beta, self.priors.m_logP0_beta, self.priors.s_logP0_beta)
            pos += 1
        if self.seasonal_mode == "dynamic":
            m_z = float(self.priors.m_logP0_gamma)
            s_z = float(self.priors.s_logP0_gamma)
            for k in range(self.period - 1):
                x0 = float(self.x[0, pos])
                self.logP0_gamma = self._slice_logP0_scalar(self.logP0_gamma, x0, float(self.m0_gamma[k]), m_z, s_z)
                pos += 1

    # --- Deterministic params (RW–MH with Normal priors) ---
    @staticmethod
    def _mh_accept(logacc: float) -> bool:
        return np.log(np.random.rand()) < min(0.0, logacc)

    def _update_det_scalar(
        self,
        key: str,
        cur: float,
        step: float,
        logprior: Callable[[float], float],
        apply_prop: Callable[[float], None],
    ) -> float:
        prop = cur + np.random.normal(0.0, step)

        # likelihood at proposal
        apply_prop(prop)
        mu_prop = self._mu_vec()
        ll_prop = self._ll_gauss(mu_prop, self.sigma2)

        # likelihood at current
        apply_prop(cur)
        mu_cur = self._mu_vec()
        ll_cur = self._ll_gauss(mu_cur, self.sigma2)

        self.proposals[key] += 1
        logacc = (ll_prop + logprior(prop)) - (ll_cur + logprior(cur))
        if self._mh_accept(logacc):
            apply_prop(prop)
            self.accept[key] += 1
            return prop
        return cur

    def update_deterministic_params(self) -> None:
        if self.level_mode == "deterministic":
            def lp(x): return -0.5 * ((x - self.priors.m_m0_alpha) / self.priors.s_m0_alpha) ** 2
            def apply(v): setattr(self, "m0_alpha", float(v))
            self.m0_alpha = self._update_det_scalar("level", self.m0_alpha, self.cfg.step_level, lp, apply)

        # --- deterministic trend: conjugate Gibbs on slope ---
        if self.trend_mode == "deterministic":
            t = np.arange(self.T, dtype=float)  # design for slope (optionally center: t -= t.mean())
            # residuals: remove dynamic part and other deterministic parts
            r = self.y.copy()
            if self.dim > 0:
                H = self._H()
                for k in range(1, self.T+1):
                    r[k-1] -= float(H @ self.x[k])
            if self.level_mode == "deterministic":
                r -= self.m0_alpha
            if self.seasonal_mode == "deterministic":
                r -= self.m0_gamma[np.arange(self.T) % self.period]

            m0, s0 = float(self.priors.m_m0_beta), float(self.priors.s_m0_beta)
            sig2 = float(self.sigma2)

            prec = (t @ t) / sig2 + 1.0 / (s0**2)          # posterior precision (scalar)
            mean = ((t @ r) / sig2 + m0 / (s0**2)) / prec  # posterior mean
            var  = 1.0 / prec                               # posterior variance

            self.m0_beta = float(np.random.normal(mean, math.sqrt(var)))


        # --- deterministic seasonality: conjugate Gibbs on the (p-1) free coeffs ---
        if self.seasonal_mode == "deterministic":
            # Design Z (cache this once in __init__ ideally):
            # months are 0..p-1; columns are 0..p-2; month p-1 row is -1 for all cols
            if not hasattr(self, "_Z_season"):
                midx = np.arange(self.T) % self.period
                K = self.period - 1
                Z = np.zeros((self.T, K))
                for k in range(K):
                    Z[:, k] = (midx == k).astype(float) - (midx == K).astype(float)
                self._Z_season = Z

            # Residuals r_t = y_t - dynamic_part - other deterministic parts
            r = self.y.copy()
            if self.dim > 0:
                H = self._H()
                for t in range(1, self.T + 1):
                    r[t-1] -= float(H @ self.x[t])
            if self.level_mode == "deterministic":
                r -= self.m0_alpha
            if self.trend_mode == "deterministic":
                r -= self.m0_beta * np.arange(self.T, dtype=float)

            # Prior: θ ~ N(m, s^2 I) on the first (p-1) entries
            K = self.period - 1
            m_prior = (np.zeros(K) if self.priors.m_m0_gamma is None
                    else np.asarray(self.priors.m_m0_gamma, float).reshape(-1))
            s2 = float(self.priors.s_m0_gamma)**2
            Z = self._Z_season
            sig2 = float(self.sigma2)

            # Posterior precision and mean
            Prec = (Z.T @ Z) / sig2 + np.eye(K) / s2
            b    = (Z.T @ r) / sig2 + m_prior / s2
            mu   = np.linalg.solve(Prec, b)

            # Sample θ ~ N(mu, Prec^{-1})
            L = np.linalg.cholesky(Prec)         # Prec = L L^T
            z = np.random.randn(K)
            theta = mu + np.linalg.solve(L.T, z) # (L^T)^{-1} z ~ N(0, Prec^{-1})

            # Enforce closure by construction
            self.m0_gamma = np.r_[theta, -theta.sum()]


    # ------------------- Progress formatting ------------------- #
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
        parts = [f"[it {it + 1}/{self.cfg.n_iter}]"]
        parts.append(f"σ={math.exp(self.logsigma):.3f}")
        if self.idx_alpha is not None:
            parts.append(f"Qα={self.s_alpha**2:.4g}")
        if self.idx_beta is not None:
            parts.append(f"Qβ={self.s_beta**2:.4g}")
        if self.seasonal_mode == "dynamic":
            parts.append(f"Qγ={self.s_gamma**2:.4g}")

        def _lam(lam, present, fixed):
            if not present:
                return "-"
            if fixed is not None:
                return f"{fixed:.3g}(fix)"
            return "-" if lam is None else f"{lam:.3g}"

        parts.append(
            "λ=("
            + ",".join(
                [
                    _lam(self.lambda_alpha, self.idx_alpha is not None, self.priors.pc_alpha.lambda_s),
                    _lam(self.lambda_beta, self.idx_beta is not None, self.priors.pc_beta.lambda_s),
                    _lam(self.lambda_gamma, self.seasonal_mode == "dynamic", self.priors.pc_gamma.lambda_s),
                ]
            )
            + ")"
        )

        # m0 display
        if self.level_mode == "dynamic":
            m0a = f"{self.m0_alpha:.4g}"
        elif self.level_mode == "deterministic":
            m0a = f"{self.m0_alpha:.4g}"
        else:
            m0a = "0 (none)"

        if self.trend_mode == "dynamic":
            m0b = f"{self.m0_beta:.4g}"
        elif self.trend_mode == "deterministic":
            m0b = f"{self.m0_beta:.4g}"
        else:
            m0b = "0 (none)"

        if self.seasonal_mode == "dynamic":
            m0g = self._fmt_list(self.m0_gamma, 6, ".4g")
        elif self.seasonal_mode == "deterministic":
            m0g = self._fmt_list(self.m0_gamma[:-1], 6, ".4g")
        else:
            m0g = "0 (none)"
        parts.append(f"m0=(α={m0a}, β={m0b}, γ={m0g})")

        # P0 display
        def e(z): return f"{math.exp(z):.4g}"
        if self.level_mode == "dynamic":
            P0a = e(self.logP0_alpha)
        elif self.level_mode == "deterministic":
            P0a = "0"
        else:
            P0a = "0 (none)"

        if self.trend_mode == "dynamic":
            P0b = e(self.logP0_beta)
        elif self.trend_mode == "deterministic":
            P0b = "0"
        else:
            P0b = "0 (none)"

        if self.seasonal_mode == "dynamic":
            P0g = e(self.logP0_gamma)
        elif self.seasonal_mode == "deterministic":
            P0g = "0"
        else:
            P0g = "0 (none)"

        parts.append(f"P0=(Pα={P0a}, Pβ={P0b}, Pγ={P0g})")
        return " | ".join(parts)

    # ------------------- Adaptive RW–MH ------------------- #
    def _adapt_steps(self, it: int) -> None:
        cfg = self.cfg
        if not cfg.adapt_steps:
            return
        in_window = (cfg.adapt_until == "all") or (it < cfg.burn)
        if (it + 1) % max(1, cfg.adapt_every) != 0 or (not in_window):
            return

        k = self._adapt_round
        eta = cfg.adapt_eta0 / ((1.0 + k) ** cfg.adapt_eta_decay)
        keys: List[str] = ["logsigma"]
        if self.level_mode == "deterministic":
            keys.append("level")
        if self.trend_mode == "deterministic":
            keys.append("slope")
        if self.seasonal_mode == "deterministic":
            keys.append("season")

        target = cfg.adapt_target_1d
        changes = []
        for key in keys:
            a_now = self.accept[key]
            p_now = self.proposals[key]
            a_win = a_now - self._mh_prev_acc[key]
            p_win = p_now - self._mh_prev_prop[key]
            if p_win <= 0:
                continue
            rate = a_win / max(1, p_win)
            name = "logsigma" if key == "logsigma" else key
            s_old = getattr(cfg, f"step_{name}")
            s_new = float(np.clip(s_old * np.exp(eta * (rate - target)), cfg.step_min, cfg.step_max))
            setattr(cfg, f"step_{name}", s_new)
            self._mh_prev_acc[key] = a_now
            self._mh_prev_prop[key] = p_now
            changes.append((key, s_old, s_new, rate))

        if changes and cfg.progress:
            msg = " | ".join([f"{k}: {o:.4g}→{n:.4g} (acc={r:.2f})" for (k, o, n, r) in changes])
            print(f"  [adapt] {msg}")

        self._adapt_round += 1

    # --------------------------------- MCMC --------------------------------- #
    def run(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        save_iters = list(range(cfg.burn, cfg.n_iter, cfg.thin))
        n_kept = len(save_iters)
        keep_idx = 0

        # allocate storage
        self.keep = {"sigma": np.zeros(n_kept, float), "mu": np.zeros((n_kept, self.T), float)}
        if self.idx_alpha is not None:
            self.keep.update(
                {"Q_alpha": np.zeros(n_kept), "lambda_alpha": np.zeros(n_kept), "m0_alpha": np.zeros(n_kept), "P0_alpha": np.zeros(n_kept)}
            )
        if self.idx_beta is not None:
            self.keep.update(
                {"Q_beta": np.zeros(n_kept), "lambda_beta": np.zeros(n_kept), "m0_beta": np.zeros(n_kept), "P0_beta": np.zeros(n_kept)}
            )
        if self.seasonal_mode == "dynamic":
            self.keep.update(
                {
                    "Q_gamma": np.zeros(n_kept),
                    "lambda_gamma": np.zeros(n_kept),
                    "m0_gamma": np.zeros((n_kept, self.period - 1)),
                    "P0_gamma": np.zeros(n_kept),
                    "x": np.zeros((n_kept, self.T, self.dim)),
                }
            )
        elif self.dim > 0:
            self.keep["x"] = np.zeros((n_kept, self.T, self.dim))

        if self.level_mode == "deterministic":
            self.keep["m0_alpha"] = np.zeros(n_kept)
        if self.trend_mode == "deterministic":
            self.keep["m0_beta"] = np.zeros(n_kept)
        if self.seasonal_mode == "deterministic":
            self.keep["m0_gamma"] = np.zeros((n_kept, self.period))

        print_every = cfg.progress_every if cfg.progress_every > 0 else max(1, cfg.n_iter // 50) or 1

        for it in range(cfg.n_iter):
            # 1) FFBS (dynamic state)
            if self.dim > 0:
                self.x = self._ffbs()

            # 2) process s (slice with PC priors) + 3) λ (Gibbs)
            if self.dim > 0:
                self.update_process_sds()
                self.update_pc_lambdas()

            # 4) m0 (Gibbs) and 5) log P0 (slice) for dynamic coords
            if self.dim > 0:
                self.update_m0()
                self.update_logP0()

            # 6) deterministic params
            self.update_deterministic_params()

            # 7) logsigma (RW–MH)
            self.update_logsigma()

            # 8) adapt steps
            self._adapt_steps(it)

            # progress print
            if cfg.progress and ((it + 1) % print_every == 0 or it == cfg.n_iter - 1):
                print(self._progress_line(it))

            # save
            if it in save_iters:
                mu = self._mu_vec()
                self.keep["mu"][keep_idx, :] = mu
                self.keep["sigma"][keep_idx] = float(np.exp(self.logsigma))

                if self.idx_alpha is not None:
                    self.keep["Q_alpha"][keep_idx] = self.s_alpha**2
                    self.keep["lambda_alpha"][keep_idx] = float(self.lambda_alpha or 0.0)
                    self.keep["m0_alpha"][keep_idx] = self.m0_alpha if self.level_mode == "dynamic" else self.m0_alpha
                    self.keep["P0_alpha"][keep_idx] = float(np.exp(self.logP0_alpha)) if self.level_mode == "dynamic" else 0.0

                if self.idx_beta is not None:
                    self.keep["Q_beta"][keep_idx] = self.s_beta**2
                    self.keep["lambda_beta"][keep_idx] = float(self.lambda_beta or 0.0)
                    if self.trend_mode == "dynamic":
                        self.keep["m0_beta"][keep_idx] = self.m0_beta
                        self.keep["P0_beta"][keep_idx] = float(np.exp(self.logP0_beta))
                    elif self.trend_mode == "deterministic":
                        self.keep["m0_beta"][keep_idx] = self.m0_beta
                        self.keep["P0_beta"][keep_idx] = 0.0

                if self.seasonal_mode == "dynamic":
                    self.keep["Q_gamma"][keep_idx] = self.s_gamma**2
                    self.keep["lambda_gamma"][keep_idx] = float(self.lambda_gamma or 0.0)
                    self.keep["m0_gamma"][keep_idx, :] = self.m0_gamma
                    self.keep["P0_gamma"][keep_idx] = float(np.exp(self.logP0_gamma))

                if "x" in self.keep and self.dim > 0:
                    self.keep["x"][keep_idx, :, :] = self.x[1: self.T + 1, :]

                if self.level_mode == "deterministic":
                    self.keep["m0_alpha"][keep_idx] = self.m0_alpha
                if self.trend_mode == "deterministic":
                    self.keep["m0_beta"][keep_idx] = self.m0_beta
                if self.seasonal_mode == "deterministic":
                    self.keep["m0_gamma"][keep_idx, :] = self.m0_gamma

                keep_idx += 1

        return self.keep

    # ------------------------------- Persistence ---------------------------- #
    def save_posterior(self, out_npz_path: str, extra_meta: Optional[dict] = None) -> None:
        _ensure_dir(os.path.dirname(out_npz_path))
        arrays = dict(self.keep)
        arrays["y"] = self.y.copy()
        if "x" not in arrays:
            arrays["x"] = np.zeros((0, 0, 0))

        # truth overlays (if registered)
        if self.true_sigma is not None:
            arrays["true_sigma"] = float(self.true_sigma)
        if self.true_Q is not None:
            arrays["true_Q"] = np.asarray(self.true_Q, float)
        if self.true_mu_t is not None:
            arrays["true_mu_t"] = np.asarray(self.true_mu_t, float)
        if self.true_alpha_t is not None:
            arrays["true_alpha_t"] = np.asarray(self.true_alpha_t, float)
        if self.true_beta_t is not None:
            arrays["true_beta_t"] = np.asarray(self.true_beta_t, float)
        if self.true_gamma_t is not None:
            arrays["true_gamma_t"] = np.asarray(self.true_gamma_t, float)

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
    import sys

    import matplotlib.pyplot as plt
    import pandas as pd

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from simulator.mean_time_series import Mean_Time_Series  # assumes it's importable

    def _parse_date(s: str | None):
        if not s:
            return datetime.today()
        parts = [int(p) for p in s.split("-")]
        if len(parts) == 1:
            return datetime(parts[0], 1, 1)
        if len(parts) == 2:
            return datetime(parts[0], parts[1], 1)
        if len(parts) == 3:
            return datetime(parts[0], parts[1], parts[2])
        raise ValueError("start-date must be YYYY, YYYY-MM, or YYYY-MM-DD")

    p = argparse.ArgumentParser(
        description=(
            "Kalman-FFBS + Gibbs/MH for Gaussian DLM with PC priors on s (λ hyperpriors), "
            "Normal on m0 & log P0 (dynamic), Normal on log σ. Data simulated via Mean_Time_Series."
        )
    )

    # Simulation
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--period", type=int, default=12)
    p.add_argument("--start-date", type=str, default="2000-01-01")

    p.add_argument("--level-mode", choices=["dynamic", "deterministic", "none"], default="dynamic")
    p.add_argument("--trend-mode", choices=["dynamic", "deterministic", "none"], default="deterministic")
    p.add_argument("--seasonal-mode", choices=["dynamic", "deterministic", "none"], default="deterministic")

    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--q-level", type=float, default=0.05)
    p.add_argument("--q-trend", type=float, default=0.002)
    p.add_argument("--q-season", type=float, default=0.15)

    p.add_argument("--m0-level", type=float, default=5.0)
    p.add_argument("--v0-level", type=float, default=0.25)
    p.add_argument("--m0-trend", type=float, default=0.015)
    p.add_argument("--v0-trend", type=float, default=0.05)
    p.add_argument("--m0-season", type=str, default=None, help="comma-separated list for m0_season (length p-1)")
    p.add_argument("--v0-season", type=str, default=None, help="comma-separated list for v0_season (length p-1)")

    # Inference priors
    p.add_argument("--prior-m-sigma", type=float, default=0.0)
    p.add_argument("--prior-s-sigma", type=float, default=1.0)
    p.add_argument("--prior-m-m0-alpha", type=float, default=0.0)
    p.add_argument("--prior-s-m0-alpha", type=float, default=10.0)
    p.add_argument("--prior-m-m0-beta", type=float, default=0.0)
    p.add_argument("--prior-s-m0-beta", type=float, default=10.0)
    p.add_argument("--prior-m-m0-gamma", type=str, default=None, help="comma-separated list for m0_gamma prior (length p-1)")
    p.add_argument("--prior-s-m0-gamma", type=float, default=10, help="stddev for m0_gamma prior (shared for p-1 components)")
    p.add_argument("--prior-m-logP0-alpha", type=float, default=0.0)
    p.add_argument("--prior-s-logP0-alpha", type=float, default=1.0)
    p.add_argument("--prior-m-logP0-beta", type=float, default=0.0)
    p.add_argument("--prior-s-logP0-beta", type=float, default=1.0)
    p.add_argument("--prior-m-logP0-gamma", type=float, default=0.0)
    p.add_argument("--prior-s-logP0-gamma", type=float, default=1.0)

    # PC priors + (optional) Gamma hyperpriors on λ
    p.add_argument("--pc-frac-alpha", type=float, default=0.10)
    p.add_argument("--pc-frac-beta", type=float, default=0.10)
    p.add_argument("--pc-frac-gamma", type=float, default=0.10)
    p.add_argument("--pc-alpha-prob", type=float, default=0.05)
    p.add_argument("--pc-lambda-alpha", type=float, default=None)
    p.add_argument("--pc-lambda-beta", type=float, default=None)
    p.add_argument("--pc-lambda-gamma", type=float, default=None)
    p.add_argument("--pc-a-lambda-alpha", type=float, default=1.0)
    p.add_argument("--pc-b-lambda-alpha", type=float, default=1.0)
    p.add_argument("--pc-a-lambda-beta", type=float, default=1.0)
    p.add_argument("--pc-b-lambda-beta", type=float, default=1.0)
    p.add_argument("--pc-a-lambda-gamma", type=float, default=1.0)
    p.add_argument("--pc-b-lambda-gamma", type=float, default=1.0)

    # Sampler config & init
    p.add_argument("--n-iter", type=int, default=4000)
    p.add_argument("--burn", type=int, default=1000)
    p.add_argument("--thin", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--progress", default=True)
    p.add_argument("--progress-every", type=int, default=0)
    p.add_argument("--step-logsigma", type=float, default=0.15)
    p.add_argument("--step-level", type=float, default=0.05)
    p.add_argument("--step-slope", type=float, default=0.02)
    p.add_argument("--step-season", type=float, default=2)
    p.add_argument("--adapt-steps", default=True)
    p.add_argument("--adapt-every", type=int, default=25)
    p.add_argument("--adapt-until", choices=["burn", "all"], default="burn")
    p.add_argument("--adapt-eta0", type=float, default=0.05)
    p.add_argument("--adapt-decay", type=float, default=0.75)
    p.add_argument("--adapt-target-1d", type=float, default=0.44)
    p.add_argument("--step-min", type=float, default=1e-5)
    p.add_argument("--step-max", type=float, default=1.0)
    p.add_argument("--slice-w", type=float, default=0.4)
    p.add_argument("--slice-m", type=int, default=40)
    p.add_argument("--slice-max-shrink", type=int, default=1000)

    # Initial values (optional)
    p.add_argument("--logsigma-init", type=float, default=0.0)
    p.add_argument("--s-alpha-init", type=float, default=1e-2)
    p.add_argument("--s-beta-init", type=float, default=1e-3)
    p.add_argument("--s-gamma-init", type=float, default=1.0)
    p.add_argument("--m0-gamma-init", type=str, default=None, help="comma-separated list for m0_gamma init (length p-1)")
    p.add_argument("--logP0-alpha-init", type=float, default=0.0)
    p.add_argument("--logP0-beta-init", type=float, default=0.0)
    p.add_argument("--logP0-gamma-init", type=float, default=0.0)

    # I/O
    p.add_argument("--out-dir", type=str, default="results/simulations/DLM")

    # Plotting & prints
    p.add_argument("--plot", default=True)
    p.add_argument("--print-summary", default=True)

    args = p.parse_args()
    np.random.seed(args.seed)
    
    start_date = _parse_date(args.start_date)
    if args.m0_season is None:
        args.m0_season = np.random.normal(0.0, 10.0, size=args.period - 1).tolist()
    if args.v0_season is None:
        args.v0_season = [0.25] * (args.period - 1)
    if args.prior_m_m0_gamma is None:
        args.prior_m_m0_gamma = [0.0] * (args.period - 1)



    # ---- Simulate data ----
    # NOTE: simulator does not support level_mode="none". We emulate "none" by
    # using deterministic level with fixed zero (same effect for μ_t).
    sim_level_mode = args.level_mode if args.level_mode != "none" else "deterministic"

    mts = Mean_Time_Series(
        sigma=args.sigma,
        level_mode=sim_level_mode,
        trend_mode=args.trend_mode,
        seasonal_mode=args.seasonal_mode,
        period=args.period,
        q_level=(args.q_level if sim_level_mode == "dynamic" else 0.0),
        q_trend=(args.q_trend if args.trend_mode == "dynamic" else 0.0),
        q_season=(args.q_season if args.seasonal_mode == "dynamic" else 0.0),
        m0_level=(0.0 if args.level_mode == "none" else args.m0_level),
        v0_level=(args.v0_level if sim_level_mode == "dynamic" else 0.0),
        m0_trend=(0.0 if args.trend_mode == "none" else args.m0_trend),
        v0_trend=(args.v0_trend if args.trend_mode == "dynamic" else 0.0),
        m0_season=args.m0_season,
        v0_season=args.v0_season,
        start_date=start_date,
    )

    y = []
    for _ in range(args.T):
        mts.move()
        y.append(mts.measure())
    y = np.asarray(y, float)

    truths = mts.get_truth_paths(as_numpy=True)
    mu_T = truths["mu_t"][1: 1 + args.T]
    alpha_T = truths["alpha_t"][1: 1 + args.T]
    beta_T = truths["beta_t"][1: 1 + args.T]
    gamma_T = truths["gamma_t"][1: 1 + args.T]
    dates_T = truths["index"][: args.T]


    priors = Priors(
        m_sigma=float(args.prior_m_sigma),
        s_sigma=float(args.prior_s_sigma),
        m_m0_alpha=float(args.prior_m_m0_alpha),
        s_m0_alpha=float(args.prior_s_m0_alpha),
        m_m0_beta=float(args.prior_m_m0_beta),
        s_m0_beta=float(args.prior_s_m0_beta),
        m_m0_gamma=args.prior_m_m0_gamma,
        s_m0_gamma=float(args.prior_s_m0_gamma),
        m_logP0_alpha=float(args.prior_m_logP0_alpha),
        s_logP0_alpha=float(args.prior_s_logP0_alpha),
        m_logP0_beta=float(args.prior_m_logP0_beta),
        s_logP0_beta=float(args.prior_s_logP0_beta),
        m_logP0_gamma=float(args.prior_m_logP0_gamma),
        s_logP0_gamma=float(args.prior_s_logP0_gamma),
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
        progress_every=int(args.progress_every),
        step_logsigma=float(args.step_logsigma),
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
        slice_max_shrink=int(args.slice_max_shrink),
    )
    
    if args.m0_gamma_init is None:
        S = np.array([np.median(y[k::args.period]) for k in range(args.period)], float)
        m0_gamma_init = (S - S.mean())[:-1]

    sampler = DLMGibbs(
        y=y,
        period=int(args.period),
        level_mode=args.level_mode,
        trend_mode=args.trend_mode,
        seasonal_mode=args.seasonal_mode,
        logsigma_init=float(args.logsigma_init),
        s_alpha_init=float(args.s_alpha_init),
        s_beta_init=float(args.s_beta_init),
        s_gamma_init=float(args.s_gamma_init),
        m0_alpha_init=(0.0 if args.level_mode == "none" else float(args.m0_level)),
        logP0_alpha_init=float(args.logP0_alpha_init),
        m0_beta_init=float(args.m0_trend if args.trend_mode != "none" else 0.0),
        logP0_beta_init=float(args.logP0_beta_init),
        m0_gamma_init=args.m0_gamma_init,
        logP0_gamma_init=float(args.logP0_gamma_init),
        priors=priors,
        cfg=cfg,
    )

    # Store truths for overlays (optional)
    sampler.set_truth(sigma=mts.sigma, Q=(mts.q_level, mts.q_trend, mts.q_season),
                      m0_level=mts.m0_level, m0_trend=mts.m0_trend, m0_season=mts.m0_season,
                      P0_level=mts.v0_level, P0_trend=mts.v0_trend, P0_season=mts.v0_season)
    sampler.set_truth_paths(
        mu=mu_T,
        alpha=(alpha_T if args.level_mode == "dynamic" else None),
        beta=(beta_T if args.trend_mode == "dynamic" else None),
        gamma=(gamma_T if args.seasonal_mode == "dynamic" else None),
    )
    
    # ---- Print quick summaries ----
    if args.print_summary:
        with np.printoptions(suppress=True, precision=4):
            print("\n--- Summary (simulation) ---")
            print(f"level_mode={mts.level_mode}, trend_mode={mts.trend_mode}, seasonal_mode={mts.seasonal_mode}")
            print(f"sigma(sim)={mts.sigma}, q_level={mts.q_level if mts.level_mode=='dynamic' else 0.0}, "
                  f"q_trend={mts.q_trend if mts.trend_mode=='dynamic' else 0.0}, "
                  f"q_season={mts.q_season if mts.seasonal_mode=='dynamic' else 0.0}")
            if mts.level_mode == "none":
                print("m0_level=0.0 (none), P0_level=0.0 (none)")
            else:
                print(f"m0_level={mts.m0_level}, P0_level={mts.v0_level if mts.level_mode=='dynamic' else 0.0}")
            if mts.trend_mode == "none":
                print("m0_trend=0.0 (none), P0_trend=0.0 (none)")
            else:
                print(f"m0_trend={(0.0 if mts.trend_mode=='none' else mts.m0_trend)}, "
                      f"P0_trend={mts.v0_trend if mts.trend_mode=='dynamic' else 0.0}")
            if mts.seasonal_mode == "none":
                print(f"m0_season={[0.0]*(mts.period-1)} (none), P0_season=0.0 (none)")
            else:
                m0_season_str = np.array2string(np.array(args.m0_season), precision=2, suppress_small=True)
                if args.seasonal_mode == "dynamic":
                    P0_season_str = np.array2string(np.array(args.v0_season), precision=2, suppress_small=True)
                else:
                    P0_season_str = "0.0"
                print(f"m0_season={m0_season_str}, P0_season={P0_season_str}")
            print(f"period={args.period}, start={dates_T[0]}, end={dates_T[-1]}")
            print(f"y mean={y.mean():.3f}, sd={y.std(ddof=1):.3f}")

    # ---- Run ----
    t0 = time.time()
    post = sampler.run()
    elapsed = time.time() - t0
    print(f"Run time: {elapsed:.2f}s")

    # ---- Save posterior ----
    out_dir = os.path.join(
        args.out_dir, f"{args.level_mode}-{args.trend_mode}-{args.seasonal_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    _ensure_dir(out_dir)
    sampler.save_posterior(
        out_npz_path=os.path.join(out_dir, "posterior.npz"),
        extra_meta={"elapsed_seconds": float(elapsed)},
    )

    # ---- Pack and optionally save data frame ----
    df = pd.DataFrame(
        {
            "date": dates_T,
            "y_t": y,
            "mu_t_truth": mu_T,
            "alpha_t_truth": alpha_T,
            "beta_t_truth": beta_T,
            "gamma_t_truth": gamma_T,
        }
    )

    # ---- Print quick summaries ----
    if args.print_summary:
        with np.printoptions(suppress=True, precision=4):
            print("\n--- Summary (simulation) ---")
            print(f"level_mode={mts.level_mode}, trend_mode={mts.trend_mode}, seasonal_mode={mts.seasonal_mode}")
            print(f"sigma(sim)={mts.sigma}, q_level={mts.q_level if mts.level_mode=='dynamic' else 0.0}, "
                  f"q_trend={mts.q_trend if mts.trend_mode=='dynamic' else 0.0}, "
                  f"q_season={mts.q_season if mts.seasonal_mode=='dynamic' else 0.0}")
            if mts.level_mode == "none":
                print("m0_level=0.0 (none), P0_level=0.0 (none)")
            else:
                print(f"m0_level={mts.m0_level}, P0_level={mts.v0_level if mts.level_mode=='dynamic' else 0.0}")
            if mts.trend_mode == "none":
                print("m0_trend=0.0 (none), P0_trend=0.0 (none)")
            else:
                print(f"m0_trend={(0.0 if mts.trend_mode=='none' else mts.m0_trend)}, "
                      f"P0_trend={mts.v0_trend if mts.trend_mode=='dynamic' else 0.0}")
            if mts.seasonal_mode == "none":
                print(f"m0_season={[0.0]*(mts.period-1)} (none), P0_season=0.0 (none)")
            else:
                m0_season_str = np.array2string(np.array(args.m0_season), precision=2, suppress_small=True)
                if args.seasonal_mode == "dynamic":
                    P0_season_str = np.array2string(np.array(args.v0_season), precision=2, suppress_small=True)
                else:
                    P0_season_str = "0.0"
                print(f"m0_season={m0_season_str}, P0_season={P0_season_str}")
            print(f"period={args.period}, start={dates_T[0]}, end={dates_T[-1]}")
            print(f"y mean={y.mean():.3f}, sd={y.std(ddof=1):.3f}")

        print("\n--- Summary (posterior means) ---")
        print(f"σ: {np.mean(post['sigma']):.4g}")
        if "Q_alpha" in post:
            m = float(np.mean(post["Q_alpha"])); print(f"Q_alpha: {m:.4g}  (√Q_alpha ≈ {math.sqrt(m):.4g})")
        else:
            if args.level_mode == "deterministic":
                print("Q_alpha: n/a (level deterministic)")
            else:
                print("Q_alpha: n/a (level none)")
        if "Q_beta" in post:
            m = float(np.mean(post["Q_beta"])); print(f"Q_beta:  {m:.4g}  (√Q_beta  ≈ {math.sqrt(m):.4g})")
        else:
            if args.trend_mode == "deterministic":
                print("Q_beta: n/a (trend deterministic)")
            else:
                print("Q_beta: n/a (trend none)")
        if "Q_gamma" in post:
            m = float(np.mean(post["Q_gamma"])); print(f"Q_gamma: {m:.4g}  (√Q_gamma ≈ {math.sqrt(m):.4g})")
        else:
            if args.seasonal_mode == "deterministic":
                print("Q_gamma: n/a (season deterministic)")
            else:
                print("Q_gamma: n/a (season none)")
        if "lambda_alpha" in post:
            print(f"λ_alpha: {np.mean(post['lambda_alpha']):.4g}")
        if "lambda_beta" in post:
            print(f"λ_beta:  {np.mean(post['lambda_beta']):.4g}")
        if "lambda_gamma" in post:
            print(f"λ_gamma: {np.mean(post['lambda_gamma']):.4g}")
        if "m0_alpha" in post:
            if args.level_mode == "dynamic":
                print(f"m0_alpha: {np.mean(post['m0_alpha']):.4g}, P0_alpha: {np.mean(post['P0_alpha']):.4g}")
            elif args.level_mode == "deterministic":
                print(f"m0_alpha: {np.mean(post['m0_alpha']):.4g} (det), P0_alpha: n/a (det)")
        else:
            if args.level_mode == "none":
                print("m0_alpha: n/a (level none)")
        if "m0_beta" in post:
            if args.trend_mode == "dynamic":
                print(f"m0_beta:  {np.mean(post['m0_beta']):.4g}, P0_beta:  {np.mean(post['P0_beta']):.4g}")
            elif args.trend_mode == "deterministic":
                print(f"m0_beta:  {np.mean(post['m0_beta']):.4g} (det), P0_beta: n/a (det)")
        else:
            if args.trend_mode == "none":
                print("m0_beta: n/a (trend none)")
        if "m0_gamma" in post:
            means = np.mean(post['m0_gamma'], axis=0)
            means = np.roll(means, 1)[:-1]  # put the "zero" at the end for printing
            if args.seasonal_mode == "dynamic":
                print(f"m0_gamma: {means}, P0_gamma: {np.mean(post['P0_gamma']):.4g}")
            elif args.seasonal_mode == "deterministic":
                print(f"m0_gamma: {np.array2string(means, precision=2, suppress_small=True)}, (det), P0_gamma: n/a (det)")
        else:
            if args.seasonal_mode == "none":
                print("m0_gamma: n/a (season none)")

    # ---- Optional plots ----
    if args.plot:
        mu_hat = post["mu"].mean(axis=0)
        plt.figure(figsize=(10, 4))
        plt.plot(dates_T, y, label=r"$y_t$", linewidth=1.0)
        plt.plot(dates_T, mu_T, "--", label=r"$\mu_t$ (truth)", linewidth=1.0)
        plt.plot(dates_T, mu_hat, "-.", label=r"$\hat{\mu}_t$ (post mean)", linewidth=1.0)
        ttl = f"DLM: level={args.level_mode}, trend={args.trend_mode}, season={args.seasonal_mode}"
        plt.title(ttl)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
