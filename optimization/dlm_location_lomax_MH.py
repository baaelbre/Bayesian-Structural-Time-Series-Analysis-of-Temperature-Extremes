from __future__ import annotations

import json, math, os, warnings
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

# =============================================================================
# Small utils
# =============================================================================

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
    for k in range(3):  # try a couple of jitters if needed
        try:
            L = np.linalg.cholesky(M + (eps * (10**k)) * I)
            Y = np.linalg.solve(L, B)         # L Y = B
            return np.linalg.solve(L.T, Y)    # L^T X = Y
        except np.linalg.LinAlgError:
            continue
    return np.linalg.pinv(M) @ B  # rare fallback

# =============================================================================
# Priors & Config
# =============================================================================

@dataclass
class PCScalePrior:
    """
    Collapsed PC prior for s > 0 with λ ~ Gamma(a, b):
      s | a,b  ~ Lomax/Pareto-II(shape=a, scale=b),  p(s) ∝ (b + s)^{-(a+1)}.
    'frac' and 'alpha_prob' are heuristics for picking sensible defaults.
    """
    a: float = 1.0     # shape
    b: float = 1.0     # scale
    frac: float = 0.10 # u = frac * scale_proxy (heuristic)
    alpha_prob: float = 0.05  # P(s > u) target (heuristic)

@dataclass
class Priors:
    # Observation variance σ²: precision τ = 1/σ² ~ Gamma(a_sigma, b_sigma) (shape–rate)
    a_sigma: float = 2.0
    b_sigma: float = 1.0

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

    # Collapsed PC scale priors (Lomax) for process sds
    pc_alpha: PCScalePrior = field(default_factory=PCScalePrior)
    pc_beta:  PCScalePrior = field(default_factory=PCScalePrior)
    pc_gamma: PCScalePrior = field(default_factory=PCScalePrior)

@dataclass
class MHAdaptConfig:
    # Whether to adapt the RW step sizes at all
    adapt: bool = True
    # Restrict adaptation to burn-in only?
    burn_only: bool = True
    # Iteration range (1-indexed) where adaptation is allowed; None means unbounded
    start_iter: int = 50
    stop_iter: Optional[int] = None
    # How often to adapt (in iterations)
    adapt_every: int = 10
    # Target acceptance (Robbins–Monro target)
    target_acc: float = 0.25
    # Robbins–Monro gain schedule: gamma = c / (it + t0)^pow
    rm_c: float = 1.0
    rm_t0: float = 10.0
    rm_pow: float = 0.5  # 0.5 ≈ 1/sqrt(t) schedule
    # Bounds on the (log-sd) proposal step size
    min_step: float = 1e-4
    max_step: float = 2.5
    # Optional initial steps (per component)
    init_step_alpha: Optional[float] = None
    init_step_beta: Optional[float] = None
    init_step_gamma: Optional[float] = None
    # EWMA decay for acceptance rate display (closer to 1 = slower)
    acc_ewma_decay: float = 0.9
    # Rolling window size for quick online ESS display
    ess_window: int = 800

@dataclass
class SamplerConfig:
    n_iter: int = 4000
    burn: int = 1000
    thin: int = 1
    random_seed: Optional[int] = 7
    progress: bool = True
    progress_every: int = 0  # 0 => ~2% of n_iter

    # MH adaptation settings
    mh: MHAdaptConfig = field(default_factory=MHAdaptConfig)

# =============================================================================
# DLM Sampler (conjugate Gibbs except process s via MH with Lomax/Pareto II prior)
# =============================================================================
class DLMGibbsConjugate:
    """
    Gaussian structural DLM with:
      • FFBS for latent states
      • Conjugate Gibbs for σ², m0, P0, deterministic params
      • Collapsed PC prior (Lomax) for process s (sample log s via RW-MH)

    STATE ORDERING & SEASONAL CONVENTION:
      - Dynamic state layout: [alpha] [beta] [g1 ... g_{p-1}]
      - Seasonal vector is NEWEST-FIRST: [γ_t, γ_{t-1}, ..., γ_{t-(p-2)}]
      - Observation loads the FIRST seasonal coord (γ_t).
      - Transition for season:
            g1(t) = -sum(g1..g_{p-1})(t-1) + ε_{γ,t}
            gk(t) = g_{k-1}(t-1),  k=2..p-1
        ⇒ Q has s_γ² on the FIRST seasonal coord only.
    """

    # --------------------------- Construction --------------------------- #
    def __init__(
        self,
        y: np.ndarray,
        period: int,
        level_mode: str = "dynamic",
        trend_mode: str = "dynamic",
        seasonal_mode: str = "dynamic",
        # initial values
        m0_alpha_init: float = 0.0,
        P0_alpha_init: float = 1.0,
        m0_beta_init: float = 0.0,
        P0_beta_init: float = 1.0,
        m0_gamma_init: Optional[Sequence[float]] = None,  # len p-1 if dynamic (NEWEST-FIRST)
        P0_gamma_init: float = 1.0,
        sigma2_init: float = 1.0,
        s_alpha_init: float = 1e-2,
        s_beta_init: float = 1e-3,
        s_gamma_init: float = 1e-3,
        priors: Priors = Priors(),
        cfg: SamplerConfig = SamplerConfig(),
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
        self.idx_beta = layout.index("beta") if "beta" in layout else None
        if self.seasonal_mode == "dynamic":
            self.idx_g_start = layout.index("g1")
            self.idx_g_end = self.idx_g_start + (self.period - 2)
        else:
            self.idx_g_start = self.idx_g_end = None

        # Parameters
        self.sigma2 = float(sigma2_init)  # observation variance
        self.s_alpha = float(s_alpha_init) if self.idx_alpha is not None else 0.0
        self.s_beta  = float(s_beta_init)  if self.idx_beta  is not None else 0.0
        self.s_gamma = float(s_gamma_init) if self.seasonal_mode == "dynamic" else 0.0

        # Initial m0 and P0 for dynamic coords (P0 on variance scale)
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
                self.m0_gamma = g  # assumed NEWEST-FIRST
            self.P0_gamma = float(P0_gamma_init)
        else:
            self.m0_gamma = None
            self.P0_gamma = 0.0

        # Deterministic contributions (outside state)
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
            # full length-p vector with sum-zero (for direct indexing in μ_det)
            self.m0_gamma = np.r_[base, -float(np.sum(base))].astype(float)

        # Latent path
        self.x = np.zeros((self.T + 1, self.dim), float)
        if self.dim > 0:
            m0_vec, P0_diag = self._current_m0_P0()
            self.x[0] = np.random.multivariate_normal(
                m0_vec, np.diag(P0_diag) + 1e-10 * np.eye(self.dim)
            )
            self._propagate_initial_path(Q_init=np.full(self.dim, 1e-6, float))

        # Storage
        self.keep: Dict[str, np.ndarray] = {}

        # Optional “truth” overlays
        self.true_sigma: Optional[float] = None
        self.true_Q: Optional[np.ndarray] = None
        self.true_mu_t: Optional[np.ndarray] = None
        self.true_alpha_t: Optional[np.ndarray] = None
        self.true_beta_t: Optional[np.ndarray] = None
        self.true_gamma_t: Optional[np.ndarray] = None

        # MH state and traces
        self._mh_state = {
            "alpha": {"step": 0.2, "acc": 0.25, "n_prop": 0, "n_acc": 0},
            "beta":  {"step": 0.2, "acc": 0.25, "n_prop": 0, "n_acc": 0},
            "gamma": {"step": 0.2, "acc": 0.25, "n_prop": 0, "n_acc": 0},
            "it": 1,
        }
        # apply configured initial step sizes if provided
        if self.priors and self.cfg and hasattr(self.cfg, "mh"):
            mhc = self.cfg.mh
            if mhc.init_step_alpha is not None: self._mh_state["alpha"]["step"] = float(mhc.init_step_alpha)
            if mhc.init_step_beta  is not None: self._mh_state["beta"]["step"]  = float(mhc.init_step_beta)
            if mhc.init_step_gamma is not None: self._mh_state["gamma"]["step"] = float(mhc.init_step_gamma)

        # rolling traces for quick ESS display (store Q = s^2)
        self._mh_trace = {"alpha": [], "beta": [], "gamma": []}

        # Print collapsed PC prior heuristics (init-scale only; no lambdas)
        if self.cfg.progress:
            y = self.y
            sd1 = _robust_sd(np.diff(y)) if y.size >= 2 else 0.0
            sd2 = _robust_sd(np.diff(y, n=2)) if y.size >= 3 else 0.0
            sdg = sd1
            fmt = lambda x: "n/a" if x is None else f"{x:.4g}"
            print(f"[init] scale proxies: sd1={fmt(sd1)}, sd2={fmt(sd2)}, sdg={fmt(sdg)}")

    # --------------------- Truth overlays (optional) --------------------- #
    def set_truth(
        self,
        sigma: Optional[float] = None,
        Q: Optional[Tuple[float, float, float]] = None,
        m0_level: Optional[float] = None,
        m0_trend: Optional[float] = None,
        m0_season: Optional[Sequence[float]] = None,
        P0_level: Optional[float] = None,
        P0_trend: Optional[float] = None,
        P0_season: Optional[Sequence[float]] = None,
    ) -> None:
        """Attach ground-truth scalars for saving/diagnostics (optional)."""
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
        """Attach full truth paths for saving/diagnostics (optional)."""
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
        if self.seasonal_mode == "dynamic":
            h[self.idx_g_start] = 1.0  # FIRST coord = γ_t
        return h.reshape(1, -1)

    def _A(self) -> np.ndarray:
        if self.dim == 0:
            return np.zeros((0, 0))
        A = np.eye(self.dim)
        if self.idx_alpha is not None and self.idx_beta is not None:
            A[self.idx_alpha, self.idx_beta] = 1.0
        if self.seasonal_mode == "dynamic":
            gs, ge = self.idx_g_start, self.idx_g_end
            K = ge - gs + 1  # = p-1
            A[gs, gs:ge+1] = -1.0                 # first row: minus sum
            A[gs+1:ge+1, gs:ge] = np.eye(K-1)     # shift-down
            A[gs+1:ge+1, ge] = 0.0
        return A

    def _u(self) -> np.ndarray:
        if self.dim == 0:
            return np.zeros(0, float)
        u = np.zeros(self.dim, float)
        if (self.idx_alpha is not None) and (self.trend_mode == "deterministic"):
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
            Q[self.idx_g_start, self.idx_g_start] = self.s_gamma**2  # only first seasonal coord
        return Q

    def _mu_det(self, t: int) -> float:
        out = 0.0
        if self.level_mode == "deterministic":
            out += self.m0_alpha
        if (self.trend_mode == "deterministic") and (self.idx_alpha is None):
            out += self.m0_beta * t
        if self.seasonal_mode == "deterministic":
            out += float(self.m0_gamma[t % self.period])
        return out

    def _current_m0_P0(self) -> Tuple[np.ndarray, np.ndarray]:
        m0, P0 = [], []
        if self.idx_alpha is not None:
            m0.append(self.m0_alpha); P0.append(self.P0_alpha)
        if self.idx_beta is not None:
            m0.append(self.m0_beta);  P0.append(self.P0_beta)
        if self.seasonal_mode == "dynamic":
            m0.extend(list(self.m0_gamma))
            P0.extend([self.P0_gamma] * (self.period - 1))
        return np.asarray(m0, float), np.asarray(P0, float)

    # ------------------------- FFBS (Kalman + Carter–Kohn) ------------------------- #
    def _ffbs(self) -> np.ndarray:
        if self.dim == 0:
            return self.x.copy()
        H, A, Q, R = self._H(), self._A(), self._Q(), float(self.sigma2)
        m0_vec, P0_diag = self._current_m0_P0()
        m = np.zeros((self.T + 1, self.dim))
        C = np.zeros((self.T + 1, self.dim, self.dim))
        a = np.zeros((self.T + 1, self.dim))
        Rm = np.zeros((self.T + 1, self.dim, self.dim))
        m[0] = m0_vec
        C[0] = np.diag(P0_diag) + 1e-12 * np.eye(self.dim)
        u = self._u()

        # forward
        for t in range(1, self.T + 1):
            a[t]  = A @ m[t - 1] + u
            Rm[t] = A @ C[t - 1] @ A.T + Q
            Rm[t] = 0.5 * (Rm[t] + Rm[t].T) + 1e-12 * np.eye(self.dim)

            resid_mean = float(self.y[t - 1] - self._mu_det(t - 1))
            S = float(H @ Rm[t] @ H.T + R)
            if S <= 0:
                S = float(H @ (Rm[t] + 1e-10 * np.eye(self.dim)) @ H.T + R)
            K = (Rm[t] @ H.T) / S
            v = resid_mean - float(H @ a[t])
            m[t] = a[t] + (K.flatten() * v)
            C[t] = Rm[t] - K @ (H @ Rm[t])
            C[t] = 0.5 * (C[t] + C[t].T) + 1e-12 * np.eye(self.dim)

        # backward
        x = np.zeros_like(self.x)
        x[self.T] = np.random.multivariate_normal(m[self.T], C[self.T])
        for t in range(self.T - 1, -1, -1):
            J = C[t] @ A.T
            J = J @ _spd_solve(Rm[t + 1], np.eye(self.dim))
            mean = m[t] + J @ (x[t + 1] - (A @ m[t] + u))
            cov  = C[t] - J @ Rm[t + 1] @ J.T
            cov  = 0.5 * (cov + cov.T)
            cov += max(0.0, 1e-12 - float(np.linalg.eigvalsh(cov).min())) * np.eye(cov.shape[0])
            x[t] = np.random.multivariate_normal(mean, cov)
        return x

    def _propagate_initial_path(self, Q_init: np.ndarray) -> None:
        if self.dim == 0:
            return
        A, u = self._A(), self._u()
        for t in range(1, self.T + 1):
            self.x[t] = A @ self.x[t - 1] + u + np.random.normal(0.0, np.sqrt(Q_init), size=self.dim)

    # ------------------ Helpers: μ and residuals ------------------ #
    def _mu_vec(self) -> np.ndarray:
        H = self._H()
        mu = np.zeros(self.T, float)
        for t in range(1, self.T + 1):
            dyn = float(H @ self.x[t]) if self.dim > 0 else 0.0
            mu[t - 1] = self._mu_det(t - 1) + dyn
        return mu

    # ------------------ σ² | rest  (Gamma on precision) ------------------ #
    def update_sigma2(self) -> None:
        e = self.y - self._mu_vec()
        a = self.priors.a_sigma + 0.5 * self.T
        b = self.priors.b_sigma + 0.5 * float(e @ e)
        tau = np.random.gamma(shape=a, scale=1.0 / b)  # precision
        self.sigma2 = 1.0 / max(tau, 1e-300)

    # ------------- Innovation sums of squares (for s updates) ------------- #
    def _innovation_ss_alpha(self) -> Tuple[float, int]:
        if self.idx_alpha is None:
            return 0.0, 0
        ss = 0.0
        for t in range(1, self.T + 1):
            drift = 0.0
            if self.idx_beta is not None:           # dynamic trend
                drift = self.x[t - 1, self.idx_beta]
            elif self.trend_mode == "deterministic": # static slope
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
        if self.seasonal_mode != "dynamic":
            return 0.0, 0
        gs, ge = self.idx_g_start, self.idx_g_end
        ss = 0.0
        for t in range(1, self.T + 1):
            prev = self.x[t - 1, gs : ge + 1]
            mean_new_first = -float(np.sum(prev))
            ss += (self.x[t, gs] - mean_new_first) ** 2
        return float(ss), self.T

    # =============================================================================
    # Adaptive MH for log s under collapsed PC (Lomax) prior
    # =============================================================================
    def _logpost_logsd_lomax(self, z: float, SS: float, T_eff: int, a: float, b: float) -> float:
        """
        Log posterior (up to constant) for z = log s under Lomax(a, b) prior:
            p(s) ∝ (b + s)^(-(a+1)),  s = e^z.
        """
        return (
            -T_eff * z
            - 0.5 * SS * math.exp(-2 * z)
            - (a + 1.0) * math.log(b + math.exp(z))
            + z
        )

    def _mh_logsd_adapt(
        self,
        z_curr: float,
        SS: float,
        T_eff: int,
        a: float,
        b: float,
        step_scale: float,
        acc_ewma: float,
        it_global: int,
        *,
        do_adapt: bool,
        mhc: MHAdaptConfig,
    ) -> Tuple[float, float, float, int]:
        """
        One RW-MH step for log s with optional adaptation.
        Returns (z_new, step_new, acc_ewma_new, accepted_int)
        """
        z_prop = z_curr + np.random.normal(0.0, step_scale)
        lp_curr = self._logpost_logsd_lomax(z_curr, SS, T_eff, a, b)
        lp_prop = self._logpost_logsd_lomax(z_prop, SS, T_eff, a, b)
        log_acc = lp_prop - lp_curr

        if math.log(np.random.rand()) < log_acc:
            accepted, z_new = 1, z_prop
        else:
            accepted, z_new = 0, z_curr

        # EWMA acceptance (for display)
        acc_ewma = mhc.acc_ewma_decay * acc_ewma + (1.0 - mhc.acc_ewma_decay) * accepted

        # Adapt step size only if allowed by schedule
        step_new = step_scale
        if do_adapt:
            gamma = mhc.rm_c / ((it_global + mhc.rm_t0) ** mhc.rm_pow)
            delta = accepted - mhc.target_acc
            step_new = step_scale * math.exp(gamma * delta)
            step_new = float(np.clip(step_new, mhc.min_step, mhc.max_step))

        return z_new, step_new, acc_ewma, accepted

    def update_process_sds(self) -> None:
        """
        Adaptive MH for s_alpha, s_beta, s_gamma (log-scale proposals).
        Tunable via self.cfg.mh. Keeps EWMA acceptance and rolling ESS traces.
        """
        st = self._mh_state
        it = st["it"]
        mhc: MHAdaptConfig = self.cfg.mh

        # Should we adapt on this iteration?
        def _allow_adapt() -> bool:
            if not mhc.adapt:
                return False
            if mhc.burn_only and it > self.cfg.burn:
                return False
            if mhc.start_iter is not None and it < mhc.start_iter:
                return False
            if mhc.stop_iter is not None and it > mhc.stop_iter:
                return False
            if mhc.adapt_every <= 1:
                return True
            return (it % mhc.adapt_every) == 0

        do_adapt = _allow_adapt()

        # α component
        if self.idx_alpha is not None:
            ss, T_eff = self._innovation_ss_alpha()
            pa = self.priors.pc_alpha
            z0 = math.log(max(1e-18, self.s_alpha))
            z1, st["alpha"]["step"], st["alpha"]["acc"], acc_a = self._mh_logsd_adapt(
                z0, ss, T_eff, pa.a, pa.b, st["alpha"]["step"], st["alpha"]["acc"], it,
                do_adapt=do_adapt, mhc=mhc
            )
            self.s_alpha = math.exp(z1)
            st["alpha"]["n_prop"] += 1
            st["alpha"]["n_acc"]  += acc_a
            tr = self._mh_trace["alpha"]; tr.append(self.s_alpha**2)
            if len(tr) > mhc.ess_window: del tr[:len(tr)-mhc.ess_window]

        # β component
        if self.idx_beta is not None:
            ss, T_eff = self._innovation_ss_beta()
            pb = self.priors.pc_beta
            z0 = math.log(max(1e-18, self.s_beta))
            z1, st["beta"]["step"], st["beta"]["acc"], acc_b = self._mh_logsd_adapt(
                z0, ss, T_eff, pb.a, pb.b, st["beta"]["step"], st["beta"]["acc"], it,
                do_adapt=do_adapt, mhc=mhc
            )
            self.s_beta = math.exp(z1)
            st["beta"]["n_prop"] += 1
            st["beta"]["n_acc"]  += acc_b
            tr = self._mh_trace["beta"]; tr.append(self.s_beta**2)
            if len(tr) > mhc.ess_window: del tr[:len(tr)-mhc.ess_window]

        # γ component
        if self.seasonal_mode == "dynamic":
            ss, T_eff = self._innovation_ss_gamma()
            pg = self.priors.pc_gamma
            z0 = math.log(max(1e-18, self.s_gamma))
            z1, st["gamma"]["step"], st["gamma"]["acc"], acc_g = self._mh_logsd_adapt(
                z0, ss, T_eff, pg.a, pg.b, st["gamma"]["step"], st["gamma"]["acc"], it,
                do_adapt=do_adapt, mhc=mhc
            )
            self.s_gamma = math.exp(z1)
            st["gamma"]["n_prop"] += 1
            st["gamma"]["n_acc"]  += acc_g
            tr = self._mh_trace["gamma"]; tr.append(self.s_gamma**2)
            if len(tr) > mhc.ess_window: del tr[:len(tr)-mhc.ess_window]

        st["it"] += 1

    # --- m0 | P0, x0  (Normal)  +  P0 | m0, x0  (Inv-Gamma) for dynamic coords --- #
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
                raise ValueError("priors.m_m0_gamma must be length p-1 (newest-first)")
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

    # --- Deterministic params (all conjugate) --- #
    def update_deterministic_params(self) -> None:
        # Intercept (level) if deterministic
        if self.level_mode == "deterministic":
            r = self.y.copy()
            if self.dim > 0:
                H = self._H()
                for t in range(1, self.T + 1):
                    r[t - 1] -= float(H @ self.x[t])
            if self.seasonal_mode == "deterministic":
                r -= self.m0_gamma[np.arange(self.T) % self.period]
            s2 = float(self.sigma2)
            m0, s0 = float(self.priors.m_m0_alpha), float(self.priors.s_m0_alpha)
            prec = self.T / s2 + 1.0 / (s0**2)
            mean = ((r.sum() / s2) + m0 / (s0**2)) / prec
            var = 1.0 / prec
            self.m0_alpha = float(np.random.normal(mean, math.sqrt(var)))

        # Deterministic slope:
        if self.trend_mode == "deterministic":
            if self.idx_alpha is not None:
                # α_t = α_{t-1} + β + ε_αt  ⇒  Δα_t ~ N(β, s_α^2)
                d = self.x[1:, self.idx_alpha] - self.x[:-1, self.idx_alpha]
                s2 = float(self.s_alpha**2) if self.s_alpha > 0 else 1e-12
                m0, s0 = float(self.priors.m_m0_beta), float(self.priors.s_m0_beta)
                prec = (self.T / s2) + 1.0 / (s0**2)
                mean = ((float(np.sum(d)) / s2) + m0 / (s0**2)) / prec
                var = 1.0 / prec
                self.m0_beta = float(np.random.normal(mean, math.sqrt(var)))
            else:
                # No dynamic α: observation-based regression
                t = np.arange(self.T, dtype=float)
                r = self.y.copy()
                if self.dim > 0:
                    H = self._H()
                    for k in range(1, self.T + 1):
                        r[k - 1] -= float(H @ self.x[k])
                if self.level_mode == "deterministic":
                    r -= self.m0_alpha
                if self.seasonal_mode == "deterministic":
                    r -= self.m0_gamma[np.arange(self.T) % self.period]
                m0, s0 = float(self.priors.m_m0_beta), float(self.priors.s_m0_beta)
                sig2 = float(self.sigma2)
                prec = (t @ t) / sig2 + 1.0 / (s0**2)
                mean = ((t @ r) / sig2 + m0 / (s0**2)) / prec
                var = 1.0 / prec
                self.m0_beta = float(np.random.normal(mean, math.sqrt(var)))

        # Deterministic seasonality (p-1 free coeffs → full p with sum-zero)
        if self.seasonal_mode == "deterministic":
            if not hasattr(self, "_Z_season"):
                midx = np.arange(self.T) % self.period
                K = self.period - 1
                Z = np.zeros((self.T, K))
                for k in range(K):
                    Z[:, k] = (midx == k).astype(float) - (midx == K).astype(float)
                self._Z_season = Z

            r = self.y.copy()
            if self.dim > 0:
                H = self._H()
                for t in range(1, self.T + 1):
                    r[t - 1] -= float(H @ self.x[t])
            if self.level_mode == "deterministic":
                r -= self.m0_alpha
            if (self.idx_alpha is None) and (self.trend_mode == "deterministic"):
                r -= self.m0_beta * np.arange(self.T, dtype=float)

            K = self.period - 1
            m_prior = (
                np.zeros(K)
                if self.priors.m_m0_gamma is None
                else np.asarray(self.priors.m_m0_gamma, float).reshape(-1)
            )
            s2 = float(self.priors.s_m0_gamma) ** 2
            Z = self._Z_season
            sig2 = float(self.sigma2)
            Prec = (Z.T @ Z) / sig2 + np.eye(K) / s2
            b = (Z.T @ r) / sig2 + m_prior / s2
            mu = np.linalg.solve(Prec, b)
            L = np.linalg.cholesky(Prec)
            z = np.random.randn(K)
            theta = mu + np.linalg.solve(L.T, z)
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

    @staticmethod
    def _ess_lag1(x: Sequence[float]) -> float:
        """Very quick ESS via lag-1 autocorrelation; returns len(x) if undefined."""
        x = np.asarray(x, float)
        n = x.size
        if n < 3:
            return float(n)
        x0, x1 = x[:-1], x[1:]
        v0 = np.var(x0)
        v1 = np.var(x1)
        if v0 <= 0 or v1 <= 0:
            return float(n)
        r = np.corrcoef(x0, x1)[0, 1]
        r = float(np.clip(r, -0.99, 0.99))
        return float(n) * (1.0 - r) / (1.0 + r)

    def _progress_line(self, it: int) -> str:
        parts = [f"[it {it + 1}/{self.cfg.n_iter}]"]
        parts.append(f"σ={math.sqrt(self.sigma2):.3f}")
        if self.idx_alpha is not None:
            parts.append(f"Qα={self.s_alpha**2:.4g}")
        if self.idx_beta is not None:
            parts.append(f"Qβ={self.s_beta**2:.4g}")
        if self.seasonal_mode == "dynamic":
            parts.append(f"Qγ={self.s_gamma**2:.4g}")
        if self.level_mode != "none":
            parts.append(
                f"m0α={self.m0_alpha:.4g} "
                f"P0α={(self.P0_alpha if self.level_mode=='dynamic' else 0.0):.4g}"
            )
        if self.trend_mode != "none":
            parts.append(
                f"m0β={self.m0_beta:.4g} "
                f"P0β={(self.P0_beta if self.trend_mode=='dynamic' else 0.0):.4g}"
            )
        if self.seasonal_mode != "none":
            g = self._fmt_list(
                (self.m0_gamma if self.seasonal_mode == "dynamic" else self.m0_gamma[:-1]), 6, ".4g"
            )
            parts.append(f"m0γ={g} P0γ={(self.P0_gamma if self.seasonal_mode=='dynamic' else 0.0):.4g}")
        return " | ".join(parts)

    def _progress_line_mh(self) -> str:
        """Compact line with MH tuning stats: step, acc(EWMA), ESS(window)."""
        items = []
        for nm in ["alpha", "beta", "gamma"]:
            if nm == "alpha" and (self.idx_alpha is None):  continue
            if nm == "beta"  and (self.idx_beta is None):   continue
            if nm == "gamma" and (self.seasonal_mode != "dynamic"): continue
            st = self._mh_state[nm]
            step = st["step"]
            acc  = st["acc"]
            tr   = self._mh_trace[nm]
            ess  = int(self._ess_lag1(tr)) if len(tr) >= 5 else len(tr)
            items.append(f"{nm[0]}: step={step:.3f} acc={acc:.2f} ESS≈{ess}")
        return "  [MH] " + " | ".join(items) if items else "  [MH] -"

    # --------------------------------- MCMC --------------------------------- #
    def run(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        save_iters = list(range(cfg.burn, cfg.n_iter, cfg.thin))
        n_kept, keep_idx = len(save_iters), 0

        # allocate storage
        self.keep = {"sigma": np.zeros(n_kept, float), "mu": np.zeros((n_kept, self.T), float)}
        if self.idx_alpha is not None:
            self.keep.update({"Q_alpha": np.zeros(n_kept),
                              "m0_alpha": np.zeros(n_kept), "P0_alpha": np.zeros(n_kept)})
        if self.idx_beta is not None:
            self.keep.update({"Q_beta": np.zeros(n_kept),
                              "m0_beta": np.zeros(n_kept), "P0_beta": np.zeros(n_kept)})
        if self.seasonal_mode == "dynamic":
            self.keep.update({"Q_gamma": np.zeros(n_kept),
                              "m0_gamma": np.zeros((n_kept, self.period - 1)), "P0_gamma": np.zeros(n_kept),
                              "x": np.zeros((n_kept, self.T, self.dim))})
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
            # 1) FFBS
            if self.dim > 0:
                self.x = self._ffbs()

            # 2) process s (RW-MH with collapsed PC prior)
            if self.dim > 0:
                self.update_process_sds()

            # 3) m0 (Gibbs) and 4) P0 (Gibbs Inv-Gamma)
            if self.dim > 0:
                self.update_m0()
                self.update_P0()

            # 5) deterministic params (all conjugate)
            self.update_deterministic_params()

            # 6) σ² (Gibbs via precision)
            self.update_sigma2()

            # progress
            if cfg.progress and ((it + 1) % print_every == 0 or it == cfg.n_iter - 1):
                print(self._progress_line(it))
                print(self._progress_line_mh())

            # save
            if it in save_iters:
                mu = self._mu_vec()
                self.keep["mu"][keep_idx, :] = mu
                self.keep["sigma"][keep_idx] = math.sqrt(self.sigma2)
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
                    self.keep["m0_alpha"][keep_idx] = self.m0_alpha
                if self.trend_mode == "deterministic":
                    self.keep["m0_beta"][keep_idx] = self.m0_beta
                if self.seasonal_mode == "deterministic":
                    self.keep["m0_gamma"][keep_idx, :] = self.m0_gamma
                keep_idx += 1

        return self.keep

    # ------------------------------- Persistence ---------------------------- #
    def save_posterior(self, out_npz_path: str, extra_meta: Optional[dict] = None) -> None:
        os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)
        arrays = dict(self.keep)
        arrays["y"] = self.y.copy()
        if "x" not in arrays:
            arrays["x"] = np.zeros((0, 0, 0))
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
    import argparse, os, sys, time, math
    from datetime import datetime
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    # --------------------------------------------------------------------------
    # Setup and imports
    # --------------------------------------------------------------------------
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(base_dir)
    from simulator.mean_time_series import Mean_Time_Series  # newest-first convention

    # --------------------------------------------------------------------------
    # Helper parsing utilities
    # --------------------------------------------------------------------------
    def _parse_date(s: str | None):
        if not s:
            return datetime.today()
        parts = [int(p) for p in s.split("-")]
        if len(parts) == 1:
            return datetime(parts[0], 1, 1)
        elif len(parts) == 2:
            return datetime(parts[0], parts[1], 1)
        elif len(parts) == 3:
            return datetime(parts[0], parts[1], parts[2])
        raise ValueError("start-date must be YYYY, YYYY-MM, or YYYY-MM-DD")

    def _csv_floats_or_none(s: str | None):
        if s is None:
            return None
        s = s.strip()
        if s == "":
            return None
        return [float(z) for z in s.split(",") if z.strip() != ""]

    def _window_to_decay(w: int) -> float:
        """Map an 'EWMA window' intuition to an EWMA decay in (0,1)."""
        w = max(2, int(w))
        return 1.0 - 2.0 / (w + 1)

    # --------------------------------------------------------------------------
    # Command-line arguments
    # --------------------------------------------------------------------------
    p = argparse.ArgumentParser(
        description=(
            "Kalman FFBS + conjugate/PC Gibbs for Gaussian DLM "
            "(level/trend/season). Uses newest-first seasonal ordering "
            "and adaptive MH random-walk for process s (collapsed PC prior)."
        )
    )

    # Simulation
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--period", type=int, default=4)
    p.add_argument("--start-date", type=str, default="2000-01-01")
    p.add_argument("--level-mode", choices=["dynamic", "deterministic"], default="dynamic")
    p.add_argument("--trend-mode", choices=["dynamic", "deterministic", "none"], default="dynamic")
    p.add_argument("--seasonal-mode", choices=["dynamic", "deterministic", "none"], default="dynamic")
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--q-level", type=float, default=0.001)
    p.add_argument("--q-trend", type=float, default=0.00002)
    p.add_argument("--q-season", type=float, default=0.005)
    p.add_argument("--m0-level", type=float, default=3.0)
    p.add_argument("--v0-level", type=float, default=0.25)
    p.add_argument("--m0-trend", type=float, default=0.015)
    p.add_argument("--v0-trend", type=float, default=0.05)
    p.add_argument("--m0-season", type=str, default=None)
    p.add_argument("--v0-season", type=str, default=None)

    # Priors and PC parameters
    p.add_argument("--prior-a-sigma", type=float, default=2.0)
    p.add_argument("--prior-b-sigma", type=float, default=1.0)
    p.add_argument("--prior-m-m0-alpha", type=float, default=0.0)
    p.add_argument("--prior-s-m0-alpha", type=float, default=10.0)
    p.add_argument("--prior-m-m0-beta", type=float, default=0.0)
    p.add_argument("--prior-s-m0-beta", type=float, default=10.0)
    p.add_argument("--prior-m-m0-gamma", type=str, default=None)
    p.add_argument("--prior-s-m0-gamma", type=float, default=5.0)
    p.add_argument("--prior-a-P0-alpha", type=float, default=5.0)
    p.add_argument("--prior-b-P0-alpha", type=float, default=1.0)
    p.add_argument("--prior-a-P0-beta", type=float, default=5.0)
    p.add_argument("--prior-b-P0-beta", type=float, default=1.0)
    p.add_argument("--prior-a-P0-gamma", type=float, default=5.0)
    p.add_argument("--prior-b-P0-gamma", type=float, default=1.0)
    p.add_argument("--pc-a-alpha", type=float, default=4.0)
    p.add_argument("--pc-b-alpha", type=float, default=0.5)
    p.add_argument("--pc-a-beta", type=float, default=4.0)
    p.add_argument("--pc-b-beta", type=float, default=0.5)
    p.add_argument("--pc-a-gamma", type=float, default=4.0)
    p.add_argument("--pc-b-gamma", type=float, default=0.5)
    p.add_argument("--pc-frac", type=float, default=0.10)
    p.add_argument("--pc-alpha-prob", type=float, default=0.05)

    # Sampler configuration (general)
    p.add_argument("--n-iter", type=int, default=8000)
    p.add_argument("--burn", type=int, default=4000)
    p.add_argument("--thin", type=int, default=2)
    p.add_argument("--seed", type=int, default=40)
    p.add_argument("--progress", default=True)
    p.add_argument("--progress-every", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="results/simulations/DLM")
    p.add_argument("--plot", default=True)
    p.add_argument("--print-summary", default=True)

    # Initial inference values
    p.add_argument("--sigma-init", type=float, default=1.0)
    p.add_argument("--s-alpha-init", type=float, default=1e-1)
    p.add_argument("--s-beta-init", type=float, default=1e-1)
    p.add_argument("--s-gamma-init", type=float, default=1e-1)
    p.add_argument("--m0-gamma-init", type=str, default=None)
    p.add_argument("--P0-alpha-init", type=float, default=0.25)
    p.add_argument("--P0-beta-init", type=float, default=0.05)
    p.add_argument("--P0-gamma-init", type=float, default=0.25)

    # --------------------- Adaptive MH step-size controls ----------------------
    p.add_argument("--mh-target", type=float, default=0.44,
                   help="Target acceptance rate for RW-MH on log s.")
    p.add_argument("--mh-init-step-alpha", type=float, default=0.15)
    p.add_argument("--mh-init-step-beta",  type=float, default=0.15)
    p.add_argument("--mh-init-step-gamma", type=float, default=0.15)
    p.add_argument("--mh-min-step", type=float, default=1e-4)
    p.add_argument("--mh-max-step", type=float, default=5.0)
    p.add_argument("--mh-adapt-every", type=int, default=10,
                   help="Adaptation cadence in iterations (1=every iter).")
    p.add_argument("--mh-adapt-window", type=int, default=50,
                   help="EWMA window (or batch size) for acceptance smoothing.")
    p.add_argument("--mh-adapt-burn-only", default=True,
                   help="If set, adapt only during burn-in; freeze afterward.")
    p.add_argument("--mh-adapt-until", type=int, default=0,
                   help="Optional hard stop iter for adaptation (0=>auto).")

    args = p.parse_args()
    np.random.seed(args.seed)

    # --------------------------------------------------------------------------
    # Simulate a synthetic time series
    # --------------------------------------------------------------------------
    start_date = _parse_date(args.start_date)
    m0_season = _csv_floats_or_none(args.m0_season) or [1.0] * (args.period - 1)
    v0_season = _csv_floats_or_none(args.v0_season) or [0.25] * (args.period - 1)

    mts = Mean_Time_Series(
        sigma=args.sigma,
        period=args.period,
        level_mode=args.level_mode,
        trend_mode=args.trend_mode,
        seasonal_mode=args.seasonal_mode,
        q_level=args.q_level,
        q_trend=args.q_trend,
        q_season=args.q_season,
        m0_level=args.m0_level,
        v0_level=args.v0_level,
        m0_trend=args.m0_trend,
        v0_trend=args.v0_trend,
        m0_season=m0_season,
        v0_season=v0_season,
        start_date=start_date,
    )

    y = np.array([mts.move() or mts.measure() for _ in range(args.T)], float)
    truths = mts.get_truth_paths(as_numpy=True)
    mu_T = truths["mu_t"][1:1 + args.T]
    dates_T = truths["index"][:args.T]

    # --------------------------------------------------------------------------
    # Build priors
    # --------------------------------------------------------------------------
    pri_gamma_vec = _csv_floats_or_none(args.prior_m_m0_gamma)
    priors = Priors(
        a_sigma=args.prior_a_sigma, b_sigma=args.prior_b_sigma,
        m_m0_alpha=args.prior_m_m0_alpha, s_m0_alpha=args.prior_s_m0_alpha,
        m_m0_beta=args.prior_m_m0_beta, s_m0_beta=args.prior_s_m0_beta,
        m_m0_gamma=None if pri_gamma_vec is None else pri_gamma_vec,
        s_m0_gamma=args.prior_s_m0_gamma,
        a_P0_alpha=args.prior_a_P0_alpha, b_P0_alpha=args.prior_b_P0_alpha,
        a_P0_beta=args.prior_a_P0_beta, b_P0_beta=args.prior_b_P0_beta,
        a_P0_gamma=args.prior_a_P0_gamma, b_P0_gamma=args.prior_b_P0_gamma,
        pc_alpha=PCScalePrior(a=args.pc_a_alpha, b=args.pc_b_alpha,
                              frac=args.pc_frac, alpha_prob=args.pc_alpha_prob),
        pc_beta=PCScalePrior(a=args.pc_a_beta, b=args.pc_b_beta,
                             frac=args.pc_frac, alpha_prob=args.pc_alpha_prob),
        pc_gamma=PCScalePrior(a=args.pc_a_gamma, b=args.pc_b_gamma,
                              frac=args.pc_frac, alpha_prob=args.pc_alpha_prob),
    )

    # --------------------------------------------------------------------------
    # Sampler config: build MHAdaptConfig, then SamplerConfig(mh=...)
    # --------------------------------------------------------------------------
    mh_cfg = MHAdaptConfig(
        adapt=True,
        burn_only=bool(args.mh_adapt_burn_only),
        start_iter=1,
        stop_iter=(None if args.mh_adapt_until <= 0 else int(args.mh_adapt_until)),
        adapt_every=max(1, int(args.mh_adapt_every)),
        target_acc=float(args.mh_target),
        rm_c=1.0,
        rm_t0=10.0,
        rm_pow=0.5,
        min_step=float(args.mh_min_step),
        max_step=float(args.mh_max_step),
        init_step_alpha=float(args.mh_init_step_alpha),
        init_step_beta=float(args.mh_init_step_beta),
        init_step_gamma=float(args.mh_init_step_gamma),
        acc_ewma_decay=_window_to_decay(int(args.mh_adapt_window)),
        ess_window=800,
    )

    cfg = SamplerConfig(
        n_iter=int(args.n_iter), burn=int(args.burn), thin=int(args.thin),
        random_seed=int(args.seed), progress=bool(args.progress),
        progress_every=int(args.progress_every),
        mh=mh_cfg,
    )

    # --------------------------------------------------------------------------
    # Instantiate sampler
    # --------------------------------------------------------------------------
    m0_gamma_init = (
        [float(z) for z in args.m0_gamma_init.split(",")] if args.m0_gamma_init else None
    )

    sampler = DLMGibbsConjugate(
        y=y, period=args.period,
        level_mode=args.level_mode, trend_mode=args.trend_mode, seasonal_mode=args.seasonal_mode,
        sigma2_init=args.sigma_init ** 2,
        s_alpha_init=args.s_alpha_init, s_beta_init=args.s_beta_init, s_gamma_init=args.s_gamma_init,
        m0_alpha_init=args.m0_level, m0_beta_init=args.m0_trend,
        P0_alpha_init=args.P0_alpha_init, P0_beta_init=args.P0_beta_init,
        m0_gamma_init=m0_gamma_init, P0_gamma_init=args.P0_gamma_init,
        priors=priors, cfg=cfg,
    )

    sampler.set_truth(
        sigma=mts.sigma, Q=(mts.q_level, mts.q_trend, mts.q_season),
        m0_level=mts.m0_level, m0_trend=mts.m0_trend, m0_season=mts.m0_season,
        P0_level=mts.v0_level, P0_trend=mts.v0_trend, P0_season=mts.v0_season,
    )
    sampler.set_truth_paths(mu=mu_T)

    # ------------------------------ Run banner --------------------------------
    if args.print_summary:
        print(f"\nSimulated {args.T} observations (σ={mts.sigma}) with modes "
              f"{args.level_mode}/{args.trend_mode}/{args.seasonal_mode}.\n")
        m = cfg.mh
        print("Adaptive MH settings:")
        print(f"  target={m.target_acc:.3f}, cadence=every {m.adapt_every} iters, "
              f"{'burn-in only' if m.burn_only else 'adapt throughout'}, "
              f"start_iter={m.start_iter}, stop_iter={m.stop_iter or 'auto'}")
        print(f"  init steps: α={m.init_step_alpha:.3f}, β={m.init_step_beta:.3f}, "
              f"γ={m.init_step_gamma:.3f} (clamped to [{m.min_step}, {m.max_step}])\n")

    # --------------------------------------------------------------------------
    # Run sampler
    # --------------------------------------------------------------------------
    t0 = time.time()
    post = sampler.run()
    elapsed = time.time() - t0
    print(f"\n[Run completed in {elapsed:.1f}s]")

    out_dir = os.path.join(
        args.out_dir,
        f"{args.level_mode}-{args.trend_mode}-{args.seasonal_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(out_dir, exist_ok=True)
    sampler.save_posterior(
        out_npz_path=os.path.join(out_dir, "posterior.npz"),
        extra_meta={
            "elapsed_seconds": float(elapsed),
            "mh_settings": {
                "target": m.target_acc,
                "init_step_alpha": m.init_step_alpha,
                "init_step_beta": m.init_step_beta,
                "init_step_gamma": m.init_step_gamma,
                "min_step": m.min_step,
                "max_step": m.max_step,
                "adapt_every": m.adapt_every,
                "acc_ewma_decay": m.acc_ewma_decay,
                "burn_only": bool(m.burn_only),
                "start_iter": m.start_iter,
                "stop_iter": m.stop_iter,
            },
        },
    )

    # --------------------------------------------------------------------------
    # Summary output and optional plotting
    # --------------------------------------------------------------------------
    if args.print_summary:
        print("\n--- Posterior means ---")
        print(f"σ = {np.mean(post['sigma']):.4f}")
        for k in ["alpha", "beta", "gamma"]:
            key = f"Q_{k}"
            if key in post:
                mQ = np.mean(post[key])
                print(f"Q_{k} = {mQ:.4g} (√Q ≈ {math.sqrt(mQ):.4g})")

        # MH specifics (final steps + smoothed acc)
        if hasattr(sampler, "_mh_state"):
            print("\n--- MH specifics (final) ---")
            for k in ["alpha", "beta", "gamma"]:
                if k in sampler._mh_state:
                    sst = sampler._mh_state[k]
                    step = sst.get("step", float("nan"))
                    acc  = sst.get("acc", float("nan"))
                    nacc = sst.get("n_acc", None)
                    natt = sst.get("n_prop", None)
                    ratio = (f"{nacc}/{natt}" if (nacc is not None and natt is not None) else "—")
                    print(f"  {k}: step={step:.4f}, acc≈{acc:.3f} ({ratio})")

    if args.plot:
        mu_hat = post["mu"].mean(axis=0)
        plt.figure(figsize=(10, 4))
        plt.plot(dates_T, y, label="y_t", lw=1)
        plt.plot(dates_T, mu_T, "--", label="μ_t (truth)")
        plt.plot(dates_T, mu_hat, "-.", label="μ̂_t (post mean)")
        plt.title(f"DLM ({args.level_mode}/{args.trend_mode}/{args.seasonal_mode})")
        plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()
