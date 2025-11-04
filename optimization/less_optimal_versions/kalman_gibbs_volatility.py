from __future__ import annotations

import json, math, os, time, warnings
from dataclasses import dataclass, asdict, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

# =============================================================================
# Small utils (as before)
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
# Priors & Config (compatible with your previous code; add sigma-block PC priors)
# =============================================================================
@dataclass
class PCPrior:
    lambda_s: Optional[float] = None
    a_lambda: float = 1.0
    b_lambda: float = 1.0
    frac: float = 0.10
    alpha_prob: float = 0.05   # P(s > u) = alpha_prob with u = frac * scale_proxy

@dataclass
class Priors:
    # obs variance precision tau ~ Gamma(a_sigma,b_sigma); only used if σ is constant
    a_sigma: float = 2.0
    b_sigma: float = 1.0

    # m0 priors (mean block and deterministic components)
    m_m0_alpha: float = 0.0; s_m0_alpha: float = 10.0
    m_m0_beta: float = 0.0;  s_m0_beta: float  = 10.0
    m_m0_gamma: Optional[Sequence[float]] = None
    s_m0_gamma: float = 5.0

    # Initial-state variances P0 ~ InvGamma(a,b)
    a_P0_alpha: float = 2.0; b_P0_alpha: float = 1.0
    a_P0_beta: float  = 2.0; b_P0_beta: float  = 1.0
    a_P0_gamma: float = 2.0; b_P0_gamma: float = 1.0

    # === NEW: η (log σ) priors if dynamic ===
    m_m0_alpha_sig: float = 0.0; s_m0_alpha_sig: float = 5.0
    m_m0_beta_sig: float  = 0.0; s_m0_beta_sig:  float = 5.0
    m_m0_gamma_sig: Optional[Sequence[float]] = None
    s_m0_gamma_sig: float = 3.0

    a_P0_alpha_sig: float = 2.0; b_P0_alpha_sig: float = 1.0
    a_P0_beta_sig: float  = 2.0; b_P0_beta_sig:  float = 1.0
    a_P0_gamma_sig: float = 2.0; b_P0_gamma_sig: float = 1.0

    # PC priors (mean block)
    pc_alpha: PCPrior = field(default_factory=PCPrior)
    pc_beta:  PCPrior = field(default_factory=PCPrior)
    pc_gamma: PCPrior = field(default_factory=PCPrior)

    # PC priors (sigma/log-sigma block)
    pc_alpha_sig: PCPrior = field(default_factory=PCPrior)
    pc_beta_sig:  PCPrior = field(default_factory=PCPrior)
    pc_gamma_sig: PCPrior = field(default_factory=PCPrior)

@dataclass
class SamplerConfig:
    n_iter: int = 4000
    burn: int = 1000
    thin: int = 1
    random_seed: Optional[int] = 7
    progress: bool = True
    progress_every: int = 0

    # Slice (log s only)
    slice_w: float = 0.4
    slice_m: int = 40
    slice_max_shrink: int = 1000

    # === RBPF config ===
    n_particles: int = 64
    ess_resample: float = 0.5  # resample when ESS/N < this
    resample_method: str = "systematic"  # or "multinomial"

# =============================================================================
# RBPF building blocks (sigma block dynamic)
# =============================================================================
def _season_A(period: int) -> np.ndarray:
    p = int(period); m = p - 1
    if m <= 0: return np.zeros((0,0))
    A = np.zeros((m, m), float)
    A[0, :] = -1.0
    if m > 1:
        A[1:, :-1] = np.eye(m - 1)
    return A

class RBPFHetero:
    """
    Rao–Blackwellized PF for a structural DLM with dynamic log-scale η.
    - Particles carry η-state (level/trend/seasonal over log σ_t).
    - For each particle, the mean block (μ-state) is conditionally linear-Gaussian ⇒
      run a Kalman filter/smoother for μ given σ_t = exp(η_t).
    - Parameter updates follow your PC prior + slice/IGamma scheme using the most
      recent sampled paths (μ and η innovations).
    """
    # --------------------------- Construction --------------------------- #
    def __init__(
        self,
        y: np.ndarray,
        period: int,
        # μ block modes
        level_mode: str = "dynamic",
        trend_mode: str = "dynamic",
        seasonal_mode: str = "dynamic",
        # η block modes (log-sigma). If all "none/deterministic", RBPF not needed.
        level_mode_sigma: str = "dynamic",
        trend_mode_sigma: str = "none",
        seasonal_mode_sigma: str = "none",
        # Initial values (μ block)
        m0_alpha_init: float = 0.0, P0_alpha_init: float = 1.0,
        m0_beta_init: float  = 0.0, P0_beta_init: float  = 1.0,
        m0_gamma_init: Optional[Sequence[float]] = None, P0_gamma_init: float = 1.0,
        s_alpha_init: float = 1e-2, s_beta_init: float = 1e-3, s_gamma_init: float = 1e-3,
        # Initial values (η block)
        m0_alpha_sig_init: float = 0.0, P0_alpha_sig_init: float = 0.25,
        m0_beta_sig_init: float  = 0.0, P0_beta_sig_init: float  = 0.05,
        m0_gamma_sig_init: Optional[Sequence[float]] = None, P0_gamma_sig_init: float = 0.25,
        s_alpha_sig_init: float = 1e-2, s_beta_sig_init: float = 1e-3, s_gamma_sig_init: float = 1e-3,
        # Priors / config
        priors: Priors = Priors(),
        cfg: SamplerConfig = SamplerConfig(),
    ):
        self.y = np.asarray(y, float)
        self.T = int(self.y.size)
        self.period = int(period)
        if self.period < 2:
            raise ValueError("period must be >= 2")

        # Modes
        ok = {"dynamic", "deterministic", "none"}
        for m in (level_mode, trend_mode, seasonal_mode,
                  level_mode_sigma, trend_mode_sigma, seasonal_mode_sigma):
            if m not in ok:
                raise ValueError("invalid mode")
        if trend_mode == "dynamic" and level_mode != "dynamic":
            raise ValueError("dynamic trend requires dynamic level")
        if trend_mode_sigma == "dynamic" and level_mode_sigma != "dynamic":
            raise ValueError("dynamic η-trend requires dynamic η-level")

        self.level_mode, self.trend_mode, self.seasonal_mode = level_mode, trend_mode, seasonal_mode
        self.level_mode_sig, self.trend_mode_sig, self.seasonal_mode_sig = (
            level_mode_sigma, trend_mode_sigma, seasonal_mode_sigma
        )

        self.priors, self.cfg = priors, cfg
        if cfg.random_seed is not None:
            np.random.seed(cfg.random_seed)

        # μ layout
        layout_mu: List[str] = []
        if level_mode == "dynamic": layout_mu.append("alpha")
        if trend_mode == "dynamic": layout_mu.append("beta")
        if seasonal_mode == "dynamic": layout_mu.extend([f"g{k}" for k in range(1, period)])
        self.layout_mu = layout_mu
        self.dim_mu = len(layout_mu)
        self.idx_alpha = layout_mu.index("alpha") if "alpha" in layout_mu else None
        self.idx_beta  = layout_mu.index("beta")  if "beta"  in layout_mu else None
        if seasonal_mode == "dynamic":
            self.idx_g_start = layout_mu.index("g1")
            self.idx_g_end   = self.idx_g_start + (period - 2)
        else:
            self.idx_g_start = self.idx_g_end = None

        # η layout
        layout_sig: List[str] = []
        if level_mode_sigma == "dynamic": layout_sig.append("alpha_sig")
        if trend_mode_sigma == "dynamic": layout_sig.append("beta_sig")
        if seasonal_mode_sigma == "dynamic": layout_sig.extend([f"gs{k}" for k in range(1, period)])
        self.layout_sig = layout_sig
        self.dim_sig = len(layout_sig)
        self.idx_alpha_sig = layout_sig.index("alpha_sig") if "alpha_sig" in layout_sig else None
        self.idx_beta_sig  = layout_sig.index("beta_sig")  if "beta_sig"  in layout_sig else None
        if seasonal_mode_sigma == "dynamic":
            self.idx_gs_start = layout_sig.index("gs1")
            self.idx_gs_end   = self.idx_gs_start + (period - 2)
        else:
            self.idx_gs_start = self.idx_gs_end = None

        # Process sds (μ)
        self.s_alpha = float(s_alpha_init) if self.idx_alpha is not None else 0.0
        self.s_beta  = float(s_beta_init)  if self.idx_beta  is not None else 0.0
        self.s_gamma = float(s_gamma_init) if seasonal_mode == "dynamic" else 0.0
        # Process sds (η)
        self.s_alpha_sig = float(s_alpha_sig_init) if self.idx_alpha_sig is not None else 0.0
        self.s_beta_sig  = float(s_beta_sig_init)  if self.idx_beta_sig  is not None else 0.0
        self.s_gamma_sig = float(s_gamma_sig_init) if seasonal_mode_sigma == "dynamic" else 0.0

        # PC λ inits
        self.lambda_alpha, self.lambda_beta, self.lambda_gamma = self._init_pc(self.y)
        self.lambda_alpha_sig, self.lambda_beta_sig, self.lambda_gamma_sig = self._init_pc_sigma(self.y)

        # Initial m0/P0 (μ)
        self.m0_alpha = float(m0_alpha_init) if self.idx_alpha is not None else 0.0
        self.P0_alpha = float(P0_alpha_init) if self.idx_alpha is not None else 0.0
        self.m0_beta  = float(m0_beta_init)  if self.idx_beta  is not None else 0.0
        self.P0_beta  = float(P0_beta_init)  if self.idx_beta  is not None else 0.0
        if seasonal_mode == "dynamic":
            if m0_gamma_init is None:
                self.m0_gamma = np.zeros(self.period - 1, float)
            else:
                g = np.asarray(m0_gamma_init, float); assert g.size == self.period - 1
                self.m0_gamma = g
            self.P0_gamma = float(P0_gamma_init)
        else:
            self.m0_gamma = None; self.P0_gamma = 0.0

        # Initial m0/P0 (η)
        self.m0_alpha_sig = float(m0_alpha_sig_init) if self.idx_alpha_sig is not None else 0.0
        self.P0_alpha_sig = float(P0_alpha_sig_init) if self.idx_alpha_sig is not None else 0.0
        self.m0_beta_sig  = float(m0_beta_sig_init)  if self.idx_beta_sig  is not None else 0.0
        self.P0_beta_sig  = float(P0_beta_sig_init)  if self.idx_beta_sig  is not None else 0.0
        if seasonal_mode_sigma == "dynamic":
            if m0_gamma_sig_init is None:
                self.m0_gamma_sig = np.zeros(self.period - 1, float)
            else:
                gs = np.asarray(m0_gamma_sig_init, float); assert gs.size == self.period - 1
                self.m0_gamma_sig = gs
            self.P0_gamma_sig = float(P0_gamma_sig_init)
        else:
            self.m0_gamma_sig = None; self.P0_gamma_sig = 0.0

        # Deterministic contributions (μ)
        if self.level_mode == "deterministic": self.m0_alpha = float(self.priors.m_m0_alpha)
        if self.trend_mode == "deterministic": self.m0_beta  = float(self.priors.m_m0_beta)
        if self.seasonal_mode == "deterministic":
            base = (np.zeros(self.period - 1, float)
                    if self.priors.m_m0_gamma is None
                    else np.asarray(self.priors.m_m0_gamma, float))
            last = -float(np.sum(base)); self.m0_gamma = np.r_[base, last]
        # Deterministic contributions (η)
        if self.level_mode_sig == "deterministic": self.m0_alpha_sig = float(self.priors.m_m0_alpha_sig)
        if self.trend_mode_sig == "deterministic": self.m0_beta_sig  = float(self.priors.m_m0_beta_sig)
        if self.seasonal_mode_sig == "deterministic":
            base = (np.zeros(self.period - 1, float)
                    if self.priors.m_m0_gamma_sig is None
                    else np.asarray(self.priors.m_m0_gamma_sig, float))
            last = -float(np.sum(base)); self.m0_gamma_sig = np.r_[base, last]

        # Storage for kept draws
        self.keep: Dict[str, np.ndarray] = {}

    # --------------------- PC λ initialization --------------------- #
    def _init_pc(self, y: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        sd1 = _robust_sd(np.diff(y)) if y.size >= 2 else 0.0
        sd2 = _robust_sd(np.diff(y, n=2)) if y.size >= 3 else 0.0
        def _cal(pc: PCPrior, sd: float) -> float:
            u = max(1e-12, pc.frac * max(1e-12, sd))
            return float(-math.log(max(1e-12, pc.alpha_prob)) / u)
        la = (self.priors.pc_alpha.lambda_s if self.priors.pc_alpha.lambda_s is not None
              else (_cal(self.priors.pc_alpha, sd1) if self.idx_alpha is not None else None))
        lb = (self.priors.pc_beta.lambda_s if self.priors.pc_beta.lambda_s is not None
              else (_cal(self.priors.pc_beta, sd2) if self.idx_beta is not None else None))
        lg = (self.priors.pc_gamma.lambda_s if self.priors.pc_gamma.lambda_s is not None
              else (_cal(self.priors.pc_gamma, sd1) if self.seasonal_mode == "dynamic" else None))
        return (None if la is None else float(la),
                None if lb is None else float(lb),
                None if lg is None else float(lg))

    def _init_pc_sigma(self, y: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        # heuristic: use |Δ log |y - median|| as rough scale proxy for η
        resid = np.abs(y - np.median(y)) + 1e-6
        logr = np.log(resid)
        sd1 = _robust_sd(np.diff(logr)) if logr.size >= 2 else 0.0
        sd2 = _robust_sd(np.diff(logr, n=2)) if logr.size >= 3 else 0.0
        def _cal(pc: PCPrior, sd: float) -> float:
            u = max(1e-12, pc.frac * max(1e-12, sd))
            return float(-math.log(max(1e-12, pc.alpha_prob)) / u)
        la = (self.priors.pc_alpha_sig.lambda_s if self.priors.pc_alpha_sig.lambda_s is not None
              else (_cal(self.priors.pc_alpha_sig, sd1) if self.idx_alpha_sig is not None else None))
        lb = (self.priors.pc_beta_sig.lambda_s if self.priors.pc_beta_sig.lambda_s is not None
              else (_cal(self.priors.pc_beta_sig, sd2) if self.idx_beta_sig is not None else None))
        lg = (self.priors.pc_gamma_sig.lambda_s if self.priors.pc_gamma_sig.lambda_s is not None
              else (_cal(self.priors.pc_gamma_sig, sd1) if self.seasonal_mode_sig == "dynamic" else None))
        return (None if la is None else float(la),
                None if lb is None else float(lb),
                None if lg is None else float(lg))

    # ----------------------------- μ model matrices ----------------------------- #
    def _H_mu(self) -> np.ndarray:
        if self.dim_mu == 0: return np.zeros((1,0))
        h = np.zeros(self.dim_mu, float)
        if self.idx_alpha is not None: h[self.idx_alpha] = 1.0
        if self.seasonal_mode == "dynamic": h[self.idx_g_start] = 1.0
        return h.reshape(1, -1)

    def _A_mu(self) -> np.ndarray:
        if self.dim_mu == 0: return np.zeros((0,0))
        A = np.eye(self.dim_mu)
        if self.idx_alpha is not None and self.idx_beta is not None:
            A[self.idx_alpha, self.idx_beta] = 1.0
        if self.seasonal_mode == "dynamic":
            gs, ge = self.idx_g_start, self.idx_g_end
            K = ge - gs + 1
            A[gs, gs:ge+1] = -1.0
            A[gs+1:ge+1, gs:ge] = np.eye(K-1)
            A[gs+1:ge+1, ge] = 0.0
        return A

    def _u_mu(self) -> np.ndarray:
        if self.dim_mu == 0: return np.zeros(0)
        u = np.zeros(self.dim_mu, float)
        if (self.idx_alpha is not None) and (self.trend_mode == "deterministic"):
            u[self.idx_alpha] = float(self.m0_beta)
        return u

    def _Q_mu(self) -> np.ndarray:
        if self.dim_mu == 0: return np.zeros((0,0))
        Q = np.zeros((self.dim_mu, self.dim_mu))
        if self.idx_alpha is not None and self.s_alpha > 0: Q[self.idx_alpha, self.idx_alpha] = self.s_alpha**2
        if self.idx_beta  is not None and self.s_beta  > 0: Q[self.idx_beta,  self.idx_beta ] = self.s_beta**2
        if self.seasonal_mode == "dynamic" and self.s_gamma > 0:
            Q[self.idx_g_start, self.idx_g_start] = self.s_gamma**2
        return Q

    # ----------------------------- η model matrices ----------------------------- #
    def _A_sig(self) -> np.ndarray:
        if self.dim_sig == 0: return np.zeros((0,0))
        A = np.eye(self.dim_sig)
        if self.idx_alpha_sig is not None and self.idx_beta_sig is not None:
            A[self.idx_alpha_sig, self.idx_beta_sig] = 1.0
        if self.seasonal_mode_sig == "dynamic":
            gs, ge = self.idx_gs_start, self.idx_gs_end
            K = ge - gs + 1
            A[gs, gs:ge+1] = -1.0
            A[gs+1:ge+1, gs:ge] = np.eye(K-1)
            A[gs+1:ge+1, ge] = 0.0
        return A

    def _u_sig(self) -> np.ndarray:
        if self.dim_sig == 0: return np.zeros(0)
        u = np.zeros(self.dim_sig, float)
        if (self.idx_alpha_sig is not None) and (self.trend_mode_sig == "deterministic"):
            u[self.idx_alpha_sig] = float(self.m0_beta_sig)
        return u

    def _Q_sig(self) -> np.ndarray:
        if self.dim_sig == 0: return np.zeros((0,0))
        Q = np.zeros((self.dim_sig, self.dim_sig))
        if self.idx_alpha_sig is not None and self.s_alpha_sig > 0: Q[self.idx_alpha_sig, self.idx_alpha_sig] = self.s_alpha_sig**2
        if self.idx_beta_sig  is not None and self.s_beta_sig  > 0: Q[self.idx_beta_sig,  self.idx_beta_sig ] = self.s_beta_sig**2
        if self.seasonal_mode_sig == "dynamic" and self.s_gamma_sig > 0:
            Q[self.idx_gs_start, self.idx_gs_start] = self.s_gamma_sig**2
        return Q

    # -------------------- Slice for log s -------------------- #
    def _slice(self, f: Callable[[float], float], z0: float) -> float:
        w, m, limit = float(self.cfg.slice_w), int(self.cfg.slice_m), int(self.cfg.slice_max_shrink)
        y_star = f(z0) - np.random.exponential(1.0)
        u = np.random.rand()
        L = z0 - u * w; R = L + w
        j = int(np.floor(m * np.random.rand())); k = (m - 1) - j
        while j > 0 and f(L) > y_star: L -= w; j -= 1
        while k > 0 and f(R) > y_star: R += w; k -= 1
        for _ in range(limit):
            z_prop = np.random.uniform(L, R)
            if f(z_prop) >= y_star: return z_prop
            if z_prop < z0: L = z_prop
            else:           R = z_prop
        return z0

    def _slice_logsd(self, z0: float, SS: float, T_eff: int, lam: float) -> float:
        def f(z: float) -> float:
            return -(T_eff * z) - 0.5 * SS * math.exp(-2 * z) - lam * math.exp(z) + z
        return self._slice(f, z0)

    # ------------------ Innovation SS helpers (μ) ------------------ #
    def _innovation_ss_alpha(self, x: np.ndarray) -> Tuple[float,int]:
        if self.idx_alpha is None: return 0.0,0
        ss = 0.0
        for t in range(1, self.T + 1):
            drift = 0.0
            if self.idx_beta is not None: drift = x[t-1, self.idx_beta]
            elif self.trend_mode == "deterministic": drift = float(self.m0_beta)
            mean = x[t-1, self.idx_alpha] + drift
            ss += (x[t, self.idx_alpha] - mean)**2
        return float(ss), self.T

    def _innovation_ss_beta(self, x: np.ndarray) -> Tuple[float,int]:
        if self.idx_beta is None: return 0.0,0
        d = x[1:, self.idx_beta] - x[:-1, self.idx_beta]
        return float(np.sum(d*d)), self.T

    def _innovation_ss_gamma(self, x: np.ndarray) -> Tuple[float,int]:
        if self.seasonal_mode != "dynamic": return 0.0,0
        gs, ge = self.idx_g_start, self.idx_g_end
        ss = 0.0
        for t in range(1, self.T + 1):
            prev = x[t-1, gs:ge+1]
            mean_new_first = -float(np.sum(prev))
            ss += (x[t, gs] - mean_new_first)**2
        return float(ss), self.T

    # ------------------ Innovation SS helpers (η) ------------------ #
    def _innovation_ss_alpha_sig(self, eta: np.ndarray) -> Tuple[float,int]:
        if self.idx_alpha_sig is None: return 0.0,0
        ss = 0.0
        for t in range(1, self.T + 1):
            drift = 0.0
            if self.idx_beta_sig is not None: drift = eta[t-1, self.idx_beta_sig]
            elif self.trend_mode_sig == "deterministic": drift = float(self.m0_beta_sig)
            mean = eta[t-1, self.idx_alpha_sig] + drift
            ss += (eta[t, self.idx_alpha_sig] - mean)**2
        return float(ss), self.T

    def _innovation_ss_beta_sig(self, eta: np.ndarray) -> Tuple[float,int]:
        if self.idx_beta_sig is None: return 0.0,0
        d = eta[1:, self.idx_beta_sig] - eta[:-1, self.idx_beta_sig]
        return float(np.sum(d*d)), self.T

    def _innovation_ss_gamma_sig(self, eta: np.ndarray) -> Tuple[float,int]:
        if self.seasonal_mode_sig != "dynamic": return 0.0,0
        gs, ge = self.idx_gs_start, self.idx_gs_end
        ss = 0.0
        for t in range(1, self.T + 1):
            prev = eta[t-1, gs:ge+1]
            mean_new_first = -float(np.sum(prev))
            ss += (eta[t, gs] - mean_new_first)**2
        return float(ss), self.T

    # ------------------ Deterministic μ contribution ------------------ #
    def _mu_det(self, t: int) -> float:
        out = 0.0
        if self.level_mode == "deterministic": out += self.m0_alpha
        if (self.trend_mode == "deterministic") and (self.idx_alpha is None):
            out += self.m0_beta * t
        if self.seasonal_mode == "deterministic":
            out += float(self.m0_gamma[t % self.period])
        return out

    # ------------------ RBPF core ------------------ #
    def _resample_idx(self, w: np.ndarray) -> np.ndarray:
        N = len(w)
        if self.cfg.resample_method == "multinomial":
            return np.random.choice(N, size=N, replace=True, p=w)
        # systematic
        u0 = np.random.rand() / N
        cdf = np.cumsum(w)
        idx = np.zeros(N, dtype=int)
        j = 0
        for n in range(N):
            u = u0 + n / N
            while u > cdf[j]:
                j += 1
            idx[n] = j
        return idx

    def _kalman_forward_for_particle(self, sig_t: np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        """
        Run KF for μ given σ_t = exp(η_t) (provided as sd_t).
        Returns filtered means m_t and covs C_t (t=0..T), and predictive μ means a_t, R_t.
        """
        if self.dim_mu == 0:
            m = np.zeros((self.T+1, 0)); C = np.zeros((self.T+1,0,0))
            a = np.zeros_like(m); R = np.zeros_like(C)
            return m, C, a, R

        H, A, Q = self._H_mu(), self._A_mu(), self._Q_mu()
        m = np.zeros((self.T+1, self.dim_mu))
        C = np.zeros((self.T+1, self.dim_mu, self.dim_mu))
        a = np.zeros_like(m); R = np.zeros_like(C)

        # prior
        m0_vec, P0_diag = [], []
        if self.idx_alpha is not None: m0_vec.append(self.m0_alpha); P0_diag.append(self.P0_alpha)
        if self.idx_beta  is not None: m0_vec.append(self.m0_beta);  P0_diag.append(self.P0_beta)
        if self.seasonal_mode == "dynamic":
            m0_vec.extend(list(self.m0_gamma)); P0_diag.extend([self.P0_gamma]*(self.period-1))
        m[0] = np.asarray(m0_vec, float)
        C[0] = np.diag(np.asarray(P0_diag)) + 1e-12*np.eye(self.dim_mu)

        u = self._u_mu()
        for t in range(1, self.T+1):
            a[t]  = A @ m[t-1] + u
            R[t]  = A @ C[t-1] @ A.T + Q
            R[t]  = 0.5*(R[t]+R[t].T) + 1e-12*np.eye(self.dim_mu)

            s2 = float(sig_t[t-1]**2)
            S = float(H @ R[t] @ H.T + s2)
            if S <= 0: S = float(H @ (R[t] + 1e-10*np.eye(self.dim_mu)) @ H.T + s2)
            K = (R[t] @ H.T) / S
            ytil = float(self.y[t-1] - self._mu_det(t-1)) - float(H @ a[t])
            m[t] = a[t] + (K.flatten()*ytil)
            C[t] = R[t] - K @ (H @ R[t])
            C[t] = 0.5*(C[t]+C[t].T) + 1e-12*np.eye(self.dim_mu)
        return m, C, a, R

    def _loglik_predictive(self, mu_pred: float, var_pred: float, y: float, s2: float) -> float:
        # y ~ N(mu_pred + μ_det, var_pred_obs) with var_pred_obs = H R H' + s2; but caller already supplies mu_pred, var_pred (scalar)
        v = float(y - mu_pred)
        S = float(var_pred + s2)
        return -0.5*(math.log(2*math.pi*S) + v*v / S)

    def _run_rbpf_once(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        One RBPF pass:
          - particles over η path
          - per-particle KF over μ
        Returns: sampled μ path (T+1,dim_mu), sampled η path (T+1,dim_sig),
                 and sd_t = exp(η_t) for t=1..T used in obs; plus μ_t mean path (H*x)
        """
        N = int(self.cfg.n_particles)
        A_sig, u_sig, Q_sig = self._A_sig(), self._u_sig(), self._Q_sig()
        H_mu, A_mu, Q_mu = self._H_mu(), self._A_mu(), self._Q_mu()

        # init particles for η_0
        if self.dim_sig > 0:
            m0_sig, P0_sig = [], []
            if self.idx_alpha_sig is not None: m0_sig.append(self.m0_alpha_sig); P0_sig.append(self.P0_alpha_sig)
            if self.idx_beta_sig  is not None: m0_sig.append(self.m0_beta_sig);  P0_sig.append(self.P0_beta_sig)
            if self.seasonal_mode_sig == "dynamic":
                m0_sig.extend(list(self.m0_gamma_sig)); P0_sig.extend([self.P0_gamma_sig]*(self.period-1))
            m0_sig = np.asarray(m0_sig,float); P0_sig = np.asarray(P0_sig,float)
            eta0_particles = np.random.multivariate_normal(m0_sig, np.diag(P0_sig)+1e-12*np.eye(self.dim_sig), size=N)
        else:
            eta0_particles = np.zeros((N,0))

        # per-particle KF state for μ
        if self.dim_mu > 0:
            m0_mu, P0_mu = [], []
            if self.idx_alpha is not None: m0_mu.append(self.m0_alpha); P0_mu.append(self.P0_alpha)
            if self.idx_beta  is not None: m0_mu.append(self.m0_beta);  P0_mu.append(self.P0_beta)
            if self.seasonal_mode == "dynamic":
                m0_mu.extend(list(self.m0_gamma)); P0_mu.extend([self.P0_gamma]*(self.period-1))
            m_particles = np.tile(np.asarray(m0_mu,float), (N,1))
            C_particles = np.tile(np.diag(np.asarray(P0_mu,float))+1e-12*np.eye(self.dim_mu), (N,1,1))
        else:
            m_particles = np.zeros((N,0)); C_particles = np.zeros((N,0,0))

        # storage
        eta_particles = np.zeros((self.T+1, N, self.dim_sig))
        eta_particles[0] = eta0_particles
        ancestors = np.zeros((self.T+1, N), dtype=int)

        logw = np.zeros(N)
        u_mu = self._u_mu()

        # forward PF with embedded KFs
        for t in range(1, self.T+1):
            # propagate η
            if self.dim_sig > 0:
                noise = np.random.multivariate_normal(np.zeros(self.dim_sig), Q_sig + 1e-12*np.eye(self.dim_sig), size=N)
                eta_t = eta_particles[t-1] @ A_sig.T + u_sig + noise
                eta_particles[t] = eta_t
                sd_t = np.exp(eta_t[:, self.idx_alpha_sig] if self.idx_alpha_sig is not None else eta_t[:,0])  # use first coord as level
            else:
                eta_particles[t] = eta_particles[t-1]
                sd_t = np.ones(N)

            # Update μ KFs and compute incremental weights
            var_obs = np.zeros(N)
            mu_pred = np.zeros(N)
            for i in range(N):
                if self.dim_mu == 0:
                    # only deterministic μ; obs mean = μ_det
                    mu_star = self._mu_det(t-1); var_star = 0.0
                else:
                    # predict
                    a = A_mu @ m_particles[i] + u_mu
                    R = A_mu @ C_particles[i] @ A_mu.T + Q_mu
                    R = 0.5*(R+R.T) + 1e-12*np.eye(self.dim_mu)
                    S = float(H_mu @ R @ H_mu.T + sd_t[i]**2)
                    v = float(self.y[t-1] - self._mu_det(t-1)) - float(H_mu @ a)
                    K = (R @ H_mu.T) / S
                    m_particles[i] = a + (K.flatten()*v)
                    C_particles[i] = R - K @ (H_mu @ R)
                    C_particles[i] = 0.5*(C_particles[i]+C_particles[i].T) + 1e-12*np.eye(self.dim_mu)
                    mu_star = float(H_mu @ a) + self._mu_det(t-1)
                    var_star = float(H_mu @ R @ H_mu.T)
                mu_pred[i] = mu_star; var_obs[i] = var_star + sd_t[i]**2
                logw[i] += -0.5*(math.log(2*math.pi*var_obs[i]) + (self.y[t-1]-mu_star)**2/var_obs[i])

            # normalize weights & resample if ESS low
            w = np.exp(logw - np.max(logw))
            w = w / np.sum(w)
            ESS = 1.0 / np.sum(w*w)
            if ESS < self.cfg.ess_resample * N:
                idx = self._resample_idx(w)
                # resample particles and KFs
                eta_particles[t] = eta_particles[t][idx]
                m_particles = m_particles[idx]
                C_particles = C_particles[idx]
                ancestors[t] = idx
                logw = np.zeros(N)
            else:
                ancestors[t] = np.arange(N)

        # pick final particle
        if self.cfg.ess_resample < 1.1:  # if we ever resampled, weights are reset
            w_final = np.ones(N)/N
        else:
            w_final = np.exp(logw - np.max(logw)); w_final /= np.sum(w_final)
        kT = np.random.choice(N, p=w_final)

        # backtrack index path
        idx_path = np.zeros(self.T+1, dtype=int); idx_path[self.T] = kT
        for t in range(self.T-1, -1, -1):
            idx_path[t] = ancestors[t+1][idx_path[t+1]]

        # build sampled η path and implied sd
        eta_path = np.zeros((self.T+1, self.dim_sig))
        for t in range(self.T+1):
            eta_path[t] = eta_particles[t, idx_path[t]]
        sd_t = np.exp(eta_path[1:, self.idx_alpha_sig] if self.idx_alpha_sig is not None else eta_path[1:,0])

        # run full FFBS for μ given sd_t (Rao-Blackwellization ⇒ smoother)
        x_mu = self._ffbs_mu_given_sigma(sd_t)
        mu_t = self._mu_series_from_state(x_mu)
        return x_mu, eta_path, sd_t, mu_t

    def _ffbs_mu_given_sigma(self, sd_t: np.ndarray) -> np.ndarray:
        if self.dim_mu == 0:
            return np.zeros((self.T+1, 0))
        H, A, Q = self._H_mu(), self._A_mu(), self._Q_mu()
        m = np.zeros((self.T + 1, self.dim_mu))
        C = np.zeros((self.T + 1, self.dim_mu, self.dim_mu))
        a = np.zeros((self.T + 1, self.dim_mu))
        Rm = np.zeros((self.T + 1, self.dim_mu, self.dim_mu))

        # prior
        m0_vec, P0_diag = [], []
        if self.idx_alpha is not None: m0_vec.append(self.m0_alpha); P0_diag.append(self.P0_alpha)
        if self.idx_beta  is not None: m0_vec.append(self.m0_beta);  P0_diag.append(self.P0_beta)
        if self.seasonal_mode == "dynamic":
            m0_vec.extend(list(self.m0_gamma)); P0_diag.extend([self.P0_gamma]*(self.period-1))
        m[0] = np.asarray(m0_vec, float)
        C[0] = np.diag(np.asarray(P0_diag)) + 1e-12*np.eye(self.dim_mu)
        u = self._u_mu()

        # forward KF with known sd_t
        for t in range(1, self.T + 1):
            a[t]  = A @ m[t - 1] + u
            Rm[t] = A @ C[t - 1] @ A.T + Q
            Rm[t] = 0.5*(Rm[t]+Rm[t].T) + 1e-12*np.eye(self.dim_mu)
            resid = float(self.y[t - 1] - self._mu_det(t - 1))
            S = float(H @ Rm[t] @ H.T + sd_t[t-1]**2)
            if S <= 0: S = float(H @ (Rm[t] + 1e-10*np.eye(self.dim_mu)) @ H.T + sd_t[t-1]**2)
            K = (Rm[t] @ H.T) / S
            v = resid - float(H @ a[t])
            m[t] = a[t] + (K.flatten()*v)
            C[t] = Rm[t] - K @ (H @ Rm[t])
            C[t] = 0.5*(C[t]+C[t].T) + 1e-12*np.eye(self.dim_mu)

        # backward sample
        x = np.zeros((self.T+1, self.dim_mu))
        x[self.T] = np.random.multivariate_normal(m[self.T], C[self.T])
        for t in range(self.T - 1, -1, -1):
            J = C[t] @ A.T
            J = J @ _spd_solve(Rm[t + 1], np.eye(self.dim_mu))
            mean = m[t] + J @ (x[t + 1] - (A @ m[t] + u))
            cov  = C[t] - J @ Rm[t + 1] @ J.T
            cov  = 0.5*(cov+cov.T)
            cov += max(0.0, 1e-12 - float(np.linalg.eigvalsh(cov).min()))*np.eye(cov.shape[0])
            x[t] = np.random.multivariate_normal(mean, cov)
        return x

    def _mu_series_from_state(self, x_mu: np.ndarray) -> np.ndarray:
        H = self._H_mu()
        mu = np.zeros(self.T, float)
        for t in range(1, self.T+1):
            dyn = float(H @ x_mu[t]) if self.dim_mu > 0 else 0.0
            mu[t-1] = self._mu_det(t-1) + dyn
        return mu

    # -------------------- Parameter updates -------------------- #
    def _gibbs_lambda_single(self, which: str) -> None:
        if which == "alpha" and (self.priors.pc_alpha.lambda_s is None) and (self.idx_alpha is not None):
            pc = self.priors.pc_alpha
            self.lambda_alpha = float(np.random.gamma(pc.a_lambda+1.0, 1.0/(pc.b_lambda + max(0.0,self.s_alpha))))
        if which == "beta" and (self.priors.pc_beta.lambda_s is None) and (self.idx_beta is not None):
            pc = self.priors.pc_beta
            self.lambda_beta = float(np.random.gamma(pc.a_lambda+1.0, 1.0/(pc.b_lambda + max(0.0,self.s_beta))))
        if which == "gamma" and (self.priors.pc_gamma.lambda_s is None) and (self.seasonal_mode == "dynamic"):
            pc = self.priors.pc_gamma
            self.lambda_gamma = float(np.random.gamma(pc.a_lambda+1.0, 1.0/(pc.b_lambda + max(0.0,self.s_gamma))))

    def _gibbs_lambda_single_sig(self, which: str) -> None:
        if which == "alpha" and (self.priors.pc_alpha_sig.lambda_s is None) and (self.idx_alpha_sig is not None):
            pc = self.priors.pc_alpha_sig
            self.lambda_alpha_sig = float(np.random.gamma(pc.a_lambda+1.0, 1.0/(pc.b_lambda + max(0.0,self.s_alpha_sig))))
        if which == "beta" and (self.priors.pc_beta_sig.lambda_s is None) and (self.idx_beta_sig is not None):
            pc = self.priors.pc_beta_sig
            self.lambda_beta_sig = float(np.random.gamma(pc.a_lambda+1.0, 1.0/(pc.b_lambda + max(0.0,self.s_beta_sig))))
        if which == "gamma" and (self.priors.pc_gamma_sig.lambda_s is None) and (self.seasonal_mode_sig == "dynamic"):
            pc = self.priors.pc_gamma_sig
            self.lambda_gamma_sig = float(np.random.gamma(pc.a_lambda+1.0, 1.0/(pc.b_lambda + max(0.0,self.s_gamma_sig))))

    def _update_process_sds_mu(self, x_mu: np.ndarray) -> None:
        if self.idx_alpha is not None and (self.lambda_alpha is not None):
            ss, T_eff = self._innovation_ss_alpha(x_mu)
            z = self._slice_logsd(math.log(max(1e-18, self.s_alpha)), ss, T_eff, float(self.lambda_alpha))
            self.s_alpha = float(math.exp(z))
        if self.idx_beta is not None and (self.lambda_beta is not None):
            ss, T_eff = self._innovation_ss_beta(x_mu)
            z = self._slice_logsd(math.log(max(1e-18, self.s_beta)), ss, T_eff, float(self.lambda_beta))
            self.s_beta = float(math.exp(z))
        if self.seasonal_mode == "dynamic" and (self.lambda_gamma is not None):
            ss, T_eff = self._innovation_ss_gamma(x_mu)
            z = self._slice_logsd(math.log(max(1e-18, self.s_gamma)), ss, T_eff, float(self.lambda_gamma))
            self.s_gamma = float(math.exp(z))
        # update lambdas
        self._gibbs_lambda_single("alpha")
        self._gibbs_lambda_single("beta")
        self._gibbs_lambda_single("gamma")

    def _update_process_sds_sig(self, eta_path: np.ndarray) -> None:
        if self.idx_alpha_sig is not None and (self.lambda_alpha_sig is not None):
            ss, T_eff = self._innovation_ss_alpha_sig(eta_path)
            z = self._slice_logsd(math.log(max(1e-18, self.s_alpha_sig)), ss, T_eff, float(self.lambda_alpha_sig))
            self.s_alpha_sig = float(math.exp(z))
        if self.idx_beta_sig is not None and (self.lambda_beta_sig is not None):
            ss, T_eff = self._innovation_ss_beta_sig(eta_path)
            z = self._slice_logsd(math.log(max(1e-18, self.s_beta_sig)), ss, T_eff, float(self.lambda_beta_sig))
            self.s_beta_sig = float(math.exp(z))
        if self.seasonal_mode_sig == "dynamic" and (self.lambda_gamma_sig is not None):
            ss, T_eff = self._innovation_ss_gamma_sig(eta_path)
            z = self._slice_logsd(math.log(max(1e-18, self.s_gamma_sig)), ss, T_eff, float(self.lambda_gamma_sig))
            self.s_gamma_sig = float(math.exp(z))
        # update lambdas
        self._gibbs_lambda_single_sig("alpha")
        self._gibbs_lambda_single_sig("beta")
        self._gibbs_lambda_single_sig("gamma")

    # ------------------------------- MCMC driver ------------------------------- #
    def run(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        save_iters = list(range(cfg.burn, cfg.n_iter, cfg.thin))
        n_kept, keep_idx = len(save_iters), 0

        # allocate
        self.keep = {
            "mu": np.zeros((n_kept, self.T), float),
            "sigma": np.zeros((n_kept, self.T), float),
        }
        if self.dim_mu > 0:
            self.keep.update({
                "x_mu": np.zeros((n_kept, self.T, self.dim_mu)),
                "Q_alpha": np.zeros(n_kept) if self.idx_alpha is not None else np.zeros(0),
                "Q_beta":  np.zeros(n_kept) if self.idx_beta  is not None else np.zeros(0),
                "Q_gamma": np.zeros(n_kept) if self.seasonal_mode == "dynamic" else np.zeros(0),
                "lambda_alpha": np.zeros(n_kept) if self.idx_alpha is not None else np.zeros(0),
                "lambda_beta":  np.zeros(n_kept) if self.idx_beta  is not None else np.zeros(0),
                "lambda_gamma": np.zeros(n_kept) if self.seasonal_mode == "dynamic" else np.zeros(0),
                "m0_alpha": np.zeros(n_kept) if self.idx_alpha is not None else np.zeros(0),
                "P0_alpha": np.zeros(n_kept) if self.idx_alpha is not None else np.zeros(0),
                "m0_beta":  np.zeros(n_kept) if self.idx_beta  is not None else np.zeros(0),
                "P0_beta":  np.zeros(n_kept) if self.idx_beta  is not None else np.zeros(0),
                "m0_gamma": np.zeros((n_kept, self.period-1)) if self.seasonal_mode == "dynamic" else np.zeros((0,0)),
                "P0_gamma": np.zeros(n_kept) if self.seasonal_mode == "dynamic" else np.zeros(0),
            })
        # sigma block traces
        if self.dim_sig > 0:
            self.keep.update({
                "eta": np.zeros((n_kept, self.T+1, self.dim_sig)),
                "Q_alpha_sig": np.zeros(n_kept) if self.idx_alpha_sig is not None else np.zeros(0),
                "Q_beta_sig":  np.zeros(n_kept) if self.idx_beta_sig  is not None else np.zeros(0),
                "Q_gamma_sig": np.zeros(n_kept) if self.seasonal_mode_sig == "dynamic" else np.zeros(0),
                "lambda_alpha_sig": np.zeros(n_kept) if self.idx_alpha_sig is not None else np.zeros(0),
                "lambda_beta_sig":  np.zeros(n_kept) if self.idx_beta_sig  is not None else np.zeros(0),
                "lambda_gamma_sig": np.zeros(n_kept) if self.seasonal_mode_sig == "dynamic" else np.zeros(0),
                "m0_alpha_sig": np.zeros(n_kept) if self.idx_alpha_sig is not None else np.zeros(0),
                "P0_alpha_sig": np.zeros(n_kept) if self.idx_alpha_sig is not None else np.zeros(0),
                "m0_beta_sig":  np.zeros(n_kept) if self.idx_beta_sig  is not None else np.zeros(0),
                "P0_beta_sig":  np.zeros(n_kept) if self.idx_beta_sig  is not None else np.zeros(0),
                "m0_gamma_sig": np.zeros((n_kept, self.period-1)) if self.seasonal_mode_sig == "dynamic" else np.zeros((0,0)),
                "P0_gamma_sig": np.zeros(n_kept) if self.seasonal_mode_sig == "dynamic" else np.zeros(0),
            })

        print_every = cfg.progress_every if cfg.progress_every > 0 else max(1, cfg.n_iter // 50) or 1

        # Current states (initialize with priors)
        x_mu_cur = np.zeros((self.T+1, self.dim_mu))
        eta_cur  = np.zeros((self.T+1, self.dim_sig))

        for it in range(cfg.n_iter):
            # 1) RBPF pass ⇒ sampled paths x_mu_cur, eta_cur, sd_t, and μ_t
            x_mu_cur, eta_cur, sd_t, mu_t = self._run_rbpf_once()

            # 2) update μ process sds (PC prior via slice/Gibbs)
            self._update_process_sds_mu(x_mu_cur)

            # 3) update η process sds (PC prior via slice/Gibbs)
            if self.dim_sig > 0:
                self._update_process_sds_sig(eta_cur)

            # 4) update m0/P0 for μ (conjugate given x0)
            #    (same scalars as in your previous code)
            def _gibbs_m0_scalar(x0, m_prior, s_prior, P0):
                prec = 1.0/(s_prior**2) + 1.0/max(1e-18,P0)
                var = 1.0/prec
                mean = var*(m_prior/(s_prior**2) + x0/max(1e-18,P0))
                return float(np.random.normal(mean, math.sqrt(var)))

            pos = 0
            if self.idx_alpha is not None:
                self.m0_alpha = _gibbs_m0_scalar(x_mu_cur[0, pos], self.priors.m_m0_alpha, self.priors.s_m0_alpha, self.P0_alpha)
                a = self.priors.a_P0_alpha + 0.5
                b = self.priors.b_P0_alpha + 0.5*(x_mu_cur[0, pos]-self.m0_alpha)**2
                self.P0_alpha = 1.0/np.random.gamma(a, 1.0/b); pos += 1
            if self.idx_beta is not None:
                self.m0_beta  = _gibbs_m0_scalar(x_mu_cur[0, pos], self.priors.m_m0_beta,  self.priors.s_m0_beta,  self.P0_beta)
                a = self.priors.a_P0_beta + 0.5
                b = self.priors.b_P0_beta + 0.5*(x_mu_cur[0, pos]-self.m0_beta)**2
                self.P0_beta  = 1.0/np.random.gamma(a, 1.0/b); pos += 1
            if self.seasonal_mode == "dynamic":
                m_prior = (np.zeros(self.period-1) if self.priors.m_m0_gamma is None
                           else np.asarray(self.priors.m_m0_gamma,float))
                s = float(self.priors.s_m0_gamma)
                diffsq = 0.0
                for k in range(self.period-1):
                    self.m0_gamma[k] = _gibbs_m0_scalar(x_mu_cur[0, pos+k], float(m_prior[k]), s, self.P0_gamma)
                    diffsq += (x_mu_cur[0, pos+k]-self.m0_gamma[k])**2
                a = self.priors.a_P0_gamma + 0.5*(self.period-1)
                b = self.priors.b_P0_gamma + 0.5*diffsq
                self.P0_gamma = 1.0/np.random.gamma(a, 1.0/b)

            # 5) update m0/P0 for η (conjugate given η0)
            pos = 0
            if self.idx_alpha_sig is not None:
                self.m0_alpha_sig = _gibbs_m0_scalar(eta_cur[0, pos], self.priors.m_m0_alpha_sig, self.priors.s_m0_alpha_sig, self.P0_alpha_sig)
                a = self.priors.a_P0_alpha_sig + 0.5
                b = self.priors.b_P0_alpha_sig + 0.5*(eta_cur[0, pos]-self.m0_alpha_sig)**2
                self.P0_alpha_sig = 1.0/np.random.gamma(a, 1.0/b); pos += 1
            if self.idx_beta_sig is not None:
                self.m0_beta_sig  = _gibbs_m0_scalar(eta_cur[0, pos], self.priors.m_m0_beta_sig,  self.priors.s_m0_beta_sig,  self.P0_beta_sig)
                a = self.priors.a_P0_beta_sig + 0.5
                b = self.priors.b_P0_beta_sig + 0.5*(eta_cur[0, pos]-self.m0_beta_sig)**2
                self.P0_beta_sig  = 1.0/np.random.gamma(a, 1.0/b); pos += 1
            if self.seasonal_mode_sig == "dynamic":
                m_prior = (np.zeros(self.period-1) if self.priors.m_m0_gamma_sig is None
                           else np.asarray(self.priors.m_m0_gamma_sig,float))
                s = float(self.priors.s_m0_gamma_sig)
                diffsq = 0.0
                for k in range(self.period-1):
                    self.m0_gamma_sig[k] = _gibbs_m0_scalar(eta_cur[0, pos+k], float(m_prior[k]), s, self.P0_gamma_sig)
                    diffsq += (eta_cur[0, pos+k]-self.m0_gamma_sig[k])**2
                a = self.priors.a_P0_gamma_sig + 0.5*(self.period-1)
                b = self.priors.b_P0_gamma_sig + 0.5*diffsq
                self.P0_gamma_sig = 1.0/np.random.gamma(a, 1.0/b)

            # progress
            if cfg.progress and ((it + 1) % (cfg.progress_every or max(1, cfg.n_iter//50)) == 0 or it == cfg.n_iter - 1):
                parts = [f"[it {it+1}/{cfg.n_iter}]",
                         f"Qμ: ({self.s_alpha**2 if self.idx_alpha is not None else 0:.3g},"
                         f"{self.s_beta**2  if self.idx_beta  is not None else 0:.3g},"
                         f"{self.s_gamma**2 if self.seasonal_mode=='dynamic' else 0:.3g})",
                         f"Qη: ({self.s_alpha_sig**2 if self.idx_alpha_sig is not None else 0:.3g},"
                         f"{self.s_beta_sig**2  if self.idx_beta_sig  is not None else 0:.3g},"
                         f"{self.s_gamma_sig**2 if self.seasonal_mode_sig=='dynamic' else 0:.3g})"]
                print(" | ".join(parts))

            # save
            if it in save_iters:
                self.keep["mu"][keep_idx, :] = mu_t
                self.keep["sigma"][keep_idx, :] = np.exp(eta_cur[1:, self.idx_alpha_sig] if self.dim_sig>0 else 0.0)
                if self.dim_mu > 0:
                    self.keep["x_mu"][keep_idx, :, :] = x_mu_cur[1:self.T+1, :]
                    if self.idx_alpha is not None:
                        self.keep["Q_alpha"][keep_idx] = self.s_alpha**2
                        self.keep["lambda_alpha"][keep_idx] = float(self.lambda_alpha or 0.0)
                        self.keep["m0_alpha"][keep_idx] = self.m0_alpha
                        self.keep["P0_alpha"][keep_idx] = self.P0_alpha
                    if self.idx_beta is not None:
                        self.keep["Q_beta"][keep_idx] = self.s_beta**2
                        self.keep["lambda_beta"][keep_idx] = float(self.lambda_beta or 0.0)
                        self.keep["m0_beta"][keep_idx] = self.m0_beta
                        self.keep["P0_beta"][keep_idx] = self.P0_beta
                    if self.seasonal_mode == "dynamic":
                        self.keep["Q_gamma"][keep_idx] = self.s_gamma**2
                        self.keep["lambda_gamma"][keep_idx] = float(self.lambda_gamma or 0.0)
                        self.keep["m0_gamma"][keep_idx, :] = self.m0_gamma
                        self.keep["P0_gamma"][keep_idx] = self.P0_gamma
                if self.dim_sig > 0:
                    self.keep["eta"][keep_idx, :, :] = eta_cur
                    if self.idx_alpha_sig is not None:
                        self.keep["Q_alpha_sig"][keep_idx] = self.s_alpha_sig**2
                        self.keep["lambda_alpha_sig"][keep_idx] = float(self.lambda_alpha_sig or 0.0)
                        self.keep["m0_alpha_sig"][keep_idx] = self.m0_alpha_sig
                        self.keep["P0_alpha_sig"][keep_idx] = self.P0_alpha_sig
                    if self.idx_beta_sig is not None:
                        self.keep["Q_beta_sig"][keep_idx] = self.s_beta_sig**2
                        self.keep["lambda_beta_sig"][keep_idx] = float(self.lambda_beta_sig or 0.0)
                        self.keep["m0_beta_sig"][keep_idx] = self.m0_beta_sig
                        self.keep["P0_beta_sig"][keep_idx] = self.P0_beta_sig
                    if self.seasonal_mode_sig == "dynamic":
                        self.keep["Q_gamma_sig"][keep_idx] = self.s_gamma_sig**2
                        self.keep["lambda_gamma_sig"][keep_idx] = float(self.lambda_gamma_sig or 0.0)
                        self.keep["m0_gamma_sig"][keep_idx, :] = self.m0_gamma_sig
                        self.keep["P0_gamma_sig"][keep_idx] = self.P0_gamma_sig
                keep_idx += 1

        return self.keep

    # ------------------------------- Persistence ---------------------------- #
    def save_posterior(self, out_npz_path: str, extra_meta: Optional[dict] = None) -> None:
        _ensure_dir(os.path.dirname(out_npz_path))
        arrays = dict(self.keep)
        arrays["y"] = self.y.copy()
        np.savez_compressed(out_npz_path, **arrays)

        meta = {
            "T": int(self.T),
            "period": int(self.period),
            "modes": {
                "level_mode": self.level_mode, "trend_mode": self.trend_mode, "seasonal_mode": self.seasonal_mode,
                "level_mode_sigma": self.level_mode_sig, "trend_mode_sigma": self.trend_mode_sig,
                "seasonal_mode_sigma": self.seasonal_mode_sig,
            },
            "layout_mu": list(self.layout_mu),
            "layout_sig": list(self.layout_sig),
            "cfg": asdict(self.cfg),
            "priors": asdict(self.priors),
        }
        if extra_meta: meta.update(extra_meta)
        with open(out_npz_path.replace(".npz", ".meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

# =============================================================================
# Unified front-end: choose RBPF if σ is dynamic; else use Kalman-Gibbs
# =============================================================================
class DLMUnifiedSampler:
    """
    Wrapper that chooses:
      - Homoskedastic: DLMGibbsConjugate (your original sampler).
      - Heteroskedastic (any dynamic η): RBPFHetero.
    Public API: run() -> keep dict with 'mu', and now also 'sigma'.
    """
    def __init__(self, *, y: np.ndarray, period: int,
                 level_mode: str = "dynamic", trend_mode: str = "dynamic", seasonal_mode: str = "dynamic",
                 # η (log σ) modes; if all deterministic/none -> homoskedastic (σ constant learned via Gamma)
                 level_mode_sigma: str = "none", trend_mode_sigma: str = "none", seasonal_mode_sigma: str = "none",
                 priors: Priors = Priors(), cfg: SamplerConfig = SamplerConfig(),
                 # pass-through of inits (μ and η) with same names as above classes...
                 **kwargs):
        self.y = np.asarray(y, float)
        self.hetero = (level_mode_sigma == "dynamic" or trend_mode_sigma == "dynamic" or seasonal_mode_sigma == "dynamic")
        self.cfg, self.priors = cfg, priors

        if not self.hetero:
            # ---- HOMOSKEDASTIC path: use DLMGibbsConjugate and filter kwargs ----
            from optimization.dlm_location import DLMGibbsConjugate  

            gibbs_kwargs = dict(
                y=y,
                period=int(period),
                level_mode=str(level_mode),
                trend_mode=str(trend_mode),
                seasonal_mode=str(seasonal_mode),

                # initial values the Gibbs engine actually knows about
                sigma2_init=float(sigma_init) ** 2,
                s_alpha_init=float(s_alpha_init),
                s_beta_init=float(s_beta_init),
                s_gamma_init=float(s_gamma_init),

                m0_alpha_init=(0.0 if level_mode == "none" else float(m0_level)),
                P0_alpha_init=float(P0_alpha_init),
                m0_beta_init=(float(m0_trend) if trend_mode != "none" else 0.0),
                P0_beta_init=float(P0_beta_init),
                m0_gamma_init=list(m0_gamma_init) if m0_gamma_init is not None else None,  # newest-first, length p-1
                P0_gamma_init=float(P0_gamma_init),

                priors=priors,
                cfg=cfg,
            )

            # Do NOT include any *_sig_* keys here.
            self.engine = DLMGibbsConjugate(**gibbs_kwargs)

        else:
            # ---- HETEROSKEDASTIC path: use your RBPF engine and pass *_sig_* args there ----
            from dlm_rbpf_engine import DLMRBPF  # whatever your RBPF class is
            rbpf_kwargs = dict(
                y=y,
                period=int(period),
                # mean-block modes
                level_mode=str(level_mode),
                trend_mode=str(trend_mode),
                seasonal_mode=str(seasonal_mode),
                # sigma-block modes
                level_mode_sigma=str(level_mode_sigma),
                trend_mode_sigma=str(trend_mode_sigma),
                seasonal_mode_sigma=str(seasonal_mode_sigma),

                # μ inits
                s_alpha_init=float(s_alpha_init),
                s_beta_init=float(s_beta_init),
                s_gamma_init=float(s_gamma_init),
                m0_alpha_init=(0.0 if level_mode == "none" else float(m0_level)),
                P0_alpha_init=float(P0_alpha_init),
                m0_beta_init=(float(m0_trend) if trend_mode != "none" else 0.0),
                P0_beta_init=float(P0_beta_init),
                m0_gamma_init=list(m0_gamma_init) if m0_gamma_init is not None else None,
                P0_gamma_init=float(P0_gamma_init),

                # η inits (only RBPF cares)
                m0_alpha_sig_init=float(m0_level_sig_init),
                P0_alpha_sig_init=float(P0_alpha_sig_init),
                m0_beta_sig_init=0.0,
                P0_beta_sig_init=float(P0_beta_sig_init),
                m0_gamma_sig_init=list(m0_gamma_sig_init) if m0_gamma_sig_init is not None else None,
                P0_gamma_sig_init=float(P0_gamma_sig_init),
                s_alpha_sig_init=float(s_alpha_sig_init),
                s_beta_sig_init=float(s_beta_sig_init),
                s_gamma_sig_init=float(s_gamma_sig_init),

                priors=priors,
                cfg=cfg,
            )
            self.engine = DLMRBPF(**rbpf_kwargs)


        self.hetero_modes = dict(level_mode_sigma=level_mode_sigma,
                                 trend_mode_sigma=trend_mode_sigma,
                                 seasonal_mode_sigma=seasonal_mode_sigma)

    def run(self) -> Dict[str, np.ndarray]:
        post = self.engine.run()
        # normalize outputs to include 'sigma'
        if "sigma" not in post:
            # homoskedastic case: derive from stored scalar sigma samples if present, else from mu residual variance
            if "sigma" in post:  # unlikely; just in case
                pass
            elif "sigma" not in post and "mu" in post:
                # fallback: use posterior mean sigma scalar from engine.keep['sigma'] vector
                if "sigma" in post and post["sigma"].ndim == 1:
                    sc = np.mean(post["sigma"])
                    post["sigma"] = np.tile(sc, (post["mu"].shape[0], post["mu"].shape[1]))
                elif "sigma" not in post and "mu" in post and "sigma" in self.engine.keep:
                    sc = np.mean(self.engine.keep["sigma"])
                    post["sigma"] = np.tile(sc, (post["mu"].shape[0], post["mu"].shape[1]))
                else:
                    # if the original class stores sqrt sigma2 as 'sigma'
                    if "sigma" in self.engine.keep and self.engine.keep["sigma"].ndim == 1:
                        sc = np.mean(self.engine.keep["sigma"])
                        post["sigma"] = np.tile(sc, (post["mu"].shape[0], post["mu"].shape[1]))
        return post

    def save_posterior(self, out_npz_path: str, extra_meta: Optional[dict] = None) -> None:
        extra = extra_meta or {}
        extra.update({"heteroskedastic": bool(self.hetero), "sigma_modes": self.hetero_modes})
        self.engine.save_posterior(out_npz_path, extra_meta=extra)

# ------------------------- CLI / Example run ------------------------------- #
if __name__ == "__main__":
    import argparse, os, time, math, sys
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from datetime import datetime

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(base_dir)

    # simulator that supports parallel mean/log-sigma blocks (newest-first convention)
    from simulator.mean_time_series_volatility import Mean_Time_Series

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

    p = argparse.ArgumentParser(
        description=(
            "Unified DLM sampler: "
            "• Homoskedastic → Kalman FFBS + Gibbs (conjugate/PC) "
            "• Heteroskedastic → RBPF (η=ln σ) with embedded Kalman and PC hyperpriors.\n"
            "Seasonal state uses newest-first; observation loads the first seasonal coord."
        )
    )

    # --- Simulation controls ---
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--period", type=int, default=12)
    p.add_argument("--start-date", type=str, default="2000-01-01")

    # μ-block modes
    p.add_argument("--level-mode",   choices=["dynamic", "deterministic", "none"], default="dynamic")
    p.add_argument("--trend-mode",   choices=["dynamic", "deterministic", "none"], default="dynamic")
    p.add_argument("--seasonal-mode",choices=["dynamic", "deterministic", "none"], default="dynamic")

    # η-block (log σ) modes — if any is "dynamic" we switch to RBPF
    p.add_argument("--level-mode-sigma",   choices=["dynamic","deterministic","none"], default="deterministic")
    p.add_argument("--trend-mode-sigma",   choices=["dynamic","deterministic","none"], default="none")
    p.add_argument("--seasonal-mode-sigma",choices=["dynamic","deterministic","none"], default="none")

    # Simulation scales (for generator)
    p.add_argument("--sigma", type=float, default=2.0)  # base scale used only if η-block is deterministic zeros
    # μ innovations (simulator)
    p.add_argument("--q-level",  type=float, default=0.05)
    p.add_argument("--q-trend",  type=float, default=0.002)
    p.add_argument("--q-season", type=float, default=0.15)
    # η innovations (simulator) — only used if η modes set to dynamic
    p.add_argument("--q-level-sigma",  type=float, default=0.00)
    p.add_argument("--q-trend-sigma",  type=float, default=0.00)
    p.add_argument("--q-season-sigma", type=float, default=0.00)

    # μ priors / fixed (simulator)
    p.add_argument("--m0-level",  type=float, default=3.0)
    p.add_argument("--v0-level",  type=float, default=0.25)
    p.add_argument("--m0-trend",  type=float, default=0.015)
    p.add_argument("--v0-trend",  type=float, default=0.05)
    p.add_argument("--m0-season", type=str, default=None, help="comma-separated (length p-1, newest-first)")
    p.add_argument("--v0-season", type=str, default=None, help="comma-separated (length p-1)")

    # η priors / fixed (simulator; for ln σ)
    p.add_argument("--m0-level-sigma",  type=float, default=0.0)
    p.add_argument("--v0-level-sigma",  type=float, default=0.0)
    p.add_argument("--m0-trend-sigma",  type=float, default=0.00)
    p.add_argument("--v0-trend-sigma",  type=float, default=0.50)
    p.add_argument("--m0-season-sigma", type=str, default=None, help="comma-separated (length p-1, newest-first)")
    p.add_argument("--v0-season-sigma", type=str, default=None, help="comma-separated (length p-1)")

    # --- Inference priors (Gibbs/PC, mean block) ---
    p.add_argument("--prior-a-sigma", type=float, default=2.0)  # used when homoskedastic
    p.add_argument("--prior-b-sigma", type=float, default=1.0)

    p.add_argument("--prior-m-m0-alpha", type=float, default=0.0)
    p.add_argument("--prior-s-m0-alpha", type=float, default=10.0)
    p.add_argument("--prior-m-m0-beta",  type=float, default=0.0)
    p.add_argument("--prior-s-m0-beta",  type=float, default=10.0)
    p.add_argument("--prior-m-m0-gamma", type=str,   default=None, help="comma-separated (length p-1, newest-first)")
    p.add_argument("--prior-s-m0-gamma", type=float, default=5.0)

    p.add_argument("--prior-a-P0-alpha", type=float, default=2.0)
    p.add_argument("--prior-b-P0-alpha", type=float, default=1.0)
    p.add_argument("--prior-a-P0-beta",  type=float, default=2.0)
    p.add_argument("--prior-b-P0-beta",  type=float, default=1.0)
    p.add_argument("--prior-a-P0-gamma", type=float, default=2.0)
    p.add_argument("--prior-b-P0-gamma", type=float, default=1.0)

    # --- Inference priors (Gibbs/PC, sigma block for RBPF) ---
    p.add_argument("--prior-m-m0-alpha-sig", type=float, default=0.0)
    p.add_argument("--prior-s-m0-alpha-sig", type=float, default=5.0)
    p.add_argument("--prior-m-m0-beta-sig",  type=float, default=0.0)
    p.add_argument("--prior-s-m0-beta-sig",  type=float, default=5.0)
    p.add_argument("--prior-m-m0-gamma-sig", type=str,   default=None, help="comma-separated (length p-1, newest-first)")
    p.add_argument("--prior-s-m0-gamma-sig", type=float, default=3.0)

    p.add_argument("--prior-a-P0-alpha-sig", type=float, default=2.0)
    p.add_argument("--prior-b-P0-alpha-sig", type=float, default=1.0)
    p.add_argument("--prior-a-P0-beta-sig",  type=float, default=2.0)
    p.add_argument("--prior-b-P0-beta-sig",  type=float, default=1.0)
    p.add_argument("--prior-a-P0-gamma-sig", type=float, default=2.0)
    p.add_argument("--prior-b-P0-gamma-sig", type=float, default=1.0)

    # --- PC priors for process sds (mean block) ---
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

    # --- PC priors for process sds (sigma block) ---
    p.add_argument("--pc-frac-alpha-sig",  type=float, default=0.10)
    p.add_argument("--pc-frac-beta-sig",   type=float, default=0.10)
    p.add_argument("--pc-frac-gamma-sig",  type=float, default=0.10)
    p.add_argument("--pc-lambda-alpha-sig", type=float, default=None)
    p.add_argument("--pc-lambda-beta-sig",  type=float, default=None)
    p.add_argument("--pc-lambda-gamma-sig", type=float, default=None)
    p.add_argument("--pc-a-lambda-alpha-sig", type=float, default=1.0)
    p.add_argument("--pc-b-lambda-alpha-sig", type=float, default=1.0)
    p.add_argument("--pc-a-lambda-beta-sig",  type=float, default=1.0)
    p.add_argument("--pc-b-lambda-beta-sig",  type=float, default=1.0)
    p.add_argument("--pc-a-lambda-gamma-sig", type=float, default=1.0)
    p.add_argument("--pc-b-lambda-gamma-sig", type=float, default=1.0)

    # --- Sampler config & slice ---
    p.add_argument("--n-iter", type=int, default=4000)
    p.add_argument("--burn", type=int, default=1000)
    p.add_argument("--thin", type=int, default=1)
    p.add_argument("--seed", type=int, default=40)
    p.add_argument("--progress", default=True)
    p.add_argument("--progress-every", type=int, default=0)
    p.add_argument("--slice-w", type=float, default=0.4)
    p.add_argument("--slice-m", type=int, default=40)
    p.add_argument("--slice-max-shrink", type=int, default=1000)

    # RBPF config
    p.add_argument("--n-particles", type=int, default=128)
    p.add_argument("--ess-resample", type=float, default=0.5)
    p.add_argument("--resample-method", choices=["systematic","multinomial"], default="systematic")

    # --- Initial values for inference (μ) ---
    p.add_argument("--sigma-init", type=float, default=2.0)  # used only if homoskedastic (sd)
    p.add_argument("--s-alpha-init", type=float, default=1e-2)
    p.add_argument("--s-beta-init",  type=float, default=1e-3)
    p.add_argument("--s-gamma-init", type=float, default=1.0)
    p.add_argument("--m0-gamma-init", type=str, default=None, help="comma-separated (length p-1, newest-first)")
    p.add_argument("--P0-alpha-init", type=float, default=0.25)
    p.add_argument("--P0-beta-init",  type=float, default=0.05)
    p.add_argument("--P0-gamma-init", type=float, default=0.25)

    # --- Initial values for inference (η) ---
    p.add_argument("--s-alpha-sig-init", type=float, default=0.05)
    p.add_argument("--s-beta-sig-init",  type=float, default=0.01)
    p.add_argument("--s-gamma-sig-init", type=float, default=0.10)
    p.add_argument("--m0-gamma-sig-init", type=str, default=None, help="comma-separated (length p-1, newest-first)")
    p.add_argument("--P0-alpha-sig-init", type=float, default=0.25)
    p.add_argument("--P0-beta-sig-init",  type=float, default=0.05)
    p.add_argument("--P0-gamma-sig-init", type=float, default=0.25)
    p.add_argument("--m0-level-sig-init", type=float, default=0.0)  # η level prior mean

    # --- I/O & plotting ---
    p.add_argument("--out-dir", type=str, default="results/simulations/DLM")
    p.add_argument("--plot", default=True)
    p.add_argument("--print-summary", default=True)

    args = p.parse_args()
    np.random.seed(args.seed)

    start_date = _parse_date(args.start_date)

    # seasonal priors for simulator
    m0_season = _csv_floats_or_none(args.m0_season) or [0.0]*(args.period-1)
    v0_season = _csv_floats_or_none(args.v0_season) or [0.25]*(args.period-1)
    m0_season_sig = _csv_floats_or_none(args.m0_season_sigma) or [0.0]*(args.period-1)
    v0_season_sig = _csv_floats_or_none(args.v0_season_sigma) or (
        [0.0]*(args.period-1) if args.seasonal_mode_sigma=="deterministic" else [0.5]*(args.period-1)
    )

    # --- Simulate data ---
    # When μ-level is "none", simulator expects deterministic 0 level; keep mean block consistent
    sim_level_mode_mu = args.level_mode if args.level_mode != "none" else "deterministic"
    mts = Mean_Time_Series(
        # μ block
        level_mode=sim_level_mode_mu,
        trend_mode=args.trend_mode,
        seasonal_mode=args.seasonal_mode,
        # η block
        level_mode_sigma=args.level_mode_sigma,
        trend_mode_sigma=args.trend_mode_sigma,
        seasonal_mode_sigma=args.seasonal_mode_sigma,
        # period
        period=args.period,
        # μ innovations
        q_level=(args.q_level if sim_level_mode_mu == "dynamic" else 0.0),
        q_trend=(args.q_trend if args.trend_mode == "dynamic" else 0.0),
        q_season=(args.q_season if args.seasonal_mode == "dynamic" else 0.0),
        # η innovations
        q_level_sigma=(args.q_level_sigma if args.level_mode_sigma == "dynamic" else 0.0),
        q_trend_sigma=(args.q_trend_sigma if args.trend_mode_sigma == "dynamic" else 0.0),
        q_season_sigma=(args.q_season_sigma if args.seasonal_mode_sigma == "dynamic" else 0.0),
        # μ priors/fixed
        m0_level=(0.0 if args.level_mode == "none" else args.m0_level),
        v0_level=(args.v0_level if sim_level_mode_mu == "dynamic" else 0.0),
        m0_trend=(0.0 if args.trend_mode == "none" else args.m0_trend),
        v0_trend=(args.v0_trend if args.trend_mode == "dynamic" else 0.0),
        m0_season=m0_season, v0_season=v0_season,
        # η priors/fixed
        m0_level_sigma=args.m0_level_sigma,
        v0_level_sigma=args.v0_level_sigma,
        m0_trend_sigma=(0.0 if args.trend_mode_sigma == "none" else args.m0_trend_sigma),
        v0_trend_sigma=args.v0_trend_sigma,
        m0_season_sigma=m0_season_sig, v0_season_sigma=v0_season_sig,
        # time
        start_date=start_date,
    )

    y = []
    for _ in range(args.T):
        mts.move()
        y.append(mts.measure())
    y = np.asarray(y, float)

    truths = mts.get_truth_paths(as_numpy=True)
    dates_T   = truths["index"][: args.T]
    mu_T      = truths["mu_t"][1 : 1 + args.T]
    sigma_T   = truths["sigma_t"][1 : 1 + args.T]

    # --- Initial seasonal means for μ (newest-first, length p-1) ---
    if args.m0_gamma_init is not None:
        m0_gamma_init = [float(z) for z in args.m0_gamma_init.split(",") if z.strip() != ""]
    else:
        # crude: de-meaned median-of-season; take first p-1 entries (newest-first)
        S = np.array([np.median(y[k::args.period]) for k in range(args.period)], float)
        base = S - S.mean()
        m0_gamma_init = base[: args.period - 1].tolist()

    # --- Initial seasonal means for η (if used) ---
    if args.m0_gamma_sig_init is not None:
        m0_gamma_sig_init = [float(z) for z in args.m0_gamma_sig_init.split(",") if z.strip() != ""]
    else:
        m0_gamma_sig_init = [0.0] * (args.period - 1)

    # --- Priors and config ---
    pri_gamma_vec = _csv_floats_or_none(args.prior_m_m0_gamma)
    pri_gamma_sig_vec = _csv_floats_or_none(args.prior_m_m0_gamma_sig)

    priors = Priors(
        # obs variance (homoskedastic path)
        a_sigma=float(args.prior_a_sigma),
        b_sigma=float(args.prior_b_sigma),
        # μ
        m_m0_alpha=float(args.prior_m_m0_alpha), s_m0_alpha=float(args.prior_s_m0_alpha),
        m_m0_beta=float(args.prior_m_m0_beta),   s_m0_beta=float(args.prior_s_m0_beta),
        m_m0_gamma=None if pri_gamma_vec is None else pri_gamma_vec, s_m0_gamma=float(args.prior_s_m0_gamma),
        a_P0_alpha=float(args.prior_a_P0_alpha), b_P0_alpha=float(args.prior_b_P0_alpha),
        a_P0_beta=float(args.prior_a_P0_beta),   b_P0_beta=float(args.prior_b_P0_beta),
        a_P0_gamma=float(args.prior_a_P0_gamma), b_P0_gamma=float(args.prior_b_P0_gamma),
        pc_alpha=PCPrior(lambda_s=(None if args.pc_lambda_alpha is None else float(args.pc_lambda_alpha)),
                         a_lambda=float(args.pc_a_lambda_alpha), b_lambda=float(args.pc_b_lambda_alpha),
                         frac=float(args.pc_frac_alpha), alpha_prob=float(args.pc_alpha_prob)),
        pc_beta=PCPrior(lambda_s=(None if args.pc_lambda_beta is None else float(args.pc_lambda_beta)),
                        a_lambda=float(args.pc_a_lambda_beta),  b_lambda=float(args.pc_b_lambda_beta),
                        frac=float(args.pc_frac_beta),  alpha_prob=float(args.pc_alpha_prob)),
        pc_gamma=PCPrior(lambda_s=(None if args.pc_lambda_gamma is None else float(args.pc_lambda_gamma)),
                         a_lambda=float(args.pc_a_lambda_gamma), b_lambda=float(args.pc_b_lambda_gamma),
                         frac=float(args.pc_frac_gamma), alpha_prob=float(args.pc_alpha_prob)),
        # η (used by RBPF)
        m_m0_alpha_sig=float(args.prior_m_m0_alpha_sig), s_m0_alpha_sig=float(args.prior_s_m0_alpha_sig),
        m_m0_beta_sig=float(args.prior_m_m0_beta_sig),   s_m0_beta_sig=float(args.prior_s_m0_beta_sig),
        m_m0_gamma_sig=None if pri_gamma_sig_vec is None else pri_gamma_sig_vec,
        s_m0_gamma_sig=float(args.prior_s_m0_gamma_sig),
        a_P0_alpha_sig=float(args.prior_a_P0_alpha_sig), b_P0_alpha_sig=float(args.prior_b_P0_alpha_sig),
        a_P0_beta_sig=float(args.prior_a_P0_beta_sig),   b_P0_beta_sig=float(args.prior_b_P0_beta_sig),
        a_P0_gamma_sig=float(args.prior_a_P0_gamma_sig), b_P0_gamma_sig=float(args.prior_b_P0_gamma_sig),
        pc_alpha_sig=PCPrior(lambda_s=(None if args.pc_lambda_alpha_sig is None else float(args.pc_lambda_alpha_sig)),
                             a_lambda=float(args.pc_a_lambda_alpha_sig), b_lambda=float(args.pc_b_lambda_alpha_sig),
                             frac=float(args.pc_frac_alpha_sig), alpha_prob=float(args.pc_alpha_prob)),
        pc_beta_sig=PCPrior(lambda_s=(None if args.pc_lambda_beta_sig is None else float(args.pc_lambda_beta_sig)),
                            a_lambda=float(args.pc_a_lambda_beta_sig),  b_lambda=float(args.pc_b_lambda_beta_sig),
                            frac=float(args.pc_frac_beta_sig),  alpha_prob=float(args.pc_alpha_prob)),
        pc_gamma_sig=PCPrior(lambda_s=(None if args.pc_lambda_gamma_sig is None else float(args.pc_lambda_gamma_sig)),
                             a_lambda=float(args.pc_a_lambda_gamma_sig), b_lambda=float(args.pc_b_lambda_gamma_sig),
                             frac=float(args.pc_frac_gamma_sig), alpha_prob=float(args.pc_alpha_prob)),
    )

    cfg = SamplerConfig(
        n_iter=int(args.n_iter),
        burn=int(args.burn),
        thin=int(args.thin),
        random_seed=int(args.seed),
        progress=bool(args.progress),
        progress_every=int(args.progress_every),
        slice_w=float(args.slice_w),
        slice_m=int(args.slice_m),
        slice_max_shrink=int(args.slice_max_shrink),
        # RBPF
        n_particles=int(args.n_particles),
        ess_resample=float(args.ess_resample),
        resample_method=str(args.resample_method),
    )

    # --- Build and run unified sampler ---
    sampler = DLMUnifiedSampler(
        y=y,
        period=int(args.period),
        # μ
        level_mode=args.level_mode,
        trend_mode=args.trend_mode,
        seasonal_mode=args.seasonal_mode,
        # η
        level_mode_sigma=args.level_mode_sigma,
        trend_mode_sigma=args.trend_mode_sigma,
        seasonal_mode_sigma=args.seasonal_mode_sigma,
        # μ inits
        sigma2_init=float(args.sigma_init) ** 2,   # only used if homoskedastic
        s_alpha_init=float(args.s_alpha_init),
        s_beta_init=float(args.s_beta_init),
        s_gamma_init=float(args.s_gamma_init),
        m0_alpha_init=(0.0 if args.level_mode == "none" else float(args.m0_level)),
        P0_alpha_init=float(args.P0_alpha_init),
        m0_beta_init=float(args.m0_trend if args.trend_mode != "none" else 0.0),
        P0_beta_init=float(args.P0_beta_init),
        m0_gamma_init=m0_gamma_init,            # μ seasonal newest-first, length p-1
        P0_gamma_init=float(args.P0_gamma_init),
        # η inits (RBPF only)
        m0_alpha_sig_init=float(args.m0_level_sig_init),
        P0_alpha_sig_init=float(args.P0_alpha_sig_init),
        m0_beta_sig_init=0.0,
        P0_beta_sig_init=float(args.P0_beta_sig_init),
        m0_gamma_sig_init=m0_gamma_sig_init,    # η seasonal newest-first, length p-1
        P0_gamma_sig_init=float(args.P0_gamma_sig_init),
        s_alpha_sig_init=float(args.s_alpha_sig_init),
        s_beta_sig_init=float(args.s_beta_sig_init),
        s_gamma_sig_init=float(args.s_gamma_sig_init),
        priors=priors,
        cfg=cfg,
    )

    if args.print_summary:
        with np.printoptions(suppress=True, precision=4):
            print("\n--- Summary (simulation) ---")
            print(f"[μ]  level={mts.level_mode_mu}, trend={mts.trend_mode_mu}, season={mts.seasonal_mode_mu}")
            print(f"[η]  level={mts.level_mode_sigma}, trend={mts.trend_mode_sigma}, season={mts.seasonal_mode_sigma}")
            print(f"[μ]  q_level={mts.q_level}, q_trend={mts.q_trend}, q_season={mts.q_season}")
            print(f"[η]  q_level={mts.q_level_sigma}, q_trend={mts.q_trend_sigma}, q_season={mts.q_season_sigma}")
            print(f"period={args.period}, start={dates_T[0]}, end={dates_T[-1]}")
            print(f"y mean={y.mean():.3f}, sd={y.std(ddof=1):.3f}")
            print(f"avg sigma(truth)={sigma_T.mean():.3f}, sd sigma(truth)={sigma_T.std(ddof=1):.3f}")

    t0 = time.time()
    post = sampler.run()
    elapsed = time.time() - t0
    print(f"Run time: {elapsed:.2f}s")

    out_dir = os.path.join(
        args.out_dir,
        f"{args.level_mode}-{args.trend_mode}-{args.seasonal_mode}"
        f"_sig({args.level_mode_sigma}-{args.trend_mode_sigma}-{args.seasonal_mode_sigma})_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(out_dir, exist_ok=True)
    # save using engine's saver to include sigma-mode metadata
    sampler.save_posterior(
        out_npz_path=os.path.join(out_dir, "posterior.npz"),
        extra_meta={"elapsed_seconds": float(elapsed)}
    )

    # ---- Print quick summaries ----
    if args.print_summary:
        with np.printoptions(suppress=True, precision=4):
            print("\n--- Summary (posterior means) ---")
            mu_hat = post["mu"].mean(axis=0)
            sig_hat = post["sigma"].mean(axis=0) if "sigma" in post else np.full(args.T, np.nan)
            print(f"μ̂_t mean (first 3): {mu_hat[:3]}")
            print(f"σ̂_t mean (first 3): {sig_hat[:3]}")
            # mean-block process noise
            if "Q_alpha" in post and post["Q_alpha"].size:
                m = float(np.mean(post["Q_alpha"])); print(f"Q_alpha: {m:.4g} (√≈{math.sqrt(m):.4g})")
            else:
                print("Q_alpha: n/a")
            if "Q_beta" in post and post["Q_beta"].size:
                m = float(np.mean(post["Q_beta"]));  print(f"Q_beta:  {m:.4g} (√≈{math.sqrt(m):.4g})")
            else:
                print("Q_beta:  n/a")
            if "Q_gamma" in post and np.size(post["Q_gamma"])>0:
                m = float(np.mean(post["Q_gamma"])); print(f"Q_gamma: {m:.4g} (√≈{math.sqrt(m):.4g})")
            else:
                print("Q_gamma: n/a")
            # sigma-block process noise (if RBPF)
            if "Q_alpha_sig" in post and post["Q_alpha_sig"].size:
                m = float(np.mean(post["Q_alpha_sig"])); print(f"Q_alpha_sig: {m:.4g} (√≈{math.sqrt(m):.4g})")
            if "Q_beta_sig" in post and post["Q_beta_sig"].size:
                m = float(np.mean(post["Q_beta_sig"]));  print(f"Q_beta_sig:  {m:.4g} (√≈{math.sqrt(m):.4g})")
            if "Q_gamma_sig" in post and np.size(post["Q_gamma_sig"])>0:
                m = float(np.mean(post["Q_gamma_sig"])); print(f"Q_gamma_sig: {m:.4g} (√≈{math.sqrt(m):.4g})")

    # ---- Plots ----
    if args.plot:
        mu_hat = post["mu"].mean(axis=0)
        sigma_hat = post["sigma"].mean(axis=0) if "sigma" in post else np.full(args.T, np.nan)

        # Figure 1: y and μ
        plt.figure(figsize=(10, 4))
        plt.plot(dates_T, y, label=r"$y_t$", linewidth=1.0)
        plt.plot(dates_T, mu_T, "--", label=r"$\mu_t$ (truth)", linewidth=1.0)
        plt.plot(dates_T, mu_hat, "-.", label=r"$\hat{\mu}_t$ (post mean)", linewidth=1.0)
        ttl = (f"DLM μ: level={args.level_mode}, trend={args.trend_mode}, season={args.seasonal_mode} | "
               f"σ modes: {args.level_mode_sigma},{args.trend_mode_sigma},{args.seasonal_mode_sigma}")
        plt.title(ttl)
        plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

        # Figure 2: σ paths (if available)
        if np.isfinite(sigma_hat).all():
            plt.figure(figsize=(10, 4))
            plt.plot(dates_T, sigma_T, "--", label=r"$\sigma_t$ (truth)", linewidth=1.0)
            plt.plot(dates_T, sigma_hat, "-.", label=r"$\hat{\sigma}_t$ (post mean)", linewidth=1.0)
            plt.title("Scale (σ) paths")
            plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()
