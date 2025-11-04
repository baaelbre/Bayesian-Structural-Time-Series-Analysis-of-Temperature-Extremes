# %% inference/dlm_full_gibbs_sv.py
from __future__ import annotations

import json, math, os, time, warnings
from dataclasses import dataclass, asdict, field
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
# Kim–Shephard–Chib (1998) 7-component log(ε^2) mixture
# log(ε^2) ≈ m_j + sqrt(v_j) * N(0,1) with probabilities p_j
# (commonly used constants; see Kim et al., 1998; Omori et al., 2007)
# We use the half-log transform z_t = 0.5*log r_t^2, so:
# z_t = eta_t + 0.5*m_j + sqrt(0.25*v_j)*N(0,1)
# =============================================================================
_KSC7_p = np.array([0.00730, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.25750])
_KSC7_m = np.array([ -10.12999,  -3.97281,   -8.56686,  -2.77786,  -1.24245,  -0.14610,   1.00000])
_KSC7_v = np.array([   5.79596,   2.61369,    5.17950,   1.79518,   1.16770,   0.73504,   0.34802])
# for z_t = 0.5 log r_t^2:
_KSC7_mh = 0.5 * _KSC7_m
_KSC7_vh = 0.25 * _KSC7_v


# =============================================================================
# Priors & Config
# =============================================================================
@dataclass
class PCPrior:
    lambda_s: Optional[float] = None   # fixed if provided
    a_lambda: float = 1.0
    b_lambda: float = 1.0
    frac: float = 0.10
    alpha_prob: float = 0.05

@dataclass
class PriorsBlock:
    # m0 priors (for dynamic x0 means or deterministic params)
    m_m0_alpha: float = 0.0
    s_m0_alpha: float = 10.0
    m_m0_beta: float  = 0.0
    s_m0_beta: float  = 10.0
    m_m0_gamma: Optional[Sequence[float]] = None   # len p-1 (NEWEST-FIRST)
    s_m0_gamma: float = 5.0

    # Initial-state variances P0 ~ InvGamma(a,b)
    a_P0_alpha: float = 2.0
    b_P0_alpha: float = 1.0
    a_P0_beta:  float = 2.0
    b_P0_beta:  float = 1.0
    a_P0_gamma: float = 2.0
    b_P0_gamma: float = 1.0

    # PC priors for process sds
    pc_alpha: PCPrior = field(default_factory=PCPrior)
    pc_beta:  PCPrior = field(default_factory=PCPrior)
    pc_gamma: PCPrior = field(default_factory=PCPrior)

@dataclass
class PriorsFull:
    # Observation precision prior for *homoskedastic fallback only* (rarely used now)
    a_sigma: float = 2.0
    b_sigma: float = 1.0
    # Separate priors for μ-block and η-block
    mu: PriorsBlock = field(default_factory=PriorsBlock)
    sig: PriorsBlock = field(default_factory=PriorsBlock)

@dataclass
class SamplerConfig:
    n_iter: int = 8000
    burn: int = 3000
    thin: int = 1
    random_seed: Optional[int] = 7
    progress: bool = True
    progress_every: int = 0  # 0 => ~2% of n_iter

    # Slice sampler
    slice_w: float = 0.4
    slice_m: int = 40
    slice_max_shrink: int = 1000

@dataclass
class SpikeSlabConfig:
    enabled: bool = False
    # μ-block
    pi_alpha_mu_on: float = 0.5
    pi_beta_mu_on:  float = 0.5
    pi_gamma_mu_on: float = 0.5
    # η-block
    pi_alpha_sig_on: float = 0.5
    pi_beta_sig_on:  float = 0.5
    pi_gamma_sig_on: float = 0.5


# =============================================================================
# Helper: seasonal transition (newest-first) for dimension p-1
# =============================================================================
def _season_F(period: int) -> np.ndarray:
    p = int(period)
    m = p - 1
    if m <= 0:
        return np.zeros((0, 0))
    F = np.zeros((m, m))
    F[0, :] = -1.0
    if m > 1:
        F[1:, :-1] = np.eye(m - 1)
    return F


# =============================================================================
# Full sampler with μ and η blocks
# =============================================================================
class DLMGibbsFull:
    """
    Full structural Gaussian model:

      y_t ~ N( μ_t ,  exp(2*η_t) )

    Parallel blocks:
      μ_t = deterministic_mu(t) + H_mu x_mu,t         (linear Gaussian)
      η_t = deterministic_sig(t) + H_sig x_sig,t      (SV via KSC mixture on log r_t^2)

    Modes per block: level in {dynamic, deterministic}, trend in {dynamic, deterministic, none},
    season in {dynamic, deterministic, none}. Seasonal state is newest-first (length p-1),
    observation loads the FIRST seasonal coord.

    Inference:
      • μ-block: FFBS with time-varying R_t = exp(2*η_t) known given η-path
      • η-block: SV step — sample mixture indicators s_t, build pseudo-obs
                 z_t = 0.5 log r_t^2 = η_t + m_{s_t} + e_t, e_t ~ N(0, v_{s_t})
                 then FFBS on η-state.
      • Process sds: PC priors + optional Gamma hyperpriors (slice on log s)
      • Optional spike-and-slab per component in both blocks (collapsed flips)
    """

    # --------------------------- Construction --------------------------- #
    def __init__(
        self,
        y: np.ndarray,
        period: int,

        # μ-block modes
        level_mode_mu: str = "dynamic",
        trend_mode_mu: str = "dynamic",
        seasonal_mode_mu: str = "dynamic",

        # η-block modes
        level_mode_sig: str = "deterministic",
        trend_mode_sig: str = "none",
        seasonal_mode_sig: str = "none",

        # Initial values (μ-block)
        m0_alpha_mu_init: float = 0.0,
        P0_alpha_mu_init: float = 1.0,
        m0_beta_mu_init:  float = 0.0,
        P0_beta_mu_init:  float = 1.0,
        m0_gamma_mu_init: Optional[Sequence[float]] = None,  # len p-1, newest-first
        P0_gamma_mu_init: float = 1.0,
        s_alpha_mu_init:  float = 1e-2,
        s_beta_mu_init:   float = 1e-3,
        s_gamma_mu_init:  float = 1e-3,

        # Initial values (η-block)
        m0_alpha_sig_init: float = 0.0,
        P0_alpha_sig_init: float = 0.0,     # 0 → deterministic unless dynamic
        m0_beta_sig_init:  float = 0.0,
        P0_beta_sig_init:  float = 1.0,
        m0_gamma_sig_init: Optional[Sequence[float]] = None,  # len p-1, newest-first
        P0_gamma_sig_init: float = 1.0,
        s_alpha_sig_init:  float = 0.0,
        s_beta_sig_init:   float = 0.0,
        s_gamma_sig_init:  float = 0.0,

        priors: PriorsFull = PriorsFull(),
        cfg: SamplerConfig = SamplerConfig(),
        spike: Optional[SpikeSlabConfig] = None,
    ):
        # Data
        self.y = np.asarray(y, float)
        self.T = int(self.y.size)
        self.period = int(period)
        if self.period < 2:
            raise ValueError("period must be >= 2")

        # Modes validation
        ok = {"dynamic", "deterministic", "none"}
        for mm in (level_mode_mu, trend_mode_mu, seasonal_mode_mu):
            if mm not in ok: raise ValueError("invalid μ-mode")
        for mm in (level_mode_sig, trend_mode_sig, seasonal_mode_sig):
            if mm not in ok: raise ValueError("invalid η-mode")
        if trend_mode_mu == "dynamic" and level_mode_mu != "dynamic":
            raise ValueError("μ dynamic trend requires μ dynamic level")
        if trend_mode_sig == "dynamic" and level_mode_sig != "dynamic":
            raise ValueError("η dynamic trend requires η dynamic level")

        self.level_mode_mu, self.trend_mode_mu, self.seasonal_mode_mu = level_mode_mu, trend_mode_mu, seasonal_mode_mu
        self.level_mode_sig, self.trend_mode_sig, self.seasonal_mode_sig = level_mode_sig, trend_mode_sig, seasonal_mode_sig

        # Priors / config
        self.priors = priors
        self.cfg = cfg
        if cfg.random_seed is not None:
            np.random.seed(cfg.random_seed)

        # Spike&slab
        self.spike = spike or SpikeSlabConfig(enabled=False)

        # Layout μ-block
        self._layout_mu: List[str] = []
        if self.level_mode_mu == "dynamic": self._layout_mu.append("alpha_mu")
        if self.trend_mode_mu == "dynamic": self._layout_mu.append("beta_mu")
        if self.seasonal_mode_mu == "dynamic":
            self._layout_mu.extend([f"gmu{k}" for k in range(1, self.period)])
        self.dim_mu = len(self._layout_mu)
        self.idx_alpha_mu = self._layout_mu.index("alpha_mu") if "alpha_mu" in self._layout_mu else None
        self.idx_beta_mu  = self._layout_mu.index("beta_mu")  if "beta_mu"  in self._layout_mu else None
        if self.seasonal_mode_mu == "dynamic":
            self.idx_gmu_start = self._layout_mu.index("gmu1")
            self.idx_gmu_end   = self.idx_gmu_start + (self.period - 2)

        # Layout η-block
        self._layout_sig: List[str] = []
        if self.level_mode_sig == "dynamic": self._layout_sig.append("alpha_sig")
        if self.trend_mode_sig == "dynamic": self._layout_sig.append("beta_sig")
        if self.seasonal_mode_sig == "dynamic":
            self._layout_sig.extend([f"gsig{k}" for k in range(1, self.period)])
        self.dim_sig = len(self._layout_sig)
        self.idx_alpha_sig = self._layout_sig.index("alpha_sig") if "alpha_sig" in self._layout_sig else None
        self.idx_beta_sig  = self._layout_sig.index("beta_sig")  if "beta_sig"  in self._layout_sig else None
        if self.seasonal_mode_sig == "dynamic":
            self.idx_gsig_start = self._layout_sig.index("gsig1")
            self.idx_gsig_end   = self.idx_gsig_start + (self.period - 2)

        # Parameters: process sds (both blocks) + PC λ inits
        self.s_alpha_mu, self.s_beta_mu, self.s_gamma_mu = float(s_alpha_mu_init), float(s_beta_mu_init), float(s_gamma_mu_init)
        self.s_alpha_sig, self.s_beta_sig, self.s_gamma_sig = float(s_alpha_sig_init), float(s_beta_sig_init), float(s_gamma_sig_init)
        self.lambda_alpha_mu, self.lambda_beta_mu, self.lambda_gamma_mu = self._init_pc_lambdas_block(self.priors.mu, target="mu")
        self.lambda_alpha_sig, self.lambda_beta_sig, self.lambda_gamma_sig = self._init_pc_lambdas_block(self.priors.sig, target="sig")

        # Spike&slab indicators (default ON if present)
        self.z_alpha_mu = 1 if self.idx_alpha_mu is not None else 0
        self.z_beta_mu  = 1 if self.idx_beta_mu  is not None else 0
        self.z_gamma_mu = 1 if self.seasonal_mode_mu == "dynamic" else 0
        self.z_alpha_sig = 1 if self.idx_alpha_sig is not None else 0
        self.z_beta_sig  = 1 if self.idx_beta_sig  is not None else 0
        self.z_gamma_sig = 1 if self.seasonal_mode_sig == "dynamic" else 0

        # Initial m0/P0 (μ)
        self.m0_alpha_mu = float(m0_alpha_mu_init) if self.idx_alpha_mu is not None else 0.0
        self.P0_alpha_mu = float(P0_alpha_mu_init) if self.idx_alpha_mu is not None else 0.0
        self.m0_beta_mu  = float(m0_beta_mu_init)  if self.idx_beta_mu  is not None else 0.0
        self.P0_beta_mu  = float(P0_beta_mu_init)  if self.idx_beta_mu  is not None else 0.0
        if self.seasonal_mode_mu == "dynamic":
            if m0_gamma_mu_init is None:
                self.m0_gamma_mu = np.zeros(self.period - 1)
            else:
                g = np.asarray(m0_gamma_mu_init, float)
                if g.size != self.period - 1:
                    raise ValueError("m0_gamma_mu_init must have length p-1 (newest-first)")
                self.m0_gamma_mu = g
            self.P0_gamma_mu = float(P0_gamma_mu_init)
        else:
            # for deterministic/no season, we’ll store full length-p vector later
            self.m0_gamma_mu = None
            self.P0_gamma_mu = 0.0

        # Initial m0/P0 (η)
        self.m0_alpha_sig = float(m0_alpha_sig_init) if self.idx_alpha_sig is not None else 0.0
        self.P0_alpha_sig = float(P0_alpha_sig_init) if self.idx_alpha_sig is not None else 0.0
        self.m0_beta_sig  = float(m0_beta_sig_init)  if self.idx_beta_sig  is not None else 0.0
        self.P0_beta_sig  = float(P0_beta_sig_init)  if self.idx_beta_sig  is not None else 0.0
        if self.seasonal_mode_sig == "dynamic":
            if m0_gamma_sig_init is None:
                self.m0_gamma_sig = np.zeros(self.period - 1)
            else:
                g = np.asarray(m0_gamma_sig_init, float)
                if g.size != self.period - 1:
                    raise ValueError("m0_gamma_sig_init must have length p-1 (newest-first)")
                self.m0_gamma_sig = g
            self.P0_gamma_sig = float(P0_gamma_sig_init)
        else:
            self.m0_gamma_sig = None
            self.P0_gamma_sig = 0.0

        # Deterministic contributions (outside state)
        # μ
        if self.level_mode_mu == "deterministic":
            self.m0_alpha_mu = float(self.priors.mu.m_m0_alpha)
        if self.trend_mode_mu == "deterministic":
            self.m0_beta_mu = float(self.priors.mu.m_m0_beta)
        if self.seasonal_mode_mu == "deterministic":
            base = np.zeros(self.period - 1) if self.priors.mu.m_m0_gamma is None else np.asarray(self.priors.mu.m_m0_gamma, float)
            if base.size != self.period - 1:
                raise ValueError("priors.mu.m_m0_gamma must have length p-1")
            self.m0_gamma_mu = np.r_[base, -base.sum()]

        # η
        if self.level_mode_sig == "deterministic":
            self.m0_alpha_sig = float(self.priors.sig.m_m0_alpha)
        if self.trend_mode_sig == "deterministic":
            self.m0_beta_sig = float(self.priors.sig.m_m0_beta)
        if self.seasonal_mode_sig == "deterministic":
            base = np.zeros(self.period - 1) if self.priors.sig.m_m0_gamma is None else np.asarray(self.priors.sig.m_m0_gamma, float)
            if base.size != self.period - 1:
                raise ValueError("priors.sig.m_m0_gamma must have length p-1")
            self.m0_gamma_sig = np.r_[base, -base.sum()]

        # Latent paths
        self.x_mu  = np.zeros((self.T + 1, self.dim_mu))
        self.x_sig = np.zeros((self.T + 1, self.dim_sig))
        if self.dim_mu > 0:
            m0, P0 = self._current_m0_P0_block(target="mu")
            self.x_mu[0] = np.random.multivariate_normal(m0, np.diag(P0) + 1e-10*np.eye(self.dim_mu))
        if self.dim_sig > 0:
            m0, P0 = self._current_m0_P0_block(target="sig")
            self.x_sig[0] = np.random.multivariate_normal(m0, np.diag(P0) + 1e-10*np.eye(self.dim_sig))

        # Storage
        self.keep: Dict[str, np.ndarray] = {}

    # --------------------- PC λ initialization per block --------------------- #
    def _init_pc_lambdas_block(self, pri: PriorsBlock, target: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        y = self.y
        sd1 = _robust_sd(np.diff(y)) if y.size >= 2 else 0.0
        sd2 = _robust_sd(np.diff(y, n=2)) if y.size >= 3 else 0.0
        sdg = sd1

        def _cal(pc: PCPrior, sd: float) -> float:
            u = max(1e-12, pc.frac * max(1e-12, sd))
            return float(-math.log(max(1e-12, pc.alpha_prob)) / u)

        # presence depends on whether that coord exists in the block
        if target == "mu":
            la = (pri.pc_alpha.lambda_s if pri.pc_alpha.lambda_s is not None else (_cal(pri.pc_alpha, sd1) if self.idx_alpha_mu is not None else None))
            lb = (pri.pc_beta.lambda_s  if pri.pc_beta.lambda_s  is not None else (_cal(pri.pc_beta,  sd2) if self.idx_beta_mu  is not None else None))
            lg = (pri.pc_gamma.lambda_s if pri.pc_gamma.lambda_s is not None else (_cal(pri.pc_gamma, sdg) if self.seasonal_mode_mu == "dynamic" else None))
        else:
            la = (pri.pc_alpha.lambda_s if pri.pc_alpha.lambda_s is not None else (_cal(pri.pc_alpha, sd1) if self.idx_alpha_sig is not None else None))
            lb = (pri.pc_beta.lambda_s  if pri.pc_beta.lambda_s  is not None else (_cal(pri.pc_beta,  sd2) if self.idx_beta_sig  is not None else None))
            lg = (pri.pc_gamma.lambda_s if pri.pc_gamma.lambda_s is not None else (_cal(pri.pc_gamma, sdg) if self.seasonal_mode_sig == "dynamic" else None))
        return (None if la is None else float(la),
                None if lb is None else float(lb),
                None if lg is None else float(lg))

    # ----------------------------- Matrices per block ----------------------------- #
    def _H_mu(self) -> np.ndarray:
        if self.dim_mu == 0: return np.zeros((1,0))
        h = np.zeros(self.dim_mu)
        if self.idx_alpha_mu is not None: h[self.idx_alpha_mu] = 1.0
        if self.seasonal_mode_mu == "dynamic": h[self.idx_gmu_start] = 1.0
        return h.reshape(1,-1)

    def _H_sig(self) -> np.ndarray:
        if self.dim_sig == 0: return np.zeros((1,0))
        h = np.zeros(self.dim_sig)
        if self.idx_alpha_sig is not None: h[self.idx_alpha_sig] = 1.0
        if self.seasonal_mode_sig == "dynamic": h[self.idx_gsig_start] = 1.0
        return h.reshape(1,-1)

    def _A_block(self, target: str) -> np.ndarray:
        if target == "mu":
            dim = self.dim_mu
            if dim == 0: return np.zeros((0,0))
            A = np.eye(dim)
            if self.idx_alpha_mu is not None and self.idx_beta_mu is not None:
                A[self.idx_alpha_mu, self.idx_beta_mu] = 1.0
            if self.seasonal_mode_mu == "dynamic":
                gs, ge = self.idx_gmu_start, self.idx_gmu_end
                K = ge - gs + 1
                A[gs, gs:ge+1] = -1.0
                A[gs+1:ge+1, gs:ge] = np.eye(K-1); A[gs+1:ge+1, ge] = 0.0
            return A
        else:
            dim = self.dim_sig
            if dim == 0: return np.zeros((0,0))
            A = np.eye(dim)
            if self.idx_alpha_sig is not None and self.idx_beta_sig is not None:
                A[self.idx_alpha_sig, self.idx_beta_sig] = 1.0
            if self.seasonal_mode_sig == "dynamic":
                gs, ge = self.idx_gsig_start, self.idx_gsig_end
                K = ge - gs + 1
                A[gs, gs:ge+1] = -1.0
                A[gs+1:ge+1, gs:ge] = np.eye(K-1); A[gs+1:ge+1, ge] = 0.0
            return A

    def _u_block(self, target: str) -> np.ndarray:
        if target == "mu":
            if self.dim_mu == 0: return np.zeros(0)
            u = np.zeros(self.dim_mu)
            if (self.idx_alpha_mu is not None) and (self.trend_mode_mu == "deterministic"):
                u[self.idx_alpha_mu] = float(self.m0_beta_mu)
            return u
        else:
            if self.dim_sig == 0: return np.zeros(0)
            u = np.zeros(self.dim_sig)
            if (self.idx_alpha_sig is not None) and (self.trend_mode_sig == "deterministic"):
                u[self.idx_alpha_sig] = float(self.m0_beta_sig)
            return u

    def _Q_block(self, target: str) -> np.ndarray:
        if target == "mu":
            if self.dim_mu == 0: return np.zeros((0,0))
            Q = np.zeros((self.dim_mu, self.dim_mu))
            if self.idx_alpha_mu is not None and self.z_alpha_mu and self.s_alpha_mu > 0:
                Q[self.idx_alpha_mu, self.idx_alpha_mu] = self.s_alpha_mu**2
            if self.idx_beta_mu  is not None and self.z_beta_mu  and self.s_beta_mu  > 0:
                Q[self.idx_beta_mu,  self.idx_beta_mu]  = self.s_beta_mu**2
            if self.seasonal_mode_mu == "dynamic" and self.z_gamma_mu and self.s_gamma_mu > 0:
                Q[self.idx_gmu_start, self.idx_gmu_start] = self.s_gamma_mu**2
            return Q
        else:
            if self.dim_sig == 0: return np.zeros((0,0))
            Q = np.zeros((self.dim_sig, self.dim_sig))
            if self.idx_alpha_sig is not None and self.z_alpha_sig and self.s_alpha_sig > 0:
                Q[self.idx_alpha_sig, self.idx_alpha_sig] = self.s_alpha_sig**2
            if self.idx_beta_sig  is not None and self.z_beta_sig  and self.s_beta_sig  > 0:
                Q[self.idx_beta_sig,  self.idx_beta_sig]  = self.s_beta_sig**2
            if self.seasonal_mode_sig == "dynamic" and self.z_gamma_sig and self.s_gamma_sig > 0:
                Q[self.idx_gsig_start, self.idx_gsig_start] = self.s_gamma_sig**2
            return Q

    # ------------------ Deterministic parts μ_t and η_t ------------------ #
    def _mu_det(self, t: int) -> float:
        out = 0.0
        if self.level_mode_mu == "deterministic": out += self.m0_alpha_mu
        if (self.trend_mode_mu == "deterministic") and (self.idx_alpha_mu is None):
            out += self.m0_beta_mu * t
        if self.seasonal_mode_mu == "deterministic":
            out += float(self.m0_gamma_mu[t % self.period])
        return out

    def _eta_det(self, t: int) -> float:
        out = 0.0
        if self.level_mode_sig == "deterministic": out += self.m0_alpha_sig
        if (self.trend_mode_sig == "deterministic") and (self.idx_alpha_sig is None):
            out += self.m0_beta_sig * t
        if self.seasonal_mode_sig == "deterministic":
            out += float(self.m0_gamma_sig[t % self.period])
        return out

    def _current_m0_P0_block(self, target: str) -> Tuple[np.ndarray, np.ndarray]:
        if target == "mu":
            m0, P0 = [], []
            if self.idx_alpha_mu is not None: m0.append(self.m0_alpha_mu); P0.append(self.P0_alpha_mu)
            if self.idx_beta_mu  is not None: m0.append(self.m0_beta_mu ); P0.append(self.P0_beta_mu)
            if self.seasonal_mode_mu == "dynamic":
                m0.extend(list(self.m0_gamma_mu)); P0.extend([self.P0_gamma_mu]*(self.period-1))
            return np.asarray(m0,float), np.asarray(P0,float)
        else:
            m0, P0 = [], []
            if self.idx_alpha_sig is not None: m0.append(self.m0_alpha_sig); P0.append(self.P0_alpha_sig)
            if self.idx_beta_sig  is not None: m0.append(self.m0_beta_sig ); P0.append(self.P0_beta_sig)
            if self.seasonal_mode_sig == "dynamic":
                m0.extend(list(self.m0_gamma_sig)); P0.extend([self.P0_gamma_sig]*(self.period-1))
            return np.asarray(m0,float), np.asarray(P0,float)

    # ------------------------- FFBS for μ with R_t ------------------------- #
    def _ffbs_mu_given_eta(self, R_t: np.ndarray) -> np.ndarray:
        if self.dim_mu == 0:
            # still return shape (T+1,0)
            return np.zeros_like(self.x_mu)
        H, A, Q = self._H_mu(), self._A_block("mu"), self._Q_block("mu")
        m0, P0 = self._current_m0_P0_block("mu")
        m = np.zeros((self.T+1, self.dim_mu))
        C = np.zeros((self.T+1, self.dim_mu, self.dim_mu))
        a = np.zeros((self.T+1, self.dim_mu))
        Rm= np.zeros((self.T+1, self.dim_mu, self.dim_mu))
        m[0]= m0; C[0]= np.diag(P0) + 1e-12*np.eye(self.dim_mu)
        u = self._u_block("mu")

        # forward
        for t in range(1, self.T+1):
            a[t]  = A @ m[t-1] + u
            Rm[t] = A @ C[t-1] @ A.T + Q
            Rm[t] = 0.5*(Rm[t]+Rm[t].T) + 1e-12*np.eye(self.dim_mu)

            resid_mean = float(self.y[t-1] - self._mu_det(t-1))
            Robs = float(R_t[t-1])
            S = float(H @ Rm[t] @ H.T + Robs)
            if S <= 0: S = float(H @ (Rm[t]+1e-10*np.eye(self.dim_mu)) @ H.T + Robs)
            K = (Rm[t] @ H.T) / S
            v = resid_mean - float(H @ a[t])
            m[t] = a[t] + (K.flatten()*v)
            C[t] = Rm[t] - K @ (H @ Rm[t])
            C[t] = 0.5*(C[t]+C[t].T) + 1e-12*np.eye(self.dim_mu)

        # backward
        x = np.zeros_like(self.x_mu)
        x[self.T] = np.random.multivariate_normal(m[self.T], C[self.T])
        for t in range(self.T-1, -1, -1):
            J = C[t] @ A.T
            J = J @ _spd_solve(Rm[t+1], np.eye(self.dim_mu))
            mean = m[t] + J @ (x[t+1] - (A @ m[t] + u))
            cov  = C[t] - J @ Rm[t+1] @ J.T
            cov  = 0.5*(cov+cov.T)
            cov += max(0.0, 1e-12 - float(np.linalg.eigvalsh(cov).min()))*np.eye(cov.shape[0])
            x[t] = np.random.multivariate_normal(mean, cov)
        return x

    # ------------------------- SV step for η (KSC) ------------------------- #
    def _sample_ksc_indicators(self, z: np.ndarray, eta_mean: np.ndarray) -> np.ndarray:
        # z_t = 0.5*log r_t^2 ; model: z_t = eta_t + m_j + N(0, v_j)
        # We compute posterior weights for s_t ∈ {1..7} and sample.
        J = _KSC7_p.size
        T = z.size
        s = np.zeros(T, dtype=int)
        for t in range(T):
            w = np.zeros(J)
            for j in range(J):
                mean = eta_mean[t] + _KSC7_mh[j]
                var  = _KSC7_vh[j]
                w[j] = _KSC7_p[j] * math.exp(-0.5*((z[t]-mean)**2)/var) / math.sqrt(2*math.pi*var)
            wsum = w.sum()
            if not np.isfinite(wsum) or wsum <= 0:
                w[:] = _KSC7_p; wsum = w.sum()
            w /= wsum
            s[t] = int(np.random.choice(J, p=w))
        return s

    def _ffbs_eta_given_s(self, z: np.ndarray, s_idx: np.ndarray) -> np.ndarray:
        # Build pseudo-obs: z_t = eta_t + eps_t,  eps_t ~ N(m_j, v_j)
        if self.dim_sig == 0:
            return np.zeros_like(self.x_sig)
        H, A, Q = self._H_sig(), self._A_block("sig"), self._Q_block("sig")
        m0, P0 = self._current_m0_P0_block("sig")
        m = np.zeros((self.T+1, self.dim_sig))
        C = np.zeros((self.T+1, self.dim_sig, self.dim_sig))
        a = np.zeros((self.T+1, self.dim_sig))
        Rm= np.zeros((self.T+1, self.dim_sig, self.dim_sig))
        m[0]= m0; C[0]= np.diag(P0) + 1e-12*np.eye(self.dim_sig)
        u = self._u_block("sig")

        # forward (heteroskedastic Gaussian with known R_t = v_{s_t})
        for t in range(1, self.T+1):
            a[t]  = A @ m[t-1] + u
            Rm[t] = A @ C[t-1] @ A.T + Q
            Rm[t] = 0.5*(Rm[t]+Rm[t].T) + 1e-12*np.eye(self.dim_sig)

            # pseudo observation y* = z_t - m_{s_t}; variance v_{s_t}
            ystar = float(z[t-1] - _KSC7_mh[s_idx[t-1]])
            Robs  = float(_KSC7_vh[s_idx[t-1]])
            S = float(H @ Rm[t] @ H.T + Robs)
            if S <= 0: S = float(H @ (Rm[t]+1e-10*np.eye(self.dim_sig)) @ H.T + Robs)
            K = (Rm[t] @ H.T) / S
            v = ystar - float(H @ a[t])
            m[t] = a[t] + (K.flatten()*v)
            C[t] = Rm[t] - K @ (H @ Rm[t])
            C[t] = 0.5*(C[t]+C[t].T) + 1e-12*np.eye(self.dim_sig)

        # backward
        x = np.zeros_like(self.x_sig)
        x[self.T] = np.random.multivariate_normal(m[self.T], C[self.T])
        for t in range(self.T-1, -1, -1):
            J = C[t] @ A.T
            J = J @ _spd_solve(Rm[t+1], np.eye(self.dim_sig))
            mean = m[t] + J @ (x[t+1] - (A @ m[t] + u))
            cov  = C[t] - J @ Rm[t+1] @ J.T
            cov  = 0.5*(cov+cov.T)
            cov += max(0.0, 1e-12 - float(np.linalg.eigvalsh(cov).min()))*np.eye(cov.shape[0])
            x[t] = np.random.multivariate_normal(mean, cov)
        return x

    # -------------------- Sufficient stats for process sds -------------------- #
    def _innovation_ss_level(self, x: np.ndarray, idx_level: int, idx_trend: Optional[int],
                             trend_mode: str, m0_beta: float) -> Tuple[float,int]:
        if idx_level is None: return 0.0, 0
        ss = 0.0
        for t in range(1, self.T+1):
            drift = 0.0
            if idx_trend is not None: drift = x[t-1, idx_trend]
            elif trend_mode == "deterministic": drift = float(m0_beta)
            mean = x[t-1, idx_level] + drift
            ss += (x[t, idx_level] - mean)**2
        return float(ss), self.T

    def _innovation_ss_trend(self, x: np.ndarray, idx_trend: Optional[int]) -> Tuple[float,int]:
        if idx_trend is None: return 0.0, 0
        d = x[1:, idx_trend] - x[:-1, idx_trend]
        return float(np.sum(d*d)), self.T

    def _innovation_ss_season_first(self, x: np.ndarray, gs: Optional[int], ge: Optional[int]) -> Tuple[float,int]:
        if gs is None: return 0.0, 0
        ss = 0.0
        for t in range(1, self.T+1):
            prev = x[t-1, gs:ge+1]
            mean_new_first = -float(np.sum(prev))
            ss += (x[t, gs] - mean_new_first)**2
        return float(ss), self.T

    # -------------------- Slice for log s with PC prior -------------------- #
    def _slice(self, f: Callable[[float], float], z0: float) -> float:
        w, m, limit = float(self.cfg.slice_w), int(self.cfg.slice_m), int(self.cfg.slice_max_shrink)
        y_star = f(z0) - np.random.exponential(1.0)
        u = np.random.rand()
        L = z0 - u*w; R = L + w
        j = int(np.floor(m*np.random.rand())); k = (m-1) - j
        while j>0 and f(L)>y_star: L -= w; j -= 1
        while k>0 and f(R)>y_star: R += w; k -= 1
        for _ in range(limit):
            z_prop = np.random.uniform(L,R)
            if f(z_prop) >= y_star:
                return z_prop
            if z_prop < z0: L = z_prop
            else: R = z_prop
        return z0

    def _slice_logsd(self, z0: float, SS: float, T_eff: int, lam: float) -> float:
        def f(z: float) -> float:
            return -(T_eff * z) - 0.5 * SS * math.exp(-2*z) - lam * math.exp(z) + z
        return self._slice(f, z0)

    # -------------------- Update process sds + λ (both blocks) -------------------- #
    def _update_process_sds_block(self, target: str) -> None:
        if target == "mu":
            # level
            if self.idx_alpha_mu is not None and self.z_alpha_mu and (self.lambda_alpha_mu is not None):
                ss, Te = self._innovation_ss_level(self.x_mu, self.idx_alpha_mu, self.idx_beta_mu,
                                                   self.trend_mode_mu, self.m0_beta_mu)
                z = self._slice_logsd(math.log(max(1e-18, self.s_alpha_mu)), ss, Te, float(self.lambda_alpha_mu))
                self.s_alpha_mu = float(math.exp(z))
            elif self.idx_alpha_mu is not None and not self.z_alpha_mu:
                self.s_alpha_mu = 0.0
            # trend
            if self.idx_beta_mu is not None and self.z_beta_mu and (self.lambda_beta_mu is not None):
                ss, Te = self._innovation_ss_trend(self.x_mu, self.idx_beta_mu)
                z = self._slice_logsd(math.log(max(1e-18, self.s_beta_mu)), ss, Te, float(self.lambda_beta_mu))
                self.s_beta_mu = float(math.exp(z))
            elif self.idx_beta_mu is not None and not self.z_beta_mu:
                self.s_beta_mu = 0.0
            # season (first coord)
            if (self.seasonal_mode_mu == "dynamic") and self.z_gamma_mu and (self.lambda_gamma_mu is not None):
                ss, Te = self._innovation_ss_season_first(self.x_mu, self.idx_gmu_start, self.idx_gmu_end)
                z = self._slice_logsd(math.log(max(1e-18, self.s_gamma_mu)), ss, Te, float(self.lambda_gamma_mu))
                self.s_gamma_mu = float(math.exp(z))
            elif self.seasonal_mode_mu == "dynamic" and not self.z_gamma_mu:
                self.s_gamma_mu = 0.0

        else:  # sig
            if self.idx_alpha_sig is not None and self.z_alpha_sig and (self.lambda_alpha_sig is not None):
                ss, Te = self._innovation_ss_level(self.x_sig, self.idx_alpha_sig, self.idx_beta_sig,
                                                   self.trend_mode_sig, self.m0_beta_sig)
                z = self._slice_logsd(math.log(max(1e-18, self.s_alpha_sig)), ss, Te, float(self.lambda_alpha_sig))
                self.s_alpha_sig = float(math.exp(z))
            elif self.idx_alpha_sig is not None and not self.z_alpha_sig:
                self.s_alpha_sig = 0.0
            if self.idx_beta_sig is not None and self.z_beta_sig and (self.lambda_beta_sig is not None):
                ss, Te = self._innovation_ss_trend(self.x_sig, self.idx_beta_sig)
                z = self._slice_logsd(math.log(max(1e-18, self.s_beta_sig)), ss, Te, float(self.lambda_beta_sig))
                self.s_beta_sig = float(math.exp(z))
            elif self.idx_beta_sig is not None and not self.z_beta_sig:
                self.s_beta_sig = 0.0
            if (self.seasonal_mode_sig == "dynamic") and self.z_gamma_sig and (self.lambda_gamma_sig is not None):
                ss, Te = self._innovation_ss_season_first(self.x_sig, self.idx_gsig_start, self.idx_gsig_end)
                z = self._slice_logsd(math.log(max(1e-18, self.s_gamma_sig)), ss, Te, float(self.lambda_gamma_sig))
                self.s_gamma_sig = float(math.exp(z))
            elif self.seasonal_mode_sig == "dynamic" and not self.z_gamma_sig:
                self.s_gamma_sig = 0.0

    def _gibbs_lambda_single(self, which: str) -> None:
        # which in {"alpha_mu","beta_mu","gamma_mu","alpha_sig","beta_sig","gamma_sig"}
        if which.endswith("_mu"):
            pri = self.priors.mu
            if which == "alpha_mu" and pri.pc_alpha.lambda_s is None and self.idx_alpha_mu is not None:
                self.lambda_alpha_mu = float(np.random.gamma(pri.pc_alpha.a_lambda + 1.0,
                                                             1.0 / (pri.pc_alpha.b_lambda + max(0.0, self.s_alpha_mu))))
            if which == "beta_mu" and pri.pc_beta.lambda_s is None and self.idx_beta_mu is not None:
                self.lambda_beta_mu  = float(np.random.gamma(pri.pc_beta.a_lambda + 1.0,
                                                             1.0 / (pri.pc_beta.b_lambda + max(0.0, self.s_beta_mu))))
            if which == "gamma_mu" and pri.pc_gamma.lambda_s is None and self.seasonal_mode_mu == "dynamic":
                self.lambda_gamma_mu = float(np.random.gamma(pri.pc_gamma.a_lambda + 1.0,
                                                             1.0 / (pri.pc_gamma.b_lambda + max(0.0, self.s_gamma_mu))))
        else:
            pri = self.priors.sig
            if which == "alpha_sig" and pri.pc_alpha.lambda_s is None and self.idx_alpha_sig is not None:
                self.lambda_alpha_sig = float(np.random.gamma(pri.pc_alpha.a_lambda + 1.0,
                                                              1.0 / (pri.pc_alpha.b_lambda + max(0.0, self.s_alpha_sig))))
            if which == "beta_sig" and pri.pc_beta.lambda_s is None and self.idx_beta_sig is not None:
                self.lambda_beta_sig  = float(np.random.gamma(pri.pc_beta.a_lambda + 1.0,
                                                              1.0 / (pri.pc_beta.b_lambda + max(0.0, self.s_beta_sig))))
            if which == "gamma_sig" and pri.pc_gamma.lambda_s is None and self.seasonal_mode_sig == "dynamic":
                self.lambda_gamma_sig = float(np.random.gamma(pri.pc_gamma.a_lambda + 1.0,
                                                              1.0 / (pri.pc_gamma.b_lambda + max(0.0, self.s_gamma_sig))))

    def update_pc_lambdas(self) -> None:
        for w in ["alpha_mu","beta_mu","gamma_mu","alpha_sig","beta_sig","gamma_sig"]:
            self._gibbs_lambda_single(w)

    # ---------------- Deterministic params (conjugate) per block ---------------- #
    @staticmethod
    def _gibbs_m0_scalar(x0: float, m_prior: float, s_prior: float, P0: float) -> float:
        prec = 1.0/(s_prior**2) + 1.0/max(1e-18, P0)
        var  = 1.0/prec
        mean = var*(m_prior/(s_prior**2) + x0/max(1e-18, P0))
        return float(np.random.normal(mean, math.sqrt(var)))

    def update_m0_P0_block(self, target: str) -> None:
        pri = self.priors.mu if target=="mu" else self.priors.sig
        if target == "mu" and self.dim_mu>0:
            pos = 0
            if self.idx_alpha_mu is not None:
                self.m0_alpha_mu = self._gibbs_m0_scalar(float(self.x_mu[0,pos]), pri.m_m0_alpha, pri.s_m0_alpha, self.P0_alpha_mu); pos+=1
            if self.idx_beta_mu is not None:
                self.m0_beta_mu  = self._gibbs_m0_scalar(float(self.x_mu[0,pos]), pri.m_m0_beta,  pri.s_m0_beta,  self.P0_beta_mu);  pos+=1
            if self.seasonal_mode_mu == "dynamic":
                m_prior = np.zeros(self.period-1) if pri.m_m0_gamma is None else np.asarray(pri.m_m0_gamma,float)
                if m_prior.size != self.period-1: raise ValueError("priors.mu.m_m0_gamma length p-1")
                s = float(pri.s_m0_gamma)
                for k in range(self.period-1):
                    self.m0_gamma_mu[k] = self._gibbs_m0_scalar(float(self.x_mu[0,pos+k]), float(m_prior[k]), s, self.P0_gamma_mu)

            # P0 updates
            pos = 0
            if self.idx_alpha_mu is not None:
                a = pri.a_P0_alpha + 0.5
                b = pri.b_P0_alpha + 0.5*(float(self.x_mu[0,pos])-self.m0_alpha_mu)**2
                self.P0_alpha_mu = 1.0/np.random.gamma(a, 1.0/b); pos+=1
            if self.idx_beta_mu is not None:
                a = pri.a_P0_beta + 0.5
                b = pri.b_P0_beta + 0.5*(float(self.x_mu[0,pos])-self.m0_beta_mu)**2
                self.P0_beta_mu = 1.0/np.random.gamma(a, 1.0/b); pos+=1
            if self.seasonal_mode_mu == "dynamic":
                diffsq = 0.0
                for k in range(self.period-1):
                    diffsq += (float(self.x_mu[0,pos+k])-float(self.m0_gamma_mu[k]))**2
                a = pri.a_P0_gamma + 0.5*(self.period-1)
                b = pri.b_P0_gamma + 0.5*diffsq
                self.P0_gamma_mu = 1.0/np.random.gamma(a, 1.0/b)

        if target == "sig" and self.dim_sig>0:
            pos = 0
            if self.idx_alpha_sig is not None:
                self.m0_alpha_sig = self._gibbs_m0_scalar(float(self.x_sig[0,pos]), pri.m_m0_alpha, pri.s_m0_alpha, self.P0_alpha_sig); pos+=1
            if self.idx_beta_sig is not None:
                self.m0_beta_sig  = self._gibbs_m0_scalar(float(self.x_sig[0,pos]), pri.m_m0_beta,  pri.s_m0_beta,  self.P0_beta_sig);  pos+=1
            if self.seasonal_mode_sig == "dynamic":
                m_prior = np.zeros(self.period-1) if pri.m_m0_gamma is None else np.asarray(pri.m_m0_gamma,float)
                if m_prior.size != self.period-1: raise ValueError("priors.sig.m_m0_gamma length p-1")
                s = float(pri.s_m0_gamma)
                for k in range(self.period-1):
                    self.m0_gamma_sig[k] = self._gibbs_m0_scalar(float(self.x_sig[0,pos+k]), float(m_prior[k]), s, self.P0_gamma_sig)

            # P0 updates
            pos = 0
            if self.idx_alpha_sig is not None:
                a = pri.a_P0_alpha + 0.5
                b = pri.b_P0_alpha + 0.5*(float(self.x_sig[0,pos])-self.m0_alpha_sig)**2
                self.P0_alpha_sig = 1.0/np.random.gamma(a, 1.0/b); pos+=1
            if self.idx_beta_sig is not None:
                a = pri.a_P0_beta + 0.5
                b = pri.b_P0_beta + 0.5*(float(self.x_sig[0,pos])-self.m0_beta_sig)**2
                self.P0_beta_sig = 1.0/np.random.gamma(a, 1.0/b); pos+=1
            if self.seasonal_mode_sig == "dynamic":
                diffsq = 0.0
                for k in range(self.period-1):
                    diffsq += (float(self.x_sig[0,pos+k])-float(self.m0_gamma_sig[k]))**2
                a = pri.a_P0_gamma + 0.5*(self.period-1)
                b = pri.b_P0_gamma + 0.5*diffsq
                self.P0_gamma_sig = 1.0/np.random.gamma(a, 1.0/b)

    # ---------------- Deterministische params uit residuen ---------------- #
    def _update_deterministic_mu(self, R_t: np.ndarray) -> None:
        # r = y - dyn_mu; weighted by R_t
        r = self.y.copy()
        if self.dim_mu > 0:
            H = self._H_mu()
            for t in range(1, self.T+1):
                r[t-1] -= float(H @ self.x_mu[t])
        w = 1.0/np.asarray(R_t, float)  # precision weights

        # level (if deterministic)
        if self.level_mode_mu == "deterministic":
            rr = r.copy()
            if (self.trend_mode_mu == "deterministic") and (self.idx_alpha_mu is None):
                t = np.arange(self.T, dtype=float); rr -= self.m0_beta_mu * t
            if self.seasonal_mode_mu == "deterministic":
                rr -= self.m0_gamma_mu[np.arange(self.T)%self.period]
            s2 = float(self.priors.mu.s_m0_alpha)**2
            prec = float(np.sum(w)) + 1.0/s2
            mean = (float(np.sum(w*rr)) + self.priors.mu.m_m0_alpha/s2)/prec
            var  = 1.0/prec
            self.m0_alpha_mu = float(np.random.normal(mean, math.sqrt(var)))

        # slope (if deterministic and α not dynamic)
        if self.trend_mode_mu == "deterministic" and self.idx_alpha_mu is None:
            rr = r.copy()
            if self.level_mode_mu == "deterministic":
                rr -= self.m0_alpha_mu
            if self.seasonal_mode_mu == "deterministic":
                rr -= self.m0_gamma_mu[np.arange(self.T)%self.period]
            t = np.arange(self.T, dtype=float)
            s2 = float(self.priors.mu.s_m0_beta)**2
            prec = float(np.sum(w*(t*t))) + 1.0/s2
            mean = (float(np.sum(w*t*rr)) + self.priors.mu.m_m0_beta/s2)/prec
            var  = 1.0/prec
            self.m0_beta_mu = float(np.random.normal(mean, math.sqrt(var)))

        # season (deterministic contrasts)
        if self.seasonal_mode_mu == "deterministic":
            midx = np.arange(self.T)%self.period
            K = self.period-1
            Z = np.zeros((self.T, K))
            for k in range(K):
                Z[:,k] = (midx==k).astype(float) - (midx==K).astype(float)
            rr = r.copy()
            if self.level_mode_mu == "deterministic":
                rr -= self.m0_alpha_mu
            if (self.trend_mode_mu == "deterministic") and (self.idx_alpha_mu is None):
                rr -= self.m0_beta_mu * np.arange(self.T, dtype=float)
            # weighted ridge
            sig2 = 1.0  # absorbed in w
            S = (Z.T * w) @ Z
            b = (Z.T * w) @ rr
            s2 = float(self.priors.mu.s_m0_gamma)**2
            Prec = S + np.eye(K)/s2
            mu = np.linalg.solve(Prec, b + (np.zeros(K) if self.priors.mu.m_m0_gamma is None
                                            else np.asarray(self.priors.mu.m_m0_gamma,float)/s2))
            L = np.linalg.cholesky(Prec)
            theta = mu + np.linalg.solve(L.T, np.random.randn(K))
            self.m0_gamma_mu = np.r_[theta, -theta.sum()]

    def _update_deterministic_eta(self, z: np.ndarray, s_idx: np.ndarray) -> None:
        # SV pseudo-obs: y*_t = z_t - m_{s_t} with variance v_{s_t}
        ystar = z - _KSC7_mh[s_idx]
        w = 1.0/_KSC7_vh[s_idx]  # precision weights

        # remove dynamic part
        if self.dim_sig > 0:
            H = self._H_sig()
            for t in range(1, self.T+1):
                ystar[t-1] -= float(H @ self.x_sig[t])

        # level
        if self.level_mode_sig == "deterministic":
            rr = ystar.copy()
            if (self.trend_mode_sig == "deterministic") and (self.idx_alpha_sig is None):
                t = np.arange(self.T, dtype=float); rr -= self.m0_beta_sig * t
            if self.seasonal_mode_sig == "deterministic":
                rr -= self.m0_gamma_sig[np.arange(self.T)%self.period]
            s2 = float(self.priors.sig.s_m0_alpha)**2
            prec = float(np.sum(w)) + 1.0/s2
            mean = (float(np.sum(w*rr)) + self.priors.sig.m_m0_alpha/s2)/prec
            var  = 1.0/prec
            self.m0_alpha_sig = float(np.random.normal(mean, math.sqrt(var)))

        # slope
        if self.trend_mode_sig == "deterministic" and self.idx_alpha_sig is None:
            rr = ystar.copy()
            if self.level_mode_sig == "deterministic":
                rr -= self.m0_alpha_sig
            if self.seasonal_mode_sig == "deterministic":
                rr -= self.m0_gamma_sig[np.arange(self.T)%self.period]
            t = np.arange(self.T, dtype=float)
            s2 = float(self.priors.sig.s_m0_beta)**2
            prec = float(np.sum(w*(t*t))) + 1.0/s2
            mean = (float(np.sum(w*t*rr)) + self.priors.sig.m_m0_beta/s2)/prec
            var  = 1.0/prec
            self.m0_beta_sig = float(np.random.normal(mean, math.sqrt(var)))

        # season
        if self.seasonal_mode_sig == "deterministic":
            midx = np.arange(self.T)%self.period
            K = self.period-1
            Z = np.zeros((self.T, K))
            for k in range(K):
                Z[:,k] = (midx==k).astype(float) - (midx==K).astype(float)
            rr = ystar.copy()
            if self.level_mode_sig == "deterministic":
                rr -= self.m0_alpha_sig
            if (self.trend_mode_sig == "deterministic") and (self.idx_alpha_sig is None):
                rr -= self.m0_beta_sig * np.arange(self.T, dtype=float)
            S = (Z.T * w) @ Z
            b = (Z.T * w) @ rr
            s2 = float(self.priors.sig.s_m0_gamma)**2
            Prec = S + np.eye(K)/s2
            mu = np.linalg.solve(Prec, b + (np.zeros(K) if self.priors.sig.m_m0_gamma is None
                                            else np.asarray(self.priors.sig.m_m0_gamma,float)/s2))
            L = np.linalg.cholesky(Prec)
            theta = mu + np.linalg.solve(L.T, np.random.randn(K))
            self.m0_gamma_sig = np.r_[theta, -theta.sum()]

    # ---------------- Helper: paths to μ_t and η_t ---------------- #
    def _mu_vec(self) -> np.ndarray:
        H = self._H_mu()
        mu = np.zeros(self.T)
        for t in range(1, self.T+1):
            dyn = float(H @ self.x_mu[t]) if self.dim_mu>0 else 0.0
            mu[t-1] = self._mu_det(t-1) + dyn
        return mu

    def _eta_vec(self) -> np.ndarray:
        H = self._H_sig()
        eta = np.zeros(self.T)
        for t in range(1, self.T+1):
            dyn = float(H @ self.x_sig[t]) if self.dim_sig>0 else 0.0
            eta[t-1] = self._eta_det(t-1) + dyn
        return eta

    # ------------------------------- MCMC ------------------------------- #
    def run(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        save_iters = list(range(cfg.burn, cfg.n_iter, cfg.thin))
        n_kept, keep_idx = len(save_iters), 0

        # allocate
        self.keep = {
            "mu":  np.zeros((n_kept, self.T)),
            "eta": np.zeros((n_kept, self.T)),
            "sigma": np.zeros((n_kept, self.T)),  # time-varying σ_t
        }
        # keep variances/process + states
        if self.dim_mu>0:
            self.keep.update({
                "Q_alpha_mu": np.zeros(n_kept),
                "Q_beta_mu":  np.zeros(n_kept),
                "Q_gamma_mu": np.zeros(n_kept),
                "x_mu": np.zeros((n_kept, self.T, self.dim_mu)),
                "m0_alpha_mu": np.zeros(n_kept),
                "m0_beta_mu":  np.zeros(n_kept),
                "P0_alpha_mu": np.zeros(n_kept),
                "P0_beta_mu":  np.zeros(n_kept),
                "P0_gamma_mu": np.zeros(n_kept),
                "lambda_alpha_mu": np.zeros(n_kept),
                "lambda_beta_mu":  np.zeros(n_kept),
                "lambda_gamma_mu": np.zeros(n_kept),
            })
            if self.seasonal_mode_mu == "dynamic":
                self.keep["m0_gamma_mu"] = np.zeros((n_kept, self.period-1))
            if self.spike.enabled:
                self.keep.update({"z_alpha_mu": np.zeros(n_kept, dtype=int),
                                  "z_beta_mu":  np.zeros(n_kept, dtype=int),
                                  "z_gamma_mu": np.zeros(n_kept, dtype=int)})
        if self.dim_sig>0:
            self.keep.update({
                "Q_alpha_sig": np.zeros(n_kept),
                "Q_beta_sig":  np.zeros(n_kept),
                "Q_gamma_sig": np.zeros(n_kept),
                "x_sig": np.zeros((n_kept, self.T, self.dim_sig)),
                "m0_alpha_sig": np.zeros(n_kept),
                "m0_beta_sig":  np.zeros(n_kept),
                "P0_alpha_sig": np.zeros(n_kept),
                "P0_beta_sig":  np.zeros(n_kept),
                "P0_gamma_sig": np.zeros(n_kept),
                "lambda_alpha_sig": np.zeros(n_kept),
                "lambda_beta_sig":  np.zeros(n_kept),
                "lambda_gamma_sig": np.zeros(n_kept),
            })
            if self.seasonal_mode_sig == "dynamic":
                self.keep["m0_gamma_sig"] = np.zeros((n_kept, self.period-1))
            if self.spike.enabled:
                self.keep.update({"z_alpha_sig": np.zeros(n_kept, dtype=int),
                                  "z_beta_sig":  np.zeros(n_kept, dtype=int),
                                  "z_gamma_sig": np.zeros(n_kept, dtype=int)})

        print_every = cfg.progress_every if cfg.progress_every>0 else max(1, cfg.n_iter//50) or 1

        for it in range(cfg.n_iter):
            # ---------- 0) Build current η_t and R_t ----------
            eta_t = self._eta_vec()
            R_t = np.exp(2.0 * eta_t)

            # ---------- 1) μ-block: FFBS given R_t ----------
            if self.dim_mu>0:
                self.x_mu = self._ffbs_mu_given_eta(R_t)

            # ---------- 2) η-block: SV (KSC) ----------
            # residuals r_t using updated μ
            mu_t = self._mu_vec()
            r = self.y - mu_t
            # stabilize tiny values to avoid log(0)
            r2 = np.maximum(r*r, 1e-12)
            z  = 0.5*np.log(r2)  # half-log squares
            # “prior mean” of eta for indicator sampling = current eta_t
            eta_t = self._eta_vec()
            s_idx = self._sample_ksc_indicators(z, eta_t)
            if self.dim_sig>0:
                self.x_sig = self._ffbs_eta_given_s(z, s_idx)

            # ---------- 3) Process sds (slice) + 4) λ (Gibbs) ----------
            if self.dim_mu>0:  self._update_process_sds_block("mu")
            if self.dim_sig>0: self._update_process_sds_block("sig")
            self.update_pc_lambdas()

            # ---------- 5) m0 / P0 (Gibbs) ----------
            if self.dim_mu>0:  self.update_m0_P0_block("mu")
            if self.dim_sig>0: self.update_m0_P0_block("sig")

            # ---------- 6) Deterministic params ----------
            if (self.level_mode_mu=="deterministic" or
                self.trend_mode_mu=="deterministic" or
                self.seasonal_mode_mu=="deterministic"):
                self._update_deterministic_mu(R_t)

            if (self.level_mode_sig=="deterministic" or
                self.trend_mode_sig=="deterministic" or
                self.seasonal_mode_sig=="deterministic"):
                self._update_deterministic_eta(z, s_idx)

            # (Optional) Spike&slab flips — we keep them OFF by default here to keep run stable.
            # If you want them: replicate your collapsed MH for μ with hetero-KF; for η,
            # use SV pseudo-likelihood (like _ffbs_eta_given_s forward filter loglik).

            # ---------- Progress ----------
            if cfg.progress and ((it+1)%print_every==0 or it==cfg.n_iter-1):
                parts = [f"[it {it+1}/{cfg.n_iter}]",]
                parts.append(f"mean(Qμ)=({self.s_alpha_mu**2:.3g},{self.s_beta_mu**2:.3g},{self.s_gamma_mu**2:.3g})")
                parts.append(f"mean(Qη)=({self.s_alpha_sig**2:.3g},{self.s_beta_sig**2:.3g},{self.s_gamma_sig**2:.3g})")
                parts.append(f"η̄={eta_t.mean():.3f}, σ̄={np.exp(eta_t).mean():.3f}")
                print(" | ".join(parts))

            # ---------- Save ----------
            if it in save_iters:
                mu_t = self._mu_vec()
                eta_t = self._eta_vec()
                self.keep["mu"][keep_idx,:]   = mu_t
                self.keep["eta"][keep_idx,:]  = eta_t
                self.keep["sigma"][keep_idx,:]= np.exp(eta_t)
                if self.dim_mu>0:
                    self.keep["x_mu"][keep_idx,:,:] = self.x_mu[1:self.T+1,:]
                    self.keep["Q_alpha_mu"][keep_idx] = self.s_alpha_mu**2
                    self.keep["Q_beta_mu"][keep_idx]  = self.s_beta_mu**2
                    self.keep["Q_gamma_mu"][keep_idx] = self.s_gamma_mu**2
                    self.keep["m0_alpha_mu"][keep_idx]= self.m0_alpha_mu
                    self.keep["m0_beta_mu"][keep_idx] = self.m0_beta_mu
                    self.keep["P0_alpha_mu"][keep_idx]= self.P0_alpha_mu
                    self.keep["P0_beta_mu"][keep_idx] = self.P0_beta_mu
                    self.keep["P0_gamma_mu"][keep_idx]= self.P0_gamma_mu
                    self.keep["lambda_alpha_mu"][keep_idx]= float(self.lambda_alpha_mu or 0.0)
                    self.keep["lambda_beta_mu"][keep_idx] = float(self.lambda_beta_mu  or 0.0)
                    self.keep["lambda_gamma_mu"][keep_idx]= float(self.lambda_gamma_mu or 0.0)
                    if "m0_gamma_mu" in self.keep:
                        self.keep["m0_gamma_mu"][keep_idx,:] = (self.m0_gamma_mu if self.seasonal_mode_mu=="dynamic"
                                                                else self.m0_gamma_mu[:-1])
                if self.dim_sig>0:
                    self.keep["x_sig"][keep_idx,:,:] = self.x_sig[1:self.T+1,:]
                    self.keep["Q_alpha_sig"][keep_idx] = self.s_alpha_sig**2
                    self.keep["Q_beta_sig"][keep_idx]  = self.s_beta_sig**2
                    self.keep["Q_gamma_sig"][keep_idx] = self.s_gamma_sig**2
                    self.keep["m0_alpha_sig"][keep_idx]= self.m0_alpha_sig
                    self.keep["m0_beta_sig"][keep_idx] = self.m0_beta_sig
                    self.keep["P0_alpha_sig"][keep_idx]= self.P0_alpha_sig
                    self.keep["P0_beta_sig"][keep_idx] = self.P0_beta_sig
                    self.keep["P0_gamma_sig"][keep_idx]= self.P0_gamma_sig
                    self.keep["lambda_alpha_sig"][keep_idx]= float(self.lambda_alpha_sig or 0.0)
                    self.keep["lambda_beta_sig"][keep_idx] = float(self.lambda_beta_sig  or 0.0)
                    self.keep["lambda_gamma_sig"][keep_idx]= float(self.lambda_gamma_sig or 0.0)
                    if "m0_gamma_sig" in self.keep:
                        self.keep["m0_gamma_sig"][keep_idx,:] = (self.m0_gamma_sig if self.seasonal_mode_sig=="dynamic"
                                                                else self.m0_gamma_sig[:-1])
                keep_idx += 1

        return self.keep

    # ------------------------------- Persistence ------------------------------ #
    def save_posterior(self, out_npz_path: str, extra_meta: Optional[dict]=None) -> None:
        _ensure_dir(os.path.dirname(out_npz_path))
        arrays = dict(self.keep)
        arrays["y"] = self.y.copy()
        if "x_mu" not in arrays:  arrays["x_mu"]  = np.zeros((0,0,0))
        if "x_sig" not in arrays: arrays["x_sig"] = np.zeros((0,0,0))
        np.savez_compressed(out_npz_path, **arrays)

        meta = {
            "T": int(self.T),
            "period": int(self.period),
            "modes": {
                "mu":  {"level": self.level_mode_mu, "trend": self.trend_mode_mu, "season": self.seasonal_mode_mu},
                "sig": {"level": self.level_mode_sig, "trend": self.trend_mode_sig, "season": self.seasonal_mode_sig},
            },
            "layout_mu": list(self._layout_mu),
            "layout_sig": list(self._layout_sig),
            "cfg": asdict(self.cfg),
            "priors": asdict(self.priors),
        }
        if extra_meta: meta.update(extra_meta)
        meta_path = out_npz_path.replace(".npz", ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[save] Posterior -> {out_npz_path}")
        print(f"[save] Metadata  -> {meta_path}")

    if __name__ == "__main__":
        pass