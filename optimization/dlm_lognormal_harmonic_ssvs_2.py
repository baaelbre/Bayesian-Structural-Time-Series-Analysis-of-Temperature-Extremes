from __future__ import annotations
"""
Harmonic Gaussian DLM with **non-centered SSVS** on u=ln s, **sticky inclusion** for δ,
**switch-anchoring** for deterministic <-> dynamic toggles, and **dimension-reduced FFBS**.

What changed vs your last version
---------------------------------
• Sticky SSVS: persistence-aware odds for δ (discourages flip-flopping).
• Switch anchoring: when a block switches 0→1, dynamic x(0) and P0 are warm-started from the
  deterministic params; when 1→0, we immediately absorb the smoothed dynamic path into deterministic
  m0’s via a one-step ridge update so μ stays continuous.
• One-iteration innovation guard: after β 0→1 we *omit β* from α-innovation SS once to avoid spuriously
  large Qα.
• u=ln s clamping: prevents pathological Q after mode switches.
• Update order adjusted: SSVS(δ,u) → handle switches → deterministic-parameter update → σ² collapse.
"""

import json, math, os, sys, time, warnings
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------- optional helpers (FFT maps) ----------
try:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(base_dir)
    from optimization.harmonic_helpers import (
        center_and_report_dummies_full,
        dummies_full_to_harmonics_fft,
        harmonics_to_dummies_full_fft,
    )
except Exception:
    def center_and_report_dummies_full(x, tol=1e-12):
        x = np.asarray(x, float); return x - x.mean()
    def dummies_full_to_harmonics_fft(x, K, use_nyquist):
        s = len(x); t = np.arange(s); omegas = 2*np.pi*np.arange(1, K+1)/s
        cos = [(2/s) * np.sum(x*np.cos(w*t)) for w in omegas]
        sin = [(2/s) * np.sum(x*np.sin(w*t)) for w in omegas]
        nyq = None
        if use_nyquist and s % 2 == 0:
            nyq = (1/s) * np.sum(x*((-1.0)**t))
        return np.array(cos), np.array(sin), nyq
    def harmonics_to_dummies_full_fft(s, cos_coefs, sin_coefs, use_nyquist, nyq_coef):
        t = np.arange(s); omegas = 2*np.pi*np.arange(1, len(cos_coefs)+1)/s
        out = np.zeros(s)
        for k, w in enumerate(omegas):
            out += cos_coefs[k]*np.cos(w*t) + sin_coefs[k]*np.sin(w*t)
        if use_nyquist:
            out += (0.0 if nyq_coef is None else nyq_coef)*((-1.0)**t)
        return out

# ===================== small utils =====================

def _mad(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    if v.size == 0: return 0.0
    m = np.median(v)
    return float(np.median(np.abs(v - m)))

def _robust_sd(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    return _mad(v) / 1.4826 if v.size else 0.0

def _spd_solve(M: np.ndarray, B: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    n = M.shape[0]; I = np.eye(n)
    for k in range(3):
        try:
            L = np.linalg.cholesky(M + (eps * (10**k)) * I)
            Y = np.linalg.solve(L, B)
            return np.linalg.solve(L.T, Y)
        except np.linalg.LinAlgError:
            continue
    return np.linalg.pinv(M) @ B

def _slice_sample(logpdf, z0: float, rng: np.random.Generator,
                  w: float = 1.0, m: int = 10, max_shrink: int = 1000) -> float:
    z0 = float(z0)
    logy = float(logpdf(z0)) - rng.exponential(1.0)
    u = rng.uniform(0.0, 1.0); L = z0 - u * w; R = L + w
    J = int(rng.integers(0, m + 1)) if m > 0 else 0
    K = (m - 1 - J) if m > 0 else 0
    while (J > 0) and (logpdf(L) > logy): L -= w; J -= 1
    while (K > 0) and (logpdf(R) > logy): R += w; K -= 1
    it = 0
    while it < max_shrink:
        z1 = rng.uniform(L, R)
        if logpdf(z1) >= logy: return float(z1)
        if z1 < z0: L = z1
        else: R = z1
        it += 1
    return float(z0)

# ===================== priors/config =====================

@dataclass
class SSVSLogScalePrior:
    m: float = -5.0   # mean for u = ln s
    v1: float = 4.0   # slab variance on u
    v0: float = 0.04  # spike variance on u (small)
    pi: float = 0.5   # prior inclusion prob

@dataclass
class Priors:
    # σ² prior (Gamma on precision)
    a_sigma: float = 2.5
    b_sigma: float = 2.5

    # Initial state means (Normals) and seasonal deterministic prior scale
    m_m0_alpha: float = 0.0; s_m0_alpha: float = 10.0
    m_m0_beta:  float = 0.0; s_m0_beta:  float = 10.0
    m_m0_nyq:   float = 0.0
    s_m0_harm:  float = 5.0
    m_m0_cos: Optional[Sequence[float]] = None
    m_m0_sin: Optional[Sequence[float]] = None

    # P0 priors (inv-gamma) for dynamic initial states
    a_P0_alpha: float = 2.0; b_P0_alpha: float = 1.0
    a_P0_beta:  float = 2.0; b_P0_beta:  float = 1.0
    a_P0_harm:  float = 2.0; b_P0_harm:  float = 1.0

    # SSVS priors on u = ln s
    ssvs_u_alpha: SSVSLogScalePrior = field(default_factory=lambda: SSVSLogScalePrior(m=-5.0, v1=2.0, v0=0.04, pi=0.5))
    ssvs_u_beta:  SSVSLogScalePrior = field(default_factory=lambda: SSVSLogScalePrior(m=-7.0, v1=2.0, v0=0.04, pi=0.5))
    ssvs_u_gamma: SSVSLogScalePrior = field(default_factory=lambda: SSVSLogScalePrior(m=-6.0, v1=2.0, v0=0.04, pi=0.5))

@dataclass
class SamplerConfig:
    n_iter: int = 20000
    burn: int = 5000
    thin: int = 2
    random_seed: Optional[int] = 42
    progress: bool = True
    progress_every: int = 0
    print_dummies_every: int = 0
    slice_w: float = 1.0
    slice_m: int = 10

    # Sticky SSVS controls (per-block persistence odds multipliers)
    stick_on_alpha: float = 20.0   # odds multiplier if prev δ=1
    stick_off_alpha: float = 5.0   # odds multiplier for staying at 0
    stick_on_beta: float = 20.0
    stick_off_beta: float = 5.0
    stick_on_gamma: float = 10.0
    stick_off_gamma: float = 3.0

    # u=ln s clamps (prevents explosive Q during/after switches)
    u_min_alpha: float = -12.0; u_max_alpha: float = 6.0
    u_min_beta:  float = -16.0; u_max_beta:  float = 6.0
    u_min_gamma: float = -12.0; u_max_gamma: float = 6.0

    # Switch anchoring knobs
    P0_small_alpha: float = 1e-3
    P0_small_beta:  float = 1e-3
    P0_small_harm:  float = 1e-3
    guard_beta_in_alpha_ss_iters: int = 1  # how many iterations to omit β in α-SS after β 0→1

# ===================== main sampler =====================

class HarmonicDLM_SSVS_NC_Reduced:
    """
    Harmonic DLM with **non-centered SSVS** and **reduced-dimension FFBS**.
    δ controls inclusion of blocks in the state:
      δ_α, δ_β ∈ {0,1}, δ_γ ∈ {0,1}, with constraint δ_β=1 ⇒ δ_α=1.
    """

    def __init__(
        self,
        y: np.ndarray,
        period: int,
        harmonics: Optional[int] = None,
        use_nyquist: Optional[bool] = None,

        # indicators
        delta_alpha_init: int = 1,
        delta_beta_init:  int = 1,
        delta_gamma_init: int = 1,

        # log-scale inits u=ln s
        u_alpha_init: float = -5.0,
        u_beta_init:  float = -7.0,
        u_gamma_init: float = -6.0,

        # initial state means/vars (for dynamic use)
        m0_alpha_init: float = 0.0, P0_alpha_init: float = 0.25,
        m0_beta_init:  float = 0.0, P0_beta_init:  float = 0.05,
        m0_cos_init: Optional[Sequence[float]] = None,
        m0_sin_init: Optional[Sequence[float]] = None,
        m0_nyq_init: Optional[float] = 0.0,
        P0_harm_init: float = 0.25,

        sigma2_init: float = 1.0,

        priors: Priors = Priors(),
        cfg: SamplerConfig = SamplerConfig(),
    ):
        self.y = np.asarray(y, float); self.T = int(self.y.size)
        self.s = int(period)
        if self.s < 2: raise ValueError("period must be >= 2")

        K_full = (self.s - 1) // 2
        self.K = K_full if harmonics is None else int(harmonics)
        if not (0 <= self.K <= K_full): raise ValueError(f"K must be in [0, {K_full}]")
        even = (self.s % 2) == 0
        self.use_nyq = bool(use_nyquist and even) if use_nyquist is not None else bool(even and (self.K >= (self.s//2 - 1)))

        self.priors, self.cfg = priors, cfg
        self._rng = np.random.default_rng(cfg.random_seed) if cfg.random_seed is not None else np.random.default_rng()

        # centered time index
        self.t = np.arange(self.T, dtype=float)
        self.tc = self.t - (self.T - 1.0)/2.0

        # harmonic caches
        self._omegas = 2.0 * np.pi * (np.arange(1, self.K + 1, dtype=float)) / float(self.s)
        self._cosw = np.cos(self._omegas); self._sinw = np.sin(self._omegas)

        # full (max) layout reference
        self.full_layout: List[str] = ["alpha", "beta"] + [z for k in range(1, self.K+1) for z in (f"c{k}", f"s{k}")]
        if self.use_nyq: self.full_layout.append("nyq")
        self.full_dim = len(self.full_layout)

        # indicators (+ enforce β ⇒ α)
        self.delta_alpha = int(bool(delta_alpha_init))
        self.delta_beta  = int(bool(delta_beta_init))
        if self.delta_beta==1: self.delta_alpha = 1
        self.delta_gamma = int(bool(delta_gamma_init))
        self._prev_delta_alpha = self.delta_alpha
        self._prev_delta_beta  = self.delta_beta
        self._prev_delta_gamma = self.delta_gamma

        # post-switch guards
        self._alpha_ss_guard_ctr = 0

        # log-scales u (non-centered)
        self.u_alpha = float(u_alpha_init)
        self.u_beta  = float(u_beta_init)
        self.u_gamma = float(u_gamma_init)

        # initial means/vars
        self.m0_alpha = float(m0_alpha_init); self.P0_alpha = float(P0_alpha_init)
        self.m0_beta  = float(m0_beta_init);  self.P0_beta  = float(P0_beta_init)
        if self.K > 0:
            if m0_cos_init is None: m0_cos_init = np.zeros(self.K)
            if m0_sin_init is None: m0_sin_init = np.zeros(self.K)
            if len(m0_cos_init)!=self.K or len(m0_sin_init)!=self.K:
                raise ValueError("m0_cos_init/m0_sin_init must have length K")
            self.m0_cos = np.asarray(m0_cos_init, float)
            self.m0_sin = np.asarray(m0_sin_init, float)
        else:
            self.m0_cos = np.zeros(0); self.m0_sin = np.zeros(0)
        self.m0_nyq = (float(m0_nyq_init) if self.use_nyq else None)
        self.P0_harm = float(P0_harm_init)

        self.sigma2 = float(sigma2_init)

        # latent storage (projected to full layout; NaN where deterministic)
        self.x_store = np.full((self.T, self.full_dim), np.nan, float)

        if self.cfg.progress:
            sd1 = _robust_sd(np.diff(self.y)) if self.T>=2 else 0.0
            sd2 = _robust_sd(np.diff(self.y, n=2)) if self.T>=3 else 0.0
            print(f"[init] sd1={sd1:.4g}, sd2={sd2:.4g} | K={self.K} nyq={self.use_nyq}")

        self.true_mu_t = None
        self.keep: Dict[str, np.ndarray] = {}

    # ---------- dynamic sub-layout given δ ----------
    def _dynamic_layout(self) -> Tuple[List[str], Dict[str,int]]:
        names: List[str] = []
        if self.delta_alpha==1: names.append("alpha")
        if self.delta_beta==1:  names.append("beta")
        if self.delta_gamma==1:
            for k in range(1, self.K+1): names += [f"c{k}", f"s{k}"]
            if self.use_nyq: names.append("nyq")
        idx = {nm:i for i,nm in enumerate(names)}
        return names, idx

    def _full_index(self, name: str) -> int:
        return self.full_layout.index(name)

    # ---------- model matrices for current dynamic set ----------
    def _A(self, dyn_names: List[str], idx: Dict[str,int]) -> np.ndarray:
        d = len(dyn_names); A = np.eye(d)
        if ("alpha" in idx) and ("beta" in idx):
            A[idx["alpha"], idx["beta"]] = 1.0
        if self.delta_gamma==1:
            for k in range(1, self.K+1):
                c, s = f"c{k}", f"s{k}"
                if (c in idx) and (s in idx):
                    i, j = idx[c], idx[s]
                    co, si = float(self._cosw[k-1]), float(self._sinw[k-1])
                    A[i, i] = co; A[i, j] = si; A[j, i] = -si; A[j, j] = co
            if self.use_nyq and ("nyq" in idx):
                A[idx["nyq"], idx["nyq"]] = -1.0
        return A

    def _Q(self, dyn_names: List[str], idx: Dict[str,int]) -> np.ndarray:
        d = len(dyn_names); Q = np.zeros((d,d))
        # clamped s
        s_alpha = math.exp(min(max(self.u_alpha, self.cfg.u_min_alpha), self.cfg.u_max_alpha))
        s_beta  = math.exp(min(max(self.u_beta,  self.cfg.u_min_beta ), self.cfg.u_max_beta ))
        s_gamma = math.exp(min(max(self.u_gamma, self.cfg.u_min_gamma), self.cfg.u_max_gamma))
        if "alpha" in idx: Q[idx["alpha"], idx["alpha"]] = s_alpha**2
        if "beta"  in idx: Q[idx["beta"],  idx["beta"]]  = s_beta**2
        if self.delta_gamma==1:
            for k in range(1, self.K+1):
                c, s = f"c{k}", f"s{k}"
                if c in idx: Q[idx[c], idx[c]] = s_gamma**2
                if s in idx: Q[idx[s], idx[s]] = s_gamma**2
            if self.use_nyq and ("nyq" in idx): Q[idx["nyq"], idx["nyq"]] = s_gamma**2
        return Q

    def _H(self, dyn_names: List[str], idx: Dict[str,int]) -> np.ndarray:
        d = len(dyn_names); h = np.zeros(d)
        if "alpha" in idx: h[idx["alpha"]] = 1.0
        if self.delta_gamma==1:
            for k in range(1, self.K+1):
                c = f"c{k}"
                if c in idx: h[idx[c]] = 1.0
            if self.use_nyq and ("nyq" in idx): h[idx["nyq"]] = 1.0
        return h.reshape(1,-1)

    def _u_drift(self, dyn_names: List[str], idx: Dict[str,int]) -> np.ndarray:
        # centered time ⇒ deterministic trend stays in μ_det only
        return np.zeros(len(dyn_names))

    # ---------- deterministic mean ----------
    def _season_det_t(self, t: int) -> float:
        if self.delta_gamma==1: return 0.0
        val = 0.0
        for k in range(1, self.K+1):
            w = self._omegas[k-1]
            val += self.m0_cos[k-1]*math.cos(w*t) + self.m0_sin[k-1]*math.sin(w*t)
        if self.use_nyq and (self.m0_nyq is not None):
            val += float(self.m0_nyq)*((-1.0)**t)
        return float(val)

    def _mu_det_t(self, t: int) -> float:
        out = 0.0
        if self.delta_alpha==0: out += self.m0_alpha
        if self.delta_beta==0:  out += self.m0_beta * self.tc[t]
        out += self._season_det_t(t)
        return out

    # ---------- FFBS on dynamic subspace ----------
    def _ffbs(self) -> Tuple[np.ndarray, List[str], Dict[str,int]]:
        dyn_names, idx = self._dynamic_layout()
        d = len(dyn_names)
        if d == 0:
            return np.zeros((self.T+1, 0)), dyn_names, idx

        A = self._A(dyn_names, idx)
        Q = self._Q(dyn_names, idx)
        H = self._H(dyn_names, idx)
        R = float(self.sigma2)
        u = self._u_drift(dyn_names, idx)

        m0_vec, P0_diag = self._current_m0P0_for_dyn(dyn_names)
        m = np.zeros((self.T+1, d)); C = np.zeros((self.T+1, d, d))
        a = np.zeros((self.T+1, d)); Rm = np.zeros((self.T+1, d, d))
        m[0] = m0_vec
        C[0] = np.diag(P0_diag) + 1e-12*np.eye(d)

        for t in range(1, self.T+1):
            a[t]  = A @ m[t-1] + u
            Rm[t] = A @ C[t-1] @ A.T + Q
            Rm[t] = 0.5*(Rm[t]+Rm[t].T) + 1e-12*np.eye(d)
            resid = float(self.y[t-1] - self._mu_det_t(t-1))
            S = float(H @ Rm[t] @ H.T + R)
            K = (Rm[t] @ H.T) / S
            v = resid - float(H @ a[t])
            m[t] = a[t] + (K.flatten() * v)
            C[t] = Rm[t] - K @ (H @ Rm[t])
            C[t] = 0.5*(C[t]+C[t].T) + 1e-12*np.eye(d)

        x_dyn = np.zeros((self.T+1, d))
        x_dyn[self.T] = np.random.multivariate_normal(m[self.T], C[self.T])
        for t in range(self.T-1, -1, -1):
            J = C[t] @ A.T
            J = J @ _spd_solve(Rm[t+1], np.eye(d))
            mean = m[t] + J @ (x_dyn[t+1] - (A @ m[t] + u))
            cov  = C[t] - J @ Rm[t+1] @ J.T
            cov  = 0.5*(cov+cov.T)
            lam_min = float(np.linalg.eigvalsh(cov).min())
            if lam_min < 1e-12:
                cov += (1e-12 - lam_min)*np.eye(d)
            x_dyn[t] = np.random.multivariate_normal(mean, cov)

        return x_dyn, dyn_names, idx

    def _current_m0P0_for_dyn(self, dyn_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        m0, P0 = [], []
        for nm in dyn_names:
            if nm == "alpha": m0.append(self.m0_alpha); P0.append(self.P0_alpha)
            elif nm == "beta": m0.append(self.m0_beta);  P0.append(self.P0_beta)
            elif nm.startswith("c") or nm.startswith("s") or nm=="nyq":
                m0.append(self._season_m0_of(nm)); P0.append(self.P0_harm)
            else:
                raise RuntimeError("unknown name in dyn layout")
        return np.asarray(m0,float), np.asarray(P0,float)

    def _season_m0_of(self, nm: str) -> float:
        if nm=="nyq": return float(0.0 if self.m0_nyq is None else self.m0_nyq)
        if nm.startswith("c"):
            k = int(nm[1:]) - 1; return float(self.m0_cos[k])
        if nm.startswith("s"):
            k = int(nm[1:]) - 1; return float(self.m0_sin[k])
        return 0.0

    def _project_to_full(self, x_dyn: np.ndarray, dyn_names: List[str], idx: Dict[str,int]) -> None:
        self.x_store[:] = np.nan
        if x_dyn.shape[1] == 0: return
        for nm in dyn_names:
            j_dyn = idx[nm]
            j_full = self._full_index(nm)
            self.x_store[:, j_full] = x_dyn[1:, j_dyn]

    def _mu_vec(self, x_dyn: np.ndarray, dyn_names: List[str], idx: Dict[str,int]) -> np.ndarray:
        mu = np.zeros(self.T)
        if x_dyn.shape[1] > 0:
            H = self._H(dyn_names, idx)
        for t in range(1, self.T+1):
            dyn_contrib = float(H @ x_dyn[t]) if x_dyn.shape[1] > 0 else 0.0
            mu[t-1] = self._mu_det_t(t-1) + dyn_contrib
        return mu

    # ---------- innovation SS per block ----------
    def _innov_ss_alpha(self, x_dyn: np.ndarray, dyn_names: List[str], idx: Dict[str,int]) -> Tuple[float,int]:
        if "alpha" not in idx: return 0.0, 0
        if self._alpha_ss_guard_ctr > 0:
            # guard: treat as if β weren't dynamic in α-SS once
            ss = 0.0
            for t in range(1, self.T+1):
                mean = x_dyn[t-1, idx["alpha"]]
                ss += (x_dyn[t, idx["alpha"]] - mean)**2
            return float(ss), self.T
        ss = 0.0
        for t in range(1, self.T+1):
            if "beta" in idx:
                mean = x_dyn[t-1, idx["alpha"]] + x_dyn[t-1, idx["beta"]]
            else:
                mean = x_dyn[t-1, idx["alpha"]]
            ss += (x_dyn[t, idx["alpha"]] - mean)**2
        return float(ss), self.T

    def _innov_ss_beta(self, x_dyn: np.ndarray, dyn_names: List[str], idx: Dict[str,int]) -> Tuple[float,int]:
        if "beta" not in idx: return 0.0, 0
        d = x_dyn[1:, idx["beta"]] - x_dyn[:-1, idx["beta"]]
        return float(np.sum(d*d)), self.T

    def _innov_ss_gamma(self, x_dyn: np.ndarray, dyn_names: List[str], idx: Dict[str,int]) -> Tuple[float,int]:
        if self.delta_gamma==0: return 0.0, 0
        ss = 0.0; per = 0
        for k in range(1, self.K+1):
            c, s = f"c{k}", f"s{k}"
            if (c in idx) and (s in idx):
                i, j = idx[c], idx[s]
                co, si = float(self._cosw[k-1]), float(self._sinw[k-1])
                R = np.array([[co, si], [-si, co]], float)
                for t in range(1, self.T+1):
                    prev = x_dyn[t-1, [i,j]]
                    mean = R @ prev
                    diff = x_dyn[t, [i,j]] - mean
                    ss += float(diff @ diff)
        per += 2*self.K
        if self.use_nyq and ("nyq" in idx):
            j = idx["nyq"]
            for t in range(1, self.T+1):
                mean = -x_dyn[t-1, j]
                diff = x_dyn[t, j] - mean
                ss += float(diff*diff)
            per += 1
        return float(ss), int(self.T*per)

    # ===================== SSVS updates on u and sticky deltas =====================

    def _logpost_u(self, u: float, SS: float, n: int, m: float, v: float) -> float:
        return (-n*u - 0.5*SS*math.exp(-2.0*u)) - 0.5*((u-m)**2)/max(v,1e-300)

    @staticmethod
    def _logit(x: float) -> float:
        x = max(min(x, 1-1e-12), 1e-12); return math.log(x) - math.log(1-x)

    def _update_u_delta_sticky(self, *, u0: float, SS: float, n: int, prior: SSVSLogScalePrior,
                               prev_delta: int, stick_on: float, stick_off: float,
                               u_min: float, u_max: float, rng) -> Tuple[float,int]:
        # sample u with current delta-variance (spike vs slab selected by prev delta for stability)
        v_cur = prior.v1 if prev_delta==1 else prior.v0
        logpdf = lambda uu: self._logpost_u(float(uu), SS, n, prior.m, v_cur)
        u = _slice_sample(logpdf, u0, rng, w=self.cfg.slice_w, m=self.cfg.slice_m)
        # clamp u
        u = float(min(max(u, u_min), u_max))

        # baseline spike–slab odds using p(δ=1|u) ∝ N(u|m, v1)*π vs N(u|m, v0)*(1-π)
        log1 = -0.5*math.log(2*math.pi*prior.v1) - 0.5*((u-prior.m)**2)/prior.v1 + math.log(prior.pi)
        log0 = -0.5*math.log(2*math.pi*prior.v0) - 0.5*((u-prior.m)**2)/prior.v0 + math.log(1.0 - prior.pi)
        log_odds = log1 - log0
        # sticky multiplier on odds
        log_odds += math.log(stick_on) if prev_delta==1 else math.log(stick_off)
        p1 = 1.0 / (1.0 + math.exp(-max(min(log_odds, 50.0), -50.0)))
        delta = int(rng.uniform() < p1)
        return u, delta

    def update_nc_ssvs(self, x_dyn: np.ndarray, dyn_names: List[str], idx: Dict[str,int]) -> None:
        rng = self._rng
        # α
        SS, n = self._innov_ss_alpha(x_dyn, dyn_names, idx)
        uA, dA = self._update_u_delta_sticky(
            u0=self.u_alpha, SS=SS, n=n, prior=self.priors.ssvs_u_alpha,
            prev_delta=self._prev_delta_alpha,
            stick_on=self.cfg.stick_on_alpha, stick_off=self.cfg.stick_off_alpha,
            u_min=self.cfg.u_min_alpha, u_max=self.cfg.u_max_alpha, rng=rng,
        )
        self.u_alpha, self.delta_alpha = uA, dA
        # β
        SS, n = self._innov_ss_beta(x_dyn, dyn_names, idx)
        uB, dB = self._update_u_delta_sticky(
            u0=self.u_beta, SS=SS, n=n, prior=self.priors.ssvs_u_beta,
            prev_delta=self._prev_delta_beta,
            stick_on=self.cfg.stick_on_beta, stick_off=self.cfg.stick_off_beta,
            u_min=self.cfg.u_min_beta, u_max=self.cfg.u_max_beta, rng=rng,
        )
        self.u_beta, self.delta_beta  = uB, dB
        # enforce β ⇒ α
        if self.delta_beta==1: self.delta_alpha = 1
        # γ
        SS, n = self._innov_ss_gamma(x_dyn, dyn_names, idx)
        uG, dG = self._update_u_delta_sticky(
            u0=self.u_gamma, SS=SS, n=n, prior=self.priors.ssvs_u_gamma,
            prev_delta=self._prev_delta_gamma,
            stick_on=self.cfg.stick_on_gamma, stick_off=self.cfg.stick_off_gamma,
            u_min=self.cfg.u_min_gamma, u_max=self.cfg.u_max_gamma, rng=rng,
        )
        self.u_gamma, self.delta_gamma = uG, dG

    # ===================== conjugate updates for dynamic starts =====================

    @staticmethod
    def _gibbs_m0_scalar(x0: float, m_prior: float, s_prior: float, P0: float) -> float:
        prec = 1.0/(s_prior**2) + 1.0/max(P0,1e-18)
        var = 1.0/prec
        mean = var*(m_prior/(s_prior**2) + x0/max(P0,1e-18))
        return float(np.random.normal(mean, math.sqrt(var)))

    def update_m0_P0_dynamic_starts(self, x_dyn0: Dict[str,float]) -> None:
        if self.delta_alpha==1:
            self.m0_alpha = self._gibbs_m0_scalar(x_dyn0["alpha"], self.priors.m_m0_alpha, self.priors.s_m0_alpha, self.P0_alpha)
        if self.delta_beta==1:
            self.m0_beta  = self._gibbs_m0_scalar(x_dyn0["beta"],  self.priors.m_m0_beta,  self.priors.s_m0_beta,  self.P0_beta)
        if self.delta_gamma==1 and self.K>0:
            s0 = float(self.priors.s_m0_harm)
            m_cos = (np.zeros(self.K) if self.priors.m_m0_cos is None else np.asarray(self.priors.m_m0_cos,float))
            m_sin = (np.zeros(self.K) if self.priors.m_m0_sin is None else np.asarray(self.priors.m_m0_sin,float))
            for k in range(self.K):
                self.m0_cos[k] = self._gibbs_m0_scalar(x_dyn0[f"c{k+1}"], float(m_cos[k]), s0, self.P0_harm)
                self.m0_sin[k] = self._gibbs_m0_scalar(x_dyn0[f"s{k+1}"], float(m_sin[k]), s0, self.P0_harm)
            if self.use_nyq:
                self.m0_nyq = self._gibbs_m0_scalar(x_dyn0["nyq"], float(self.priors.m_m0_nyq), s0, self.P0_harm)

        # P0’s
        if self.delta_alpha==1:
            a = self.priors.a_P0_alpha + 0.5
            b = self.priors.b_P0_alpha + 0.5*(x_dyn0["alpha"] - self.m0_alpha)**2
            self.P0_alpha = 1.0/np.random.gamma(shape=a, scale=1.0/max(b,1e-300))
        if self.delta_beta==1:
            a = self.priors.a_P0_beta + 0.5
            b = self.priors.b_P0_beta + 0.5*(x_dyn0["beta"] - self.m0_beta)**2
            self.P0_beta = 1.0/np.random.gamma(shape=a, scale=1.0/max(b,1e-300))
        if self.delta_gamma==1:
            diffsq = 0.0; Ktot = 2*self.K + (1 if self.use_nyq else 0)
            targets = [*self.m0_cos, *self.m0_sin] + ([float(self.m0_nyq)] if self.use_nyq else [])
            names = (["c"+str(i+1) for i in range(self.K)] +
                     ["s"+str(i+1) for i in range(self.K)] +
                     (["nyq"] if self.use_nyq else []))
            for k,name in enumerate(names):
                diffsq += (x_dyn0[name] - targets[k])**2
            a = self.priors.a_P0_harm + 0.5*Ktot
            b = self.priors.b_P0_harm + 0.5*diffsq
            self.P0_harm = 1.0/np.random.gamma(shape=a, scale=1.0/max(b,1e-300))

    # ---------- Switch anchoring (deterministic ↔ dynamic) ----------
    def _handle_switches(self, x_dyn: np.ndarray, dyn_names: List[str], idx: Dict[str,int]) -> None:
        """Warm-start dynamic blocks on 0→1 and absorb into deterministic on 1→0.
        Also set the α-innovation guard after β 0→1.
        """
        # helpers
        def dyn_at_t(t1: int) -> float:
            if x_dyn.shape[1]==0: return 0.0
            H = self._H(dyn_names, idx); return float(H @ x_dyn[t1])

        # β switch
        if (self._prev_delta_beta == 0) and (self.delta_beta == 1):
            # warm-start β dynamics from current deterministic slope
            self.P0_beta = min(self.P0_beta, self.cfg.P0_small_beta)
            # small innovation at start to avoid huge Qβ instantly
            self.u_beta = max(self.u_beta, self.cfg.u_min_beta + 2.0)
            # guard α-SS for one iteration
            self._alpha_ss_guard_ctr = max(self._alpha_ss_guard_ctr, self.cfg.guard_beta_in_alpha_ss_iters)
        elif (self._prev_delta_beta == 1) and (self.delta_beta == 0):
            # absorb dynamic β into deterministic β via one-step ridge using current draws
            tc = self.tc
            # residual after removing dynamic α & season at current iteration
            r = np.array([self.y[i] - dyn_at_t(i+1) - (self.m0_alpha if self.delta_alpha==0 else 0.0)
                          - (self._season_det_t(i)) for i in range(self.T)], float)
            s2 = float(self.sigma2); m0, s0 = float(self.priors.m_m0_beta), float(self.priors.s_m0_beta)
            prec = (tc@tc)/s2 + 1.0/(s0**2); mean = ((tc@r)/s2 + m0/(s0**2))/prec
            self.m0_beta = float(mean)

        # α switch
        if (self._prev_delta_alpha == 0) and (self.delta_alpha == 1):
            self.P0_alpha = min(self.P0_alpha, self.cfg.P0_small_alpha)
            self.u_alpha = max(self.u_alpha, self.cfg.u_min_alpha + 2.0)
        elif (self._prev_delta_alpha == 1) and (self.delta_alpha == 0):
            # absorb α dynamic mean level immediately
            r = np.array([self.y[i] - dyn_at_t(i+1) - (self.m0_beta*self.tc[i] if self.delta_beta==0 else 0.0)
                          - self._season_det_t(i) for i in range(self.T)], float)
            s2 = float(self.sigma2); m0, s0 = float(self.priors.m_m0_alpha), float(self.priors.s_m0_alpha)
            prec = self.T/s2 + 1.0/(s0**2); mean = (r.sum()/s2 + m0/(s0**2))/prec
            self.m0_alpha = float(mean)

        # γ switch (deterministic seasonal <-> dynamic)
        if (self._prev_delta_gamma == 0) and (self.delta_gamma == 1):
            self.P0_harm = min(self.P0_harm, self.cfg.P0_small_harm)
            self.u_gamma = max(self.u_gamma, self.cfg.u_min_gamma + 2.0)
        elif (self._prev_delta_gamma == 1) and (self.delta_gamma == 0):
            # absorb dynamic season into deterministic harmonics by LS with ridge prior
            t = np.arange(self.T, dtype=float)
            Zcols = []
            for k in range(1, self.K+1):
                w = self._omegas[k-1]; Zcols += [np.cos(w*t), np.sin(w*t)]
            if self.use_nyq: Zcols.append(((-1.0)**t))
            Z = np.column_stack(Zcols) if len(Zcols)>0 else np.zeros((self.T,0))
            def resid(i):
                return self.y[i] - dyn_at_t(i+1) - (self.m0_alpha if self.delta_alpha==0 else 0.0) \
                       - (self.m0_beta*self.tc[i] if self.delta_beta==0 else 0.0)
            r = np.array([resid(i) for i in range(self.T)], float)
            p = Z.shape[1]
            if p>0:
                s2p = float(self.priors.s_m0_harm)**2; m_prior = np.zeros(p)
                if (self.priors.m_m0_cos is not None) and (self.priors.m_m0_sin is not None):
                    if (len(self.priors.m_m0_cos)==self.K) and (len(self.priors.m_m0_sin)==self.K):
                        m_prior[:2*self.K:2] = np.asarray(self.priors.m_m0_cos,float)
                        m_prior[1:2*self.K:2] = np.asarray(self.priors.m_m0_sin,float)
                if self.use_nyq and (p>2*self.K): m_prior[-1] = float(self.priors.m_m0_nyq)
                sig2 = float(self.sigma2)
                Prec = (Z.T@Z)/sig2 + np.eye(p)/s2p
                b = (Z.T@r)/sig2 + m_prior/s2p
                theta = _spd_solve(Prec, b)
                if self.K>0:
                    self.m0_cos = theta[:2*self.K:2].copy()
                    self.m0_sin = theta[1:2*self.K:2].copy()
                if self.use_nyq: self.m0_nyq = float(theta[-1])

        # countdown guard
        if self._alpha_ss_guard_ctr > 0:
            self._alpha_ss_guard_ctr -= 1

        # update prev flags
        self._prev_delta_alpha = self.delta_alpha
        self._prev_delta_beta  = self.delta_beta
        self._prev_delta_gamma = self.delta_gamma

    # ---------- Deterministic parameter updates (given *current* δ) ----------
    def update_deterministic_params(self, x_dyn: np.ndarray, dyn_names: List[str], idx: Dict[str,int]) -> None:
        def dyn_at_t(t1: int) -> float:
            if x_dyn.shape[1]==0: return 0.0
            H = self._H(dyn_names, idx); return float(H @ x_dyn[t1])

        # α deterministic
        if self.delta_alpha==0:
            r = np.array([self.y[t] - dyn_at_t(t+1) - self._season_det_t(t) - (self.m0_beta*self.tc[t] if self.delta_beta==0 else 0.0)
                          for t in range(self.T)], float)
            s2 = float(self.sigma2); m0, s0 = float(self.priors.m_m0_alpha), float(self.priors.s_m0_alpha)
            prec = self.T/s2 + 1.0/(s0**2); mean = ((r.sum()/s2) + m0/(s0**2))/prec
            self.m0_alpha = float(np.random.normal(mean, math.sqrt(1.0/prec)))

        # β deterministic (centered time)
        if self.delta_beta==0:
            tc = self.tc
            r = np.array([self.y[i] - dyn_at_t(i+1) - self._season_det_t(i) - (self.m0_alpha if self.delta_alpha==0 else 0.0)
                          for i in range(self.T)], float)
            s2 = float(self.sigma2); m0, s0 = float(self.priors.m_m0_beta), float(self.priors.s_m0_beta)
            prec = (tc@tc)/s2 + 1.0/(s0**2); mean = ((tc@r)/s2 + m0/(s0**2))/prec
            self.m0_beta = float(np.random.normal(mean, math.sqrt(1.0/prec)))

        # γ deterministic
        if self.delta_gamma==0 and (self.K>0 or self.use_nyq):
            t = np.arange(self.T, dtype=float)
            Zcols = []
            for k in range(1, self.K+1):
                w = self._omegas[k-1]; Zcols += [np.cos(w*t), np.sin(w*t)]
            if self.use_nyq: Zcols.append(((-1.0)**t))
            Z = np.column_stack(Zcols) if len(Zcols)>0 else np.zeros((self.T,0))
            r = np.array([self.y[i] - dyn_at_t(i+1) - (self.m0_alpha if self.delta_alpha==0 else 0.0)
                          - (self.m0_beta*self.tc[i] if self.delta_beta==0 else 0.0)
                          for i in range(self.T)], float)
            p = Z.shape[1]
            if p>0:
                s2p = float(self.priors.s_m0_harm)**2; m_prior = np.zeros(p)
                if (self.priors.m_m0_cos is not None) and (self.priors.m_m0_sin is not None):
                    if (len(self.priors.m_m0_cos)==self.K) and (len(self.priors.m_m0_sin)==self.K):
                        m_prior[:2*self.K:2] = np.asarray(self.priors.m_m0_cos,float)
                        m_prior[1:2*self.K:2] = np.asarray(self.priors.m_m0_sin,float)
                if self.use_nyq and (p>2*self.K): m_prior[-1] = float(self.priors.m_m0_nyq)
                sig2 = float(self.sigma2)
                Prec = (Z.T@Z)/sig2 + np.eye(p)/s2p
                b = (Z.T@r)/sig2 + m_prior/s2p
                L = np.linalg.cholesky(Prec)
                mu = _spd_solve(Prec, b)
                theta = mu + np.linalg.solve(L.T, np.random.randn(p))
                if self.K>0:
                    self.m0_cos = theta[:2*self.K:2].copy()
                    self.m0_sin = theta[1:2*self.K:2].copy()
                if self.use_nyq: self.m0_nyq = float(theta[-1])

    # ---------- Collapsed σ² update w.r.t. deterministic regressors ----------
    def _update_sigma2_collapsed(self, x_dyn: np.ndarray, dyn_names: List[str], idx: Dict[str,int]) -> None:
        H = self._H(dyn_names, idx) if x_dyn.shape[1]>0 else None
        mu_dyn = np.zeros(self.T, float)
        if x_dyn.shape[1] > 0:
            for t in range(1, self.T+1):
                mu_dyn[t-1] = float(H @ x_dyn[t])
        r = self.y - mu_dyn

        cols = []
        m0_list = []
        s2_list = []
        if self.delta_alpha==0:
            cols.append(np.ones(self.T))
            m0_list.append(self.priors.m_m0_alpha)
            s2_list.append(self.priors.s_m0_alpha**2)
        if self.delta_beta==0:
            cols.append(self.tc)
            m0_list.append(self.priors.m_m0_beta)
            s2_list.append(self.priors.s_m0_beta**2)
        if self.delta_gamma==0:
            t = np.arange(self.T, dtype=float)
            for k in range(1, self.K + 1):
                w = self._omegas[k-1]
                cols.append(np.cos(w * t)); cols.append(np.sin(w * t))
                m0_list.append(0.0 if self.priors.m_m0_cos is None else float(self.priors.m_m0_cos[k-1]))
                s2_list.append(self.priors.s_m0_harm**2)
                m0_list.append(0.0 if self.priors.m_m0_sin is None else float(self.priors.m_m0_sin[k-1]))
                s2_list.append(self.priors.s_m0_harm**2)
            if self.use_nyq:
                cols.append(((-1.0) ** t))
                m0_list.append(float(self.priors.m_m0_nyq))
                s2_list.append(self.priors.s_m0_harm**2)

        if len(cols) == 0:
            e = r
            a = self.priors.a_sigma + 0.5 * self.T
            b = self.priors.b_sigma + 0.5 * float(e @ e)
            tau = np.random.gamma(shape=a, scale=1.0 / max(b,1e-300))
            self.sigma2 = 1.0 / max(tau, 1e-300)
            return

        Z = np.column_stack(cols)
        m0 = np.asarray(m0_list, float)
        S0 = np.diag(np.asarray(s2_list, float))
        A0 = np.linalg.inv(S0)
        A0N = A0 + Z.T @ Z
        rhs = A0 @ m0 + Z.T @ r
        mN  = _spd_solve(A0N, rhs)
        quad0 = float(m0 @ (A0 @ m0))
        quadN = float(mN @ (A0N @ mN))
        aN = self.priors.a_sigma + 0.5 * self.T
        bN = self.priors.b_sigma + 0.5 * (float(r @ r) + quad0 - quadN)
        tau = np.random.gamma(shape=aN, scale=1.0 / max(bN,1e-300))  # precision
        self.sigma2 = 1.0 / max(tau, 1e-300)

    # ===================== progress =====================

    @staticmethod
    def _fmt_list(vals, max_elems: int = 6, fmt: str = ".4g") -> str:
        if vals is None: return "-"
        v = np.asarray(vals,float).ravel()
        if v.size==0: return "[]"
        if v.size<=max_elems: return "[" + ", ".join(f"{x:{fmt}}" for x in v) + "]"
        head = ", ".join(f"{x:{fmt}}" for x in v[:max_elems]); return f"[{head}, …]"

    def _progress_line(self, it: int) -> str:
        parts = [f"[it {it+1}/{self.cfg.n_iter}]",
                 f"σ={math.sqrt(self.sigma2):.3f}",
                 f"Qα≈{math.exp(2*self.u_alpha):.4g}",
                 f"Qβ≈{math.exp(2*self.u_beta):.4g}"]
        if self.K>0 or self.use_nyq: parts.append(f"Qγ≈{math.exp(2*self.u_gamma):.4g}")
        parts.append(f"δα={self.delta_alpha} δβ={self.delta_beta} δγ={self.delta_gamma}")
        parts.append(f"m0α={self.m0_alpha:.4g} P0α={self.P0_alpha:.4g}")
        parts.append(f"m0β={self.m0_beta:.4g} P0β={self.P0_beta:.4g}")
        if self.K>0 or self.use_nyq:
            parts.append(f"m0cos={self._fmt_list(self.m0_cos,6)} m0sin={self._fmt_list(self.m0_sin,6)}"
                         + (f" nyq={0.0 if (self.m0_nyq is None) else float(self.m0_nyq):.4g}" if self.use_nyq else ""))
            parts.append(f"P0harm={self.P0_harm:.4g}")
        return " | ".join(parts)

    def _maybe_print_dummies(self, it: int) -> None:
        n = int(self.cfg.print_dummies_every)
        if n<=0: return
        if (it+1)%n!=0 and it!=self.cfg.n_iter-1: return
        d = harmonics_to_dummies_full_fft(
            s=self.s,
            cos_coefs=self.m0_cos if self.K>0 else np.zeros(0),
            sin_coefs=self.m0_sin if self.K>0 else np.zeros(0),
            use_nyquist=self.use_nyq,
            nyq_coef=self.m0_nyq,
        )
        print(f"seasonal dummies = {self._fmt_list(d, max_elems=self.s, fmt='.4f')}")

    # ===================== run =====================

    def run(self) -> Dict[str,np.ndarray]:
        cfg = self.cfg
        save_iters = list(range(cfg.burn, cfg.n_iter, cfg.thin))
        n_kept, keep_idx = len(save_iters), 0

        self.keep = {
            "sigma": np.zeros(n_kept),
            "mu": np.zeros((n_kept, self.T)),
            "u_alpha": np.zeros(n_kept), "u_beta": np.zeros(n_kept),
            "u_gamma": np.zeros(n_kept if (self.K>0 or self.use_nyq) else 0),
            "delta_alpha": np.zeros(n_kept, int),
            "delta_beta":  np.zeros(n_kept, int),
            "delta_gamma": np.zeros(n_kept, int),
            "m0_alpha": np.zeros(n_kept), "P0_alpha": np.zeros(n_kept),
            "m0_beta":  np.zeros(n_kept), "P0_beta":  np.zeros(n_kept),
            "m0_cos": np.zeros((n_kept, self.K)),
            "m0_sin": np.zeros((n_kept, self.K)),
            "m0_nyq": np.zeros(n_kept) if self.use_nyq else np.zeros(0),
            "P0_harm": np.zeros(n_kept),
            "x": np.zeros((n_kept, self.T, self.full_dim))
        }

        print_every = cfg.progress_every if cfg.progress_every>0 else max(1, cfg.n_iter//50) or 1

        for it in range(cfg.n_iter):
            # 1) FFBS on current dynamic subset
            x_dyn, dyn_names, idx = self._ffbs()

            # 2) Project path to full layout and update dynamic-starts m0/P0
            self._project_to_full(x_dyn, dyn_names, idx)
            x0_map = {nm: float(x_dyn[0, idx[nm]]) for nm in dyn_names}
            self.update_m0_P0_dynamic_starts(x0_map)

            # 3) Sticky SSVS on u and δ (enforce β ⇒ α inside)
            self.update_nc_ssvs(x_dyn, dyn_names, idx)

            # 4) Handle switches (anchoring + guards), then update prev deltas
            self._handle_switches(x_dyn, dyn_names, idx)

            # 5) Deterministic parameter updates given *current* δ
            self.update_deterministic_params(x_dyn, dyn_names, idx)

            # 6) Build μ and **collapsed** σ² update
            mu = self._mu_vec(x_dyn, dyn_names, idx)
            self._update_sigma2_collapsed(x_dyn, dyn_names, idx)

            # progress
            if cfg.progress and ((it+1)%print_every==0 or it==cfg.n_iter-1):
                print(self._progress_line(it))
            self._maybe_print_dummies(it)

            # save
            if it in save_iters:
                self.keep["mu"][keep_idx,:] = mu
                self.keep["sigma"][keep_idx] = math.sqrt(self.sigma2)
                self.keep["u_alpha"][keep_idx] = self.u_alpha
                self.keep["u_beta"][keep_idx]  = self.u_beta
                if self.K>0 or self.use_nyq: self.keep["u_gamma"][keep_idx] = self.u_gamma
                self.keep["delta_alpha"][keep_idx] = self.delta_alpha
                self.keep["delta_beta"][keep_idx]  = self.delta_beta
                self.keep["delta_gamma"][keep_idx] = self.delta_gamma
                self.keep["m0_alpha"][keep_idx] = self.m0_alpha
                self.keep["P0_alpha"][keep_idx] = self.P0_alpha
                self.keep["m0_beta"][keep_idx]  = self.m0_beta
                self.keep["P0_beta"][keep_idx]  = self.P0_beta
                if self.K>0:
                    self.keep["m0_cos"][keep_idx,:] = self.m0_cos
                    self.keep["m0_sin"][keep_idx,:] = self.m0_sin
                if self.use_nyq:
                    self.keep["m0_nyq"][keep_idx] = 0.0 if (self.m0_nyq is None) else float(self.m0_nyq)
                self.keep["P0_harm"][keep_idx] = self.P0_harm
                self.keep["x"][keep_idx,:,:] = self.x_store
                keep_idx += 1

        return self.keep

    # ---------- persistence & truth ----------
    def set_truth_paths(self, mu: Optional[np.ndarray] = None, **_) -> None:
        self.true_mu_t = None if mu is None else np.asarray(mu, float)

    def save_posterior(self, out_npz_path: str, extra_meta: Optional[dict] = None) -> None:
        os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)
        arrays = dict(self.keep); arrays["y"] = self.y.copy()
        if self.true_mu_t is not None: arrays["true_mu_t"] = np.asarray(self.true_mu_t, float)
        np.savez_compressed(out_npz_path, **arrays)
        meta = {
            "T": int(self.T), "period": int(self.s),
            "K": int(self.K), "use_nyquist": bool(self.use_nyq),
            "full_layout": list(self.full_layout),
            "cfg": asdict(self.cfg),
            "priors": asdict(self.priors),
        }
        if extra_meta: meta.update(extra_meta)
        meta_path = out_npz_path.replace(".npz", ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f: json.dump(meta, f, indent=2)
        print(f"[save] Posterior -> {out_npz_path}\n[save] Metadata  -> {meta_path}")

# ---------------- CLI (compact; mirrors your options) ----------------
if __name__ == "__main__":
    import argparse
    from datetime import datetime
    from simulator.mean_time_series_harmonic import Mean_Time_Series

    def _parse_date(s: str | None):
        if not s:
            from datetime import datetime as _dt
            return _dt.today()
        parts = [int(p) for p in s.split("-")]
        if len(parts) == 1: return datetime(parts[0], 1, 1)
        if len(parts) == 2: return datetime(parts[0], parts[1], 1)
        if len(parts) == 3: return datetime(parts[0], parts[1], parts[2])
        raise ValueError("date must be YYYY or YYYY-MM or YYYY-MM-DD")

    def _parse_csv_maybe(s: Optional[str]) -> Optional[List[float]]:
        if s is None: return None
        s = s.strip()
        if s == "": return None
        return [float(z) for z in s.split(",")]

    p = argparse.ArgumentParser(description="Harmonic DLM with NC-SSVS and reduced-dimension FFBS (stability-fixed).")
    # sim
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--period", type=int, default=4)
    p.add_argument("--start-date", type=str, default="2000-01-01")
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--harmonics", type=int, default=None)
    p.add_argument("--use-nyquist", type=int, default=1)
    p.add_argument("--level-mode", choices=["dynamic", "deterministic"], default="deterministic")
    p.add_argument("--trend-mode", choices=["dynamic", "deterministic", "none"], default="deterministic")
    p.add_argument("--seasonal-mode", choices=["dynamic", "deterministic", "none"], default="deterministic")

    p.add_argument("--q-level", type=float, default=0.05)
    p.add_argument("--q-trend", type=float, default=2e-6)
    p.add_argument("--q-season", type=float, default=1e-4)
    
    p.add_argument("--m0-level", type=float, default=5.0)
    p.add_argument("--v0-level", type=float, default=0.25)
    p.add_argument("--m0-trend", type=float, default=0.02)
    p.add_argument("--v0-trend", type=float, default=0.05)
    p.add_argument("--sim-m0-cos", type=str, default=None)
    p.add_argument("--sim-m0-sin", type=str, default=None)
    p.add_argument("--sim-m0-nyq", type=float, default=0.0)
    p.add_argument("--season-dummies", type=str, default="1,1,1,-3")

    # sampler config
    p.add_argument("--n-iter", type=int, default=20000)
    p.add_argument("--burn", type=int, default=5000)
    p.add_argument("--thin", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--progress", type=int, default=1)
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--print-dummies-every", type=int, default=0)
    p.add_argument("--slice-w", type=float, default=1.0)
    p.add_argument("--slice-m", type=int, default=10)

    # SSVS priors on u
    p.add_argument("--u-alpha-m", type=float, default=-5.0)
    p.add_argument("--u-alpha-v1", type=float, default=4.0)
    p.add_argument("--u-alpha-v0", type=float, default=0.04)
    p.add_argument("--u-alpha-pi", type=float, default=0.5)
    p.add_argument("--u-beta-m", type=float, default=-7.0)
    p.add_argument("--u-beta-v1", type=float, default=4.0)
    p.add_argument("--u-beta-v0", type=float, default=0.04)
    p.add_argument("--u-beta-pi", type=float, default=0.5)
    p.add_argument("--u-gamma-m", type=float, default=-6.0)
    p.add_argument("--u-gamma-v1", type=float, default=4.0)
    p.add_argument("--u-gamma-v0", type=float, default=0.04)
    p.add_argument("--u-gamma-pi", type=float, default=0.5)

    # obs prior
    p.add_argument("--prior-a-sigma", type=float, default=2.0)
    p.add_argument("--prior-b-sigma", type=float, default=1.0)

    # m0 priors
    p.add_argument("--prior-m-m0-alpha", type=float, default=0.0)
    p.add_argument("--prior-s-m0-alpha", type=float, default=10.0)
    p.add_argument("--prior-m-m0-beta", type=float, default=0.0)
    p.add_argument("--prior-s-m0-beta", type=float, default=10.0)
    p.add_argument("--prior-m-m0-cos", type=str, default=None)
    p.add_argument("--prior-m-m0-sin", type=str, default=None)
    p.add_argument("--prior-m-m0-nyq", type=float, default=0.0)
    p.add_argument("--prior-s-m0-harm", type=float, default=5.0)

    # P0 priors
    p.add_argument("--prior-a-P0-alpha", type=float, default=2.0)
    p.add_argument("--prior-b-P0-alpha", type=float, default=1.0)
    p.add_argument("--prior-a-P0-beta", type=float, default=2.0)
    p.add_argument("--prior-b-P0-beta", type=float, default=1.0)
    p.add_argument("--prior-a-P0-harm", type=float, default=2.0)
    p.add_argument("--prior-b-P0-harm", type=float, default=1.0)

    # inits
    p.add_argument("--sigma-init", type=float, default=1.0)
    p.add_argument("--P0-alpha-init", type=float, default=0.25)
    p.add_argument("--P0-beta-init", type=float, default=0.05)
    p.add_argument("--P0-harm-init", type=float, default=0.25)
    p.add_argument("--m0-cos-init", type=str, default=None)
    p.add_argument("--m0-sin-init", type=str, default=None)
    p.add_argument("--m0-nyq-init", type=float, default=None)
    p.add_argument("--delta-alpha-init", type=int, default=1, choices=[0, 1])
    p.add_argument("--delta-beta-init",  type=int, default=1, choices=[0, 1])
    p.add_argument("--delta-gamma-init", type=int, default=1, choices=[0, 1])
    p.add_argument("--u-alpha-init", type=float, default=-5.0)
    p.add_argument("--u-beta-init",  type=float, default=-7.0)
    p.add_argument("--u-gamma-init", type=float, default=-6.0)

    # output
    p.add_argument("--out-dir", type=str, default="results/simulations/DLM_harm_NC_SSVS_reduced")
    p.add_argument("--plot", type=int, default=1)
    p.add_argument("--print-summary", type=int, default=1)

    args = p.parse_args()
    np.random.seed(args.seed)

    if args.harmonics is None:
        args.harmonics = (args.period - 1) // 2
    use_nyq = bool(int(args.use_nyquist))

    # parse csv
    def _csv(s): return _parse_csv_maybe(s)
    def _parse_csv_maybe(s: Optional[str]) -> Optional[List[float]]:
        if s is None: return None
        s = s.strip()
        if s == "": return None
        return [float(z) for z in s.split(",")]

    sim_m0_cos = _csv(args.sim_m0_cos); sim_m0_sin = _csv(args.sim_m0_sin)
    pri_m_cos  = _csv(args.prior_m_m0_cos); pri_m_sin = _csv(args.prior_m_m0_sin)
    m0_cos_init = _csv(args.m0_cos_init); m0_sin_init = _csv(args.m0_sin_init)
    season_dummies = _csv(args.season_dummies)

    # simulator
    mts = Mean_Time_Series(
        sigma=args.sigma, period=args.period,
        level_mode=args.level_mode, trend_mode=args.trend_mode, seasonal_mode=args.seasonal_mode,
        season_harmonics=args.harmonics, season_use_nyquist=use_nyq,
        q_level=args.q_level, q_trend=args.q_trend, q_season=args.q_season,
        m0_level=args.m0_level, v0_level=args.v0_level,
        m0_trend=args.m0_trend, v0_trend=args.v0_trend,
        m0_cos=sim_m0_cos, m0_sin=sim_m0_sin, m0_nyq=args.sim_m0_nyq,
        season_dummies=season_dummies,
        start_date=_parse_date(args.start_date),
        rng=np.random.default_rng(args.seed),
    )
    y = np.array([mts.move() or mts.measure() for _ in range(args.T)], float)
    truth = mts.get_truth_paths(as_numpy=True)
    mu_T = truth["mu_t"][1:1+args.T]
    dates_T = truth["index"][:args.T]

    # seasonal init from dummies if needed
    if (m0_cos_init is None or m0_sin_init is None) and (season_dummies is not None):
        if len(season_dummies) != args.period:
            raise ValueError(f"--season-dummies must have length period={args.period}")
        centered = center_and_report_dummies_full(season_dummies, tol=1e-12)
        cos_coefs, sin_coefs, nyq_val = dummies_full_to_harmonics_fft(centered, K=args.harmonics, use_nyquist=use_nyq)
        if m0_cos_init is None: m0_cos_init = list(cos_coefs)
        if m0_sin_init is None: m0_sin_init = list(sin_coefs)
        if args.m0_nyq_init is None and use_nyq:
            args.m0_nyq_init = float(0.0 if nyq_val is None else nyq_val)
    if m0_cos_init is None: m0_cos_init = [0.0] * args.harmonics
    if m0_sin_init is None: m0_sin_init = [0.0] * args.harmonics
    if args.m0_nyq_init is None: args.m0_nyq_init = 0.0

    # priors/config
    priors = Priors(
        a_sigma=args.prior_a_sigma, b_sigma=args.prior_b_sigma,
        m_m0_alpha=args.prior_m_m0_alpha, s_m0_alpha=args.prior_s_m0_alpha,
        m_m0_beta=args.prior_m_m0_beta,  s_m0_beta=args.prior_s_m0_beta,
        m_m0_cos=None if pri_m_cos is None else pri_m_cos,
        m_m0_sin=None if pri_m_sin is None else pri_m_sin,
        m_m0_nyq=args.prior_m_m0_nyq, s_m0_harm=args.prior_s_m0_harm,
        a_P0_alpha=args.prior_a_P0_alpha, b_P0_alpha=args.prior_b_P0_alpha,
        a_P0_beta=args.prior_a_P0_beta,   b_P0_beta=args.prior_b_P0_beta,
        a_P0_harm=args.prior_a_P0_harm,   b_P0_harm=args.prior_b_P0_harm,
        ssvs_u_alpha=SSVSLogScalePrior(m=args.u_alpha_m, v1=args.u_alpha_v1, v0=args.u_alpha_v0, pi=args.u_alpha_pi),
        ssvs_u_beta =SSVSLogScalePrior(m=args.u_beta_m , v1=args.u_beta_v1 , v0=args.u_beta_v0 , pi=args.u_beta_pi ),
        ssvs_u_gamma=SSVSLogScalePrior(m=args.u_gamma_m, v1=args.u_gamma_v1, v0=args.u_gamma_v0, pi=args.u_gamma_pi),
    )
    cfg = SamplerConfig(
        n_iter=int(args.n_iter), burn=int(args.burn), thin=int(args.thin),
        random_seed=int(args.seed), progress=bool(args.progress),
        progress_every=int(args.progress_every),
        print_dummies_every=int(args.print_dummies_every),
        slice_w=float(args.slice_w), slice_m=int(args.slice_m),
    )

    sampler = HarmonicDLM_SSVS_NC_Reduced(
        y=y, period=args.period, harmonics=args.harmonics, use_nyquist=use_nyq,
        delta_alpha_init=args.delta_alpha_init, delta_beta_init=args.delta_beta_init, delta_gamma_init=args.delta_gamma_init,
        u_alpha_init=args.u_alpha_init, u_beta_init=args.u_beta_init, u_gamma_init=args.u_gamma_init,
        m0_alpha_init=args.m0_level, m0_beta_init=args.m0_trend,
        P0_alpha_init=args.P0_alpha_init, P0_beta_init=args.P0_beta_init, P0_harm_init=args.P0_harm_init,
        m0_cos_init=m0_cos_init, m0_sin_init=m0_sin_init, m0_nyq_init=args.m0_nyq_init,
        sigma2_init=args.sigma_init**2,
        priors=priors, cfg=cfg,
    )
    sampler.set_truth_paths(mu=mu_T)

    if args.print_summary:
        print(f"\nSimulated {args.T} obs (σ={mts.sigma:.3g}). K={sampler.K}, Nyquist={sampler.use_nyq}")
        print("SSVS(u=ln s) priors:",
              f"α{vars(priors.ssvs_u_alpha)} β{vars(priors.ssvs_u_beta)} γ{vars(priors.ssvs_u_gamma)}")

    t0 = time.time(); post = sampler.run(); elapsed = time.time() - t0
    print(f"[Run completed in {elapsed:.1f}s]")

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(args.out_dir, f"NCSSVS_reduced_K{sampler.K}_nyq{int(bool(sampler.use_nyq))}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    sampler.save_posterior(
        out_npz_path=os.path.join(out_dir, "posterior.npz"),
        extra_meta={"elapsed_seconds": float(elapsed),
                    "ssvs_u": {"alpha": asdict(priors.ssvs_u_alpha),
                               "beta":  asdict(priors.ssvs_u_beta),
                               "gamma": asdict(priors.ssvs_u_gamma)}}
    )

    if args.print_summary:
        print("--- Posterior means ---")
        print(f"σ = {np.mean(post['sigma']):.4f}")
        print(f"Pr(δ_α=1) ≈ {np.mean(post['delta_alpha']):.3f}")
        print(f"Pr(δ_β=1) ≈ {np.mean(post['delta_beta']):.3f}")
        if post["delta_gamma"].size:
            print(f"Pr(δ_γ=1) ≈ {np.mean(post['delta_gamma']):.3f}")

    if args.plot:
        import matplotlib.pyplot as plt
        mu_hat = post["mu"].mean(axis=0)
        plt.figure(figsize=(10, 4))
        plt.plot(dates_T, y, label="y_t", lw=1)
        if sampler.true_mu_t is not None:
            plt.plot(dates_T, mu_T, "--", label="μ_t (truth)")
        plt.plot(dates_T, mu_hat, "-.", label="μ̂_t (post mean)")
        plt.title(f"Harmonic DLM + NC-SSVS (reduced FFBS, stabilized) | K={sampler.K}, nyq={sampler.use_nyq}")
        plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()
