from __future__ import annotations
"""
Gaussian structural time-series with harmonic seasonality (cos/sin + optional Nyquist)
FFBS + Gibbs for Gaussian parts, and **LogNormal priors on dimensionless taus**:

    s_alpha = c_alpha * tau_alpha,   ln tau_alpha ~ N(mu, sd^2)
    s_beta  = c_beta  * tau_beta
    s_gamma = c_gamma * tau_gamma
    (optional) sigma = c_eps * tau_eps

We slice-sample z = ln(tau) with SS* = SS / c^2, so geometry is scale-free.

This file expects:
  center_and_report_dummies_full, dummies_full_to_harmonics_fft, harmonics_to_dummies_full_fft
in optimization.harmonic_helpers
"""
import json, math, os, sys, time, warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
warnings.filterwarnings("ignore", category=DeprecationWarning)

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_dir)
from optimization.harmonic_helpers import (
    center_and_report_dummies_full,
    dummies_full_to_harmonics_fft,
    harmonics_to_dummies_full_fft,
)

# -------------------- small utils --------------------
def _mad(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    if v.size == 0: return 0.0
    m = np.median(v)
    return float(np.median(np.abs(v - m)))

def _robust_sd(v: np.ndarray) -> float:
    return _mad(np.asarray(v, float)) / 1.4826 if np.size(v) else 0.0

def _spd_solve(M: np.ndarray, B: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    I = np.eye(M.shape[0])
    for k in range(3):
        try:
            L = np.linalg.cholesky(M + (eps * (10**k)) * I)
            return np.linalg.solve(L.T, np.linalg.solve(L, B))
        except np.linalg.LinAlgError:
            pass
    return np.linalg.pinv(M) @ B

# -------------------- slice sampler --------------------
def _slice_sample(logpdf, z0: float, rng: np.random.Generator,
                  w: float = 1.0, m: int = 10, max_shrink: int = 1000) -> float:
    z0 = float(z0)
    logy = float(logpdf(z0)) - rng.exponential(1.0)
    u = rng.uniform(0.0, 1.0)
    L = z0 - u * w
    R = L + w
    J = int(rng.integers(0, m + 1)) if m > 0 else 0
    K = (m - 1 - J) if m > 0 else 0
    while (J > 0) and (logpdf(L) > logy):
        L -= w; J -= 1
    while (K > 0) and (logpdf(R) > logy):
        R += w; K -= 1
    it = 0
    while it < max_shrink:
        z1 = rng.uniform(L, R)
        if logpdf(z1) >= logy: return float(z1)
        if z1 < z0: L = z1
        else: R = z1
        it += 1
    return float(z0)

# -------------------- priors & config --------------------
@dataclass
class Priors:
    # observation variance (Gamma on precision) baseline; can be replaced by ln tau eps below
    a_sigma: float = 2.0
    b_sigma: float = 1.0

    # m0 priors
    m_m0_alpha: float = 0.0; s_m0_alpha: float = 10.0
    m_m0_beta:  float = 0.0; s_m0_beta:  float = 10.0
    m_m0_nyq:   float = 0.0
    s_m0_harm:  float = 5.0
    m_m0_cos: Optional[Sequence[float]] = None
    m_m0_sin: Optional[Sequence[float]] = None

    # P0 priors (Inv-Gamma through Gamma on precision)
    a_P0_alpha: float = 2.0; b_P0_alpha: float = 1.0
    a_P0_beta:  float = 2.0; b_P0_beta:  float = 1.0
    a_P0_harm:  float = 2.0; b_P0_harm:  float = 1.0

    # LogNormal on dimensionless taus (process)
    ln_tau_alpha_mu: float = 0.0; ln_tau_alpha_sd: float = 1.0
    ln_tau_beta_mu:  float = 0.0; ln_tau_beta_sd:  float = 1.0
    ln_tau_gamma_mu: float = 0.0; ln_tau_gamma_sd: float = 1.0

    # Optional LogNormal on tau_eps (observation). If use_ln_tau_eps=False we ignore these.
    ln_tau_eps_mu: float = 0.0
    ln_tau_eps_sd: float = 1.0

@dataclass
class SamplerConfig:
    n_iter: int = 4000; burn: int = 1000; thin: int = 1
    random_seed: Optional[int] = 7
    progress: bool = True; progress_every: int = 0
    print_dummies_every: int = 0
    slice_w: float = 1.0; slice_m: int = 10
    # τ reparam controls
    use_dimensionless_scales: bool = True
    # data-driven default scales (if None, computed from y)
    c_alpha: Optional[float] = None
    c_beta:  Optional[float] = None
    c_gamma: Optional[float] = None
    c_eps:   Optional[float] = None
    # observation update choice
    use_ln_tau_eps: bool = False

# -------------------- model --------------------
class DLMGibbsHarmonic:
    def __init__(self,
                 y: np.ndarray, period: int,
                 harmonics: Optional[int] = None,
                 use_nyquist: Optional[bool] = None,
                 level_mode: str = "dynamic",
                 trend_mode: str = "dynamic",
                 seasonal_mode: str = "dynamic",
                 m0_alpha_init: float = 0.0, P0_alpha_init: float = 1.0,
                 m0_beta_init: float = 0.0,  P0_beta_init:  float = 1.0,
                 m0_cos_init: Optional[Sequence[float]] = None,
                 m0_sin_init: Optional[Sequence[float]] = None,
                 m0_nyq_init: float = 0.0,   P0_harm_init: float = 1.0,
                 # inits now given as s_k (still OK); we convert to tau via scales in init
                 sigma2_init: float = 1.0, s_alpha_init: float = 1e-2,
                 s_beta_init: float = 1e-3, s_gamma_init: float = 1e-3,
                 priors: Priors = Priors(), cfg: SamplerConfig = SamplerConfig()):
        # data/spec
        self.y = np.asarray(y, float); self.T = int(self.y.size)
        self.s = int(period);  assert self.s >= 2
        K_full = (self.s - 1)//2
        self.K = K_full if harmonics is None else int(harmonics)
        even = (self.s % 2) == 0
        self.use_nyq = bool(even and (self.K >= (self.s//2 - 1))) if use_nyquist is None else bool(use_nyquist and even)
        for m in (level_mode, trend_mode, seasonal_mode):
            if m not in {"dynamic","deterministic","none"}: raise ValueError("invalid mode")
        if trend_mode == "dynamic" and level_mode != "dynamic":
            raise ValueError("dynamic trend requires dynamic level")
        self.level_mode, self.trend_mode, self.seasonal_mode = level_mode, trend_mode, seasonal_mode

        self.priors, self.cfg = priors, cfg
        self._rng = np.random.default_rng(cfg.random_seed) if cfg.random_seed is not None else np.random.default_rng()

        # harmonic caches
        self._omegas = 2.0*np.pi*(np.arange(1, self.K+1, dtype=float))/float(self.s)
        self._cosw, self._sinw = np.cos(self._omegas), np.sin(self._omegas)

        # layout
        layout: List[str] = []
        if level_mode   == "dynamic": layout.append("alpha")
        if trend_mode   == "dynamic": layout.append("beta")
        if seasonal_mode== "dynamic":
            for k in range(1, self.K+1): layout += [f"c{k}", f"s{k}"]
            if self.use_nyq: layout.append("nyq")
        self._layout = layout; self.dim = len(layout)
        self.idx_alpha = layout.index("alpha") if "alpha" in layout else None
        self.idx_beta  = layout.index("beta")  if "beta"  in layout else None
        def _idx_pair(k:int)->int:
            pos=0
            if self.idx_alpha is not None: pos+=1
            if self.idx_beta  is not None: pos+=1
            return pos+2*(k-1)
        self._idx_pair=_idx_pair
        if seasonal_mode=="dynamic":
            self.idx_first_season = (_idx_pair(1) if self.K>0 else None)
            self.idx_nyq = (None if not self.use_nyq else
                            ((1 if self.idx_alpha is not None else 0)
                             + (1 if self.idx_beta  is not None else 0)
                             + 2*self.K))
        else:
            self.idx_first_season=None; self.idx_nyq=None

        # ---------------- scales c_* (dimensioned) ----------------
        if cfg.use_dimensionless_scales:
            sd1 = _robust_sd(np.diff(self.y)) if self.T>=2 else 1.0
            sd2 = _robust_sd(np.diff(self.y, n=2)) if self.T>=3 else max(sd1,1.0)
            seas_diff = _robust_sd(self.y[self.s:]-self.y[:-self.s]) if self.T>self.s else _robust_sd(self.y)
            sdy = _robust_sd(self.y)
            self.c_alpha = float(cfg.c_alpha if cfg.c_alpha is not None else max(sd1, 1e-8))
            self.c_beta  = float(cfg.c_beta  if cfg.c_beta  is not None else max(sd2, 1e-8))
            self.c_gamma = float(cfg.c_gamma if cfg.c_gamma is not None else max(seas_diff, 1e-8))
            self.c_eps   = float(cfg.c_eps   if cfg.c_eps   is not None else max(sdy, 1e-8))
        else:
            self.c_alpha = self.c_beta = self.c_gamma = self.c_eps = 1.0

        # ---------------- parameters / inits ----------------
        # observation (keep as variance; optional τ param in update)
        self.sigma2 = float(sigma2_init)

        # process SDs as s = c * tau  ⇒ store tau, expose s via property
        def _init_tau(s_init: float, c: float) -> float:
            return float(max(s_init, 1e-16) / max(c, 1e-16))
        self.tau_alpha = _init_tau(s_alpha_init, self.c_alpha) if self.idx_alpha is not None else 0.0
        self.tau_beta  = _init_tau(s_beta_init,  self.c_beta ) if self.idx_beta  is not None else 0.0
        self.tau_gamma = _init_tau(s_gamma_init, self.c_gamma) if seasonal_mode=="dynamic" else 0.0

        self.m0_alpha = float(m0_alpha_init) if self.idx_alpha is not None else 0.0
        self.P0_alpha = float(P0_alpha_init) if self.idx_alpha is not None else 0.0
        self.m0_beta  = float(m0_beta_init)  if self.idx_beta  is not None else 0.0
        self.P0_beta  = float(P0_beta_init)  if self.idx_beta  is not None else 0.0
        if self.K>0:
            if m0_cos_init is None: m0_cos_init = np.zeros(self.K)
            if m0_sin_init is None: m0_sin_init = np.zeros(self.K)
            if len(m0_cos_init)!=self.K or len(m0_sin_init)!=self.K:
                raise ValueError("m0_cos_init/m0_sin_init must have length K")
            self.m0_cos = np.asarray(m0_cos_init,float); self.m0_sin = np.asarray(m0_sin_init,float)
        else:
            self.m0_cos = np.zeros(0); self.m0_sin = np.zeros(0)
        self.m0_nyq = (float(m0_nyq_init) if self.use_nyq else None)
        self.P0_harm = float(P0_harm_init)

        # latent path (x_0 ~ N(m0, P0))
        self.x = np.zeros((self.T+1, self.dim), float)
        if self.dim>0:
            m0_vec, P0_diag = self._current_m0_P0()
            self.x[0] = np.random.multivariate_normal(m0_vec, np.diag(P0_diag)+1e-12*np.eye(self.dim))
            self._propagate_initial_path(Q_init=np.full(self.dim, 1e-6))

        # storage
        self.keep: Dict[str, np.ndarray] = {}
        if self.cfg.progress:
            nyq_tag = f"nyq={self.use_nyq}"
            print(f"[init] scales c: α={self.c_alpha:.4g}, β={self.c_beta:.4g}, γ={self.c_gamma:.4g}, ε={self.c_eps:.4g} | K={self.K} {nyq_tag}")

        # optional truth overlays
        self.true_sigma=None; self.true_Q=None; self.true_mu_t=None
        self.true_alpha_t=None; self.true_beta_t=None; self.true_gamma_t=None

    # ---------- convenience: current s_ from tau ----------
    @property
    def s_alpha(self)->float: return float(self.c_alpha*self.tau_alpha)
    @property
    def s_beta (self)->float: return float(self.c_beta *self.tau_beta )
    @property
    def s_gamma(self)->float: return float(self.c_gamma*self.tau_gamma)

    # ---------- truth overlays ----------
    def set_truth(self, **kwargs)->None:
        self.true_sigma = float(kwargs["sigma"]) if "sigma" in kwargs and kwargs["sigma"] is not None else None
        self.true_Q = np.asarray(kwargs["Q"], float) if "Q" in kwargs and kwargs["Q"] is not None else None
        self.true_mu_t = np.asarray(kwargs["mu"], float) if "mu" in kwargs and kwargs["mu"] is not None else None
        self.true_alpha_t = np.asarray(kwargs["alpha"], float) if "alpha" in kwargs and kwargs["alpha"] is not None else None
        self.true_beta_t  = np.asarray(kwargs["beta"],  float) if "beta"  in kwargs and kwargs["beta"]  is not None else None
        self.true_gamma_t = np.asarray(kwargs["gamma"], float) if "gamma" in kwargs and kwargs["gamma"] is not None else None

    def set_truth_paths(self, mu: Optional[np.ndarray]=None, **_)->None:
        self.true_mu_t = None if mu is None else np.asarray(mu, float)

    # ---------- model matrices ----------
    def _H(self)->np.ndarray:
        if self.dim==0: return np.zeros((1,0))
        h = np.zeros(self.dim)
        if self.idx_alpha is not None: h[self.idx_alpha]=1.0
        if self.seasonal_mode=="dynamic" and self.K>0:
            for k in range(1,self.K+1):
                i=self._idx_pair(k); h[i]=1.0
            if self.use_nyq: h[self.idx_nyq]=1.0
        return h.reshape(1,-1)

    def _A(self)->np.ndarray:
        if self.dim==0: return np.zeros((0,0))
        A = np.eye(self.dim)
        if self.idx_alpha is not None and self.idx_beta is not None:
            A[self.idx_alpha, self.idx_beta] = 1.0
        if self.seasonal_mode=="dynamic" and self.K>0:
            for k in range(1,self.K+1):
                i=self._idx_pair(k); co,si = float(self._cosw[k-1]), float(self._sinw[k-1])
                A[i,i]=co; A[i,i+1]=si; A[i+1,i]=-si; A[i+1,i+1]=co
            if self.use_nyq: A[self.idx_nyq, self.idx_nyq] = -1.0
        return A

    def _u(self)->np.ndarray:
        if self.dim==0: return np.zeros(0)
        u=np.zeros(self.dim)
        if (self.idx_alpha is not None) and (self.trend_mode=="deterministic"):
            u[self.idx_alpha]=float(self.m0_beta)
        return u

    def _Q(self)->np.ndarray:
        if self.dim==0: return np.zeros((0,0))
        Q=np.zeros((self.dim,self.dim))
        if self.idx_alpha is not None and self.tau_alpha>0:
            Q[self.idx_alpha, self.idx_alpha] = self.s_alpha**2
        if self.idx_beta is not None and self.tau_beta>0:
            Q[self.idx_beta, self.idx_beta] = self.s_beta**2
        if self.seasonal_mode=="dynamic" and self.tau_gamma>0:
            for k in range(1,self.K+1):
                i=self._idx_pair(k)
                v=self.s_gamma**2
                Q[i,i]=v; Q[i+1,i+1]=v
            if self.use_nyq:
                Q[self.idx_nyq, self.idx_nyq] = self.s_gamma**2
        return Q

    # ---------- deterministic mean pieces ----------
    def _season_det(self, t:int)->float:
        if self.seasonal_mode!="deterministic": return 0.0
        val=0.0
        for k in range(1,self.K+1):
            w=self._omegas[k-1]
            val += self.m0_cos[k-1]*math.cos(w*t) + self.m0_sin[k-1]*math.sin(w*t)
        if self.use_nyq and (self.m0_nyq is not None):
            val += float(self.m0_nyq) * ((-1.0)**t)
        return float(val)

    def _mu_det(self, t:int)->float:
        out=0.0
        if self.level_mode=="deterministic": out+=self.m0_alpha
        if (self.trend_mode=="deterministic") and (self.idx_alpha is None): out += self.m0_beta*t
        out += self._season_det(t)
        return out

    def _current_m0_P0(self)->Tuple[np.ndarray,np.ndarray]:
        m0,P0=[],[]
        if self.idx_alpha is not None: m0.append(self.m0_alpha); P0.append(self.P0_alpha)
        if self.idx_beta  is not None: m0.append(self.m0_beta ); P0.append(self.P0_beta )
        if self.seasonal_mode=="dynamic":
            for k in range(self.K):
                m0 += [self.m0_cos[k], self.m0_sin[k]]
                P0 += [self.P0_harm,   self.P0_harm]
            if self.use_nyq:
                m0.append(float(0.0 if self.m0_nyq is None else self.m0_nyq)); P0.append(self.P0_harm)
        return np.asarray(m0,float), np.asarray(P0,float)

    # ---------- FFBS ----------
    def _ffbs(self)->np.ndarray:
        if self.dim==0: return self.x.copy()
        H,A,Q,R = self._H(), self._A(), self._Q(), float(self.sigma2)
        m0_vec, P0_diag = self._current_m0_P0()
        m=np.zeros((self.T+1,self.dim)); C=np.zeros((self.T+1,self.dim,self.dim))
        a=np.zeros((self.T+1,self.dim)); Rm=np.zeros((self.T+1,self.dim,self.dim))
        m[0]=m0_vec; C[0]=np.diag(P0_diag)+1e-12*np.eye(self.dim)
        u=self._u()
        for t in range(1,self.T+1):
            a[t]  = A @ m[t-1] + u
            Rm[t] = A @ C[t-1] @ A.T + Q
            Rm[t] = 0.5*(Rm[t]+Rm[t].T) + 1e-12*np.eye(self.dim)
            resid_mean = float(self.y[t-1] - self._mu_det(t-1))
            S = float(H @ Rm[t] @ H.T + R)
            if S<=0: S=float(H @ (Rm[t]+1e-10*np.eye(self.dim)) @ H.T + R)
            K  = (Rm[t] @ H.T) / S
            v  = resid_mean - float(H @ a[t])
            m[t]= a[t] + (K.flatten()*v)
            C[t]= Rm[t] - K @ (H @ Rm[t])
            C[t]= 0.5*(C[t]+C[t].T) + 1e-12*np.eye(self.dim)
        x=np.zeros_like(self.x)
        x[self.T] = np.random.multivariate_normal(m[self.T], C[self.T])
        for t in range(self.T-1,-1,-1):
            J = C[t] @ A.T
            J = J @ _spd_solve(Rm[t+1], np.eye(self.dim))
            mean = m[t] + J @ (x[t+1] - (A @ m[t] + u))
            cov  = C[t] - J @ Rm[t+1] @ J.T
            cov  = 0.5*(cov+cov.T)
            eigmin = float(np.linalg.eigvalsh(cov).min())
            if eigmin < 1e-12: cov += (1e-12-eigmin)*np.eye(cov.shape[0])
            x[t] = np.random.multivariate_normal(mean, cov)
        return x

    def _propagate_initial_path(self, Q_init: np.ndarray)->None:
        if self.dim==0: return
        A,u = self._A(), self._u()
        for t in range(1,self.T+1):
            self.x[t] = A @ self.x[t-1] + u + np.random.normal(0.0, np.sqrt(Q_init), size=self.dim)

    # ---------- helpers ----------
    def _mu_vec(self)->np.ndarray:
        H=self._H(); mu=np.zeros(self.T)
        for t in range(1,self.T+1):
            dyn = float(H @ self.x[t]) if self.dim>0 else 0.0
            mu[t-1] = self._mu_det(t-1) + dyn
        return mu

    # ---------- σ^2 updates ----------
    def update_sigma2(self)->None:
        e = self.y - self._mu_vec()
        if not self.cfg.use_ln_tau_eps:
            # Gamma on precision with scale correction
            SS_star = float(e @ e) / (self.c_eps**2)
            a = self.priors.a_sigma + 0.5*self.T
            b = self.priors.b_sigma + 0.5*SS_star
            tau = np.random.gamma(shape=a, scale=1.0/b)
            self.sigma2 = (self.c_eps**2) / max(tau, 1e-300)
        else:
            # LogNormal prior on tau_eps with slice on z=ln tau_eps
            SS_star = float(e @ e) / (self.c_eps**2)
            T_eff = self.T
            mu, sd = float(self.priors.ln_tau_eps_mu), float(self.priors.ln_tau_eps_sd)
            def _logpost(z: float)->float:
                # ll(z) = -T_eff*z - 0.5*SS_star*exp(-2z)
                return (-T_eff*z - 0.5*SS_star*math.exp(-2.0*z)) - 0.5*((z-mu)/sd)**2
            z0 = 0.5*math.log(max(self.sigma2,1e-300)) - math.log(self.c_eps)  # since sigma = c_eps*exp(z)
            z  = _slice_sample(_logpost, z0, self._rng, w=self.cfg.slice_w, m=self.cfg.slice_m)
            self.sigma2 = (self.c_eps**2) * math.exp(2.0*z)

    # ---------- innovation SS ----------
    def _innovation_ss_alpha(self)->Tuple[float,int]:
        if self.idx_alpha is None: return 0.0, 0
        ss=0.0
        for t in range(1,self.T+1):
            drift=0.0
            if self.idx_beta is not None: drift = self.x[t-1, self.idx_beta]
            elif self.trend_mode=="deterministic": drift = float(self.m0_beta)
            mean = self.x[t-1, self.idx_alpha] + drift
            ss += (self.x[t, self.idx_alpha] - mean)**2
        return float(ss), self.T

    def _innovation_ss_beta(self)->Tuple[float,int]:
        if self.idx_beta is None: return 0.0, 0
        d = self.x[1:, self.idx_beta] - self.x[:-1, self.idx_beta]
        return float(np.sum(d*d)), self.T

    def _innovation_ss_gamma(self)->Tuple[float,int]:
        if self.seasonal_mode!="dynamic": return 0.0, 0
        ss = 0.0; per_step=0
        for k in range(1,self.K+1):
            i = self._idx_pair(k)
            co, si = float(self._cosw[k-1]), float(self._sinw[k-1])
            R = np.array([[co, si], [-si, co]], float)
            for t in range(1,self.T+1):
                prev = self.x[t-1, i:i+2]
                mean = R @ prev
                err  = self.x[t, i:i+2] - mean
                ss  += float(err @ err)
        per_step += 2*self.K
        if self.use_nyq:
            j=self.idx_nyq
            for t in range(1,self.T+1):
                mean = -self.x[t-1, j]
                err  = self.x[t, j] - mean
                ss  += float(err*err)
            per_step += 1
        return float(ss), int(self.T*per_step)

    # ---------- τ updates (slice on ln tau) ----------
    @staticmethod
    def _logpost_z_tau(z: float, SS_star: float, T_eff: int, mu: float, sd: float)->float:
        # ll(z) = -T_eff*z - 0.5*SS_star*exp(-2z); lp(z) = -0.5*((z-mu)/sd)^2
        return -T_eff*z - 0.5*SS_star*math.exp(-2.0*z) - 0.5*((z-mu)/sd)**2

    def update_process_Q_lognormal_tau(self)->None:
        rng=self._rng; w=self.cfg.slice_w; m=self.cfg.slice_m
        # α
        if self.idx_alpha is not None:
            SS, T_eff = self._innovation_ss_alpha()
            SS_star = SS / (self.c_alpha**2)
            mu, sd = float(self.priors.ln_tau_alpha_mu), float(self.priors.ln_tau_alpha_sd)
            z0 = math.log(max(self.tau_alpha, 1e-16))
            z  = _slice_sample(lambda z: self._logpost_z_tau(z, SS_star, T_eff, mu, sd), z0, rng, w=w, m=m)
            self.tau_alpha = float(math.exp(z))
        # β
        if self.idx_beta is not None:
            SS, T_eff = self._innovation_ss_beta()
            SS_star = SS / (self.c_beta**2)
            mu, sd = float(self.priors.ln_tau_beta_mu), float(self.priors.ln_tau_beta_sd)
            z0 = math.log(max(self.tau_beta, 1e-16))
            z  = _slice_sample(lambda z: self._logpost_z_tau(z, SS_star, T_eff, mu, sd), z0, rng, w=w, m=m)
            self.tau_beta = float(math.exp(z))
        # γ
        if self.seasonal_mode=="dynamic":
            SS, T_eff = self._innovation_ss_gamma()
            SS_star = SS / (self.c_gamma**2)
            mu, sd = float(self.priors.ln_tau_gamma_mu), float(self.priors.ln_tau_gamma_sd)
            z0 = math.log(max(self.tau_gamma, 1e-16))
            z  = _slice_sample(lambda z: self._logpost_z_tau(z, SS_star, T_eff, mu, sd), z0, rng, w=w, m=m)
            self.tau_gamma = float(math.exp(z))

    # ---------- m0 / P0 ----------
    @staticmethod
    def _gibbs_m0_scalar(x0: float, m_prior: float, s_prior: float, P0: float)->float:
        prec = 1.0/(s_prior**2) + 1.0/max(1e-18, P0)
        var  = 1.0/prec
        mean = var*( m_prior/(s_prior**2) + x0/max(1e-18, P0) )
        return float(np.random.normal(mean, math.sqrt(var)))

    def update_m0(self)->None:
        if self.dim==0: return
        pos=0
        if self.idx_alpha is not None:
            self.m0_alpha = self._gibbs_m0_scalar(float(self.x[0,pos]), self.priors.m_m0_alpha, self.priors.s_m0_alpha, self.P0_alpha); pos+=1
        if self.idx_beta is not None:
            self.m0_beta  = self._gibbs_m0_scalar(float(self.x[0,pos]), self.priors.m_m0_beta,  self.priors.s_m0_beta,  self.P0_beta ); pos+=1
        if self.seasonal_mode=="dynamic":
            s0=float(self.priors.s_m0_harm)
            m_cos = (np.zeros(self.K) if self.priors.m_m0_cos is None else np.asarray(self.priors.m_m0_cos,float))
            m_sin = (np.zeros(self.K) if self.priors.m_m0_sin is None else np.asarray(self.priors.m_m0_sin,float))
            if m_cos.size!=self.K or m_sin.size!=self.K: raise ValueError("priors.m_m0_cos/m_m0_sin length K")
            for k in range(self.K):
                self.m0_cos[k] = self._gibbs_m0_scalar(float(self.x[0, pos+2*k    ]), float(m_cos[k]), s0, self.P0_harm)
                self.m0_sin[k] = self._gibbs_m0_scalar(float(self.x[0, pos+2*k +1]), float(m_sin[k]), s0, self.P0_harm)
            if self.use_nyq:
                j=pos+2*self.K
                self.m0_nyq = self._gibbs_m0_scalar(float(self.x[0,j]), float(self.priors.m_m0_nyq), s0, self.P0_harm)

    def update_P0(self)->None:
        if self.dim==0: return
        pos=0
        if self.idx_alpha is not None:
            a=self.priors.a_P0_alpha+0.5
            b=self.priors.b_P0_alpha+0.5*(float(self.x[0,pos])-self.m0_alpha)**2
            self.P0_alpha = 1.0/np.random.gamma(shape=a, scale=1.0/b); pos+=1
        if self.idx_beta is not None:
            a=self.priors.a_P0_beta+0.5
            b=self.priors.b_P0_beta+0.5*(float(self.x[0,pos])-self.m0_beta )**2
            self.P0_beta = 1.0/np.random.gamma(shape=a, scale=1.0/b); pos+=1
        if self.seasonal_mode=="dynamic":
            diffsq=0.0; Ktot=2*self.K+(1 if self.use_nyq else 0)
            target = [*self.m0_cos, *self.m0_sin] + ([float(self.m0_nyq)] if self.use_nyq else [])
            for k in range(Ktot):
                diffsq += (float(self.x[0,pos+k]) - float(target[k]))**2
            a=self.priors.a_P0_harm + 0.5*Ktot
            b=self.priors.b_P0_harm + 0.5*diffsq
            self.P0_harm = 1.0/np.random.gamma(shape=a, scale=1.0/b)

    # ---------- deterministic params ----------
    def update_deterministic_params(self)->None:
        # level
        if self.level_mode=="deterministic":
            r=self.y.copy()
            if self.dim>0:
                H=self._H()
                for t in range(1,self.T+1): r[t-1] -= float(H @ self.x[t])
            r -= np.array([self._season_det(t) for t in range(self.T)], float)
            if (self.idx_alpha is None) and (self.trend_mode=="deterministic"):
                r -= self.m0_beta*np.arange(self.T,float)
            s2=float(self.sigma2); m0,s0 = float(self.priors.m_m0_alpha), float(self.priors.s_m0_alpha)
            prec = self.T/s2 + 1.0/(s0**2); mean = ((r.sum()/s2)+m0/(s0**2))/prec
            self.m0_alpha = float(np.random.normal(mean, math.sqrt(1.0/prec)))
        # trend
        if self.trend_mode=="deterministic":
            if self.idx_alpha is not None:
                d = self.x[1:, self.idx_alpha] - self.x[:-1, self.idx_alpha]
                s2 = float(self.s_alpha**2) if self.tau_alpha>0 else 1e-12
                m0,s0 = float(self.priors.m_m0_beta), float(self.priors.s_m0_beta)
                prec = (self.T/s2) + 1.0/(s0**2)
                mean = ((float(np.sum(d))/s2) + m0/(s0**2))/prec
                self.m0_beta = float(np.random.normal(mean, math.sqrt(1.0/prec)))
            else:
                t = np.arange(self.T,float)
                r = self.y.copy()
                if self.dim>0:
                    H=self._H()
                    for k in range(1,self.T+1): r[k-1] -= float(H @ self.x[k])
                if self.level_mode=="deterministic": r -= self.m0_alpha
                r -= np.array([self._season_det(tt) for tt in range(self.T)], float)
                m0,s0 = float(self.priors.m_m0_beta), float(self.priors.s_m0_beta)
                sig2=float(self.sigma2)
                Prec=(t@t)/sig2 + 1.0/(s0**2)
                mu=((t@r)/sig2 + m0/(s0**2))/Prec
                L=np.sqrt(1.0/Prec)
                self.m0_beta = float(np.random.normal(mu, L))
        # season (deterministic)
        if self.seasonal_mode=="deterministic":
            t=np.arange(self.T,float); Zcols=[]
            for k in range(1,self.K+1):
                w=self._omegas[k-1]; Zcols += [np.cos(w*t), np.sin(w*t)]
            if self.use_nyq: Zcols.append(((-1.0)**t))
            Z = np.column_stack(Zcols) if Zcols else np.zeros((self.T,0))
            r=self.y.copy()
            if self.dim>0:
                H=self._H()
                for k in range(1,self.T+1): r[k-1] -= float(H @ self.x[k])
            if self.level_mode=="deterministic": r -= self.m0_alpha
            if (self.idx_alpha is None) and (self.trend_mode=="deterministic"): r -= self.m0_beta*t
            p=Z.shape[1]; s2=float(self.priors.s_m0_harm)**2
            m_prior=np.zeros(p)
            if (self.priors.m_m0_cos is not None) and (self.priors.m_m0_sin is not None):
                if len(self.priors.m_m0_cos)==self.K and len(self.priors.m_m0_sin)==self.K:
                    m_prior[:2*self.K:2] = np.asarray(self.priors.m_m0_cos,float)
                    m_prior[1:2*self.K:2] = np.asarray(self.priors.m_m0_sin,float)
            if self.use_nyq and (p>2*self.K): m_prior[-1]=float(self.priors.m_m0_nyq)
            sig2=float(self.sigma2)
            Prec=(Z.T@Z)/sig2 + np.eye(p)/s2
            b=(Z.T@r)/sig2 + m_prior/s2
            mu=np.linalg.solve(Prec,b)
            L=np.linalg.cholesky(Prec)
            theta = mu + np.linalg.solve(L.T, np.random.randn(p))
            if self.K>0:
                self.m0_cos = theta[:2*self.K:2].copy()
                self.m0_sin = theta[1:2*self.K:2].copy()
            if self.use_nyq: self.m0_nyq = float(theta[-1])

    # ---------- progress & printing ----------
    @staticmethod
    def _fmt_list(vals, max_elems:int=6, fmt:str=".4g")->str:
        if vals is None: return "-"
        v=np.asarray(vals,float).ravel()
        if v.size==0: return "[]"
        if v.size<=max_elems: return "[" + ", ".join(f"{x:{fmt}}" for x in v) + "]"
        head = ", ".join(f"{x:{fmt}}" for x in v[:max_elems])
        return f"[{head}, …]"

    def _progress_line(self, it:int)->str:
        parts=[f"[it {it+1}/{self.cfg.n_iter}]", f"σ={math.sqrt(self.sigma2):.3f}"]
        if self.idx_alpha is not None: parts.append(f"Qα={(self.s_alpha**2):.4g} (τ={self.tau_alpha:.3g})")
        if self.idx_beta  is not None: parts.append(f"Qβ={(self.s_beta**2):.4g} (τ={self.tau_beta:.3g})")
        if self.seasonal_mode=="dynamic": parts.append(f"Qγ={(self.s_gamma**2):.4g} (τ={self.tau_gamma:.3g})")
        if self.level_mode!="none":
            parts.append(f"m0α={self.m0_alpha:.4g} P0α={(self.P0_alpha if self.level_mode=='dynamic' else 0.0):.4g}")
        if self.trend_mode!="none":
            parts.append(f"m0β={self.m0_beta:.4g} P0β={(self.P0_beta if self.trend_mode=='dynamic' else 0.0):.4g}")
        if self.seasonal_mode!="none":
            parts.append(f"m0cos={self._fmt_list(self.m0_cos,6)} m0sin={self._fmt_list(self.m0_sin,6)}"
                         + (f" nyq={0.0 if (self.m0_nyq is None) else float(self.m0_nyq):.4g}" if self.use_nyq else ""))
            if self.seasonal_mode=="dynamic": parts.append(f"P0harm={self.P0_harm:.4g}")
        return " | ".join(parts)

    def _maybe_print_dummies(self, it:int)->None:
        n=int(self.cfg.print_dummies_every)
        if n<=0: return
        if (it+1)%n!=0 and it!=self.cfg.n_iter-1: return
        if self.seasonal_mode in ("dynamic","deterministic"):
            d = harmonics_to_dummies_full_fft(
                s=self.s,
                cos_coefs=self.m0_cos if self.K>0 else np.zeros(0),
                sin_coefs=self.m0_sin if self.K>0 else np.zeros(0),
                use_nyquist=self.use_nyq,
                nyq_coef=self.m0_nyq,
            )
            print(f"seasonal dummies = {self._fmt_list(d, max_elems=self.s, fmt='.4f')}")

    # ---------- MCMC ----------
    def run(self)->Dict[str,np.ndarray]:
        cfg=self.cfg
        save_iters = list(range(cfg.burn, cfg.n_iter, cfg.thin))
        n_kept, keep_idx = len(save_iters), 0
        self.keep = {"sigma": np.zeros(n_kept,float), "mu": np.zeros((n_kept,self.T),float)}
        if self.idx_alpha is not None:
            self.keep.update({"Q_alpha": np.zeros(n_kept), "tau_alpha": np.zeros(n_kept),
                              "m0_alpha": np.zeros(n_kept), "P0_alpha": np.zeros(n_kept)})
        if self.idx_beta is not None:
            self.keep.update({"Q_beta": np.zeros(n_kept), "tau_beta": np.zeros(n_kept),
                              "m0_beta": np.zeros(n_kept), "P0_beta": np.zeros(n_kept)})
        if self.seasonal_mode=="dynamic":
            self.keep.update({
                "Q_gamma": np.zeros(n_kept), "tau_gamma": np.zeros(n_kept),
                "m0_cos": np.zeros((n_kept,self.K)), "m0_sin": np.zeros((n_kept,self.K)),
                "m0_nyq": np.zeros(n_kept) if self.use_nyq else np.zeros(0),
                "P0_harm": np.zeros(n_kept),
                "x": np.zeros((n_kept,self.T,self.dim)) if self.dim>0 else np.zeros((0,0,0))
            })
        else:
            self.keep.update({
                "m0_cos": np.zeros((n_kept,self.K)),
                "m0_sin": np.zeros((n_kept,self.K)),
                "m0_nyq": np.zeros(n_kept) if self.use_nyq else np.zeros(0)
            })
            if self.dim>0: self.keep["x"] = np.zeros((n_kept,self.T,self.dim))

        print_every = cfg.progress_every if cfg.progress_every>0 else max(1, cfg.n_iter//50) or 1

        for it in range(cfg.n_iter):
            if self.dim>0: self.x = self._ffbs()
            if self.dim>0: self.update_process_Q_lognormal_tau()
            if self.dim>0:
                self.update_m0()
                self.update_P0()
            self.update_deterministic_params()
            self.update_sigma2()

            if cfg.progress and ((it+1)%print_every==0 or it==cfg.n_iter-1):
                print(self._progress_line(it))
            self._maybe_print_dummies(it)

            if it in save_iters:
                mu=self._mu_vec()
                self.keep["mu"][keep_idx,:]=mu
                self.keep["sigma"][keep_idx]=math.sqrt(self.sigma2)
                if self.idx_alpha is not None:
                    self.keep["Q_alpha"][keep_idx]=self.s_alpha**2
                    self.keep["tau_alpha"][keep_idx]=self.tau_alpha
                    self.keep["m0_alpha"][keep_idx]=self.m0_alpha
                    self.keep["P0_alpha"][keep_idx]=self.P0_alpha
                if self.idx_beta is not None:
                    self.keep["Q_beta"][keep_idx]=self.s_beta**2
                    self.keep["tau_beta"][keep_idx]=self.tau_beta
                    self.keep["m0_beta"][keep_idx]=self.m0_beta
                    self.keep["P0_beta"][keep_idx]=self.P0_beta
                if self.seasonal_mode=="dynamic":
                    self.keep["Q_gamma"][keep_idx]=self.s_gamma**2
                    self.keep["tau_gamma"][keep_idx]=self.tau_gamma
                    self.keep["m0_cos"][keep_idx,:]=self.m0_cos
                    self.keep["m0_sin"][keep_idx,:]=self.m0_sin
                    if self.use_nyq: self.keep["m0_nyq"][keep_idx]=0.0 if (self.m0_nyq is None) else float(self.m0_nyq)
                    self.keep["P0_harm"][keep_idx]=self.P0_harm
                else:
                    self.keep["m0_cos"][keep_idx,:]=self.m0_cos
                    self.keep["m0_sin"][keep_idx,:]=self.m0_sin
                    if self.use_nyq: self.keep["m0_nyq"][keep_idx]=0.0 if (self.m0_nyq is None) else float(self.m0_nyq)
                if "x" in self.keep and self.dim>0: self.keep["x"][keep_idx,:,:] = self.x[1:self.T+1,:]
                keep_idx+=1
        return self.keep

    # ---------- persistence ----------
    def save_posterior(self, out_npz_path: str, extra_meta: Optional[dict]=None)->None:
        os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)
        arrays = dict(self.keep); arrays["y"]=self.y.copy()
        if "x" not in arrays: arrays["x"]=np.zeros((0,0,0))
        if self.true_sigma is not None: arrays["true_sigma"]=float(self.true_sigma)
        if self.true_Q is not None: arrays["true_Q"]=np.asarray(self.true_Q,float)
        if self.true_mu_t is not None: arrays["true_mu_t"]=np.asarray(self.true_mu_t,float)
        for nm in ("true_alpha_t","true_beta_t","true_gamma_t"):
            if getattr(self, nm, None) is not None: arrays[nm]=np.asarray(getattr(self,nm),float)
        np.savez_compressed(out_npz_path, **arrays)

        meta = {
            "T": int(self.T), "dim": int(self.dim), "period": int(self.s),
            "harmonics": int(self.K), "use_nyquist": bool(self.use_nyq),
            "modes": {"level_mode": self.level_mode, "trend_mode": self.trend_mode, "seasonal_mode": self.seasonal_mode},
            "layout": list(self._layout),
            "cfg": asdict(self.cfg),
            "priors": asdict(self.priors),
            "scales": {"c_alpha": self.c_alpha, "c_beta": self.c_beta, "c_gamma": self.c_gamma, "c_eps": self.c_eps},
        }
        if extra_meta: meta.update(extra_meta)
        meta_path = out_npz_path.replace(".npz", ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f: json.dump(meta, f, indent=2)
        print(f"[save] Posterior -> {out_npz_path}\n[save] Metadata  -> {meta_path}")


# ---------------- CLI / example run ----------------
if __name__ == "__main__":
    import argparse
    from datetime import datetime
    from simulator.mean_time_series_harmonic import Mean_Time_Series

    def _parse_date(s: str | None):
        if not s:
            from datetime import datetime as _dt
            return _dt.today()
        parts=[int(p) for p in s.split("-")]
        if len(parts)==1: return datetime(parts[0],1,1)
        if len(parts)==2: return datetime(parts[0],parts[1],1)
        if len(parts)==3: return datetime(parts[0],parts[1],parts[2])
        raise ValueError("start-date must be YYYY, YYYY-MM, or YYYY-MM-DD")

    def _parse_csv_maybe(s: Optional[str])->Optional[List[float]]:
        if s is None: return None
        s=s.strip()
        if s=="": return None
        return [float(z) for z in s.split(",")]

    p=argparse.ArgumentParser(
        description=("Gaussian DLM with harmonics. FFBS + Gibbs. "
                     "LogNormal on dimensionless taus for process noises; optional for obs.")
    )
    # simulation
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--period", type=int, default=4)
    p.add_argument("--start-date", type=str, default="2000-01-01")
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--level-mode", choices=["dynamic","deterministic"], default="dynamic")
    p.add_argument("--trend-mode", choices=["dynamic","deterministic","none"], default="dynamic")
    p.add_argument("--seasonal-mode", choices=["dynamic","deterministic","none"], default="dynamic")
    p.add_argument("--q-level", type=float, default=0.05)
    p.add_argument("--q-trend", type=float, default=0.000002)
    p.add_argument("--q-season", type=float, default=0.0001)
    p.add_argument("--m0-level", type=float, default=5.0)
    p.add_argument("--v0-level", type=float, default=0.25)
    p.add_argument("--m0-trend", type=float, default=0.02)
    p.add_argument("--v0-trend", type=float, default=0.05)

    # harmonics
    p.add_argument("--harmonics", type=int, default=None)
    p.add_argument("--use-nyquist", type=int, default=1)

    # simulator season init (optional)
    p.add_argument("--sim-m0-cos", type=str, default=None)
    p.add_argument("--sim-m0-sin", type=str, default=None)
    p.add_argument("--sim-m0-nyq", type=float, default=0.0)
    p.add_argument("--season-dummies", type=str, default='1,1,1,-3')

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

    # observation variance prior
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

    # initial inference values (still in s-space; converted to tau using c’s)
    p.add_argument("--sigma-init", type=float, default=1.0)
    p.add_argument("--s-alpha-init", type=float, default=1e-2)
    p.add_argument("--s-beta-init", type=float, default=1e-3)
    p.add_argument("--s-gamma-init", type=float, default=1e-3)
    p.add_argument("--P0-alpha-init", type=float, default=0.25)
    p.add_argument("--P0-beta-init", type=float, default=0.05)
    p.add_argument("--P0-harm-init", type=float, default=0.25)
    p.add_argument("--m0-cos-init", type=str, default=None)
    p.add_argument("--m0-sin-init", type=str, default=None)
    p.add_argument("--m0-nyq-init", type=float, default=None)

    # LogNormal priors on taus
    p.add_argument("--ln-tau-alpha-mu", type=float, default=0.0)
    p.add_argument("--ln-tau-alpha-sd", type=float, default=1.0)
    p.add_argument("--ln-tau-beta-mu",  type=float, default=0.0)
    p.add_argument("--ln-tau-beta-sd",  type=float, default=1.0)
    p.add_argument("--ln-tau-gamma-mu", type=float, default=0.0)
    p.add_argument("--ln-tau-gamma-sd", type=float, default=1.0)

    # Optional LogNormal on tau_eps
    p.add_argument("--use-ln-tau-eps", type=int, default=1)
    p.add_argument("--ln-tau-eps-mu", type=float, default=0.0)
    p.add_argument("--ln-tau-eps-sd", type=float, default=1.0)

    # override scales if desired
    p.add_argument("--c-alpha", type=float, default=None)
    p.add_argument("--c-beta",  type=float, default=None)
    p.add_argument("--c-gamma", type=float, default=None)
    p.add_argument("--c-eps",   type=float, default=None)

    # output
    p.add_argument("--out-dir", type=str, default="results/simulations/DLM_harm_tau")
    p.add_argument("--plot", type=int, default=1)
    p.add_argument("--print-summary", type=int, default=1)

    args = p.parse_args()
    np.random.seed(args.seed)

    if args.harmonics is None: args.harmonics = (args.period - 1)//2
    use_nyq = bool(int(args.use_nyquist))

    # lists
    def _csv(s): return None if s is None else ([float(z) for z in s.strip().split(",")] if s.strip()!="" else None)
    sim_m0_cos=_csv(args.sim_m0_cos); sim_m0_sin=_csv(args.sim_m0_sin)
    pri_m_cos =_csv(args.prior_m_m0_cos); pri_m_sin =_csv(args.prior_m_m0_sin)
    m0_cos_init=_csv(args.m0_cos_init); m0_sin_init=_csv(args.m0_sin_init)
    season_dummies=_csv(args.season_dummies)

    # simulator
    mts = Mean_Time_Series(
        sigma=args.sigma, period=args.period,
        level_mode=args.level_mode, trend_mode=args.trend_mode, seasonal_mode=args.seasonal_mode,
        season_harmonics=args.harmonics, season_use_nyquist=use_nyq,
        q_level=args.q_level, q_trend=args.q_trend, q_season=args.q_season,
        m0_level=args.m0_level, v0_level=args.v0_level,
        m0_trend=(0.0 if args.trend_mode=="none" else args.m0_trend), v0_trend=args.v0_trend,
        m0_cos=sim_m0_cos, m0_sin=sim_m0_sin, m0_nyq=args.sim_m0_nyq,
        season_dummies=season_dummies,
        start_date=_parse_date(args.start_date),
        rng=np.random.default_rng(args.seed),
    )
    y = np.array([mts.move() or mts.measure() for _ in range(args.T)], float)
    truth = mts.get_truth_paths(as_numpy=True)
    mu_T = truth["mu_t"][1:1+args.T]; dates_T = truth["index"][:args.T]

    # priors
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
        ln_tau_alpha_mu=args.ln_tau_alpha_mu, ln_tau_alpha_sd=args.ln_tau_alpha_sd,
        ln_tau_beta_mu=args.ln_tau_beta_mu,   ln_tau_beta_sd=args.ln_tau_beta_sd,
        ln_tau_gamma_mu=args.ln_tau_gamma_mu, ln_tau_gamma_sd=args.ln_tau_gamma_sd,
        ln_tau_eps_mu=args.ln_tau_eps_mu,     ln_tau_eps_sd=args.ln_tau_eps_sd,
    )
    cfg = SamplerConfig(
        n_iter=int(args.n_iter), burn=int(args.burn), thin=int(args.thin),
        random_seed=int(args.seed), progress=bool(args.progress),
        progress_every=int(args.progress_every),
        print_dummies_every=int(args.print_dummies_every),
        slice_w=float(args.slice_w), slice_m=int(args.slice_m),
        use_dimensionless_scales=True,
        c_alpha=args.c_alpha, c_beta=args.c_beta, c_gamma=args.c_gamma, c_eps=args.c_eps,
        use_ln_tau_eps=bool(int(args.use_ln_tau_eps)),
    )

    # derive sampler seasonal inits from dummies if needed
    if (m0_cos_init is None or m0_sin_init is None) and (season_dummies is not None):
        if len(season_dummies)!=args.period:
            raise ValueError(f"--season-dummies must have length period={args.period}")
        centered = center_and_report_dummies_full(season_dummies, tol=1e-12)
        cos_coefs, sin_coefs, nyq_val = dummies_full_to_harmonics_fft(centered, K=args.harmonics, use_nyquist=use_nyq)
        if m0_cos_init is None: m0_cos_init = list(cos_coefs)
        if m0_sin_init is None: m0_sin_init = list(sin_coefs)
        if args.m0_nyq_init is None and use_nyq: args.m0_nyq_init = float(0.0 if nyq_val is None else nyq_val)
    if m0_cos_init is None: m0_cos_init = [0.0]*args.harmonics
    if m0_sin_init is None: m0_sin_init = [0.0]*args.harmonics
    if args.m0_nyq_init is None: args.m0_nyq_init = 0.0

    sampler = DLMGibbsHarmonic(
        y=y, period=args.period,
        harmonics=args.harmonics, use_nyquist=use_nyq,
        level_mode=args.level_mode, trend_mode=args.trend_mode, seasonal_mode=args.seasonal_mode,
        sigma2_init=args.sigma_init**2,
        s_alpha_init=args.s_alpha_init, s_beta_init=args.s_beta_init, s_gamma_init=args.s_gamma_init,
        m0_alpha_init=args.m0_level, m0_beta_init=(0.0 if args.trend_mode=="none" else args.m0_trend),
        P0_alpha_init=args.P0_alpha_init, P0_beta_init=args.P0_beta_init, P0_harm_init=args.P0_harm_init,
        m0_cos_init=m0_cos_init, m0_sin_init=m0_sin_init, m0_nyq_init=args.m0_nyq_init,
        priors=priors, cfg=cfg,
    )
    sampler.set_truth_paths(mu=mu_T)

    if args.print_summary:
        print(f"\nSimulated {args.T} obs (σ={mts.sigma:.3g}) | modes {args.level_mode}/{args.trend_mode}/{args.seasonal_mode}")
        print(f"Harmonics: K={sampler.K}, Nyquist={sampler.use_nyq}")
        print(f"LN priors (ln τ): α~N({priors.ln_tau_alpha_mu:.2f},{priors.ln_tau_alpha_sd:.2f}²), "
              f"β~N({priors.ln_tau_beta_mu:.2f},{priors.ln_tau_beta_sd:.2f}²), "
              f"γ~N({priors.ln_tau_gamma_mu:.2f},{priors.ln_tau_gamma_sd:.2f}²) | "
              f"obs via {'LN τ_eps' if cfg.use_ln_tau_eps else 'Gamma(precision)'}")

    t0=time.time()
    post = sampler.run()
    elapsed=time.time()-t0
    print(f"[Run completed in {elapsed:.1f}s]")

    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir=os.path.join(args.out_dir, f"{args.level_mode}-{args.trend_mode}-{args.seasonal_mode}_K{sampler.K}_nyq{int(bool(sampler.use_nyq))}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    sampler.save_posterior(
        out_npz_path=os.path.join(out_dir,"posterior.npz"),
        extra_meta={
            "elapsed_seconds": float(elapsed),
            "ln_tau_priors": {
                "alpha": {"mu": priors.ln_tau_alpha_mu, "sd": priors.ln_tau_alpha_sd},
                "beta":  {"mu": priors.ln_tau_beta_mu,  "sd": priors.ln_tau_beta_sd},
                "gamma": {"mu": priors.ln_tau_gamma_mu, "sd": priors.ln_tau_gamma_sd},
                "eps":   {"mu": priors.ln_tau_eps_mu,   "sd": priors.ln_tau_eps_sd, "used": bool(cfg.use_ln_tau_eps)},
            },
        },
    )

    if args.print_summary:
        print("--- Posterior means ---")
        print(f"σ = {np.mean(post['sigma']):.4f}")
        for k in ["alpha","beta","gamma"]:
            key=f"Q_{k}"
            if key in post:
                mQ=np.mean(post[key])
                print(f"Q_{k} = {mQ:.4g} (√Q ≈ {math.sqrt(max(mQ,0.0)):.4g})")

    if args.plot:
        import matplotlib.pyplot as plt
        mu_hat = post["mu"].mean(axis=0)
        plt.figure(figsize=(10,4))
        plt.plot(dates_T, y, label="y_t", lw=1)
        plt.plot(dates_T, mu_T, "--", label="μ_t (truth)")
        plt.plot(dates_T, mu_hat, "-.", label="μ̂_t (post mean)")
        plt.title(f"DLM harmonic (τ-param): {args.level_mode}/{args.trend_mode}/{args.seasonal_mode} | K={sampler.K}, nyq={sampler.use_nyq}")
        plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()
