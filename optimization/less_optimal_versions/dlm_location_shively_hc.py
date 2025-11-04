# dlm_gibbs_shively_selection.py
from __future__ import annotations

import json, math, os, warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
warnings.filterwarnings("ignore", category=DeprecationWarning)

# =============================================================================
# Small utils
# =============================================================================
def _mad(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    if v.size == 0: return 0.0
    m = np.median(v)
    return float(np.median(np.abs(v - m)))

def _robust_sd(v: np.ndarray) -> float:
    return _mad(np.asarray(v, float)) / 1.4826 if v.size else 0.0

def _spd_solve(M: np.ndarray, B: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    n = M.shape[0]; I = np.eye(n)
    for k in range(3):
        try:
            L = np.linalg.cholesky(M + (eps*(10**k))*I)
            return np.linalg.solve(L.T, np.linalg.solve(L, B))
        except np.linalg.LinAlgError:
            pass
    return np.linalg.pinv(M) @ B

# =============================================================================
# Priors & Config
# =============================================================================
@dataclass
class Priors:
    # Observation variance σ²: precision τ = 1/σ² ~ Gamma(a_sigma, b_sigma)
    a_sigma: float = 2.0
    b_sigma: float = 1.0

    # m0 priors (used both for dynamic x0 means and deterministic params)
    m_m0_alpha: float = 0.0;  s_m0_alpha: float = 10.0
    m_m0_beta:  float = 0.0;  s_m0_beta:  float = 10.0
    m_m0_gamma: Optional[Sequence[float]] = None  # length p-1 (newest-first)
    s_m0_gamma: float = 5.0

    # Initial-state variances P0 ~ InvGamma(a,b)
    a_P0_alpha: float = 2.0; b_P0_alpha: float = 1.0
    a_P0_beta:  float = 2.0; b_P0_beta:  float = 1.0
    a_P0_gamma: float = 2.0; b_P0_gamma: float = 1.0

    # Half-Cauchy scales (process SDs)
    hc_scale_alpha: float = 0.5
    hc_scale_beta:  float = 0.5
    hc_scale_gamma: float = 0.5

@dataclass
class SamplerConfig:
    n_iter: int = 6000
    burn: int = 2000
    thin: int = 1
    random_seed: Optional[int] = 7
    progress: bool = True
    progress_every: int = 0   # 0 → ~2% cadence

@dataclass
class SelectionConfig:
    enabled: bool = True
    # how often to run a Shively search inside MCMC (set 0 to disable in-chain flips)
    search_every: int = 0
    # how many parameter draws to use when scoring (posterior predictive averaging)
    score_draws: int = 32
    # prior on models: dict like {("dyn","dyn","dyn"): log_prior, ...}, else uniform
    log_model_priors: Optional[Dict[Tuple[str,str,str], float]] = None
    # greedy flip if MAP model beats current by this many nats (≈ log BF)
    adopt_thresh: float = 2.0

# =============================================================================
# Structural DLM + conjugate Gibbs (σ² IG, P0 IG, Half-Cauchy via IG mixture) + FFBS
# =============================================================================
class DLMGibbsConjugate:
    """
    Gaussian structural DLM with:
      • FFBS for latent states
      • Conjugate Gibbs for σ², m0, P0
      • Half-Cauchy on process SDs via inverse-gamma mixtures (pure Gibbs)
      • Integrated Shively-style model selection (posterior predictive scores)

    State layout (dynamic parts): [alpha] [beta] [g1 ... g_{p-1}]
    Seasonal is NEWEST-FIRST; observation loads g1.
    """
    # --------------------------- construction --------------------------- #
    def __init__(
        self,
        y: np.ndarray,
        period: int,
        level_mode: str = "dynamic",
        trend_mode: str = "dynamic",
        seasonal_mode: str = "dynamic",
        # initial values
        m0_alpha_init: float = 0.0, P0_alpha_init: float = 1.0,
        m0_beta_init:  float = 0.0, P0_beta_init:  float = 1.0,
        m0_gamma_init: Optional[Sequence[float]] = None,  # len p-1 newest-first
        P0_gamma_init: float = 1.0,
        sigma2_init: float = 1.0,
        s_alpha_init: float = 1e-2,
        s_beta_init:  float = 1e-3,
        s_gamma_init: float = 1e-3,
        priors: Priors = Priors(),
        cfg: SamplerConfig = SamplerConfig(),
        sel: SelectionConfig = SelectionConfig(),
    ):
        # Data
        self.y = np.asarray(y, float); self.T = int(self.y.size)
        self.period = int(period); assert self.period >= 2

        # Modes
        ok = {"dynamic","deterministic","none"}
        if {level_mode,trend_mode,seasonal_mode} - ok:
            raise ValueError("invalid mode")
        if trend_mode=="dynamic" and level_mode!="dynamic":
            raise ValueError("dynamic trend requires dynamic level")
        self.level_mode, self.trend_mode, self.seasonal_mode = level_mode, trend_mode, seasonal_mode

        # Priors / cfg
        self.priors, self.cfg, self.sel = priors, cfg, sel
        if cfg.random_seed is not None: np.random.seed(cfg.random_seed)

        # Dynamic layout
        layout: List[str] = []
        if self.level_mode == "dynamic": layout.append("alpha")
        if self.trend_mode == "dynamic": layout.append("beta")
        if self.seasonal_mode == "dynamic": layout.extend([f"g{k}" for k in range(1, self.period)])
        self._layout = layout; self.dim = len(layout)

        self.idx_alpha = layout.index("alpha") if "alpha" in layout else None
        self.idx_beta  = layout.index("beta")  if "beta"  in layout else None
        if self.seasonal_mode == "dynamic":
            self.idx_g_start = layout.index("g1")
            self.idx_g_end   = self.idx_g_start + (self.period-2)
        else:
            self.idx_g_start = self.idx_g_end = None

        # Parameters
        self.sigma2 = float(sigma2_init)
        self.s_alpha = float(s_alpha_init) if self.idx_alpha is not None else 0.0
        self.s_beta  = float(s_beta_init)  if self.idx_beta  is not None else 0.0
        self.s_gamma = float(s_gamma_init) if self.seasonal_mode == "dynamic" else 0.0

        # Aux (Half-Cauchy mixtures) a_k ~ InvGamma(1/2, 1/A^2)
        self.a_alpha = 1.0; self.a_beta = 1.0; self.a_gamma = 1.0

        # Initial m0 and P0
        self.m0_alpha = float(m0_alpha_init) if self.idx_alpha is not None else 0.0
        self.P0_alpha = float(P0_alpha_init) if self.idx_alpha is not None else 0.0
        self.m0_beta  = float(m0_beta_init)  if self.idx_beta  is not None else 0.0
        self.P0_beta  = float(P0_beta_init)  if self.idx_beta  is not None else 0.0
        if self.seasonal_mode == "dynamic":
            if m0_gamma_init is None:
                self.m0_gamma = np.zeros(self.period-1, float)
            else:
                g = np.asarray(m0_gamma_init, float)
                if g.size != self.period-1: raise ValueError("m0_gamma_init length must be p-1")
                self.m0_gamma = g
            self.P0_gamma = float(P0_gamma_init)
        else:
            self.m0_gamma = None; self.P0_gamma = 0.0

        # Deterministic components (outside state)
        if self.level_mode == "deterministic":
            self.m0_alpha = float(self.priors.m_m0_alpha)
        if self.trend_mode == "deterministic":
            self.m0_beta  = float(self.priors.m_m0_beta)
        if self.seasonal_mode == "deterministic":
            base = np.zeros(self.period - 1) if self.priors.m_m0_gamma is None else np.asarray(self.priors.m_m0_gamma,float)
            if base.size != self.period-1: raise ValueError("priors.m_m0_gamma must be length p-1")
            self.m0_gamma = np.r_[base, -base.sum()].astype(float)

        # Latent path
        self.x = np.zeros((self.T+1, self.dim), float)
        if self.dim>0:
            m0_vec, P0_diag = self._current_m0_P0()
            self.x[0] = np.random.multivariate_normal(m0_vec, np.diag(P0_diag)+1e-10*np.eye(self.dim))
            self._propagate_initial_path(Q_init=np.full(self.dim, 1e-6))

        # storage
        self.keep: Dict[str,np.ndarray] = {}

        # init scale print
        if self.cfg.progress:
            y = self.y
            sd1 = _robust_sd(np.diff(y)) if y.size>=2 else 0.0
            sd2 = _robust_sd(np.diff(y,2)) if y.size>=3 else 0.0
            print(f"[init] sd1≈{sd1:.4g} sd2≈{sd2:.4g}")

    # ----------------------------- matrices ----------------------------- #
    def _H(self) -> np.ndarray:
        if self.dim==0: return np.zeros((1,0))
        h = np.zeros(self.dim); 
        if self.idx_alpha is not None: h[self.idx_alpha]=1.0
        if self.seasonal_mode=="dynamic": h[self.idx_g_start]=1.0
        return h.reshape(1,-1)

    def _A(self) -> np.ndarray:
        if self.dim==0: return np.zeros((0,0))
        A = np.eye(self.dim)
        if (self.idx_alpha is not None) and (self.idx_beta is not None):
            A[self.idx_alpha, self.idx_beta] = 1.0
        if self.seasonal_mode=="dynamic":
            gs, ge = self.idx_g_start, self.idx_g_end
            K = ge-gs+1
            A[gs, gs:ge+1] = -1.0
            A[gs+1:ge+1, gs:ge] = np.eye(K-1)
            A[gs+1:ge+1, ge] = 0.0
        return A

    def _u(self) -> np.ndarray:
        if self.dim==0: return np.zeros(0)
        u = np.zeros(self.dim)
        if (self.idx_alpha is not None) and (self.trend_mode=="deterministic"):
            u[self.idx_alpha] = float(self.m0_beta)
        return u

    def _Q(self) -> np.ndarray:
        if self.dim==0: return np.zeros((0,0))
        Q = np.zeros((self.dim,self.dim))
        if self.idx_alpha is not None and self.s_alpha>0: Q[self.idx_alpha,self.idx_alpha]=self.s_alpha**2
        if self.idx_beta  is not None and self.s_beta >0: Q[self.idx_beta, self.idx_beta ]=self.s_beta**2
        if self.seasonal_mode=="dynamic" and self.s_gamma>0: Q[self.idx_g_start,self.idx_g_start]=self.s_gamma**2
        return Q

    def _mu_det(self, t:int) -> float:
        out=0.0
        if self.level_mode=="deterministic": out += self.m0_alpha
        if (self.trend_mode=="deterministic") and (self.idx_alpha is None): out += self.m0_beta * t
        if self.seasonal_mode=="deterministic": out += float(self.m0_gamma[t % self.period])
        return out

    def _current_m0_P0(self) -> Tuple[np.ndarray,np.ndarray]:
        m0, P0 = [], []
        if self.idx_alpha is not None: m0.append(self.m0_alpha); P0.append(self.P0_alpha)
        if self.idx_beta  is not None: m0.append(self.m0_beta ); P0.append(self.P0_beta)
        if self.seasonal_mode=="dynamic":
            m0.extend(list(self.m0_gamma)); P0.extend([self.P0_gamma]*(self.period-1))
        return np.asarray(m0,float), np.asarray(P0,float)

    # ------------------------- FFBS ------------------------- #
    def _ffbs(self) -> np.ndarray:
        if self.dim==0: return self.x.copy()
        H,A,Q,R = self._H(), self._A(), self._Q(), float(self.sigma2)
        m0_vec, P0_diag = self._current_m0_P0()
        m = np.zeros((self.T+1,self.dim)); C = np.zeros((self.T+1,self.dim,self.dim))
        a = np.zeros((self.T+1,self.dim)); Rm = np.zeros((self.T+1,self.dim,self.dim))
        m[0]=m0_vec; C[0]=np.diag(P0_diag)+1e-12*np.eye(self.dim)
        u = self._u()

        # forward filter
        for t in range(1,self.T+1):
            a[t]  = A @ m[t-1] + u
            Rm[t] = A @ C[t-1] @ A.T + Q
            Rm[t] = 0.5*(Rm[t]+Rm[t].T)+1e-12*np.eye(self.dim)

            resid = float(self.y[t-1] - self._mu_det(t-1))
            S = float(H @ Rm[t] @ H.T + R)
            if not np.isfinite(S) or S<=0: S = float(H @ (Rm[t]+1e-10*np.eye(self.dim)) @ H.T + R)
            K = (Rm[t] @ H.T)/S
            v = resid - float(H @ a[t])
            m[t] = a[t] + (K.flatten()*v)
            C[t] = Rm[t] - K @ (H @ Rm[t])
            C[t] = 0.5*(C[t]+C[t].T)+1e-12*np.eye(self.dim)

        # backward sample
        x = np.zeros_like(self.x)
        x[self.T] = np.random.multivariate_normal(m[self.T], C[self.T])
        for t in range(self.T-1,-1,-1):
            J = C[t] @ A.T
            J = J @ _spd_solve(Rm[t+1], np.eye(self.dim))
            mean = m[t] + J @ (x[t+1] - (A @ m[t] + u))
            cov  = C[t] - J @ Rm[t+1] @ J.T
            cov  = 0.5*(cov+cov.T)
            lam = float(np.minimum(0.0, np.linalg.eigvalsh(cov).min()))
            if lam < 1e-14: cov += (1e-12 - lam)*np.eye(self.dim)
            x[t] = np.random.multivariate_normal(mean, cov)
        return x

    def _propagate_initial_path(self, Q_init: np.ndarray) -> None:
        if self.dim==0: return
        A,u = self._A(), self._u()
        for t in range(1,self.T+1):
            self.x[t] = A @ self.x[t-1] + u + np.random.normal(0.0, np.sqrt(Q_init), size=self.dim)

    def _mu_vec(self) -> np.ndarray:
        H = self._H(); mu = np.zeros(self.T)
        for t in range(1,self.T+1):
            dyn = float(H @ self.x[t]) if self.dim>0 else 0.0
            mu[t-1] = self._mu_det(t-1) + dyn
        return mu

    # ------------------ σ² | rest (IG via Gamma on precision) ------------------ #
    def update_sigma2(self) -> None:
        e = self.y - self._mu_vec()
        a = self.priors.a_sigma + 0.5*self.T
        b = self.priors.b_sigma + 0.5*float(e@e)
        tau = np.random.gamma(shape=a, scale=1.0/b)   # precision
        self.sigma2 = 1.0 / max(tau, 1e-300)

    # ------------- innovation SS for Q updates ------------- #
    def _innovation_ss_alpha(self) -> Tuple[float,int]:
        if self.idx_alpha is None: return 0.0,0
        ss = 0.0
        for t in range(1,self.T+1):
            drift = 0.0
            if self.idx_beta is not None: drift = self.x[t-1,self.idx_beta]
            elif self.trend_mode=="deterministic": drift = float(self.m0_beta)
            mean = self.x[t-1,self.idx_alpha] + drift
            ss += (self.x[t,self.idx_alpha] - mean)**2
        return float(ss), self.T

    def _innovation_ss_beta(self) -> Tuple[float,int]:
        if self.idx_beta is None: return 0.0,0
        d = self.x[1:,self.idx_beta] - self.x[:-1,self.idx_beta]
        return float(np.sum(d*d)), self.T

    def _innovation_ss_gamma(self) -> Tuple[float,int]:
        if self.seasonal_mode!="dynamic": return 0.0,0
        gs, ge = self.idx_g_start, self.idx_g_end
        ss=0.0
        for t in range(1,self.T+1):
            prev = self.x[t-1, gs:ge+1]
            mean_new_first = -float(np.sum(prev))
            ss += (self.x[t,gs] - mean_new_first)**2
        return float(ss), self.T

    # ------------------ Half-Cauchy via IG mixture (pure Gibbs) ------------------ #
    @staticmethod
    def _rinvgamma(shape: float, scale: float) -> float:
        return 1.0 / np.random.gamma(shape, 1.0/scale)

    def update_process_Q_halfcauchy(self) -> None:
        # α
        if self.idx_alpha is not None:
            SS, T_eff = self._innovation_ss_alpha(); A = float(self.priors.hc_scale_alpha)
            Q_shape = 0.5*T_eff + 0.5; Q_scale = 0.5*SS + 1.0/max(self.a_alpha,1e-300)
            Q_alpha = self._rinvgamma(Q_shape, Q_scale); self.s_alpha = math.sqrt(max(Q_alpha,0.0))
            a_shape = 1.0; a_scale = (1.0/(A*A)) + (1.0/max(Q_alpha,1e-300))
            self.a_alpha = self._rinvgamma(a_shape, a_scale)

        # β
        if self.idx_beta is not None:
            SS, T_eff = self._innovation_ss_beta(); A = float(self.priors.hc_scale_beta)
            Q_shape = 0.5*T_eff + 0.5; Q_scale = 0.5*SS + 1.0/max(self.a_beta,1e-300)
            Q_beta = self._rinvgamma(Q_shape, Q_scale); self.s_beta = math.sqrt(max(Q_beta,0.0))
            a_shape = 1.0; a_scale = (1.0/(A*A)) + (1.0/max(Q_beta,1e-300))
            self.a_beta = self._rinvgamma(a_shape, a_scale)

        # γ
        if self.seasonal_mode=="dynamic":
            SS, T_eff = self._innovation_ss_gamma(); A = float(self.priors.hc_scale_gamma)
            Q_shape = 0.5*T_eff + 0.5; Q_scale = 0.5*SS + 1.0/max(self.a_gamma,1e-300)
            Q_gamma = self._rinvgamma(Q_shape, Q_scale); self.s_gamma = math.sqrt(max(Q_gamma,0.0))
            a_shape = 1.0; a_scale = (1.0/(A*A)) + (1.0/max(Q_gamma,1e-300))
            self.a_gamma = self._rinvgamma(a_shape, a_scale)

    # ------------------ m0 and P0 ------------------ #
    @staticmethod
    def _gibbs_m0_scalar(x0: float, m_prior: float, s_prior: float, P0: float) -> float:
        prec = 1.0/(s_prior**2) + 1.0/max(P0,1e-18)
        var = 1.0/prec
        mean = var * (m_prior/(s_prior**2) + x0/max(P0,1e-18))
        return float(np.random.normal(mean, math.sqrt(var)))

    def update_m0(self) -> None:
        if self.dim==0: return
        pos=0
        if self.idx_alpha is not None:
            self.m0_alpha = self._gibbs_m0_scalar(float(self.x[0,pos]), self.priors.m_m0_alpha, self.priors.s_m0_alpha, self.P0_alpha); pos+=1
        if self.idx_beta  is not None:
            self.m0_beta  = self._gibbs_m0_scalar(float(self.x[0,pos]), self.priors.m_m0_beta , self.priors.s_m0_beta , self.P0_beta ); pos+=1
        if self.seasonal_mode=="dynamic":
            m_prior = (np.zeros(self.period-1) if self.priors.m_m0_gamma is None else np.asarray(self.priors.m_m0_gamma,float))
            if m_prior.size != self.period-1: raise ValueError("priors.m_m0_gamma length p-1")
            s = float(self.priors.s_m0_gamma)
            for k in range(self.period-1):
                self.m0_gamma[k] = self._gibbs_m0_scalar(float(self.x[0,pos+k]), float(m_prior[k]), s, self.P0_gamma)

    def update_P0(self) -> None:
        if self.dim==0: return
        pos=0
        if self.idx_alpha is not None:
            a = self.priors.a_P0_alpha + 0.5
            b = self.priors.b_P0_alpha + 0.5*(float(self.x[0,pos]) - self.m0_alpha)**2
            self.P0_alpha = 1.0/np.random.gamma(a, 1.0/b); pos+=1
        if self.idx_beta is not None:
            a = self.priors.a_P0_beta + 0.5
            b = self.priors.b_P0_beta + 0.5*(float(self.x[0,pos]) - self.m0_beta)**2
            self.P0_beta = 1.0/np.random.gamma(a, 1.0/b); pos+=1
        if self.seasonal_mode=="dynamic":
            diffsq = 0.0
            for k in range(self.period-1): diffsq += (float(self.x[0,pos+k]) - float(self.m0_gamma[k]))**2
            a = self.priors.a_P0_gamma + 0.5*(self.period-1)
            b = self.priors.b_P0_gamma + 0.5*diffsq
            self.P0_gamma = 1.0/np.random.gamma(a, 1.0/b)

    def update_deterministic_params(self) -> None:
        # level μ0
        if self.level_mode=="deterministic":
            r = self.y.copy()
            if self.dim>0:
                H = self._H()
                for t in range(1,self.T+1): r[t-1] -= float(H @ self.x[t])
            if self.seasonal_mode=="deterministic":
                r -= self.m0_gamma[np.arange(self.T)%self.period]
            s2 = float(self.sigma2)
            m0,s0 = float(self.priors.m_m0_alpha), float(self.priors.s_m0_alpha)
            prec = self.T/s2 + 1.0/(s0**2)
            mean = ((r.sum()/s2) + m0/(s0**2))/prec
            self.m0_alpha = float(np.random.normal(mean, math.sqrt(1.0/prec)))
        # deterministic trend
        if self.trend_mode=="deterministic":
            if self.idx_alpha is not None:
                d = self.x[1:,self.idx_alpha] - self.x[:-1,self.idx_alpha]
                s2 = float(self.s_alpha**2) if self.s_alpha>0 else 1e-12
                m0,s0 = float(self.priors.m_m0_beta), float(self.priors.s_m0_beta)
                prec = (self.T/s2) + 1.0/(s0**2)
                mean = ((float(np.sum(d))/s2) + m0/(s0**2))/prec
                self.m0_beta = float(np.random.normal(mean, math.sqrt(1.0/prec)))
            else:
                t = np.arange(self.T, dtype=float)
                r = self.y.copy()
                if self.dim>0:
                    H=self._H()
                    for k in range(1,self.T+1): r[k-1] -= float(H @ self.x[k])
                if self.level_mode=="deterministic": r -= self.m0_alpha
                if self.seasonal_mode=="deterministic": r -= self.m0_gamma[np.arange(self.T)%self.period]
                m0,s0 = float(self.priors.m_m0_beta), float(self.priors.s_m0_beta)
                sig2 = float(self.sigma2)
                Prec = (t@t)/sig2 + 1.0/(s0**2)
                mean = ((t@r)/sig2 + m0/(s0**2))/Prec
                self.m0_beta = float(np.random.normal(mean, math.sqrt(1.0/Prec)))
        # deterministic seasonal vector (sum-to-zero)
        if self.seasonal_mode=="deterministic":
            if not hasattr(self, "_Z_season"):
                midx = np.arange(self.T)%self.period
                K = self.period-1
                Z = np.zeros((self.T,K))
                for k in range(K):
                    Z[:,k] = (midx==k).astype(float) - (midx==K).astype(float)
                self._Z_season = Z
            r = self.y.copy()
            if self.dim>0:
                H=self._H()
                for t in range(1,self.T+1): r[t-1] -= float(H @ self.x[t])
            if self.level_mode=="deterministic": r -= self.m0_alpha
            if (self.idx_alpha is None) and (self.trend_mode=="deterministic"):
                r -= self.m0_beta*np.arange(self.T,dtype=float)
            K=self.period-1
            m_prior = (np.zeros(K) if self.priors.m_m0_gamma is None
                       else np.asarray(self.priors.m_m0_gamma,float).reshape(-1))
            s2 = float(self.priors.s_m0_gamma)**2
            Z = self._Z_season; sig2 = float(self.sigma2)
            Prec = (Z.T@Z)/sig2 + np.eye(K)/s2
            b = (Z.T@r)/sig2 + m_prior/s2
            mu = np.linalg.solve(Prec,b)
            L = np.linalg.cholesky(Prec)
            theta = mu + np.linalg.solve(L.T, np.random.randn(K))
            self.m0_gamma = np.r_[theta, -theta.sum()]

    # ------------------- selection helper: KF predictive log score ------------------- #
    def _kf_logscore(self, modes: Tuple[str,str,str], sigma2: float, s_alpha: float, s_beta: float, s_gamma: float) -> float:
        """
        One-step-ahead predictive log-likelihood sum for given block modes and variances.
        Deterministic pieces use current m0_*; dynamic uses same initial P0 and m0_*.
        """
        lvl, trd, seas = modes
        # build matrices for candidate
        idx_alpha = (lvl=="dynamic")
        idx_beta  = (trd=="dynamic")
        idx_seas  = (seas=="dynamic")
        dim = (1 if idx_alpha else 0) + (1 if idx_beta else 0) + ((self.period-1) if idx_seas else 0)
        if dim==0:
            mu_det = np.array([ ( (self.priors.m_m0_alpha if lvl=="deterministic" else 0.0)
                                 + ((self.priors.m_m0_beta*t) if (trd=="deterministic" and not idx_alpha) else 0.0)
                                 + (self.m0_gamma[t%self.period] if seas=="deterministic" else 0.0) )
                               for t in range(self.T)], float)
            e = self.y - mu_det
            return float(-0.5*np.sum(np.log(2*np.pi*sigma2) + (e*e)/sigma2))
        # index maps
        L=[]; 
        if idx_alpha: L.append("alpha")
        if idx_beta:  L.append("beta")
        if idx_seas:  L.extend([f"g{k}" for k in range(1,self.period)])
        ia = L.index("alpha") if "alpha" in L else None
        ib = L.index("beta")  if "beta"  in L else None
        if idx_seas: igs = L.index("g1"); ige = igs+(self.period-2)
        # matrices
        H = np.zeros(dim); 
        if ia is not None: H[ia]=1.0
        if idx_seas: H[igs]=1.0
        H = H.reshape(1,-1)
        A = np.eye(dim)
        if (ia is not None) and (ib is not None): A[ia,ib]=1.0
        if idx_seas:
            K = (self.period-1)
            A[igs, igs:igs+K] = -1.0
            if K>1:
                A[igs+1:igs+K, igs:igs+K-1] = np.eye(K-1)
                A[igs+1:igs+K, igs+K-1] = 0.0
        Q = np.zeros((dim,dim))
        if ia is not None and s_alpha>0: Q[ia,ia]=s_alpha**2
        if ib is not None and s_beta >0: Q[ib,ib ]=s_beta**2
        if idx_seas and s_gamma>0: Q[igs,igs]=s_gamma**2

        # initial moments: reuse current m0/P0 for matching blocks
        m0=[]; P0=[]
        if ia is not None: m0.append(self.m0_alpha); P0.append(self.P0_alpha)
        if ib is not None: m0.append(self.m0_beta ); P0.append(self.P0_beta)
        if idx_seas: m0.extend(list(self.m0_gamma)); P0.extend([self.P0_gamma]*(self.period-1))
        m = np.asarray(m0,float) if m0 else np.zeros(0)
        C = np.diag(np.asarray(P0,float)) if P0 else np.zeros((0,0))
        u = np.zeros(dim); 
        if (ia is not None) and (trd=="deterministic"): u[ia] = float(self.m0_beta)

        logscore=0.0
        for t in range(self.T):
            # predict
            a = A @ m + u
            Rm = A @ C @ A.T + Q
            # deterministic mean
            mu_det = 0.0
            if lvl=="deterministic": mu_det += self.m0_alpha
            if (trd=="deterministic") and (ia is None): mu_det += self.m0_beta * t
            if seas=="deterministic": mu_det += float(self.m0_gamma[t % self.period])
            # predict y
            yhat = mu_det + (float(H @ a) if dim>0 else 0.0)
            S = float(H @ Rm @ H.T + sigma2)
            v = float(self.y[t] - yhat)
            logscore += -0.5*(math.log(2*math.pi*S) + (v*v)/S)
            # update
            if dim>0:
                K = (Rm @ H.T)/S
                m = a + (K.flatten()*v)
                C = Rm - K @ (H @ Rm)
                C = 0.5*(C+C.T)+1e-12*np.eye(dim)
        return float(logscore)

    # ------------------- Shively-style model selection wrapper ------------------- #
    def _candidate_models(self) -> List[Tuple[str,str,str]]:
        # level in {dynamic, deterministic}, trend in {dynamic, deterministic, none}, seasonal in {dynamic, deterministic, none}
        c = []
        for lvl in ("dynamic","deterministic"):
            for trd in ("dynamic","deterministic","none"):
                if trd=="dynamic" and lvl!="dynamic": continue
                for seas in ("dynamic","deterministic","none"):
                    c.append((lvl,trd,seas))
        return c

    def shively_search(self, param_draws: Optional[int]=None, log_model_priors: Optional[Dict[Tuple[str,str,str],float]]=None) -> Dict[str,object]:
        """
        Compute posterior model probabilities by averaging KF one-step-ahead log scores
        across a small bundle of posterior draws of (σ, s_alpha, s_beta, s_gamma) pulled
        from self.keep (or using current state if keep is empty).
        """
        models = self._candidate_models()
        if param_draws is None: param_draws = self.sel.score_draws
        # Collect draws
        draws = []
        if self.keep:
            N = next(iter(self.keep.values())).shape[0]
            idx = np.random.choice(N, size=min(param_draws,N), replace=False)
            for i in idx:
                sig = float(self.keep["sigma"][i]) if "sigma" in self.keep else math.sqrt(self.sigma2)
                sa  = float(np.sqrt(self.keep["Q_alpha"][i])) if "Q_alpha" in self.keep else float(self.s_alpha)
                sb  = float(np.sqrt(self.keep["Q_beta"][i]))  if "Q_beta"  in self.keep else float(self.s_beta )
                sg  = float(np.sqrt(self.keep["Q_gamma"][i])) if "Q_gamma" in self.keep else float(self.s_gamma)
                draws.append((sig,sa,sb,sg))
        else:
            draws = [(math.sqrt(self.sigma2), self.s_alpha, self.s_beta, self.s_gamma)]
        # score
        log_scores = np.zeros(len(models))
        for j, M in enumerate(models):
            s=0.0
            for (sig,sa,sb,sg) in draws:
                s += self._kf_logscore(M, sigma2=sig**2, s_alpha=sa, s_beta=sb, s_gamma=sg)
            log_scores[j] = s / len(draws)
        # add model priors (log scale)
        if log_model_priors:
            logp = np.array([log_model_priors.get(M, 0.0) for M in models], float)
            log_scores = log_scores + logp
        # normalize
        m = float(log_scores.max())
        w = np.exp(log_scores - m); probs = w/float(w.sum())
        return {"models": models, "log_scores": log_scores, "post_probs": probs}

    def maybe_adopt_map_model(self, res: Dict[str,object], thresh: float = 2.0) -> None:
        """Greedy switch to MAP model if it beats current by `thresh` nats."""
        models = res["models"]; log_scores = res["log_scores"]
        cur = (self.level_mode, self.trend_mode, self.seasonal_mode)
        cur_idx = models.index(cur)
        map_idx = int(np.argmax(log_scores))
        gain = float(log_scores[map_idx] - log_scores[cur_idx])
        if gain >= thresh:
            self.level_mode, self.trend_mode, self.seasonal_mode = models[map_idx]
            # rebuild layout + indices
            layout: List[str] = []
            if self.level_mode=="dynamic": layout.append("alpha")
            if self.trend_mode=="dynamic": layout.append("beta")
            if self.seasonal_mode=="dynamic": layout.extend([f"g{k}" for k in range(1,self.period)])
            self._layout = layout; self.dim = len(layout)
            self.idx_alpha = layout.index("alpha") if "alpha" in layout else None
            self.idx_beta  = layout.index("beta")  if "beta"  in layout else None
            if self.seasonal_mode=="dynamic":
                self.idx_g_start = layout.index("g1"); self.idx_g_end = self.idx_g_start + (self.period-2)
            else:
                self.idx_g_start = self.idx_g_end = None
            # shrink process sds if a block turned deterministic
            if self.level_mode!="dynamic": self.s_alpha = 0.0
            if self.trend_mode!="dynamic": self.s_beta  = 0.0
            if self.seasonal_mode!="dynamic": self.s_gamma = 0.0

    # ------------------- progress line ------------------- #
    def _progress_line(self, it: int) -> str:
        parts = [f"[it {it+1}/{self.cfg.n_iter}]"]
        parts.append(f"σ={math.sqrt(self.sigma2):.3f}")
        if self.idx_alpha is not None: parts.append(f"Qα={self.s_alpha**2:.4g}")
        if self.idx_beta  is not None: parts.append(f"Qβ={self.s_beta**2:.4g}")
        if self.seasonal_mode=="dynamic": parts.append(f"Qγ={self.s_gamma**2:.4g}")
        if self.level_mode!="none":
            parts.append(f"m0α={self.m0_alpha:.4g} P0α={(self.P0_alpha if self.idx_alpha is not None else 0.0):.4g}")
        if self.trend_mode!="none":
            parts.append(f"m0β={self.m0_beta:.4g} P0β={(self.P0_beta if self.idx_beta is not None else 0.0):.4g}")
        if self.seasonal_mode!="none":
            g = self.m0_gamma if self.seasonal_mode=="deterministic" else self.m0_gamma
            gtxt = "["+", ".join(f"{z:.4g}" for z in (g if g is not None else [])) + "]"
            parts.append(f"m0γ={gtxt} P0γ={(self.P0_gamma if self.seasonal_mode=='dynamic' else 0.0):.4g}")
        return " | ".join(parts)

    # --------------------------------- MCMC --------------------------------- #
    def run(self) -> Dict[str,np.ndarray]:
        cfg = self.cfg
        save_iters = list(range(cfg.burn, cfg.n_iter, cfg.thin))
        n_kept, keep_idx = len(save_iters), 0
        # allocate
        self.keep = {"sigma": np.zeros(n_kept), "mu": np.zeros((n_kept, self.T))}
        if self.idx_alpha is not None: self.keep.update({"Q_alpha": np.zeros(n_kept), "m0_alpha": np.zeros(n_kept), "P0_alpha": np.zeros(n_kept)})
        if self.idx_beta  is not None: self.keep.update({"Q_beta":  np.zeros(n_kept), "m0_beta":  np.zeros(n_kept), "P0_beta":  np.zeros(n_kept)})
        if self.seasonal_mode=="dynamic":
            self.keep.update({"Q_gamma": np.zeros(n_kept),
                              "m0_gamma": np.zeros((n_kept, self.period-1)),
                              "P0_gamma": np.zeros(n_kept),
                              "x": np.zeros((n_kept, self.T, self.dim))})
        elif self.dim>0:
            self.keep["x"] = np.zeros((n_kept, self.T, self.dim))
        if self.level_mode=="deterministic":  self.keep["m0_alpha"] = np.zeros(n_kept)
        if self.trend_mode=="deterministic":  self.keep["m0_beta"]  = np.zeros(n_kept)
        if self.seasonal_mode=="deterministic": self.keep["m0_gamma"] = np.zeros((n_kept, self.period))

        print_every = cfg.progress_every if cfg.progress_every>0 else max(1, cfg.n_iter//50)

        for it in range(cfg.n_iter):
            # 1) FFBS
            if self.dim>0: self.x = self._ffbs()

            # 2) Q (Half-Cauchy mixtures)
            if self.dim>0: self.update_process_Q_halfcauchy()

            # 3) m0, 4) P0
            if self.dim>0:
                self.update_m0()
                self.update_P0()

            # 5) deterministic params
            self.update_deterministic_params()

            # 6) σ²
            self.update_sigma2()

            # 7) optional in-chain Shively search / greedy adopt
            if self.sel.enabled and self.sel.search_every and ((it+1)%self.sel.search_every==0):
                res = self.shively_search(param_draws=self.sel.score_draws, log_model_priors=self.sel.log_model_priors)
                self.maybe_adopt_map_model(res, thresh=self.sel.adopt_thresh)

            # progress
            if self.cfg.progress and ((it+1)%print_every==0 or it==cfg.n_iter-1):
                print(self._progress_line(it))

            # save
            if it in save_iters:
                mu = self._mu_vec()
                self.keep["mu"][keep_idx,:] = mu
                self.keep["sigma"][keep_idx] = math.sqrt(self.sigma2)
                if self.idx_alpha is not None:
                    self.keep["Q_alpha"][keep_idx] = self.s_alpha**2
                    self.keep["m0_alpha"][keep_idx] = self.m0_alpha
                    self.keep["P0_alpha"][keep_idx] = self.P0_alpha
                if self.idx_beta is not None:
                    self.keep["Q_beta"][keep_idx] = self.s_beta**2
                    self.keep["m0_beta"][keep_idx] = self.m0_beta
                    self.keep["P0_beta"][keep_idx] = self.P0_beta
                if self.seasonal_mode=="dynamic":
                    self.keep["Q_gamma"][keep_idx] = self.s_gamma**2
                    self.keep["m0_gamma"][keep_idx,:] = self.m0_gamma
                    self.keep["P0_gamma"][keep_idx] = self.P0_gamma
                if "x" in self.keep and self.dim>0:
                    self.keep["x"][keep_idx,:,:] = self.x[1:self.T+1,:]
                keep_idx += 1

        return self.keep

    # ------------------------------- I/O ------------------------------- #
    def save_posterior(self, out_npz_path: str, extra_meta: Optional[dict]=None) -> None:
        os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)
        arrays = dict(self.keep); arrays["y"] = self.y.copy()
        if "x" not in arrays: arrays["x"] = np.zeros((0,0,0))
        np.savez_compressed(out_npz_path, **arrays)
        meta = {
            "T": int(self.T), "dim": int(self.dim), "period": int(self.period),
            "modes": {"level_mode": self.level_mode, "trend_mode": self.trend_mode, "seasonal_mode": self.seasonal_mode},
            "layout": list(self._layout),
            "cfg": asdict(self.cfg), "priors": asdict(self.priors), "selection": asdict(self.sel)
        }
        if extra_meta: meta.update(extra_meta)
        with open(out_npz_path.replace(".npz",".meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[save] Posterior -> {out_npz_path}")

# ------------------------- CLI / Example ------------------------------- #
if __name__ == "__main__":
    import argparse, time
    import matplotlib.pyplot as plt
    import numpy as np

    # --- args ---
    p = argparse.ArgumentParser("Gaussian DLM with Gibbs+FFBS and Shively-style model selection")
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--period", type=int, default=12)
    p.add_argument("--level-mode", choices=["dynamic","deterministic"], default="dynamic")
    p.add_argument("--trend-mode", choices=["dynamic","deterministic","none"], default="dynamic")
    p.add_argument("--seasonal-mode", choices=["dynamic","deterministic","none"], default="dynamic")

    p.add_argument("--sigma", type=float, default=1.5)
    p.add_argument("--q-level", type=float, default=1e-3)
    p.add_argument("--q-trend", type=float, default=2e-5)
    p.add_argument("--q-season", type=float, default=5e-5)

    p.add_argument("--sigma-init", type=float, default=1.0)
    p.add_argument("--s-alpha-init", type=float, default=1e-2)
    p.add_argument("--s-beta-init",  type=float, default=1e-3)
    p.add_argument("--s-gamma-init", type=float, default=1e-3)

    p.add_argument("--prior-a-sigma", type=float, default=2.0)
    p.add_argument("--prior-b-sigma", type=float, default=1.0)
    p.add_argument("--prior-a-P0-alpha", type=float, default=5.0)
    p.add_argument("--prior-b-P0-alpha", type=float, default=1.0)
    p.add_argument("--prior-a-P0-beta",  type=float, default=5.0)
    p.add_argument("--prior-b-P0-beta",  type=float, default=1.0)
    p.add_argument("--prior-a-P0-gamma", type=float, default=5.0)
    p.add_argument("--prior-b-P0-gamma", type=float, default=1.0)

    p.add_argument("--prior-m-m0-alpha", type=float, default=0.0)
    p.add_argument("--prior-s-m0-alpha", type=float, default=10.0)
    p.add_argument("--prior-m-m0-beta",  type=float, default=0.0)
    p.add_argument("--prior-s-m0-beta",  type=float, default=10.0)
    p.add_argument("--prior-s-m0-gamma", type=float, default=5.0)

    p.add_argument("--hc-scale-alpha", type=float, default=0.5)
    p.add_argument("--hc-scale-beta",  type=float, default=0.5)
    p.add_argument("--hc-scale-gamma", type=float, default=0.5)

    p.add_argument("--n-iter", type=int, default=8000)
    p.add_argument("--burn", type=int, default=4000)
    p.add_argument("--thin", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--progress", action="store_true")
    p.add_argument("--progress-every", type=int, default=1)

    # selection options
    p.add_argument("--sel-enabled", action="store_true")
    p.add_argument("--sel-search-every", type=int, default=0)
    p.add_argument("--sel-score-draws", type=int, default=32)
    p.add_argument("--sel-adopt-thresh", type=float, default=2.0)

    args = p.parse_args()
    np.random.seed(args.seed)

    # --- simulate simple STS y_t ---
    T = args.T; pper = args.period
    # true seasonal pattern (sum-to-zero over period)
    base = np.sin(2*np.pi*np.arange(pper)/pper); base -= base.mean()
    gtruth = np.tile(base, 1 + T//pper)[:T]
    alpha = np.zeros(T); beta = np.zeros(T)
    ql, qt, qg = args.q_level, args.q_trend, args.q_season
    for t in range(1,T):
        if args.level_mode=="dynamic": alpha[t] = alpha[t-1] + (beta[t-1] if args.trend_mode!="none" else 0.0) + np.random.normal(0,np.sqrt(ql))
        if args.trend_mode=="dynamic": beta[t]  = beta[t-1]  + np.random.normal(0,np.sqrt(qt))
    if args.seasonal_mode!="none": pass  # deterministic seasonal is handled via base in the mean below
    mu = alpha + gtruth + (0.0)  # + deterministic trend handled if needed
    y  = mu + np.random.normal(0, args.sigma, size=T)

    # --- priors and configs ---
    pri_gamma = None
    pri = Priors(
        a_sigma=args.prior_a_sigma, b_sigma=args.prior_b_sigma,
        m_m0_alpha=args.prior_m_m0_alpha, s_m0_alpha=args.prior_s_m0_alpha,
        m_m0_beta=args.prior_m_m0_beta, s_m0_beta=args.prior_s_m0_beta,
        m_m0_gamma=pri_gamma, s_m0_gamma=args.prior_s_m0_gamma,
        a_P0_alpha=args.prior_a_P0_alpha, b_P0_alpha=args.prior_b_P0_alpha,
        a_P0_beta=args.prior_a_P0_beta,   b_P0_beta=args.prior_b_P0_beta,
        a_P0_gamma=args.prior_a_P0_gamma, b_P0_gamma=args.prior_b_P0_gamma,
        hc_scale_alpha=args.hc_scale_alpha, hc_scale_beta=args.hc_scale_beta, hc_scale_gamma=args.hc_scale_gamma
    )
    cfg = SamplerConfig(n_iter=args.n_iter, burn=args.burn, thin=args.thin,
                        random_seed=args.seed, progress=args.progress, progress_every=args.progress_every)
    sel = SelectionConfig(enabled=bool(args.sel_enabled),
                          search_every=int(args.sel_search_every),
                          score_draws=int(args.sel_score_draws),
                          adopt_thresh=float(args.sel_adopt_thresh),
                          log_model_priors=None)

    # --- run ---
    sampler = DLMGibbsConjugate(
        y=y, period=pper,
        level_mode=args.level_mode, trend_mode=args.trend_mode, seasonal_mode=args.seasonal_mode,
        sigma2_init=args.sigma_init**2, s_alpha_init=args.s_alpha_init, s_beta_init=args.s_beta_init, s_gamma_init=args.s_gamma_init,
        priors=pri, cfg=cfg, sel=sel
    )
    t0 = time.time()
    post = sampler.run()
    print(f"\n[Run done in {time.time()-t0:.1f}s]")

    # offline Shively search (optional, using kept draws)
    if sel.enabled:
        res = sampler.shively_search()
        models = res["models"]; probs = res["post_probs"]
        print("\n[Model posterior probabilities]")
        order = np.argsort(-probs)
        for j in order[:8]:
            print(f"  {models[j]} : {probs[j]:.3f}")

    # quick plot
    if True:
        import matplotlib.pyplot as plt
        mu_hat = post["mu"].mean(axis=0)
        plt.figure(figsize=(10,4))
        plt.plot(y, color="k", lw=1, label="y")
        plt.plot(mu_hat, ls="--", lw=2, label="μ̂ (post mean)")
        plt.legend(); plt.tight_layout(); plt.show()
