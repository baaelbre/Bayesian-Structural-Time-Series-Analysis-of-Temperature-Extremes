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
    med = np.median(v)
    return float(np.median(np.abs(v - med)))

def _robust_sd(v: np.ndarray) -> float:
    return _mad(np.asarray(v, float)) / 1.4826 if v.size else 0.0

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

# ---------- Slice sampler on (a, b) for log-density f ----------
def _slice_sample_bounded(logf, x0: float, a: float, b: float, w: float = 1.0, m: int = 100, rng=np.random):
    x0 = float(np.clip(x0, a + 1e-18, b - 1e-18))
    fx = logf(x0)
    y = fx - np.log(1.0 / max(rng.uniform(), 1e-300))
    # initial bracket
    L = max(a, x0 - w * rng.uniform())
    R = min(b, L + w)
    j = int(m * rng.uniform()); k = (m - 1) - j
    while j > 0 and L > a and logf(L) > y:
        L = max(a, L - w); j -= 1
    while k > 0 and R < b and logf(R) > y:
        R = min(b, R + w); k -= 1
    for _ in range(100):
        x_new = rng.uniform(L, R)
        if logf(x_new) >= y:
            return float(np.clip(x_new, a + 1e-18, b - 1e-18))
        if x_new < x0:
            L = x_new
        else:
            R = x_new
    return float(np.clip(x0, a + 1e-18, b - 1e-18))

# ---------- Log-sum-exp ----------
def _logsumexp(vals: Sequence[float]) -> float:
    if len(vals) == 0: 
        return float("-inf")
    a = np.asarray(vals, float)
    m = np.max(a)
    return float(m + np.log(np.sum(np.exp(a - m))))

# ---------- Stable integral for slab evidence ∫_0^{zbar} q^{-T/2} exp(-SS/(2q)) dq ----------
# Use Gauss–Legendre on s in (0,1) with transform q = zbar * s^2 for more mass near 0.
_GLX, _GLW = np.polynomial.legendre.leggauss(64)  # 64-pt is robust and still fast
def _log_evidence_slab(SS: float, T_eff: int, zbar: float) -> float:
    if T_eff <= 0:
        # no increments -> degenerate (zero likelihood curvature); treat evidence ~ log(zbar)
        return math.log(max(zbar, 1e-300))
    if SS <= 0:
        # perfect fit -> integral dominated at q=zbar; approximate by finite integral mass
        # ∫_0^{zbar} q^{-T/2} dq = zbar^{1 - T/2} / (1 - T/2) if T != 2
        if abs(1 - 0.5*T_eff) > 1e-8:
            return math.log(max(zbar,1e-300)) * (1 - 0.5*T_eff)
        else:
            # T=2 => log divergence -> just return large number
            return 50.0
    x = 0.5 * (_GLX + 1.0)  # map [-1,1] -> [0,1]
    w = 0.5 * _GLW
    s = x
    q = zbar * s * s
    # integrand: q^{-T/2} exp(-SS/(2q)) * dq; dq = 2*zbar*s ds
    with np.errstate(divide='ignore', over='ignore'):
        log_integrand = (-0.5*T_eff)*np.log(q) - (SS/(2.0*q)) + np.log(2.0*zbar*s)
    logw = np.log(np.maximum(w, 1e-300))
    return _logsumexp(logw + log_integrand)

# =============================================================================
# Priors & Config — Shively-style spike & bounded-uniform slab
# =============================================================================

@dataclass
class PriorsShively:
    # Uniform slab bounds for process variances (zeta bars)
    zbar_alpha: float = 1.0
    zbar_beta:  float = 1.0
    zbar_gamma: float = 1.0
    # Inclusion probabilities
    pi_alpha: float = 0.5
    pi_beta:  float = 0.5
    pi_gamma: float = 0.5
    # Observation variance prior: sigma^2 ~ Uniform(0, sigma2_bar)
    sigma2_bar: float = 10.0

    # Static (deterministic) parameter priors (conjugate normals)
    m_m0_alpha: float = 0.0
    s_m0_alpha: float = 10.0
    m_m0_beta:  float = 0.0
    s_m0_beta:  float = 10.0
    m_m0_gamma: Optional[Sequence[float]] = None  # length p-1
    s_m0_gamma: float = 5.0

    # Dynamic initial-state priors for P0 (weak IG)
    a_P0_alpha: float = 5.0; b_P0_alpha: float = 1.0
    a_P0_beta:  float = 5.0; b_P0_beta:  float = 1.0
    a_P0_gamma: float = 5.0; b_P0_gamma: float = 1.0

@dataclass
class SamplerConfig:
    n_iter: int = 10000
    burn: int = 5000
    thin: int = 2
    random_seed: Optional[int] = 40
    progress: bool = True
    progress_every: int = 0  # 0 => ~2% of n_iter

# =============================================================================
# DLM Sampler — Shively spike & slab (bounded-uniform slab on Q, spike at 0)
# =============================================================================

class DLMGibbs_Shively:
    """
    y_t = μ_t + ε_t,  ε_t ~ N(0, σ²),  σ² ~ Uniform(0, σ̄²)

    Components (level α, slope β, seasonal g):
      δ_k = 1  → dynamic (random walk), Q_k ~ Uniform(0, ζ̄_k)
      δ_k = 0  → static, Q_k = 0, update θ_k conjugately; exclude k from FFBS

    Seasonal is the dummy-rotation with innovation on the first seasonal coord only when dynamic.
    """

    # --------------------------- Construction --------------------------- #
    def __init__(
        self,
        y: np.ndarray,
        period: int,
        sigma2_init: float = 1.0,
        # initial values
        m0_alpha_init: float = 0.0, P0_alpha_init: float = 0.25,
        m0_beta_init:  float = 0.0, P0_beta_init:  float = 0.05,
        m0_gamma_init: Optional[Sequence[float]] = None, P0_gamma_init: float = 0.25,
        # SSVS init
        Q_alpha_init: float = 1e-2, Q_beta_init: float = 1e-3, Q_gamma_init: float = 1e-3,
        delta_alpha_init: int = 1, delta_beta_init: int = 1, delta_gamma_init: int = 1,
        # optional static inits
        theta_alpha_init: float = 0.0, theta_beta_init: float = 0.0,
        theta_gamma_init: Optional[Sequence[float]] = None,  # length p-1
        priors: PriorsShively = PriorsShively(),
        cfg: SamplerConfig = SamplerConfig(),
    ):
        # Data
        self.y = np.asarray(y, float); self.T = int(self.y.size)
        self.period = int(period)
        if self.period < 2: raise ValueError("period must be >= 2")

        self.priors, self.cfg = priors, cfg
        if cfg.random_seed is not None: np.random.seed(cfg.random_seed)

        # layout
        self.idx_alpha, self.idx_beta = 0, 1
        self.idx_g_start = 2
        self.K_seas = self.period - 1
        self.dim_full = 2 + self.K_seas

        # Observation variance
        self.sigma2 = float(sigma2_init)

        # Dynamic initial-state params (used when δ=1)
        self.m0_alpha = float(m0_alpha_init); self.P0_alpha = float(P0_alpha_init)
        self.m0_beta  = float(m0_beta_init);  self.P0_beta  = float(P0_beta_init)
        if m0_gamma_init is None:
            self.m0_gamma = np.zeros(self.K_seas, float)
        else:
            g = np.asarray(m0_gamma_init, float).reshape(-1)
            if g.size != self.K_seas:
                raise ValueError("m0_gamma_init must have length p-1 (newest-first)")
            self.m0_gamma = g
        self.P0_gamma = float(P0_gamma_init)

        # Spike–slab: process variances and inclusion indicators
        self.Q_alpha = float(Q_alpha_init)
        self.Q_beta  = float(Q_beta_init)
        self.Q_gamma = float(Q_gamma_init)
        self.delta_alpha = int(delta_alpha_init)
        self.delta_beta  = int(delta_beta_init)
        self.delta_gamma = int(delta_gamma_init)
        self.pdelta_alpha = 0.5
        self.pdelta_beta  = 0.5
        self.pdelta_gamma = 0.5


        # Static parameters θ (used when δ=0)
        self.theta_alpha = float(theta_alpha_init)
        self.theta_beta  = float(theta_beta_init)
        if self.K_seas > 0:
            thg = np.zeros(self.K_seas, float) if theta_gamma_init is None else np.asarray(theta_gamma_init, float)
            if thg.size != self.K_seas:
                raise ValueError("theta_gamma_init must have length p-1 (newest-first)")
            self.theta_gamma = np.r_[thg, -np.sum(thg)]  # full p-vector with sum-to-zero
        else:
            self.theta_gamma = np.zeros(0, float)

        # Latent path container (dynamic coords only per iteration; allocate max and reuse)
        self.x = np.zeros((self.T + 1, self.dim_full), float)

        # Storage
        self.keep: Dict[str, np.ndarray] = {}

        if self.cfg.progress:
            y = self.y
            sd1 = _robust_sd(np.diff(y)) if y.size >= 2 else 0.0
            sd2 = _robust_sd(np.diff(y, n=2)) if y.size >= 3 else 0.0
            print(f"[init] scale proxies: sd1={sd1:.4g}, sd2={sd2:.4g}")

    # ----------------------------- Helpers ----------------------------- #

    def _dynamic_indices(self) -> List[int]:
        idx = []
        if self.delta_alpha == 1: idx.append(self.idx_alpha)
        if self.delta_beta  == 1: idx.append(self.idx_beta)
        if self.delta_gamma == 1 and self.K_seas > 0:
            idx.extend(list(range(self.idx_g_start, self.idx_g_start + self.K_seas)))
        return idx

    def _H_full(self) -> np.ndarray:
        h = np.zeros(self.dim_full, float)
        if self.delta_alpha == 1:
            h[self.idx_alpha] = 1.0
        if self.delta_gamma == 1 and self.K_seas > 0:
            h[self.idx_g_start] = 1.0
        return h.reshape(1, -1)

    def _A_full(self) -> np.ndarray:
        A = np.eye(self.dim_full)
        if self.delta_alpha == 1 and self.delta_beta == 1:
            A[self.idx_alpha, self.idx_beta] = 1.0
        if self.delta_gamma == 1 and self.K_seas > 0:
            gs = self.idx_g_start; ge = gs + self.K_seas - 1
            A[gs, gs:ge+1] = -1.0
            if self.K_seas > 1:
                A[gs+1:ge+1, gs:ge] = np.eye(self.K_seas - 1)
                A[gs+1:ge+1, ge] = 0.0
        return A

    def _u_full(self) -> np.ndarray:
        u = np.zeros(self.dim_full, float)
        if self.delta_alpha == 1 and self.delta_beta == 0:
            u[self.idx_alpha] = float(self.theta_beta)
        return u

    def _Q_full(self) -> np.ndarray:
        Q = np.zeros((self.dim_full, self.dim_full))
        if self.delta_alpha == 1 and self.Q_alpha > 0: Q[self.idx_alpha, self.idx_alpha] = self.Q_alpha
        if self.delta_beta  == 1 and self.Q_beta  > 0: Q[self.idx_beta,  self.idx_beta]  = self.Q_beta
        if self.delta_gamma == 1 and self.K_seas > 0 and self.Q_gamma > 0:
            Q[self.idx_g_start, self.idx_g_start] = self.Q_gamma
        return Q

    def _mu_vec(self) -> np.ndarray:
        mu = np.zeros(self.T, float)
        if self.delta_alpha == 1:
            mu += self.x[1:self.T+1, self.idx_alpha]
        else:
            mu += self.theta_alpha + self.theta_beta * np.arange(self.T, dtype=float)
        if self.K_seas > 0:
            if self.delta_gamma == 1:
                mu += self.x[1:self.T+1, self.idx_g_start]
            else:
                mu += self.theta_gamma[np.arange(self.T) % self.period]
        return mu

    # ------------------------- FFBS on dynamic subset ------------------------- #
    def _ffbs(self):
        dyn_idx = self._dynamic_indices()
        d = len(dyn_idx)
        if d == 0:
            self.x[:] = 0.0
            return

        H_full, A_full, Q_full, u_full = self._H_full(), self._A_full(), self._Q_full(), self._u_full()
        H = H_full[:, dyn_idx]
        A = A_full[np.ix_(dyn_idx, dyn_idx)]
        Q = Q_full[np.ix_(dyn_idx, dyn_idx)]
        u = u_full[dyn_idx]

        m0_list, P0_list = [], []
        for j in dyn_idx:
            if j == self.idx_alpha: m0_list.append(self.m0_alpha); P0_list.append(self.P0_alpha)
            elif j == self.idx_beta: m0_list.append(self.m0_beta); P0_list.append(self.P0_beta)
            else:
                off = j - self.idx_g_start
                m0_list.append(self.m0_gamma[off])
                P0_list.append(self.P0_gamma)
        m0 = np.asarray(m0_list, float)
        P0 = np.diag(np.asarray(P0_list, float)) + 1e-12*np.eye(d)

        m = np.zeros((self.T + 1, d))
        C = np.zeros((self.T + 1, d, d))
        a = np.zeros((self.T + 1, d))
        Rm = np.zeros((self.T + 1, d, d))
        m[0] = m0; C[0] = P0
        R = float(self.sigma2)

        det = np.zeros(self.T, float)
        if self.delta_alpha == 0:
            det += self.theta_alpha + self.theta_beta * np.arange(self.T, dtype=float)
        if self.K_seas > 0 and self.delta_gamma == 0:
            det += self.theta_gamma[np.arange(self.T) % self.period]

        for t in range(1, self.T + 1):
            a[t]  = A @ m[t - 1] + u
            Rm[t] = A @ C[t - 1] @ A.T + Q
            Rm[t] = 0.5*(Rm[t]+Rm[t].T) + 1e-12*np.eye(d)

            ytil = self.y[t - 1] - det[t - 1]
            S = float(H @ Rm[t] @ H.T + R)
            K = (Rm[t] @ H.T) / S
            v = ytil - float(H @ a[t])
            m[t] = a[t] + (K.flatten() * v)
            C[t] = Rm[t] - K @ (H @ Rm[t])
            C[t] = 0.5*(C[t]+C[t].T) + 1e-12*np.eye(d)

        xs = np.zeros((self.T + 1, d))
        xs[self.T] = np.random.multivariate_normal(m[self.T], C[self.T])
        for t in range(self.T - 1, -1, -1):
            J = C[t] @ A.T
            J = J @ _spd_solve(Rm[t + 1], np.eye(d))
            mean = m[t] + J @ (xs[t + 1] - a[t + 1])
            cov  = C[t] - J @ Rm[t + 1] @ J.T
            cov  = 0.5*(cov+cov.T)
            evmin = float(np.linalg.eigvalsh(cov).min())
            if evmin < 1e-12:
                cov += (1e-12 - evmin) * np.eye(d)
            xs[t] = np.random.multivariate_normal(mean, cov)

        self.x[:] = 0.0
        self.x[:, dyn_idx] = xs

    # ------------- Innovation sums of squares (for evidence & Q updates) -------------
    def _innovation_ss_alpha(self) -> Tuple[float, int]:
        if self.delta_alpha != 1: return 0.0, 0
        ss = 0.0
        for t in range(1, self.T + 1):
            drift = (self.x[t - 1, self.idx_beta] if self.delta_beta == 1 else self.theta_beta)
            mean = self.x[t - 1, self.idx_alpha] + drift
            ss += (self.x[t, self.idx_alpha] - mean) ** 2
        return float(ss), self.T

    def _innovation_ss_beta(self) -> Tuple[float, int]:
        if self.delta_beta != 1: return 0.0, 0
        d = self.x[1:, self.idx_beta] - self.x[:-1, self.idx_beta]
        return float(np.sum(d * d)), self.T

    def _innovation_ss_gamma(self) -> Tuple[float, int]:
        if self.delta_gamma != 1 or self.K_seas == 0: return 0.0, 0
        gs = self.idx_g_start; ge = gs + self.K_seas - 1
        ss = 0.0
        for t in range(1, self.T + 1):
            prev = self.x[t - 1, gs:ge + 1]
            mean_new_first = -float(np.sum(prev))
            ss += (self.x[t, gs] - mean_new_first) ** 2
        return float(ss), self.T

    # ------------------- Shively: posterior P(δ=1|y) via slab evidence -------------------
    def _update_delta_from_evidence(self, SS: float, T_eff: int, zbar: float, pi: float) -> Tuple[int, float]:
        # Spike δ=0: approximate degenerate likelihood with tiny ε
        eps = 1e-12
        log_like_spike = -0.5 * T_eff * math.log(2*math.pi*eps) - 0.5 * SS / eps if T_eff > 0 else 0.0
        log_m0 = math.log(max(1.0 - pi, 1e-300)) + log_like_spike

        # Slab δ=1: integrate Q ~ Uniform(0, zbar)
        log_int = _log_evidence_slab(SS, T_eff, zbar)
        log_m1 = math.log(max(pi, 1e-300)) + (log_int - math.log(max(zbar, 1e-300)))

        m = max(log_m0, log_m1)
        p1 = math.exp(log_m1 - m) / (math.exp(log_m0 - m) + math.exp(log_m1 - m))
        delta = 1 if np.random.uniform() < p1 else 0
        return delta, float(p1)


    # ------------------- Q | path, δ=1, slab prior Uniform(0,zbar) -------------------
    def _update_Q_slab(self, Q_curr: float, SS: float, T_eff: int, zbar: float) -> float:
        if T_eff <= 0: return 0.0
        def logpost(q):
            if q <= 0.0 or q >= zbar: return -np.inf
            return (-0.5*T_eff)*math.log(q) - (SS/(2.0*q))  # + log(1/zbar) cancels in slice
        start = Q_curr if 0 < Q_curr < zbar else min(zbar * 0.5, max(1e-6, SS / max(T_eff,1)))
        return _slice_sample_bounded(logpost, start, 0.0, zbar, w=min(zbar, 1.0))

    # --- Dynamic initial state updates (only when δ=1) --- #
    @staticmethod
    def _gibbs_m0_scalar(x0: float, m_prior: float, s_prior: float, P0: float) -> float:
        prec = 1.0 / (s_prior**2) + 1.0 / max(1e-18, P0)
        var = 1.0 / prec
        mean = var * (m_prior / (s_prior**2) + x0 / max(1e-18, P0))
        return float(np.random.normal(mean, math.sqrt(var)))

    def _update_m0_P0_dynamic(self):
        pos_map = {self.idx_alpha: ("m0_alpha","P0_alpha", self.priors.m_m0_alpha, self.priors.s_m0_alpha),
                   self.idx_beta:  ("m0_beta","P0_beta",   self.priors.m_m0_beta,  self.priors.s_m0_beta)}
        for j in [self.idx_alpha, self.idx_beta]:
            if (j == self.idx_alpha and self.delta_alpha==1) or (j==self.idx_beta and self.delta_beta==1):
                m_name, P_name, m_prior, s_prior = pos_map[j]
                x0 = float(self.x[0, j])
                m = self._gibbs_m0_scalar(x0, float(m_prior), float(s_prior), float(getattr(self, P_name)))
                setattr(self, m_name, m)
                a = getattr(self.priors, f"a_P0_{'alpha' if j==self.idx_alpha else 'beta'}")
                b = getattr(self.priors, f"b_P0_{'alpha' if j==self.idx_alpha else 'beta'}")
                P = 1.0 / np.random.gamma(shape=float(a) + 0.5,
                                          scale=1.0 / (float(b) + 0.5 * (x0 - m) ** 2))
                setattr(self, P_name, float(P))
        if self.delta_gamma == 1 and self.K_seas > 0:
            s = float(self.priors.s_m0_gamma)
            m_prior_vec = np.zeros(self.K_seas, float) if self.priors.m_m0_gamma is None \
                          else np.asarray(self.priors.m_m0_gamma, float)
            for k in range(self.K_seas):
                x0 = float(self.x[0, self.idx_g_start + k])
                self.m0_gamma[k] = self._gibbs_m0_scalar(x0, float(m_prior_vec[k]), s, self.P0_gamma)
            diffsq = float(np.sum((self.x[0, self.idx_g_start:self.idx_g_start+self.K_seas] - self.m0_gamma)**2))
            a = float(self.priors.a_P0_gamma) + 0.5 * self.K_seas
            b = float(self.priors.b_P0_gamma) + 0.5 * diffsq
            self.P0_gamma = float(1.0 / np.random.gamma(shape=a, scale=1.0 / b))

    # --- Static parameter updates (only when δ=0) --- #
    def _update_theta_alpha_static(self):
        r = self.y.copy()
        if self.delta_gamma == 1 and self.K_seas > 0:
            r -= self.x[1:self.T+1, self.idx_g_start]
        elif self.K_seas > 0 and self.delta_gamma == 0:
            r -= self.theta_gamma[np.arange(self.T) % self.period]
        if self.delta_alpha == 0:
            r -= self.theta_beta * np.arange(self.T, dtype=float)
        s2 = float(self.sigma2)
        m0, s0 = float(self.priors.m_m0_alpha), float(self.priors.s_m0_alpha)
        prec = self.T / s2 + 1.0 / (s0**2)
        mean = ((r.sum() / s2) + m0 / (s0**2)) / prec
        var  = 1.0 / prec
        self.theta_alpha = float(np.random.normal(mean, math.sqrt(var)))

    def _update_theta_beta_static(self):
        if self.delta_alpha == 1:
            d = self.x[1:, self.idx_alpha] - self.x[:-1, self.idx_alpha]
            s2 = float(max(self.Q_alpha, 1e-6))
            m0, s0 = float(self.priors.m_m0_beta), float(self.priors.s_m0_beta)
            prec = (self.T / s2) + 1.0 / (s0**2)
            mean = ((float(np.sum(d)) / s2) + m0 / (s0**2)) / prec
            var  = 1.0 / prec
            self.theta_beta = float(np.random.normal(mean, math.sqrt(var)))
        else:
            t = np.arange(self.T, dtype=float)
            r = self.y.copy()
            r -= self.theta_alpha
            if self.delta_gamma == 1 and self.K_seas > 0:
                r -= self.x[1:self.T+1, self.idx_g_start]
            elif self.K_seas > 0 and self.delta_gamma == 0:
                r -= self.theta_gamma[np.arange(self.T) % self.period]
            s2 = float(self.sigma2)
            m0, s0 = float(self.priors.m_m0_beta), float(self.priors.s_m0_beta)
            prec = (t @ t) / s2 + 1.0 / (s0**2)
            mean = ((t @ r) / s2 + m0 / (s0**2)) / prec
            var  = 1.0 / prec
            self.theta_beta = float(np.random.normal(mean, math.sqrt(var)))

    def _update_theta_gamma_static(self):
        if self.K_seas == 0: return
        midx = np.arange(self.T) % self.period
        K = self.K_seas
        Z = np.zeros((self.T, K))
        for k in range(K):
            Z[:, k] = (midx == k).astype(float) - (midx == K).astype(float)
        r = self.y.copy()
        if self.delta_alpha == 1:
            r -= self.x[1:self.T+1, self.idx_alpha]
        else:
            r -= (self.theta_alpha + self.theta_beta * np.arange(self.T, dtype=float))
        s2 = float(self.sigma2)
        m_prior = np.zeros(K) if self.priors.m_m0_gamma is None \
                  else np.asarray(self.priors.m_m0_gamma, float).reshape(-1)
        s02 = float(self.priors.s_m0_gamma)**2
        Prec = (Z.T @ Z) / s2 + np.eye(K) / s02
        b    = (Z.T @ r) / s2 + m_prior / s02
        mu   = np.linalg.solve(Prec, b)
        L    = np.linalg.cholesky(Prec)
        z    = np.random.randn(K)
        theta_free = mu + np.linalg.solve(L.T, z)
        self.theta_gamma = np.r_[theta_free, -theta_free.sum()]

    # ------------------ σ² | rest  with Uniform(0, sigma2_bar) prior ------------------ #
    def _update_sigma2(self):
        e = self.y - self._mu_vec()
        SS = float(e @ e)
        T = self.T
        sbar = float(self.priors.sigma2_bar)
        if SS <= 0:
            self.sigma2 = min(sbar, 1.0)
            return
        # log posterior: -(T/2) log s2 - SS/(2 s2), truncated (0, sbar)
        def logpost(s2):
            if s2 <= 0 or s2 >= sbar: return -np.inf
            return -0.5*T*math.log(s2) - SS/(2.0*s2)
        start = min(max(self.sigma2, 1e-6), sbar * 0.9)
        self.sigma2 = _slice_sample_bounded(logpost, start, 0.0, sbar, w=sbar*0.5)

    # --------------------------------- MCMC --------------------------------- #
    def run(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        save_iters = list(range(cfg.burn, cfg.n_iter, cfg.thin))
        n_kept = len(save_iters); keep_idx = 0

        # allocate storage
        self.keep = {
            "sigma": np.zeros(n_kept, float),
            "mu": np.zeros((n_kept, self.T), float),
            "Q_alpha": np.zeros(n_kept), "Q_beta": np.zeros(n_kept),
            "delta_alpha": np.zeros(n_kept, int), "delta_beta": np.zeros(n_kept, int),
            "p_delta_alpha": np.zeros(n_kept, float), "p_delta_beta": np.zeros(n_kept, float),
            "m0_alpha": np.zeros(n_kept), "P0_alpha": np.zeros(n_kept),
            "m0_beta":  np.zeros(n_kept), "P0_beta":  np.zeros(n_kept),
            "theta_alpha": np.zeros(n_kept), "theta_beta": np.zeros(n_kept),
        }
        if self.K_seas > 0:
            self.keep.update({
                "Q_gamma": np.zeros(n_kept),
                "delta_gamma": np.zeros(n_kept, int),
                "p_delta_gamma": np.zeros(n_kept, float),
                "m0_gamma": np.zeros((n_kept, self.K_seas)),
                "P0_gamma": np.zeros(n_kept),
                "theta_gamma": np.zeros((n_kept, self.period)),
            })


        print_every = cfg.progress_every if cfg.progress_every > 0 else max(1, cfg.n_iter // 50) or 1

        for it in range(cfg.n_iter):
            # 1) FFBS on dynamic subset (based on current deltas)
            self._ffbs()

            # 2) Update deltas from evidences, then Q's if δ=1
            # α
            SS_a, T_a = self._innovation_ss_alpha()
            self.delta_alpha, self.pdelta_alpha = self._update_delta_from_evidence(
                SS_a, T_a, self.priors.zbar_alpha, self.priors.pi_alpha
            )
            if self.delta_alpha == 1:
                self.Q_alpha = self._update_Q_slab(self.Q_alpha, SS_a, T_a, self.priors.zbar_alpha)
            else:
                self.Q_alpha = 0.0

            # β
            SS_b, T_b = self._innovation_ss_beta()
            self.delta_beta, self.pdelta_beta = self._update_delta_from_evidence(
                SS_b, T_b, self.priors.zbar_beta, self.priors.pi_beta
            )
            if self.delta_beta == 1:
                self.Q_beta = self._update_Q_slab(self.Q_beta, SS_b, T_b, self.priors.zbar_beta)
            else:
                self.Q_beta = 0.0

            # γ
            if self.K_seas > 0:
                SS_g, T_g = self._innovation_ss_gamma()
                self.delta_gamma, self.pdelta_gamma = self._update_delta_from_evidence(
                    SS_g, T_g, self.priors.zbar_gamma, self.priors.pi_gamma
                )
                if self.delta_gamma == 1:
                    self.Q_gamma = self._update_Q_slab(self.Q_gamma, SS_g, T_g, self.priors.zbar_gamma)
                else:
                    self.Q_gamma = 0.0


            # 3) Update dynamic initial states where δ=1
            self._update_m0_P0_dynamic()

            # 4) Update static parameters where δ=0
            if self.delta_alpha == 0: self._update_theta_alpha_static()
            if self.delta_beta  == 0: self._update_theta_beta_static()
            if self.K_seas > 0 and self.delta_gamma == 0: self._update_theta_gamma_static()

            # 5) Update σ²
            self._update_sigma2()

            # progress
            if cfg.progress and ((it + 1) % print_every == 0 or it == cfg.n_iter - 1):
                parts = [f"[it {it + 1}/{cfg.n_iter}]",
                        f"σ={math.sqrt(self.sigma2):.3f}",
                        f"Qα={self.Q_alpha:.4g} δα={self.delta_alpha} P(δα=1)={self.pdelta_alpha:.3f}",
                        f"Qβ={self.Q_beta:.4g} δβ={self.delta_beta} P(δβ=1)={self.pdelta_beta:.3f}"]
                if self.K_seas > 0:
                    parts.append(f"Qγ={self.Q_gamma:.4g} δγ={self.delta_gamma} P(δγ=1)={self.pdelta_gamma:.3f}")
                if self.delta_alpha == 1:
                    parts.append(f"m0α={self.m0_alpha:.4g} P0α={self.P0_alpha:.4g}")
                else:
                    parts.append(f"θα={self.theta_alpha:.4g}")
                if self.delta_beta == 1:
                    parts.append(f"m0β={self.m0_beta:.4g} P0β={self.P0_beta:.4g}")
                else:
                    parts.append(f"θβ={self.theta_beta:.4g}")
                if self.K_seas > 0:
                    if self.delta_gamma == 1:
                        gtxt = "[" + ", ".join(f"{x:.4g}" for x in self.m0_gamma) + "]"
                        parts.append(f"m0γ={gtxt} P0γ={self.P0_gamma:.4g}")
                    else:
                        gtxt = "[" + ", ".join(f"{x:.4g}" for x in self.theta_gamma) + "]"
                        parts.append(f"θγ={gtxt}")
                print(" | ".join(parts))

            # 6) Save
            if it in save_iters:
                self.keep["mu"][keep_idx, :] = self._mu_vec()
                self.keep["sigma"][keep_idx] = math.sqrt(self.sigma2)
                self.keep["Q_alpha"][keep_idx] = self.Q_alpha
                self.keep["Q_beta"][keep_idx]  = self.Q_beta
                self.keep["delta_alpha"][keep_idx] = self.delta_alpha
                self.keep["delta_beta"][keep_idx]  = self.delta_beta
                self.keep["m0_alpha"][keep_idx] = self.m0_alpha
                self.keep["P0_alpha"][keep_idx] = self.P0_alpha
                self.keep["m0_beta"][keep_idx]  = self.m0_beta
                self.keep["P0_beta"][keep_idx]  = self.P0_beta
                self.keep["theta_alpha"][keep_idx] = self.theta_alpha
                self.keep["theta_beta"][keep_idx]  = self.theta_beta
                self.keep["p_delta_alpha"][keep_idx] = self.pdelta_alpha
                self.keep["p_delta_beta"][keep_idx]  = self.pdelta_beta

                if self.K_seas > 0:
                    self.keep["Q_gamma"][keep_idx] = self.Q_gamma
                    self.keep["delta_gamma"][keep_idx] = self.delta_gamma
                    self.keep["m0_gamma"][keep_idx, :] = self.m0_gamma
                    self.keep["P0_gamma"][keep_idx] = self.P0_gamma
                    self.keep["theta_gamma"][keep_idx, :] = self.theta_gamma
                    self.keep["p_delta_gamma"][keep_idx] = self.pdelta_gamma
                keep_idx += 1

        # Best pattern among saved draws (optional)
        if self.K_seas > 0:
            pats = np.c_[self.keep["delta_alpha"], self.keep["delta_beta"], self.keep["delta_gamma"]]
        else:
            pats = np.c_[self.keep["delta_alpha"], self.keep["delta_beta"]]
        uniq, counts = np.unique(pats, axis=0, return_counts=True)
        best = uniq[np.argmax(counts)]
        self.best_model_pattern_ = tuple(int(x) for x in best.tolist())

        return self.keep

    # ------------------------------- Persistence ---------------------------- #
    def save_posterior(self, out_npz_path: str, extra_meta: Optional[dict] = None) -> None:
        os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)
        arrays = dict(self.keep)
        arrays["y"] = self.y.copy()
        arrays["best_model_pattern"] = np.array(self.best_model_pattern_, int)
        np.savez_compressed(out_npz_path, **arrays)

        meta = {
            "T": int(self.T),
            "period": int(self.period),
            "cfg": asdict(self.cfg),
            "priors": asdict(self.priors),
            "best_model_pattern": self.best_model_pattern_,
        }
        if extra_meta: meta.update(extra_meta)
        meta_path = out_npz_path.replace(".npz", ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[save] Posterior -> {out_npz_path}")
        print(f"[save] Metadata  -> {meta_path}")

# ------------------------- CLI / Example run ------------------------------- #
if __name__ == "__main__":
    import argparse, os, sys, time
    from datetime import datetime
    import numpy as np
    import matplotlib.pyplot as plt

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(base_dir)
    from simulator.mean_time_series import Mean_Time_Series  # newest-first convention

    def _parse_date(s: str | None):
        if not s: return datetime.today()
        parts = [int(p) for p in s.split("-")]
        if len(parts) == 1:  return datetime(parts[0], 1, 1)
        if len(parts) == 2:  return datetime(parts[0], parts[1], 1)
        if len(parts) == 3:  return datetime(parts[0], parts[1], parts[2])
        raise ValueError("start-date must be YYYY, YYYY-MM, or YYYY-MM-DD")

    def _csv_floats_or_none(s: str | None):
        if s is None: return None
        s = s.strip()
        if s == "": return None
        return [float(z) for z in s.split(",") if z.strip() != ""]

    p = argparse.ArgumentParser(
        description=("Kalman FFBS + Shively spike–slab for process variances in a Gaussian DLM "
                     "(level/trend/season). Newest-first seasonal ordering. "
                     "Progress shows Q, δ and m0/P0 (if dynamic) or θ (if static).")
    )

    # Simulation (for Mean_Time_Series only; the sampler always runs spike–slab)
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--period", type=int, default=4)
    p.add_argument("--start-date", type=str, default="2000-01-01")
    p.add_argument("--level-mode", choices=["dynamic", "deterministic"], default="dynamic")
    p.add_argument("--trend-mode", choices=["dynamic", "deterministic", "none"], default="deterministic")
    p.add_argument("--seasonal-mode", choices=["dynamic", "deterministic", "none"], default="deterministic")
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--q-level", type=float, default=0.001)
    p.add_argument("--q-trend", type=float, default=0.00002)
    p.add_argument("--q-season", type=float, default=0.00005)
    p.add_argument("--m0-level", type=float, default=3.0)
    p.add_argument("--v0-level", type=float, default=0.25)
    p.add_argument("--m0-trend", type=float, default=0.015)
    p.add_argument("--v0-trend", type=float, default=0.05)
    p.add_argument("--m0-season", type=str, default=None)
    p.add_argument("--v0-season", type=str, default=None)

    # Shively priors & bounds
    p.add_argument("--zbar-alpha", type=float, default=1.0)
    p.add_argument("--zbar-beta",  type=float, default=1.0)
    p.add_argument("--zbar-gamma", type=float, default=1.0)
    p.add_argument("--pi-alpha", type=float, default=0.5)
    p.add_argument("--pi-beta",  type=float, default=0.5)
    p.add_argument("--pi-gamma", type=float, default=0.5)
    p.add_argument("--sigma2-bar", type=float, default=10.0)

    # Static priors
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

    # Sampler configuration
    p.add_argument("--n-iter", type=int, default=10000)
    p.add_argument("--burn", type=int, default=5000)
    p.add_argument("--thin", type=int, default=2)
    p.add_argument("--seed", type=int, default=40)
    p.add_argument("--progress", default=True)
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--out-dir", type=str, default="results/simulations/DLM_Shively")
    p.add_argument("--plot", default=True)
    p.add_argument("--print-summary", default=True)

    # Initial inference values
    p.add_argument("--sigma-init", type=float, default=1.0)
    p.add_argument("--Q-alpha-init", type=float, default=1e-2)
    p.add_argument("--Q-beta-init",  type=float, default=1e-3)
    p.add_argument("--Q-gamma-init", type=float, default=1e-3)
    p.add_argument("--delta-alpha-init", type=int, choices=[0,1], default=1)
    p.add_argument("--delta-beta-init",  type=int, choices=[0,1], default=1)
    p.add_argument("--delta-gamma-init", type=int, choices=[0,1], default=1)
    p.add_argument("--m0-gamma-init", type=str, default=None)
    p.add_argument("--P0-alpha-init", type=float, default=0.25)
    p.add_argument("--P0-beta-init", type=float, default=0.05)
    p.add_argument("--P0-gamma-init", type=float, default=0.25)
    p.add_argument("--theta-alpha-init", type=float, default=0.0)
    p.add_argument("--theta-beta-init",  type=float, default=0.0)
    p.add_argument("--theta-gamma-init", type=str, default=None)

    args = p.parse_args()
    np.random.seed(args.seed)

    # Simulate with requested modes (affects only data gen)
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

    # Priors
    pri_gamma_vec = _csv_floats_or_none(args.prior_m_m0_gamma)
    priors = PriorsShively(
        zbar_alpha=args.zbar_alpha, zbar_beta=args.zbar_beta, zbar_gamma=args.zbar_gamma,
        pi_alpha=args.pi_alpha, pi_beta=args.pi_beta, pi_gamma=args.pi_gamma,
        sigma2_bar=args.sigma2_bar,
        m_m0_alpha=args.prior_m_m0_alpha, s_m0_alpha=args.prior_s_m0_alpha,
        m_m0_beta=args.prior_m_m0_beta,  s_m0_beta=args.prior_s_m0_beta,
        m_m0_gamma=None if pri_gamma_vec is None else pri_gamma_vec,
        s_m0_gamma=args.prior_s_m0_gamma,
        a_P0_alpha=args.prior_a_P0_alpha, b_P0_alpha=args.prior_b_P0_alpha,
        a_P0_beta=args.prior_a_P0_beta,  b_P0_beta=args.prior_b_P0_beta,
        a_P0_gamma=args.prior_a_P0_gamma, b_P0_gamma=args.prior_b_P0_gamma,
    )

    cfg = SamplerConfig(
        n_iter=int(args.n_iter), burn=int(args.burn), thin=int(args.thin),
        random_seed=int(args.seed), progress=bool(args.progress),
        progress_every=int(args.progress_every),
    )

    # Initial static seasonal
    theta_gamma_init = _csv_floats_or_none(args.theta_gamma_init)

    # Sampler
    m0_gamma_init = _csv_floats_or_none(args.m0_gamma_init)
    sampler = DLMGibbs_Shively(
        y=y, period=args.period,
        sigma2_init=args.sigma_init ** 2,
        Q_alpha_init=args.Q_alpha_init, Q_beta_init=args.Q_beta_init, Q_gamma_init=args.Q_gamma_init,
        delta_alpha_init=args.delta_alpha_init, delta_beta_init=args.delta_beta_init, delta_gamma_init=args.delta_gamma_init,
        m0_alpha_init=args.m0_level, m0_beta_init=args.m0_trend,
        P0_alpha_init=args.P0_alpha_init, P0_beta_init=args.P0_beta_init,
        m0_gamma_init=m0_gamma_init, P0_gamma_init=args.P0_gamma_init,
        theta_alpha_init=args.theta_alpha_init, theta_beta_init=args.theta_beta_init,
        theta_gamma_init=theta_gamma_init,
        priors=priors, cfg=cfg,
    )

    # Truth overlays (optional)
    true_sigma = mts.sigma
    mu_truth = mu_T

    if args.print_summary:
        print(f"\nSimulated {args.T} observations (σ={true_sigma}).")
        print("Shively spike–slab bounds (Uniform slab on Q; spike at 0):")
        print(f"  ζ̄_alpha={priors.zbar_alpha:.3g}, ζ̄_beta={priors.zbar_beta:.3g}, ζ̄_gamma={priors.zbar_gamma:.3g}")
        print(f"  π_alpha={priors.pi_alpha:.2f}, π_beta={priors.pi_beta:.2f}, π_gamma={priors.pi_gamma:.2f}")
        print(f"  σ² ~ Uniform(0, {priors.sigma2_bar:.3g})\n")

    # Run
    import time
    t0 = time.time()
    post = sampler.run()
    elapsed = time.time() - t0
    print(f"\n[Run completed in {elapsed:.1f}s]")

    out_dir = os.path.join(
        args.out_dir,
        f"shively_{args.level_mode}-{args.trend_mode}-{args.seasonal_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(out_dir, exist_ok=True)
    sampler.save_posterior(
        out_npz_path=os.path.join(out_dir, "posterior.npz"),
        extra_meta={
            "elapsed_seconds": float(elapsed),
            "Shively": {
                "zbar": {"alpha": priors.zbar_alpha, "beta": priors.zbar_beta, "gamma": priors.zbar_gamma},
                "pi":   {"alpha": priors.pi_alpha,  "beta": priors.pi_beta,  "gamma": priors.pi_gamma},
                "sigma2_bar": priors.sigma2_bar,
            },
        },
    )

    # Summary
    if args.print_summary:
        print("\n--- Posterior means (saved draws) ---")
        print(f"σ = {np.mean(post['sigma']):.4f}")
        for key in ["Q_alpha", "Q_beta"] + (["Q_gamma"] if "Q_gamma" in post else []):
            mQ = np.mean(post[key]); print(f"{key} = {mQ:.4g} (√Q ≈ {math.sqrt(max(mQ,0.0)):.4g})")
        print(f"\nBest (MAP-ish) model pattern (δα, δβ, δγ or without γ): {sampler.best_model_pattern_}")

    # Plot
    if args.plot:
        mu_hat = post["mu"].mean(axis=0)
        plt.figure(figsize=(10, 4))
        plt.plot(dates_T, y, label="y_t", lw=1)
        plt.plot(dates_T, mu_truth, "--", label="μ_t (truth)")
        plt.plot(dates_T, mu_hat, "-.", label="μ̂_t (post mean)")
        plt.title(f"DLM Shively spike–slab ({args.level_mode}/{args.trend_mode}/{args.seasonal_mode})")
        plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()
