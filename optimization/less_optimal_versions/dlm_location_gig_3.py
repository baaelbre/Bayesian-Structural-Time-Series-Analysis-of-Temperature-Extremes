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

# --------------- Slice sampler on (0, ∞) for log-density f --------------- #

def _slice_sample_positive(logf, x0: float, w: float = 1.0, m: int = 100, rng=np.random):
    x = max(1e-12, float(x0))
    fx = logf(x)
    u = rng.uniform()
    y = fx - np.log(1.0/u)
    L = x - w * rng.uniform()
    R = L + w
    j = int(m * rng.uniform()); k = (m - 1) - j
    while j > 0 and L > 0 and logf(max(L, 1e-18)) > y:
        L -= w; j -= 1
    while k > 0 and logf(R) > y:
        R += w; k -= 1
    if L <= 0: L = 1e-18
    for _ in range(100):
        x_new = rng.uniform(L, R)
        if logf(x_new) >= y:
            return max(1e-18, x_new)
        if x_new < x:
            L = x_new
        else:
            R = x_new
    return max(1e-18, x)

# =============================================================================
# Priors & Config
# =============================================================================

@dataclass
class PriorsFS:
    # σ² prior: precision τ=1/σ² ~ Gamma(a_sigma, b_sigma) (shape–rate)
    a_sigma: float = 2.0
    b_sigma: float = 1.0

    # m0 priors for initial dynamic state means (and as priors for static θ)
    m_m0_alpha: float = 0.0
    s_m0_alpha: float = 10.0
    m_m0_beta: float  = 0.0
    s_m0_beta: float  = 10.0
    m_m0_gamma: Optional[Sequence[float]] = None  # length p-1 (newest-first)
    s_m0_gamma: float = 5.0

    # Initial-state variances P0 ~ InvGamma(a, b)
    a_P0_alpha: float = 5.0; b_P0_alpha: float = 1.0
    a_P0_beta:  float = 5.0; b_P0_beta:  float = 1.0
    a_P0_gamma: float = 5.0; b_P0_gamma: float = 1.0

    # FS non-centered prior on process SD ψ_k: ψ_k | σ² ~ N(0, B0_k σ²), Q_k = ψ_k²
    B0_alpha: float = 10.0
    B0_beta:  float = 10.0
    B0_gamma: float = 10.0

    # SSVS inclusion priors π_k = P(δ_k=1)
    pi_alpha: float = 0.5
    pi_beta:  float = 0.5
    pi_gamma: float = 0.5

@dataclass
class SamplerConfig:
    n_iter: int = 10000
    burn: int = 5000
    thin: int = 2
    random_seed: Optional[int] = 40
    progress: bool = True
    progress_every: int = 0  # 0 => ~2% of n_iter

# =============================================================================
# DLM Sampler — FS-SSVS (always full model, SSVS selects blocks)
# =============================================================================

class DLMGibbs_FS_SSVS:
    """
    y_t = μ_t + ε_t,  ε_t ~ N(0, σ²)
    μ_t = (α_t or θα) + (seasonal dynamic g_t or static θγ[t mod p]) + (trend if α static: θβ * t)
    State (when dynamic):
      α_t = α_{t-1} + (β_{t-1} or θβ) + ηα_t,     ηα_t ~ N(0, Qα) if δα=1 else 0
      β_t = β_{t-1} + ηβ_t,                       ηβ_t ~ N(0, Qβ) if δβ=1 else 0
      seasonal: dummy-rotation with only first coord driven by ηγ_t ~ N(0, Qγ) if δγ=1

    SSVS:
      δk∈{0,1}. If δk=1 → dynamic block participates in FFBS; we sample (m0_k, P0_k) and Q_k.
      If δk=0 → block is static: FFBS excludes it and we sample θ_k via conjugate updates.

    Progress line shows:
      • σ, Q⋅, δ⋅
      • For each block: if δ=1 → m0⋅ and P0⋅ ; if δ=0 → θ⋅
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
        Q_alpha_init: float = 1e-3, Q_beta_init: float = 1e-4, Q_gamma_init: float = 1e-3,
        delta_alpha_init: int = 1, delta_beta_init: int = 1, delta_gamma_init: int = 1,
        # optional static inits
        theta_alpha_init: float = 0.0, theta_beta_init: float = 0.0,
        theta_gamma_init: Optional[Sequence[float]] = None,  # length p-1 (newest-first)
        priors: PriorsFS = PriorsFS(),
        cfg: SamplerConfig = SamplerConfig(),
    ):
        # Data
        self.y = np.asarray(y, float); self.T = int(self.y.size)
        self.period = int(period)
        if self.period < 2: raise ValueError("period must be >= 2")

        self.priors, self.cfg = priors, cfg
        if cfg.random_seed is not None: np.random.seed(cfg.random_seed)

        # Fixed layout (always full): alpha, beta, seasonal (p-1)
        self.layout: List[str] = ["alpha", "beta"] + [f"g{k}" for k in range(1, self.period)]
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

        # SSVS: process variances and inclusion indicators
        self.Q_alpha = float(Q_alpha_init)
        self.Q_beta  = float(Q_beta_init)
        self.Q_gamma = float(Q_gamma_init)
        self.delta_alpha = int(delta_alpha_init)
        self.delta_beta  = int(delta_beta_init)
        self.delta_gamma = int(delta_gamma_init)

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

        # Caches & storage
        self.keep: Dict[str, np.ndarray] = {}
        self._delta_counts = np.zeros((0,))  # set in run

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
        """Observation row for full state (alpha and first seasonal coord enter)."""
        h = np.zeros(self.dim_full, float)
        if self.delta_alpha == 1:
            h[self.idx_alpha] = 1.0
        # for seasonal dynamic, observation depends on first seasonal coord only
        if self.delta_gamma == 1 and self.K_seas > 0:
            h[self.idx_g_start] = 1.0
        return h.reshape(1, -1)

    def _A_full(self) -> np.ndarray:
        """Transition for full state; seasonal rotation with innovation on first coord only."""
        A = np.eye(self.dim_full)
        # alpha gets + beta (dynamic) or + theta_beta (static) in the transition when alpha is dynamic
        if self.delta_alpha == 1:
            if self.delta_beta == 1:
                A[self.idx_alpha, self.idx_beta] = 1.0
            else:
                # handled via constant control u below
                pass
        # seasonal rotation (K_seas = p-1)
        if self.delta_gamma == 1 and self.K_seas > 0:
            gs = self.idx_g_start; ge = gs + self.K_seas - 1
            A[gs, gs:ge+1] = -1.0
            if self.K_seas > 1:
                A[gs+1:ge+1, gs:ge] = np.eye(self.K_seas - 1)
                A[gs+1:ge+1, ge] = 0.0
        return A

    def _u_full(self) -> np.ndarray:
        """Deterministic drift for transitions (only needed if alpha dynamic and beta static)."""
        u = np.zeros(self.dim_full, float)
        if self.delta_alpha == 1 and self.delta_beta == 0:
            u[self.idx_alpha] = float(self.theta_beta)
        return u

    def _Q_full(self) -> np.ndarray:
        """Process covariance for full state, zeros for static blocks."""
        Q = np.zeros((self.dim_full, self.dim_full))
        if self.delta_alpha == 1 and self.Q_alpha > 0: Q[self.idx_alpha, self.idx_alpha] = self.Q_alpha
        if self.delta_beta  == 1 and self.Q_beta  > 0: Q[self.idx_beta,  self.idx_beta]  = self.Q_beta
        if self.delta_gamma == 1 and self.K_seas > 0 and self.Q_gamma > 0:
            Q[self.idx_g_start, self.idx_g_start] = self.Q_gamma
        return Q

    def _mu_vec(self) -> np.ndarray:
        """Build μ_t from current x, θ, and deltas."""
        mu = np.zeros(self.T, float)
        # alpha contribution
        if self.delta_alpha == 1:
            alpha_t = self.x[1:self.T+1, self.idx_alpha]
            mu += alpha_t
        else:
            mu += self.theta_alpha
            # if alpha is static, slope becomes a linear trend term in μ_t
            mu += self.theta_beta * np.arange(self.T, dtype=float)

        # seasonal
        if self.K_seas > 0:
            if self.delta_gamma == 1:
                # observation uses g1 only (already added via H*x in filter; here add explicitly)
                g1_t = self.x[1:self.T+1, self.idx_g_start]
                mu += g1_t
            else:
                mu += self.theta_gamma[np.arange(self.T) % self.period]
        return mu

    # ------------------------- FFBS on dynamic subset ------------------------- #
    def _ffbs(self):
        dyn_idx = self._dynamic_indices()
        d = len(dyn_idx)
        if d == 0:
            # no dynamic states; keep x zeros
            self.x[:] = 0.0
            return

        # Build reduced model matrices
        H_full, A_full, Q_full, u_full = self._H_full(), self._A_full(), self._Q_full(), self._u_full()
        H = H_full[:, dyn_idx]
        A = A_full[np.ix_(dyn_idx, dyn_idx)]
        Q = Q_full[np.ix_(dyn_idx, dyn_idx)]
        u = u_full[dyn_idx]

        # initial mean/cov for dynamic coords
        m0_list = []
        P0_list = []
        for j in dyn_idx:
            if j == self.idx_alpha: m0_list.append(self.m0_alpha); P0_list.append(self.P0_alpha)
            elif j == self.idx_beta: m0_list.append(self.m0_beta); P0_list.append(self.P0_beta)
            else:  # seasonal
                off = j - self.idx_g_start
                m0_list.append(self.m0_gamma[off])
                P0_list.append(self.P0_gamma)
        m0 = np.asarray(m0_list, float)
        P0 = np.diag(np.asarray(P0_list, float)) + 1e-12*np.eye(d)

        # Filtering
        m = np.zeros((self.T + 1, d))
        C = np.zeros((self.T + 1, d, d))
        a = np.zeros((self.T + 1, d))
        Rm = np.zeros((self.T + 1, d, d))
        m[0] = m0; C[0] = P0
        R = float(self.sigma2)

        # deterministic chunks in μ_t (pieces not represented in H*x)
        det = np.zeros(self.T, float)
        if self.delta_alpha == 0:
            det += self.theta_alpha + self.theta_beta * np.arange(self.T, dtype=float)
        if self.K_seas > 0 and self.delta_gamma == 0:
            det += self.theta_gamma[np.arange(self.T) % self.period]

        for t in range(1, self.T + 1):
            a[t]  = A @ m[t - 1] + u
            Rm[t] = A @ C[t - 1] @ A.T + Q
            Rm[t] = 0.5*(Rm[t]+Rm[t].T) + 1e-12*np.eye(d)

            # observation residual after subtracting deterministic part
            ytil = self.y[t - 1] - det[t - 1]
            S = float(H @ Rm[t] @ H.T + R)
            K = (Rm[t] @ H.T) / S
            v = ytil - float(H @ a[t])
            m[t] = a[t] + (K.flatten() * v)
            C[t] = Rm[t] - K @ (H @ Rm[t])
            C[t] = 0.5*(C[t]+C[t].T) + 1e-12*np.eye(d)

        # Backward sampling
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

        # write back to full container
        self.x[:] = 0.0
        self.x[:, dyn_idx] = xs

    # ------------- Innovation sums of squares (for Q updates) ------------- #
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

    # ------------------- FS slab log posterior for Q_k ------------------- #
    def _logpost_Q_fs(self, Q: float, SS: float, T_eff: int, B0: float, sigma2: float) -> float:
        if Q <= 0: return -np.inf
        lam = 0.5 * (1.0 - T_eff)
        return (lam - 1.0) * math.log(Q) - 0.5 * (SS / Q) - 0.5 * (Q / (B0 * sigma2))

    def _log_marginal_slab_laplace(self, SS: float, T_eff: int, B0: float, sigma2: float) -> float:
        T1 = T_eff + 1.0
        b  = B0 * sigma2
        disc = T1*T1 + 4.0 * SS / max(b, 1e-300)
        q_hat = 0.5 * b * (-T1 + math.sqrt(disc))
        q_hat = max(q_hat, 1e-18)
        lam = 0.5 * (1.0 - T_eff)
        g = (lam - 1.0) * math.log(q_hat) - 0.5 * (SS / q_hat) - 0.5 * (q_hat / b)
        curv = (lam - 1.0) / (q_hat*q_hat) + SS / (q_hat**3 + 1e-300)
        curv = max(curv, 1e-300)
        return g + 0.5 * math.log(2.0 * math.pi) - 0.5 * math.log(curv)

    def _update_delta_block(self, SS: float, T_eff: int, B0: float, sigma2: float, pi: float) -> int:
        # δ=1 (slab) via Laplace approx; δ=0 (spike) as tiny ε-variance proxy
        log_m1 = self._log_marginal_slab_laplace(SS, T_eff, B0, sigma2)
        eps = 1e-12 * max(sigma2, 1.0)
        log_m0 = -0.5 * T_eff * math.log(2.0 * math.pi * eps) - 0.5 * SS / max(eps, 1e-300)
        log_post1 = math.log(max(pi, 1e-12)) + log_m1
        log_post0 = math.log(max(1.0 - pi, 1e-12)) + log_m0
        mmax = max(log_post0, log_post1)
        p1 = math.exp(log_post1 - mmax) / (math.exp(log_post0 - mmax) + math.exp(log_post1 - mmax))
        return 1 if np.random.uniform() < p1 else 0

    def _update_Q_block(self, Q_curr: float, SS: float, T_eff: int, B0: float, sigma2: float) -> float:
        if T_eff <= 0: return 0.0
        T1 = T_eff + 1.0; b = B0 * sigma2
        disc = T1*T1 + 4.0 * SS / max(b, 1e-300)
        mode = 0.5 * b * (-T1 + math.sqrt(disc))
        w = max(1e-6, 0.5 * (mode + 1e-6))
        return _slice_sample_positive(lambda q: self._logpost_Q_fs(q, SS, T_eff, B0, sigma2),
                                      max(Q_curr, mode if mode > 0 else 1e-6), w=w)

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
        # alpha, beta
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
        # seasonal (each coord shares P0_gamma)
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
        # remove dynamic pieces from μ_t except alpha
        if self.delta_gamma == 1 and self.K_seas > 0:
            g1 = self.x[1:self.T+1, self.idx_g_start]
            r -= g1
        elif self.K_seas > 0 and self.delta_gamma == 0:
            r -= self.theta_gamma[np.arange(self.T) % self.period]
        if self.delta_alpha == 0:
            # remove linear trend if slope treated as observation trend (alpha static)
            r -= self.theta_beta * np.arange(self.T, dtype=float)
        s2 = float(self.sigma2)
        m0, s0 = float(self.priors.m_m0_alpha), float(self.priors.s_m0_alpha)
        prec = self.T / s2 + 1.0 / (s0**2)
        mean = ((r.sum() / s2) + m0 / (s0**2)) / prec
        var  = 1.0 / prec
        self.theta_alpha = float(np.random.normal(mean, math.sqrt(var)))

    def _update_theta_beta_static(self):
        # Two cases: (i) alpha dynamic → θβ affects alpha transition; use differences in alpha_t
        #            (ii) alpha static → θβ enters μ_t as linear time trend; use regression on y
        if self.delta_alpha == 1:
            # regress (alpha_t - alpha_{t-1}) on constant 1 with noise Q_alpha≈0 proxy? Use σ² via FFBS-residualization:
            d = self.x[1:, self.idx_alpha] - self.x[:-1, self.idx_alpha]
            s2 = float(self.Q_alpha if self.delta_alpha==1 else self.sigma2)
            m0, s0 = float(self.priors.m_m0_beta), float(self.priors.s_m0_beta)
            prec = (self.T / max(s2,1e-12)) + 1.0 / (s0**2)
            mean = ((float(np.sum(d)) / max(s2,1e-12)) + m0 / (s0**2)) / prec
            var  = 1.0 / prec
            self.theta_beta = float(np.random.normal(mean, math.sqrt(var)))
        else:
            # alpha static: θβ contributes as linear trend in y
            t = np.arange(self.T, dtype=float)
            r = self.y.copy()
            # remove alpha static intercept
            r -= self.theta_alpha
            # remove seasonal
            if self.delta_gamma == 1 and self.K_seas > 0:
                g1 = self.x[1:self.T+1, self.idx_g_start]
                r -= g1
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
        # remove alpha contribution
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

    # ------------------ σ² | rest  ------------------ #
    def _update_sigma2(self):
        e = self.y - self._mu_vec()
        a = float(self.priors.a_sigma) + 0.5 * self.T
        b = float(self.priors.b_sigma) + 0.5 * float(e @ e)
        tau = np.random.gamma(shape=a, scale=1.0 / b)
        self.sigma2 = 1.0 / max(tau, 1e-300)

    # ------------------- Progress formatting ------------------- #
    def _progress_line(self, it: int) -> str:
        parts = [f"[it {it + 1}/{self.cfg.n_iter}]",
                 f"σ={math.sqrt(self.sigma2):.3f}",
                 f"Qα={self.Q_alpha:.4g} δα={self.delta_alpha}",
                 f"Qβ={self.Q_beta:.4g} δβ={self.delta_beta}"]
        if self.K_seas > 0:
            parts.append(f"Qγ={self.Q_gamma:.4g} δγ={self.delta_gamma}")

        # Report m0/P0 when δ=1; θ when δ=0
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

        return " | ".join(parts)

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
            "m0_alpha": np.zeros(n_kept), "P0_alpha": np.zeros(n_kept),
            "m0_beta":  np.zeros(n_kept), "P0_beta":  np.zeros(n_kept),
            "theta_alpha": np.zeros(n_kept), "theta_beta": np.zeros(n_kept),
        }
        if self.K_seas > 0:
            self.keep.update({
                "Q_gamma": np.zeros(n_kept),
                "delta_gamma": np.zeros(n_kept, int),
                "m0_gamma": np.zeros((n_kept, self.K_seas)),
                "P0_gamma": np.zeros(n_kept),
                "theta_gamma": np.zeros((n_kept, self.period)),
            })

        # track inclusion frequencies
        inc_counts = np.zeros(3, int)  # alpha, beta, gamma

        print_every = cfg.progress_every if cfg.progress_every > 0 else max(1, cfg.n_iter // 50) or 1

        for it in range(cfg.n_iter):
            # 1) FFBS on dynamic subset
            self._ffbs()

            # 2) SSVS updates: deltas and Q's
            # α
            SS, T_eff = self._innovation_ss_alpha()
            self.delta_alpha = self._update_delta_block(SS, T_eff, self.priors.B0_alpha, self.sigma2, self.priors.pi_alpha)
            if self.delta_alpha == 1:
                self.Q_alpha = self._update_Q_block(self.Q_alpha, SS, T_eff, self.priors.B0_alpha, self.sigma2)
            else:
                self.Q_alpha = 0.0
            # β
            SS, T_eff = self._innovation_ss_beta()
            self.delta_beta = self._update_delta_block(SS, T_eff, self.priors.B0_beta, self.sigma2, self.priors.pi_beta)
            if self.delta_beta == 1:
                self.Q_beta = self._update_Q_block(self.Q_beta, SS, T_eff, self.priors.B0_beta, self.sigma2)
            else:
                self.Q_beta = 0.0
            # γ
            if self.K_seas > 0:
                SS, T_eff = self._innovation_ss_gamma()
                self.delta_gamma = self._update_delta_block(SS, T_eff, self.priors.B0_gamma, self.sigma2, self.priors.pi_gamma)
                if self.delta_gamma == 1:
                    self.Q_gamma = self._update_Q_block(self.Q_gamma, SS, T_eff, self.priors.B0_gamma, self.sigma2)
                else:
                    self.Q_gamma = 0.0

            # 3) Update dynamic initial states (m0, P0) where δ=1
            self._update_m0_P0_dynamic()

            # 4) Update static parameters θ where δ=0
            if self.delta_alpha == 0: self._update_theta_alpha_static()
            if self.delta_beta  == 0: self._update_theta_beta_static()
            if self.K_seas > 0 and self.delta_gamma == 0: self._update_theta_gamma_static()

            # 5) Update σ²
            self._update_sigma2()

            # progress
            if cfg.progress and ((it + 1) % print_every == 0 or it == cfg.n_iter - 1):
                print(self._progress_line(it))

            # tally inclusions
            inc_counts[0] += int(self.delta_alpha == 1)
            inc_counts[1] += int(self.delta_beta  == 1)
            if self.K_seas > 0: inc_counts[2] += int(self.delta_gamma == 1)

            # save
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
                if self.K_seas > 0:
                    self.keep["Q_gamma"][keep_idx] = self.Q_gamma
                    self.keep["delta_gamma"][keep_idx] = self.delta_gamma
                    self.keep["m0_gamma"][keep_idx, :] = self.m0_gamma
                    self.keep["P0_gamma"][keep_idx] = self.P0_gamma
                    self.keep["theta_gamma"][keep_idx, :] = self.theta_gamma
                keep_idx += 1

        # inclusion probabilities (over all iterations, not only saved)
        inc_probs = inc_counts / float(cfg.n_iter)
        self.inclusion_probabilities_ = {
            "alpha": float(inc_probs[0]),
            "beta":  float(inc_probs[1]),
            "gamma": float(inc_probs[2] if self.K_seas>0 else np.nan),
        }

        # MAP model among saved draws
        # pattern = (δα, δβ, δγ)
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
        arrays["inclusion_probabilities"] = np.array([
            self.inclusion_probabilities_.get("alpha", np.nan),
            self.inclusion_probabilities_.get("beta",  np.nan),
            self.inclusion_probabilities_.get("gamma", np.nan),
        ], float)
        arrays["best_model_pattern"] = np.array(self.best_model_pattern_, int)
        np.savez_compressed(out_npz_path, **arrays)

        meta = {
            "T": int(self.T),
            "period": int(self.period),
            "layout": list(self.layout),
            "cfg": asdict(self.cfg),
            "priors": asdict(self.priors),
            "inclusion_probabilities": self.inclusion_probabilities_,
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
        description=("Kalman FFBS + FS-SSVS for process variances in a Gaussian DLM "
                     "(level/trend/season). Newest-first seasonal ordering. "
                     "Progress shows Q, δ and m0/P0 (if dynamic) or θ (if static).")
    )

    # Simulation (for Mean_Time_Series only; the sampler always runs full SSVS)
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

    # Priors
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

    # FS scales and SSVS priors
    p.add_argument("--B0-alpha", type=float, default=10.0)
    p.add_argument("--B0-beta",  type=float, default=10.0)
    p.add_argument("--B0-gamma", type=float, default=10.0)
    p.add_argument("--pi-alpha", type=float, default=0.5)
    p.add_argument("--pi-beta",  type=float, default=0.5)
    p.add_argument("--pi-gamma", type=float, default=0.5)

    # Sampler configuration
    p.add_argument("--n-iter", type=int, default=10000)
    p.add_argument("--burn", type=int, default=5000)
    p.add_argument("--thin", type=int, default=2)
    p.add_argument("--seed", type=int, default=40)
    p.add_argument("--progress", default=True)
    p.add_argument("--progress-every", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="results/simulations/DLM_FS_SSVS")
    p.add_argument("--plot", default=True)
    p.add_argument("--print-summary", default=True)

    # Initial inference values
    p.add_argument("--sigma-init", type=float, default=1.0)
    p.add_argument("--Q-alpha-init", type=float, default=1e-1)
    p.add_argument("--Q-beta-init",  type=float, default=1e-1)
    p.add_argument("--Q-gamma-init", type=float, default=1e-1)
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
    priors = PriorsFS(
        a_sigma=args.prior_a_sigma, b_sigma=args.prior_b_sigma,
        m_m0_alpha=args.prior_m_m0_alpha, s_m0_alpha=args.prior_s_m0_alpha,
        m_m0_beta=args.prior_m_m0_beta, s_m0_beta=args.prior_s_m0_beta,
        m_m0_gamma=None if pri_gamma_vec is None else pri_gamma_vec,
        s_m0_gamma=args.prior_s_m0_gamma,
        a_P0_alpha=args.prior_a_P0_alpha, b_P0_alpha=args.prior_b_P0_alpha,
        a_P0_beta=args.prior_a_P0_beta, b_P0_beta=args.prior_b_P0_beta,
        a_P0_gamma=args.prior_a_P0_gamma, b_P0_gamma=args.prior_b_P0_gamma,
        B0_alpha=args.B0_alpha, B0_beta=args.B0_beta, B0_gamma=args.B0_gamma,
        pi_alpha=args.pi_alpha, pi_beta=args.pi_beta, pi_gamma=args.pi_gamma,
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
    sampler = DLMGibbs_FS_SSVS(
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

    # Truth overlays (optional in this version: just for plotting)
    true_sigma = mts.sigma
    mu_truth = mu_T

    if args.print_summary:
        print(f"\nSimulated {args.T} observations (σ={true_sigma}) with modes "
              f"{args.level_mode}/{args.trend_mode}/{args.seasonal_mode}.\n")
        print("FS scales (ψ_k | σ² ~ N(0, B0_k σ²)) and SSVS priors:")
        print(f"  B0_alpha={priors.B0_alpha:.3g}, B0_beta={priors.B0_beta:.3g}, B0_gamma={priors.B0_gamma:.3g}")
        print(f"  π_alpha={priors.pi_alpha:.2f}, π_beta={priors.pi_beta:.2f}, π_gamma={priors.pi_gamma:.2f}\n")

    # Run
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
            "FS": {
                "B0": {"alpha": priors.B0_alpha, "beta": priors.B0_beta, "gamma": priors.B0_gamma},
                "pi": {"alpha": priors.pi_alpha, "beta": priors.pi_beta, "gamma": priors.pi_gamma},
            },
        },
    )

    # Summary
    if args.print_summary:
        print("\n--- Inclusion probabilities (over all iterations) ---")
        for k, v in sampler.inclusion_probabilities_.items():
            print(f"P(δ_{k}=1) ≈ {v:.3f}")
        print(f"\nBest (MAP) model pattern (δα, δβ, δγ): {sampler.best_model_pattern_}")

        print("\n--- Posterior means (saved draws) ---")
        print(f"σ = {np.mean(post['sigma']):.4f}")
        for key in ["Q_alpha", "Q_beta"] + (["Q_gamma"] if "Q_gamma" in post else []):
            mQ = np.mean(post[key]); print(f"{key} = {mQ:.4g} (√Q ≈ {math.sqrt(max(mQ,0.0)):.4g})")

    # Plot
    if args.plot:
        mu_hat = post["mu"].mean(axis=0)
        plt.figure(figsize=(10, 4))
        plt.plot(dates_T, y, label="y_t", lw=1)
        plt.plot(dates_T, mu_truth, "--", label="μ_t (truth)")
        plt.plot(dates_T, mu_hat, "-.", label="μ̂_t (post mean)")
        plt.title(f"DLM FS-SSVS ({args.level_mode}/{args.trend_mode}/{args.seasonal_mode})")
        plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()
