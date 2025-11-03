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

def _logsumexp(vals: Sequence[float]) -> float:
    if len(vals) == 0: 
        return float("-inf")
    a = np.asarray(vals, float)
    m = np.max(a)
    return float(m + np.log(np.sum(np.exp(a - m))))

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
    pi_alpha: float = 0.1
    pi_beta:  float = 0.1
    pi_gamma: float = 0.1

@dataclass
class SamplerConfig:
    n_iter: int = 10000
    burn: int = 5000
    thin: int = 2
    random_seed: Optional[int] = 40
    progress: bool = True
    progress_every: int = 0  # 0 => ~2% of n_iter

# =============================================================================
# DLM Sampler — FS-SSVS with JOINT δ-move (choose deltas FIRST, then FFBS)
# =============================================================================

class DLMGibbs_FS_SSVS:
    """
    y_t = μ_t + ε_t,  ε_t ~ N(0, σ²)
    μ_t = (α_t or θα) + (seasonal dynamic g_t or static θγ[t mod p]) + (trend if α static: θβ * t)

    State (when dynamic):
      α_t = α_{t-1} + (β_{t-1} or θβ) + ηα_t,     ηα_t ~ N(0, Qα) if δα=1 else 0
      β_t = β_{t-1} + ηβ_t,                       ηβ_t ~ N(0, Qβ) if δβ=1 else 0
      seasonal: dummy-rotation; innovation on first coord only when δγ=1

    Progress lines print WINDOWED MEANS over the last k iterations (k = progress_every).
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

        # Layout
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

        # Latent path container (max size; we only fill dynamic columns)
        self.x = np.zeros((self.T + 1, self.dim_full), float)

        # Storage & progress window
        self.keep: Dict[str, np.ndarray] = {}
        self._last_delta_stats: Dict[str, Dict[str, float]] = {}
        self._win = None  # rolling window accumulator (set in run)

        if self.cfg.progress:
            y = self.y
            sd1 = _robust_sd(np.diff(y)) if y.size >= 2 else 0.0
            sd2 = _robust_sd(np.diff(y, n=2)) if y.size >= 3 else 0.0
            print(f"[init] scale proxies: sd1={sd1:.4g}, sd2={sd2:.4g}")

    # ----------------------------- Residuals for evidences ----------------------------- #
    def _residual_minus_alpha(self) -> np.ndarray:
        r = self.y.copy()
        # remove seasonal
        if self.K_seas > 0:
            if self.delta_gamma == 1:
                r -= self.x[1:self.T+1, self.idx_g_start]
            else:
                r -= self.theta_gamma[np.arange(self.T) % self.period]
        # remove (β trend in obs) only when alpha static
        if self.delta_alpha == 0:
            r -= self.theta_beta * np.arange(self.T, dtype=float)
        return r

    def _residual_minus_gamma(self) -> np.ndarray:
        r = self.y.copy()
        # remove level contribution
        if self.delta_alpha == 1:
            r -= self.x[1:self.T+1, self.idx_alpha]
        else:
            r -= (self.theta_alpha + self.theta_beta * np.arange(self.T, dtype=float))
        return r

    # ----------------------------- Evidences: static & dynamic ----------------------------- #
    @staticmethod
    def _log_evidence_static_reg(r: np.ndarray, Z: np.ndarray, m0: np.ndarray, S0: np.ndarray | float, sigma2: float) -> float:
        T = len(r); K = Z.shape[1]
        if np.isscalar(S0): 
            S0 = (float(S0)**2) * np.eye(K)
        S0_inv = np.linalg.inv(S0)
        Prec = S0_inv + (Z.T @ Z) / sigma2
        b    = S0_inv @ m0 + (Z.T @ r) / sigma2
        s0_ld = np.linalg.slogdet(S0)[1]
        pr_ld = np.linalg.slogdet(Prec)[1]
        mu    = np.linalg.solve(Prec, b)
        quad  = (r @ r) / sigma2 + float(m0.T @ S0_inv @ m0) - float(mu.T @ Prec @ mu)
        return -0.5 * (T*np.log(2*np.pi*sigma2) + quad + s0_ld - pr_ld)

    @staticmethod
    def _loglik_kf_rw_1d(r: np.ndarray, Q: float, R: float) -> float:
        # 1D random walk: F=1, H=1, diffuse start
        x_pred = 0.0; P_pred = 1e6
        ll = 0.0
        for t in range(len(r)):
            S = P_pred + R
            v = r[t] - x_pred
            ll += -0.5*(np.log(2*np.pi*S) + (v*v)/S)
            K = P_pred / S
            x_filt = x_pred + K*v
            P_filt = (1.0 - K)*P_pred
            x_pred = x_filt
            P_pred = P_filt + Q
        return float(ll)

    def _log_evidence_dyn_rw_fs(self, r: np.ndarray, B0: float, obs_var: float) -> float:
        # FS prior on Q: p(Q|σ²) ∝ Q^{λ-1} exp(-Q/(B0*σ²)), with λ = 0.5*(1-T)
        T = len(r)
        lam = 0.5*(1.0 - T)
        R = float(obs_var)

        def logpost(Q):
            if Q <= 0: return -np.inf
            ll = self._loglik_kf_rw_1d(r, Q, R)
            prior = (lam-1.0)*np.log(Q) - Q/(B0*R)
            return ll + prior

        # Mode find (coarse-to-fine on positive line)
        q = max(1e-8, B0*R)
        f0 = logpost(q)
        for s in (0.25, 0.5, 2.0, 4.0, 8.0):
            q_try = max(1e-12, s*q)
            f_try = logpost(q_try)
            if f_try > f0:
                q, f0 = q_try, f_try
        # Newton-like using symmetric finite differences
        for _ in range(12):
            eps = max(1e-8, 0.1*q)
            f1 = logpost(q - eps); f2 = logpost(q + eps)
            g  = (f2 - f1) / (2*eps)
            H  = (f2 - 2*f0 + f1) / (eps*eps)
            if H <= 0: break
            q_new = max(1e-12, q + g/H)
            f_new = logpost(q_new)
            if abs(f_new - f0) < 1e-6: 
                q, f0 = q_new, f_new
                break
            q, f0 = q_new, f_new
        # Laplace correction
        eps = max(1e-8, 0.05*q)
        f1 = logpost(q - eps); f2 = logpost(q + eps)
        H  = (f2 - 2*f0 + f1) / (eps*eps)
        H  = max(H, 1e-9)
        return f0 + 0.5*np.log(2*np.pi/H)

    # ------------------------- FFBS on dynamic subset ------------------------- #
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

    # ------------- Innovation sums of squares (for Q updates) ------------- #
    def _innovation_ss_alpha(self) -> Tuple[float, int]:
        ss = 0.0
        for t in range(1, self.T + 1):
            drift = (self.x[t - 1, self.idx_beta] if self.delta_beta == 1 else self.theta_beta)
            mean = self.x[t - 1, self.idx_alpha] + drift
            ss += (self.x[t, self.idx_alpha] - mean) ** 2
        return float(ss), self.T

    def _innovation_ss_beta(self) -> Tuple[float, int]:
        d = self.x[1:, self.idx_beta] - self.x[:-1, self.idx_beta]
        return float(np.sum(d * d)), self.T

    def _innovation_ss_gamma(self) -> Tuple[float, int]:
        if self.K_seas == 0: return 0.0, 0
        gs = self.idx_g_start; ge = gs + self.K_seas - 1
        ss = 0.0
        for t in range(1, self.T + 1):
            prev = self.x[t - 1, gs:ge + 1]
            mean_new_first = -float(np.sum(prev))
            ss += (self.x[t, gs] - mean_new_first) ** 2
        return float(ss), self.T

    # ------------------- Select δ jointly (before FFBS) ------------------- #
    def _pattern_list(self) -> List[Tuple[int, int, int]]:
        pats = [(1,1,1), (1,0,1), (1,1,0), (0,0,1), (0,0,0)]
        if self.K_seas == 0:
            pats = [(1,1,0), (1,0,0), (0,0,0)]
        return pats

    def _delta_evidences(self) -> Dict[str, Dict[str, float]]:
        """Compute per-block log-evidence for dynamic (1) and static (0) using residuals."""
        stats: Dict[str, Dict[str, float]] = {}

        # α: static vs dynamic on residual minus seasonal and (if static α) trend
        r_a = self._residual_minus_alpha()
        Z1 = np.ones((self.T, 1))
        m0a = np.array([self.priors.m_m0_alpha], float)
        s0a = self.priors.s_m0_alpha
        logE_a0 = self._log_evidence_static_reg(r_a, Z1, m0a, s0a, self.sigma2)
        logE_a1 = self._log_evidence_dyn_rw_fs(r_a, self.priors.B0_alpha, self.sigma2)
        stats["alpha"] = {"logE0": logE_a0, "logE1": logE_a1}

        # γ (seasonal): static dummy design with sum-to-zero vs dynamic RW on first coord
        if self.K_seas > 0:
            r_g = self._residual_minus_gamma()
            K = self.K_seas
            midx = np.arange(self.T) % self.period
            Zg = np.zeros((self.T, K))
            for k in range(K):
                Zg[:, k] = (midx == k).astype(float) - (midx == K).astype(float)
            m0g = np.zeros(K) if self.priors.m_m0_gamma is None else np.asarray(self.priors.m_m0_gamma, float)
            s0g = self.priors.s_m0_gamma
            logE_g0 = self._log_evidence_static_reg(r_g, Zg, m0g, s0g, self.sigma2)
            logE_g1 = self._log_evidence_dyn_rw_fs(r_g, self.priors.B0_gamma, self.sigma2)
            stats["gamma"] = {"logE0": logE_g0, "logE1": logE_g1}

        # β (trend): only meaningful if α is dynamic (admissible patterns enforce β=0 if α=0)
        # Use alpha increments as data for β block evidences
        if self.delta_alpha == 1:
            a_inc = self.x[1:, self.idx_alpha] - self.x[:-1, self.idx_alpha]
            # static beta: d_t = θβ + noise, noise var ≈ Qα (proxy by max of current Qα and tiny floor)
            Qalpha_eff = max(self.Q_alpha, 1e-8)
            Zb = np.ones((self.T, 1))
            m0b = np.array([self.priors.m_m0_beta], float)
            s0b = self.priors.s_m0_beta
            logE_b0 = self._log_evidence_static_reg(a_inc, Zb, m0b, s0b, Qalpha_eff)
            # dynamic beta: β_t random walk observed through d_t ≈ β_{t-1} + ξ_t (obs var ≈ Qα)
            logE_b1 = self._log_evidence_dyn_rw_fs(a_inc, self.priors.B0_beta, Qalpha_eff)
        else:
            # if α is currently static, we can't form a_inc from states; treat β evidences as neutral
            logE_b0 = 0.0
            logE_b1 = -1e300  # forbid β dynamic if α static
        stats["beta"] = {"logE0": logE_b0, "logE1": logE_b1}

        return stats

    def _joint_select_deltas(self):
        evid = self._delta_evidences()
        pats = self._pattern_list()

        logs = []
        for (da, db, dg) in pats:
            lp = 0.0
            # prior on indicators
            for flag, pi in zip((da, db, dg), (self.priors.pi_alpha, self.priors.pi_beta, self.priors.pi_gamma)):
                p = min(max(pi,1e-12), 1-1e-12)
                lp += math.log(p if flag==1 else (1-p))
            # block evidences
            logm = (evid["alpha"]["logE1"] if da==1 else evid["alpha"]["logE0"])
            logm += (evid["beta"]["logE1"]  if db==1 else evid["beta"]["logE0"])
            if self.K_seas > 0:
                logm += (evid["gamma"]["logE1"] if dg==1 else evid["gamma"]["logE0"])
            logs.append(lp + logm)

        m = max(logs)
        w = np.exp(np.asarray(logs) - m); w /= w.sum()
        idx = np.random.choice(len(pats), p=w)
        self.delta_alpha, self.delta_beta, self.delta_gamma = pats[idx]

        # per-block posterior inclusion probs (by marginalizing over patterns)
        def _marg_prob(which: int, val: int) -> float:
            mask = [int(p[which] == val) for p in pats]
            num = _logsumexp([logs[i] for i in range(len(pats)) if mask[i]])
            den = _logsumexp(logs)
            return float(np.exp(num - den))

        p1_a = _marg_prob(0, 1)
        p1_b = _marg_prob(1, 1)
        p1_g = _marg_prob(2, 1) if self.K_seas>0 else float("nan")

        self._last_delta_stats = {
            "alpha": {"p1": p1_a, "logE1": evid["alpha"]["logE1"], "logE0": evid["alpha"]["logE0"]},
            "beta":  {"p1": p1_b, "logE1": evid["beta"]["logE1"],  "logE0": evid["beta"]["logE0"]},
        }
        if self.K_seas>0:
            self._last_delta_stats["gamma"] = {
                "p1": p1_g, "logE1": evid["gamma"]["logE1"], "logE0": evid["gamma"]["logE0"]
            }

        # enforce identifiability: if α is static -> β must be static
        if self.delta_alpha == 0:
            self.delta_beta = 0

        return self._last_delta_stats

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
    def _mu_vec(self) -> np.ndarray:
        mu = np.zeros(self.T, float)
        if self.delta_alpha == 1:
            mu += self.x[1:self.T+1, self.idx_alpha]
        else:
            mu += self.theta_alpha
            mu += self.theta_beta * np.arange(self.T, dtype=float)
        if self.K_seas > 0:
            if self.delta_gamma == 1:
                mu += self.x[1:self.T+1, self.idx_g_start]
            else:
                mu += self.theta_gamma[np.arange(self.T) % self.period]
        return mu

    def _update_theta_alpha_static(self):
        r = self._residual_minus_alpha()
        s2 = float(self.sigma2)
        m0, s0 = float(self.priors.m_m0_alpha), float(self.priors.s_m0_alpha)
        prec = self.T / s2 + 1.0 / (s0**2)
        mean = ((r.sum() / s2) + m0 / (s0**2)) / prec
        var  = 1.0 / prec
        self.theta_alpha = float(np.random.normal(mean, math.sqrt(var)))

    def _update_theta_beta_static(self):
        if self.delta_alpha == 1:
            d = self.x[1:, self.idx_alpha] - self.x[:-1, self.idx_alpha]
            s2 = float(max(self.Q_alpha, 1e-10))
            m0, s0 = float(self.priors.m_m0_beta), float(self.priors.s_m0_beta)
            prec = (self.T / s2) + 1.0 / (s0**2)
            mean = ((float(np.sum(d)) / s2) + m0 / (s0**2)) / prec
            var  = 1.0 / prec
            self.theta_beta = float(np.random.normal(mean, math.sqrt(var)))
        else:
            t = np.arange(self.T, dtype=float)
            r = self.y.copy()
            r -= self.theta_alpha
            if self.K_seas > 0:
                if self.delta_gamma == 1:
                    r -= self.x[1:self.T+1, self.idx_g_start]
                else:
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
        r = self._residual_minus_gamma()
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

    # ------------------- FS Q-updates (only if δ=1) ------------------- #
    def _logpost_Q_fs(self, Q: float, SS: float, T_eff: int, B0: float, sigma2: float) -> float:
        if Q <= 0: return -np.inf
        lam = 0.5 * (1.0 - T_eff)
        return (lam - 1.0) * math.log(Q) - 0.5 * (SS / Q) - 0.5 * (Q / (B0 * sigma2))

    def _update_Q_block(self, Q_curr: float, SS: float, T_eff: int, B0: float, sigma2: float) -> float:
        if T_eff <= 0: return 0.0
        # initialize at FS mode of the slab posterior
        T1 = T_eff + 1.0; b = B0 * sigma2
        disc = T1*T1 + 4.0 * SS / max(b, 1e-300)
        mode = 0.5 * b * (-T1 + math.sqrt(disc))
        w = max(1e-6, 0.5 * (mode + 1e-6))
        return _slice_sample_positive(lambda q: self._logpost_Q_fs(q, SS, T_eff, B0, sigma2),
                                      max(Q_curr, mode if mode > 0 else 1e-6), w=w)

    # ------------------- Rolling window accounting ------------------- #
    def _window_reset(self, print_every: int):
        K = self.K_seas
        self._win = {
            "n": 0,
            "sigma": 0.0,
            "Q_alpha": 0.0, "Q_beta": 0.0, "Q_gamma": 0.0,
            "delta_alpha": 0.0, "delta_beta": 0.0, "delta_gamma": 0.0,
            "m0_alpha": 0.0, "P0_alpha": 0.0, "m0_beta": 0.0, "P0_beta": 0.0,
            "theta_alpha": 0.0, "theta_beta": 0.0,
            "m0_gamma": np.zeros(max(K,0), float),
            "P0_gamma": 0.0,
            "theta_gamma": np.zeros(self.period if K>0 else 0, float),
            "p1_alpha": 0.0, "p1_beta": 0.0, "p1_gamma": 0.0,
            "logE1_alpha": 0.0, "logE0_alpha": 0.0,
            "logE1_beta":  0.0, "logE0_beta":  0.0,
            "logE1_gamma": 0.0, "logE0_gamma": 0.0,
            "print_every": int(print_every),
        }

    def _window_push(self, delta_stats: Dict[str, Dict[str, float]]):
        w = self._win; w["n"] += 1
        w["sigma"] += math.sqrt(self.sigma2)
        w["Q_alpha"] += self.Q_alpha; w["Q_beta"] += self.Q_beta
        if self.K_seas>0: w["Q_gamma"] += self.Q_gamma

        w["delta_alpha"] += float(self.delta_alpha)
        w["delta_beta"]  += float(self.delta_beta)
        if self.K_seas>0: w["delta_gamma"] += float(self.delta_gamma)

        w["m0_alpha"] += self.m0_alpha; w["P0_alpha"] += self.P0_alpha
        w["m0_beta"]  += self.m0_beta;  w["P0_beta"]  += self.P0_beta
        w["theta_alpha"] += self.theta_alpha; w["theta_beta"] += self.theta_beta

        if self.K_seas>0:
            w["m0_gamma"] += self.m0_gamma
            w["P0_gamma"] += self.P0_gamma
            w["theta_gamma"] += self.theta_gamma

        # inclusion probabilities and evidences (per-iteration stats)
        w["p1_alpha"]    += delta_stats["alpha"]["p1"]
        w["logE1_alpha"] += delta_stats["alpha"]["logE1"]; w["logE0_alpha"] += delta_stats["alpha"]["logE0"]
        w["p1_beta"]     += delta_stats["beta"]["p1"]
        w["logE1_beta"]  += delta_stats["beta"]["logE1"];  w["logE0_beta"]  += delta_stats["beta"]["logE0"]
        if self.K_seas>0 and "gamma" in delta_stats:
            w["p1_gamma"]     += delta_stats["gamma"]["p1"]
            w["logE1_gamma"]  += delta_stats["gamma"]["logE1"]; w["logE0_gamma"] += delta_stats["gamma"]["logE0"]

    def _progress_line_window_mean(self, it: int) -> str:
        w = self._win; n = max(1, w["n"])
        f = lambda x: x / n
        parts = [f"[it {it + 1}/{self.cfg.n_iter}]",
                 f"σ={f(w['sigma']):.3f}",
                 f"Qα={f(w['Q_alpha']):.4g} δα={f(w['delta_alpha']):.3f}",
                 f"Qβ={f(w['Q_beta']):.4g} δβ={f(w['delta_beta']):.3f}"]
        if self.K_seas > 0:
            parts.append(f"Qγ={f(w['Q_gamma']):.4g} δγ={f(w['delta_gamma']):.3f}")

        # Display dynamic vs static per block based on mean δ ≥ 0.5 in window
        show_dyn_alpha = (f(w["delta_alpha"]) >= 0.5)
        show_dyn_beta  = (f(w["delta_beta"])  >= 0.5)
        show_dyn_gamma = (f(w["delta_gamma"]) >= 0.5) if self.K_seas>0 else False

        if show_dyn_alpha:
            parts.append(f"m0α={f(w['m0_alpha']):.4g} P0α={f(w['P0_alpha']):.4g}")
        else:
            parts.append(f"θα={f(w['theta_alpha']):.4g}")

        if show_dyn_beta:
            parts.append(f"m0β={f(w['m0_beta']):.4g} P0β={f(w['P0_beta']):.4g}")
        else:
            parts.append(f"θβ={f(w['theta_beta']):.4g}")

        if self.K_seas > 0:
            if show_dyn_gamma:
                gtxt = "[" + ", ".join(f"{x:.4g}" for x in (w["m0_gamma"]/n)) + "]"
                parts.append(f"m0γ={gtxt} P0γ={f(w['P0_gamma']):.4g}")
            else:
                gtxt = "[" + ", ".join(f"{x:.4g}" for x in (w["theta_gamma"]/n)) + "]"
                parts.append(f"θγ={gtxt}")

        tail = f"P(δα=1)={f(w['p1_alpha']):.3f} P(δβ=1)={f(w['p1_beta']):.3f}"
        if self.K_seas > 0: tail += f" P(δγ=1)={f(w['p1_gamma']):.3f}"
        parts.append(tail)

        evid = (f"logEα[1/0]={f(w['logE1_alpha']):.2f}/{f(w['logE0_alpha']):.2f}  "
                f"logEβ[1/0]={f(w['logE1_beta']):.2f}/{f(w['logE0_beta']):.2f}")
        if self.K_seas > 0:
            evid += f"  logEγ[1/0]={f(w['logE1_gamma']):.2f}/{f(w['logE0_gamma']):.2f}"
        parts.append(evid)

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

        inc_counts = np.zeros(3, int)  # alpha, beta, gamma
        print_every = cfg.progress_every if cfg.progress_every > 0 else max(1, cfg.n_iter // 50) or 1
        self._window_reset(print_every)

        for it in range(cfg.n_iter):
            # 1) Select δ jointly using local evidences (conditional on current other parts)
            delta_stats = self._joint_select_deltas()

            # 2) FFBS on dynamic subset (skip if all δ=0)
            self._ffbs()

            # 3) Update Q's under chosen pattern (slice on FS slab posterior)
            if self.delta_alpha == 1:
                SS_a, T_a = self._innovation_ss_alpha()
                self.Q_alpha = self._update_Q_block(self.Q_alpha, SS_a, T_a, self.priors.B0_alpha, self.sigma2)
            else:
                self.Q_alpha = 0.0
            if self.delta_beta == 1:
                SS_b, T_b = self._innovation_ss_beta()
                self.Q_beta  = self._update_Q_block(self.Q_beta,  SS_b, T_b, self.priors.B0_beta,  self.sigma2)
            else:
                self.Q_beta = 0.0
            if self.K_seas > 0:
                if self.delta_gamma == 1:
                    SS_g, T_g = self._innovation_ss_gamma()
                    self.Q_gamma = self._update_Q_block(self.Q_gamma, SS_g, T_g, self.priors.B0_gamma, self.sigma2)
                else:
                    self.Q_gamma = 0.0

            # 4) Update dynamic initial states (m0, P0) where δ=1
            self._update_m0_P0_dynamic()

            # 5) Update static parameters θ where δ=0
            if self.delta_alpha == 0: self._update_theta_alpha_static()
            if self.delta_beta  == 0: self._update_theta_beta_static()
            if self.K_seas > 0 and self.delta_gamma == 0: self._update_theta_gamma_static()

            # 6) Update σ²
            self._update_sigma2()

            # Push into rolling window
            self._window_push(delta_stats)

            # progress (windowed means)
            if cfg.progress and (((it + 1) % print_every == 0) or (it == cfg.n_iter - 1)):
                print(self._progress_line_window_mean(it))
                self._window_reset(print_every)

            # tally inclusions (for overall probabilities)
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

        inc_probs = inc_counts / float(cfg.n_iter)
        self.inclusion_probabilities_ = {
            "alpha": float(inc_probs[0]),
            "beta":  float(inc_probs[1]),
            "gamma": float(inc_probs[2] if self.K_seas>0 else np.nan),
        }

        # MAP model among saved draws
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
        description=("Kalman FFBS + FS-SSVS (JOINT δ move) for process variances in a Gaussian DLM "
                     "(level/trend/season). Newest-first seasonal ordering. "
                     "Progress shows Q, δ and m0/P0 (if dynamic) or θ (if static).")
    )

    # Simulation
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--period", type=int, default=4)
    p.add_argument("--start-date", type=str, default="2000-01-01")
    p.add_argument("--level-mode", choices=["dynamic", "deterministic"], default="deterministic")
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
    p.add_argument("--pi-alpha", type=float, default=0.1)
    p.add_argument("--pi-beta",  type=float, default=0.1)
    p.add_argument("--pi-gamma", type=float, default=0.1)

    # Sampler configuration
    p.add_argument("--n-iter", type=int, default=10000)
    p.add_argument("--burn", type=int, default=5000)
    p.add_argument("--thin", type=int, default=2)
    p.add_argument("--seed", type=int, default=40)
    p.add_argument("--progress", default=True)
    p.add_argument("--progress-every", type=int, default=10)
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

    # Truth overlays (optional)
    true_sigma = mts.sigma
    mu_truth = mu_T

    if args.print_summary:
        print(f"\nSimulated {args.T} observations (σ={true_sigma}) with modes "
              f"{args.level_mode}/{args.trend_mode}/{args.seasonal_mode}.\n")
        print("FS scales (ψ_k | σ² ~ N(0, B0_k σ²)) and SSVS priors:")
        print(f"  B0_alpha={priors.B0_alpha:.3g}, B0_beta={priors.B0_beta:.3g}, B0_gamma={priors.B0_gamma:.3g}")
        print(f"  π_alpha={priors.pi_alpha:.2f}, π_beta={priors.pi_beta:.2f}, π_gamma={priors.pi_gamma:.2f}\n")

    # Run
    import time
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
        plt.title(f"DLM FS-SSVS (JOINT δ first) ({args.level_mode}/{args.trend_mode}/{args.seasonal_mode})")
        plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()
