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
    """
    Univariate slice sampler on (0, ∞). logf: callable(q) returning log-density up to const.
    x0 > 0 initial. w: bracket width. m: max steps to expand/shrink.
    """
    x = max(1e-12, float(x0))
    fx = logf(x)
    # Draw vertical level
    u = rng.uniform()
    y = fx - np.log(1.0/u)
    # Create bracket [L, R]
    L = x - w * rng.uniform()
    R = L + w
    # Expand bracket
    j = int(m * rng.uniform()); k = (m - 1) - j
    while j > 0 and L > 0 and logf(max(L, 1e-18)) > y:
        L -= w; j -= 1
    while k > 0 and logf(R) > y:
        R += w; k -= 1
    if L <= 0: L = 1e-18
    # Sample from the slice
    for _ in range(100):
        x_new = rng.uniform(L, R)
        if logf(x_new) >= y:
            return max(1e-18, x_new)
        if x_new < x:
            L = x_new
        else:
            R = x_new
    return max(1e-18, x)  # fallback

# =============================================================================
# Priors & Config
# =============================================================================

@dataclass
class PriorsFS:
    # Observation variance σ²: precision τ = 1/σ² ~ Gamma(a_sigma, b_sigma) (shape–rate)
    a_sigma: float = 2.0
    b_sigma: float = 1.0

    # m0 priors (both for dynamic x0 means and deterministic components)
    m_m0_alpha: float = 0.0
    s_m0_alpha: float = 10.0
    m_m0_beta: float = 0.0
    s_m0_beta: float = 10.0
    m_m0_gamma: Optional[Sequence[float]] = None  # length p-1 if provided (NEWEST-FIRST)
    s_m0_gamma: float = 5.0

    # Initial-state variances P0 ~ InvGamma(a_P0_*, b_P0_*)
    a_P0_alpha: float = 2.0
    b_P0_alpha: float = 1.0
    a_P0_beta: float  = 2.0
    b_P0_beta: float  = 1.0
    a_P0_gamma: float = 2.0
    b_P0_gamma: float = 1.0

    # FS scales for ψ (process SD): ψ_k | σ² ~ N(0, B0_k σ²)  ⇒ Q_k = ψ_k²
    B0_alpha: float = 10.0
    B0_beta:  float = 10.0
    B0_gamma: float = 10.0

    # Spike–slab inclusion probabilities π_k = P(δ_k=1)
    pi_alpha: float = 0.5
    pi_beta:  float = 0.5
    pi_gamma: float = 0.5

@dataclass
class SamplerConfig:
    n_iter: int = 4000
    burn: int = 1000
    thin: int = 1
    random_seed: Optional[int] = 7
    progress: bool = True
    progress_every: int = 0  # 0 => ~2% of n_iter

# =============================================================================
# DLM Sampler — FS prior (non-centered), with spike–slab
# =============================================================================

class DLMGibbs_FS_SSVS:
    """
    Structural Gaussian DLM with FS prior and SSVS over blocks:
      - Blocks k ∈ {alpha (level), beta (slope), gamma (seasonal)}
      - Indicators δ_k ∈ {0,1}: 1 = dynamic (RW with Q_k>0), 0 = static parameter (no state noise)
      - If δ_k=0, block k is excluded from FFBS and sampled conjugately as a coefficient.
      - If δ_beta=1, enforce δ_alpha=1 (dynamic trend requires dynamic level).

    FS prior: ψ_k = sqrt(Q_k) | σ² ~ N(0, B0_k σ²), so Q_k has a GIG-like posterior.
    """

    # --------------------------- Construction --------------------------- #
    def __init__(
        self,
        y: np.ndarray,
        period: int,
        # initial values
        sigma2_init: float = 1.0,
        Q_alpha_init: float = 1e-2,
        Q_beta_init:  float = 1e-3,
        Q_gamma_init: float = 1e-3,
        delta_alpha_init: int = 1,
        delta_beta_init:  int = 1,
        delta_gamma_init: int = 1,
        # static-parameter inits (used when δ_k=0)
        theta_alpha_init: float = 0.0,              # intercept if δ_alpha = 0
        theta_beta_init:  float = 0.0,              # fixed slope if δ_beta = 0
        theta_gamma_init: Optional[Sequence[float]] = None,  # seasonal p-1, newest-first
        # priors / cfg
        priors: PriorsFS = PriorsFS(),
        cfg: SamplerConfig = SamplerConfig(),
    ):
        self.y = np.asarray(y, float)
        self.T = int(self.y.size)
        self.p = int(period)
        if self.p < 2:
            raise ValueError("period must be >= 2")

        self.priors, self.cfg = priors, cfg
        if cfg.random_seed is not None:
            np.random.seed(cfg.random_seed)

        # Indicators (start anywhere; chain will learn)
        self.delta_alpha = int(delta_alpha_init)
        self.delta_beta  = int(delta_beta_init)
        self.delta_gamma = int(delta_gamma_init)

        # Enforce dynamic trend => dynamic level
        if self.delta_beta == 1 and self.delta_alpha == 0:
            self.delta_alpha = 1

        # Observation variance
        self.sigma2 = float(sigma2_init)

        # Process variances (for dynamic blocks)
        self.Q_alpha = float(Q_alpha_init)
        self.Q_beta  = float(Q_beta_init)
        self.Q_gamma = float(Q_gamma_init)

        # Static coefficients (for δ_k = 0)
        self.theta_alpha = float(theta_alpha_init)
        self.theta_beta  = float(theta_beta_init)
        if theta_gamma_init is None:
            self.theta_gamma = np.zeros(self.p)  # full p with sum-to-zero constraint
            if self.p > 1:
                base = np.zeros(self.p - 1)
                self.theta_gamma[:-1] = base
                self.theta_gamma[-1]  = -base.sum()
        else:
            g = np.asarray(theta_gamma_init, float)
            if g.size != self.p - 1:
                raise ValueError("theta_gamma_init must have length p-1 (newest-first)")
            self.theta_gamma = np.r_[g, -np.sum(g)]

        # Initial state containers (only for dynamic coords)
        self._rebuild_layout()
        if self.ndim_dyn > 0:
            self.x = np.zeros((self.T + 1, self.ndim_dyn))
            # diffuse-ish start for dynamics
            self.x[0] = 0.0
        else:
            self.x = np.zeros((self.T + 1, 0))

        # Optional truth overlays
        self.true_sigma = None
        self.true_Q = None
        self.true_mu_t = None

        # Storage
        self.keep: Dict[str, np.ndarray] = {}

        # Print scale proxies
        if self.cfg.progress:
            y = self.y
            sd1 = _robust_sd(np.diff(y)) if y.size >= 2 else 0.0
            sd2 = _robust_sd(np.diff(y, n=2)) if y.size >= 3 else 0.0
            print(f"[init] scale proxies: sd1={sd1:.4g}, sd2={sd2:.4g}, sdg={sd1:.4g}")

    # --------------------------- Layout rebuild --------------------------- #
    def _rebuild_layout(self) -> None:
        """
        Build dynamic-state layout given current deltas.
        If δ_beta=1, force δ_alpha=1.
        Dynamic order: [alpha?, beta?, seasonal g1..g_{p-1}?]
        """
        if self.delta_beta == 1 and self.delta_alpha == 0:
            self.delta_alpha = 1

        layout = []
        self.idx_alpha = None
        self.idx_beta = None
        self.idx_g_start = None
        self.idx_g_end = None

        if self.delta_alpha == 1:
            self.idx_alpha = len(layout); layout.append("alpha")
        if self.delta_beta == 1:
            self.idx_beta  = len(layout); layout.append("beta")
        if self.delta_gamma == 1 and self.p > 1:
            self.idx_g_start = len(layout)
            layout.extend([f"g{k}" for k in range(1, self.p)])
            self.idx_g_end = len(layout) - 1

        self._layout_dyn = layout
        self.ndim_dyn = len(layout)

    # ------------------ Design pieces for current δ ------------------ #
    def _A_dyn(self) -> np.ndarray:
        """
        Transition for dynamic part only.
        - If alpha & beta dynamic: local linear trend coupling.
        - If alpha dynamic & beta static: drift u_alpha = theta_beta.
        - If seasonal dynamic: dummy rotation (shock only first seasonal coord).
        """
        if self.ndim_dyn == 0:
            return np.zeros((0, 0))
        A = np.eye(self.ndim_dyn)
        # alpha depends on beta when both dynamic
        if self.idx_alpha is not None and self.idx_beta is not None:
            A[self.idx_alpha, self.idx_beta] = 1.0
        # seasonal rotation
        if (self.idx_g_start is not None) and (self.idx_g_end is not None):
            gs, ge = self.idx_g_start, self.idx_g_end
            K = ge - gs + 1
            A[gs, gs:ge+1] = -1.0
            if K > 1:
                A[gs+1:ge+1, gs:ge] = np.eye(K-1)
                A[gs+1:ge+1, ge] = 0.0
        return A

    def _u_dyn(self) -> np.ndarray:
        if self.ndim_dyn == 0:
            return np.zeros(0)
        u = np.zeros(self.ndim_dyn)
        # if alpha dynamic and beta static, add deterministic drift = theta_beta
        if (self.idx_alpha is not None) and (self.delta_beta == 0):
            u[self.idx_alpha] = float(self.theta_beta)
        return u

    def _Q_dyn(self) -> np.ndarray:
        if self.ndim_dyn == 0:
            return np.zeros((0, 0))
        Q = np.zeros((self.ndim_dyn, self.ndim_dyn))
        if (self.idx_alpha is not None) and (self.Q_alpha > 0.0):
            Q[self.idx_alpha, self.idx_alpha] = self.Q_alpha
        if (self.idx_beta is not None) and (self.Q_beta  > 0.0):
            Q[self.idx_beta,  self.idx_beta]  = self.Q_beta
        if (self.idx_g_start is not None) and (self.Q_gamma > 0.0):
            Q[self.idx_g_start, self.idx_g_start] = self.Q_gamma
        return Q

    def _H_dyn(self) -> np.ndarray:
        """Only dynamic contribution to observation."""
        if self.ndim_dyn == 0:
            return np.zeros((1, 0))
        h = np.zeros(self.ndim_dyn)
        if self.idx_alpha is not None:
            h[self.idx_alpha] = 1.0
        if self.idx_g_start is not None:
            h[self.idx_g_start] += 1.0
        return h.reshape(1, -1)

    def _Z_static(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build (Z, theta) for static contributions in observation:
        - If δ_alpha=0: intercept θ_α.
        - If δ_gamma=0: seasonal dummies θ_γ (with sum-to-zero).
        Trend θ_β never enters observation (it drives α's evolution).
        """
        cols = []
        thetas = []
        if self.delta_alpha == 0:
            cols.append(np.ones(self.T))
            thetas.append(self.theta_alpha)
        if self.delta_gamma == 0 and self.p > 1:
            k = self.p - 1
            midx = np.arange(self.T) % self.p
            Zg = np.zeros((self.T, k))
            for j in range(k):
                Zg[:, j] = (midx == j).astype(float) - (midx == k).astype(float)
            cols.append(Zg)                       # matrix block
            thetas.append(self.theta_gamma[:-1])  # p-1 free, last is -sum
        if not cols:
            return np.zeros((self.T, 0)), np.zeros(0)
        # stack columns (handle blocks)
        mats = []
        for c in cols:
            if c.ndim == 1:
                mats.append(c.reshape(-1, 1))
            else:
                mats.append(c)
        Z = np.concatenate(mats, axis=1)
        theta = np.concatenate([np.atleast_1d(t).astype(float) for t in thetas])
        return Z, theta

    # ------------------------- FFBS over dynamic part ------------------------- #
    def _ffbs_dyn(self) -> np.ndarray:
        if self.ndim_dyn == 0:
            return np.zeros((self.T + 1, 0))
        H, A, Q, R = self._H_dyn(), self._A_dyn(), self._Q_dyn(), float(self.sigma2)
        # mean contribution of static blocks to observations
        Z, theta = self._Z_static()
        mu_static = Z @ theta if Z.shape[1] > 0 else 0.0

        # Kalman forward
        m = np.zeros((self.T + 1, self.ndim_dyn))
        C = np.zeros((self.T + 1, self.ndim_dyn, self.ndim_dyn))
        a = np.zeros((self.T + 1, self.ndim_dyn))
        Rm = np.zeros((self.T + 1, self.ndim_dyn, self.ndim_dyn))
        m[0] = 0.0
        C[0] = 1e6 * np.eye(self.ndim_dyn)  # diffuse start
        u = self._u_dyn()

        for t in range(1, self.T + 1):
            a[t]  = A @ m[t - 1] + u
            Rm[t] = A @ C[t - 1] @ A.T + Q
            Rm[t] = 0.5 * (Rm[t] + Rm[t].T) + 1e-12 * np.eye(self.ndim_dyn)

            resid = float(self.y[t - 1] - (mu_static[t - 1] if np.ndim(mu_static) else mu_static))
            S = float(H @ Rm[t] @ H.T + R)
            K = (Rm[t] @ H.T) / S
            v = resid - float(H @ a[t])
            m[t] = a[t] + (K.flatten() * v)
            C[t] = Rm[t] - K @ (H @ Rm[t])
            C[t] = 0.5 * (C[t] + C[t].T) + 1e-12 * np.eye(self.ndim_dyn)

        # backward simulation
        x = np.zeros((self.T + 1, self.ndim_dyn))
        x[self.T] = np.random.multivariate_normal(m[self.T], C[self.T])
        I = np.eye(self.ndim_dyn)
        for t in range(self.T - 1, -1, -1):
            J = C[t] @ A.T
            J = J @ _spd_solve(Rm[t + 1], I)
            mean = m[t] + J @ (x[t + 1] - (A @ m[t] + u))
            cov  = C[t] - J @ Rm[t + 1] @ J.T
            cov  = 0.5 * (cov + cov.T)
            cov += max(0.0, 1e-12 - float(np.linalg.eigvalsh(cov).min())) * I
            x[t] = np.random.multivariate_normal(mean, cov)
        return x

    # ------------------ Helpers: μ and residuals ------------------ #
    def _mu_vec(self, x_dyn: np.ndarray) -> np.ndarray:
        H = self._H_dyn()
        Z, theta = self._Z_static()
        mu_static = Z @ theta if Z.shape[1] > 0 else 0.0
        mu = np.zeros(self.T, float)
        for t in range(1, self.T + 1):
            dyn = float(H @ x_dyn[t]) if self.ndim_dyn > 0 else 0.0
            mu[t - 1] = dyn + (mu_static[t - 1] if np.ndim(mu_static) else mu_static)
        return mu

    # ------------------ σ² | rest  (Gamma on precision) ------------------ #
    def update_sigma2(self, x_dyn: np.ndarray) -> None:
        e = self.y - self._mu_vec(x_dyn)
        a = self.priors.a_sigma + 0.5 * self.T
        b = self.priors.b_sigma + 0.5 * float(e @ e)
        tau = np.random.gamma(shape=a, scale=1.0 / b)  # precision
        self.sigma2 = 1.0 / max(tau, 1e-300)

    # ------------- Innovation SS (for Q updates) ------------- #
    def _innovation_ss_alpha(self, x_dyn: np.ndarray) -> Tuple[float, int]:
        if self.idx_alpha is None:
            return 0.0, 0
        ss = 0.0
        for t in range(1, self.T + 1):
            drift = 0.0
            if self.idx_beta is not None:
                drift = x_dyn[t - 1, self.idx_beta]
            elif self.delta_beta == 0:
                drift = float(self.theta_beta)
            mean = x_dyn[t - 1, self.idx_alpha] + drift
            ss += (x_dyn[t, self.idx_alpha] - mean) ** 2
        return float(ss), self.T

    def _innovation_ss_beta(self, x_dyn: np.ndarray) -> Tuple[float, int]:
        if self.idx_beta is None:
            return 0.0, 0
        d = x_dyn[1:, self.idx_beta] - x_dyn[:-1, self.idx_beta]
        return float(np.sum(d * d)), self.T

    def _innovation_ss_gamma(self, x_dyn: np.ndarray) -> Tuple[float, int]:
        if self.idx_g_start is None:
            return 0.0, 0
        gs, ge = self.idx_g_start, self.idx_g_end
        ss = 0.0
        for t in range(1, self.T + 1):
            prev = x_dyn[t - 1, gs:ge + 1]
            mean_new_first = -float(np.sum(prev))
            ss += (x_dyn[t, gs] - mean_new_first) ** 2
        return float(ss), self.T

    # ------------------- FS slab & indicator updates ------------------- #
    def _logpost_Q_fs(self, Q: float, SS: float, T_eff: int, B0: float, sigma2: float) -> float:
        if Q <= 0: return -np.inf
        lam = 0.5 * (1.0 - T_eff)
        return (lam - 1.0) * np.log(Q) - 0.5 * (SS / Q) - 0.5 * (Q / (B0 * sigma2))

    def _log_marginal_slab_laplace(self, SS: float, T_eff: int, B0: float, sigma2: float) -> float:
        T1 = T_eff + 1.0
        b = B0 * sigma2
        disc = T1 * T1 + 4.0 * SS / max(b, 1e-300)
        q_hat = 0.5 * b * ( -T1 + math.sqrt(disc) ); q_hat = max(q_hat, 1e-18)
        lam = 0.5 * (1.0 - T_eff)
        g = (lam - 1.0) * math.log(q_hat) - 0.5 * (SS / q_hat) - 0.5 * (q_hat / b)
        curv = (lam - 1.0) / (q_hat * q_hat) + SS / (q_hat**3 + 1e-300)
        curv = max(curv, 1e-300)
        return g + 0.5 * math.log(2.0 * math.pi) - 0.5 * math.log(curv)

    def _update_delta(self, SS: float, T_eff: int, B0: float, sigma2: float, pi: float) -> int:
        log_m1 = self._log_marginal_slab_laplace(SS, T_eff, B0, sigma2)
        eps = 1e-12 * max(sigma2, 1.0)
        log_m0 = -0.5 * T_eff * math.log(2.0 * math.pi * eps) - 0.5 * SS / max(eps, 1e-300)
        log_post1 = math.log(max(pi, 1e-12)) + log_m1
        log_post0 = math.log(max(1.0 - pi, 1e-12)) + log_m0
        mmax = max(log_post0, log_post1)
        p1 = math.exp(log_post1 - mmax) / (math.exp(log_post0 - mmax) + math.exp(log_post1 - mmax))
        return 1 if np.random.uniform() < p1 else 0

    def _update_Q(self, Q_curr: float, SS: float, T_eff: int, B0: float, sigma2: float) -> float:
        if T_eff <= 0: return 0.0
        logf = lambda q: self._logpost_Q_fs(q, SS, T_eff, B0, sigma2)
        T1 = T_eff + 1.0
        b = B0 * sigma2
        disc = T1 * T1 + 4.0 * SS / max(b, 1e-300)
        mode = 0.5 * b * ( -T1 + math.sqrt(disc) )
        w = max(1e-6, 0.5 * (mode + 1e-6))
        return _slice_sample_positive(logf, max(Q_curr, mode if mode>0 else 1e-6), w=w)

    # --------------- Static-parameter (δ=0) conjugate updates --------------- #
    def _update_theta_alpha(self, x_dyn: np.ndarray) -> None:
        if self.delta_alpha == 1: return
        # residual after removing dynamic contribution and static seasonal
        Z, theta = self._Z_static()
        H = self._H_dyn()
        mu_dyn = np.array([float(H @ x_dyn[t]) for t in range(1, self.T + 1)]) if self.ndim_dyn>0 else 0.0
        r = self.y - (Z @ theta) - mu_dyn if np.ndim(mu_dyn) else self.y - (Z @ theta) - mu_dyn
        s2 = float(self.sigma2)
        m0, s0 = float(self.priors.m_m0_alpha), float(self.priors.s_m0_alpha)
        prec = self.T / s2 + 1.0 / (s0**2)
        mean = ((r.sum() / s2) + m0 / (s0**2)) / prec
        var = 1.0 / prec
        self.theta_alpha = float(np.random.normal(mean, math.sqrt(var)))

    def _update_theta_beta(self, x_dyn: np.ndarray) -> None:
        if self.delta_beta == 1: return
        # If alpha is dynamic: use α-increments to learn slope θβ
        m0, s0 = float(self.priors.m_m0_beta), float(self.priors.s_m0_beta)
        if self.delta_alpha == 1 and self.idx_alpha is not None:
            a = x_dyn[1:, self.idx_alpha] - x_dyn[:-1, self.idx_alpha]
            s2 = max(self.Q_alpha, 1e-12)  # α innovations variance
            prec = (self.T / s2) + 1.0 / (s0**2)
            mean = ((float(np.sum(a)) / s2) + m0 / (s0**2)) / prec
            var = 1.0 / prec
            self.theta_beta = float(np.random.normal(mean, math.sqrt(var)))
        else:
            # fallback: regress y on time if alpha is static as well
            t = np.arange(self.T, dtype=float)
            Z, theta = self._Z_static()
            mu_stat = Z @ theta if Z.shape[1] > 0 else 0.0
            r = self.y - (mu_stat if np.ndim(mu_stat) else mu_stat)
            sig2 = float(self.sigma2)
            prec = (t @ t) / sig2 + 1.0 / (s0**2)
            mean = ((t @ r) / sig2 + m0 / (s0**2)) / prec
            var = 1.0 / prec
            self.theta_beta = float(np.random.normal(mean, math.sqrt(var)))

    def _update_theta_gamma(self, x_dyn: np.ndarray) -> None:
        if self.delta_gamma == 1 or self.p <= 1: return
        # seasonal dummies with sum-to-zero
        k = self.p - 1
        midx = np.arange(self.T) % self.p
        Zg = np.zeros((self.T, k))
        for j in range(k):
            Zg[:, j] = (midx == j).astype(float) - (midx == k).astype(float)
        H = self._H_dyn()
        mu_dyn = np.array([float(H @ x_dyn[t]) for t in range(1, self.T + 1)]) if self.ndim_dyn>0 else 0.0
        r = self.y - (mu_dyn if np.ndim(mu_dyn) else mu_dyn)
        if self.delta_alpha == 0:
            r -= self.theta_alpha
        sig2 = float(self.sigma2)
        s2 = float(self.priors.s_m0_gamma) ** 2
        Prec = (Zg.T @ Zg) / sig2 + np.eye(k) / s2
        b = (Zg.T @ r) / sig2
        mu = np.linalg.solve(Prec, b)
        L = np.linalg.cholesky(Prec)
        z = np.random.randn(k)
        theta = mu + np.linalg.solve(L.T, z)
        self.theta_gamma = np.r_[theta, -theta.sum()]

    # -------------------- Progress formatting (unchanged) -------------------- #
    def _progress_line(self, it: int) -> str:
        parts = [f"[it {it + 1}/{self.cfg.n_iter}]"]
        parts.append(f"σ={math.sqrt(self.sigma2):.3f}")
        parts.append(f"Qα={self.Q_alpha:.4g} δα={self.delta_alpha}")
        parts.append(f"Qβ={self.Q_beta:.4g} δβ={self.delta_beta}")
        parts.append(f"Qγ={self.Q_gamma:.4g} δγ={self.delta_gamma}")
        parts.append(f"m0α/θα={self.theta_alpha:.4g}")
        parts.append(f"m0β/θβ={self.theta_beta:.4g}")
        parts.append(f"m0γ/θγ={_format_gamma(self.theta_gamma, 5)}")
        return " | ".join(parts)

    # --------------------------------- MCMC --------------------------------- #
    def run(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        save_iters = list(range(cfg.burn, cfg.n_iter, cfg.thin))
        n_kept, keep_idx = len(save_iters), 0

        # allocate storage
        self.keep = {
            "sigma": np.zeros(n_kept),
            "mu": np.zeros((n_kept, self.T)),
            "Q_alpha": np.zeros(n_kept), "Q_beta": np.zeros(n_kept), "Q_gamma": np.zeros(n_kept),
            "delta_alpha": np.zeros(n_kept, int), "delta_beta": np.zeros(n_kept, int), "delta_gamma": np.zeros(n_kept, int),
            "theta_alpha": np.zeros(n_kept), "theta_beta": np.zeros(n_kept),
            "theta_gamma": np.zeros((n_kept, self.p)),
        }
        # dynamic states saved at observation times
        self.keep["x_dyn"] = np.zeros((n_kept, self.T, 0))  # resized on-the-fly

        print_every = cfg.progress_every if cfg.progress_every > 0 else max(1, cfg.n_iter // 50) or 1

        for it in range(cfg.n_iter):
            # 0) Rebuild layout given current deltas
            self._rebuild_layout()

            # 1) FFBS on dynamic subset
            x_dyn = self._ffbs_dyn()

            # 2) SSVS: update δ_k using innovation SS (from x_dyn)
            SS_a, Te_a = self._innovation_ss_alpha(x_dyn)
            new_delta_alpha = self._update_delta(SS_a, Te_a, self.priors.B0_alpha, self.sigma2, self.priors.pi_alpha)

            SS_b, Te_b = self._innovation_ss_beta(x_dyn)
            new_delta_beta  = self._update_delta(SS_b, Te_b, self.priors.B0_beta,  self.sigma2, self.priors.pi_beta)

            SS_g, Te_g = self._innovation_ss_gamma(x_dyn)
            new_delta_gamma = self._update_delta(SS_g, Te_g, self.priors.B0_gamma, self.sigma2, self.priors.pi_gamma)

            # Enforce δβ=1 ⇒ δα=1
            if new_delta_beta == 1 and new_delta_alpha == 0:
                new_delta_alpha = 1
            self.delta_alpha, self.delta_beta, self.delta_gamma = new_delta_alpha, new_delta_beta, new_delta_gamma

            # 3) Given δ, update Q_k when dynamic; else keep at 0
            if self.delta_alpha == 1:
                self.Q_alpha = self._update_Q(self.Q_alpha, SS_a, Te_a, self.priors.B0_alpha, self.sigma2)
            else:
                self.Q_alpha = 0.0
            if self.delta_beta == 1:
                self.Q_beta  = self._update_Q(self.Q_beta,  SS_b, Te_b, self.priors.B0_beta,  self.sigma2)
            else:
                self.Q_beta  = 0.0
            if self.delta_gamma == 1:
                self.Q_gamma = self._update_Q(self.Q_gamma, SS_g, Te_g, self.priors.B0_gamma, self.sigma2)
            else:
                self.Q_gamma = 0.0

            # 4) Static parameters (conjugate) when δ=0
            self._update_theta_alpha(x_dyn)
            self._update_theta_beta(x_dyn)
            self._update_theta_gamma(x_dyn)

            # 5) σ²
            self.update_sigma2(x_dyn)

            # 6) progress
            if cfg.progress and ((it + 1) % print_every == 0 or it == cfg.n_iter - 1):
                print(self._progress_line(it))

            # 7) save
            if it in save_iters:
                # resize x_dyn storage lazily to current dim
                if self.keep["x_dyn"].shape[2] != self.ndim_dyn:
                    new = np.zeros((n_kept, self.T, self.ndim_dyn))
                    new[:, :, :min(self.keep["x_dyn"].shape[2], self.ndim_dyn)] = self.keep["x_dyn"]
                    self.keep["x_dyn"] = new
                self.keep["mu"][keep_idx, :] = self._mu_vec(x_dyn)
                self.keep["sigma"][keep_idx] = math.sqrt(self.sigma2)
                self.keep["Q_alpha"][keep_idx] = self.Q_alpha
                self.keep["Q_beta"][keep_idx]  = self.Q_beta
                self.keep["Q_gamma"][keep_idx] = self.Q_gamma
                self.keep["delta_alpha"][keep_idx] = self.delta_alpha
                self.keep["delta_beta"][keep_idx]  = self.delta_beta
                self.keep["delta_gamma"][keep_idx] = self.delta_gamma
                self.keep["theta_alpha"][keep_idx] = self.theta_alpha
                self.keep["theta_beta"][keep_idx]  = self.theta_beta
                self.keep["theta_gamma"][keep_idx, :] = self.theta_gamma
                self.keep["x_dyn"][keep_idx, :, :] = x_dyn[1:self.T+1, :]
                keep_idx += 1

        # -------------------- Model-search summaries (PIPs & best models) --------------------
        deltas = np.c_[self.keep["delta_alpha"], self.keep["delta_beta"], self.keep["delta_gamma"]]
        pips = deltas.mean(axis=0)  # posterior inclusion probabilities
        # frequency table of visited models
        rows, counts = np.unique(deltas, axis=0, return_counts=True)
        order = np.argsort(-counts)
        rows, counts = rows[order], counts[order]
        best_map = rows[0].tolist()
        # median-probability model (include if PIP >= 0.5)
        best_mpm = (pips >= 0.5).astype(int).tolist()

        self.keep["model_search"] = {
            "pips": {"alpha": float(pips[0]), "beta": float(pips[1]), "gamma": float(pips[2])},
            "visited_models": [{"delta": r.tolist(), "freq": int(c)} for r, c in zip(rows, counts)],
            "best_map": {"delta": best_map, "freq": int(counts[0])},
            "best_mpm": {"delta": best_mpm},
        }
        # print a compact summary
        print("\n[Model search]")
        print(f"  PIP(alpha,beta,gamma) = ({pips[0]:.3f}, {pips[1]:.3f}, {pips[2]:.3f})")
        print(f"  MAP model (δα,δβ,δγ)  = {best_map} (freq {counts[0]})")
        print(f"  MPM model (δα,δβ,δγ)  = {best_mpm}")

        return self.keep


def _format_gamma(g: np.ndarray, k: int = 5) -> str:
    g = np.asarray(g).ravel()
    if g.size == 0: return "[]"
    if g.size <= k: return "[" + ", ".join(f"{x:.4g}" for x in g) + "]"
    return "[" + ", ".join(f"{x:.4g}" for x in g[:k]) + ", …]"

# ------------------------- CLI / Example run ------------------------------- #
if __name__ == "__main__":
    import argparse, os, sys, time
    from datetime import datetime
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(base_dir)
    from simulator.mean_time_series import Mean_Time_Series  # newest-first convention

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

    # --------------------------------------------------------------------------
    # Command-line arguments
    # --------------------------------------------------------------------------
    p = argparse.ArgumentParser(
        description=(
            "Gaussian DLM with FS prior + SSVS over level/trend/seasonal blocks. "
            "Sampler always runs model search (δ_k ∈ {0,1}); simulation modes only "
            "affect the synthetic data generator (Mean_Time_Series)."
        )
    )

    # Simulation (for Mean_Time_Series only)
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--period", type=int, default=4)
    p.add_argument("--start-date", type=str, default="2000-01-01")
    p.add_argument("--level-mode", choices=["dynamic", "deterministic"], default="dynamic")
    p.add_argument("--trend-mode", choices=["dynamic", "deterministic", "none"], default="dynamic")
    p.add_argument("--seasonal-mode", choices=["dynamic", "deterministic", "none"], default="dynamic")
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

    # Priors (FS). P0 priors kept for compatibility, although not used by SSVS sampler.
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

    # Frühwirth–Schnatter scales: ψ_k | σ² ~ N(0, B0_k σ²)
    p.add_argument("--B0-alpha", type=float, default=10.0)
    p.add_argument("--B0-beta",  type=float, default=10.0)
    p.add_argument("--B0-gamma", type=float, default=10.0)

    # Spike–slab inclusion probabilities π_k
    p.add_argument("--pi-alpha", type=float, default=0.5)
    p.add_argument("--pi-beta",  type=float, default=0.5)
    p.add_argument("--pi-gamma", type=float, default=0.5)

    # Sampler configuration
    p.add_argument("--n-iter", type=int, default=5000)
    p.add_argument("--burn", type=int, default=1000)
    p.add_argument("--thin", type=int, default=2)
    p.add_argument("--seed", type=int, default=40)
    p.add_argument("--progress", default=True)
    p.add_argument("--progress-every", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="results/simulations/DLM_FS_SSVS")
    p.add_argument("--plot", default=True)
    p.add_argument("--print-summary", default=True)

    # Initial inference values (for SSVS sampler)
    p.add_argument("--sigma-init", type=float, default=1.0)
    p.add_argument("--Q-alpha-init", type=float, default=1e-1)
    p.add_argument("--Q-beta-init",  type=float, default=1e-1)
    p.add_argument("--Q-gamma-init", type=float, default=1e-1)
    p.add_argument("--delta-alpha-init", type=int, choices=[0, 1], default=1)
    p.add_argument("--delta-beta-init",  type=int, choices=[0, 1], default=1)
    p.add_argument("--delta-gamma-init", type=int, choices=[0, 1], default=1)
    p.add_argument("--theta-alpha-init", type=float, default=0.0)
    p.add_argument("--theta-beta-init",  type=float, default=0.0)
    p.add_argument("--theta-gamma-init", type=str, default=None)  # CSV p-1 newest-first

    args = p.parse_args()
    np.random.seed(args.seed)

    # Simulate
    start_date = _parse_date(args.start_date)
    m0_season = _csv_floats_or_none(args.m0_season) if hasattr(args, "m0_season") else None
    if m0_season is None:
        m0_season = [1.0] * (args.period - 1)
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

    # Priors (FS)
    from dataclasses import asdict  # used later for meta
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

    # Sampler
    theta_gamma_init = (
        [float(z) for z in args.theta_gamma_init.split(",")] if args.theta_gamma_init else None
    )

    sampler = DLMGibbs_FS_SSVS(
        y=y, period=args.period,
        sigma2_init=args.sigma_init ** 2,
        Q_alpha_init=args.Q_alpha_init, Q_beta_init=args.Q_beta_init, Q_gamma_init=args.Q_gamma_init,
        delta_alpha_init=args.delta_alpha_init, delta_beta_init=args.delta_beta_init, delta_gamma_init=args.delta_gamma_init,
        theta_alpha_init=args.theta_alpha_init, theta_beta_init=args.theta_beta_init, theta_gamma_init=theta_gamma_init,
        priors=priors, cfg=cfg,
    )

    # Truth overlays (optional)
    sampler.true_sigma = mts.sigma
    sampler.true_Q = np.array([mts.q_level, mts.q_trend, mts.q_season], float)
    sampler.true_mu_t = mu_T

    if args.print_summary:
        print(f"\nSimulated {args.T} observations (σ={mts.sigma}) with SIM modes "
              f"{args.level_mode}/{args.trend_mode}/{args.seasonal_mode}.\n")
        print("FS scales for process SDs (ψ_k | σ² ~ N(0, B0_k σ²)) and spike priors:")
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
            "model_search": post.get("model_search", {}),
        },
    )

    # Summary + model search report
    if args.print_summary:
        print("\n--- Posterior means ---")
        print(f"σ = {np.mean(post['sigma']):.4f}")
        for k in ["alpha", "beta", "gamma"]:
            key = f"Q_{k}"
            if key in post:
                mQ = float(np.mean(post[key]))
                print(f"Q_{k} = {mQ:.4g} (√Q ≈ {math.sqrt(max(mQ, 0.0)):.4g})")

        ms = post.get("model_search", {})
        if ms:
            pips = ms["pips"]
            print("\n--- Model search ---")
            print(f"PIP(alpha)={pips['alpha']:.3f}, PIP(beta)={pips['beta']:.3f}, PIP(gamma)={pips['gamma']:.3f}")
            print(f"MAP model (δα,δβ,δγ) = {ms['best_map']['delta']} (freq {ms['best_map']['freq']})")
            print(f"MPM model (δα,δβ,δγ) = {ms['best_mpm']['delta']}")

    if args.plot:
        mu_hat = post["mu"].mean(axis=0)
        plt.figure(figsize=(10, 4))
        plt.plot(dates_T, y, label="y_t", lw=1)
        plt.plot(dates_T, mu_T, "--", label="μ_t (truth)")
        plt.plot(dates_T, mu_hat, "-.", label="μ̂_t (post mean)")
        plt.title(f"DLM-FS-SSVS (sim: {args.level_mode}/{args.trend_mode}/{args.seasonal_mode})")
        plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()
