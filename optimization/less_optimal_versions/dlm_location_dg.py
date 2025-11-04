# dlm_gibbs_doublegamma.py
from __future__ import annotations

import json, math, os, warnings
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Sequence, Tuple, List

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

# =============================================================================
# Priors & Config
# =============================================================================

@dataclass
class Priors:
    # σ²: precision τ = 1/σ² ~ Gamma(a_sigma, b_sigma) (shape–rate)
    a_sigma: float = 2.0
    b_sigma: float = 1.0

    # m0 priors
    m_m0_alpha: float = 0.0
    s_m0_alpha: float = 10.0
    m_m0_beta:  float = 0.0
    s_m0_beta:  float = 10.0
    m_m0_gamma: Optional[Sequence[float]] = None
    s_m0_gamma: float = 5.0

    # P0 ~ InvGamma
    a_P0_alpha: float = 2.0; b_P0_alpha: float = 1.0
    a_P0_beta:  float = 2.0; b_P0_beta:  float = 1.0
    a_P0_gamma: float = 2.0; b_P0_gamma: float = 1.0

    # ===== Double-Gamma hyper (heterogeneous) =====
    # local shape (shared); if you want per-block: add a_xi_alpha/beta/gamma and use them below
    a_xi: float = 1.0

    # per-block global-scale priors for κ_k^2
    d1_alpha: float = 1.0; d2_alpha: float = 1.0
    d1_beta:  float = 1.0; d2_beta:  float = 1.0
    d1_gamma: float = 1.0; d2_gamma: float = 1.0


@dataclass
class SamplerConfig:
    n_iter: int = 10000
    burn: int = 5000
    thin: int = 2
    random_seed: Optional[int] = 7
    progress: bool = True
    progress_every: int = 0  # 0 => ~2% of n_iter

# =============================================================================
# DLM Sampler with Double-Gamma shrinkage on process variances
# =============================================================================
class DLMGibbsDoubleGamma:
    """
    Gaussian structural DLM with:
      • FFBS for latent states (centered)
      • Conjugate Gibbs for σ², m0, P0, deterministic params
      • Double-Gamma shrinkage for process variances Q_k = s_k^2 (k ∈ {α,β,γ})
        using N(0, 1/τ_k) prior for s_k, τ_k ~ Ga(a_xi, a_xi * κ² / 2), κ² ~ Ga(d1,d2)
        — τ_k and κ² updated via Gibbs, s_k via a short MH step on log s_k.
    Deterministic components are estimated outside the state, as before.

    Seasonal state is NEWEST-FIRST; observation loads the first seasonal coord.
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
        m0_alpha_init: float = 0.0, P0_alpha_init: float = 1.0,
        m0_beta_init:  float = 0.0, P0_beta_init:  float = 1.0,
        m0_gamma_init: Optional[Sequence[float]] = None, P0_gamma_init: float = 1.0,
        sigma2_init: float = 1.0,
        s_alpha_init: float = 1e-2, s_beta_init: float = 1e-3, s_gamma_init: float = 1e-3,
        priors: Priors = Priors(),
        cfg: SamplerConfig = SamplerConfig(),
        # MH step sizes on log s_k
        mh_log_s_step: Tuple[float, float, float] = (0.2, 0.2, 0.2),
    ):
        # Data
        self.y = np.asarray(y, float).ravel()
        self.T = int(self.y.size)
        self.period = int(period)
        if self.period < 2: raise ValueError("period must be >= 2")

        # Modes
        ok = {"dynamic", "deterministic", "none"}
        if level_mode not in ok or trend_mode not in ok or seasonal_mode not in ok:
            raise ValueError("invalid mode")
        if trend_mode == "dynamic" and level_mode != "dynamic":
            raise ValueError("dynamic trend requires dynamic level")
        self.level_mode, self.trend_mode, self.seasonal_mode = level_mode, trend_mode, seasonal_mode

        # Priors / cfg
        self.priors, self.cfg = priors, cfg
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
            self.idx_g_start = layout.index("g1"); self.idx_g_end = self.idx_g_start + (self.period - 2)
        else:
            self.idx_g_start = self.idx_g_end = None

        # Parameters
        self.sigma2 = float(sigma2_init)
        self.s_alpha = float(s_alpha_init) if self.idx_alpha is not None else 0.0
        self.s_beta  = float(s_beta_init)  if self.idx_beta  is not None else 0.0
        self.s_gamma = float(s_gamma_init) if self.seasonal_mode == "dynamic" else 0.0

        # ==== DG hyper/aux ====
        self.a_xi = float(priors.a_xi)
        self.d1_alpha, self.d2_alpha = float(priors.d1_alpha), float(priors.d2_alpha)
        self.d1_beta,  self.d2_beta  = float(priors.d1_beta),  float(priors.d2_beta)
        self.d1_gamma, self.d2_gamma = float(priors.d1_gamma), float(priors.d2_gamma)
        # component-specific κ² (globals) and τ (locals)
        self.kappa2_alpha = 1.0
        self.kappa2_beta  = 1.0
        self.kappa2_gamma = 1.0

        self.tau_alpha = 1.0
        self.tau_beta  = 1.0
        self.tau_gamma = 1.0

        # MH step sizes on log s
        self.mh_log_s_step = dict(alpha=mh_log_s_step[0], beta=mh_log_s_step[1], gamma=mh_log_s_step[2])
        self._mh_acc = dict(alpha=0, beta=0, gamma=0); self._mh_tr = dict(alpha=0, beta=0, gamma=0)

        # Initial m0/P0 for dynamic coords
        self.m0_alpha = float(m0_alpha_init) if self.idx_alpha is not None else 0.0
        self.P0_alpha = float(P0_alpha_init) if self.idx_alpha is not None else 0.0
        self.m0_beta  = float(m0_beta_init)  if self.idx_beta  is not None else 0.0
        self.P0_beta  = float(P0_beta_init)  if self.idx_beta  is not None else 0.0
        if self.seasonal_mode == "dynamic":
            if m0_gamma_init is None:
                self.m0_gamma = np.zeros(self.period - 1, float)
            else:
                g = np.asarray(m0_gamma_init, float)
                if g.size != self.period - 1: raise ValueError("m0_gamma_init must have length p-1 (newest-first)")
                self.m0_gamma = g
            self.P0_gamma = float(P0_gamma_init)
        else:
            self.m0_gamma = None; self.P0_gamma = 0.0

        # Deterministic contributions (outside state, as before)
        if self.level_mode == "deterministic":
            self.m0_alpha = float(self.priors.m_m0_alpha)
        if self.trend_mode == "deterministic":
            self.m0_beta = float(self.priors.m_m0_beta)
        if self.seasonal_mode == "deterministic":
            base = (np.zeros(self.period - 1, float) if self.priors.m_m0_gamma is None
                    else np.asarray(self.priors.m_m0_gamma, float))
            if base.size != self.period - 1: raise ValueError("priors.m_m0_gamma must be length p-1")
            self.m0_gamma = np.r_[base, -float(np.sum(base))].astype(float)

        # Latent path (centered)
        self.x = np.zeros((self.T + 1, self.dim), float)
        if self.dim > 0:
            m0_vec, P0_diag = self._current_m0_P0()
            self.x[0] = np.random.multivariate_normal(m0_vec, np.diag(P0_diag) + 1e-10 * np.eye(self.dim))
            self._propagate_initial_path(Q_init=np.full(self.dim, 1e-6, float))

        # Storage
        self.keep: Dict[str, np.ndarray] = {}

        # Print scale proxies
        if self.cfg.progress:
            y = self.y
            sd1 = _robust_sd(np.diff(y)) if y.size >= 2 else 0.0
            sd2 = _robust_sd(np.diff(y, n=2)) if y.size >= 3 else 0.0
            print(f"[init] scale proxies: sd1={sd1:.4g}, sd2={sd2:.4g}")
            
        # ---- Optional truth overlays (for simulation benchmarking) ----
        self.true_sigma: Optional[float] = None
        self.true_Q: Optional[np.ndarray] = None
        self.true_m0_level: Optional[float] = None
        self.true_m0_trend: Optional[float] = None
        self.true_m0_season: Optional[np.ndarray] = None
        self.true_P0_level: Optional[float] = None
        self.true_P0_trend: Optional[float] = None
        self.true_P0_season: Optional[np.ndarray] = None

        self.true_mu_t: Optional[np.ndarray] = None
        self.true_alpha_t: Optional[np.ndarray] = None
        self.true_beta_t: Optional[np.ndarray] = None
        self.true_gamma_t: Optional[np.ndarray] = None


    # ----------------------------- Model matrices ----------------------------- #
    def _H(self) -> np.ndarray:
        if self.dim == 0: return np.zeros((1, 0))
        h = np.zeros(self.dim, float)
        if self.idx_alpha is not None: h[self.idx_alpha] = 1.0
        if self.seasonal_mode == "dynamic": h[self.idx_g_start] = 1.0
        return h.reshape(1, -1)

    def _A(self) -> np.ndarray:
        if self.dim == 0: return np.zeros((0, 0))
        A = np.eye(self.dim)
        if self.idx_alpha is not None and self.idx_beta is not None:
            A[self.idx_alpha, self.idx_beta] = 1.0
        if self.seasonal_mode == "dynamic":
            gs, ge = self.idx_g_start, self.idx_g_end
            K = ge - gs + 1
            A[gs, gs:ge+1] = -1.0
            A[gs+1:ge+1, gs:ge] = np.eye(K-1)
            A[gs+1:ge+1, ge] = 0.0
        return A

    def _u(self) -> np.ndarray:
        if self.dim == 0: return np.zeros(0, float)
        u = np.zeros(self.dim, float)
        if (self.idx_alpha is not None) and (self.trend_mode == "deterministic"):
            u[self.idx_alpha] = float(self.m0_beta)
        return u

    def _Q(self) -> np.ndarray:
        if self.dim == 0: return np.zeros((0, 0))
        Q = np.zeros((self.dim, self.dim))
        if self.idx_alpha is not None and self.s_alpha > 0: Q[self.idx_alpha, self.idx_alpha] = self.s_alpha**2
        if self.idx_beta  is not None and self.s_beta  > 0: Q[self.idx_beta,  self.idx_beta ] = self.s_beta**2
        if self.seasonal_mode == "dynamic" and self.s_gamma > 0:
            Q[self.idx_g_start, self.idx_g_start] = self.s_gamma**2
        return Q

    def _mu_det(self, t: int) -> float:
        out = 0.0
        if self.level_mode == "deterministic": out += self.m0_alpha
        if (self.trend_mode == "deterministic") and (self.idx_alpha is None): out += self.m0_beta * t
        if self.seasonal_mode == "deterministic": out += float(self.m0_gamma[t % self.period])
        return out

    def _current_m0_P0(self) -> Tuple[np.ndarray, np.ndarray]:
        m0, P0 = [], []
        if self.idx_alpha is not None: m0.append(self.m0_alpha); P0.append(self.P0_alpha)
        if self.idx_beta  is not None: m0.append(self.m0_beta ); P0.append(self.P0_beta)
        if self.seasonal_mode == "dynamic":
            m0.extend(list(self.m0_gamma)); P0.extend([self.P0_gamma] * (self.period - 1))
        return np.asarray(m0, float), np.asarray(P0, float)

    # ------------------------- FFBS (centered) ------------------------- #
    def _ffbs(self) -> np.ndarray:
        if self.dim == 0: return self.x.copy()
        H, A, Q, R = self._H(), self._A(), self._Q(), float(self.sigma2)
        m0_vec, P0_diag = self._current_m0_P0()
        m = np.zeros((self.T + 1, self.dim)); C = np.zeros((self.T + 1, self.dim, self.dim))
        a = np.zeros((self.T + 1, self.dim)); Rm = np.zeros((self.T + 1, self.dim, self.dim))
        m[0] = m0_vec; C[0] = np.diag(P0_diag) + 1e-12 * np.eye(self.dim)
        u = self._u()

        # forward
        for t in range(1, self.T + 1):
            a[t]  = A @ m[t - 1] + u
            Rm[t] = A @ C[t - 1] @ A.T + Q
            Rm[t] = 0.5 * (Rm[t] + Rm[t].T) + 1e-12 * np.eye(self.dim)

            resid_mean = float(self.y[t - 1] - self._mu_det(t - 1))
            S = float(H @ Rm[t] @ H.T + R)
            if S <= 0: S = float(H @ (Rm[t] + 1e-10 * np.eye(self.dim)) @ H.T + R)
            K = (Rm[t] @ H.T) / S
            v = resid_mean - float(H @ a[t])
            m[t] = a[t] + (K.flatten() * v)
            C[t] = Rm[t] - K @ (H @ Rm[t])
            C[t] = 0.5 * (C[t] + C[t].T) + 1e-12 * np.eye(self.dim)

        # backward
        x = np.zeros_like(self.x)
        x[self.T] = np.random.multivariate_normal(m[self.T], C[self.T])
        for t in range(self.T - 1, -1, -1):
            J = C[t] @ A.T; J = J @ _spd_solve(Rm[t + 1], np.eye(self.dim))
            mean = m[t] + J @ (x[t + 1] - (A @ m[t] + u))
            cov  = C[t] - J @ Rm[t + 1] @ J.T
            cov  = 0.5 * (cov + cov.T)
            evmin = float(np.linalg.eigvalsh(cov).min())
            if evmin < 1e-12: cov += (1e-12 - evmin) * np.eye(cov.shape[0])
            x[t] = np.random.multivariate_normal(mean, cov)
        return x

    def _propagate_initial_path(self, Q_init: np.ndarray) -> None:
        if self.dim == 0: return
        A, u = self._A(), self._u()
        for t in range(1, self.T + 1):
            self.x[t] = A @ self.x[t - 1] + u + np.random.normal(0.0, np.sqrt(Q_init), size=self.dim)

    # ------------------ Helpers: μ and residuals ------------------ #
    def _mu_vec(self) -> np.ndarray:
        H = self._H(); mu = np.zeros(self.T, float)
        for t in range(1, self.T + 1):
            dyn = float(H @ self.x[t]) if self.dim > 0 else 0.0
            mu[t - 1] = self._mu_det(t - 1) + dyn
        return mu

    # innovations sums of squares for each block
    def _innovation_ss_alpha(self) -> Tuple[float, int]:
        if self.idx_alpha is None: return 0.0, 0
        ss = 0.0
        for t in range(1, self.T + 1):
            drift = 0.0
            if self.idx_beta is not None: drift = self.x[t - 1, self.idx_beta]
            elif self.trend_mode == "deterministic": drift = float(self.m0_beta)
            mean = self.x[t - 1, self.idx_alpha] + drift
            ss += (self.x[t, self.idx_alpha] - mean) ** 2
        return float(ss), self.T

    def _innovation_ss_beta(self) -> Tuple[float, int]:
        if self.idx_beta is None: return 0.0, 0
        d = self.x[1:, self.idx_beta] - self.x[:-1, self.idx_beta]
        return float(np.sum(d * d)), self.T

    def _innovation_ss_gamma(self) -> Tuple[float, int]:
        if self.seasonal_mode != "dynamic": return 0.0, 0
        gs, ge = self.idx_g_start, self.idx_g_end
        ss = 0.0
        for t in range(1, self.T + 1):
            prev = self.x[t - 1, gs:ge + 1]
            mean_new_first = -float(np.sum(prev))
            ss += (self.x[t, gs] - mean_new_first) ** 2
        return float(ss), self.T

    # ------------------ σ² | rest  (Gamma on precision) ------------------ #
    def update_sigma2(self) -> None:
        e = self.y - self._mu_vec()
        a = self.priors.a_sigma + 0.5 * self.T
        b = self.priors.b_sigma + 0.5 * float(e @ e)
        tau = np.random.gamma(shape=a, scale=1.0 / b)
        self.sigma2 = 1.0 / max(tau, 1e-300)

    # =============================================================================
    # Double-Gamma updates for process variances Q_k = s_k^2
    # =============================================================================
    def _mh_update_s(self, name: str, s_curr: float, tau: float, ss: float, T_eff: int) -> float:
        if T_eff == 0: return s_curr
        step = self.mh_log_s_step[name]
        log_s_curr = math.log(max(s_curr, 1e-12))
        # propose on log-scale
        log_s_prop = log_s_curr + np.random.normal(0.0, step)
        s_prop = math.exp(log_s_prop)

        def logpost(s: float) -> float:
            if s <= 0: return -np.inf
            # RW(1) likelihood for innovations (Gaussian with var s^2)
            lp = - (T_eff/2) * math.log(s*s) - ss / (2 * s*s)
            # DG prior: s | tau ~ N(0, 1/tau)
            lp += 0.5 * math.log(tau) - 0.5 * tau * (s*s)
            return lp

        lp_curr = logpost(math.exp(log_s_curr)) + log_s_curr  # Jacobian for log-transform
        lp_prop = logpost(s_prop) + log_s_prop
        acc = lp_prop - lp_curr

        self._mh_tr[name] += 1
        if math.log(np.random.rand()) < acc:
            self._mh_acc[name] += 1
            return s_prop
        return s_curr

    def _gibbs_tau(self, s2: float, kappa2: float) -> float:
        # tau | s, kappa2  ~ Ga(a_xi + 1/2,  a_xi*kappa2/2 + s^2/2)
        a = self.a_xi + 0.5
        b = self.a_xi * kappa2 / 2.0 + 0.5 * s2
        return np.random.gamma(shape=a, scale=1.0 / b)

    def _gibbs_kappa2_single(self, tau_k: float, d1: float, d2: float) -> float:
        # κ_k^2 | τ_k  ~ Ga(d1 + a_xi,  d2 + 0.5*a_xi*τ_k)
        shape = float(d1) + self.a_xi
        rate  = float(d2) + 0.5 * self.a_xi * float(tau_k)
        return np.random.gamma(shape=shape, scale=1.0 / rate)


    def update_process_Q_doublegamma(self) -> None:
        # α
        if self.idx_alpha is not None:
            ss, T_eff = self._innovation_ss_alpha()
            self.s_alpha = self._mh_update_s("alpha", self.s_alpha, self.tau_alpha, ss, T_eff)
            self.tau_alpha = self._gibbs_tau(self.s_alpha**2, self.kappa2_alpha)
            self.kappa2_alpha = self._gibbs_kappa2_single(
                self.tau_alpha, self.priors.d1_alpha, self.priors.d2_alpha
            )

        # β
        if self.idx_beta is not None:
            ss, T_eff = self._innovation_ss_beta()
            self.s_beta = self._mh_update_s("beta", self.s_beta, self.tau_beta, ss, T_eff)
            self.tau_beta = self._gibbs_tau(self.s_beta**2, self.kappa2_beta)
            self.kappa2_beta = self._gibbs_kappa2_single(
                self.tau_beta, self.priors.d1_beta, self.priors.d2_beta
            )

        # γ (first seasonal coord only)
        if self.seasonal_mode == "dynamic":
            ss, T_eff = self._innovation_ss_gamma()
            self.s_gamma = self._mh_update_s("gamma", self.s_gamma, self.tau_gamma, ss, T_eff)
            self.tau_gamma = self._gibbs_tau(self.s_gamma**2, self.kappa2_gamma)
            self.kappa2_gamma = self._gibbs_kappa2_single(
                self.tau_gamma, self.priors.d1_gamma, self.priors.d2_gamma
            )


    # --- m0 | P0, x0  +  P0 | m0, x0 (same as before; conjugate) --- #
    @staticmethod
    def _gibbs_m0_scalar(x0: float, m_prior: float, s_prior: float, P0: float) -> float:
        prec = 1.0 / (s_prior**2) + 1.0 / max(1e-18, P0)
        var = 1.0 / prec
        mean = var * (m_prior / (s_prior**2) + x0 / max(1e-18, P0))
        return float(np.random.normal(mean, math.sqrt(var)))

    def update_m0(self) -> None:
        if self.dim == 0: return
        pos = 0
        if self.idx_alpha is not None:
            self.m0_alpha = self._gibbs_m0_scalar(float(self.x[0, pos]), self.priors.m_m0_alpha, self.priors.s_m0_alpha, self.P0_alpha)
            pos += 1
        if self.idx_beta is not None:
            self.m0_beta = self._gibbs_m0_scalar(float(self.x[0, pos]), self.priors.m_m0_beta, self.priors.s_m0_beta, self.P0_beta)
            pos += 1
        if self.seasonal_mode == "dynamic":
            m_prior = (np.zeros(self.period - 1, float) if self.priors.m_m0_gamma is None
                       else np.asarray(self.priors.m_m0_gamma, float))
            if m_prior.size != self.period - 1: raise ValueError("priors.m_m0_gamma must be length p-1")
            s = float(self.priors.s_m0_gamma)
            for k in range(self.period - 1):
                self.m0_gamma[k] = self._gibbs_m0_scalar(float(self.x[0, pos + k]), float(m_prior[k]), s, self.P0_gamma)

    def update_P0(self) -> None:
        if self.dim == 0: return
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
            for k in range(self.period - 1): diffsq += (float(self.x[0, pos + k]) - float(self.m0_gamma[k])) ** 2
            a = self.priors.a_P0_gamma + 0.5 * (self.period - 1)
            b = self.priors.b_P0_gamma + 0.5 * diffsq
            self.P0_gamma = 1.0 / np.random.gamma(shape=a, scale=1.0 / b)

    # --- Deterministic params (unchanged) --- #
    def update_deterministic_params(self) -> None:
        # Same conjugate blocks as your original code:
        # level (intercept) if deterministic, trend if deterministic, seasonal dummies if deterministic
        if self.level_mode == "deterministic":
            r = self.y.copy()
            if self.dim > 0:
                H = self._H()
                for t in range(1, self.T + 1): r[t - 1] -= float(H @ self.x[t])
            if self.seasonal_mode == "deterministic":
                r -= self.m0_gamma[np.arange(self.T) % self.period]
            s2 = float(self.sigma2)
            m0, s0 = float(self.priors.m_m0_alpha), float(self.priors.s_m0_alpha)
            prec = self.T / s2 + 1.0 / (s0**2)
            mean = ((r.sum() / s2) + m0 / (s0**2)) / prec
            var = 1.0 / prec
            self.m0_alpha = float(np.random.normal(mean, math.sqrt(var)))

        if self.trend_mode == "deterministic":
            if self.idx_alpha is not None:
                d = self.x[1:, self.idx_alpha] - self.x[:-1, self.idx_alpha]
                s2 = float(self.s_alpha**2) if self.s_alpha > 0 else 1e-12
                m0, s0 = float(self.priors.m_m0_beta), float(self.priors.s_m0_beta)
                prec = (self.T / s2) + 1.0 / (s0**2)
                mean = ((float(np.sum(d)) / s2) + m0 / (s0**2)) / prec
                var = 1.0 / prec
                self.m0_beta = float(np.random.normal(mean, math.sqrt(var)))
            else:
                t = np.arange(self.T, dtype=float)
                r = self.y.copy()
                if self.dim > 0:
                    H = self._H()
                    for k in range(1, self.T + 1): r[k - 1] -= float(H @ self.x[k])
                if self.level_mode == "deterministic": r -= self.m0_alpha
                if self.seasonal_mode == "deterministic": r -= self.m0_gamma[np.arange(self.T) % self.period]
                m0, s0 = float(self.priors.m_m0_beta), float(self.priors.s_m0_beta)
                sig2 = float(self.sigma2)
                prec = (t @ t) / sig2 + 1.0 / (s0**2)
                mean = ((t @ r) / sig2 + m0 / (s0**2)) / prec
                var = 1.0 / prec
                self.m0_beta = float(np.random.normal(mean, math.sqrt(var)))

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
                for t in range(1, self.T + 1): r[t - 1] -= float(H @ self.x[t])
            if self.level_mode == "deterministic": r -= self.m0_alpha
            if (self.idx_alpha is None) and (self.trend_mode == "deterministic"): r -= self.m0_beta * np.arange(self.T, dtype=float)

            K = self.period - 1
            m_prior = (np.zeros(K) if self.priors.m_m0_gamma is None
                       else np.asarray(self.priors.m_m0_gamma, float).reshape(-1))
            s2 = float(self.priors.s_m0_gamma) ** 2
            Z = self._Z_season; sig2 = float(self.sigma2)
            Prec = (Z.T @ Z) / sig2 + np.eye(K) / s2
            b = (Z.T @ r) / sig2 + m_prior / s2
            mu = np.linalg.solve(Prec, b)
            L = np.linalg.cholesky(Prec)
            z = np.random.randn(K)
            theta = mu + np.linalg.solve(L.T, z)
            self.m0_gamma = np.r_[theta, -theta.sum()]

    # ------------------- Progress formatting ------------------- #
    def _mh_rate(self, k: str) -> float:
        t = max(1, self._mh_tr[k]); return 100.0 * (self._mh_acc[k] / t)

    def _progress_line(self, it: int) -> str:
        parts = [f"[it {it + 1}/{self.cfg.n_iter}]", f"σ={math.sqrt(self.sigma2):.3f}"]
        if self.idx_alpha is not None: parts.append(f"Qα={self.s_alpha**2:.4g} (MH {self._mh_rate('alpha'):.1f}%)")
        if self.idx_beta  is not None: parts.append(f"Qβ={self.s_beta**2:.4g} (MH {self._mh_rate('beta'):.1f}%)")
        if self.seasonal_mode == "dynamic": parts.append(f"Qγ={self.s_gamma**2:.4g} (MH {self._mh_rate('gamma'):.1f}%)")
        return " | ".join(parts)
    
        # --------------------- Truth overlays (optional) --------------------- #
    def set_truth(
        self,
        sigma: Optional[float] = None,
        Q: Optional[Sequence[float]] = None,
        m0_level: Optional[float] = None,
        m0_trend: Optional[float] = None,
        m0_season: Optional[Sequence[float]] = None,
        P0_level: Optional[float] = None,
        P0_trend: Optional[float] = None,
        P0_season: Optional[Sequence[float]] = None,
    ) -> None:
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
        self.true_mu_t = None if mu is None else np.asarray(mu, float)
        self.true_alpha_t = None if alpha is None else np.asarray(alpha, float)
        self.true_beta_t = None if beta is None else np.asarray(beta, float)
        self.true_gamma_t = None if gamma is None else np.asarray(gamma, float)


    # --------------------------------- MCMC --------------------------------- #
    def run(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        save_iters = list(range(cfg.burn, cfg.n_iter, cfg.thin))
        n_kept, keep_idx = len(save_iters), 0

        # allocate storage (compatible with previous version)
        self.keep = {"sigma": np.zeros(n_kept, float), "mu": np.zeros((n_kept, self.T), float)}
        if self.idx_alpha is not None:
            self.keep.update({"Q_alpha": np.zeros(n_kept), "m0_alpha": np.zeros(n_kept), "P0_alpha": np.zeros(n_kept)})
        if self.idx_beta is not None:
            self.keep.update({"Q_beta": np.zeros(n_kept), "m0_beta": np.zeros(n_kept), "P0_beta": np.zeros(n_kept)})
        if self.seasonal_mode == "dynamic":
            self.keep.update({"Q_gamma": np.zeros(n_kept),
                              "m0_gamma": np.zeros((n_kept, self.period - 1)), "P0_gamma": np.zeros(n_kept),
                              "x": np.zeros((n_kept, self.T, self.dim))})
        elif self.dim > 0:
            self.keep["x"] = np.zeros((n_kept, self.T, self.dim))
        if self.level_mode == "deterministic":   self.keep["m0_alpha"] = np.zeros(n_kept)
        if self.trend_mode == "deterministic":   self.keep["m0_beta"]  = np.zeros(n_kept)
        if self.seasonal_mode == "deterministic": self.keep["m0_gamma"] = np.zeros((n_kept, self.period))

        print_every = cfg.progress_every if cfg.progress_every > 0 else max(1, cfg.n_iter // 50) or 1

        for it in range(cfg.n_iter):
            # 1) FFBS (centered)
            if self.dim > 0:
                self.x = self._ffbs()

            # 2) DG process variances
            if self.dim > 0:
                self.update_process_Q_doublegamma()

            # 3) m0 and 4) P0 (conjugate)
            if self.dim > 0:
                self.update_m0()
                self.update_P0()

            # 5) deterministic parameters (conjugate)
            self.update_deterministic_params()

            # 6) σ² (Gibbs)
            self.update_sigma2()

            # progress
            if cfg.progress and ((it + 1) % print_every == 0 or it == cfg.n_iter - 1):
                print(self._progress_line(it))

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

        # include truths if present (for compatibility with older runners)
        if self.true_sigma is not None: arrays["true_sigma"] = float(self.true_sigma)
        if self.true_Q is not None: arrays["true_Q"] = np.asarray(self.true_Q, float)
        if self.true_mu_t is not None: arrays["true_mu_t"] = np.asarray(self.true_mu_t, float)
        if self.true_alpha_t is not None: arrays["true_alpha_t"] = np.asarray(self.true_alpha_t, float)
        if self.true_beta_t is not None: arrays["true_beta_t"] = np.asarray(self.true_beta_t, float)
        if self.true_gamma_t is not None: arrays["true_gamma_t"] = np.asarray(self.true_gamma_t, float)

        np.savez_compressed(out_npz_path, **arrays)

        meta = {
            "T": int(self.T), "dim": int(self.dim), "period": int(self.period),
            "modes": {
                "level_mode": self.level_mode,
                "trend_mode": self.trend_mode,
                "seasonal_mode": self.seasonal_mode,
            },
            "layout": list(self._layout),
            "cfg": asdict(self.cfg),
            "priors": asdict(self.priors),
            "dg_hyper": {
                "a_xi": self.a_xi,
                "d_alpha": [self.priors.d1_alpha, self.priors.d2_alpha],
                "d_beta":  [self.priors.d1_beta,  self.priors.d2_beta],
                "d_gamma": [self.priors.d1_gamma, self.priors.d2_gamma],
                "kappa2_end": {
                "alpha": float(self.kappa2_alpha),
                "beta":  float(self.kappa2_beta),
                "gamma": float(self.kappa2_gamma),
                },
            },
            "mh_accept": {k: float(self._mh_rate(k)) for k in ("alpha","beta","gamma")},
        }
        if extra_meta: meta.update(extra_meta)
        meta_path = out_npz_path.replace(".npz", ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[save] Posterior -> {out_npz_path}")
        print(f"[save] Metadata  -> {meta_path}")


if __name__ == "__main__":
    import argparse, os, sys, time
    from datetime import datetime
    import numpy as np
    import matplotlib.pyplot as plt

    # optional simulator (keeps your original wiring)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(base_dir)
    try:
        from simulator.mean_time_series import Mean_Time_Series  # newest-first convention
        HAS_SIM = True
    except Exception:
        HAS_SIM = False

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

    p = argparse.ArgumentParser(
        description=(
            "Kalman FFBS + conjugate Gibbs for Gaussian DLM "
            "(level/trend/season) with Double-Gamma shrinkage on process variances. "
            "Uses newest-first seasonal ordering; deterministic components are estimated "
            "outside the state (no tiny-Q hacks)."
        )
    )

    # Simulation (optional; if simulator is missing, y must be provided via --y-csv)
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--period", type=int, default=4)
    p.add_argument("--start-date", type=str, default="2000-01-01")
    p.add_argument("--level-mode", choices=["dynamic", "deterministic"], default="dynamic")
    p.add_argument("--trend-mode", choices=["dynamic", "deterministic", "none"], default="dynamic")
    p.add_argument("--seasonal-mode", choices=["dynamic", "deterministic", "none"], default="none")

    # If simulating:
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

    # Or provide data directly:
    p.add_argument("--y-csv", type=str, default=None, help="Path to CSV with a single column y")

    # Priors (observation variance, initials, DG hyper)
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

    # Double-Gamma hyperparameters
    p.add_argument("--dg-a-xi", type=float, default=0.1, help="local shape a_xi (1 ≈ Bayesian Lasso)")
    p.add_argument("--dg-d1-alpha", type=float, default=1.0)
    p.add_argument("--dg-d2-alpha", type=float, default=1e3)
    p.add_argument("--dg-d1-beta",  type=float, default=1.0)
    p.add_argument("--dg-d2-beta",  type=float, default=1e3)
    p.add_argument("--dg-d1-gamma", type=float, default=1.0)
    p.add_argument("--dg-d2-gamma", type=float, default=1e3)

    # d2 = d1/(2 s_target^2)

    # MH step sizes on log s_k (alpha, beta, gamma)
    p.add_argument("--mh-log-s-steps", type=str, default="0.2,0.2,0.2",
                   help="Comma-separated RW stddevs for log s_alpha, log s_beta, log s_gamma")

    # Sampler configuration
    p.add_argument("--n-iter", type=int, default=5000)
    p.add_argument("--burn", type=int, default=1000)
    p.add_argument("--thin", type=int, default=1)
    p.add_argument("--seed", type=int, default=40)
    p.add_argument("--progress", default=True)
    p.add_argument("--progress-every", type=int, default=100)
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

    args = p.parse_args()
    np.random.seed(args.seed)

    # Parse MH steps
    try:
        mh_steps_tuple = tuple(float(z) for z in args.mh_log_s_steps.split(","))
        if len(mh_steps_tuple) != 3:
            raise ValueError
    except Exception:
        raise ValueError("--mh-log-s-steps must be three comma-separated floats, e.g. 0.2,0.2,0.2")

    # Data (simulate if possible and no CSV provided)
    if args.y_csv is not None:
        y = np.loadtxt(args.y_csv, delimiter=",").astype(float).ravel()
        mu_T = None  # no truth available
    else:
        if not HAS_SIM:
            raise RuntimeError("No simulator found and no --y-csv provided.")
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

    # Priors
    pri_gamma_vec = _csv_floats_or_none(args.prior_m_m0_gamma)
    priors = Priors(
        a_sigma=args.prior_a_sigma, b_sigma=args.prior_b_sigma,
        m_m0_alpha=args.prior_m_m0_alpha, s_m0_alpha=args.prior_s_m0_alpha,
        m_m0_beta=args.prior_m_m0_beta,  s_m0_beta=args.prior_s_m0_beta,
        m_m0_gamma=None if pri_gamma_vec is None else pri_gamma_vec,
        s_m0_gamma=args.prior_s_m0_gamma,
        a_P0_alpha=args.prior_a_P0_alpha, b_P0_alpha=args.prior_b_P0_alpha,
        a_P0_beta=args.prior_a_P0_beta,   b_P0_beta=args.prior_b_P0_beta,
        a_P0_gamma=args.prior_a_P0_gamma, b_P0_gamma=args.prior_b_P0_gamma,
        a_xi=args.dg_a_xi,
        d1_alpha=args.dg_d1_alpha, d2_alpha=args.dg_d2_alpha,
        d1_beta=args.dg_d1_beta,   d2_beta=args.dg_d2_beta,
        d1_gamma=args.dg_d1_gamma, d2_gamma=args.dg_d2_gamma,
    )


    cfg = SamplerConfig(
        n_iter=int(args.n_iter), burn=int(args.burn), thin=int(args.thin),
        random_seed=int(args.seed), progress=bool(args.progress),
        progress_every=int(args.progress_every),
    )

    # Sampler
    m0_gamma_init = (
        [float(z) for z in args.m0_gamma_init.split(",")] if args.m0_gamma_init else None
    )

    sampler = DLMGibbsDoubleGamma(
        y=y, period=args.period,
        level_mode=args.level_mode, trend_mode=args.trend_mode, seasonal_mode=args.seasonal_mode,
        sigma2_init=args.sigma_init ** 2,
        s_alpha_init=args.s_alpha_init, s_beta_init=args.s_beta_init, s_gamma_init=args.s_gamma_init,
        m0_alpha_init=args.m0_level, m0_beta_init=args.m0_trend,
        P0_alpha_init=args.P0_alpha_init, P0_beta_init=args.P0_beta_init,
        m0_gamma_init=m0_gamma_init, P0_gamma_init=args.P0_gamma_init,
        priors=priors, cfg=cfg, mh_log_s_step=mh_steps_tuple,
    )

    # Truth overlays (if simulated)
    if args.y_csv is None and HAS_SIM:
        sampler.set_truth(
            sigma=mts.sigma, Q=(mts.q_level, mts.q_trend, mts.q_season),
            m0_level=mts.m0_level, m0_trend=mts.m0_trend, m0_season=mts.m0_season,
            P0_level=mts.v0_level, P0_trend=mts.v0_trend, P0_season=mts.v0_season,
        )
        if mu_T is not None:
            sampler.set_truth_paths(mu=mu_T)

    # Run
    t0 = time.time()
    post = sampler.run()
    elapsed = time.time() - t0
    print(f"\n[Run completed in {elapsed:.1f}s]")

    # Save
    out_dir = os.path.join(
        args.out_dir,
        f"{args.level_mode}-{args.trend_mode}-{args.seasonal_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(out_dir, exist_ok=True)
    sampler.save_posterior(
        out_npz_path=os.path.join(out_dir, "posterior.npz"),
        extra_meta={
            "elapsed_seconds": float(elapsed),
            "dg_hyper": {"a_xi": priors.a_xi, 
                         "d1_alpha": priors.d1_alpha, "d2_alpha": priors.d2_alpha,
                         "d1_beta": priors.d1_beta,   "d2_beta": priors.d2_beta,
                         "d1_gamma": priors.d1_gamma,  "d2_gamma": priors.d2_gamma},
            "mh_accept": {
                "alpha": sampler._mh_rate("alpha"),
                "beta": sampler._mh_rate("beta"),
                "gamma": sampler._mh_rate("gamma"),
            },
        },
    )

    # Summary + plot
    if args.print_summary:
        print("\n--- Posterior means ---")
        print(f"σ = {np.mean(post['sigma']):.4f}")
        for k in ["alpha", "beta", "gamma"]:
            key = f"Q_{k}"
            if key in post:
                mQ = np.mean(post[key])
                print(f"Q_{k} = {mQ:.4g} (√Q ≈ {math.sqrt(mQ):.4g})")
        print("\nMH acceptance rates for log s_k:")
        if "Q_alpha" in post:
            print(f"  alpha: {sampler._mh_rate('alpha'):.1f}%")
        if "Q_beta" in post:
            print(f"  beta : {sampler._mh_rate('beta'):.1f}%")
        if "Q_gamma" in post or args.seasonal_mode == "dynamic":
            print(f"  gamma: {sampler._mh_rate('gamma'):.1f}%")

    if args.plot:
        mu_hat = post["mu"].mean(axis=0)
        x_axis = np.arange(len(y))
        plt.figure(figsize=(10, 4))
        plt.plot(x_axis, y, label="y_t", lw=1)
        if mu_T is not None:
            plt.plot(x_axis, mu_T, "--", label="μ_t (truth)")
        plt.plot(x_axis, mu_hat, "-.", label="μ̂_t (post mean)")
        ttl = f"DLM-DG ({args.level_mode}/{args.trend_mode}/{args.seasonal_mode})"
        plt.title(ttl); plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()
