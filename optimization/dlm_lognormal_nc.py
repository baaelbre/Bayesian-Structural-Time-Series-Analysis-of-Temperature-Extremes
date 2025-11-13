from __future__ import annotations
"""
Gaussian structural time–series model with **dummy-rotation seasonality** (no harmonics)
— **Non‑Centered** parameterization via **Durbin–Koopman Simulation Smoother**
— Gibbs for Gaussian parts — and **log‑normal priors on process SDs** (slice on log‑SD).

Key differences vs centered FFBS version
----------------------------------------
• Primary latents are **state disturbances** w_t, initial state x_0, and **obs disturbances** ε_t.
  We draw (x_0, w_1:T, ε_1:T) using the Durbin–Koopman simulation smoother and reconstruct x.
• Process SD updates use the **true innovation energy** SS = \sum_t w_{k,t}^2 for each block k∈{α,β,γ}.
• Seasonality uses the **dummy-rotation** (newest-first) (p-1)-vector with \sum_{j=0}^{p-1} γ_{t,j} = 0.
• Deterministic pieces (level/trend/season) updated by conjugate normals.
• Observation variance σ² updated by Gamma on precision; can target residuals or ε-draws.

Notes
-----
- Season vector is NEWEST-FIRST of length (p-1). The implied last dummy is −sum of the (p−1).
- In dynamic mode, the **rotation** is:   γ_t = [ −1' γ_{t-1}; I_{p-2} 0 ] γ_{t-1} + e_t,
  i.e. γ_{t,1} is refreshed (its conditional mean is −1'γ_{t-1}), rest shift down.
"""

import json, math, os, sys, time, warnings
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
# Univariate slice sampler (stepping-out + shrinkage) on R
# =============================================================================
class Slice1D:
    def __init__(self, h, w: float = 1.0, m: int = 20, rng: Optional[np.random.Generator] = None):
        self.h = h; self.w = float(w); self.m = int(m)
        self.rng = rng if rng is not None else np.random.default_rng()

    def _safe(self, z: float) -> float:
        try:
            val = float(self.h(z))
            return val if np.isfinite(val) else -np.inf
        except Exception:
            return -np.inf

    def sample(self, z0: float, n: int = 1) -> np.ndarray:
        out = np.empty(n, float)
        z = float(z0)
        for i in range(n):
            logy = self._safe(z) - self.rng.exponential(1.0)
            w = self.w
            L = z - w * self.rng.random(); R = L + w
            J = int(self.rng.integers(0, self.m)); K = self.m - 1 - J
            while J > 0 and self._safe(L) > logy:
                L -= w; J -= 1
            while K > 0 and self._safe(R) > logy:
                R += w; K -= 1
            while True:
                z_new = self.rng.uniform(L, R)
                if self._safe(z_new) >= logy:
                    z = z_new; break
                elif z_new < z: L = z_new
                else: R = z_new
            out[i] = z
        return out

# =============================================================================
# Priors & Config (Log-Normal priors on process SDs)
# =============================================================================

@dataclass
class Priors:
    # Observation variance σ²: precision τ = 1/σ² ~ Gamma(a_sigma, b_sigma) (shape–rate)
    a_sigma: float = 2.0
    b_sigma: float = 1.0

    # m0 priors (used both for dynamic x0 means and deterministic params)
    m_m0_alpha: float = 0.0
    s_m0_alpha: float = 10.0
    m_m0_beta:  float = 0.0
    s_m0_beta:  float = 10.0
    m_m0_gamma: Optional[Sequence[float]] = None  # len p-1 (newest-first)
    s_m0_gamma: float = 5.0

    # Initial-state variances P0 ~ InvGamma(a, b)
    a_P0_alpha: float = 2.0
    b_P0_alpha: float = 1.0
    a_P0_beta:  float = 2.0
    b_P0_beta:  float = 1.0
    a_P0_gamma: float = 2.0
    b_P0_gamma: float = 1.0  # shared across p-1 seasonal coords

    # Log-Normal priors for process SDs: s_k ~ LogNormal(mu, sd^2)
    mu_log_s_alpha: float = -2.3
    sd_log_s_alpha: float = 0.7
    mu_log_s_beta:  float = -3.5
    sd_log_s_beta:  float = 0.7
    mu_log_s_gamma: float = -3.0
    sd_log_s_gamma: float = 0.7

@dataclass
class SamplerConfig:
    n_iter: int = 5000
    burn: int = 1000
    thin: int = 2
    random_seed: Optional[int] = 40
    progress: bool = True
    progress_every: int = 0  # 0 => ~2% of n_iter
    # slice sampler knobs (shared for all z=log s updates)
    slice_w: float = 1.0
    slice_m: int = 20
    # which σ² update: 'resid' (from y-μ) or 'eps' (from DK ε draws)
    sigma_update: str = "resid"

# =============================================================================
# DLM Sampler — Disturbance‑NCP + Dummy Rotation Seasonality
# =============================================================================

class DLMDisturbanceNCP:
    """
    Linear Gaussian SSM in **disturbance** NCP with dummy-rotation seasonality.

    States (if dynamic):
      x_t = A x_{t-1} + u + w_t,  w_t ~ N(0, Q),  ε_t ~ N(0, σ²)
      y_t = H x_t + μ_det(t) + ε_t

    We sample (x_0, w_1:T, ε_1:T) with the **Durbin–Koopman simulation smoother** and reconstruct x.
    Variance learning uses w_t sums of squares (per block) with log‑normal priors on s.
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
        m0_alpha_init: float = 0.0,
        P0_alpha_init: float = 1.0,
        m0_beta_init: float = 0.0,
        P0_beta_init: float = 1.0,
        m0_gamma_init: Optional[Sequence[float]] = None,  # len p-1 (NEWEST-FIRST)
        P0_gamma_init: float = 1.0,
        sigma2_init: float = 1.0,
        s_alpha_init: float = 1e-2,
        s_beta_init: float = 1e-3,
        s_gamma_init: float = 1e-3,
        priors: Priors = Priors(),
        cfg: SamplerConfig = SamplerConfig(),
    ):
        self.y = np.asarray(y, float)
        self.T = int(self.y.size)
        self.period = int(period)
        if self.period < 2:
            raise ValueError("period must be >= 2")

        ok = {"dynamic", "deterministic", "none"}
        if level_mode not in ok or trend_mode not in ok or seasonal_mode not in ok:
            raise ValueError("invalid mode")
        if trend_mode == "dynamic" and level_mode != "dynamic":
            raise ValueError("dynamic trend requires dynamic level")
        self.level_mode, self.trend_mode, self.seasonal_mode = level_mode, trend_mode, seasonal_mode

        self.priors, self.cfg = priors, cfg
        self._rng = np.random.default_rng(cfg.random_seed)

        # Dynamic layout (newest-first seasonal)
        layout: List[str] = []
        if self.level_mode == "dynamic":  layout.append("alpha")
        if self.trend_mode == "dynamic":  layout.append("beta")
        if self.seasonal_mode == "dynamic":
            layout.extend([f"g{k}" for k in range(1, self.period)])
        self._layout = layout
        self.dim = len(layout)

        self.idx_alpha = layout.index("alpha") if "alpha" in layout else None
        self.idx_beta  = layout.index("beta")  if "beta"  in layout else None
        if self.seasonal_mode == "dynamic":
            self.idx_g_start = layout.index("g1")
            self.idx_g_end   = self.idx_g_start + (self.period - 2)
        else:
            self.idx_g_start = self.idx_g_end = None

        # Parameters
        self.sigma2  = float(sigma2_init)
        self.s_alpha = float(s_alpha_init) if self.idx_alpha is not None else 0.0
        self.s_beta  = float(s_beta_init)  if self.idx_beta  is not None else 0.0
        self.s_gamma = float(s_gamma_init) if self.seasonal_mode == "dynamic" else 0.0

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
                if g.size != self.period - 1:
                    raise ValueError("m0_gamma_init must have length p-1 (NEWEST-FIRST)")
                self.m0_gamma = g
            self.P0_gamma = float(P0_gamma_init)
        else:
            self.m0_gamma = None
            self.P0_gamma = 0.0

        # Deterministic contributions
        if self.level_mode == "deterministic":
            self.m0_alpha = float(self.priors.m_m0_alpha)
        if self.trend_mode == "deterministic":
            self.m0_beta  = float(self.priors.m_m0_beta)
        if self.seasonal_mode == "deterministic":
            base = (
                np.zeros(self.period - 1, float)
                if self.priors.m_m0_gamma is None
                else np.asarray(self.priors.m_m0_gamma, float)
            )
            if base.size != self.period - 1:
                raise ValueError("priors.m_m0_gamma must have length p-1")
            self.m0_gamma = np.r_[base, -float(np.sum(base))].astype(float)

        # Latent containers (x for convenience; w, eps for NCP energy)
        self.x   = np.zeros((self.T + 1, self.dim), float)
        self.w   = np.zeros((self.T + 1, self.dim), float)  # w[0] stores x0−m0 proxy
        self.eps = np.zeros(self.T, float)

        # Initialize x path crudely (not needed, but helps warmup diagnostics)
        if self.dim > 0:
            m0_vec, P0_diag = self._current_m0_P0()
            self.x[0] = self._rng.multivariate_normal(m0_vec, np.diag(P0_diag) + 1e-10 * np.eye(self.dim))
            self._propagate_initial_path(Q_init=np.full(self.dim, 1e-6, float))

        # Storage
        self.keep: Dict[str, np.ndarray] = {}

        # Truth overlays
        self.true_sigma: Optional[float] = None
        self.true_Q: Optional[np.ndarray] = None
        self.true_mu_t: Optional[np.ndarray] = None

        # Print scale proxies
        if self.cfg.progress:
            y = self.y
            sd1 = _robust_sd(np.diff(y)) if y.size >= 2 else 0.0
            sd2 = _robust_sd(np.diff(y, n=2)) if y.size >= 3 else 0.0
            print(f"[init] scale proxies: sd1={sd1:.4g}, sd2={sd2:.4g}")

        # slice handles for z = log s (built lazily inside update in case params change)

    # ----------------------------- Model matrices ----------------------------- #
    def _H(self) -> np.ndarray:
        if self.dim == 0:
            return np.zeros((1, 0))
        h = np.zeros(self.dim, float)
        if self.idx_alpha is not None: h[self.idx_alpha] = 1.0
        if self.seasonal_mode == "dynamic": h[self.idx_g_start] = 1.0
        return h.reshape(1, -1)

    def _A(self) -> np.ndarray:
        if self.dim == 0:
            return np.zeros((0, 0))
        A = np.eye(self.dim)
        if self.idx_alpha is not None and self.idx_beta is not None:
            A[self.idx_alpha, self.idx_beta] = 1.0
        if self.seasonal_mode == "dynamic":
            gs, ge = self.idx_g_start, self.idx_g_end
            K = ge - gs + 1  # p-1
            A[gs, gs:ge+1] = -1.0
            if K > 1:
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
        if self.idx_alpha is not None and self.s_alpha > 0:
            Q[self.idx_alpha, self.idx_alpha] = self.s_alpha**2
        if self.idx_beta is not None and self.s_beta > 0:
            Q[self.idx_beta, self.idx_beta] = self.s_beta**2
        if self.seasonal_mode == "dynamic" and self.s_gamma > 0:
            Q[self.idx_g_start, self.idx_g_start] = self.s_gamma**2
        return Q

    def _mu_det(self, t: int) -> float:
        out = 0.0
        if self.level_mode == "deterministic": out += self.m0_alpha
        if (self.trend_mode == "deterministic") and (self.idx_alpha is None):
            out += self.m0_beta * t
        if self.seasonal_mode == "deterministic":
            out += float(self.m0_gamma[t % self.period])
        return out

    def _current_m0_P0(self) -> Tuple[np.ndarray, np.ndarray]:
        m0, P0 = [], []
        if self.idx_alpha is not None:
            m0.append(self.m0_alpha); P0.append(self.P0_alpha)
        if self.idx_beta is not None:
            m0.append(self.m0_beta);  P0.append(self.P0_beta)
        if self.seasonal_mode == "dynamic":
            m0.extend(list(self.m0_gamma))
            P0.extend([self.P0_gamma] * (self.period - 1))
        return np.asarray(m0, float), np.asarray(P0, float)

    # ====================== Durbin–Koopman simulation smoother ======================

    def _filter(self, y_series: Optional[np.ndarray] = None):
        yv = self.y if y_series is None else np.asarray(y_series, float)
        if self.dim == 0:
            a = np.zeros((self.T + 1, 0)); Rm = np.zeros((self.T + 1, 0, 0))
            v = np.zeros(self.T); F = np.ones(self.T) * self.sigma2
            K = np.zeros((self.T + 1, 0)); m = np.zeros_like(a); C = np.zeros_like(Rm)
            return a, Rm, v, F, K, m, C
        H, A, Q, R = self._H(), self._A(), self._Q(), float(self.sigma2)
        m0_vec, P0_diag = self._current_m0_P0()
        m = np.zeros((self.T + 1, self.dim))
        C = np.zeros((self.T + 1, self.dim, self.dim))
        a = np.zeros((self.T + 1, self.dim))
        Rm = np.zeros((self.T + 1, self.dim, self.dim))
        v = np.zeros(self.T); F = np.zeros(self.T)
        K = np.zeros((self.T + 1, self.dim))
        m[0] = m0_vec; C[0] = np.diag(P0_diag) + 1e-12 * np.eye(self.dim)
        u = self._u()
        for t in range(1, self.T + 1):
            a[t]  = A @ m[t - 1] + u
            Rm[t] = A @ C[t - 1] @ A.T + Q
            Rm[t] = 0.5*(Rm[t] + Rm[t].T) + 1e-12*np.eye(self.dim)
            y_det = self._mu_det(t - 1)
            v[t-1] = float(yv[t - 1] - y_det - H @ a[t])
            F[t-1] = float(H @ Rm[t] @ H.T + R)
            if F[t-1] <= 0:
                F[t-1] = float(H @ (Rm[t] + 1e-10*np.eye(self.dim)) @ H.T + R)
            Kt = (Rm[t] @ H.T) / F[t-1]
            K[t] = Kt.flatten()
            m[t] = a[t] + Kt.flatten()*v[t-1]
            C[t] = Rm[t] - Kt @ (H @ Rm[t])
            C[t] = 0.5*(C[t] + C[t].T) + 1e-12*np.eye(self.dim)
        return a, Rm, v, F, K, m, C

    def _rts_smoother(self, a, Rm, K, m_filt, C_filt) -> np.ndarray:
        if self.dim == 0: return np.zeros((self.T + 1, 0))
        A = self._A(); u = self._u()
        x_sm = np.zeros_like(m_filt)
        x_sm[self.T] = m_filt[self.T]
        for t in range(self.T - 1, -1, -1):
            J = C_filt[t] @ A.T
            Rmt1 = 0.5*(Rm[t+1] + Rm[t+1].T) + 1e-12*np.eye(self.dim)
            J = np.linalg.solve(Rmt1.T, J.T).T
            x_pred = A @ m_filt[t] + u
            x_sm[t] = m_filt[t] + J @ (x_sm[t+1] - x_pred)
        return x_sm

    def _simulate_smoother_draw(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (x[0..T], w[1..T], eps[1..T]). w[0]=x0−m0 is stored in w[:,] padding."""
        if self.dim == 0:
            # No dynamic state; disturbances are pure obs noise draws
            eps = self._rng.normal(0.0, math.sqrt(self.sigma2), size=self.T)
            return np.zeros((self.T + 1, 0)), np.zeros((self.T + 1, 0)), eps

        H, A, Q, R = self._H(), self._A(), self._Q(), float(self.sigma2)
        m0_vec, P0_diag = self._current_m0_P0()
        a, Rm, v, F, K, m_filt, C_filt = self._filter()

        # 1) simulate from prior to build pseudo series y^+
        z0 = self._rng.multivariate_normal(np.zeros(self.dim), np.diag(P0_diag) + 1e-12*np.eye(self.dim))
        w_plus = self._rng.multivariate_normal(np.zeros(self.dim), Q, size=self.T)
        e_plus = self._rng.normal(0.0, math.sqrt(R), size=self.T)
        x_plus = np.zeros((self.T + 1, self.dim))
        x_plus[0] = m0_vec + z0
        u = self._u()
        for t in range(1, self.T + 1):
            x_plus[t] = A @ x_plus[t-1] + u + w_plus[t-1]
        y_det = np.array([self._mu_det(t) for t in range(self.T)], float)
        y_plus = (H @ x_plus[1:].T).ravel() + y_det + e_plus

        # 2) smooth real y and pseudo y^+
        x_hat = self._rts_smoother(a, Rm, K, m_filt, C_filt)
        a_p, Rm_p, v_p, F_p, K_p, m_p, C_p = self._filter(y_plus)
        x_hat_p = self._rts_smoother(a_p, Rm_p, K_p, m_p, C_p)

        # 3) Simulation smoother identity
        x = x_hat + (x_plus - x_hat_p)

        # 4) Recover disturbances
        w = np.zeros_like(x)
        w[0] = x[0] - m0_vec
        for t in range(1, self.T + 1):
            w[t] = x[t] - (A @ x[t-1] + u)
        # 5) Recover ε_t from pseudo trick too
        #    ε = (y - μ_det - H x) + (e_plus - (y_plus - μ_det - H x_plus))
        resid_real = self.y - y_det - (H @ x[1:].T).ravel()
        resid_plus = y_plus - y_det - (H @ x_plus[1:].T).ravel()
        eps = resid_real + (e_plus - resid_plus)
        return x, w, eps

    # ------------------ Helpers: μ and residuals ------------------ #
    def _mu_vec(self) -> np.ndarray:
        H = self._H()
        mu = np.zeros(self.T, float)
        for t in range(1, self.T + 1):
            dyn = float(H @ self.x[t]) if self.dim > 0 else 0.0
            mu[t - 1] = self._mu_det(t - 1) + dyn
        return mu

    # ------------------ σ² | rest  (Gamma on precision) ------------------ #
    def update_sigma2(self) -> None:
        if self.cfg.sigma_update == "eps" and self.eps.size == self.T:
            e = self.eps
        else:
            e = self.y - self._mu_vec()
        a = self.priors.a_sigma + 0.5 * self.T
        b = self.priors.b_sigma + 0.5 * float(e @ e)
        tau = np.random.gamma(shape=a, scale=1.0 / b)  # precision
        self.sigma2 = 1.0 / max(tau, 1e-300)

    # ------------- Innovation sums of squares (from disturbances) ------------- #
    def _innovation_ss_alpha(self) -> Tuple[float, int]:
        if (self.idx_alpha is None) or (self.dim == 0):
            return 0.0, 0
        v = self.w[1:, self.idx_alpha]
        return float(v @ v), int(v.size)

    def _innovation_ss_beta(self) -> Tuple[float, int]:
        if (self.idx_beta is None) or (self.dim == 0):
            return 0.0, 0
        v = self.w[1:, self.idx_beta]
        return float(v @ v), int(v.size)

    def _innovation_ss_gamma(self) -> Tuple[float, int]:
        if (self.seasonal_mode != "dynamic") or (self.dim == 0):
            return 0.0, 0
        gs, ge = self.idx_g_start, self.idx_g_end
        v = self.w[1:, gs]
        return float(v @ v), int(v.size)

    # =============================================================================
    # Process SDs with log-normal priors: ln s ~ N(mu, sd^2) via slice on z=ln s
    # =============================================================================

    def _logpost_z(self, z: float, SS: float, T_eff: int, mu: float, sd: float) -> float:
        # For w_t ~ N(0, s^2): in z = ln s,  ll(z) = -T_eff z - 0.5 SS e^{-2z}
        ll = -T_eff * z - 0.5 * SS * math.exp(-2.0 * z)
        lp = -0.5 * ((z - mu) ** 2) / (sd ** 2)
        return ll + lp

    def update_process_Q_lognormal(self) -> None:
        rng = self._rng
        # α
        if self.idx_alpha is not None:
            SS, T_eff = self._innovation_ss_alpha()
            mu, sd = float(self.priors.mu_log_s_alpha), float(self.priors.sd_log_s_alpha)
            z0 = math.log(max(self.s_alpha, 1e-16))
            sl = Slice1D(lambda z: self._logpost_z(float(z), SS, T_eff, mu, sd),
                         w=self.cfg.slice_w, m=self.cfg.slice_m, rng=rng)
            z = float(sl.sample(z0, 1)[0])
            self.s_alpha = float(math.exp(z))
        # β
        if self.idx_beta is not None:
            SS, T_eff = self._innovation_ss_beta()
            mu, sd = float(self.priors.mu_log_s_beta), float(self.priors.sd_log_s_beta)
            z0 = math.log(max(self.s_beta, 1e-16))
            sl = Slice1D(lambda z: self._logpost_z(float(z), SS, T_eff, mu, sd),
                         w=self.cfg.slice_w, m=self.cfg.slice_m, rng=rng)
            z = float(sl.sample(z0, 1)[0])
            self.s_beta = float(math.exp(z))
        # γ
        if self.seasonal_mode == "dynamic":
            SS, T_eff = self._innovation_ss_gamma()
            mu, sd = float(self.priors.mu_log_s_gamma), float(self.priors.sd_log_s_gamma)
            z0 = math.log(max(self.s_gamma, 1e-16))
            sl = Slice1D(lambda z: self._logpost_z(float(z), SS, T_eff, mu, sd),
                         w=self.cfg.slice_w, m=self.cfg.slice_m, rng=rng)
            z = float(sl.sample(z0, 1)[0])
            self.s_gamma = float(math.exp(z))

    # --- m0 | P0, x0 (Normal); P0 | m0, x0 (Inv-Gamma) --- #
    @staticmethod
    def _gibbs_m0_scalar(x0: float, m_prior: float, s_prior: float, P0: float) -> float:
        prec = 1.0/(s_prior**2) + 1.0/max(1e-18, P0)
        var = 1.0/prec
        mean = var*(m_prior/(s_prior**2) + x0/max(1e-18, P0))
        var = float(np.clip(var, 1e-6, 1e6))
        mean = float(np.clip(mean, -1e6, 1e6))
        return float(np.random.normal(mean, math.sqrt(var)))

    def update_m0(self) -> None:
        if self.dim == 0: return
        pos = 0
        if self.idx_alpha is not None:
            self.m0_alpha = self._gibbs_m0_scalar(
                float(self.x[0, pos]), self.priors.m_m0_alpha, self.priors.s_m0_alpha, self.P0_alpha
            ); pos += 1
        if self.idx_beta is not None:
            self.m0_beta = self._gibbs_m0_scalar(
                float(self.x[0, pos]), self.priors.m_m0_beta, self.priors.s_m0_beta, self.P0_beta
            ); pos += 1
        if self.seasonal_mode == "dynamic":
            m_prior = (
                np.zeros(self.period - 1, float)
                if self.priors.m_m0_gamma is None
                else np.asarray(self.priors.m_m0_gamma, float)
            )
            if m_prior.size != self.period - 1:
                raise ValueError("priors.m_m0_gamma must be length p-1 (NEWEST-FIRST)")
            s = float(self.priors.s_m0_gamma)
            for k in range(self.period - 1):
                self.m0_gamma[k] = self._gibbs_m0_scalar(float(self.x[0, pos + k]), float(m_prior[k]), s, self.P0_gamma)

    def update_P0(self) -> None:
        if self.dim == 0: return
        pos = 0
        if self.idx_alpha is not None:
            a = self.priors.a_P0_alpha + 0.5
            b = self.priors.b_P0_alpha + 0.5 * (float(self.x[0, pos]) - self.m0_alpha) ** 2
            self.P0_alpha = 1.0 / np.random.gamma(shape=a, scale=1.0 / b); pos += 1
            self.P0_alpha = float(np.clip(self.P0_alpha, 1e-4, 1e4))
        if self.idx_beta is not None:
            a = self.priors.a_P0_beta + 0.5
            b = self.priors.b_P0_beta + 0.5 * (float(self.x[0, pos]) - self.m0_beta) ** 2
            self.P0_beta = 1.0 / np.random.gamma(shape=a, scale=1.0 / b); pos += 1
            self.P0_beta = float(np.clip(self.P0_beta, 1e-4, 1e4))
        if self.seasonal_mode == "dynamic":
            diffsq = 0.0
            for k in range(self.period - 1):
                diffsq += (float(self.x[0, pos + k]) - float(self.m0_gamma[k])) ** 2
            a = self.priors.a_P0_gamma + 0.5 * (self.period - 1)
            b = self.priors.b_P0_gamma + 0.5 * diffsq
            self.P0_gamma = 1.0 / np.random.gamma(shape=a, scale=1.0 / b)
            self.P0_gamma = float(np.clip(self.P0_gamma, 1e-4, 1e4))

    # --- Deterministic parameter updates (conjugate) --- #
    def update_deterministic_params(self) -> None:
        # deterministic level
        if self.level_mode == "deterministic":
            r = self.y.copy()
            if self.dim > 0:
                H = self._H()
                for t in range(1, self.T + 1):
                    r[t - 1] -= float(H @ self.x[t])
            if self.seasonal_mode == "deterministic":
                r -= self.m0_gamma[np.arange(self.T) % self.period]
            s2 = float(self.sigma2)
            m0, s0 = float(self.priors.m_m0_alpha), float(self.priors.s_m0_alpha)
            prec = self.T / s2 + 1.0 / (s0**2)
            mean = ((r.sum() / s2) + m0 / (s0**2)) / prec
            var = 1.0 / prec
            self.m0_alpha = float(np.random.normal(mean, math.sqrt(var)))

        # deterministic trend
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
                    for k in range(1, self.T + 1):
                        r[k - 1] -= float(H @ self.x[k])
                if self.level_mode == "deterministic":
                    r -= self.m0_alpha
                if self.seasonal_mode == "deterministic":
                    r -= self.m0_gamma[np.arange(self.T) % self.period]
                m0, s0 = float(self.priors.m_m0_beta), float(self.priors.s_m0_beta)
                sig2 = float(self.sigma2)
                prec = (t @ t) / sig2 + 1.0 / (s0**2)
                mean = ((t @ r) / sig2 + m0 / (s0**2)) / prec
                var = 1.0 / prec
                self.m0_beta = float(np.random.normal(mean, math.sqrt(var)))

        # deterministic season: regress on (p-1) deviation-coded dummies
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
                for t in range(1, self.T + 1):
                    r[t - 1] -= float(H @ self.x[t])
            if self.level_mode == "deterministic":
                r -= self.m0_alpha
            if (self.idx_alpha is None) and (self.trend_mode == "deterministic"):
                r -= self.m0_beta * np.arange(self.T, dtype=float)

            K = self.period - 1
            m_prior = (
                np.zeros(K)
                if self.priors.m_m0_gamma is None
                else np.asarray(self.priors.m_m0_gamma, float).reshape(-1)
            )
            s2 = float(self.priors.s_m0_gamma) ** 2
            Z = self._Z_season
            sig2 = float(self.sigma2)
            Prec = (Z.T @ Z) / sig2 + np.eye(K) / s2
            b = (Z.T @ r) / sig2 + m_prior / s2
            mu = np.linalg.solve(Prec, b)
            L = np.linalg.cholesky(Prec)
            z = np.random.randn(K)
            theta = mu + np.linalg.solve(L.T, z)
            self.m0_gamma = np.r_[theta, -theta.sum()]

    # ------------------- Progress formatting ------------------- #
    @staticmethod
    def _fmt_list(vals, max_elems: int = 6, fmt: str = ".4g") -> str:
        if vals is None: return "-"
        v = np.asarray(vals, float).ravel()
        if v.size == 0: return "[]"
        if v.size <= max_elems:
            return "[" + ", ".join(f"{x:{fmt}}" for x in v) + "]"
        head = ", ".join(f"{x:{fmt}}" for x in v[:max_elems])
        return f"[{head}, …]"

    def _progress_line(self, it: int) -> str:
        parts = [f"[it {it + 1}/{self.cfg.n_iter}]", f"σ={math.sqrt(self.sigma2):.3f}"]
        if self.idx_alpha is not None: parts.append(f"Qα={self.s_alpha**2:.4g}")
        if self.idx_beta  is not None: parts.append(f"Qβ={self.s_beta**2:.4g}")
        if self.seasonal_mode == "dynamic": parts.append(f"Qγ={self.s_gamma**2:.4g}")
        if self.level_mode != "none":
            parts.append(f"m0α={self.m0_alpha:.4g} P0α={(self.P0_alpha if self.level_mode=='dynamic' else 0.0):.4g}")
        if self.trend_mode != "none":
            parts.append(f"m0β={self.m0_beta:.4g} P0β={(self.P0_beta if self.trend_mode=='dynamic' else 0.0):.4g}")
        if self.seasonal_mode != "none":
            g = self._fmt_list((self.m0_gamma if self.seasonal_mode == "dynamic" else self.m0_gamma[:-1]), 6, ".4g")
            parts.append(f"m0γ={g} P0γ={(self.P0_gamma if self.seasonal_mode=='dynamic' else 0.0):.4g}")
        parts.append(f"σ-update={self.cfg.sigma_update}")
        return " | ".join(parts)

    # --------------------------------- MCMC --------------------------------- #
    def run(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        save_iters = list(range(cfg.burn, cfg.n_iter, cfg.thin))
        n_kept, keep_idx = len(save_iters), 0

        # allocate storage
        self.keep = {"sigma": np.zeros(n_kept, float), "mu": np.zeros((n_kept, self.T), float)}
        if self.idx_alpha is not None:
            self.keep.update({"Q_alpha": np.zeros(n_kept),
                              "m0_alpha": np.zeros(n_kept), "P0_alpha": np.zeros(n_kept)})
        if self.idx_beta is not None:
            self.keep.update({"Q_beta": np.zeros(n_kept),
                              "m0_beta": np.zeros(n_kept), "P0_beta": np.zeros(n_kept)})
        if self.seasonal_mode == "dynamic":
            self.keep.update({"Q_gamma": np.zeros(n_kept),
                              "m0_gamma": np.zeros((n_kept, self.period - 1)), "P0_gamma": np.zeros(n_kept),
                              "x": np.zeros((n_kept, self.T, self.dim))})
        elif self.dim > 0:
            self.keep["x"] = np.zeros((n_kept, self.T, self.dim))
        if self.level_mode == "deterministic":
            self.keep["m0_alpha"] = np.zeros(n_kept)
        if self.trend_mode == "deterministic":
            self.keep["m0_beta"] = np.zeros(n_kept)
        if self.seasonal_mode == "deterministic":
            self.keep["m0_gamma"] = np.zeros((n_kept, self.period))

        print_every = cfg.progress_every if cfg.progress_every > 0 else max(1, cfg.n_iter // 50) or 1

        for it in range(cfg.n_iter):
            # 1) Draw (x, w, eps) via disturbance simulation smoother
            if self.dim > 0:
                self.x, self.w, self.eps = self._simulate_smoother_draw()
            else:
                # no dynamic state: draw eps for sigma update only
                self.eps = self._rng.normal(0.0, math.sqrt(self.sigma2), size=self.T)

            # 2) Update process SDs from disturbance energy (log-normal via slice)
            if self.dim > 0:
                self.update_process_Q_lognormal()

            # 3) Update m0 and 4) P0
            if self.dim > 0:
                self.update_m0()
                self.update_P0()

            # 5) Deterministic parameters
            self.update_deterministic_params()

            # 6) σ²
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
        arrays["eps"] = getattr(self, "eps", np.zeros(0))
        arrays["w_last"] = self.w[-1] if hasattr(self, "w") else np.zeros(0)
        if self.true_sigma is not None:
            arrays["true_sigma"] = float(self.true_sigma)
        if self.true_Q is not None:
            arrays["true_Q"] = np.asarray(self.true_Q, float)
        if self.true_mu_t is not None:
            arrays["true_mu_t"] = np.asarray(self.true_mu_t, float)
        np.savez_compressed(out_npz_path, **arrays)

        meta = {
            "T": int(self.T),
            "dim": int(self.dim),
            "period": int(self.period),
            "modes": {
                "level_mode": self.level_mode,
                "trend_mode": self.trend_mode,
                "seasonal_mode": self.seasonal_mode,
            },
            "layout": list(self._layout),
            "cfg": asdict(self.cfg),
            "priors": asdict(self.priors),
        }
        if extra_meta: meta.update(extra_meta)
        meta_path = out_npz_path.replace(".npz", ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[save] Posterior -> {out_npz_path}")
        print(f"[save] Metadata  -> {meta_path}")

    # --------------------- Truth overlays (optional) --------------------- #
    def set_truth_paths(self, mu: Optional[np.ndarray] = None, **_) -> None:
        self.true_mu_t = None if mu is None else np.asarray(mu, float)

    # ------------------------------ Init helpers ------------------------------ #
    def _propagate_initial_path(self, Q_init: np.ndarray) -> None:
        if self.dim == 0: return
        A, u = self._A(), self._u()
        for t in range(1, self.T + 1):
            self.x[t] = A @ self.x[t - 1] + u + np.random.normal(0.0, np.sqrt(Q_init), size=self.dim)

# ------------------------- CLI / Example run ------------------------------- #
if __name__ == "__main__":
    import argparse
    from datetime import datetime
    import matplotlib.pyplot as plt


    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(base_dir)
    from simulator.mean_time_series import Mean_Time_Series  # newest-first

    def _parse_date(s: str | None):
        if not s: return datetime.today()
        parts = [int(p) for p in s.split("-")]
        if   len(parts) == 1: return datetime(parts[0], 1, 1)
        elif len(parts) == 2: return datetime(parts[0], parts[1], 1)
        elif len(parts) == 3: return datetime(parts[0], parts[1], parts[2])
        raise ValueError("start-date must be YYYY, YYYY-MM, or YYYY-MM-DD")

    def _csv_floats_or_none(s: str | None):
        if s is None: return None
        s = s.strip()
        if s == "":   return None
        return [float(z) for z in s.split(",") if z.strip() != ""]

    p = argparse.ArgumentParser(
        description=(
            "DLM (dummy-rotation) in **disturbance NCP** with DK simulation smoother.\n"
            "Log‑normal priors on process SDs updated via slice on log‑SD.\n"
            "Deterministic pieces by conjugate normals; σ² from residuals or ε-draws."
        )
    )

    # Simulation
    p.add_argument("--T", type=int, default=600)
    p.add_argument("--period", type=int, default=12)
    p.add_argument("--start-date", type=str, default="2000-01-01")
    p.add_argument("--level-mode", choices=["dynamic", "deterministic"], default="dynamic")
    p.add_argument("--trend-mode", choices=["dynamic", "deterministic", "none"], default="dynamic")
    p.add_argument("--seasonal-mode", choices=["dynamic", "deterministic", "none"], default="dynamic")
    p.add_argument("--sigma", type=float, default=1.5)
    p.add_argument("--q-level", type=float, default=1e-3)
    p.add_argument("--q-trend", type=float, default=2e-4)
    p.add_argument("--q-season", type=float, default=5e-4)
    p.add_argument("--m0-level", type=float, default=3.0)
    p.add_argument("--v0-level", type=float, default=0.05)
    p.add_argument("--m0-trend", type=float, default=0.02)
    p.add_argument("--v0-trend", type=float, default=0.05)
    p.add_argument("--m0-season", type=str, default=None)
    p.add_argument("--v0-season", type=str, default=None)

    # Priors
    p.add_argument("--prior-a-sigma", type=float, default=2.0)
    p.add_argument("--prior-b-sigma", type=float, default=1.0)
    p.add_argument("--prior-m-m0-alpha", type=float, default=0.0)
    p.add_argument("--prior-s-m0-alpha", type=float, default=10.0)
    p.add_argument("--prior-m-m0-beta",  type=float, default=0.0)
    p.add_argument("--prior-s-m0-beta",  type=float, default=10.0)
    p.add_argument("--prior-m-m0-gamma", type=str, default=None)
    p.add_argument("--prior-s-m0-gamma", type=float, default=5)
    p.add_argument("--prior-a-P0-alpha", type=float, default=5.0)
    p.add_argument("--prior-b-P0-alpha", type=float, default=1.0)
    p.add_argument("--prior-a-P0-beta",  type=float, default=5.0)
    p.add_argument("--prior-b-P0-beta",  type=float, default=1.0)
    p.add_argument("--prior-a-P0-gamma", type=float, default=5.0)
    p.add_argument("--prior-b-P0-gamma", type=float, default=1.0)

    # Log-Normal priors (process SDs)
    p.add_argument("--ln-mu-alpha", type=float, default=-2.0)
    p.add_argument("--ln-sd-alpha", type=float, default=0.7)
    p.add_argument("--ln-mu-beta",  type=float, default=-3.0)
    p.add_argument("--ln-sd-beta",  type=float, default=0.7)
    p.add_argument("--ln-mu-gamma", type=float, default=-4.0)
    p.add_argument("--ln-sd-gamma", type=float, default=0.7)

    # Sampler configuration
    p.add_argument("--n-iter", type=int, default=8000)
    p.add_argument("--burn", type=int, default=4000)
    p.add_argument("--thin", type=int, default=2)
    p.add_argument("--seed", type=int, default=40)
    p.add_argument("--progress", default=True)
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--slice-w", type=float, default=1.0)
    p.add_argument("--slice-m", type=int, default=20)
    p.add_argument("--sigma-update", choices=["resid", "eps"], default="eps")
    p.add_argument("--out-dir", type=str, default="results/simulations/DLM_NCP")
    p.add_argument("--plot", action="store_true")

    # Initial inference values
    p.add_argument("--sigma-init", type=float, default=1.0)
    p.add_argument("--s-alpha-init", type=float, default=1e-1)
    p.add_argument("--s-beta-init",  type=float, default=1e-1)
    p.add_argument("--s-gamma-init", type=float, default=1e-1)
    p.add_argument("--m0-gamma-init", type=str, default=None)
    p.add_argument("--P0-alpha-init", type=float, default=0.25)
    p.add_argument("--P0-beta-init",  type=float, default=0.05)
    p.add_argument("--P0-gamma-init", type=float, default=0.25)

    args = p.parse_args()
    np.random.seed(args.seed)

    start_date = _parse_date(args.start_date)
    m0_season = [5.0] * (args.period - 1)
    v0_season = [0.25] * (args.period - 1)
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
    pri_gamma_vec = [0.0] * (args.period - 1)
    priors = Priors(
        a_sigma=args.prior_a_sigma, b_sigma=args.prior_b_sigma,
        m_m0_alpha=args.prior_m_m0_alpha, s_m0_alpha=args.prior_s_m0_alpha,
        m_m0_beta=args.prior_m_m0_beta,   s_m0_beta=args.prior_s_m0_beta,
        m_m0_gamma=pri_gamma_vec,
        s_m0_gamma=args.prior_s_m0_gamma,
        a_P0_alpha=args.prior_a_P0_alpha, b_P0_alpha=args.prior_b_P0_alpha,
        a_P0_beta=args.prior_a_P0_beta,   b_P0_beta=args.prior_b_P0_beta,
        a_P0_gamma=args.prior_a_P0_gamma, b_P0_gamma=args.prior_b_P0_gamma,
        mu_log_s_alpha=args.ln_mu_alpha, sd_log_s_alpha=args.ln_sd_alpha,
        mu_log_s_beta=args.ln_mu_beta,   sd_log_s_beta=args.ln_sd_beta,
        mu_log_s_gamma=args.ln_mu_gamma, sd_log_s_gamma=args.ln_sd_gamma
    )

    cfg = SamplerConfig(
        n_iter=int(args.n_iter), burn=int(args.burn), thin=int(args.thin),
        random_seed=int(args.seed), progress=bool(args.progress),
        progress_every=int(args.progress_every),
        slice_w=float(args.slice_w), slice_m=int(args.slice_m),
        sigma_update=str(args.sigma_update),
    )

    # Sampler
    m0_gamma_init = (
        [float(z) for z in (args.m0_gamma_init or "").split(",")] if args.m0_gamma_init else None
    )

    sampler = DLMDisturbanceNCP(
        y=y, period=args.period,
        level_mode=args.level_mode, trend_mode=args.trend_mode, seasonal_mode=args.seasonal_mode,
        sigma2_init=args.sigma_init ** 2,
        s_alpha_init=args.s_alpha_init, s_beta_init=args.s_beta_init, s_gamma_init=args.s_gamma_init,
        m0_alpha_init=args.m0_level, m0_beta_init=args.m0_trend,
        P0_alpha_init=args.P0_alpha_init, P0_beta_init=args.P0_beta_init,
        m0_gamma_init=m0_gamma_init, P0_gamma_init=args.P0_gamma_init,
        priors=priors, cfg=cfg,
    )

    if HAVE_SIM:
        sampler.true_sigma = mts.sigma
        sampler.true_Q = np.array([mts.q_level, mts.q_trend, mts.q_season])
        sampler.set_truth_paths(mu=mu_T)

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
            "lognormal_priors": {
                "alpha": {"mu": priors.mu_log_s_alpha, "sd": priors.sd_log_s_alpha},
                "beta" : {"mu": priors.mu_log_s_beta , "sd": priors.sd_log_s_beta },
                "gamma": {"mu": priors.mu_log_s_gamma, "sd": priors.sd_log_s_gamma},
            },
            "slice": {"w": cfg.slice_w, "m": cfg.slice_m},
            "sigma_update": cfg.sigma_update,
        },
    )

    # Quick plot
    if args.plot:
        mu_hat = post["mu"].mean(axis=0)
        plt.figure(figsize=(10, 4))
        plt.plot(y, label="y_t", lw=1)
        plt.plot(mu_T, "--", label="μ_t (truth)")
        plt.plot(mu_hat, "-.", label="μ̂_t (post mean)")
        plt.title(f"DLM-NCP (dummy-rotation) {args.level_mode}/{args.trend_mode}/{args.seasonal_mode}")
        plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()
