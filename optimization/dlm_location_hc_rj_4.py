from __future__ import annotations

"""
Reversible‑Jump Gibbs sampler for a Gaussian structural DLM (level/trend/seasonality).

Highlights
----------
• FFBS for latent states (when blocks are dynamic).
• Conjugate Gibbs updates for observation variance (σ²), initial state means (m0_*), and
  initial state variances (P0_*).
• Half‑Cauchy priors on process standard deviations via inverse‑gamma scale mixtures
  (pure Gibbs for Q_k = s_k^2 and the auxiliary a_k).
• RJ moves toggle each block among {dynamic, deterministic, none} with a simple model prior.
  Acceptance uses a Kalman filter log‑likelihood (integrating out states) + parameter priors.
• "Escape move": if proposing trend→dynamic while level≠dynamic, flip level→dynamic jointly.
• Plotter‑friendly storage: always allocate full arrays; fill N/A with NaN. Modes stored as
  integers (0: dynamic, 1: deterministic, 2: none).

Seasonal convention
-------------------
- Period p ≥ 2. We use (p-1) explicit seasonal states with newest‑first ordering.
- Observation loads the first seasonal state (γ_t).
- Transition for season block (dynamic):
    g1(t) = -sum(g1..g_{p-1})(t-1) + ε_{γ,t}
    gk(t) = g_{k-1}(t-1),  k = 2..p-1
  ⇒ Q has s_γ² on the FIRST seasonal coord only.

Notes
-----
- RJ acceptance uses a collapsed Gaussian likelihood via the Kalman filter, evaluated under the
  *proposed* vs *current* modes and parameters (with symmetric proposal choices), hence the ratio
  reduces to [Δ loglik + Δ logprior].
- When a block becomes dynamic, its new parameters (m0, P0, s) are proposed from their priors.
  When it becomes deterministic, the deterministic parameter is proposed from its prior.
- Latent states are sampled by FFBS only *after* the RJ stage for the accepted model.
"""

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
# Priors & Config
# =============================================================================

@dataclass
class Priors:
    # Observation variance σ²: precision τ = 1/σ² ~ Gamma(a_sigma, b_sigma) (shape–rate)
    a_sigma: float = 2.0
    b_sigma: float = 1.0

    # m0 priors (used both for dynamic x0 means and for deterministic components)
    m_m0_alpha: float = 0.0
    s_m0_alpha: float = 10.0
    m_m0_beta: float = 0.0
    s_m0_beta: float = 10.0
    m_m0_gamma: Optional[Sequence[float]] = None  # length p-1 if provided (NEWEST-FIRST)
    s_m0_gamma: float = 5.0

    # Initial-state variances P0 ~ InvGamma(a_P0_*, b_P0_*)
    a_P0_alpha: float = 2.0
    b_P0_alpha: float = 1.0
    a_P0_beta: float = 2.0
    b_P0_beta: float = 1.0
    a_P0_gamma: float = 2.0
    b_P0_gamma: float = 1.0   # shared across p-1 seasonal coords

    # Half-Cauchy scales for process SDs (s_alpha, s_beta, s_gamma)
    hc_scale_alpha: float = 0.5
    hc_scale_beta:  float = 0.5
    hc_scale_gamma: float = 0.5

@dataclass
class ModelPrior:
    # Categorical priors on each block's mode. Must be nonnegative; will be renormalized.
    level:  Tuple[float, float, float] = (0.9, 0.1, 0.0)  # (dyn, det, none)
    trend:  Tuple[float, float, float] = (0.9, 0.1, 0.0)
    season: Tuple[float, float, float] = (0.9, 0.1, 0.0)

    def as_logits(self) -> Dict[str, np.ndarray]:
        out = {}
        for k, v in dict(level=self.level, trend=self.trend, season=self.season).items():
            a = np.asarray(v, float)
            a = a / max(1e-300, a.sum())
            out[k] = np.log(a + 1e-300)
        return out

@dataclass
class SamplerConfig:
    n_iter: int = 4000
    burn: int = 1000
    thin: int = 1
    random_seed: Optional[int] = 7
    progress: bool = True
    progress_every: int = 0  # 0 => ~2% of n_iter
    rj_moves_per_iter: int = 3


# =============================================================================
# Core class with RJ
# =============================================================================
class DLMRJGibbs:
    """Gaussian structural DLM with conjugate Gibbs + RJ over block modes.

    Modes per block in {"dynamic", "deterministic", "none"}, with constraint
    trend==dynamic ⇒ level==dynamic (enforced and via escape move).
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
        m0_gamma_init: Optional[Sequence[float]] = None,  # len p-1 if dynamic (NEWEST-FIRST)
        P0_gamma_init: float = 1.0,
        sigma2_init: float = 1.0,
        s_alpha_init: float = 1e-2,
        s_beta_init: float = 1e-3,
        s_gamma_init: float = 1e-3,
        priors: Priors = Priors(),
        model_prior: ModelPrior = ModelPrior(),
        cfg: SamplerConfig = SamplerConfig(),
    ):
        # Data
        self.y = np.asarray(y, float)
        self.T = int(self.y.size)
        self.period = int(period)
        if self.period < 2:
            raise ValueError("period must be >= 2")

        self.priors = priors
        self.model_prior = model_prior.as_logits()  # store as logits
        self.cfg = cfg
        if cfg.random_seed is not None:
            np.random.seed(cfg.random_seed)

        # Modes
        ok = {"dynamic", "deterministic", "none"}
        if level_mode not in ok or trend_mode not in ok or seasonal_mode not in ok:
            raise ValueError("invalid mode")
        if trend_mode == "dynamic" and level_mode != "dynamic":
            # enforce constraint
            level_mode = "dynamic"
        self.level_mode, self.trend_mode, self.seasonal_mode = level_mode, trend_mode, seasonal_mode

        # Parameters
        self.sigma2 = float(sigma2_init)  # observation variance
        self.s_alpha = float(s_alpha_init)
        self.s_beta  = float(s_beta_init)
        self.s_gamma = float(s_gamma_init)

        # Aux vars for Half‑Cauchy mixtures (a_k ~ InvGamma(1/2, 1/A_k^2), post a|Q ~ InvGamma(1, ...))
        self.a_alpha = 1.0
        self.a_beta  = 1.0
        self.a_gamma = 1.0

        # Initial m0 and P0 for dynamic coords (on variance scale)
        self.m0_alpha = float(m0_alpha_init)
        self.P0_alpha = float(P0_alpha_init)
        self.m0_beta  = float(m0_beta_init)
        self.P0_beta  = float(P0_beta_init)
        if m0_gamma_init is None:
            self.m0_gamma = np.zeros(self.period - 1, float)
        else:
            g = np.asarray(m0_gamma_init, float)
            if g.size != self.period - 1:
                raise ValueError("m0_gamma_init must have length p-1 (newest-first)")
            self.m0_gamma = g
        self.P0_gamma = float(P0_gamma_init)

        # Deterministic aliases (used when block is deterministic)
        # For season deterministic, store full length‑p vector summing to zero.
        self.season_det = np.r_[self.m0_gamma, -float(np.sum(self.m0_gamma))]

        # Layout & latent path
        self._refresh_layout()  # sets indices & dim
        self.x = np.zeros((self.T + 1, self._layout_dim), float)
        if self._layout_dim > 0:
            m0_vec, P0_diag = self._current_m0_P0()
            self.x[0] = np.random.multivariate_normal(m0_vec, np.diag(P0_diag) + 1e-10 * np.eye(self._layout_dim))
            self._propagate_initial_path(Q_init=np.full(self._layout_dim, 1e-6, float))

        # Stats for RJ
        self.rj_stats = {"level": {"proposed": 0, "accepted": 0},
                         "trend": {"proposed": 0, "accepted": 0},
                         "season": {"proposed": 0, "accepted": 0}}

        # Optional truth overlays
        self.true_sigma = None
        self.true_Q = None
        self.true_mu_t = None
        self.true_alpha_t = None
        self.true_beta_t = None
        self.true_gamma_t = None

        # Progress scale proxies
        if self.cfg.progress:
            y = self.y
            sd1 = _robust_sd(np.diff(y)) if y.size >= 2 else 0.0
            sd2 = _robust_sd(np.diff(y, n=2)) if y.size >= 3 else 0.0
            print(f"[init] scale proxies: sd1={sd1:.4g}, sd2={sd2:.4g}")

    # ----------------------------- Model matrices ----------------------------- #
    def _refresh_layout(self) -> None:
        layout: List[str] = []
        self.idx_alpha = self.idx_beta = None
        self.idx_g_start = self.idx_g_end = None
        if self.level_mode == "dynamic":
            self.idx_alpha = len(layout); layout.append("alpha")
        if self.trend_mode == "dynamic":
            self.idx_beta = len(layout); layout.append("beta")
        if self.seasonal_mode == "dynamic":
            self.idx_g_start = len(layout)
            layout.extend([f"g{k}" for k in range(1, self.period)])
            self.idx_g_end = len(layout) - 1
        self._layout = layout
        self._layout_dim = len(layout)

    def _H(self) -> np.ndarray:
        if self._layout_dim == 0:
            return np.zeros((1, 0))
        h = np.zeros(self._layout_dim, float)
        if self.idx_alpha is not None:
            h[self.idx_alpha] = 1.0
        if self.seasonal_mode == "dynamic":
            h[self.idx_g_start] = 1.0
        return h.reshape(1, -1)

    def _A(self) -> np.ndarray:
        if self._layout_dim == 0:
            return np.zeros((0, 0))
        A = np.eye(self._layout_dim)
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
        if self._layout_dim == 0:
            return np.zeros(0, float)
        u = np.zeros(self._layout_dim, float)
        if (self.idx_alpha is not None) and (self.trend_mode == "deterministic"):
            u[self.idx_alpha] = float(self.m0_beta)
        return u

    def _Q(self) -> np.ndarray:
        if self._layout_dim == 0:
            return np.zeros((0, 0))
        Q = np.zeros((self._layout_dim, self._layout_dim))
        if self.idx_alpha is not None and self.s_alpha > 0:
            Q[self.idx_alpha, self.idx_alpha] = self.s_alpha**2
        if self.idx_beta is not None and self.s_beta > 0:
            Q[self.idx_beta, self.idx_beta] = self.s_beta**2
        if self.seasonal_mode == "dynamic" and self.s_gamma > 0:
            Q[self.idx_g_start, self.idx_g_start] = self.s_gamma**2
        return Q

    def _mu_det(self, t: int) -> float:
        out = 0.0
        if self.level_mode == "deterministic":
            out += self.m0_alpha
        if (self.trend_mode == "deterministic") and (self.idx_alpha is None):
            out += self.m0_beta * t
        if self.seasonal_mode == "deterministic":
            out += float(self.season_det[t % self.period])
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

    # ------------------------- FFBS (Kalman + Carter–Kohn) ------------------------- #
    def _ffbs(self) -> np.ndarray:
        if self._layout_dim == 0:
            return self.x.copy()
        H, A, Q, R = self._H(), self._A(), self._Q(), float(self.sigma2)
        m0_vec, P0_diag = self._current_m0_P0()
        m = np.zeros((self.T + 1, self._layout_dim))
        C = np.zeros((self.T + 1, self._layout_dim, self._layout_dim))
        a = np.zeros((self.T + 1, self._layout_dim))
        Rm = np.zeros((self.T + 1, self._layout_dim, self._layout_dim))
        m[0] = m0_vec
        C[0] = np.diag(P0_diag) + 1e-12 * np.eye(self._layout_dim)
        u = self._u()

        # forward
        for t in range(1, self.T + 1):
            a[t]  = A @ m[t - 1] + u
            Rm[t] = A @ C[t - 1] @ A.T + Q
            Rm[t] = 0.5 * (Rm[t] + Rm[t].T) + 1e-12 * np.eye(self._layout_dim)

            resid_mean = float(self.y[t - 1] - self._mu_det(t - 1))
            S = float(H @ Rm[t] @ H.T + R)
            if S <= 0:
                S = float(H @ (Rm[t] + 1e-10 * np.eye(self._layout_dim)) @ H.T + R)
            K = (Rm[t] @ H.T) / S
            v = resid_mean - float(H @ a[t])
            m[t] = a[t] + (K.flatten() * v)
            C[t] = Rm[t] - K @ (H @ Rm[t])
            C[t] = 0.5 * (C[t] + C[t].T) + 1e-12 * np.eye(self._layout_dim)

        # backward
        x = np.zeros_like(self.x)
        x[self.T] = np.random.multivariate_normal(m[self.T], C[self.T]) if self._layout_dim>0 else np.zeros(self._layout_dim)
        for t in range(self.T - 1, -1, -1):
            J = C[t] @ A.T
            J = J @ _spd_solve(Rm[t + 1], np.eye(self._layout_dim))
            mean = m[t] + J @ (x[t + 1] - (A @ m[t] + u))
            cov  = C[t] - J @ Rm[t + 1] @ J.T
            cov  = 0.5 * (cov + cov.T)
            cov += max(0.0, 1e-12 - float(np.linalg.eigvalsh(cov).min())) * np.eye(cov.shape[0])
            x[t] = np.random.multivariate_normal(mean, cov)
        return x

    def _propagate_initial_path(self, Q_init: np.ndarray) -> None:
        if self._layout_dim == 0:
            return
        A, u = self._A(), self._u()
        for t in range(1, self.T + 1):
            self.x[t] = A @ self.x[t - 1] + u + np.random.normal(0.0, np.sqrt(Q_init), size=self._layout_dim)

    # ------------------ Helpers: μ and residuals ------------------ #
    def _mu_vec(self) -> np.ndarray:
        H = self._H()
        mu = np.zeros(self.T, float)
        for t in range(1, self.T + 1):
            dyn = float(H @ self.x[t]) if self._layout_dim > 0 else 0.0
            mu[t - 1] = self._mu_det(t - 1) + dyn
        return mu

    # ------------------ σ² | rest  (Gamma on precision) ------------------ #
    def update_sigma2(self) -> None:
        e = self.y - self._mu_vec()
        a = self.priors.a_sigma + 0.5 * self.T
        b = self.priors.b_sigma + 0.5 * float(e @ e)
        tau = np.random.gamma(shape=a, scale=1.0 / b)  # precision
        self.sigma2 = 1.0 / max(tau, 1e-300)

    # ------------- Innovation sums of squares (for Q updates) ------------- #
    def _innovation_ss_alpha(self) -> Tuple[float, int]:
        if self.idx_alpha is None:
            return 0.0, 0
        ss = 0.0
        for t in range(1, self.T + 1):
            drift = 0.0
            if self.idx_beta is not None:
                drift = self.x[t - 1, self.idx_beta]
            elif self.trend_mode == "deterministic":
                drift = float(self.m0_beta)
            mean = self.x[t - 1, self.idx_alpha] + drift
            ss += (self.x[t, self.idx_alpha] - mean) ** 2
        return float(ss), self.T

    def _innovation_ss_beta(self) -> Tuple[float, int]:
        if self.idx_beta is None:
            return 0.0, 0
        d = self.x[1:, self.idx_beta] - self.x[:-1, self.idx_beta]
        return float(np.sum(d * d)), self.T

    def _innovation_ss_gamma(self) -> Tuple[float, int]:
        if self.seasonal_mode != "dynamic":
            return 0.0, 0
        gs, ge = self.idx_g_start, self.idx_g_end
        ss = 0.0
        for t in range(1, self.T + 1):
            prev = self.x[t - 1, gs : ge + 1]
            mean_new_first = -float(np.sum(prev))
            ss += (self.x[t, gs] - mean_new_first) ** 2
        return float(ss), self.T

    # =============================================================================
    # Pure Gibbs for process variances with Half‑Cauchy priors
    # =============================================================================
    @staticmethod
    def _sample_invgamma(shape: float, scale: float) -> float:
        return 1.0 / np.random.gamma(shape, 1.0 / scale)

    def update_process_Q_halfcauchy(self) -> None:
        # α
        if self.idx_alpha is not None:
            SS, T_eff = self._innovation_ss_alpha()
            A = float(self.priors.hc_scale_alpha)
            shape_Q = 0.5 * T_eff + 0.5
            scale_Q = 0.5 * SS + 1.0 / max(self.a_alpha, 1e-300)
            Q_alpha = self._sample_invgamma(shape_Q, scale_Q)
            self.s_alpha = math.sqrt(max(Q_alpha, 0.0))
            # a | Q
            shape_a = 1.0
            scale_a = (1.0 / (A * A)) + (1.0 / max(Q_alpha, 1e-300))
            self.a_alpha = self._sample_invgamma(shape_a, scale_a)

        # β
        if self.idx_beta is not None:
            SS, T_eff = self._innovation_ss_beta()
            A = float(self.priors.hc_scale_beta)
            shape_Q = 0.5 * T_eff + 0.5
            scale_Q = 0.5 * SS + 1.0 / max(self.a_beta, 1e-300)
            Q_beta = self._sample_invgamma(shape_Q, scale_Q)
            self.s_beta = math.sqrt(max(Q_beta, 0.0))
            shape_a = 1.0
            scale_a = (1.0 / (A * A)) + (1.0 / max(Q_beta, 1e-300))
            self.a_beta = self._sample_invgamma(shape_a, scale_a)

        # γ
        if self.seasonal_mode == "dynamic":
            SS, T_eff = self._innovation_ss_gamma()
            A = float(self.priors.hc_scale_gamma)
            shape_Q = 0.5 * T_eff + 0.5
            scale_Q = 0.5 * SS + 1.0 / max(self.a_gamma, 1e-300)
            Q_gamma = self._sample_invgamma(shape_Q, scale_Q)
            self.s_gamma = math.sqrt(max(Q_gamma, 0.0))
            shape_a = 1.0
            scale_a = (1.0 / (A * A)) + (1.0 / max(Q_gamma, 1e-300))
            self.a_gamma = self._sample_invgamma(shape_a, scale_a)

    # --- m0 | P0, x0 (Normal) + P0 | m0, x0 (Inv‑Gamma) --- #
    @staticmethod
    def _gibbs_m0_scalar(x0: float, m_prior: float, s_prior: float, P0: float) -> float:
        prec = 1.0 / (s_prior**2) + 1.0 / max(1e-18, P0)
        var = 1.0 / prec
        mean = var * (m_prior / (s_prior**2) + x0 / max(1e-18, P0))
        return float(np.random.normal(mean, math.sqrt(var)))

    def update_m0(self) -> None:
        if self._layout_dim == 0:
            return
        pos = 0
        if self.idx_alpha is not None:
            self.m0_alpha = self._gibbs_m0_scalar(
                float(self.x[0, pos]), self.priors.m_m0_alpha, self.priors.s_m0_alpha, self.P0_alpha
            )
            pos += 1
        if self.idx_beta is not None:
            self.m0_beta = self._gibbs_m0_scalar(
                float(self.x[0, pos]), self.priors.m_m0_beta, self.priors.s_m0_beta, self.P0_beta
            )
            pos += 1
        if self.seasonal_mode == "dynamic":
            m_prior = (
                np.zeros(self.period - 1, float)
                if self.priors.m_m0_gamma is None
                else np.asarray(self.priors.m_m0_gamma, float)
            )
            if m_prior.size != self.period - 1:
                raise ValueError("priors.m_m0_gamma must be length p-1 (newest-first)")
            s = float(self.priors.s_m0_gamma)
            for k in range(self.period - 1):
                self.m0_gamma[k] = self._gibbs_m0_scalar(float(self.x[0, pos + k]), float(m_prior[k]), s, self.P0_gamma)

    def update_P0(self) -> None:
        if self._layout_dim == 0:
            return
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
            for k in range(self.period - 1):
                diffsq += (float(self.x[0, pos + k]) - float(self.m0_gamma[k])) ** 2
            a = self.priors.a_P0_gamma + 0.5 * (self.period - 1)
            b = self.priors.b_P0_gamma + 0.5 * diffsq
            self.P0_gamma = 1.0 / np.random.gamma(shape=a, scale=1.0 / b)

    # --- Deterministic params (all conjugate) --- #
    def update_deterministic_params(self) -> None:
        # level(det)
        if self.level_mode == "deterministic":
            r = self.y.copy()
            if self._layout_dim > 0:
                H = self._H()
                for t in range(1, self.T + 1):
                    r[t - 1] -= float(H @ self.x[t])
            if self.seasonal_mode == "deterministic":
                r -= self.season_det[np.arange(self.T) % self.period]
            s2 = float(self.sigma2)
            m0, s0 = float(self.priors.m_m0_alpha), float(self.priors.s_m0_alpha)
            prec = self.T / s2 + 1.0 / (s0**2)
            mean = ((r.sum() / s2) + m0 / (s0**2)) / prec
            var = 1.0 / prec
            self.m0_alpha = float(np.random.normal(mean, math.sqrt(var)))

        # trend(det)
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
                if self._layout_dim > 0:
                    H = self._H()
                    for k in range(1, self.T + 1):
                        r[k - 1] -= float(H @ self.x[k])
                if self.level_mode == "deterministic":
                    r -= self.m0_alpha
                if self.seasonal_mode == "deterministic":
                    r -= self.season_det[np.arange(self.T) % self.period]
                m0, s0 = float(self.priors.m_m0_beta), float(self.priors.s_m0_beta)
                sig2 = float(self.sigma2)
                prec = (t @ t) / sig2 + 1.0 / (s0**2)
                mean = ((t @ r) / sig2 + m0 / (s0**2)) / prec
                var = 1.0 / prec
                self.m0_beta = float(np.random.normal(mean, math.sqrt(var)))

        # season(det)
        if self.seasonal_mode == "deterministic":
            if not hasattr(self, "_Z_season"):
                midx = np.arange(self.T) % self.period
                K = self.period - 1
                Z = np.zeros((self.T, K))
                for k in range(K):
                    Z[:, k] = (midx == k).astype(float) - (midx == K).astype(float)
                self._Z_season = Z

            r = self.y.copy()
            if self._layout_dim > 0:
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
            self.season_det = np.r_[theta, -theta.sum()]

    # -------------------- Kalman loglik for RJ acceptance -------------------- #
    def _kalman_loglik(self, modes: Tuple[str, str, str], params: Dict[str, np.ndarray | float]) -> float:
        # Build dimension & matrices based on proposed modes and params
        level_mode, trend_mode, seasonal_mode = modes
        dim = (1 if level_mode == "dynamic" else 0) + (1 if trend_mode == "dynamic" else 0) + (self.period - 1 if seasonal_mode == "dynamic" else 0)
        if dim == 0:
            # pure deterministic mean
            mu = np.zeros(self.T)
            if level_mode == "deterministic":
                mu += params.get("m0_alpha_det", 0.0)
            if trend_mode == "deterministic" and level_mode != "dynamic":
                mu += params.get("m0_beta_det", 0.0) * np.arange(self.T)
            if seasonal_mode == "deterministic":
                season_det = params.get("season_det", np.zeros(self.period))
                mu += season_det[np.arange(self.T) % self.period]
            e = self.y - mu
            R = float(params["sigma2"])  # observation variance
            return -0.5 * (self.T * math.log(2 * math.pi * R) + float(e @ e) / R)

        # Build H, A, Q, u, m0, P0 under proposed modes/params
        # indices in this temporary layout
        idx = {}
        pos = 0
        if level_mode == "dynamic":
            idx["alpha"] = pos; pos += 1
        if trend_mode == "dynamic":
            idx["beta"] = pos; pos += 1
        if seasonal_mode == "dynamic":
            idx["g_start"] = pos
            pos += self.period - 1
            idx["g_end"] = pos - 1

        H = np.zeros((1, dim))
        if "alpha" in idx:
            H[0, idx["alpha"]] = 1.0
        if seasonal_mode == "dynamic":
            H[0, idx["g_start"]] = 1.0

        A = np.eye(dim)
        if "alpha" in idx and "beta" in idx:
            A[idx["alpha"], idx["beta"]] = 1.0
        if seasonal_mode == "dynamic":
            gs, ge = idx["g_start"], idx["g_end"]
            K = ge - gs + 1
            A[gs, gs:ge+1] = -1.0
            if K > 1:
                A[gs+1:ge+1, gs:ge] = np.eye(K-1)
                A[gs+1:ge+1, ge] = 0.0

        Q = np.zeros((dim, dim))
        if "alpha" in idx:
            Q[idx["alpha"], idx["alpha"]] = float(params.get("Q_alpha", 0.0))
        if "beta" in idx:
            Q[idx["beta"], idx["beta"]] = float(params.get("Q_beta", 0.0))
        if seasonal_mode == "dynamic":
            Q[idx["g_start"], idx["g_start"]] = float(params.get("Q_gamma", 0.0))

        u = np.zeros(dim)
        if ("alpha" in idx) and (trend_mode == "deterministic"):
            u[idx["alpha"]] = float(params.get("m0_beta_det", 0.0))

        # Initials
        m0 = []
        P0 = []
        if "alpha" in idx:
            m0.append(float(params.get("m0_alpha", 0.0)))
            P0.append(float(params.get("P0_alpha", 1.0)))
        if "beta" in idx:
            m0.append(float(params.get("m0_beta", 0.0)))
            P0.append(float(params.get("P0_beta", 1.0)))
        if seasonal_mode == "dynamic":
            m0.extend(list(np.asarray(params.get("m0_gamma", np.zeros(self.period - 1)), float)))
            P0.extend([float(params.get("P0_gamma", 1.0))] * (self.period - 1))
        m0 = np.asarray(m0, float)
        P0 = np.asarray(P0, float)

        # Deterministic μ contribution
        def mu_det_t(t: int) -> float:
            out = 0.0
            if level_mode == "deterministic":
                out += float(params.get("m0_alpha_det", 0.0))
            if (trend_mode == "deterministic") and ("alpha" not in idx):
                out += float(params.get("m0_beta_det", 0.0)) * t
            if seasonal_mode == "deterministic":
                season_det = np.asarray(params.get("season_det", np.zeros(self.period)))
                out += float(season_det[t % self.period])
            return out

        R = float(params["sigma2"])  # observation variance

        # Kalman filter loglik
        m = m0.copy()
        C = np.diag(P0) + 1e-12 * np.eye(dim)
        loglik = 0.0
        for t in range(self.T):
            a = A @ m + u
            Rm = A @ C @ A.T + Q
            Rm = 0.5 * (Rm + Rm.T) + 1e-12 * np.eye(dim)
            y_t = float(self.y[t] - mu_det_t(t))
            S = float(H @ Rm @ H.T + R)
            if S <= 0:
                S = float(H @ (Rm + 1e-10 * np.eye(dim)) @ H.T + R)
            v = y_t - float(H @ a)
            loglik += -0.5 * (math.log(2 * math.pi * S) + (v * v) / S)
            K = (Rm @ H.T) / S
            m = a + (K.flatten() * v)
            C = Rm - K @ (H @ Rm)
            C = 0.5 * (C + C.T) + 1e-12 * np.eye(dim)
        return float(loglik)

    # --------------------- Priors for RJ (log) --------------------- #
    def _log_prior_params(self, modes: Tuple[str, str, str], params: Dict[str, float | np.ndarray]) -> float:
        lp = 0.0
        # model prior
        enc = {"dynamic": 0, "deterministic": 1, "none": 2}
        lp += float(self.model_prior["level"][enc[modes[0]]])
        lp += float(self.model_prior["trend"][enc[modes[1]]])
        lp += float(self.model_prior["season"][enc[modes[2]]])

        # σ² prior: τ ~ Gamma(a,b) ⇒ p(σ²) ∝ (σ²)^{-a-1} exp(-b/σ²)
        sigma2 = float(params["sigma2"])  # use current σ² (not proposing new σ² here)
        a, b = self.priors.a_sigma, self.priors.b_sigma
        # p(σ²) in σ² scale (improper constant aside). Use Jeffreys‑consistent form:
        lp += -(a + 1.0) * math.log(sigma2) - b / sigma2

        # Dynamic blocks: m0 ~ N(m,s²); P0 ~ InvGamma(a,b); Q ~ scale‑mixture Half‑Cauchy.
        def log_inv_gamma(x, a, b):
            return a * math.log(b) - math.lgamma(a) - (a + 1) * math.log(x) - b / x
        def log_norm(x, m, s):
            return -0.5 * (math.log(2 * math.pi * s * s) + ((x - m) ** 2) / (s * s))
        # Half‑Cauchy on s via IG mixture ⇒ marginal p(Q)=? We'll use the IG conditional at fixed a_k.
        # For acceptance stability we include the conditional p(Q|a) and p(a) terms.
        def log_inv_gamma_cond(q, a_sh, a_sc):
            return log_inv_gamma(q, a_sh, a_sc)

        # level
        if modes[0] == "dynamic":
            lp += log_norm(float(params.get("m0_alpha", 0.0)), self.priors.m_m0_alpha, self.priors.s_m0_alpha)
            lp += log_inv_gamma(float(params.get("P0_alpha", 1.0)), self.priors.a_P0_alpha, self.priors.b_P0_alpha)
            # Q_alpha | a_alpha ~ InvGamma(1/2, 1/a_alpha); a_alpha ~ InvGamma(1/2, 1/A^2)
            Q = float(params.get("Q_alpha", 1.0)); A = self.priors.hc_scale_alpha
            a_aux = float(params.get("a_alpha", 1.0))
            lp += log_inv_gamma_cond(Q, 0.5, 1.0 / max(a_aux, 1e-300))
            lp += log_inv_gamma(a_aux, 0.5, 1.0 / (A * A))
        elif modes[0] == "deterministic":
            lp += log_norm(float(params.get("m0_alpha_det", 0.0)), self.priors.m_m0_alpha, self.priors.s_m0_alpha)

        # trend
        if modes[1] == "dynamic":
            lp += log_norm(float(params.get("m0_beta", 0.0)), self.priors.m_m0_beta, self.priors.s_m0_beta)
            lp += log_inv_gamma(float(params.get("P0_beta", 1.0)), self.priors.a_P0_beta, self.priors.b_P0_beta)
            Q = float(params.get("Q_beta", 1.0)); A = self.priors.hc_scale_beta
            a_aux = float(params.get("a_beta", 1.0))
            lp += log_inv_gamma_cond(Q, 0.5, 1.0 / max(a_aux, 1e-300))
            lp += log_inv_gamma(a_aux, 0.5, 1.0 / (A * A))
        elif modes[1] == "deterministic":
            lp += log_norm(float(params.get("m0_beta_det", 0.0)), self.priors.m_m0_beta, self.priors.s_m0_beta)

        # season
        if modes[2] == "dynamic":
            m_prior = (
                np.zeros(self.period - 1)
                if self.priors.m_m0_gamma is None
                else np.asarray(self.priors.m_m0_gamma, float)
            )
            for k in range(self.period - 1):
                lp += log_norm(
                    float(np.asarray(params.get("m0_gamma", np.zeros(self.period - 1)))[k]),
                    float(m_prior[k]), float(self.priors.s_m0_gamma)
                )
            lp += log_inv_gamma(float(params.get("P0_gamma", 1.0)), self.priors.a_P0_gamma, self.priors.b_P0_gamma)
            Q = float(params.get("Q_gamma", 1.0)); A = self.priors.hc_scale_gamma
            a_aux = float(params.get("a_gamma", 1.0))
            lp += log_inv_gamma_cond(Q, 0.5, 1.0 / max(a_aux, 1e-300))
            lp += log_inv_gamma(a_aux, 0.5, 1.0 / (A * A))
        elif modes[2] == "deterministic":
            # (p-1) free levels with sum‑to‑zero closure (we store full p with last implied)
            K = self.period - 1
            m_prior = (
                np.zeros(K)
                if self.priors.m_m0_gamma is None
                else np.asarray(self.priors.m_m0_gamma, float)
            )
            for k in range(K):
                lp += log_norm(float(np.asarray(params.get("season_det", np.zeros(self.period)))[k]), float(m_prior[k]), float(self.priors.s_m0_gamma))
        return lp

    # -------------------------- RJ proposal kernel -------------------------- #
    def _propose_modes_and_params(self, block: str) -> Tuple[Tuple[str, str, str], Dict[str, float | np.ndarray]]:
        # current modes
        modes_cur = (self.level_mode, self.trend_mode, self.seasonal_mode)
        modes_prop = list(modes_cur)

        # symmetric local proposals per block
        def toggle(s: str) -> str:
            if s == "dynamic": return "deterministic"
            if s == "deterministic": return "dynamic"  # prefer mixing between dyn<->det
            return "deterministic"  # from none, go to det

        if block == "level":
            modes_prop[0] = toggle(modes_prop[0])
            # if trend is dynamic, ensure level dynamic
            if modes_prop[1] == "dynamic" and modes_prop[0] != "dynamic":
                modes_prop[0] = "dynamic"
        elif block == "trend":
            modes_prop[1] = toggle(modes_prop[1])
            if modes_prop[1] == "dynamic" and modes_prop[0] != "dynamic":
                # escape move: flip level→dynamic as well
                modes_prop[0] = "dynamic"
        elif block == "season":
            modes_prop[2] = toggle(modes_prop[2])
        modes_prop = tuple(modes_prop)

        # Gather params for proposed model (draw new ones from priors when needed; otherwise reuse)
        params = {"sigma2": self.sigma2}
        # level
        if modes_prop[0] == "dynamic":
            params["m0_alpha"] = (self.m0_alpha if self.level_mode == "dynamic" else np.random.normal(self.priors.m_m0_alpha, self.priors.s_m0_alpha))
            params["P0_alpha"] = (self.P0_alpha if self.level_mode == "dynamic" else 1.0 / np.random.gamma(self.priors.a_P0_alpha, 1.0 / self.priors.b_P0_alpha))
            params["Q_alpha"]  = (self.s_alpha**2 if self.level_mode == "dynamic" else 1.0 / np.random.gamma(0.5, 2.0 * (self.a_alpha if hasattr(self, 'a_alpha') else 1.0)))
            params["a_alpha"]  = (self.a_alpha if self.level_mode == "dynamic" else 1.0 / np.random.gamma(0.5, 2.0 * (1.0 / (self.priors.hc_scale_alpha**2))))
        elif modes_prop[0] == "deterministic":
            params["m0_alpha_det"] = (self.m0_alpha if self.level_mode == "deterministic" else np.random.normal(self.priors.m_m0_alpha, self.priors.s_m0_alpha))
        # trend
        if modes_prop[1] == "dynamic":
            params["m0_beta"] = (self.m0_beta if self.trend_mode == "dynamic" else np.random.normal(self.priors.m_m0_beta, self.priors.s_m0_beta))
            params["P0_beta"] = (self.P0_beta if self.trend_mode == "dynamic" else 1.0 / np.random.gamma(self.priors.a_P0_beta, 1.0 / self.priors.b_P0_beta))
            params["Q_beta"]  = (self.s_beta**2 if self.trend_mode == "dynamic" else 1.0 / np.random.gamma(0.5, 2.0 * (self.a_beta if hasattr(self, 'a_beta') else 1.0)))
            params["a_beta"]  = (self.a_beta if self.trend_mode == "dynamic" else 1.0 / np.random.gamma(0.5, 2.0 * (1.0 / (self.priors.hc_scale_beta**2))))
        elif modes_prop[1] == "deterministic":
            params["m0_beta_det"] = (self.m0_beta if self.trend_mode == "deterministic" else np.random.normal(self.priors.m_m0_beta, self.priors.s_m0_beta))
        # season
        if modes_prop[2] == "dynamic":
            params["m0_gamma"] = (self.m0_gamma if self.seasonal_mode == "dynamic" else (np.zeros(self.period - 1) if self.priors.m_m0_gamma is None else np.asarray(self.priors.m_m0_gamma, float)) + np.random.normal(0, self.priors.s_m0_gamma, self.period - 1)*0.0)
            params["P0_gamma"] = (self.P0_gamma if self.seasonal_mode == "dynamic" else 1.0 / np.random.gamma(self.priors.a_P0_gamma, 1.0 / self.priors.b_P0_gamma))
            params["Q_gamma"]  = (self.s_gamma**2 if self.seasonal_mode == "dynamic" else 1.0 / np.random.gamma(0.5, 2.0 * (self.a_gamma if hasattr(self, 'a_gamma') else 1.0)))
            params["a_gamma"]  = (self.a_gamma if self.seasonal_mode == "dynamic" else 1.0 / np.random.gamma(0.5, 2.0 * (1.0 / (self.priors.hc_scale_gamma**2))))
        elif modes_prop[2] == "deterministic":
            if self.seasonal_mode == "deterministic":
                params["season_det"] = self.season_det.copy()
            else:
                K = self.period - 1
                base = np.zeros(K) if (self.priors.m_m0_gamma is None) else np.asarray(self.priors.m_m0_gamma, float)
                z = np.random.normal(base, self.priors.s_m0_gamma, size=K)
                params["season_det"] = np.r_[z, -z.sum()]

        return modes_prop, params

    def _accept_reject(self, modes_prop: Tuple[str, str, str], params_prop: Dict[str, float | np.ndarray]) -> bool:
        # current modes and params snapshot for log posterior
        modes_cur = (self.level_mode, self.trend_mode, self.seasonal_mode)
        params_cur: Dict[str, float | np.ndarray] = {"sigma2": self.sigma2}
        if self.level_mode == "dynamic":
            params_cur.update(dict(m0_alpha=self.m0_alpha, P0_alpha=self.P0_alpha, Q_alpha=self.s_alpha**2, a_alpha=self.a_alpha))
        elif self.level_mode == "deterministic":
            params_cur.update(dict(m0_alpha_det=self.m0_alpha))
        if self.trend_mode == "dynamic":
            params_cur.update(dict(m0_beta=self.m0_beta, P0_beta=self.P0_beta, Q_beta=self.s_beta**2, a_beta=self.a_beta))
        elif self.trend_mode == "deterministic":
            params_cur.update(dict(m0_beta_det=self.m0_beta))
        if self.seasonal_mode == "dynamic":
            params_cur.update(dict(m0_gamma=self.m0_gamma, P0_gamma=self.P0_gamma, Q_gamma=self.s_gamma**2, a_gamma=self.a_gamma))
        elif self.seasonal_mode == "deterministic":
            params_cur.update(dict(season_det=self.season_det))

        # compute Δ log posterior = (loglik_prop + logprior_prop) - (loglik_cur + logprior_cur)
        ll_cur = self._kalman_loglik(modes_cur, params_cur)
        lp_cur = self._log_prior_params(modes_cur, params_cur)
        ll_prop = self._kalman_loglik(modes_prop, params_prop)
        lp_prop = self._log_prior_params(modes_prop, params_prop)
        dlog = (ll_prop + lp_prop) - (ll_cur + lp_cur)
        if np.log(np.random.rand()) < dlog:
            # accept ⇒ overwrite modes & parameters, refresh layout and latent path dimension
            self.level_mode, self.trend_mode, self.seasonal_mode = modes_prop
            # update parameters
            if self.level_mode == "dynamic":
                self.m0_alpha = float(params_prop.get("m0_alpha", self.m0_alpha))
                self.P0_alpha = float(params_prop.get("P0_alpha", self.P0_alpha))
                self.s_alpha  = math.sqrt(max(float(params_prop.get("Q_alpha", self.s_alpha**2)), 0.0))
                self.a_alpha  = float(params_prop.get("a_alpha", self.a_alpha))
            elif self.level_mode == "deterministic":
                self.m0_alpha = float(params_prop.get("m0_alpha_det", self.m0_alpha))
            if self.trend_mode == "dynamic":
                self.m0_beta = float(params_prop.get("m0_beta", self.m0_beta))
                self.P0_beta = float(params_prop.get("P0_beta", self.P0_beta))
                self.s_beta  = math.sqrt(max(float(params_prop.get("Q_beta", self.s_beta**2)), 0.0))
                self.a_beta  = float(params_prop.get("a_beta", self.a_beta))
            elif self.trend_mode == "deterministic":
                self.m0_beta = float(params_prop.get("m0_beta_det", self.m0_beta))
            if self.seasonal_mode == "dynamic":
                self.m0_gamma = np.asarray(params_prop.get("m0_gamma", self.m0_gamma), float)
                self.P0_gamma = float(params_prop.get("P0_gamma", self.P0_gamma))
                self.s_gamma  = math.sqrt(max(float(params_prop.get("Q_gamma", self.s_gamma**2)), 0.0))
                self.a_gamma  = float(params_prop.get("a_gamma", self.a_gamma))
            elif self.seasonal_mode == "deterministic":
                self.season_det = np.asarray(params_prop.get("season_det", self.season_det), float)

            # refresh layout and resize latent path container
            old_dim = self._layout_dim
            self._refresh_layout()
            if self._layout_dim != old_dim:
                self.x = np.zeros((self.T + 1, self._layout_dim), float)
            return True
        return False

    def _rj_move_one(self) -> None:
        block = np.random.choice(["level", "trend", "season"])  # simple random scan
        self.rj_stats[block]["proposed"] += 1
        modes_prop, params_prop = self._propose_modes_and_params(block)
        acc = self._accept_reject(modes_prop, params_prop)
        if acc:
            self.rj_stats[block]["accepted"] += 1

    # ------------------- Progress formatting ------------------- #
    @staticmethod
    def _fmt_list(vals, max_elems: int = 6, fmt: str = ".4g") -> str:
        if vals is None:
            return "-"
        v = np.asarray(vals, float).ravel()
        if v.size == 0:
            return "[]"
        if v.size <= max_elems:
            return "[" + ", ".join(f"{x:{fmt}}" for x in v) + "]"
        head = ", ".join(f"{x:{fmt}}" for x in v[:max_elems])
        return f"[{head}, …]"

    def _fmt_rj_all(self) -> str:
        def f(k: str) -> str:
            s = self.rj_stats[k]
            return f"{s['accepted']}/{s['proposed']}"
        return f"L {f('level')} | T {f('trend')} | S {f('season')}"

    def _max_state_dim(self) -> int:
        return 1 + 1 + (self.period - 1)  # level + trend + seasonal max

    def _tally_modes(self) -> None:
        # no-op here; kept for symmetry with earlier API
        return

    # ------------------------------- progress ------------------------------- #
    def _progress_line(self, it: int) -> str:
        parts = [
            f"[it {it + 1}/{self.cfg.n_iter}]",
            f"σ={math.sqrt(self.sigma2):.3f}",
            f"L/T/S={self.level_mode[:3]}/{self.trend_mode[:3]}/{self.seasonal_mode[:3]}",
        ]
        if self.idx_alpha is not None:
            parts.append(f"Qα={self.s_alpha**2:.4g}")
        if self.idx_beta is not None:
            parts.append(f"Qβ={self.s_beta**2:.4g}")
        if self.seasonal_mode == "dynamic":
            parts.append(f"Qγ={self.s_gamma**2:.4g}")
        if self.level_mode == "dynamic":
            parts.append(f"m0α={self.m0_alpha:.4g} P0α={self.P0_alpha:.4g}")
        elif self.level_mode == "deterministic":
            parts.append(f"m0α(det)={self.m0_alpha:.4g}")
        if self.trend_mode == "dynamic":
            parts.append(f"m0β={self.m0_beta:.4g} P0β={self.P0_beta:.4g}")
        elif self.trend_mode == "deterministic":
            parts.append(f"m0β(det)={self.m0_beta:.4g}")
        if self.seasonal_mode == "dynamic":
            parts.append(f"m0γ={self._fmt_list(self.m0_gamma)} P0γ={self.P0_gamma:.4g}")
        elif self.seasonal_mode == "deterministic":
            parts.append(f"m0γ(det)={self._fmt_list(self.season_det[:-1])}")
        parts.append(f"RJ {self._fmt_rj_all()}")
        return " | ".join(parts)

    # ---------------------------------- run --------------------------------- #
    def run(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        save_iters = list(range(cfg.burn, cfg.n_iter, cfg.thin))
        n_kept, keep_idx = len(save_iters), 0

        # pre-allocate with max possible dim (safe for RJ growth)
        max_dim = self._max_state_dim()

        # Always allocate full set; fill with NaN where N/A. Plotter-friendly.
        self.keep = {
            "sigma": np.full(n_kept, np.nan, float),
            "mu":    np.full((n_kept, self.T), np.nan, float),
            "modes": np.zeros((n_kept, 3), int),  # 0:dyn,1:det,2:none
            "x":     np.zeros((n_kept, self.T, max_dim), float),
            # Process variances (Q = s^2)
            "Q_alpha": np.full(n_kept, np.nan, float),
            "Q_beta":  np.full(n_kept, np.nan, float),
            "Q_gamma": np.full(n_kept, np.nan, float),
            # m0 for dynamic blocks (NaN if not dynamic)
            "m0_alpha": np.full(n_kept, np.nan, float),
            "m0_beta":  np.full(n_kept, np.nan, float),
            "m0_gamma": np.full((n_kept, self.period - 1), np.nan, float),
            # P0 for dynamic blocks (NaN if not dynamic)
            "P0_alpha": np.full(n_kept, np.nan, float),
            "P0_beta":  np.full(n_kept, np.nan, float),
            "P0_gamma": np.full(n_kept, np.nan, float),
            # Deterministic aliases (filled only when deterministic)
            "m0_alpha_det": np.full(n_kept, np.nan, float),
            "m0_beta_det":  np.full(n_kept, np.nan, float),
            "season_det":   np.full((n_kept, self.period - 1), np.nan, float),
        }

        print_every = cfg.progress_every if cfg.progress_every > 0 else max(1, cfg.n_iter // 50) or 1

        for it in range(cfg.n_iter):
            # RJ block moves (before FFBS so x aligns with accepted model)
            for _ in range(cfg.rj_moves_per_iter):
                self._rj_move_one()
            self._refresh_layout()

            # 1) FFBS for current model
            if self._layout_dim > 0:
                self.x = self._ffbs()
                # 2) process Q via Half‑Cauchy IG mixtures (pure Gibbs)
                self.update_process_Q_halfcauchy()
                # 3) m0 and 4) P0
                self.update_m0()
                self.update_P0()

            # 5) deterministic params
            self.update_deterministic_params()
            # 6) σ²
            self.update_sigma2()

            if cfg.progress and ((it + 1) % print_every == 0 or it == cfg.n_iter - 1):
                print(self._progress_line(it))

            # save
            if it in save_iters:
                k = keep_idx
                mu = self._mu_vec()
                self.keep["mu"][k, :] = mu
                self.keep["sigma"][k] = math.sqrt(self.sigma2)

                enc = lambda s: 0 if s == "dynamic" else (1 if s == "deterministic" else 2)
                self.keep["modes"][k, :] = np.array(
                    [enc(self.level_mode), enc(self.trend_mode), enc(self.seasonal_mode)], int
                )

                # Q
                if self.level_mode == "dynamic":
                    self.keep["Q_alpha"][k] = self.s_alpha**2
                if self.trend_mode == "dynamic":
                    self.keep["Q_beta"][k] = self.s_beta**2
                if self.seasonal_mode == "dynamic":
                    self.keep["Q_gamma"][k] = self.s_gamma**2

                # m0/P0/det aliases
                if self.level_mode == "dynamic":
                    self.keep["m0_alpha"][k] = self.m0_alpha
                    self.keep["P0_alpha"][k] = self.P0_alpha
                elif self.level_mode == "deterministic":
                    self.keep["m0_alpha_det"][k] = self.m0_alpha

                if self.trend_mode == "dynamic":
                    self.keep["m0_beta"][k] = self.m0_beta
                    self.keep["P0_beta"][k] = self.P0_beta
                elif self.trend_mode == "deterministic":
                    self.keep["m0_beta_det"][k] = self.m0_beta

                if self.seasonal_mode == "dynamic":
                    self.keep["m0_gamma"][k, :] = np.asarray(self.m0_gamma, float)
                    self.keep["P0_gamma"][k]    = self.P0_gamma
                elif self.seasonal_mode == "deterministic":
                    self.keep["season_det"][k, :] = np.asarray(self.season_det[:-1], float)

                # latent states (cap width)
                if self._layout_dim > 0:
                    w = min(self.x.shape[1], self.keep["x"].shape[2])
                    self.keep["x"][k, :, :w] = self.x[1 : self.T + 1, :w]

                keep_idx += 1

        return self.keep

    # ----------------------------------- I/O -------------------------------- #
    def save_posterior(self, out_npz_path: str, extra_meta: Optional[dict] = None) -> None:
        os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)
        arrays = dict(self.keep)
        arrays["y"] = self.y.copy()
        if "x" not in arrays:
            arrays["x"] = np.zeros((0, 0, 0))
        # Truth overlays (optional)
        if self.true_sigma is not None:
            arrays["true_sigma"] = float(self.true_sigma)
        if self.true_Q is not None:
            arrays["true_Q"] = np.asarray(self.true_Q, float)
        if self.true_mu_t is not None:
            arrays["true_mu_t"] = np.asarray(self.true_mu_t, float)
        if self.true_alpha_t is not None:
            arrays["true_alpha_t"] = np.asarray(self.true_alpha_t, float)
        if self.true_beta_t is not None:
            arrays["true_beta_t"] = np.asarray(self.true_beta_t, float)
        if self.true_gamma_t is not None:
            arrays["true_gamma_t"] = np.asarray(self.true_gamma_t, float)
        np.savez_compressed(out_npz_path, **arrays)

        meta = {
            "T": int(self.T),
            "period": int(self.period),
            "cfg": asdict(self.cfg),
            "priors": asdict(self.priors),
            "model_prior": {k: list(np.exp(v)) for k, v in self.model_prior.items()},
            "current_modes": {
                "level": self.level_mode,
                "trend": self.trend_mode,
                "season": self.seasonal_mode,
            },
            "rj_accept": {
                b: {
                    "proposed": int(self.rj_stats[b]["proposed"]),
                    "accepted": int(self.rj_stats[b]["accepted"]),
                    "acc_rate": (
                        self.rj_stats[b]["accepted"] / max(1, self.rj_stats[b]["proposed"])
                    ),
                }
                for b in ("level", "trend", "season")
            },
            "modes_encoding": {"dynamic": 0, "deterministic": 1, "none": 2},
            "layout_labels": (
                (["alpha"] if self.level_mode == "dynamic" else [])
                + (["beta"] if self.trend_mode == "dynamic" else [])
                + ([f"g{k}" for k in range(1, self.period)] if self.seasonal_mode == "dynamic" else [])
            ),
        }
        if extra_meta:
            meta.update(extra_meta)
        meta_path = out_npz_path.replace(".npz", ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[save] Posterior -> {out_npz_path}")
        print(f"[save] Metadata  -> {meta_path}")

    # --------------------------- optional truth --------------------------- #
    def set_truth(self, sigma: Optional[float] = None, Q: Optional[Tuple[float, float, float]] = None,
                  m0_level: Optional[float] = None, m0_trend: Optional[float] = None,
                  m0_season: Optional[Sequence[float]] = None, P0_level: Optional[float] = None,
                  P0_trend: Optional[float] = None, P0_season: Optional[Sequence[float]] = None) -> None:
        self.true_sigma = None if sigma is None else float(sigma)
        self.true_Q = None if Q is None else np.asarray(Q, float)
        self.true_m0_level = None if m0_level is None else float(m0_level)
        self.true_m0_trend = None if m0_trend is None else float(m0_trend)
        self.true_m0_season = None if m0_season is None else np.asarray(m0_season, float)
        self.true_P0_level = None if P0_level is None else float(P0_level)
        self.true_P0_trend = None if P0_trend is None else float(P0_trend)
        self.true_P0_season = None if P0_season is None else np.asarray(P0_season, float)

    def set_truth_paths(self, mu: Optional[np.ndarray] = None, alpha: Optional[np.ndarray] = None,
                         beta: Optional[np.ndarray] = None, gamma: Optional[np.ndarray] = None) -> None:
        self.true_mu_t = None if mu is None else np.asarray(mu, float)
        self.true_alpha_t = None if alpha is None else np.asarray(alpha, float)
        self.true_beta_t = None if beta is None else np.asarray(beta, float)
        self.true_gamma_t = None if gamma is None else np.asarray(gamma, float)


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
            "Gaussian structural TS (level/trend/season) with RJ over block modes, "
            "FFBS + conjugate Gibbs, newest-first seasonal ordering, and Half-Cauchy "
            "(IG-mixture) priors on process SDs."
        )
    )

    # Simulation (data-generating truth)
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--period", type=int, default=4)
    p.add_argument("--start-date", type=str, default="2000-01-01")
    p.add_argument("--level-mode", choices=["dynamic", "deterministic"], default="dynamic")
    p.add_argument("--trend-mode", choices=["dynamic", "deterministic", "none"], default="deterministic")
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

    # Half-Cauchy scales for process SDs
    p.add_argument("--hc-scale-alpha", type=float, default=0.5)
    p.add_argument("--hc-scale-beta",  type=float, default=0.5)
    p.add_argument("--hc-scale-gamma", type=float, default=0.5)

    # Model prior over modes (dyn, det, none)
    p.add_argument("--mp-level",  type=str, default="0.9,0.1,0.0")
    p.add_argument("--mp-trend",  type=str, default="0.9,0.1,0.0")
    p.add_argument("--mp-season", type=str, default="0.9,0.1,0.0")

    # Sampler configuration
    p.add_argument("--n-iter", type=int, default=20000)
    p.add_argument("--burn", type=int, default=5000)
    p.add_argument("--thin", type=int, default=2)
    p.add_argument("--seed", type=int, default=40)
    p.add_argument("--progress", default=True)
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--rj-moves-per-iter", type=int, default=3)
    p.add_argument("--out-dir", type=str, default="results/simulations/DLM_RJ")
    p.add_argument("--plot", action="store_true", default=True)
    p.add_argument("--print-summary", action="store_true", default=True)

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

    # Simulate data
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
    priors = Priors(
        a_sigma=args.prior_a_sigma, b_sigma=args.prior_b_sigma,
        m_m0_alpha=args.prior_m_m0_alpha, s_m0_alpha=args.prior_s_m0_alpha,
        m_m0_beta=args.prior_m_m0_beta, s_m0_beta=args.prior_s_m0_beta,
        m_m0_gamma=None if pri_gamma_vec is None else pri_gamma_vec,
        s_m0_gamma=args.prior_s_m0_gamma,
        a_P0_alpha=args.prior_a_P0_alpha, b_P0_alpha=args.prior_b_P0_alpha,
        a_P0_beta=args.prior_a_P0_beta, b_P0_beta=args.prior_b_P0_beta,
        a_P0_gamma=args.prior_a_P0_gamma, b_P0_gamma=args.prior_b_P0_gamma,
        hc_scale_alpha=args.hc_scale_alpha,
        hc_scale_beta=args.hc_scale_beta,
        hc_scale_gamma=args.hc_scale_gamma,
    )

    # Model prior
    def _triple(s: str) -> tuple[float, float, float]:
        v = [float(z) for z in s.split(",")]
        if len(v) != 3:
            raise ValueError("Model prior triple must have 3 comma-separated values")
        return (v[0], v[1], v[2])

    model_prior = ModelPrior(
        level=_triple(args.mp_level),
        trend=_triple(args.mp_trend),
        season=_triple(args.mp_season),
    )

    cfg = SamplerConfig(
        n_iter=int(args.n_iter), burn=int(args.burn), thin=int(args.thin),
        random_seed=int(args.seed), progress=bool(args.progress),
        progress_every=int(args.progress_every), rj_moves_per_iter=int(args.rj_moves_per_iter)
    )

    # Sampler init
    m0_gamma_init = (
        [float(z) for z in args.m0_gamma_init.split(",")] if args.m0_gamma_init else None
    )

    sampler = DLMRJGibbs(
        y=y, period=args.period,
        level_mode='dynamic', trend_mode='dynamic', seasonal_mode='dynamic',
        sigma2_init=args.sigma_init ** 2,
        s_alpha_init=args.s_alpha_init, s_beta_init=args.s_beta_init, s_gamma_init=args.s_gamma_init,
        m0_alpha_init=args.m0_level, m0_beta_init=args.m0_trend,
        P0_alpha_init=args.P0_alpha_init, P0_beta_init=args.P0_beta_init,
        m0_gamma_init=m0_gamma_init, P0_gamma_init=args.P0_gamma_init,
        priors=priors, model_prior=model_prior, cfg=cfg,
    )

    sampler.set_truth(
        sigma=mts.sigma, Q=(mts.q_level, mts.q_trend, mts.q_season),
        m0_level=mts.m0_level, m0_trend=mts.m0_trend, m0_season=mts.m0_season,
        P0_level=mts.v0_level, P0_trend=mts.v0_trend, P0_season=mts.v0_season,
    )
    sampler.set_truth_paths(mu=mu_T)

    if args.print_summary:
        print(f"Simulated {args.T} observations (σ={mts.sigma}) with START modes "
              f"{args.level_mode}/{args.trend_mode}/{args.seasonal_mode}.")
        print("Half-Cauchy scales (A_k) for process SDs:")
        print(f"  A_alpha={priors.hc_scale_alpha:.3f}, A_beta={priors.hc_scale_beta:.3f}, "
              f"A_gamma={priors.hc_scale_gamma:.3f}")  
        print("Model prior (dyn, det, none):")
        print(f"  level = {args.mp_level}  trend = {args.mp_trend}  season = {args.mp_season}")

    # Run
    t0 = time.time()
    post = sampler.run()
    elapsed = time.time() - t0
    print(f"[Run completed in {elapsed:.1f}s]")

    out_dir = os.path.join(
        args.out_dir,
        f"{args.level_mode}-{args.trend_mode}-{args.seasonal_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(out_dir, exist_ok=True)
    sampler.save_posterior(
        out_npz_path=os.path.join(out_dir, "posterior.npz"),
        extra_meta={
            "elapsed_seconds": float(elapsed),
            "hc_scales": {
                "alpha": priors.hc_scale_alpha,
                "beta": priors.hc_scale_beta,
                "gamma": priors.hc_scale_gamma,
            },
        },
    )

    # Summary + plot
    if args.print_summary:
        print("--- Posterior means (kept draws) ---")
        print(f"σ = {np.nanmean(post['sigma']):.4f}")
        for k in ["alpha", "beta", "gamma"]:
            key = f"Q_{k}"
            if key in post and np.isfinite(post[key]).any():
                mQ = np.nanmean(post[key])
                if np.isfinite(mQ):
                    print(f"Q_{k} = {mQ:.4g} (√Q ≈ {math.sqrt(mQ):.4g})")
        # RJ acceptance
        print("RJ acceptance (accepted/proposed):")
        for b in ("level", "trend", "season"):
            s = sampler.rj_stats[b]
            rate = s["accepted"] / max(1, s["proposed"])
            print(f"  {b:6s}: {s['accepted']}/{s['proposed']} (rate {rate:.3f})")

    if args.plot:
        mu_hat = np.nanmean(post["mu"], axis=0)
        plt.figure(figsize=(10, 4))
        plt.plot(dates_T, y, label="y_t", lw=1)
        plt.plot(dates_T, mu_T, "--", label="μ_t (truth)")
        plt.plot(dates_T, mu_hat, "-.", label="μ̂_t (post mean)")
        plt.title(f"DLM‑RJ ({args.level_mode}/{args.trend_mode}/{args.seasonal_mode})")
        plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()
