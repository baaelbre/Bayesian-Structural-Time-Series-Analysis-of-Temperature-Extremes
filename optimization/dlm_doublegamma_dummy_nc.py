from __future__ import annotations

import json, math, os, warnings, time, sys
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
    """Robust solver for SPD systems via (possibly jittered) Cholesky."""
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
# Priors & Config (double-Gamma priors on process precisions λ_k)
# =============================================================================

@dataclass
class Priors:
    # Observation variance σ²: precision τ = 1/σ² ~ Gamma(a_sigma, b_sigma)
    a_sigma: float = 2.0
    b_sigma: float = 1.0

    # m0 priors (used both for dynamic x0 means and deterministic params)
    m_m0_alpha: float = 0.0
    s_m0_alpha: float = 10.0
    m_m0_beta:  float = 0.0
    s_m0_beta:  float = 10.0
    m_m0_gamma: Optional[Sequence[float]] = None  # len p-1 (newest-first)
    s_m0_gamma: float = 5.0

    # Initial-state variances P0 ~ Inv-Gamma(a, b) (shape–rate)
    a_P0_alpha: float = 2.0
    b_P0_alpha: float = 1.0
    a_P0_beta:  float = 2.0
    b_P0_beta:  float = 1.0
    a_P0_gamma: float = 2.0
    b_P0_gamma: float = 1.0  # shared across p-1 seasonal coords

    # Double-Gamma priors for process precisions λ_k = 1/s_k^2:
    #   λ_k | τ_k ~ Gamma(a_lambda_k, τ_k)
    #   τ_k       ~ Gamma(c_lambda_k, d_lambda_k)
    a_lambda_alpha: float = 1.0
    c_lambda_alpha: float = 1.0
    d_lambda_alpha: float = 1.0

    a_lambda_beta: float = 1.0
    c_lambda_beta: float = 1.0
    d_lambda_beta: float = 1.0

    a_lambda_gamma: float = 1.0
    c_lambda_gamma: float = 1.0
    d_lambda_gamma: float = 1.0


@dataclass
class SamplerConfig:
    n_iter: int = 5000
    burn: int = 1000
    thin: int = 2
    random_seed: Optional[int] = 40
    progress: bool = True
    progress_every: int = 0  # 0 => ~2% of n_iter


# =============================================================================
# DLM Sampler with dummy-seasonals + double-Gamma process precisions
# =============================================================================

class DLMGibbsDummyDoubleGamma:
    """
    Gaussian structural DLM with dummy-seasonal representation.

    State (newest-first season):
        [alpha] [beta] [g1 ... g_{p-1}]

    Observation:
        y_t = μ_det(t) + H x_t + ε_t,   ε_t ~ N(0, σ²)

    Centred state evolution:
        x_t = A x_{t-1} + u + w_t,   w_t ~ N(0, Q)
        Q = diag(1/λ_alpha, 1/λ_beta, 1/λ_gamma, 0, ..., 0)

    Process precisions λ_k = 1/s_k^2 have double-Gamma priors:
        λ_k | τ_k ~ Gamma(a_lambda_k, τ_k)
        τ_k       ~ Gamma(c_lambda_k, d_lambda_k)

    Full conditionals (shape–rate):
        λ_k | τ_k, w_k ~ Gamma(a_lambda_k + n_k/2,
                               τ_k + 0.5 * sum w_k^2)
        τ_k | λ_k     ~ Gamma(c_lambda_k + a_lambda_k,
                               d_lambda_k + λ_k)

    All updates for Q are thus fully conjugate; no slice sampling.
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
        m0_gamma_init: Optional[Sequence[float]] = None,  # len p-1 (NEWEST-FIRST if dynamic)
        P0_gamma_init: float = 1.0,
        sigma2_init: float = 1.0,
        # we keep these as *SD* in the interface but convert to λ_k internally
        s_alpha_init: float = 1e-2,
        s_beta_init: float = 1e-3,
        s_gamma_init: float = 1e-3,
        priors: Priors = Priors(),
        cfg: SamplerConfig = SamplerConfig(),
    ):
        # ---------------- data / spec ----------------
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
        if cfg.random_seed is not None:
            np.random.seed(cfg.random_seed)
            self._rng = np.random.default_rng(cfg.random_seed)
        else:
            self._rng = np.random.default_rng()

        # ---------------- layout (dynamic) ----------------
        layout: List[str] = []
        if self.level_mode == "dynamic":
            layout.append("alpha")
        if self.trend_mode == "dynamic":
            layout.append("beta")
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
            self.idx_g_start = None
            self.idx_g_end   = None

        # ---------------- parameters / inits ----------------
        self.sigma2 = float(sigma2_init)

        # process precisions λ_k and associated τ_k (double-Gamma hierarchy)
        # we initialise λ_k from SD guesses and τ_k near prior mean
        tiny = 1e-12
        if self.idx_alpha is not None:
            self.lambda_alpha = 1.0 / max(s_alpha_init**2, tiny)
            self.tau_alpha = self.priors.d_lambda_alpha / max(self.priors.c_lambda_alpha, tiny)
        else:
            self.lambda_alpha = 0.0
            self.tau_alpha = 0.0

        if self.idx_beta is not None:
            self.lambda_beta = 1.0 / max(s_beta_init**2, tiny)
            self.tau_beta = self.priors.d_lambda_beta / max(self.priors.c_lambda_beta, tiny)
        else:
            self.lambda_beta = 0.0
            self.tau_beta = 0.0

        if self.seasonal_mode == "dynamic":
            self.lambda_gamma = 1.0 / max(s_gamma_init**2, tiny)
            self.tau_gamma = self.priors.d_lambda_gamma / max(self.priors.c_lambda_gamma, tiny)
        else:
            self.lambda_gamma = 0.0
            self.tau_gamma = 0.0

        # m0/P0 for dynamic coordinates
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
            # full period-length dummy means with sum-to-zero
            self.m0_gamma = np.r_[base, -float(np.sum(base))].astype(float)

        # ---------------- latent path ----------------
        self.x   = np.zeros((self.T + 1, self.dim), float)
        self.eta = np.zeros((self.T, self.dim), float) if self.dim > 0 else np.zeros((0, 0))
        self.x0  = np.zeros(self.dim, float) if self.dim > 0 else np.zeros(0, float)

        if self.dim > 0:
            # draw initial x0 from m0/P0, then build x from (x0, eta, s)
            m0_vec, P0_diag = self._current_m0_P0()
            self.x0 = np.random.multivariate_normal(
                m0_vec, np.diag(P0_diag) + 1e-10 * np.eye(self.dim)
            )
            self.eta = self._rng.normal(size=(self.T, self.dim))
            S_diag_init = self._S_diag()
            self.x = self._x_from_eta(S_diag=S_diag_init)

        # CP stats cache (for λ updates)
        self._cp_stats: Dict[str, Tuple[float, int]] = {}

        # Storage
        self.keep: Dict[str, np.ndarray] = {}

        # Truth overlays
        self.true_sigma: Optional[float] = None
        self.true_Q: Optional[np.ndarray] = None
        self.true_mu_t: Optional[np.ndarray] = None
        self.true_alpha_t: Optional[np.ndarray] = None
        self.true_beta_t: Optional[np.ndarray] = None
        self.true_gamma_t: Optional[np.ndarray] = None

        # Print scale proxies
        if self.cfg.progress:
            y = self.y
            sd1 = _robust_sd(np.diff(y)) if y.size >= 2 else 0.0
            sd2 = _robust_sd(np.diff(y, n=2)) if y.size >= 3 else 0.0
            fmt = lambda x: "n/a" if x is None else f"{x:.4g}"
            print(f"[init] scale proxies: sd1={fmt(sd1)}, sd2={fmt(sd2)}")

    # --------------------- Truth overlays (optional) --------------------- #
    def set_truth(
        self,
        sigma: Optional[float] = None,
        Q: Optional[Tuple[float, float, float]] = None,
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

    # ----------------------------- Model matrices ----------------------------- #
    def _H(self) -> np.ndarray:
        if self.dim == 0:
            return np.zeros((1, 0))
        h = np.zeros(self.dim, float)
        if self.idx_alpha is not None:
            h[self.idx_alpha] = 1.0
        if self.seasonal_mode == "dynamic":
            h[self.idx_g_start] = 1.0
        return h.reshape(1, -1)

    def _A(self) -> np.ndarray:
        """Transition matrix for local level/trend + seasonal dummies (newest-first)."""
        if self.dim == 0:
            return np.zeros((0, 0))
        A = np.eye(self.dim)
        # local linear trend
        if self.idx_alpha is not None and self.idx_beta is not None:
            A[self.idx_alpha, self.idx_beta] = 1.0
        # seasonal dummy block
        if self.seasonal_mode == "dynamic":
            gs, ge = self.idx_g_start, self.idx_g_end
            K = ge - gs + 1  # = period - 1
            A[gs, gs:ge + 1] = -1.0
            A[gs + 1:ge + 1, gs:ge] = np.eye(K - 1)
            A[gs + 1:ge + 1, ge] = 0.0
        return A

    def _u(self) -> np.ndarray:
        """Drift term u in x_t = A x_{t-1} + u + w_t."""
        if self.dim == 0:
            return np.zeros(0, float)
        u = np.zeros(self.dim, float)
        if (self.idx_alpha is not None) and (self.trend_mode == "deterministic"):
            u[self.idx_alpha] = float(self.m0_beta)
        return u

    # helpers to get Q_k from λ_k
    def _Q_alpha_scalar(self) -> float:
        return 1.0 / max(self.lambda_alpha, 1e-300) if self.idx_alpha is not None else 0.0

    def _Q_beta_scalar(self) -> float:
        return 1.0 / max(self.lambda_beta, 1e-300) if self.idx_beta is not None else 0.0

    def _Q_gamma_scalar(self) -> float:
        if self.seasonal_mode != "dynamic":
            return 0.0
        return 1.0 / max(self.lambda_gamma, 1e-300)

    def _Q(self) -> np.ndarray:
        """Process covariance for the centred representation."""
        if self.dim == 0:
            return np.zeros((0, 0))
        Q = np.zeros((self.dim, self.dim))
        if self.idx_alpha is not None:
            Q[self.idx_alpha, self.idx_alpha] = self._Q_alpha_scalar()
        if self.idx_beta is not None:
            Q[self.idx_beta, self.idx_beta] = self._Q_beta_scalar()
        if self.seasonal_mode == "dynamic":
            Q[self.idx_g_start, self.idx_g_start] = self._Q_gamma_scalar()
        return Q

    # ---------------- disturbance NCP helpers ---------------- #
    def _S_diag(self) -> np.ndarray:
        """
        Diagonal of S such that w_t = S eta_t and Q = diag(S^2).
        Since Q_kk = 1/λ_k, we have s_k = sqrt(Q_kk) = 1/sqrt(λ_k).
        """
        d = np.zeros(self.dim, float)
        if self.idx_alpha is not None and self.lambda_alpha > 0:
            d[self.idx_alpha] = math.sqrt(self._Q_alpha_scalar())
        if self.idx_beta is not None and self.lambda_beta > 0:
            d[self.idx_beta] = math.sqrt(self._Q_beta_scalar())
        if self.seasonal_mode == "dynamic" and self.lambda_gamma > 0:
            d[self.idx_g_start] = math.sqrt(self._Q_gamma_scalar())
        return d

    def _x_from_eta(self, S_diag: Optional[np.ndarray] = None) -> np.ndarray:
        """Reconstruct x[0:T] from (x0, eta[1:T]) and S_diag."""
        if self.dim == 0:
            return np.zeros((self.T + 1, 0), float)
        if S_diag is None:
            S_diag = self._S_diag()
        A, u = self._A(), self._u()
        x = np.zeros((self.T + 1, self.dim), float)
        x[0] = self.x0
        for t in range(1, self.T + 1):
            noise = S_diag * self.eta[t - 1]
            x[t] = A @ x[t - 1] + u + noise
        return x

    def _update_eta_from_x(self) -> None:
        """
        Given centred state path x, compute η_t = S^{-1}(x_t - A x_{t-1} - u).
        """
        if self.dim == 0:
            return
        A, u = self._A(), self._u()
        S_diag = self._S_diag()
        S_safe = np.where(S_diag > 0, S_diag, 1e-12)
        eta = np.zeros((self.T, self.dim), float)
        for t in range(1, self.T + 1):
            innov = self.x[t] - (A @ self.x[t - 1] + u)
            eta[t - 1] = innov / S_safe
        self.eta = eta
        self.x0 = self.x[0].copy()

    # ---------------- deterministic mean pieces ---------------- #
    def _mu_det(self, t: int) -> float:
        out = 0.0
        if self.level_mode == "deterministic":
            out += self.m0_alpha
        if (self.trend_mode == "deterministic") and (self.idx_alpha is None):
            out += self.m0_beta * t
        if self.seasonal_mode == "deterministic":
            out += float(self.m0_gamma[t % self.period])
        return out

    def _current_m0_P0(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build m0 and P0 for dynamic coordinates."""
        m0, P0 = [], []
        if self.idx_alpha is not None:
            m0.append(self.m0_alpha)
            P0.append(self.P0_alpha)
        if self.idx_beta is not None:
            m0.append(self.m0_beta)
            P0.append(self.P0_beta)
        if self.seasonal_mode == "dynamic":
            m0.extend(list(self.m0_gamma))
            P0.extend([self.P0_gamma] * (self.period - 1))
        return np.asarray(m0, float), np.asarray(P0, float)

    # ------------------------- FFBS (Kalman + Carter–Kohn) ------------------------- #
    def _ffbs(self) -> np.ndarray:
        """Centred FFBS on x using Q(λ); draw x | y, λ, σ²."""
        if self.dim == 0:
            return self.x.copy()
        H, A, Q, R = self._H(), self._A(), self._Q(), float(self.sigma2)
        m0_vec, P0_diag = self._current_m0_P0()
        m = np.zeros((self.T + 1, self.dim))
        C = np.zeros((self.T + 1, self.dim, self.dim))
        a = np.zeros((self.T + 1, self.dim))
        Rm = np.zeros((self.T + 1, self.dim, self.dim))
        m[0] = m0_vec
        C[0] = np.diag(P0_diag) + 1e-12 * np.eye(self.dim)
        u = self._u()

        # forward
        for t in range(1, self.T + 1):
            a[t]  = A @ m[t - 1] + u
            Rm[t] = A @ C[t - 1] @ A.T + Q
            Rm[t] = 0.5 * (Rm[t] + Rm[t].T) + 1e-12 * np.eye(self.dim)

            resid_mean = float(self.y[t - 1] - self._mu_det(t - 1))
            S = float(H @ Rm[t] @ H.T + R)
            if S <= 0:
                S = float(H @ (Rm[t] + 1e-10 * np.eye(self.dim)) @ H.T + R)
            K = (Rm[t] @ H.T) / S
            v = resid_mean - float(H @ a[t])
            m[t] = a[t] + (K.flatten() * v)
            C[t] = Rm[t] - K @ (H @ Rm[t])
            C[t] = 0.5 * (C[t] + C[t].T) + 1e-12 * np.eye(self.dim)

        # backward
        x = np.zeros_like(self.x)
        x[self.T] = np.random.multivariate_normal(m[self.T], C[self.T])
        for t in range(self.T - 1, -1, -1):
            J = C[t] @ A.T
            J = J @ _spd_solve(Rm[t + 1], np.eye(self.dim))
            mean = m[t] + J @ (x[t + 1] - (A @ m[t] + u))
            cov  = C[t] - J @ Rm[t + 1] @ J.T
            cov  = 0.5 * (cov + cov.T)
            eigmin = float(np.linalg.eigvalsh(cov).min())
            if eigmin < 1e-12:
                cov += (1e-12 - eigmin) * np.eye(cov.shape[0])
            x[t] = np.random.multivariate_normal(mean, cov)
        return x

    # ------------------ Helpers: μ and residuals ------------------ #
    def _mu_from_x(self, x: np.ndarray) -> np.ndarray:
        H = self._H()
        mu = np.zeros(self.T, float)
        for t in range(1, self.T + 1):
            dyn = float(H @ x[t]) if self.dim > 0 else 0.0
            mu[t - 1] = self._mu_det(t - 1) + dyn
        return mu

    def _mu_vec(self) -> np.ndarray:
        """μ_t for the *current* x path."""
        if self.dim == 0:
            return np.array([self._mu_det(t) for t in range(self.T)], float)
        return self._mu_from_x(self.x)

    # ------------------ σ² | rest  (Gamma on precision) ------------------ #
    def update_sigma2(self) -> None:
        e = self.y - self._mu_vec()
        a = self.priors.a_sigma + 0.5 * self.T
        b = self.priors.b_sigma + 0.5 * float(e @ e)
        tau = np.random.gamma(shape=a, scale=1.0 / b)  # precision
        self.sigma2 = 1.0 / max(tau, 1e-300)

    # =============================================================================
    # Process precisions λ_k: double-Gamma (fully conjugate)
    # =============================================================================

    def _compute_cp_stats(self) -> None:
        """
        Precompute sufficient statistics for λ_alpha, λ_beta, λ_gamma from
        innovations w_t = x_t - A x_{t-1} - u.
        """
        if self.dim == 0:
            self._cp_stats = {}
            return

        A, u = self._A(), self._u()
        ssq_alpha = 0.0
        ssq_beta  = 0.0
        ssq_gamma = 0.0
        n_alpha = 0
        n_beta  = 0
        n_gamma = 0

        for t in range(1, self.T + 1):
            innov = self.x[t] - (A @ self.x[t - 1] + u)
            if self.idx_alpha is not None:
                val = float(innov[self.idx_alpha])
                ssq_alpha += val * val
                n_alpha += 1
            if self.idx_beta is not None:
                val = float(innov[self.idx_beta])
                ssq_beta += val * val
                n_beta += 1
            if self.seasonal_mode == "dynamic":
                val = float(innov[self.idx_g_start])
                ssq_gamma += val * val
                n_gamma += 1

        stats: Dict[str, Tuple[float, int]] = {}
        if self.idx_alpha is not None:
            stats["alpha"] = (ssq_alpha, n_alpha)
        if self.idx_beta is not None:
            stats["beta"] = (ssq_beta, n_beta)
        if self.seasonal_mode == "dynamic" and n_gamma > 0:
            stats["gamma"] = (ssq_gamma, n_gamma)
        self._cp_stats = stats

    def update_precisions_double_gamma(self) -> None:
        """
        Conjugate updates for λ_k and τ_k under the double-Gamma prior:

            λ_k | τ_k, w_k ~ Gamma(a_λk + n_k/2, τ_k + 0.5 * sum w_k^2)
            τ_k | λ_k      ~ Gamma(c_λk + a_λk, d_λk + λ_k)

        (shape–rate parameterisation).
        """
        if self.dim == 0:
            return

        self._compute_cp_stats()

        # α-block
        if self.idx_alpha is not None and "alpha" in self._cp_stats:
            ssq, n_eff = self._cp_stats["alpha"]
            a0 = self.priors.a_lambda_alpha
            c0 = self.priors.c_lambda_alpha
            d0 = self.priors.d_lambda_alpha

            a_post = a0 + 0.5 * n_eff
            b_post = self.tau_alpha + 0.5 * ssq
            self.lambda_alpha = np.random.gamma(shape=a_post, scale=1.0 / b_post)

            c_post = c0 + a0
            d_post = d0 + self.lambda_alpha
            self.tau_alpha = np.random.gamma(shape=c_post, scale=1.0 / d_post)

        # β-block
        if self.idx_beta is not None and "beta" in self._cp_stats:
            ssq, n_eff = self._cp_stats["beta"]
            a0 = self.priors.a_lambda_beta
            c0 = self.priors.c_lambda_beta
            d0 = self.priors.d_lambda_beta

            a_post = a0 + 0.5 * n_eff
            b_post = self.tau_beta + 0.5 * ssq
            self.lambda_beta = np.random.gamma(shape=a_post, scale=1.0 / b_post)

            c_post = c0 + a0
            d_post = d0 + self.lambda_beta
            self.tau_beta = np.random.gamma(shape=c_post, scale=1.0 / d_post)

        # γ-block
        if self.seasonal_mode == "dynamic" and "gamma" in self._cp_stats:
            ssq, n_eff = self._cp_stats["gamma"]
            a0 = self.priors.a_lambda_gamma
            c0 = self.priors.c_lambda_gamma
            d0 = self.priors.d_lambda_gamma

            a_post = a0 + 0.5 * n_eff
            b_post = self.tau_gamma + 0.5 * ssq
            self.lambda_gamma = np.random.gamma(shape=a_post, scale=1.0 / b_post)

            c_post = c0 + a0
            d_post = d0 + self.lambda_gamma
            self.tau_gamma = np.random.gamma(shape=c_post, scale=1.0 / d_post)

    # --- m0 | P0, x0 (Normal); P0 | m0, x0 (Inv-Gamma) --- #
    @staticmethod
    def _gibbs_m0_scalar(x0: float, m_prior: float, s_prior: float, P0: float) -> float:
        prec = 1.0 / (s_prior**2) + 1.0 / max(1e-18, P0)
        var = 1.0 / prec
        mean = var * (m_prior / (s_prior**2) + x0 / max(1e-18, P0))
        return float(np.random.normal(mean, math.sqrt(var)))

    def update_m0(self) -> None:
        if self.dim == 0:
            return
        pos = 0
        if self.idx_alpha is not None:
            self.m0_alpha = self._gibbs_m0_scalar(
                float(self.x0[pos]), self.priors.m_m0_alpha, self.priors.s_m0_alpha, self.P0_alpha
            )
            pos += 1
        if self.idx_beta is not None:
            self.m0_beta = self._gibbs_m0_scalar(
                float(self.x0[pos]), self.priors.m_m0_beta, self.priors.s_m0_beta, self.P0_beta
            )
            pos += 1
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
                self.m0_gamma[k] = self._gibbs_m0_scalar(
                    float(self.x0[pos + k]), float(m_prior[k]), s, self.P0_gamma
                )

    def update_P0(self) -> None:
        if self.dim == 0:
            return
        pos = 0
        if self.idx_alpha is not None:
            a = self.priors.a_P0_alpha + 0.5
            b = self.priors.b_P0_alpha + 0.5 * (float(self.x0[pos]) - self.m0_alpha) ** 2
            self.P0_alpha = 1.0 / np.random.gamma(shape=a, scale=1.0 / b)
            pos += 1
        if self.idx_beta is not None:
            a = self.priors.a_P0_beta + 0.5
            b = self.priors.b_P0_beta + 0.5 * (float(self.x0[pos]) - self.m0_beta) ** 2
            self.P0_beta = 1.0 / np.random.gamma(shape=a, scale=1.0 / b)
            pos += 1
        if self.seasonal_mode == "dynamic":
            diffsq = 0.0
            for k in range(self.period - 1):
                diffsq += (float(self.x0[pos + k]) - float(self.m0_gamma[k])) ** 2
            a = self.priors.a_P0_gamma + 0.5 * (self.period - 1)
            b = self.priors.b_P0_gamma + 0.5 * diffsq
            self.P0_gamma = 1.0 / np.random.gamma(shape=a, scale=1.0 / b)

    # --- Deterministic params (all conjugate) --- #
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
                s2 = self._Q_alpha_scalar()
                s2 = max(s2, 1e-12)
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

        # deterministic season: dummy regression with sum-to-zero
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
        if vals is None:
            return "-"
        v = np.asarray(vals, float).ravel()
        if v.size == 0:
            return "[]"
        if v.size <= max_elems:
            return "[" + ", ".join(f"{x:{fmt}}" for x in v) + "]"
        head = ", ".join(f"{x:{fmt}}" for x in v[:max_elems])
        return f"[{head}, …]"

    def _progress_line(self, it: int) -> str:
        parts = [f"[it {it + 1}/{self.cfg.n_iter}]"]
        parts.append(f"σ={math.sqrt(self.sigma2):.3f}")
        if self.idx_alpha is not None:
            Qa = self._Q_alpha_scalar()
            parts.append(f"Qα={Qa:.4g}")
        if self.idx_beta is not None:
            Qb = self._Q_beta_scalar()
            parts.append(f"Qβ={Qb:.4g}")
        if self.seasonal_mode == "dynamic":
            Qg = self._Q_gamma_scalar()
            parts.append(f"Qγ={Qg:.4g}")
        if self.level_mode != "none":
            parts.append(
                f"m0α={self.m0_alpha:.4g} "
                f"P0α={(self.P0_alpha if self.level_mode=='dynamic' else 0.0):.4g}"
            )
        if self.trend_mode != "none":
            parts.append(
                f"m0β={self.m0_beta:.4g} "
                f"P0β={(self.P0_beta if self.trend_mode=='dynamic' else 0.0):.4g}"
            )
        if self.seasonal_mode != "none":
            if self.seasonal_mode == "dynamic":
                g = self._fmt_list(self.m0_gamma, 6, ".4g")
                P0g = self.P0_gamma
            else:
                g = self._fmt_list(self.m0_gamma[:-1], 6, ".4g")
                P0g = 0.0
            parts.append(f"m0γ={g} P0γ={P0g:.4g}")
        return " | ".join(parts)

    # --------------------------------- MCMC --------------------------------- #
    def run(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        save_iters = list(range(cfg.burn, cfg.n_iter, cfg.thin))
        n_kept, keep_idx = len(save_iters), 0

        # allocate storage
        self.keep = {
            "sigma": np.zeros(n_kept, float),
            "mu": np.zeros((n_kept, self.T), float),
        }
        if self.idx_alpha is not None:
            self.keep.update(
                {"Q_alpha": np.zeros(n_kept),
                 "m0_alpha": np.zeros(n_kept),
                 "P0_alpha": np.zeros(n_kept)}
            )
        if self.idx_beta is not None:
            self.keep.update(
                {"Q_beta": np.zeros(n_kept),
                 "m0_beta": np.zeros(n_kept),
                 "P0_beta": np.zeros(n_kept)}
            )
        if self.seasonal_mode == "dynamic":
            self.keep.update(
                {"Q_gamma": np.zeros(n_kept),
                 "m0_gamma": np.zeros((n_kept, self.period - 1)),
                 "P0_gamma": np.zeros(n_kept)}
            )
        if self.dim > 0:
            self.keep["x"] = np.zeros((n_kept, self.T, self.dim))
        if self.level_mode == "deterministic":
            self.keep["m0_alpha"] = np.zeros(n_kept)
        if self.trend_mode == "deterministic":
            self.keep["m0_beta"] = np.zeros(n_kept)
        if self.seasonal_mode == "deterministic":
            self.keep["m0_gamma"] = np.zeros((n_kept, self.period))

        print_every = cfg.progress_every if cfg.progress_every > 0 else max(1, cfg.n_iter // 50) or 1

        for it in range(cfg.n_iter):
            # 1) FFBS: x | y, λ, σ²
            if self.dim > 0:
                self.x = self._ffbs()

                # 2) update λ_k and τ_k (double-Gamma, fully conjugate)
                self.update_precisions_double_gamma()

                # 3) CP → NCP transform: compute eta from (x, λ)
                self._update_eta_from_x()
                # (we keep x as the centred path; eta is stored for diagnostics / possible future ASIS tweaks)

            # 4) m0 and 5) P0 for dynamic coords (use x0)
            if self.dim > 0:
                self.update_m0()
                self.update_P0()

            # 6) deterministic params (level/trend/season)
            self.update_deterministic_params()

            # 7) σ² (Gibbs)
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
                    self.keep["Q_alpha"][keep_idx] = self._Q_alpha_scalar()
                    self.keep["m0_alpha"][keep_idx] = self.m0_alpha
                    self.keep["P0_alpha"][keep_idx] = self.P0_alpha
                if self.idx_beta is not None:
                    self.keep["Q_beta"][keep_idx] = self._Q_beta_scalar()
                    self.keep["m0_beta"][keep_idx] = self.m0_beta
                    self.keep["P0_beta"][keep_idx] = self.P0_beta
                if self.seasonal_mode == "dynamic":
                    self.keep["Q_gamma"][keep_idx] = self._Q_gamma_scalar()
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
        if extra_meta:
            meta.update(extra_meta)
        meta_path = out_npz_path.replace(".npz", ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[save] Posterior -> {out_npz_path}")
        print(f"[save] Metadata  -> {meta_path}")


# ------------------------- CLI / Example run ------------------------------- #
if __name__ == "__main__":
    import argparse
    from datetime import datetime
    import matplotlib.pyplot as plt

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(base_dir)
    from simulator.mean_time_series import Mean_Time_Series  # newest-first

    def _parse_date(s: str | None):
        if not s:
            return datetime.today()
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
            "Gaussian DLM with level/trend/seasonal dummies (newest-first). "
            "Kalman FFBS + conjugate Gibbs for Gaussian parts. "
            "Process precisions λ_k have double-Gamma priors (no slice sampling)."
        )
    )

    # Simulation
    p.add_argument("--T", type=int, default=1000)
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
    p.add_argument("--v0-level", type=float, default=0.05)
    p.add_argument("--m0-trend", type=float, default=1)
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
    p.add_argument("--prior-a-P0-gamma", type=float, default=5)
    p.add_argument("--prior-b-P0-gamma", type=float, default=1.0)

    # Double-Gamma hyperparams for λ_k
    p.add_argument("--prior-a-lambda-alpha", type=float, default=1.0)
    p.add_argument("--prior-c-lambda-alpha", type=float, default=1.0)
    p.add_argument("--prior-d-lambda-alpha", type=float, default=1.0)

    p.add_argument("--prior-a-lambda-beta", type=float, default=1.0)
    p.add_argument("--prior-c-lambda-beta", type=float, default=1.0)
    p.add_argument("--prior-d-lambda-beta", type=float, default=1.0)

    p.add_argument("--prior-a-lambda-gamma", type=float, default=1.0)
    p.add_argument("--prior-c-lambda-gamma", type=float, default=1.0)
    p.add_argument("--prior-d-lambda-gamma", type=float, default=1.0)

    # Sampler configuration
    p.add_argument("--n-iter", type=int, default=10000)
    p.add_argument("--burn", type=int, default=5000)
    p.add_argument("--thin", type=int, default=2)
    p.add_argument("--seed", type=int, default=40)
    p.add_argument("--progress", default=True)
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--out-dir", type=str, default="results/simulations/DLM_dummy_DG")
    p.add_argument("--plot", default=True)
    p.add_argument("--print-summary", default=True)

    # Initial inference values (as SDs; converted to λ inside)
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

    # Simulate
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
    dates_T = truths["index"][:args.T]

    # Priors
    pri_gamma_vec = [0.0] * (args.period - 1)
    if args.prior_m_m0_gamma is not None:
        pri_gamma_vec = _csv_floats_or_none(args.prior_m_m0_gamma) or pri_gamma_vec

    priors = Priors(
        a_sigma=args.prior_a_sigma, b_sigma=args.prior_b_sigma,
        m_m0_alpha=args.prior_m_m0_alpha, s_m0_alpha=args.prior_s_m0_alpha,
        m_m0_beta=args.prior_m_m0_beta,   s_m0_beta=args.prior_s_m0_beta,
        m_m0_gamma=pri_gamma_vec,
        s_m0_gamma=args.prior_s_m0_gamma,
        a_P0_alpha=args.prior_a_P0_alpha, b_P0_alpha=args.prior_b_P0_alpha,
        a_P0_beta=args.prior_a_P0_beta,   b_P0_beta=args.prior_b_P0_beta,
        a_P0_gamma=args.prior_a_P0_gamma, b_P0_gamma=args.prior_b_P0_gamma,
        a_lambda_alpha=args.prior_a_lambda_alpha,
        c_lambda_alpha=args.prior_c_lambda_alpha,
        d_lambda_alpha=args.prior_d_lambda_alpha,
        a_lambda_beta=args.prior_a_lambda_beta,
        c_lambda_beta=args.prior_c_lambda_beta,
        d_lambda_beta=args.prior_d_lambda_beta,
        a_lambda_gamma=args.prior_a_lambda_gamma,
        c_lambda_gamma=args.prior_c_lambda_gamma,
        d_lambda_gamma=args.prior_d_lambda_gamma,
    )

    cfg = SamplerConfig(
        n_iter=int(args.n_iter), burn=int(args.burn), thin=int(args.thin),
        random_seed=int(args.seed), progress=bool(args.progress),
        progress_every=int(args.progress_every),
    )

    # Sampler
    m0_gamma_init = (
        [float(z) for z in (args.m0_gamma_init or "").split(",")] if args.m0_gamma_init else None
    )

    sampler = DLMGibbsDummyDoubleGamma(
        y=y, period=args.period,
        level_mode=args.level_mode, trend_mode=args.trend_mode, seasonal_mode=args.seasonal_mode,
        sigma2_init=args.sigma_init ** 2,
        s_alpha_init=args.s_alpha_init, s_beta_init=args.s_beta_init, s_gamma_init=args.s_gamma_init,
        m0_alpha_init=args.m0_level, m0_beta_init=args.m0_trend,
        P0_alpha_init=args.P0_alpha_init, P0_beta_init=args.P0_beta_init,
        m0_gamma_init=m0_gamma_init, P0_gamma_init=args.P0_gamma_init,
        priors=priors, cfg=cfg,
    )

    sampler.set_truth(
        sigma=mts.sigma, Q=(mts.q_level, mts.q_trend, mts.q_season),
        m0_level=mts.m0_level, m0_trend=mts.m0_trend, m0_season=mts.m0_season,
        P0_level=mts.v0_level, P0_trend=mts.v0_trend, P0_season=mts.v0_season,
    )
    sampler.set_truth_paths(mu=mu_T)

    if args.print_summary:
        print(f"\nSimulated {args.T} observations (σ={mts.sigma}) with modes "
              f"{args.level_mode}/{args.trend_mode}/{args.seasonal_mode}.\n")
        print("Double-Gamma priors for process precisions λ_k:")
        print("  λ_k | τ_k ~ Gamma(a_lambda_k, τ_k)")
        print("  τ_k       ~ Gamma(c_lambda_k, d_lambda_k)\n")

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
            "double_gamma": {
                "alpha": {
                    "a_lambda": priors.a_lambda_alpha,
                    "c_lambda": priors.c_lambda_alpha,
                    "d_lambda": priors.d_lambda_alpha,
                },
                "beta": {
                    "a_lambda": priors.a_lambda_beta,
                    "c_lambda": priors.c_lambda_beta,
                    "d_lambda": priors.d_lambda_beta,
                },
                "gamma": {
                    "a_lambda": priors.a_lambda_gamma,
                    "c_lambda": priors.c_lambda_gamma,
                    "d_lambda": priors.d_lambda_gamma,
                },
            },
            "representation": "dummy_newest_first",
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
                print(f"Q_{k} = {mQ:.4g} (√Q ≈ {math.sqrt(max(mQ, 0.0)):.4g})")

    if args.plot:
        mu_hat = post["mu"].mean(axis=0)
        plt.figure(figsize=(10, 4))
        plt.plot(dates_T, y, label="y_t", lw=1)
        plt.plot(dates_T, mu_T, "--", label="μ_t (truth)")
        plt.plot(dates_T, mu_hat, "-.", label="μ̂_t (post mean)")
        plt.title(
            f"DLM dummy (double-Gamma λ): {args.level_mode}/{args.trend_mode}/{args.seasonal_mode}"
        )
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
