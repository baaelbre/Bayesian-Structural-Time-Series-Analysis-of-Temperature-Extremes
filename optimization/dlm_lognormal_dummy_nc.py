from __future__ import annotations

import json, math, os, warnings, time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
warnings.filterwarnings("ignore", category=DeprecationWarning)

# =============================================================================
# Small utils
# =============================================================================


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
# Priors & Config (FS: Normal priors on process SDs)
# =============================================================================

@dataclass
class Priors:
    # Observation variance σ²: precision τ = 1/σ² ~ Gamma(a_sigma, b_sigma) (shape–rate)
    a_sigma: float = 2.0
    b_sigma: float = 1.0

    # m0 priors (used for static level/trend/seasonal baselines)
    m_m0_alpha: float = 0.0
    s_m0_alpha: float = 10.0
    m_m0_beta: float  = 0.0
    s_m0_beta: float  = 10.0
    m_m0_gamma: Optional[Sequence[float]] = None  # len p-1 (newest-first)
    s_m0_gamma: float = 5.0

    # Initial-state variances P0 ~ InvGamma(a, b)  (kept for compatibility / storage)
    a_P0_alpha: float = 2.0
    b_P0_alpha: float = 1.0
    a_P0_beta:  float = 2.0
    b_P0_beta:  float = 1.0
    a_P0_gamma: float = 2.0
    b_P0_gamma: float = 1.0  # shared across p-1 seasonal coords

    # FS-style Normal priors for process SDs:
    #   s_k | σ² ~ N(0, B0_s_k * σ²)
    B0_s_alpha: float = 1.0
    B0_s_beta:  float = 1.0
    B0_s_gamma: float = 1.0


@dataclass
class SamplerConfig:
    n_iter: int = 5000
    burn: int = 1000
    thin: int = 2
    random_seed: Optional[int] = 40
    progress: bool = True
    progress_every: int = 0  # 0 => ~2% of n_iter
    # slice_w, slice_m kept for compatibility, but unused in FS update
    slice_w: float = 1.0
    slice_m: int = 20


# =============================================================================
# DLM Sampler (FS Normal priors on process SDs + joint regression update)
# =============================================================================

class DLMGibbsConjugate:
    """
    Gaussian structural DLM with:
      • Non-centred parametrisation of the latent states (tilde alpha, tilde beta, A, tilde gamma).
      • FFBS on the non-centred state with fixed transition G_tilde and unit process covariance Q_tilde.
      • FS-style Normal priors on process SDs s_k | σ² ~ N(0, B0_s_k σ²).
      • Joint Gaussian regression update for (m0_alpha, m0_beta, m0_gamma, s_alpha, s_beta, s_gamma).
      • Random sign switches on (s_alpha, s_beta, s_gamma) and corresponding NCP states.

    Externally:
      - Progress lines and posterior storage are expressed in terms of the centred states x_t and Q_k = s_k^2,
        exactly as in the original centred implementation.
      - self.keep['x'] stores the CENTRED states x_t (alpha, beta, gamma dummy states).
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

        # Enforce dynamic/dynamic/dynamic only
        if not (level_mode == trend_mode == seasonal_mode == "dynamic"):
            raise ValueError(
                "This version of DLMGibbsConjugate only supports "
                "level_mode=trend_mode=seasonal_mode='dynamic'."
            )
        self.level_mode, self.trend_mode, self.seasonal_mode = level_mode, trend_mode, seasonal_mode

        self.priors, self.cfg = priors, cfg
        if cfg.random_seed is not None:
            np.random.seed(cfg.random_seed)
            self._rng = np.random.default_rng(cfg.random_seed)
        else:
            self._rng = np.random.default_rng()

        # ------------------ CP layout (for storage / interpretation) ------------------ #
        layout: List[str] = ["alpha", "beta"]
        layout.extend([f"g{k}" for k in range(1, self.period)])
        self._layout = layout
        self.dim = len(layout)

        self.idx_alpha = layout.index("alpha")
        self.idx_beta = layout.index("beta")
        self.idx_g_start = layout.index("g1")
        self.idx_g_end = self.idx_g_start + (self.period - 2)

        self.K_gamma = self.period - 1  # number of seasonal params (baseline + dynamic states)

        # ------------------ Parameters ------------------ #
        # Observation variance
        self.sigma2 = float(sigma2_init)

        # Process SDs (signed, FS-style prior)
        self.s_alpha = float(s_alpha_init)
        self.s_beta = float(s_beta_init)
        self.s_gamma = float(s_gamma_init)

        # Static baselines (m0_alpha, m0_beta, m0_gamma)
        self.m0_alpha = float(m0_alpha_init)  # alpha0
        self.m0_beta = float(m0_beta_init)    # beta0

        if m0_gamma_init is None:
            self.m0_gamma = np.zeros(self.K_gamma, float)
        else:
            g = np.asarray(m0_gamma_init, float)
            if g.size != self.K_gamma:
                raise ValueError("m0_gamma_init must have length p-1 (NEWEST-FIRST)")
            self.m0_gamma = g

        # We keep P0_* for compatibility / storage, but do not update them in NCP.
        self.P0_alpha = float(P0_alpha_init)
        self.P0_beta = float(P0_beta_init)
        self.P0_gamma = float(P0_gamma_init)

        # ------------------ Seasonal baseline design ------------------ #
        # S[t, :] maps m0_gamma -> static seasonal effect at time t (sum-to-zero across p seasons)
        self._season_design = self._build_season_design()

        # ------------------ NCP layout ------------------ #
        # Non-centred state: [tilde_alpha, tilde_beta, A, tilde_g1, ..., tilde_g_{p-1}]
        if self.dim > 0:
            self.dim_ncp = self.dim + 1  # extra A-state
            self.idx_tilde_alpha = 0
            self.idx_tilde_beta = 1
            self.idx_A = 2
            self.idx_tilde_g_start = 3
            self.idx_tilde_g_end = self.idx_tilde_g_start + (self.period - 2)
        else:
            self.dim_ncp = 0

        # Latent NCP path z (tilde states)
        self.z = np.zeros((self.T + 1, self.dim_ncp), float)
        # Centred path x (for storage / mu / progress)
        self.x = np.zeros((self.T + 1, self.dim), float)

        # Build fixed NCP transition and process covariance
        if self.dim_ncp > 0:
            self._G_tilde = self._build_G_tilde()
            self._Q_tilde = self._build_Q_tilde()

        # Map initial NCP path (all zeros) to CP states (dynamic part)
        if self.dim > 0:
            self._refresh_cp_from_ncp()

        # Storage
        self.keep: Dict[str, np.ndarray] = {}

        # Truth overlays
        self.true_sigma: Optional[float] = None
        self.true_Q: Optional[np.ndarray] = None
        self.true_mu_t: Optional[np.ndarray] = None
        self.true_alpha_t: Optional[np.ndarray] = None
        self.true_beta_t: Optional[np.ndarray] = None
        self.true_gamma_t: Optional[np.ndarray] = None

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

    # ----------------------------- Seasonal baseline design ----------------------------- #
    def _build_season_design(self) -> np.ndarray:
        """
        Build static seasonal design S[t, :] mapping m0_gamma (length p-1)
        to a seasonal baseline at time t under a sum-to-zero constraint.

        Seasons are indexed 0..p-1 with period = p.
        For seasons 0..p-2, we use one-hot in coordinates 0..p-2.
        For season p-1, we use -1 in all entries (γ_p = -sum_{j=1}^{p-1} γ_j).
        """
        T, p = self.T, self.period
        K = p - 1
        S = np.zeros((T, K), float)
        for t in range(T):
            season = t % p  # 0..p-1
            if season < K:
                S[t, season] = 1.0
            else:  # last season
                S[t, :] = -1.0
        return S

    # ----------------------------- NCP model matrices ----------------------------- #
    def _build_G_tilde(self) -> np.ndarray:
        """Transition matrix for non-centred state z_t = [tilde_alpha, tilde_beta, A, tilde_gamma...]"""
        if self.dim_ncp == 0:
            return np.zeros((0, 0))
        d = self.dim_ncp
        G = np.zeros((d, d))

        ia = self.idx_tilde_alpha
        ib = self.idx_tilde_beta
        iA = self.idx_A
        gs = self.idx_tilde_g_start
        ge = self.idx_tilde_g_end
        K = ge - gs + 1  # p-1

        # tilde_alpha random walk
        G[ia, ia] = 1.0
        # tilde_beta random walk
        G[ib, ib] = 1.0
        # A_t = A_{t-1} + tilde_beta_{t-1}
        G[iA, iA] = 1.0
        G[iA, ib] = 1.0

        # Seasonal dummy rotation matrix S for tilde_gamma
        if K > 0:
            S = np.zeros((K, K))
            S[0, :] = -1.0
            if K > 1:
                S[1:, :-1] = np.eye(K - 1)
            G[gs:ge + 1, gs:ge + 1] = S

        return G

    def _build_Q_tilde(self) -> np.ndarray:
        """Unit process covariance for non-centred state."""
        if self.dim_ncp == 0:
            return np.zeros((0, 0))
        d = self.dim_ncp
        Q = np.zeros((d, d))

        ia = self.idx_tilde_alpha
        ib = self.idx_tilde_beta
        gs = self.idx_tilde_g_start
        ge = self.idx_tilde_g_end

        Q[ia, ia] = 1.0  # tilde_alpha innovations
        Q[ib, ib] = 1.0  # tilde_beta innovations
        # A has no own innovation (deterministic given tilde_beta)
        for k in range(gs, ge + 1):
            Q[k, k] = 1.0  # seasonal block innovations

        return Q

    # ------------------ Mapping NCP → CP ------------------ #
    def _refresh_cp_from_ncp(self) -> None:
        """
        Compute centred states x_t from non-centred states z_t and current parameters.

        Note: x_t carries only the *dynamic* seasonal component (s_gamma * tilde_gamma),
        while the static seasonal baseline m0_gamma enters μ_t through _season_design.
        """
        if self.dim == 0 or self.dim_ncp == 0:
            self.x = np.zeros((self.T + 1, self.dim), float)
            return

        z = self.z
        T = self.T

        alpha0 = self.m0_alpha
        beta0 = self.m0_beta
        s_alpha = self.s_alpha
        s_beta = self.s_beta
        s_gamma = self.s_gamma

        ia = self.idx_tilde_alpha
        ib = self.idx_tilde_beta
        iA = self.idx_A
        gs_t = self.idx_tilde_g_start
        ge_t = self.idx_tilde_g_end

        t_idx = np.arange(T + 1, dtype=float)

        tilde_alpha = z[:, ia]
        tilde_beta = z[:, ib]
        A_t = z[:, iA]
        tilde_gamma = z[:, gs_t:ge_t + 1]  # shape (T+1, p-1)

        beta_cp = beta0 + s_beta * tilde_beta
        alpha_cp = alpha0 + t_idx * beta0 + s_alpha * tilde_alpha + s_beta * A_t
        gamma_cp = s_gamma * tilde_gamma

        x = np.zeros((T + 1, self.dim), float)
        x[:, self.idx_alpha] = alpha_cp
        x[:, self.idx_beta] = beta_cp
        x[:, self.idx_g_start:self.idx_g_end + 1] = gamma_cp

        self.x = x

    # ------------------ Helpers: μ and residuals ------------------ #
    def _H(self) -> np.ndarray:
        """CP observation row H for the dynamic part (alpha_t + g1_t)."""
        if self.dim == 0:
            return np.zeros((1, 0))
        h = np.zeros(self.dim, float)
        h[self.idx_alpha] = 1.0
        h[self.idx_g_start] = 1.0
        return h.reshape(1, -1)

    def _mu_vec(self) -> np.ndarray:
        """
        Compute μ_t from centred states x_t and static seasonal baseline m0_gamma.

        μ_t = alpha_t + g1_t (from x_t)  +  S[t,:] @ m0_gamma
        """
        if self.dim == 0:
            return np.zeros(self.T, float)
        H = self._H()
        T = self.T
        mu_dyn = np.zeros(T, float)
        for t in range(1, T + 1):
            mu_dyn[t - 1] = float(H @ self.x[t])

        if self.period > 1 and self._season_design.size > 0:
            mu_base = self._season_design @ self.m0_gamma
        else:
            mu_base = 0.0

        return mu_dyn + mu_base

    # ------------------------- FFBS in NCP ------------------------- #
    def _ffbs_ncp(self) -> np.ndarray:
        """
        FFBS on the non-centred state z_t with fixed G_tilde, Q_tilde.

        We work with centred observations:
            y_t' = y_t - c_t,
        where
            c_t = m0_alpha + m0_beta * t + S[t,:] @ m0_gamma
        and
            y_t' = H z_t + ε_t,   ε_t ~ N(0, σ²).
        """
        if self.dim_ncp == 0:
            return self.z.copy()

        G = self._G_tilde
        Q = self._Q_tilde
        sigma2 = float(self.sigma2)

        T = self.T
        dim = self.dim_ncp

        m = np.zeros((T + 1, dim))
        C = np.zeros((T + 1, dim, dim))
        a = np.zeros((T + 1, dim))
        Rm = np.zeros((T + 1, dim, dim))

        # Initial NCP state: centred at zero with tiny covariance
        m[0] = np.zeros(dim)
        C[0] = 1e-6 * np.eye(dim)

        # Observation design for centred observations y_t' = y_t - c_t
        H_vec = np.zeros(dim)
        H_vec[self.idx_tilde_alpha] = self.s_alpha
        H_vec[self.idx_A] = self.s_beta
        H_vec[self.idx_tilde_g_start] = self.s_gamma
        H = H_vec.reshape(1, -1)

        t_idx = np.arange(1, T + 1, dtype=float)
        S = self._season_design

        # Forward pass
        for t in range(1, T + 1):
            a[t] = G @ m[t - 1]
            Rm[t] = G @ C[t - 1] @ G.T + Q
            Rm[t] = 0.5 * (Rm[t] + Rm[t].T) + 1e-12 * np.eye(dim)

            c_t = self.m0_alpha + self.m0_beta * t_idx[t - 1]
            if self.period > 1 and S.size > 0:
                c_t += float(S[t - 1] @ self.m0_gamma)

            y_center = float(self.y[t - 1] - c_t)

            S_var = float(H @ Rm[t] @ H.T + sigma2)
            if S_var <= 0:
                S_var = float(H @ (Rm[t] + 1e-10 * np.eye(dim)) @ H.T + sigma2)
            K = (Rm[t] @ H.T) / S_var
            v = y_center - float(H @ a[t])

            m[t] = a[t] + (K.flatten() * v)
            C[t] = Rm[t] - K @ (H @ Rm[t])
            C[t] = 0.5 * (C[t] + C[t].T) + 1e-12 * np.eye(dim)

        # Backward pass
        z = np.zeros_like(self.z)
        z[T] = np.random.multivariate_normal(m[T], C[T])
        for t in range(T - 1, -1, -1):
            J = C[t] @ G.T
            J = J @ _spd_solve(Rm[t + 1], np.eye(dim))
            mean = m[t] + J @ (z[t + 1] - (G @ m[t]))
            cov = C[t] - J @ Rm[t + 1] @ J.T
            cov = 0.5 * (cov + cov.T)
            cov += max(0.0, 1e-12 - float(np.linalg.eigvalsh(cov).min())) * np.eye(dim)
            z[t] = np.random.multivariate_normal(mean, cov)

        return z

    # ------------------ σ² | rest  (Gamma on precision) ------------------ #
    def update_sigma2(self) -> None:
        e = self.y - self._mu_vec()
        a = self.priors.a_sigma + 0.5 * self.T
        b = self.priors.b_sigma + 0.5 * float(e @ e)
        tau = np.random.gamma(shape=a, scale=1.0 / b)  # precision
        self.sigma2 = 1.0 / max(tau, 1e-300)

    # --------------- Joint FS update for (m0_alpha, m0_beta, m0_gamma, s_alpha, s_beta, s_gamma) --------------- #
    def update_beta_fs(self) -> None:
        """
        FS-style Gaussian regression update:

            y_t = m0_alpha + t m0_beta + S[t,:] @ m0_gamma
                  + s_alpha * tilde_alpha_t + s_beta * A_t + s_gamma * tilde_gamma1_t
                  + ε_t,

        with Normal priors:
          - (m0_alpha, m0_beta, m0_gamma) ~ N(m_prior, diag(s_m0_*^2))
          - s_k | σ² ~ N(0, B0_s_k σ²).
        """
        if self.dim_ncp == 0:
            return

        T = self.T
        ia = self.idx_tilde_alpha
        iA = self.idx_A
        gs = self.idx_tilde_g_start

        t_idx = np.arange(1, T + 1, dtype=float)
        S = self._season_design

        tilde_alpha = self.z[1:, ia]
        A_t = self.z[1:, iA]
        tilde_gamma1 = self.z[1:, gs]

        # Design matrix:
        #   X_static = [1, t, S[t,:]]  (for m0_alpha, m0_beta, m0_gamma)
        #   X_dyn    = [tilde_alpha_t, A_t, tilde_gamma1_t] (for s_alpha, s_beta, s_gamma)
        if self.K_gamma > 0:
            X_static = np.column_stack([np.ones(T, float), t_idx, S])
        else:
            X_static = np.column_stack([np.ones(T, float), t_idx])
        X_dyn = np.column_stack([tilde_alpha, A_t, tilde_gamma1])
        X = np.column_stack([X_static, X_dyn])  # shape (T, 2 + K_gamma + 3)

        y = self.y
        d = X.shape[1]  # = 2 + K_gamma + 3

        # Prior mean vector
        m_prior = np.zeros(d, float)
        m_prior[0] = self.priors.m_m0_alpha
        m_prior[1] = self.priors.m_m0_beta

        if self.K_gamma > 0:
            if self.priors.m_m0_gamma is not None:
                m_gamma_prior = np.asarray(self.priors.m_m0_gamma, float)
                if m_gamma_prior.size != self.K_gamma:
                    raise ValueError("Priors.m_m0_gamma must have length p-1")
            else:
                m_gamma_prior = np.zeros(self.K_gamma, float)
            m_prior[2 : 2 + self.K_gamma] = m_gamma_prior

        # Prior variances (diagonal)
        s2_prior = np.zeros(d, float)
        # Baselines (independent of σ²)
        s2_prior[0] = self.priors.s_m0_alpha**2
        s2_prior[1] = self.priors.s_m0_beta**2
        if self.K_gamma > 0:
            s2_prior[2 : 2 + self.K_gamma] = self.priors.s_m0_gamma**2

        # Process SDs: s_k | σ² ~ N(0, B0_s_k σ²)
        idx_s_alpha = 2 + self.K_gamma
        idx_s_beta  = 3 + self.K_gamma
        idx_s_gamma = 4 + self.K_gamma

        s2_prior[idx_s_alpha] = self.priors.B0_s_alpha * self.sigma2
        s2_prior[idx_s_beta]  = self.priors.B0_s_beta  * self.sigma2
        s2_prior[idx_s_gamma] = self.priors.B0_s_gamma * self.sigma2

        eps = 1e-12
        s2_prior = np.maximum(s2_prior, eps)
        V_prior = np.diag(s2_prior)

        XtX = X.T @ X
        Xt_y = X.T @ y
        sigma2 = self.sigma2

        V_prior_inv = np.linalg.inv(V_prior)
        prec_post = XtX / sigma2 + V_prior_inv
        cov_post = _spd_solve(prec_post, np.eye(d))
        mean_post = cov_post @ (Xt_y / sigma2 + V_prior_inv @ m_prior)

        beta = np.random.multivariate_normal(mean_post, cov_post)

        # Assign back to parameters
        self.m0_alpha = float(beta[0])
        self.m0_beta = float(beta[1])
        if self.K_gamma > 0:
            self.m0_gamma = beta[2 : 2 + self.K_gamma].copy()
        self.s_alpha = float(beta[idx_s_alpha])
        self.s_beta  = float(beta[idx_s_beta])
        self.s_gamma = float(beta[idx_s_gamma])

    # --------------- Random sign switches (FS multimodality move) --------------- #
    def random_sign_switches(self) -> None:
        """
        Flip signs of (s_alpha, s_beta, s_gamma) and corresponding NCP states
        with probability 1/2 each, leaving y|z invariant and improving mixing
        in the ±√Q directions.
        """
        if self.dim_ncp == 0:
            return

        # α block: s_alpha and tilde_alpha
        if self._rng.random() < 0.5:
            self.s_alpha *= -1.0
            self.z[:, self.idx_tilde_alpha] *= -1.0

        # β block: s_beta and BOTH tilde_beta and A
        if self._rng.random() < 0.5:
            self.s_beta *= -1.0
            self.z[:, self.idx_tilde_beta] *= -1.0
            self.z[:, self.idx_A] *= -1.0

        # γ block: s_gamma and all seasonal NCP coords
        if self.K_gamma > 0 and self._rng.random() < 0.5:
            self.s_gamma *= -1.0
            self.z[:, self.idx_tilde_g_start : self.idx_tilde_g_end + 1] *= -1.0

        self._refresh_cp_from_ncp()

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
        parts.append(f"Qα={self.s_alpha**2:.4g}")
        parts.append(f"Qβ={self.s_beta**2:.4g}")
        parts.append(f"Qγ={self.s_gamma**2:.4g}")
        parts.append(f"m0α={self.m0_alpha:.4g} P0α={self.P0_alpha:.4g}")
        parts.append(f"m0β={self.m0_beta:.4g} P0β={self.P0_beta:.4g}")
        g = self._fmt_list(self.m0_gamma, 6, ".4g")
        parts.append(f"m0γ={g} P0γ={self.P0_gamma:.4g}")
        return " | ".join(parts)

    # --------------------------------- MCMC --------------------------------- #
    def run(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        save_iters = list(range(cfg.burn, cfg.n_iter, cfg.thin))
        n_kept, keep_idx = len(save_iters), 0

        # allocate storage (same interface as before, plus signed s_*)
        self.keep = {
            "sigma": np.zeros(n_kept, float),
            "mu": np.zeros((n_kept, self.T), float),

            "Q_alpha": np.zeros(n_kept),
            "s_alpha": np.zeros(n_kept),  # signed process SD α
            "m0_alpha": np.zeros(n_kept),
            "P0_alpha": np.zeros(n_kept),

            "Q_beta": np.zeros(n_kept),
            "s_beta": np.zeros(n_kept),   # signed process SD β
            "m0_beta": np.zeros(n_kept),
            "P0_beta": np.zeros(n_kept),

            "Q_gamma": np.zeros(n_kept),
            "s_gamma": np.zeros(n_kept),  # signed process SD γ
            "m0_gamma": np.zeros((n_kept, self.period - 1)),
            "P0_gamma": np.zeros(n_kept),

            "x": np.zeros((n_kept, self.T, self.dim)),
        }

        print_every = cfg.progress_every if cfg.progress_every > 0 else max(1, cfg.n_iter // 50) or 1

        for it in range(cfg.n_iter):
            # 1) FFBS in NCP
            if self.dim_ncp > 0:
                self.z = self._ffbs_ncp()
                self._refresh_cp_from_ncp()

            # 2) FS joint regression update for (m0_*, s_*)
            if self.dim_ncp > 0:
                self.update_beta_fs()
                self._refresh_cp_from_ncp()

            # 3) Random sign switches in ±√Q directions
            if self.dim_ncp > 0:
                self.random_sign_switches()

            # 4) σ² (Gibbs, same as before)
            self.update_sigma2()

            # progress
            if cfg.progress and ((it + 1) % print_every == 0 or it == cfg.n_iter - 1):
                print(self._progress_line(it))

            # save centred posterior samples
            if it in save_iters:
                mu = self._mu_vec()
                self.keep["mu"][keep_idx, :] = mu
                self.keep["sigma"][keep_idx] = math.sqrt(self.sigma2)

                # α block
                self.keep["Q_alpha"][keep_idx] = self.s_alpha**2
                self.keep["s_alpha"][keep_idx] = self.s_alpha
                self.keep["m0_alpha"][keep_idx] = self.m0_alpha
                self.keep["P0_alpha"][keep_idx] = self.P0_alpha

                # β block
                self.keep["Q_beta"][keep_idx] = self.s_beta**2
                self.keep["s_beta"][keep_idx] = self.s_beta
                self.keep["m0_beta"][keep_idx] = self.m0_beta
                self.keep["P0_beta"][keep_idx] = self.P0_beta

                # γ block
                self.keep["Q_gamma"][keep_idx] = self.s_gamma**2
                self.keep["s_gamma"][keep_idx] = self.s_gamma
                self.keep["m0_gamma"][keep_idx, :] = self.m0_gamma
                self.keep["P0_gamma"][keep_idx] = self.P0_gamma

                if self.dim > 0:
                    self.keep["x"][keep_idx, :, :] = self.x[1 : self.T + 1, :]
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
    import argparse, sys
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
            "Kalman FFBS + FS-style Gibbs for Gaussian DLM "
            "(dynamic/dynamic/dynamic, newest-first seasonal). "
            "Process SDs have Normal priors s_k | σ² ~ N(0, B0_s_k σ²) and "
            "are updated via joint regression, with random sign switches."
        )
    )

    # Simulation
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--period", type=int, default=4)
    p.add_argument("--start-date", type=str, default="2000-01-01")
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

    # FS Normal priors (process SDs: s_k | σ² ~ N(0, B0_s_k σ²))
    p.add_argument("--prior-B0-s-alpha", type=float, default=1e-4)
    p.add_argument("--prior-B0-s-beta",  type=float, default=1e-7)
    p.add_argument("--prior-B0-s-gamma", type=float, default=1e-4)

    # Sampler configuration
    p.add_argument("--n-iter", type=int, default=10000)
    p.add_argument("--burn", type=int, default=5000)
    p.add_argument("--thin", type=int, default=2)
    p.add_argument("--seed", type=int, default=40)
    p.add_argument("--progress", default=True)
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--slice-w", type=float, default=1.0)   # unused, kept for compatibility
    p.add_argument("--slice-m", type=int, default=20)      # unused, kept for compatibility
    p.add_argument("--out-dir", type=str, default="results/simulations/DLM")
    p.add_argument("--plot", default=True)
    p.add_argument("--print-summary", default=True)

    # Initial inference values
    p.add_argument("--sigma-init", type=float, default=1.0)
    p.add_argument("--s-alpha-init", type=float, default=1e-2)
    p.add_argument("--s-beta-init",  type=float, default=1e-2)
    p.add_argument("--s-gamma-init", type=float, default=1e-2)
    p.add_argument("--m0-gamma-init", type=str, default=None)
    p.add_argument("--P0-alpha-init", type=float, default=0.25)
    p.add_argument("--P0-beta-init",  type=float, default=0.25)
    p.add_argument("--P0-gamma-init", type=float, default=0.25)

    args = p.parse_args()
    np.random.seed(args.seed)

    start_date = _parse_date(args.start_date) if hasattr(args, "start-date") else _parse_date(args.start_date)
    m0_season = [5.0] * (args.period - 1)
    v0_season = [0.25] * (args.period - 1)

    mts = Mean_Time_Series(
        sigma=args.sigma,
        period=args.period,
        level_mode="dynamic",
        trend_mode="dynamic",
        seasonal_mode="dynamic",
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
    plt.plot(y)
    plt.show()

    # Priors
    if args.prior_m_m0_gamma is not None and args.prior_m_m0_gamma.strip() != "":
        pri_gamma_vec = [float(z) for z in args.prior_m_m0_gamma.split(",")]
    else:
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
        B0_s_alpha=args.prior_B0_s_alpha,
        B0_s_beta=args.prior_B0_s_beta,
        B0_s_gamma=args.prior_B0_s_gamma,
    )

    cfg = SamplerConfig(
        n_iter=int(args.n_iter), burn=int(args.burn), thin=int(args.thin),
        random_seed=int(args.seed), progress=bool(args.progress),
        progress_every=int(args.progress_every),
        slice_w=float(args.slice_w), slice_m=int(args.slice_m),
    )

    # Sampler
    m0_gamma_init = (
        [float(z) for z in (args.m0_gamma_init or "").split(",")] if args.m0_gamma_init else None
    )

    sampler = DLMGibbsConjugate(
        y=y, period=args.period,
        level_mode="dynamic", trend_mode="dynamic", seasonal_mode="dynamic",
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
        print(f"\nSimulated {args.T} observations (σ={mts.sigma}) "
              f"with modes dynamic/dynamic/dynamic.\n")
        print("FS-style Normal priors for process SDs (s_k | σ²):")
        print(f"  alpha: s_alpha ~ N(0, {priors.B0_s_alpha:.3g} * σ²)")
        print(f"  beta : s_beta  ~ N(0, {priors.B0_s_beta :.3g} * σ²)")
        print(f"  gamma: s_gamma ~ N(0, {priors.B0_s_gamma:.3g} * σ²)\n")

    # Run
    t0 = time.time()
    post = sampler.run()
    elapsed = time.time() - t0
    print(f"\n[Run completed in {elapsed:.1f}s]")

    out_dir = os.path.join(
        args.out_dir,
        f"dynamic-dynamic-dynamic_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(out_dir, exist_ok=True)
    sampler.save_posterior(
        out_npz_path=os.path.join(out_dir, "posterior.npz"),
        extra_meta={
            "elapsed_seconds": float(elapsed),
            "fs_priors": {
                "B0_s_alpha": priors.B0_s_alpha,
                "B0_s_beta":  priors.B0_s_beta,
                "B0_s_gamma": priors.B0_s_gamma,
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

    if args.plot:
        mu_hat = post["mu"].mean(axis=0)
        plt.figure(figsize=(10, 4))
        plt.plot(dates_T, y, label="y_t", lw=1)
        plt.plot(dates_T, mu_T, "--", label="μ_t (truth)")
        plt.plot(dates_T, mu_hat, "-.", label="μ̂_t (post mean)")
        plt.title("DLM (FS Normal priors + regression) dynamic/dynamic/dynamic")
        plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()
