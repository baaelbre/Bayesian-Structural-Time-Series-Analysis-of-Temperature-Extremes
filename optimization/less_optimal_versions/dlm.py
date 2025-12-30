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
    """Solve M x = B for SPD M with a bit of regularisation."""
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


def _rand_inv_gauss(mu: float, lam: float, rng: np.random.Generator) -> float:
    """
    Draw from an inverse-Gaussian IG(mu, lam) using the Michael-Schucany-Haas method.
    """
    if mu <= 0 or lam <= 0:
        raise ValueError("Inverse-Gaussian requires mu>0, lam>0")
    v = rng.normal()
    y = v * v
    mu2 = mu * mu
    term = mu2 * y
    x = mu + term / (2.0 * lam) - (mu / (2.0 * lam)) * math.sqrt(4.0 * mu * lam * y + term * y)
    u = rng.random()
    if u <= mu / (mu + x):
        return x
    return mu2 / x


# =============================================================================
# Priors & Config (hierarchical Bayesian lasso prior on process SDs)
# =============================================================================

@dataclass
class Priors:
    # Observation variance σ²: precision τ = 1/σ² ~ Gamma(a_sigma, b_sigma) (shape–rate)
    a_sigma: float = 2.0
    b_sigma: float = 1.0

    # m0 priors (used for static level/trend/seasonal baselines)
    m0_alpha: float = 0.0
    P0_alpha: float = 10.0
    m0_beta: float  = 0.0
    P0_beta: float  = 10.0
    m0_gamma: Optional[Sequence[float]] = None  # len p-1 (newest-first)
    P0_gamma: float = 5.0

    # Hierarchical Bayesian lasso hyperparameters for process SDs:
    #   s_k | τ_k, σ² ~ N(0, σ² τ_k)
    #   τ_k | λ² ~ Exp(λ²/2)
    #   λ²  ~ Gamma(a_lambda, b_lambda)   (shape–rate)
    a_lambda: float = 1.0
    b_lambda: float = 1.0


@dataclass
class SamplerConfig:
    n_iter: int = 5000
    burn: int = 1000
    thin: int = 2
    random_seed: Optional[int] = 40
    progress: bool = True
    progress_every: int = 0  # 0 => ~2% of n_iter
    # slice_w, slice_m kept for compatibility (unused)
    slice_w: float = 1.0
    slice_m: int = 20


# =============================================================================
# DLM Sampler (Bayesian lasso prior on process SDs + regression update)
# =============================================================================

class DLMGibbsConjugate:
    """
    Gaussian structural DLM with:
      • Non-centred parametrisation of the latent states (tilde alpha, tilde beta, A,
        tilde gamma).
      • FFBS on the non-centred state with fixed transition G_tilde and unit
        process covariance Q_tilde.
      • Hierarchical Bayesian lasso prior on process SDs s_k:
            s_k | τ_k, σ² ~ N(0, σ² τ_k),
            τ_k | λ² ~ Exp(λ² / 2),
            λ² ~ Gamma(a_lambda, b_lambda).
      • Joint Gaussian regression update for (alpha0, beta0, gamma0, s_alpha,
        s_beta, s_gamma) conditional on (τ_k, λ²), *implemented in a centred
        time parametrisation*:
            α_c, β, γ, s_α, s_β, s_γ
        with t_c = t - mean(t), and then transformed back to:
            α0 = α_c - t̄ β,    β0 = β.

      • Random sign switches on (s_alpha, s_beta, s_gamma) and corresponding NCP
        states.

    Externally:
      - Progress lines and posterior storage are expressed in terms of the
        centred states x_t and Q_k = s_k^2.
      - self.keep['x'] stores the CENTRED states x_t (alpha, beta, gamma dummy
        states).
      - alpha0, beta0 keep the *original* interpretation in y_t = alpha0 + t beta0 + ...
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
        alpha0: float = 0.0,
        beta0: float = 0.0,
        gamma0: Optional[Sequence[float]] = None,  # len p-1 (NEWEST-FIRST)
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

        # Precompute time index and its centre (used for centering in regression)
        self._t_idx = np.arange(1, self.T + 1, dtype=float)  # 1..T
        self._t_center = float(self._t_idx.mean())           # t̄

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

        # Process SDs (signed)
        self.s_alpha = float(s_alpha_init)
        self.s_beta = float(s_beta_init)
        self.s_gamma = float(s_gamma_init)

        # Lasso local scales τ_k and global shrinkage λ²
        self.tau_alpha = 1.0  # mixing variance for s_alpha
        self.tau_beta = 1.0   # mixing variance for s_beta
        self.tau_gamma = 1.0  # mixing variance for s_gamma
        self.lambda2 = 1.0    # global lasso parameter λ²

        # Static baselines (alpha0, beta0, gamma0) in the ORIGINAL parametrisation
        # used by the state mapping and FFBS:
        #   y_t ≈ alpha0 + beta0 * t + season + dynamic.
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)

        if gamma0 is None:
            self.gamma0 = np.zeros(self.K_gamma, float)
        else:
            g = np.asarray(gamma0, float)
            if g.size != self.K_gamma:
                raise ValueError("gamma0 must have length p-1 (NEWEST-FIRST)")
            self.gamma0 = g

        # ------------------ Seasonal baseline design ------------------ #
        # S[t, :] maps gamma0 -> static seasonal effect at time t (sum-to-zero across p seasons)
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
        Build static seasonal design S[t, :] mapping gamma0 (length p-1)
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
        while the static seasonal baseline gamma0 enters μ_t through _season_design.
        """
        if self.dim == 0 or self.dim_ncp == 0:
            self.x = np.zeros((self.T + 1, self.dim), float)
            return

        z = self.z
        T = self.T

        alpha0 = self.alpha0
        beta0 = self.beta0
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
        Compute μ_t from centred states x_t and static seasonal baseline gamma0.

        μ_t = alpha_t + g1_t (from x_t)  +  S[t,:] @ gamma0
        """
        if self.dim == 0:
            return np.zeros(self.T, float)
        H = self._H()
        T = self.T
        mu_dyn = np.zeros(T, float)
        for t in range(1, T + 1):
            mu_dyn[t - 1] = float(H @ self.x[t])

        if self.period > 1 and self._season_design.size > 0:
            mu_base = self._season_design @ self.gamma0
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
            c_t = self.alpha0 + self.beta0 * t + self._season_design[t,:] @ self.gamma0
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

            c_t = self.alpha0 + self.beta0 * t_idx[t - 1]
            if self.period > 1 and S.size > 0:
                c_t += float(S[t - 1] @ self.gamma0)

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
        K = 3  # number of s_k blocks: alpha, beta, gamma
        ss_resid = float(e @ e)
        ss_lasso = (
            self.s_alpha**2 / max(self.tau_alpha, 1e-12)
            + self.s_beta**2  / max(self.tau_beta,  1e-12)
            + self.s_gamma**2 / max(self.tau_gamma, 1e-12)
        )
        a_post = self.priors.a_sigma + 0.5 * (self.T + K)
        b_post = self.priors.b_sigma + 0.5 * (ss_resid + ss_lasso)
        tau = np.random.gamma(shape=a_post, scale=1.0 / max(b_post, 1e-300))
        self.sigma2 = 1.0 / max(tau, 1e-300)

    # --------------- Joint regression update for (m0_*, s_*) --------------- #
    def update_beta_fs(self) -> None:
        """
        Gaussian regression update (FS style), but with hierarchical lasso prior on s_k,
        using a *centred* time covariate for better α0–β0 mixing.

        Model (reparametrised):

            y_t = α_c + β * (t - t̄) + S[t,:] γ0
                  + s_alpha * tilde_alpha_t + s_beta * A_t + s_gamma * tilde_gamma1_t
                  + ε_t,

        where t̄ is the mean of {1, …, T}. We sample θ_c = (α_c, β, γ0, s_α, s_β, s_γ),
        then transform back to the ORIGINAL parametrisation used elsewhere:

            alpha0 = α_c - t̄ * β
            beta0  = β.
        """
        if self.dim_ncp == 0:
            return

        T = self.T
        ia = self.idx_tilde_alpha
        iA = self.idx_A
        gs = self.idx_tilde_g_start

        t_idx = self._t_idx                    # 1..T
        t_center = self._t_center             # t̄
        t_c = t_idx - t_center                # centred time

        S = self._season_design

        tilde_alpha = self.z[1:, ia]
        A_t = self.z[1:, iA]
        tilde_gamma1 = self.z[1:, gs]

        # Design matrix:
        #   X_static_c = [1, (t - t̄), S[t,:]]  (for α_c, β, γ0)
        #   X_dyn      = [tilde_alpha_t, A_t, tilde_gamma1_t] (for s_alpha, s_beta, s_gamma)
        if self.K_gamma > 0:
            X_static = np.column_stack([np.ones(T, float), t_c, S])
        else:
            X_static = np.column_stack([np.ones(T, float), t_c])
        X_dyn = np.column_stack([tilde_alpha, A_t, tilde_gamma1])
        X = np.column_stack([X_static, X_dyn])  # shape (T, 2 + K_gamma + 3)

        y = self.y
        d = X.shape[1]  # = 2 + K_gamma + 3

        # Prior mean vector for θ_c = (α_c, β, γ0, s_α, s_β, s_γ)
        m_prior = np.zeros(d, float)

        # We have priors on (alpha0, beta0) in the original parametrisation.
        # Rough but practical choice: let α_c prior mean correspond to the mean path
        # implied by (m0_alpha, m0_beta) at t̄:
        #   α_c ≈ m0_alpha + m0_beta * t̄
        m_prior[0] = self.priors.m0_alpha + self.priors.m0_beta * t_center
        m_prior[1] = self.priors.m0_beta

        if self.K_gamma > 0:
            if self.priors.m0_gamma is not None:
                m_gamma_prior = np.asarray(self.priors.m0_gamma, float)
                if m_gamma_prior.size != self.K_gamma:
                    raise ValueError("Priors.m0_gamma must have length p-1")
            else:
                m_gamma_prior = np.zeros(self.K_gamma, float)
            m_prior[2 : 2 + self.K_gamma] = m_gamma_prior

        # Prior variances (diagonal)
        s2_prior = np.zeros(d, float)
        eps = 1e-12
        # Baselines (here interpreted as priors on α_c and β)
        s2_prior[0] = self.priors.P0_alpha
        s2_prior[1] = self.priors.P0_beta
        if self.K_gamma > 0:
            s2_prior[2 : 2 + self.K_gamma] = self.priors.P0_gamma

        # Process SDs: s_k | τ_k, σ² ~ N(0, σ² τ_k)
        idx_s_alpha = 2 + self.K_gamma
        idx_s_beta  = 3 + self.K_gamma
        idx_s_gamma = 4 + self.K_gamma

        sigma2 = self.sigma2

        s2_prior[idx_s_alpha] = max(sigma2 * self.tau_alpha, eps)
        s2_prior[idx_s_beta]  = max(sigma2 * self.tau_beta,  eps)
        s2_prior[idx_s_gamma] = max(sigma2 * self.tau_gamma, eps)

        s2_prior = np.maximum(s2_prior, eps)
        V_prior = np.diag(s2_prior)

        XtX = X.T @ X
        Xt_y = X.T @ y

        V_prior_inv = np.linalg.inv(V_prior)
        prec_post = XtX / sigma2 + V_prior_inv
        cov_post = _spd_solve(prec_post, np.eye(d))
        mean_post = cov_post @ (Xt_y / sigma2 + V_prior_inv @ m_prior)

        theta_c = np.random.multivariate_normal(mean_post, cov_post)

        # Extract α_c, β and transform back to (alpha0, beta0)
        alpha_c = float(theta_c[0])
        beta0 = float(theta_c[1])

        self.beta0 = beta0
        self.alpha0 = alpha_c - t_center * beta0  # ORIGINAL intercept at t=0

        # Seasonal baselines
        if self.K_gamma > 0:
            self.gamma0 = theta_c[2 : 2 + self.K_gamma].copy()

        # Process SDs (signed)
        self.s_alpha = float(theta_c[idx_s_alpha])
        self.s_beta  = float(theta_c[idx_s_beta])
        self.s_gamma = float(theta_c[idx_s_gamma])

    # --------------- Lasso local/global scale updates --------------- #
    def update_lasso_scales(self) -> None:
        """
        Update local scales τ_k and global λ² under the hierarchy:
            s_k | τ_k, σ² ~ N(0, σ² τ_k),
            τ_k | λ² ~ Exp(λ²/2),
            λ² ~ Gamma(a_lambda, b_lambda).

        Full conditionals (Park & Casella style):
          τ_k | s_k, σ², λ² ~ IG(μ_k, λ²),
             μ_k = sqrt(λ² σ² / s_k²),
          λ² | τ_k         ~ Gamma(a_lambda + K, b_lambda + 0.5 Σ τ_k),
          with K = 3 here (α, β, γ).
        """
        if self.dim_ncp == 0:
            return

        sigma2 = max(self.sigma2, 1e-12)
        a_lam = self.priors.a_lambda
        b_lam = self.priors.b_lambda
        rng = self._rng
        eps = 1e-16

        # --- local scales τ_k | s_k, σ², λ² (Inverse-Gaussian) ---
        lam2 = max(self.lambda2, 1e-12)

        # α
        s2_alpha = self.s_alpha**2
        if s2_alpha < eps:
            self.tau_alpha = 1.0
        else:
            mu_alpha = math.sqrt(lam2 * sigma2 / s2_alpha)
            self.tau_alpha = _rand_inv_gauss(mu_alpha, lam2, rng)

        # β
        s2_beta = self.s_beta**2
        if s2_beta < eps:
            self.tau_beta = 1.0
        else:
            mu_beta = math.sqrt(lam2 * sigma2 / s2_beta)
            self.tau_beta = _rand_inv_gauss(mu_beta, lam2, rng)

        # γ
        s2_gamma = self.s_gamma**2
        if s2_gamma < eps:
            self.tau_gamma = 1.0
        else:
            mu_gamma = math.sqrt(lam2 * sigma2 / s2_gamma)
            self.tau_gamma = _rand_inv_gauss(mu_gamma, lam2, rng)

        # --- global scale λ² | τ_k (Gamma) ---
        K = 3
        shape = a_lam + K
        rate = b_lam + 0.5 * (self.tau_alpha + self.tau_beta + self.tau_gamma)
        self.lambda2 = np.random.gamma(shape=shape, scale=1.0 / max(rate, 1e-12))

    # --------------- Random sign switches (multimodality move) --------------- #
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
        parts.append(f"α0={self.alpha0:.4g}")
        parts.append(f"β0={self.beta0:.4g}")
        g = self._fmt_list(self.gamma0, 6, ".4g")
        parts.append(f"γ0={g}")
        parts.append(
            "τloc=["
            + ", ".join(
                f"{x:.3g}" for x in [self.tau_alpha, self.tau_beta, self.tau_gamma]
            )
            + "]"
        )
        parts.append(f"λ²={self.lambda2:.4g}")
        return " | ".join(parts)

    # --------------------------------- MCMC --------------------------------- #
    def run(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        save_iters = list(range(cfg.burn, cfg.n_iter, cfg.thin))
        n_kept, keep_idx = len(save_iters), 0

        # allocate storage (same interface as before, plus signed s_* and lasso scales)
        self.keep = {
            "sigma": np.zeros(n_kept, float),
            "mu": np.zeros((n_kept, self.T), float),

            "Q_alpha": np.zeros(n_kept),
            "s_alpha": np.zeros(n_kept),
            "alpha0": np.zeros(n_kept),

            "Q_beta": np.zeros(n_kept),
            "s_beta": np.zeros(n_kept),
            "beta0": np.zeros(n_kept),
            
            "Q_gamma": np.zeros(n_kept),
            "s_gamma": np.zeros(n_kept),
            "gamma0": np.zeros((n_kept, self.period - 1)),

            # Lasso scales
            "tau_alpha": np.zeros(n_kept),
            "tau_beta": np.zeros(n_kept),
            "tau_gamma": np.zeros(n_kept),
            "lambda2": np.zeros(n_kept),

            "x": np.zeros((n_kept, self.T, self.dim)),
        }

        print_every = cfg.progress_every if cfg.progress_every > 0 else max(1, cfg.n_iter // 50) or 1

        for it in range(cfg.n_iter):
            # 1) FFBS in NCP
            if self.dim_ncp > 0:
                self.z = self._ffbs_ncp()
                self._refresh_cp_from_ncp()

            # 2) Regression update for (alpha0, beta0, gamma0, s_*)
            if self.dim_ncp > 0:
                self.update_beta_fs()
                self._refresh_cp_from_ncp()

            # 3) Random sign switches in ±√Q directions
            if self.dim_ncp > 0:
                self.random_sign_switches()

            # 4) Update hierarchical lasso local/global scales
            if self.dim_ncp > 0:
                self.update_lasso_scales()

            # 5) σ² (Gibbs, same as before)
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
                self.keep["alpha0"][keep_idx] = self.alpha0

                # β block
                self.keep["Q_beta"][keep_idx] = self.s_beta**2
                self.keep["s_beta"][keep_idx] = self.s_beta
                self.keep["beta0"][keep_idx] = self.beta0

                # γ block
                self.keep["Q_gamma"][keep_idx] = self.s_gamma**2
                self.keep["s_gamma"][keep_idx] = self.s_gamma
                self.keep["gamma0"][keep_idx, :] = self.gamma0

                # Lasso scales
                self.keep["tau_alpha"][keep_idx] = self.tau_alpha
                self.keep["tau_beta"][keep_idx] = self.tau_beta
                self.keep["tau_gamma"][keep_idx] = self.tau_gamma
                self.keep["lambda2"][keep_idx] = self.lambda2

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
            "time_center": float(self._t_center),
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

    p = argparse.ArgumentParser(
        description=(
            "Kalman FFBS + Gibbs for Gaussian DLM "
            "(dynamic/dynamic/dynamic, newest-first seasonal). "
            "Process SDs have a hierarchical Bayesian lasso prior "
            "s_k | τ_k, σ² ~ N(0, σ² τ_k), τ_k | λ² ~ Exp(λ²/2)."
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
    p.add_argument("--m0-trend", type=float, default=.1)
    p.add_argument("--v0-trend", type=float, default=0.05)
    p.add_argument("--m0-season", type=str, default=None)
    p.add_argument("--v0-season", type=str, default=None)

    # Priors
    p.add_argument("--prior-a-sigma", type=float, default=2.0)
    p.add_argument("--prior-b-sigma", type=float, default=1.0)
    p.add_argument("--prior-m0-alpha", type=float, default=0.0)
    p.add_argument("--prior-P0-alpha", type=float, default=10)
    p.add_argument("--prior-m0-beta",  type=float, default=0)
    p.add_argument("--prior-P0-beta",  type=float, default=10)
    p.add_argument("--prior-m0-gamma", type=str, default=None)
    p.add_argument("--prior-P0-gamma", type=float, default=5)
    # Lasso hyperparameters
    p.add_argument("--prior-a-lambda", type=float, default=0.001)
    p.add_argument("--prior-b-lambda", type=float, default=0.001)

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
    p.add_argument("--s-beta-init",  type=float, default=1e-3)
    p.add_argument("--s-gamma-init", type=float, default=1e-2)
    p.add_argument("--gamma0-init", type=str, default=None)

    args = p.parse_args()
    np.random.seed(args.seed)

    start_date = _parse_date(args.start_date)
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
    if args.prior_m0_gamma is not None and args.prior_m0_gamma.strip() != "":
        pri_gamma_vec = [float(z) for z in args.prior_m0_gamma.split(",")]
    else:
        pri_gamma_vec = [0.0] * (args.period - 1)

    priors = Priors(
        a_sigma=args.prior_a_sigma, b_sigma=args.prior_b_sigma,
        m0_alpha=args.prior_m0_alpha, P0_alpha=args.prior_P0_alpha,
        m0_beta=args.prior_m0_beta,   P0_beta=args.prior_P0_beta,
        m0_gamma=pri_gamma_vec,
        P0_gamma=args.prior_P0_gamma,
        a_lambda=args.prior_a_lambda, b_lambda=args.prior_b_lambda,
    )

    cfg = SamplerConfig(
        n_iter=int(args.n_iter), burn=int(args.burn), thin=int(args.thin),
        random_seed=int(args.seed), progress=bool(args.progress),
        progress_every=int(args.progress_every),
        slice_w=float(args.slice_w), slice_m=int(args.slice_m),
    )

    # Sampler
    gamma0_init = (
        [float(z) for z in (args.gamma0_init or "").split(",")] if args.gamma0_init else None
    )

    sampler = DLMGibbsConjugate(
        y=y, period=args.period,
        level_mode="dynamic", trend_mode="dynamic", seasonal_mode="dynamic",
        sigma2_init=args.sigma_init ** 2,
        s_alpha_init=args.s_alpha_init, s_beta_init=args.s_beta_init, s_gamma_init=args.s_gamma_init,
        alpha0=args.m0_level, beta0=args.m0_trend,
        gamma0=gamma0_init, 
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
        print("Hierarchical Bayesian lasso prior for process SDs:")
        print("  s_k | τ_k, σ² ~ N(0, σ² τ_k)")
        print("  τ_k | λ²     ~ Exp(λ²/2)")
        print(f"  λ²  ~ Gamma(a_λ={priors.a_lambda:.3g}, b_λ={priors.b_lambda:.3g})\n")

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
            "lasso_priors": {
                "a_lambda": priors.a_lambda,
                "b_lambda": priors.b_lambda,
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
        print(f"mean λ²      = {np.mean(post['lambda2']):.4g}")
        print(f"mean τ_α     = {np.mean(post['tau_alpha']):.4g}")
        print(f"mean τ_β     = {np.mean(post['tau_beta']):.4g}")
        print(f"mean τ_γ     = {np.mean(post['tau_gamma']):.4g}")

    if args.plot:
        mu_hat = post["mu"].mean(axis=0)
        plt.figure(figsize=(10, 4))
        plt.plot(dates_T, y, label="y_t", lw=1)
        plt.plot(dates_T, mu_T, "--", label="μ_t (truth)")
        plt.plot(dates_T, mu_hat, "-.", label="μ̂_t (post mean)")
        plt.title("DLM (Bayesian lasso prior on process SDs, dynamic/dynamic/dynamic)")
        plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()
