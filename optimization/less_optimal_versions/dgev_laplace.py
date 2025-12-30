from __future__ import annotations

import json, math, os, warnings, time, sys
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
    return np.linalg.pinv(M) @ B @ 1.0


def _rand_inv_gauss(mu: float, lam: float, rng: np.random.Generator) -> float:
    """
    Draw from an inverse-Gaussian IG(mu, lam) using the Michael–Schucany–Haas method.
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
# GEV log-likelihood, score & Hessian wrt μ
# =============================================================================

def gev_logpdf(y: float, mu: float, sigma: float, xi: float) -> float:
    """log f(y | mu, sigma>0, xi) under GEV parameterization (mu, sigma, xi)."""
    if sigma <= 0.0 or np.isnan(mu):
        return -np.inf
    z = (y - mu) / sigma
    u = 1.0 + xi * z
    if u <= 0.0:
        return -np.inf
    if abs(xi) < 1e-8:  # Gumbel limit
        return -np.log(sigma) - z - math.exp(-z)
    return -math.log(sigma) - (1.0 + 1.0 / xi) * math.log(u) - u ** (-1.0 / xi)


def gev_loglike_sum(y: np.ndarray, mu_vec: np.ndarray, sigma: float, xi: float) -> float:
    """Sum_t log f(y_t | mu_t, sigma, xi)."""
    if sigma <= 0.0 or np.any(np.isnan(mu_vec)):
        return -np.inf
    z = (y - mu_vec) / sigma
    u = 1.0 + xi * z
    if np.any(u <= 0.0):
        return -np.inf
    if abs(xi) < 1e-8:
        return float(np.sum(-np.log(sigma) - z - np.exp(-z)))
    return float(np.sum(-np.log(sigma) - (1.0 + 1.0 / xi) * np.log(u) - u ** (-1.0 / xi)))


def gev_score_hess_mu(y: float, mu: float, sigma: float, xi: float) -> Tuple[float, float]:
    """
    First and second derivative of log f(y | mu, sigma, xi) w.r.t. μ.
    Returns (g, h) where g = dℓ/dμ, h = d²ℓ/dμ².
    """
    z = (y - mu) / sigma
    if abs(xi) < 1e-8:
        # Gumbel limit
        e = math.exp(-z)
        g = (1.0 - e) / sigma
        h = -e / (sigma**2)
        return g, h

    u = 1.0 + xi * z
    if u <= 0.0:
        # Outside support: fall back to weak curvature
        return 0.0, -1e-8

    # ℓ = -log σ - (1 + 1/ξ) log u - u^{-1/ξ}
    # g = (1/σ)[ (ξ + 1)/u - u^{-1/ξ - 1} ]
    g = ((xi + 1.0) / u - u**(-1.0 / xi - 1.0)) / sigma

    # h = (1+ξ)/(σ²) [ ξ/u² - u^{-(1+2ξ)/ξ} ]
    h = (1.0 + xi) * (xi / (u * u) - u**(-(1.0 + 2.0 * xi) / xi)) / (sigma**2)
    return g, h


# =============================================================================
# Priors & Config (hierarchical Bayesian lasso prior on process SDs)
# =============================================================================

@dataclass
class Priors:
    # Observation parameters:
    # σ² ~ Inv-Gamma(a_sigma, b_sigma)
    a_sigma: float = 2.0
    b_sigma: float = 2.0

    # ξ ~ Uniform[xi_lower, xi_upper]
    xi_lower: float = -0.5
    xi_upper: float = 0.5

    # Baseline priors (alpha0, beta0, gamma0)
    m0_alpha: float = 0.0
    P0_alpha: float = 10.0
    m0_beta: float = 0.0
    P0_beta: float = 10.0
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

    # (kept for compatibility / CLI, not used here)
    slice_w: float = 1.0
    slice_m: int = 20
    
# =============================================================================
# DGEV Approximate Gibbs with NCP + dummy seasonality + Bayesian lasso
# =============================================================================

class DGEVLaplaceNCP:
    """
    Structural DGEV with:
      • Dynamic/dynamic/dynamic mean (alpha, beta, seasonal dummies; newest-first).
      • Non-centred parametrisation of latent states:
            z_t = [tilde_alpha_t, tilde_beta_t, A_t, tilde_g1_t, ..., tilde_g_{p-1,t}],
        with fixed transition G_tilde and unit process covariance Q_tilde.
      • Signed process SDs (s_alpha, s_beta, s_gamma) with hierarchical Bayesian lasso prior:
            s_k | τ_k, σ² ~ N(0, σ² τ_k),
            τ_k | λ² ~ Exp(λ² / 2),
            λ² ~ Gamma(a_lambda, b_lambda).
      • DGEV observation:
            Y_t | μ_t, σ, ξ ~ GEV(μ_t, σ, ξ),
            μ_t = alpha_t + g1_t + S[t,:] @ gamma0,
        where S is the static seasonal baseline design (sum-to-zero constraint).
      • Laplace approximation in μ_t:
            for each t, approximate log f(y_t | μ_t, σ, ξ) by a local Gaussian
            to construct pseudo-observations z_t with variances R_t.
      • FFBS in the non-centred state using the pseudo-observations,
        then regression update of (alpha0, beta0, gamma0, s_alpha, s_beta, s_gamma),
        random sign switches, hierarchical lasso updates, and Gibbs for σ², ξ.

      In the regression step we use a *centred time* covariate t_c = t - t̄,
      sample (α_c, β, γ0, s_α, s_β, s_γ), and transform back to
          alpha0 = α_c - t̄ β,  beta0 = β
      to reduce α0–β0 correlation.
    """

    # --------------------------- Construction --------------------------- #
    def __init__(
        self,
        y: np.ndarray,
        period: int,
        # initial values
        alpha0: float = 0.0,
        beta0: float = 0.0,
        gamma0: Optional[Sequence[float]] = None,  # len p-1 (NEWEST-FIRST)
        sigma_init: float = 1.0,
        xi_init: float = 0.0,
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

        # We enforce dynamic/dynamic/dynamic only (mirror DLMGibbsConjugate)
        self.level_mode = "dynamic"
        self.trend_mode = "dynamic"
        self.seasonal_mode = "dynamic"

        self.priors, self.cfg = priors, cfg
        if cfg.random_seed is not None:
            np.random.seed(cfg.random_seed)
            self._rng = np.random.default_rng(cfg.random_seed)
        else:
            self._rng = np.random.default_rng()

        # ---- time index and centre for regression centering ----
        self._t_idx = np.arange(1, self.T + 1, dtype=float)   # 1..T
        self._t_center = float(self._t_idx.mean())            # t̄

        # ------------------ CP layout (for storage / interpretation) ------------------ #
        layout: List[str] = ["alpha", "beta"]
        layout.extend([f"g{k}" for k in range(1, self.period)])
        self._layout = layout
        self.dim = len(layout)

        self.idx_alpha = layout.index("alpha")
        self.idx_beta = layout.index("beta")
        self.idx_g_start = layout.index("g1")
        self.idx_g_end = self.idx_g_start + (self.period - 2)

        self.K_gamma = self.period - 1  # number of seasonal baseline params

        # Observation parameters
        self.logsigma = float(math.log(max(sigma_init, 1e-12)))
        self.sigma = float(math.exp(self.logsigma))
        self.xi = float(xi_init)
        if not (priors.xi_lower <= self.xi <= priors.xi_upper):
            raise ValueError(
                f"Initial xi={xi_init} must lie in [{priors.xi_lower},{priors.xi_upper}]"
            )

        # Process SDs (signed)
        self.s_alpha = float(s_alpha_init)
        self.s_beta = float(s_beta_init)
        self.s_gamma = float(s_gamma_init)

        # Lasso local scales τ_k and global shrinkage λ²
        self.tau_alpha = 1.0
        self.tau_beta = 1.0
        self.tau_gamma = 1.0
        self.lambda2 = 1.0

        # Static baselines (original parametrisation)
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        if gamma0 is None:
            self.gamma0 = np.zeros(self.K_gamma, float)
        else:
            g = np.asarray(gamma0, float)
            if g.size != self.K_gamma:
                raise ValueError("gamma0 must have length p-1 (NEWEST-FIRST)")
            self.gamma0 = g

        # Seasonal baseline design
        self._season_design = self._build_season_design()

        # ------------------ NCP layout ------------------ #
        if self.dim > 0:
            self.dim_ncp = self.dim + 1  # extra A state
            self.idx_tilde_alpha = 0
            self.idx_tilde_beta = 1
            self.idx_A = 2
            self.idx_tilde_g_start = 3
            self.idx_tilde_g_end = self.idx_tilde_g_start + (self.period - 2)
        else:
            self.dim_ncp = 0

        # Latent NCP path z (tilde states)
        self.z = np.zeros((self.T + 1, self.dim_ncp), float)
        # Centred CP path x (for storage and μ_t)
        self.x = np.zeros((self.T + 1, self.dim), float)

        # Fixed NCP transition and unit covariance
        if self.dim_ncp > 0:
            self._G_tilde = self._build_G_tilde()
            self._Q_tilde = self._build_Q_tilde()

        # Map initial NCP path to CP states
        if self.dim > 0:
            self._refresh_cp_from_ncp()

        # Storage
        self.keep: Dict[str, np.ndarray] = {}

        # Truth overlays (optional)
        self.true_sigma: Optional[float] = None
        self.true_xi: Optional[float] = None
        self.true_Q: Optional[np.ndarray] = None
        self.true_mu_t: Optional[np.ndarray] = None
        self.true_alpha_t: Optional[np.ndarray] = None
        self.true_beta_t: Optional[np.ndarray] = None
        self.true_gamma_t: Optional[np.ndarray] = None

        # Last log-likelihood
        self.last_loglike: float = float("nan")

    # --------------------- Truth overlays (optional) --------------------- #
    def set_truth(
        self,
        sigma: Optional[float] = None,
        xi: Optional[float] = None,
        Q: Optional[Tuple[float, float, float]] = None,
        m0_level: Optional[float] = None,
        m0_trend: Optional[float] = None,
        m0_season: Optional[Sequence[float]] = None,
    ) -> None:
        self.true_sigma = None if sigma is None else float(sigma)
        self.true_xi = None if xi is None else float(xi)
        self.true_Q = None if Q is None else np.asarray(Q, float)
        self.true_m0_level = None if m0_level is None else float(m0_level)
        self.true_m0_trend = None if m0_trend is None else float(m0_trend)
        self.true_m0_season = None if m0_season is None else np.asarray(m0_season, float)

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
            else:
                S[t, :] = -1.0
        return S

    # ----------------------------- NCP model matrices ----------------------------- #
    def _build_G_tilde(self) -> np.ndarray:
        """Transition matrix for non-centred state z_t."""
        if self.dim_ncp == 0:
            return np.zeros((0, 0))
        d = self.dim_ncp
        G = np.zeros((d, d))

        ia = self.idx_tilde_alpha
        ib = self.idx_tilde_beta
        iA = self.idx_A
        gs = self.idx_tilde_g_start
        ge = self.idx_tilde_g_end
        K = ge - gs + 1

        # tilde_alpha RW
        G[ia, ia] = 1.0
        # tilde_beta RW
        G[ib, ib] = 1.0
        # A_t = A_{t-1} + tilde_beta_{t-1}
        G[iA, iA] = 1.0
        G[iA, ib] = 1.0

        # Seasonal dummy rotation S for tilde_gamma
        if K > 0:
            R = np.zeros((K, K))
            R[0, :] = -1.0
            if K > 1:
                R[1:, :-1] = np.eye(K - 1)
            G[gs:ge + 1, gs:ge + 1] = R

        return G

    def _build_Q_tilde(self) -> np.ndarray:
        """Unit process covariance for NCP."""
        if self.dim_ncp == 0:
            return np.zeros((0, 0))
        d = self.dim_ncp
        Q = np.zeros((d, d))

        ia = self.idx_tilde_alpha
        ib = self.idx_tilde_beta
        gs = self.idx_tilde_g_start
        ge = self.idx_tilde_g_end

        Q[ia, ia] = 1.0
        Q[ib, ib] = 1.0
        for k in range(gs, ge + 1):
            Q[k, k] = 1.0

        return Q

    # ------------------ Mapping NCP → CP ------------------ #
    def _refresh_cp_from_ncp(self) -> None:
        """
        Compute centred states x_t from NCP z_t and current parameters.

        alpha_t = alpha0 + t beta0 + s_alpha tilde_alpha_t + s_beta A_t
        beta_t  = beta0 + s_beta tilde_beta_t
        gamma_t = s_gamma tilde_gamma_t  (dynamic seasonal states)
        """
        if self.dim == 0 or self.dim_ncp == 0:
            self.x = np.zeros((self.T + 1, self.dim), float)
            return

        z = self.z
        T = self.T
        t_idx = np.arange(T + 1, dtype=float)

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

        tilde_alpha = z[:, ia]
        tilde_beta = z[:, ib]
        A_t = z[:, iA]
        tilde_gamma = z[:, gs_t:ge_t + 1]

        beta_cp = beta0 + s_beta * tilde_beta
        alpha_cp = alpha0 + t_idx * beta0 + s_alpha * tilde_alpha + s_beta * A_t
        gamma_cp = s_gamma * tilde_gamma

        x = np.zeros((T + 1, self.dim), float)
        x[:, self.idx_alpha] = alpha_cp
        x[:, self.idx_beta] = beta_cp
        x[:, self.idx_g_start : self.idx_g_end + 1] = gamma_cp

        self.x = x

    # ------------------ μ and observation helpers ------------------ #
    def _H_dyn(self) -> np.ndarray:
        """Observation row for the *dynamic* part (alpha_t + g1_t)."""
        if self.dim == 0:
            return np.zeros((1, 0))
        h = np.zeros(self.dim, float)
        h[self.idx_alpha] = 1.0
        h[self.idx_g_start] = 1.0
        return h.reshape(1, -1)

    def _mu_vec_from_x(self) -> np.ndarray:
        """
        Compute μ_t from centred states x_t and static seasonal baseline gamma0.

        μ_t = alpha_t + g1_t (from x_t) + S[t,:] @ gamma0.
        """
        if self.dim == 0:
            return np.zeros(self.T, float)
        H = self._H_dyn()
        T = self.T
        mu_dyn = np.zeros(T, float)
        for t in range(1, T + 1):
            mu_dyn[t - 1] = float(H @ self.x[t])

        if self.period > 1 and self._season_design.size > 0:
            mu_base = self._season_design @ self.gamma0
        else:
            mu_base = 0.0

        return mu_dyn + mu_base

    # ------------------ Laplace pseudo-observations ------------------ #
    def _build_pseudo_obs(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build Laplace-based pseudo-observations for μ_t.

        For each t, approximate
          ℓ_t(μ) ≈ ℓ_t(μ_t) + g_t (μ-μ_t) + 0.5 h_t (μ-μ_t)^2
                 = const - 0.5 w_t (z_t - μ)^2
          where w_t = -h_t > 0 and z_t = μ_t - g_t/h_t.

        We then define a regression for the dynamic part:
          z_t = (alpha_t + g1_t) + S[t,:] @ gamma0 + ε_t,  ε_t ~ N(0, R_t),
        so:
          z_star_t := z_t - S[t,:] @ gamma0 = H_dyn x_t + ε_t,  R_t = 1 / w_t.
        """
        mu = self._mu_vec_from_x()
        z_t = np.zeros(self.T, float)
        R_t = np.zeros(self.T, float)

        sigma = self.sigma
        xi = self.xi
        S = self._season_design

        for t in range(self.T):
            g, h = gev_score_hess_mu(self.y[t], mu[t], sigma, xi)
            if not np.isfinite(g) or not np.isfinite(h) or h >= -1e-8:
                h = -1e-8
                g = 0.0
            w = -h
            z_mu = mu[t] - g / h
            R_t[t] = 1.0 / w
            # subtract static seasonal part to form pseudo-observation for dynamic part
            base_t = float(S[t] @ self.gamma0) if (S.size > 0) else 0.0
            z_t[t] = z_mu - base_t

        return z_t, R_t

    # ------------------------- FFBS in NCP using Laplace pseudo-obs ------------------------- #
        # ------------------------- FFBS in NCP using Laplace pseudo-obs ------------------------- #
    def _ffbs_ncp_laplace(self) -> np.ndarray:
        """
        FFBS on the non-centred state z_t using Laplace-Gaussian pseudo-observations.

        Pseudo-observations from _build_pseudo_obs() satisfy (approximately)
            z_star_t ≈ alpha_t + g1_t + ε_t,  ε_t ~ N(0, R_t),

        with
            alpha_t = alpha0 + t * beta0 + s_alpha * tilde_alpha_t + s_beta * A_t
            g1_t    = s_gamma * tilde_gamma1_t.

        Hence
            z_star_t - (alpha0 + t * beta0)
              ≈ s_alpha * tilde_alpha_t + s_beta * A_t + s_gamma * tilde_gamma1_t
              = H_tilde z_t + ε_t,

        where H_tilde selects the NCP blocks (tilde_alpha, A, tilde_gamma1).
        We run Kalman FFBS on this linear-Gaussian model.
        """
        if self.dim_ncp == 0:
            return self.z.copy()

        G = self._G_tilde
        Q = self._Q_tilde
        T = self.T
        dim = self.dim_ncp

        # H_tilde: maps z_t → s_alpha * tilde_alpha_t + s_beta * A_t + s_gamma * tilde_gamma1_t
        H_vec = np.zeros(dim)
        H_vec[self.idx_tilde_alpha] = self.s_alpha
        H_vec[self.idx_A] = self.s_beta
        H_vec[self.idx_tilde_g_start] = self.s_gamma  # tilde_gamma1
        H = H_vec.reshape(1, -1)

        # Laplace pseudo-obs for dynamic part: z_star_t ≈ alpha_t + g1_t
        z_star, R_t = self._build_pseudo_obs()

        # Subtract deterministic level + trend so that the NCP state only carries
        # the random parts (tilde_alpha, A, tilde_gamma1).
        # t_idx is 1..T
        t_idx = self._t_idx
        det_part = self.alpha0 + t_idx * self.beta0  # shape (T,)
        y_ncp = z_star - det_part                    # what the filter "observes"

        m = np.zeros((T + 1, dim))
        C = np.zeros((T + 1, dim, dim))
        a = np.zeros((T + 1, dim))
        Rm = np.zeros((T + 1, dim, dim))

        # Prior on z_0 is N(0, tiny I)
        m[0] = np.zeros(dim)
        C[0] = 1e-6 * np.eye(dim)

        # Forward pass
        for t in range(1, T + 1):
            R_obs = float(R_t[t - 1])
            R_obs = max(R_obs, 1e-10)

            # Prediction
            a[t] = G @ m[t - 1]
            Rm[t] = G @ C[t - 1] @ G.T + Q
            Rm[t] = 0.5 * (Rm[t] + Rm[t].T) + 1e-12 * np.eye(dim)

            # One-step forecast variance
            S_var = float(H @ Rm[t] @ H.T + R_obs)
            if S_var <= 0:
                S_var = float(H @ (Rm[t] + 1e-10 * np.eye(dim)) @ H.T + R_obs)

            # Kalman gain
            K = (Rm[t] @ H.T) / S_var

            # Innovation uses y_ncp (z_star minus deterministic part)
            v = float(y_ncp[t - 1] - H @ a[t])

            # Update
            m[t] = a[t] + (K.flatten() * v)
            C[t] = Rm[t] - K @ (H @ Rm[t])
            C[t] = 0.5 * (C[t] + C[t].T) + 1e-12 * np.eye(dim)

        # Backward sampling
        z = np.zeros_like(self.z)
        z[T] = np.random.multivariate_normal(m[T], C[T])
        for t in range(T - 1, -1, -1):
            J = C[t] @ G.T
            J = J @ _spd_solve(Rm[t + 1], np.eye(dim))
            mean = m[t] + J @ (z[t + 1] - (G @ m[t]))
            cov = C[t] - J @ Rm[t + 1] @ J.T
            cov = 0.5 * (cov + cov.T)
            eigmin = float(np.linalg.eigvalsh(cov).min())
            if eigmin < 1e-12:
                cov += (1e-12 - eigmin) * np.eye(dim)
            z[t] = np.random.multivariate_normal(mean, cov)

        return z

    # ------------------ σ² and ξ updates ------------------ #
    def _log_prior_logsigma(self, logsigma: float) -> float:
        """
        Inv-Gamma prior on v = σ² with shape a,b (shape-rate):
          p(v) ∝ v^{-(a+1)} exp(-b/v)
        For ℓ = ln σ, v = exp(2ℓ) ⇒ log p(ℓ) = -2aℓ - b exp(-2ℓ) + const.
        """
        a = float(self.priors.a_sigma)
        b = float(self.priors.b_sigma)
        return -2.0 * a * logsigma - b * math.exp(-2.0 * logsigma)

    def update_logsigma(self) -> None:
        step = 0.05
        cur = self.logsigma
        prop = cur + self._rng.normal(0.0, step)
        sigma_cur, sigma_prop = float(math.exp(cur)), float(math.exp(prop))
        mu_vec = self._mu_vec_from_x()
        ll_old = gev_loglike_sum(self.y, mu_vec, sigma_cur, self.xi)
        ll_new = gev_loglike_sum(self.y, mu_vec, sigma_prop, self.xi)
        if ll_new == -np.inf:
            return
        lp_old = self._log_prior_logsigma(cur)
        lp_new = self._log_prior_logsigma(prop)
        logacc = (ll_new + lp_new) - (ll_old + lp_old)
        if np.log(self._rng.random()) < min(0.0, logacc):
            self.logsigma = prop
            self.sigma = sigma_prop

    def update_xi(self) -> None:
        """Random-walk MH for ξ with Uniform prior on [xi_lower, xi_upper]."""
        step = 0.05
        cur = self.xi
        prop = cur + self._rng.normal(0.0, step)

        lb, ub = float(self.priors.xi_lower), float(self.priors.xi_upper)
        if not (lb <= prop <= ub):
            return

        mu_vec = self._mu_vec_from_x()
        ll_old = gev_loglike_sum(self.y, mu_vec, self.sigma, cur)
        ll_new = gev_loglike_sum(self.y, mu_vec, self.sigma, prop)
        if ll_new == -np.inf:
            return
        logacc = ll_new - ll_old
        if np.log(self._rng.random()) < min(0.0, logacc):
            self.xi = prop

    # --------------- Joint regression update for (alpha0, beta0, gamma0, s_*) --------------- #
    def update_beta_fs(self) -> None:
        """
        Gaussian regression update (FS style) for
            (alpha0, beta0, gamma0, s_alpha, s_beta, s_gamma)
        using Laplace pseudo-observations z_mu.

        Reparametrisation for better mixing:

            z_mu_t = α_c + β (t - t̄) + S[t,:] γ0
                     + s_alpha tilde_alpha_t + s_beta A_t + s_gamma tilde_gamma1_t
                     + ε_t,  ε_t ~ N(0, R_t),

        where t̄ is the mean of {1,…,T}. We sample θ_c = (α_c, β, γ0, s_α, s_β, s_γ)
        and then transform back to the original parametrisation:

            alpha0 = α_c - t̄ β (time-centering)
            beta0  = β.
        """
        if self.dim_ncp == 0:
            return

        T = self.T
        ia = self.idx_tilde_alpha
        iA = self.idx_A
        gs = self.idx_tilde_g_start

        t_idx = self._t_idx
        t_center = self._t_center
        t_c = t_idx - t_center
        S = self._season_design

        # Full pseudo-observations z_mu_t
        mu = self._mu_vec_from_x()
        sigma = self.sigma
        xi = self.xi
        z_mu = np.zeros(T, float)
        R_t = np.zeros(T, float)

        for t in range(T):
            g, h = gev_score_hess_mu(self.y[t], mu[t], sigma, xi)
            if not np.isfinite(g) or not np.isfinite(h) or h >= -1e-8:
                h = -1e-8
                g = 0.0
            w = -h
            z_mu[t] = mu[t] - g / h
            R_t[t] = 1.0 / w

        # NCP regressors
        tilde_alpha = self.z[1:, ia]
        A_t = self.z[1:, iA]
        tilde_gamma1 = self.z[1:, gs]

        # Design matrices:
        # static part for α_c, β, γ0 (with centred time)
        if self.K_gamma > 0:
            X_static = np.column_stack([np.ones(T, float), t_c, S])
        else:
            X_static = np.column_stack([np.ones(T, float), t_c])

        # dynamic part for s_alpha, s_beta, s_gamma
        X_dyn = np.column_stack([tilde_alpha, A_t, tilde_gamma1])

        X = np.column_stack([X_static, X_dyn])  # shape (T, 2 + K_gamma + 3)
        d = X.shape[1]

        # Weighted regression: W = diag(1/R_t)
        w = 1.0 / np.maximum(R_t, 1e-12)
        sqrt_w = np.sqrt(w)
        Xw = X * sqrt_w[:, None]
        yw = z_mu * sqrt_w

        # Priors for θ_c = (α_c, β, γ0, s_α, s_β, s_γ)
        m_prior = np.zeros(d, float)

        # Map original priors on (alpha0, beta0) to the centred parametrisation:
        #   α_c = alpha0 + t̄ beta0
        # Use m0_alpha, m0_beta as prior means for (alpha0, beta0)
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

        s2_prior = np.zeros(d, float)
        eps = 1e-12

        # baselines: interpret P0_alpha, P0_beta as prior variances for α_c, β
        s2_prior[0] = self.priors.P0_alpha
        s2_prior[1] = self.priors.P0_beta
        if self.K_gamma > 0:
            s2_prior[2 : 2 + self.K_gamma] = self.priors.P0_gamma

        # Process SDs positions
        idx_s_alpha = 2 + self.K_gamma
        idx_s_beta = 3 + self.K_gamma
        idx_s_gamma = 4 + self.K_gamma

        # Lasso priors on s_k (pseudo-likelihood scale)
        sigma2_eff = 1.0
        s2_prior[idx_s_alpha] = max(sigma2_eff * self.tau_alpha, eps)
        s2_prior[idx_s_beta] = max(sigma2_eff * self.tau_beta, eps)
        s2_prior[idx_s_gamma] = max(sigma2_eff * self.tau_gamma, eps)

        s2_prior = np.maximum(s2_prior, eps)
        V_prior = np.diag(s2_prior)
        V_prior_inv = np.linalg.inv(V_prior)

        XtX = Xw.T @ Xw
        Xt_y = Xw.T @ yw

        prec_post = XtX + V_prior_inv
        cov_post = _spd_solve(prec_post, np.eye(d))
        mean_post = cov_post @ (Xt_y + V_prior_inv @ m_prior)

        theta_c = np.random.multivariate_normal(mean_post, cov_post)

        # Extract α_c, β and transform back
        alpha_c = float(theta_c[0])
        beta0 = float(theta_c[1])

        self.beta0 = beta0
        self.alpha0 = alpha_c - t_center * beta0

        if self.K_gamma > 0:
            self.gamma0 = theta_c[2 : 2 + self.K_gamma].copy()
        self.s_alpha = float(theta_c[idx_s_alpha])
        self.s_beta = float(theta_c[idx_s_beta])
        self.s_gamma = float(theta_c[idx_s_gamma])

    # --------------- Lasso local/global scale updates --------------- #
    def update_lasso_scales(self) -> None:
        """
        Update local τ_k and global λ² under:
            s_k | τ_k, σ² ~ N(0, σ² τ_k),
            τ_k | λ² ~ Exp(λ²/2),
            λ² ~ Gamma(a_lambda, b_lambda).

        Here we approximate σ² by current σ² from GEV, reusing the same formulas
        as in the Gaussian DLM case.
        """
        if self.dim_ncp == 0:
            return

        sigma2 = max(self.sigma**2, 1e-12)
        a_lam = self.priors.a_lambda
        b_lam = self.priors.b_lambda
        rng = self._rng
        eps = 1e-16

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

        K = 3
        shape = a_lam + K
        rate = b_lam + 0.5 * (self.tau_alpha + self.tau_beta + self.tau_gamma)
        self.lambda2 = np.random.gamma(shape=shape, scale=1.0 / max(rate, 1e-12))

    # --------------- Random sign switches --------------- #
    def random_sign_switches(self) -> None:
        """
        Flip signs of (s_alpha, s_beta, s_gamma) and corresponding NCP blocks.
        """
        if self.dim_ncp == 0:
            return

        # α
        if self._rng.random() < 0.5:
            self.s_alpha *= -1.0
            self.z[:, self.idx_tilde_alpha] *= -1.0

        # β (tilde_beta and A)
        if self._rng.random() < 0.5:
            self.s_beta *= -1.0
            self.z[:, self.idx_tilde_beta] *= -1.0
            self.z[:, self.idx_A] *= -1.0

        # γ (all seasonal NCP coords)
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
        parts.append(f"σ={self.sigma:.3f}")
        parts.append(f"ξ={self.xi:.3f}")
        parts.append(f"Qα={self.s_alpha**2:.4g}")
        parts.append(f"Qβ={self.s_beta**2:.4g}")
        parts.append(f"Qγ={self.s_gamma**2:.4g}")
        parts.append(f"α0={self.alpha0:.4g}")
        parts.append(f"β0={self.beta0:.4g}")
        parts.append(f"γ0={self._fmt_list(self.gamma0, 6, '.4g')}")
        parts.append(
            "τloc=[" +
            ", ".join(f"{x:.3g}" for x in [self.tau_alpha, self.tau_beta, self.tau_gamma]) +
            "]"
        )
        parts.append(f"λ²={self.lambda2:.4g}")
        parts.append(f"logL={self.last_loglike:.2f}")
        return " | ".join(parts)

    # --------------------------------- MCMC --------------------------------- #
    def run(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        save_iters = list(range(cfg.burn, cfg.n_iter, cfg.thin))
        n_kept, keep_idx = len(save_iters), 0

        self.keep = {
            "sigma": np.zeros(n_kept, float),
            "xi": np.zeros(n_kept, float),
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

            "tau_alpha": np.zeros(n_kept),
            "tau_beta": np.zeros(n_kept),
            "tau_gamma": np.zeros(n_kept),
            "lambda2": np.zeros(n_kept),

            "loglike": np.zeros(n_kept),
            "x": np.zeros((n_kept, self.T, self.dim)),
        }

        print_every = cfg.progress_every if cfg.progress_every > 0 else max(1, cfg.n_iter // 50) or 1

        for it in range(cfg.n_iter):
            # 1) FFBS in NCP using Laplace pseudo-obs
            if self.dim_ncp > 0:
                self.z = self._ffbs_ncp_laplace()
                self._refresh_cp_from_ncp()

            # Compute current exact GEV loglik
            mu_vec_now = self._mu_vec_from_x()
            self.last_loglike = gev_loglike_sum(self.y, mu_vec_now, self.sigma, self.xi)

            # 2) Regression update of (alpha0, beta0, gamma0, s_alpha, s_beta, s_gamma)
            if self.dim_ncp > 0:
                self.update_beta_fs()
                self._refresh_cp_from_ncp()

            # 3) Random sign switches
            if self.dim_ncp > 0:
                self.random_sign_switches()

            # 4) Hierarchical lasso scales
            if self.dim_ncp > 0:
                self.update_lasso_scales()

            # 5) Observation parameters (σ, ξ)
            self.update_logsigma()
            self.update_xi()

            # progress
            if cfg.progress and ((it + 1) % print_every == 0 or it == cfg.n_iter - 1):
                print(self._progress_line(it))

            # save
            if it in save_iters:
                mu = self._mu_vec_from_x()
                self.keep["mu"][keep_idx, :] = mu
                self.keep["sigma"][keep_idx] = self.sigma
                self.keep["xi"][keep_idx] = self.xi
                self.keep["loglike"][keep_idx] = self.last_loglike

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

                # Lasso
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
        if self.true_xi is not None:
            arrays["true_xi"] = float(self.true_xi)
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
    import argparse
    from datetime import datetime
    import matplotlib.pyplot as plt

    # make project root importable
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(base_dir)

    # seasonal extremal mean simulator with dummies
    from simulator.extremal_time_series import Extremal_Time_Series

    # ------------------------------------------------------------------ #
    # Small CLI helpers
    # ------------------------------------------------------------------ #
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

    def _csv_floats_or_none(s: Optional[str]) -> Optional[List[float]]:
        """Parse 'a,b,c' -> [a,b,c] or return None if empty/None."""
        if s is None:
            return None
        s_str = str(s).strip()
        if not s_str:
            return None
        return [float(z) for z in s_str.split(",")]

    p = argparse.ArgumentParser(
        description=(
            "Non-centred structural DGEV with seasonal dummies (newest-first) "
            "and a Bayesian lasso prior on process SDs. "
            "Latent state updated by FFBS on Laplace-Gaussian pseudo-observations; "
            "σ² via Inv-Gamma prior on σ² (through log σ), ξ via Uniform RW-MH."
        )
    )

    # Simulation
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--period", type=int, default=4)
    p.add_argument("--start-date", type=str, default="2000-01-01")

    p.add_argument("--sigma", type=float, default=3.0)
    p.add_argument("--xi", type=float, default=-0.1)

    p.add_argument("--level-mode", type=str, default="dynamic")
    p.add_argument("--trend-mode", type=str, default="dynamic")
    p.add_argument("--seasonal-mode", type=str, default="dynamic")

    p.add_argument("--q-level", type=float, default=0.001)
    p.add_argument("--q-trend", type=float, default=0.00002)
    p.add_argument("--q-season", type=float, default=0.00005)
    p.add_argument("--m0-level", type=float, default=3.0)
    p.add_argument("--v0-level", type=float, default=0.05)
    p.add_argument("--m0-trend", type=float, default=0.1)
    p.add_argument("--v0-trend", type=float, default=0.05)
    p.add_argument("--m0-season", type=str, default=None)
    p.add_argument("--v0-season", type=str, default=None)

    # Priors
    p.add_argument("--prior-a-sigma", type=float, default=2.0)
    p.add_argument("--prior-b-sigma", type=float, default=2.0)
    p.add_argument("--prior-xi-lower", type=float, default=-0.5)
    p.add_argument("--prior-xi-upper", type=float, default=0.5)

    p.add_argument("--prior-m0-alpha", type=float, default=0.0)
    p.add_argument("--prior-P0-alpha", type=float, default=10.0)
    p.add_argument("--prior-m0-beta", type=float, default=0.0)
    p.add_argument("--prior-P0-beta", type=float, default=10.0)
    p.add_argument("--prior-m0-gamma", type=str, default=None)
    p.add_argument("--prior-P0-gamma", type=float, default=5.0)

    # Lasso hyperparameters
    p.add_argument("--prior-a-lambda", type=float, default=0.001)
    p.add_argument("--prior-b-lambda", type=float, default=0.001)

    # Sampler configuration
    p.add_argument("--n-iter", type=int, default=8000)
    p.add_argument("--burn", type=int, default=4000)
    p.add_argument("--thin", type=int, default=2)
    p.add_argument("--seed", type=int, default=40)
    p.add_argument("--progress", default=True)
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--slice-w", type=float, default=1.0)
    p.add_argument("--slice-m", type=int, default=20)
    p.add_argument("--out-dir", type=str, default="results/simulations/DGEV_NCP_LASSO")
    p.add_argument("--plot", default=True)
    p.add_argument("--print-summary", default=True)

    # Initial inference values
    p.add_argument("--sigma-init", type=float, default=1.0)
    p.add_argument("--xi-init", type=float, default=-0.05)
    p.add_argument("--s-alpha-init", type=float, default=1e-2)
    p.add_argument("--s-beta-init", type=float, default=1e-3)
    p.add_argument("--s-gamma-init", type=float, default=1e-2)
    p.add_argument("--gamma0-init", type=str, default=None)

    args = p.parse_args()
    np.random.seed(args.seed)

    # --- simulate data with seasonal dummies --------------------------------
    start_date = _parse_date(args.start_date)

    # Parse seasonal hyperparameters (comma-separated) or use defaults
    args.m0_season = _csv_floats_or_none(args.m0_season)
    args.v0_season = _csv_floats_or_none(args.v0_season)

    if args.m0_season is None:
        # e.g. for period=4: [2,2,2] ⇒ last season is -sum = -6
        args.m0_season = [2.0] * (args.period - 1)
    if args.v0_season is None:
        args.v0_season = [0.01] * (args.period - 1)

    ts = Extremal_Time_Series(
        parameters=(args.sigma, args.xi),
        level_mode=args.level_mode,
        trend_mode=args.trend_mode,
        seasonal_mode=args.seasonal_mode,
        period=args.period,
        q_level=(args.q_level if args.level_mode == "dynamic" else 0.0),
        q_trend=(args.q_trend if args.trend_mode == "dynamic" else 0.0),
        q_season=(args.q_season if args.seasonal_mode == "dynamic" else 0.0),
        m0_level=args.m0_level,
        v0_level=args.v0_level,
        m0_trend=(args.m0_trend if args.trend_mode != "none" else 0.0),
        v0_trend=args.v0_trend,
        m0_season=(args.m0_season if args.seasonal_mode != "none" else None),
        v0_season=(args.v0_season if args.seasonal_mode != "none" else None),
        start_date=start_date,
    )

    y = []
    for _ in range(args.T):
        ts.move()
        y.append(ts.measure())
    y = np.asarray(y, float)

    truths = ts.get_truth_paths(as_numpy=False)
    mu_T    = np.asarray(truths["mu"][1:1 + args.T], float)
    alpha_T = (np.asarray(truths["alpha"][1:1 + args.T], float)
               if args.level_mode == "dynamic" else None)
    beta_T  = (np.asarray(truths["beta"][1:1 + args.T], float)
               if args.trend_mode == "dynamic" else None)
    gamma_T = (np.asarray(truths["gamma"][1:1 + args.T], float)
               if args.seasonal_mode == "dynamic" else None)
    dates_T = truths.get("index", np.arange(args.T))

    if args.print_summary:
        print(f"\nSimulated {args.T} DGEV observations (σ={args.sigma}, ξ={args.xi}) "
              f"with modes {args.level_mode}/{args.trend_mode}/{args.seasonal_mode}.\n")
        print("Hierarchical Bayesian lasso prior for process SDs:")
        print("  s_k | τ_k, σ² ~ N(0, σ² τ_k)")
        print("  τ_k | λ²     ~ Exp(λ²/2)")
        print(f"  λ²  ~ Gamma(a_λ={args.prior_a_lambda:.3g}, b_λ={args.prior_b_lambda:.3g})\n")

    # Priors
    if args.prior_m0_gamma is not None and args.prior_m0_gamma.strip() != "":
        pri_gamma_vec = [float(z) for z in args.prior_m0_gamma.split(",")]
    else:
        pri_gamma_vec = [0.0] * (args.period - 1)

    priors = Priors(
        a_sigma=args.prior_a_sigma,
        b_sigma=args.prior_b_sigma,
        xi_lower=args.prior_xi_lower,
        xi_upper=args.prior_xi_upper,
        m0_alpha=args.prior_m0_alpha,
        P0_alpha=args.prior_P0_alpha,
        m0_beta=args.prior_m0_beta,
        P0_beta=args.prior_P0_beta,
        m0_gamma=pri_gamma_vec,
        P0_gamma=args.prior_P0_gamma,
        a_lambda=args.prior_a_lambda,
        b_lambda=args.prior_b_lambda,
    )

    cfg = SamplerConfig(
        n_iter=int(args.n_iter),
        burn=int(args.burn),
        thin=int(args.thin),
        random_seed=int(args.seed),
        progress=bool(args.progress),
        progress_every=int(args.progress_every),
        slice_w=float(args.slice_w),
        slice_m=int(args.slice_m),
    )

    # Initial seasonal baseline for inference
    if args.gamma0_init is not None and args.gamma0_init.strip() != "":
        gamma0_init = [float(z) for z in args.gamma0_init.split(",")]
    else:
        gamma0_init = None

    sampler = DGEVLaplaceNCP(
        y=y,
        period=args.period,
        alpha0=args.m0_level,
        beta0=args.m0_trend,
        gamma0=gamma0_init,
        sigma_init=args.sigma_init,
        xi_init=args.xi_init,
        s_alpha_init=args.s_alpha_init,
        s_beta_init=args.s_beta_init,
        s_gamma_init=args.s_gamma_init,
        priors=priors,
        cfg=cfg,
    )

    sampler.set_truth(
        sigma=args.sigma,
        xi=args.xi,
        Q=(args.q_level, args.q_trend, args.q_season),
        m0_level=args.m0_level,
        m0_trend=args.m0_trend,
        m0_season=args.m0_season,
    )
    sampler.set_truth_paths(mu=mu_T, alpha=alpha_T, beta=beta_T, gamma=gamma_T)

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
        print(f"ξ = {np.mean(post['xi']):.4f}")
        for k in ["alpha", "beta", "gamma"]:
            key = f"Q_{k}"
            if key in post:
                mQ = np.mean(post[key])
                print(f"Q_{k} = {mQ:.4g} (√Q ≈ {math.sqrt(max(mQ, 0.0)):.4g})")
        print(f"mean λ²      = {np.mean(post['lambda2']):.4g}")
        print(f"mean τ_α     = {np.mean(post['tau_alpha']):.4g}")
        print(f"mean τ_β     = {np.mean(post['tau_beta']):.4g}")
        print(f"mean τ_γ     = {np.mean(post['tau_gamma']):.4g}")

    if args.plot:
        mu_hat = post["mu"].mean(axis=0)
        plt.figure(figsize=(10, 4))
        plt.plot(dates_T, y, label="y_t", lw=1)
        if sampler.true_mu_t is not None:
            plt.plot(dates_T, mu_T, "--", label="μ_t (truth)")
        plt.plot(dates_T, mu_hat, "-.", label="μ̂_t (post mean)")
        plt.title("DGEV (Laplace NCP, Bayesian lasso on process SDs)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
