# optimization/dlm_lasso.py
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import numpy as np
try:
    from .ffbs import ffbs_dlm_ncp  # type: ignore
    from .utils import rand_invgauss, spd_solve, symmetrize  # type: ignore
except ImportError:
    from ffbs import ffbs_dlm_ncp  # type: ignore
    from utils import rand_invgauss, spd_solve, symmetrize  # type: ignore

# =============================================================================
# Priors & Config
# =============================================================================
@dataclass(slots=True)
class Priors:
    # Observation variance σ²: precision τ = 1/σ² ~ Gamma(a_sigma, b_sigma) (shape–rate)
    a_sigma: float = 2.0
    b_sigma: float = 1.0

    # Initial-state priors for (alpha0, beta0) (NOT for centred-time intercept)
    m0_alpha: float = 0.0
    P0_alpha: float = 10.0
    m0_beta: float = 0.0
    P0_beta: float = 10.0

    # Seasonal baseline prior for gamma0 (length p-1 newest-first)
    m0_gamma: Optional[Sequence[float]] = None
    P0_gamma: float = 5.0

    # Hierarchical Bayesian lasso hyperparameters for process SDs (signed s_k)
    a_lambda: float = 1.0
    b_lambda: float = 1.0


@dataclass(slots=True)
class SamplerConfig:
    n_iter: int = 5000
    burn: int = 1000
    thin: int = 2
    random_seed: Optional[int] = 40
    progress: bool = True
    progress_every: int = 0  # 0 => ~2% of n_iter


# =============================================================================
# Sampler
# =============================================================================
class DLMGibbsConjugate:
    """
    Gaussian structural DLM (dynamic/dynamic/dynamic) with:
      - non-centred parametrisation (NCP) for states,
      - hierarchical Bayesian lasso prior on signed process SDs.

    Fixes applied:
      (1) Seasonal innovations: ONLY the first seasonal component gets noise (dummy seasonal).
      (2) Centred-time regression update: correct correlated Gaussian prior on (alpha_c, beta)
          induced by independent priors on (alpha0, beta0).
    """

    # --------------------------- Construction --------------------------- #
    def __init__(
        self,
        y: np.ndarray,
        period: int,
        *,
        alpha0: float = 0.0,
        beta0: float = 0.0,
        gamma0: Optional[Sequence[float]] = None,  # length p-1 newest-first
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

        if self.T < 1:
            raise ValueError("y must have length >= 1")
        if self.period < 2:
            raise ValueError("period must be >= 2")

        self.priors = priors
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.random_seed)

        # time indices
        self._t1 = np.arange(1, self.T + 1, dtype=float)  # 1..T
        self._tbar = float(self._t1.mean())
        self._t0 = np.arange(0, self.T + 1, dtype=float)  # 0..T

        # CP layout (dynamic-only states)
        self._layout: List[str] = ["alpha", "beta"] + [f"g{k}" for k in range(1, self.period)]
        self.dim = len(self._layout)

        self.idx_alpha = 0
        self.idx_beta = 1
        self.idx_g_start = 2
        self.idx_g_end = self.idx_g_start + (self.period - 2)  # inclusive
        self.K_gamma = self.period - 1  # baseline gamma0 length

        # parameters
        self.sigma2 = float(sigma2_init)

        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)

        if gamma0 is None:
            self.gamma0 = np.zeros(self.K_gamma, float)
        else:
            g = np.asarray(gamma0, float)
            if g.size != self.K_gamma:
                raise ValueError("gamma0 must have length p-1 (newest-first)")
            self.gamma0 = g.copy()

        # signed process SDs
        self.s_alpha = float(s_alpha_init)
        self.s_beta = float(s_beta_init)
        self.s_gamma = float(s_gamma_init)

        # lasso scales
        self.tau_alpha = 1.0
        self.tau_beta = 1.0
        self.tau_gamma = 1.0
        self.lambda2 = 1.0

        # seasonal baseline design S[t,:] @ gamma0
        self._S = self._build_season_design()

        # NCP layout: [tilde_alpha, tilde_beta, A, tilde_g1..tilde_g_{p-1}]
        self.dim_ncp = self.dim + 1
        self.idx_tilde_alpha = 0
        self.idx_tilde_beta = 1
        self.idx_A = 2
        self.idx_tilde_g_start = 3
        self.idx_tilde_g_end = self.idx_tilde_g_start + (self.period - 2)  # inclusive

        self.z = np.zeros((self.T + 1, self.dim_ncp), float)  # NCP path
        self.x = np.zeros((self.T + 1, self.dim), float)      # CP (dynamic states only)

        self._G_tilde = self._build_G_tilde()
        self._Q_tilde = self._build_Q_tilde()

        self._refresh_cp_from_ncp()

        # storage
        self.keep: Dict[str, np.ndarray] = {}

        # optional truth overlays (for sim diagnostics)
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

    # --------------------- Truth overlays (optional) --------------------- #
    def set_truth(
        self,
        *,
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
        *,
        mu: Optional[np.ndarray] = None,
        alpha: Optional[np.ndarray] = None,
        beta: Optional[np.ndarray] = None,
        gamma: Optional[np.ndarray] = None,
    ) -> None:
        self.true_mu_t = None if mu is None else np.asarray(mu, float)
        self.true_alpha_t = None if alpha is None else np.asarray(alpha, float)
        self.true_beta_t = None if beta is None else np.asarray(beta, float)
        self.true_gamma_t = None if gamma is None else np.asarray(gamma, float)

    # ----------------------------- Design & system matrices ----------------------------- #
    def _build_season_design(self) -> np.ndarray:
        """
        Static seasonal design S[t,:] mapping gamma0 (length p-1) to seasonal baseline.
        Season p is represented as -sum_{j=1}^{p-1} gamma_j (sum-to-zero).
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

    def _build_G_tilde(self) -> np.ndarray:
        """
        Transition for z_t = [tilde_alpha, tilde_beta, A, tilde_gamma...]

          tilde_alpha, tilde_beta are RW(1)
          A_t = A_{t-1} + tilde_beta_{t-1}
          seasonal block rotates with sum-to-zero.

        NOTE: We keep A deterministic (no innovation variance).
        """
        d = self.dim_ncp
        G = np.zeros((d, d), float)

        ia, ib, iA = self.idx_tilde_alpha, self.idx_tilde_beta, self.idx_A
        gs, ge = self.idx_tilde_g_start, self.idx_tilde_g_end
        K = ge - gs + 1  # = p-1

        # RW for tilde_alpha and tilde_beta
        G[ia, ia] = 1.0
        G[ib, ib] = 1.0

        # accumulator
        G[iA, iA] = 1.0
        G[iA, ib] = 1.0

        # seasonal rotation
        if K > 0:
            Srot = np.zeros((K, K), float)
            Srot[0, :] = -1.0
            if K > 1:
                Srot[1:, :-1] = np.eye(K - 1)
            G[gs : ge + 1, gs : ge + 1] = Srot

        return G

    def _build_Q_tilde(self) -> np.ndarray:
        """
        Unit innovations in NCP.

        Fix (1): dummy seasonal -> ONLY the first seasonal component gets noise.
          - tilde_alpha: unit noise
          - tilde_beta : unit noise
          - A          : deterministic (0)
          - seasonal   : only tilde_g1 has unit noise, others 0
        """
        d = self.dim_ncp
        Q = np.zeros((d, d), float)

        Q[self.idx_tilde_alpha, self.idx_tilde_alpha] = 1.0
        Q[self.idx_tilde_beta, self.idx_tilde_beta] = 1.0

        if self.idx_tilde_g_start <= self.idx_tilde_g_end:
            Q[self.idx_tilde_g_start, self.idx_tilde_g_start] = 1.0

        return Q

    # ----------------------------- Mapping NCP -> CP ----------------------------- #
    def _refresh_cp_from_ncp(self) -> None:
        """
        Compute CP dynamic states x_t from NCP z_t and parameters.

          beta_t  = beta0 + s_beta * tilde_beta_t
          alpha_t = alpha0 + t*beta0 + s_alpha*tilde_alpha_t + s_beta*A_t
          g_t     = s_gamma * tilde_gamma_t
        """
        z = self.z

        tilde_alpha = z[:, self.idx_tilde_alpha]
        tilde_beta = z[:, self.idx_tilde_beta]
        A_t = z[:, self.idx_A]
        tilde_gamma = z[:, self.idx_tilde_g_start : self.idx_tilde_g_end + 1]

        beta_cp = self.beta0 + self.s_beta * tilde_beta
        alpha_cp = self.alpha0 + self._t0 * self.beta0 + self.s_alpha * tilde_alpha + self.s_beta * A_t
        gamma_cp = self.s_gamma * tilde_gamma

        self.x[:, self.idx_alpha] = alpha_cp
        self.x[:, self.idx_beta] = beta_cp
        self.x[:, self.idx_g_start : self.idx_g_end + 1] = gamma_cp

    # ----------------------------- Mean / residuals ----------------------------- #
    def mu_vec(self) -> np.ndarray:
        """
        μ_t = (alpha_t + g1_t) + S[t,:] @ gamma0, returned for t=1..T as (T,).
        """
        mu_dyn = self.x[1:, self.idx_alpha] + self.x[1:, self.idx_g_start]
        mu_base = (self._S @ self.gamma0) if self._S.size else 0.0
        return mu_dyn + mu_base

    # ----------------------------- FFBS on NCP ----------------------------- #
    def ffbs_ncp(self) -> None:
        """
        Sample z_{0:T} from p(z | y, params) using NCP-aware FFBS.

          c_t = alpha0 + beta0*t + S[t,:]@gamma0
          y'_t = y_t - c_t
          y'_t = s_alpha*tilde_alpha_t + s_beta*A_t + s_gamma*tilde_g1_t + eps_t,
                 eps_t ~ N(0, sigma2)
        """
        self.z = ffbs_dlm_ncp(
            y=self.y,
            G_tilde=self._G_tilde,
            Q_tilde=self._Q_tilde,
            sigma2=float(self.sigma2),
            alpha0=float(self.alpha0),
            beta0=float(self.beta0),
            t1=self._t1,
            season_design=self._S if self._S.size else None,
            gamma0=self.gamma0 if self._S.size else None,
            s_alpha=float(self.s_alpha),
            s_beta=float(self.s_beta),
            s_gamma=float(self.s_gamma),
            idx_tilde_alpha=self.idx_tilde_alpha,
            idx_A=self.idx_A,
            idx_tilde_g_start=self.idx_tilde_g_start,
            m0=np.zeros(self.dim_ncp),
            C0=1e-6 * np.eye(self.dim_ncp),
            rng=self.rng,
            jitter=1e-12,
        )

    # ----------------------------- Gibbs updates ----------------------------- #
    def update_sigma2(self) -> None:
        """
        Update σ² via Gamma prior on precision τ = 1/σ² (shape-rate).

        Likelihood part uses residual SSR from y - mu_vec().

        Prior part for signed s_k is Gaussian given τ_k:
          s_k | τ_k, σ² ~ N(0, σ² τ_k)
        => contributes s_k^2 / τ_k to the τ-weighted sum-of-squares.
        """
        e = self.y - self.mu_vec()
        ss_resid = float(e @ e)

        ss_lasso = (
            (self.s_alpha * self.s_alpha) / max(self.tau_alpha, 1e-12)
            + (self.s_beta * self.s_beta) / max(self.tau_beta, 1e-12)
            + (self.s_gamma * self.s_gamma) / max(self.tau_gamma, 1e-12)
        )

        K = 3
        a_post = self.priors.a_sigma + 0.5 * (self.T + K)
        b_post = self.priors.b_sigma + 0.5 * (ss_resid + ss_lasso)

        tau = self.rng.gamma(shape=a_post, scale=1.0 / max(b_post, 1e-300))
        self.sigma2 = 1.0 / max(float(tau), 1e-300)

    def update_theta(self) -> None:
        """
        Joint Gaussian update for:
          θ = (alpha_c, beta, gamma0, s_alpha, s_beta, s_gamma)
        using centred time t_c = t - t̄:

          y_t = alpha_c + beta * t_c + S[t,:]gamma0
                + s_alpha*tilde_alpha_t + s_beta*A_t + s_gamma*tilde_g1_t + eps_t

        Fix (2): if we interpret Priors.{m0_alpha,P0_alpha} and Priors.{m0_beta,P0_beta}
        as independent priors on (alpha0, beta0), then the induced prior on (alpha_c, beta)
        is correlated because alpha_c = alpha0 + t̄ beta0.

        Back-transform:
          beta0  = beta
          alpha0 = alpha_c - t̄ beta
        """
        T = self.T
        t_c = self._t1 - self._tbar

        tilde_alpha = self.z[1:, self.idx_tilde_alpha]
        A_t = self.z[1:, self.idx_A]
        tilde_g1 = self.z[1:, self.idx_tilde_g_start]

        # design: static [1, t_c, S] and dynamic [tilde_alpha, A_t, tilde_g1]
        if self.K_gamma > 0:
            X_static = np.column_stack([np.ones(T), t_c, self._S])
            k0 = 2 + self.K_gamma
        else:
            X_static = np.column_stack([np.ones(T), t_c])
            k0 = 2

        X_dyn = np.column_stack([tilde_alpha, A_t, tilde_g1])
        X = np.column_stack([X_static, X_dyn])
        d = int(X.shape[1])

        eps = 1e-12

        # ------------------------------------------------------------------
        # Prior mean vector
        # ------------------------------------------------------------------
        m0 = np.zeros(d, float)

        # induced mean on (alpha_c, beta)
        m_a0 = float(self.priors.m0_alpha)
        v_a0 = max(float(self.priors.P0_alpha), eps)
        m_b0 = float(self.priors.m0_beta)
        v_b0 = max(float(self.priors.P0_beta), eps)

        m0[0] = m_a0 + self._tbar * m_b0  # E[alpha_c]
        m0[1] = m_b0                      # E[beta]

        # gamma0 mean
        if self.K_gamma > 0:
            if self.priors.m0_gamma is None:
                m_gamma = np.zeros(self.K_gamma, float)
            else:
                m_gamma = np.asarray(self.priors.m0_gamma, float)
                if m_gamma.size != self.K_gamma:
                    raise ValueError("Priors.m0_gamma must have length p-1")
            m0[2 : 2 + self.K_gamma] = m_gamma

        # s_* means are 0 already

        # ------------------------------------------------------------------
        # Prior precision matrix (block for (alpha_c,beta), diagonal for rest)
        # ------------------------------------------------------------------
        prior_prec = np.zeros((d, d), float)

        # induced covariance of (alpha_c, beta) from independent (alpha0, beta0)
        # alpha_c = alpha0 + tbar*beta0, beta = beta0
        V_ab = np.array(
            [
                [v_a0 + (self._tbar**2) * v_b0, self._tbar * v_b0],
                [self._tbar * v_b0,            v_b0],
            ],
            float,
        )
        V_ab = symmetrize(V_ab) + 1e-12 * np.eye(2)
        invV_ab = spd_solve(V_ab, np.eye(2), jitter=1e-12)
        invV_ab = symmetrize(invV_ab)
        prior_prec[0:2, 0:2] = invV_ab

        # gamma0 diagonal prior
        if self.K_gamma > 0:
            v_g = max(float(self.priors.P0_gamma), eps)
            prior_prec[2 : 2 + self.K_gamma, 2 : 2 + self.K_gamma] = (1.0 / v_g) * np.eye(self.K_gamma)

        # lasso priors: s_k | τ_k, σ² ~ N(0, σ² τ_k)
        sigma2 = max(float(self.sigma2), eps)
        prior_prec[k0 + 0, k0 + 0] = 1.0 / max(sigma2 * float(self.tau_alpha), eps)
        prior_prec[k0 + 1, k0 + 1] = 1.0 / max(sigma2 * float(self.tau_beta), eps)
        prior_prec[k0 + 2, k0 + 2] = 1.0 / max(sigma2 * float(self.tau_gamma), eps)

        # ------------------------------------------------------------------
        # Posterior Gaussian
        # ------------------------------------------------------------------
        XtX = X.T @ X
        Xty = X.T @ self.y

        prec = XtX / sigma2 + prior_prec
        prec = symmetrize(prec) + 1e-12 * np.eye(d)

        cov = spd_solve(prec, np.eye(d), jitter=1e-12)
        cov = symmetrize(cov)

        mean = cov @ (Xty / sigma2 + prior_prec @ m0)

        # numerical guard for sampling
        eigmin = float(np.linalg.eigvalsh(cov).min())
        if eigmin < 1e-12:
            cov = cov + (1e-12 - eigmin) * np.eye(d)

        theta = self.rng.multivariate_normal(mean, cov)

        alpha_c = float(theta[0])
        beta = float(theta[1])
        self.beta0 = beta
        self.alpha0 = alpha_c - self._tbar * beta

        if self.K_gamma > 0:
            self.gamma0 = theta[2 : 2 + self.K_gamma].copy()

        self.s_alpha = float(theta[k0 + 0])
        self.s_beta = float(theta[k0 + 1])
        self.s_gamma = float(theta[k0 + 2])

    def update_lasso_scales(self) -> None:
        """
        Park–Casella updates:

          τ_k | s_k, σ², λ² ~ IG(mu_k, λ²),
            mu_k = sqrt(λ² σ² / s_k²)

          λ² | τ            ~ Gamma(a_lambda + K, b_lambda + 0.5 * Σ τ_k)
        """
        sigma2 = max(self.sigma2, 1e-12)
        lam2 = max(self.lambda2, 1e-12)
        eps = 1e-16

        def _upd_tau(sk: float) -> float:
            s2 = sk * sk
            if s2 < eps:
                return 1.0
            mu = math.sqrt(lam2 * sigma2 / s2)
            return rand_invgauss(mu, lam2, self.rng)

        self.tau_alpha = _upd_tau(self.s_alpha)
        self.tau_beta = _upd_tau(self.s_beta)
        self.tau_gamma = _upd_tau(self.s_gamma)

        K = 3
        shape = self.priors.a_lambda + K
        rate = self.priors.b_lambda + 0.5 * (self.tau_alpha + self.tau_beta + self.tau_gamma)
        self.lambda2 = float(self.rng.gamma(shape=shape, scale=1.0 / max(rate, 1e-12)))

    def random_sign_switches(self) -> None:
        """
        Flip signs in symmetric directions leaving likelihood invariant:
          - α: (s_alpha, tilde_alpha)
          - β: (s_beta, tilde_beta, A)
          - γ: (s_gamma, all tilde_g)
        """
        if self.rng.random() < 0.5:
            self.s_alpha *= -1.0
            self.z[:, self.idx_tilde_alpha] *= -1.0

        if self.rng.random() < 0.5:
            self.s_beta *= -1.0
            self.z[:, self.idx_tilde_beta] *= -1.0
            self.z[:, self.idx_A] *= -1.0

        if self.K_gamma > 0 and self.rng.random() < 0.5:
            self.s_gamma *= -1.0
            self.z[:, self.idx_tilde_g_start : self.idx_tilde_g_end + 1] *= -1.0

    # ----------------------------- MCMC ----------------------------- #
    def run(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg

        save_iters = list(range(cfg.burn, cfg.n_iter, cfg.thin))
        save_set = set(save_iters)
        n_kept = len(save_iters)
        keep_idx = 0

        self.keep = {
            "sigma": np.zeros(n_kept, float),
            "mu": np.zeros((n_kept, self.T), float),

            "Q_alpha": np.zeros(n_kept, float),
            "s_alpha": np.zeros(n_kept, float),
            "alpha0": np.zeros(n_kept, float),

            "Q_beta": np.zeros(n_kept, float),
            "s_beta": np.zeros(n_kept, float),
            "beta0": np.zeros(n_kept, float),

            "Q_gamma": np.zeros(n_kept, float),
            "s_gamma": np.zeros(n_kept, float),
            "gamma0": np.zeros((n_kept, self.period - 1), float),

            "tau_alpha": np.zeros(n_kept, float),
            "tau_beta": np.zeros(n_kept, float),
            "tau_gamma": np.zeros(n_kept, float),
            "lambda2": np.zeros(n_kept, float),

            "x": np.zeros((n_kept, self.T, self.dim), float),
        }

        print_every = cfg.progress_every if cfg.progress_every > 0 else max(1, cfg.n_iter // 50)

        for it in range(cfg.n_iter):
            # 1) z | rest
            self.ffbs_ncp()
            self._refresh_cp_from_ncp()

            # 2) (alpha0,beta0,gamma0,s_*) | z, rest
            self.update_theta()
            self._refresh_cp_from_ncp()

            # 3) sign-switch move
            self.random_sign_switches()
            self._refresh_cp_from_ncp()

            # 4) lasso scales
            self.update_lasso_scales()

            # 5) sigma2
            self.update_sigma2()

            if cfg.progress and (((it + 1) % print_every == 0) or (it == cfg.n_iter - 1)):
                print(
                    f"[it {it+1}/{cfg.n_iter}] "
                    f"σ={math.sqrt(self.sigma2):.3f} | "
                    f"Qα={self.s_alpha**2:.3g} Qβ={self.s_beta**2:.3g} Qγ={self.s_gamma**2:.3g} | "
                    f"α0={self.alpha0:.3g} β0={self.beta0:.3g} | "
                    f"τ=[{self.tau_alpha:.3g},{self.tau_beta:.3g},{self.tau_gamma:.3g}] λ²={self.lambda2:.3g}"
                )

            if it in save_set:
                mu = self.mu_vec()
                self.keep["mu"][keep_idx] = mu
                self.keep["sigma"][keep_idx] = math.sqrt(self.sigma2)

                self.keep["Q_alpha"][keep_idx] = self.s_alpha**2
                self.keep["s_alpha"][keep_idx] = self.s_alpha
                self.keep["alpha0"][keep_idx] = self.alpha0

                self.keep["Q_beta"][keep_idx] = self.s_beta**2
                self.keep["s_beta"][keep_idx] = self.s_beta
                self.keep["beta0"][keep_idx] = self.beta0

                self.keep["Q_gamma"][keep_idx] = self.s_gamma**2
                self.keep["s_gamma"][keep_idx] = self.s_gamma
                self.keep["gamma0"][keep_idx] = self.gamma0

                self.keep["tau_alpha"][keep_idx] = self.tau_alpha
                self.keep["tau_beta"][keep_idx] = self.tau_beta
                self.keep["tau_gamma"][keep_idx] = self.tau_gamma
                self.keep["lambda2"][keep_idx] = self.lambda2

                self.keep["x"][keep_idx] = self.x[1:, :]
                keep_idx += 1

        return self.keep

    # ----------------------------- Persistence ----------------------------- #
    def save_posterior(self, out_npz_path: str, extra_meta: Optional[dict] = None) -> None:
        os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)

        arrays = dict(self.keep)
        arrays["y"] = self.y.copy()

        # truth overlays
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

        if self.true_m0_level is not None:
            arrays["true_m0_level"] = float(self.true_m0_level)
        if self.true_m0_trend is not None:
            arrays["true_m0_trend"] = float(self.true_m0_trend)
        if self.true_m0_season is not None:
            arrays["true_m0_season"] = np.asarray(self.true_m0_season, float)

        if self.true_P0_level is not None:
            arrays["true_P0_level"] = float(self.true_P0_level)
        if self.true_P0_trend is not None:
            arrays["true_P0_trend"] = float(self.true_P0_trend)
        if self.true_P0_season is not None:
            arrays["true_P0_season"] = np.asarray(self.true_P0_season, float)

        np.savez_compressed(out_npz_path, **arrays)

        meta = {
            "T": int(self.T),
            "dim": int(self.dim),
            "period": int(self.period),
            "layout": list(self._layout),
            "cfg": asdict(self.cfg),
            "priors": asdict(self.priors),
            "time_center": float(self._tbar),
        }
        if extra_meta:
            meta.update(extra_meta)

        meta_path = out_npz_path.replace(".npz", ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"[save] Posterior -> {out_npz_path}")
        print(f"[save] Metadata  -> {meta_path}")


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    def _parse_date(s: str | None) -> datetime:
        if not s:
            return datetime.today()
        parts = [int(p) for p in s.split("-")]
        if len(parts) == 1:
            return datetime(parts[0], 1, 1)
        if len(parts) == 2:
            return datetime(parts[0], parts[1], 1)
        if len(parts) == 3:
            return datetime(parts[0], parts[1], parts[2])
        raise ValueError("start-date must be YYYY, YYYY-MM, or YYYY-MM-DD")

    def _parse_csv_floats(s: Optional[str], expected_len: Optional[int] = None) -> Optional[List[float]]:
        if s is None:
            return None
        ss = s.strip()
        if ss == "":
            return None
        vals = [float(z) for z in ss.split(",")]
        if expected_len is not None and len(vals) != expected_len:
            raise ValueError(f"Expected {expected_len} comma-separated floats, got {len(vals)}")
        return vals

    def _parse_bool(x) -> bool:
        if isinstance(x, bool):
            return x
        if x is None:
            return False
        s = str(x).strip().lower()
        return s in ("1", "true", "t", "yes", "y", "on")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(base_dir)

    from simulator.mean_time_series import Mean_Time_Series  # type: ignore

    p = argparse.ArgumentParser(
        description=(
            "Kalman FFBS + Gibbs for Gaussian DLM "
            "(dynamic/dynamic/dynamic; newest-first seasonal baseline). "
            "Process SDs have hierarchical Bayesian lasso prior. "
            "Fixes: correlated (alpha_c,beta) prior under centring; seasonal noise only in first seasonal state."
        )
    )

    # Simulation
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--period", type=int, default=4)
    p.add_argument("--start-date", type=str, default="2000-01-01")
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--q-level", type=float, default=0.001)
    p.add_argument("--q-trend", type=float, default=0.0000002)
    p.add_argument("--q-season", type=float, default=0.00005)
    p.add_argument("--m0-level", type=float, default=3.0)
    p.add_argument("--v0-level", type=float, default=0.05)
    p.add_argument("--m0-trend", type=float, default=0.1)
    p.add_argument("--v0-trend", type=float, default=0.05)
    p.add_argument("--m0-season", type=str, default=None)  # csv or None
    p.add_argument("--v0-season", type=str, default=None)  # csv or None

    # Priors
    p.add_argument("--prior-a-sigma", type=float, default=2.0)
    p.add_argument("--prior-b-sigma", type=float, default=1.0)
    p.add_argument("--prior-m0-alpha", type=float, default=0.0)
    p.add_argument("--prior-P0-alpha", type=float, default=10.0)
    p.add_argument("--prior-m0-beta", type=float, default=0.0)
    p.add_argument("--prior-P0-beta", type=float, default=10.0)
    p.add_argument("--prior-m0-gamma", type=str, default=None)  # csv
    p.add_argument("--prior-P0-gamma", type=float, default=5.0)
    p.add_argument("--prior-a-lambda", type=float, default=0.001)
    p.add_argument("--prior-b-lambda", type=float, default=0.001)

    # Sampler
    p.add_argument("--n-iter", type=int, default=10000)
    p.add_argument("--burn", type=int, default=5000)
    p.add_argument("--thin", type=int, default=2)
    p.add_argument("--seed", type=int, default=40)
    p.add_argument("--progress", default=True)
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--out-dir", type=str, default="results/simulations/DLM")
    p.add_argument("--plot", default=True)
    p.add_argument("--print-summary", default=True)

    # Initial values
    p.add_argument("--sigma-init", type=float, default=1.0)
    p.add_argument("--s-alpha-init", type=float, default=1e-2)
    p.add_argument("--s-beta-init", type=float, default=1e-3)
    p.add_argument("--s-gamma-init", type=float, default=1e-2)
    p.add_argument("--gamma0-init", type=str, default=None)  # csv

    args = p.parse_args()

    # ---- simulation seasonal settings ----
    start_date = _parse_date(args.start_date)
    K = int(args.period) - 1
    m0_season = _parse_csv_floats(args.m0_season, expected_len=K) or ([5.0] * K)
    v0_season = _parse_csv_floats(args.v0_season, expected_len=K) or ([0.25] * K)

    mts = Mean_Time_Series(
        sigma=float(args.sigma),
        period=int(args.period),
        level_mode="dynamic",
        trend_mode="dynamic",
        seasonal_mode="dynamic",
        q_level=float(args.q_level),
        q_trend=float(args.q_trend),
        q_season=float(args.q_season),
        m0_level=float(args.m0_level),
        v0_level=float(args.v0_level),
        m0_trend=float(args.m0_trend),
        v0_trend=float(args.v0_trend),
        m0_season=m0_season,
        v0_season=v0_season,
        start_date=start_date,
    )

    # ---- simulate observations ----
    y = np.array([mts.move() or mts.measure() for _ in range(int(args.T))], float)

    truths = mts.get_truth_paths(as_numpy=True)
    mu_T = truths["mu_t"][1 : 1 + int(args.T)]
    dates_T = truths["index"][: int(args.T)]

    if _parse_bool(args.plot):
        plt.figure(figsize=(10, 3))
        plt.plot(dates_T, y, lw=1)
        plt.title("Simulated observations")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # ---- priors ----
    pri_gamma_vec = _parse_csv_floats(args.prior_m0_gamma, expected_len=K) or ([0.0] * K)

    priors = Priors(
        a_sigma=float(args.prior_a_sigma),
        b_sigma=float(args.prior_b_sigma),
        m0_alpha=float(args.prior_m0_alpha),
        P0_alpha=float(args.prior_P0_alpha),
        m0_beta=float(args.prior_m0_beta),
        P0_beta=float(args.prior_P0_beta),
        m0_gamma=pri_gamma_vec,
        P0_gamma=float(args.prior_P0_gamma),
        a_lambda=float(args.prior_a_lambda),
        b_lambda=float(args.prior_b_lambda),
    )

    cfg = SamplerConfig(
        n_iter=int(args.n_iter),
        burn=int(args.burn),
        thin=int(args.thin),
        random_seed=int(args.seed),
        progress=_parse_bool(args.progress),
        progress_every=int(args.progress_every),
    )

    gamma0_init = _parse_csv_floats(args.gamma0_init, expected_len=K)

    sampler = DLMGibbsConjugate(
        y=y,
        period=int(args.period),
        sigma2_init=float(args.sigma_init) ** 2,
        s_alpha_init=float(args.s_alpha_init),
        s_beta_init=float(args.s_beta_init),
        s_gamma_init=float(args.s_gamma_init),
        alpha0=float(args.m0_level),
        beta0=float(args.m0_trend),
        gamma0=gamma0_init,
        priors=priors,
        cfg=cfg,
    )

    sampler.set_truth(
        sigma=float(mts.sigma),
        Q=(float(mts.q_level), float(mts.q_trend), float(mts.q_season)),
        m0_level=float(mts.m0_level),
        m0_trend=float(mts.m0_trend),
        m0_season=np.asarray(mts.m0_season, float),
        P0_level=float(mts.v0_level),
        P0_trend=float(mts.v0_trend),
        P0_season=np.asarray(mts.v0_season, float),
    )
    sampler.set_truth_paths(mu=mu_T)

    if _parse_bool(args.print_summary):
        print(
            f"\nSimulated {int(args.T)} observations (σ={mts.sigma}) "
            "with modes dynamic/dynamic/dynamic.\n"
        )
        print("Hierarchical Bayesian lasso prior for process SDs:")
        print("  s_k | τ_k, σ² ~ N(0, σ² τ_k)")
        print("  τ_k | λ²     ~ Exp(λ²/2)")
        print(f"  λ²  ~ Gamma(a_λ={priors.a_lambda:.3g}, b_λ={priors.b_lambda:.3g})\n")

    # ---- run ----
    t0 = time.time()
    post = sampler.run()
    elapsed = time.time() - t0
    print(f"\n[Run completed in {elapsed:.1f}s]")

    out_dir = os.path.join(
        str(args.out_dir),
        f"dynamic-dynamic-dynamic_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(out_dir, exist_ok=True)

    sampler.save_posterior(
        out_npz_path=os.path.join(out_dir, "posterior.npz"),
        extra_meta={
            "elapsed_seconds": float(elapsed),
            "lasso_priors": {"a_lambda": priors.a_lambda, "b_lambda": priors.b_lambda},
            "fixes": {
                "correlated_prior_alpha_c_beta": True,
                "seasonal_noise_first_component_only": True,
            },
        },
    )

    # ---- summary ----
    if _parse_bool(args.print_summary):
        print("\n--- Posterior means ---")
        print(f"σ = {np.mean(post['sigma']):.4f}")
        for k in ["alpha", "beta", "gamma"]:
            key = f"Q_{k}"
            if key in post:
                mQ = float(np.mean(post[key]))
                print(f"{key} = {mQ:.4g} (√Q ≈ {math.sqrt(mQ):.4g})")
        print(f"mean λ²  = {np.mean(post['lambda2']):.4g}")
        print(f"mean τ_α = {np.mean(post['tau_alpha']):.4g}")
        print(f"mean τ_β = {np.mean(post['tau_beta']):.4g}")
        print(f"mean τ_γ = {np.mean(post['tau_gamma']):.4g}")

    # ---- plot fit ----
    if _parse_bool(args.plot):
        mu_hat = post["mu"].mean(axis=0)
        plt.figure(figsize=(10, 4))
        plt.plot(dates_T, y, label="y_t", lw=1)
        plt.plot(dates_T, mu_T, "--", label="μ_t (truth)")
        plt.plot(dates_T, mu_hat, "-.", label="μ̂_t (post mean)")
        plt.title("DLM (Bayesian lasso prior on process SDs, dynamic/dynamic/dynamic)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
