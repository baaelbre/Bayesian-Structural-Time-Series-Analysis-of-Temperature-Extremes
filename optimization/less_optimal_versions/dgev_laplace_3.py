# optimization/dgev_laplace.py
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from ffbs import ffbs_dgev_ncp_laplace_ekf  # type: ignore

from utils import (  # type: ignore
    spd_solve,
    symmetrize,
    rand_invgauss,
    gev_loglike_sum,
    gev_score_hess_mu,
)


# =============================================================================
# Priors & config (hierarchical Bayesian lasso prior on process SDs)
# =============================================================================
@dataclass(slots=True)
class Priors:
    # σ² ~ Inv-Gamma(a_sigma, b_sigma) (shape–rate on σ²)
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

    # Hierarchical Bayesian lasso hyperparameters (shape–rate for λ²)
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

    # kept for CLI compatibility (not used)
    slice_w: float = 1.0
    slice_m: int = 20


# =============================================================================
# DGEV Laplace Approximate Gibbs (NCP + dummies + Bayesian lasso)
# =============================================================================
class DGEVLaplaceNCP:
    """
    Structural DGEV with:
      - NCP latent Gaussian structure for (alpha,beta,seasonal dummies)
      - Laplace pseudo-observations for μ_t
      - FFBS in NCP using pseudo-obs for the dynamic part (alpha_t + g1_t)
      - ONE joint regression update (FS-style) for:
            (alpha0, beta0, gamma0, s_alpha, s_beta, s_gamma)
        using the full pseudo-obs z_mu (not z_star), with centred time.
      - Bayesian lasso on signed process SDs s_*:
            s_k | τ_k, sigma2_eff ~ N(0, sigma2_eff * τ_k)
            τ_k | λ² ~ IG( ... ),  λ² ~ Gamma(a_lambda+K, b_lambda+0.5 sum τ_k)
      - Random sign switches
      - RW-MH for logsigma and xi using exact GEV likelihood

    Key behaviour (matches your "old code"):
      • Build (z_mu, R_t) ONCE per iteration and reuse in FFBS + regression.
      • Joint regression update of baselines + s_* in one multivariate Gaussian draw.
    """

    def __init__(
        self,
        y: np.ndarray,
        period: int,
        *,
        alpha0: float = 0.0,
        beta0: float = 0.0,
        gamma0: Optional[Sequence[float]] = None,  # len p-1 (newest-first)
        sigma_init: float = 1.0,
        xi_init: float = 0.0,
        s_alpha_init: float = 1e-2,
        s_beta_init: float = 1e-3,
        s_gamma_init: float = 1e-3,
        priors: Priors = Priors(),
        cfg: SamplerConfig = SamplerConfig(),
        ffbs_C0_scale: float = 1e-6,   # old default
        ffbs_C0_A: float = 1e-6,       # old default (A0 also tight)
        ffbs_jitter: float = 1e-12,
        sigma2_eff: float = 1.0,       # lasso/regression scale (independent of GEV sigma)
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

        # time indices (1..T) and its mean (for centring)
        self._t1 = np.arange(1, self.T + 1, dtype=float)
        self._tbar = float(self._t1.mean())
        self._t0 = np.arange(0, self.T + 1, dtype=float)

        # system layout: CP states alpha,beta,g1..g_{p-1}
        self._layout: List[str] = ["alpha", "beta"] + [f"g{k}" for k in range(1, self.period)]
        self.dim = len(self._layout)
        self.idx_alpha = 0
        self.idx_beta = 1
        self.idx_g_start = 2
        self.idx_g_end = self.idx_g_start + (self.period - 2)  # inclusive
        self.K_gamma = self.period - 1  # baseline gamma0 length

        # baselines
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        if gamma0 is None:
            self.gamma0 = np.zeros(self.K_gamma, float)
        else:
            g = np.asarray(gamma0, float)
            if g.size != self.K_gamma:
                raise ValueError("gamma0 must have length p-1 (newest-first)")
            self.gamma0 = g.copy()

        # observation params
        self.logsigma = float(math.log(max(float(sigma_init), 1e-12)))
        self.sigma = float(math.exp(self.logsigma))
        self.xi = float(xi_init)
        if not (self.priors.xi_lower <= self.xi <= self.priors.xi_upper):
            raise ValueError("xi_init outside prior support")

        # signed process SDs
        self.s_alpha = float(s_alpha_init)
        self.s_beta = float(s_beta_init)
        self.s_gamma = float(s_gamma_init)

        # lasso scales
        self.tau_alpha = 1.0
        self.tau_beta = 1.0
        self.tau_gamma = 1.0
        self.lambda2 = 1.0

        # sigma2_eff (used in regression + lasso scale updates)
        self.sigma2_eff = float(max(sigma2_eff, 1e-12))

        # seasonal baseline design S[t,:]@gamma0
        self._S = self._build_season_design()

        # NCP layout: [tilde_alpha, tilde_beta, A, tilde_g1..tilde_g_{p-1}]
        self.dim_ncp = self.dim + 1
        self.idx_tilde_alpha = 0
        self.idx_tilde_beta = 1
        self.idx_A = 2
        self.idx_tilde_g_start = 3
        self.idx_tilde_g_end = self.idx_tilde_g_start + (self.period - 2)  # inclusive

        # paths
        self.z = np.zeros((self.T + 1, self.dim_ncp), float)  # NCP
        self.x = np.zeros((self.T + 1, self.dim), float)      # CP

        # fixed NCP system matrices
        self._G_tilde = self._build_G_tilde()
        self._Q_tilde = self._build_Q_tilde()

        # FFBS initial covariance knobs
        self._ffbs_C0_scale = float(ffbs_C0_scale)
        self._ffbs_C0_A = float(ffbs_C0_A)
        self._ffbs_jitter = float(ffbs_jitter)

        self._refresh_cp_from_ncp()

        # storage
        self.keep: Dict[str, np.ndarray] = {}
        self.last_loglike: float = float("nan")

        # truth overlays (optional)
        self.true_sigma: Optional[float] = None
        self.true_xi: Optional[float] = None
        self.true_Q: Optional[np.ndarray] = None
        self.true_mu_t: Optional[np.ndarray] = None
        self.true_alpha_t: Optional[np.ndarray] = None
        self.true_beta_t: Optional[np.ndarray] = None
        self.true_gamma_t: Optional[np.ndarray] = None

    # --------------------- truth overlays (optional) --------------------- #
    def set_truth(
        self,
        *,
        sigma: Optional[float] = None,
        xi: Optional[float] = None,
        Q: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        self.true_sigma = None if sigma is None else float(sigma)
        self.true_xi = None if xi is None else float(xi)
        self.true_Q = None if Q is None else np.asarray(Q, float)

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

    # ----------------------------- seasonal baseline design ----------------------------- #
    def _build_season_design(self) -> np.ndarray:
        """
        Static seasonal design S[t,:] for baseline gamma0 (length p-1), sum-to-zero.
        Using t=0..T-1, season = t % p.
        """
        T, p = self.T, self.period
        K = p - 1
        S = np.zeros((T, K), float)
        for t in range(T):
            season = t % p
            if season < K:
                S[t, season] = 1.0
            else:
                S[t, :] = -1.0
        return S

    # ----------------------------- NCP system matrices ----------------------------- #
    def _build_G_tilde(self) -> np.ndarray:
        d = self.dim_ncp
        G = np.zeros((d, d), float)

        ia, ib, iA = self.idx_tilde_alpha, self.idx_tilde_beta, self.idx_A
        gs, ge = self.idx_tilde_g_start, self.idx_tilde_g_end
        K = ge - gs + 1  # = p-1

        # RW for tilde_alpha, tilde_beta
        G[ia, ia] = 1.0
        G[ib, ib] = 1.0

        # accumulator A_t = A_{t-1} + tilde_beta_{t-1}
        G[iA, iA] = 1.0
        G[iA, ib] = 1.0

        # seasonal rotation for dummies (sum-to-zero)
        if K > 0:
            R = np.zeros((K, K), float)
            R[0, :] = -1.0
            if K > 1:
                R[1:, :-1] = np.eye(K - 1)
            G[gs: ge + 1, gs: ge + 1] = R

        return G

    def _build_Q_tilde(self) -> np.ndarray:
        d = self.dim_ncp
        Q = np.zeros((d, d), float)
        Q[self.idx_tilde_alpha, self.idx_tilde_alpha] = 1.0
        Q[self.idx_tilde_beta, self.idx_tilde_beta] = 1.0
        for k in range(self.idx_tilde_g_start, self.idx_tilde_g_end + 1):
            Q[k, k] = 1.0
        # A has no innovation
        return Q

    # ----------------------------- map NCP -> CP ----------------------------- #
    def _refresh_cp_from_ncp(self) -> None:
        """
        alpha_t = alpha0 + t*beta0 + s_alpha*tilde_alpha_t + s_beta*A_t
        beta_t  = beta0 + s_beta*tilde_beta_t
        g_t     = s_gamma*tilde_gamma_t
        """
        z = self.z
        tilde_alpha = z[:, self.idx_tilde_alpha]
        tilde_beta = z[:, self.idx_tilde_beta]
        A_t = z[:, self.idx_A]
        tilde_gamma = z[:, self.idx_tilde_g_start: self.idx_tilde_g_end + 1]

        beta_cp = self.beta0 + self.s_beta * tilde_beta
        alpha_cp = self.alpha0 + self._t0 * self.beta0 + self.s_alpha * tilde_alpha + self.s_beta * A_t
        gamma_cp = self.s_gamma * tilde_gamma

        self.x[:, self.idx_alpha] = alpha_cp
        self.x[:, self.idx_beta] = beta_cp
        self.x[:, self.idx_g_start: self.idx_g_end + 1] = gamma_cp

    # ----------------------------- mean vector ----------------------------- #
    def mu_vec(self) -> np.ndarray:
        """
        mu_t = alpha_t + g1_t + S[t,:]@gamma0 for t=1..T
        """
        mu_dyn = self.x[1:, self.idx_alpha] + self.x[1:, self.idx_g_start]
        mu_base = (self._S @ self.gamma0) if self._S.size else 0.0
        return mu_dyn + mu_base

    def ffbs_ncp_laplace_ekf(self) -> tuple[np.ndarray, np.ndarray]:
        """
        One-pass EKF-like Laplace FFBS:
        - builds (z_mu, R_t) around mu0_t = offset_t + H a_t (a_t = G m_{t-1})
        - samples z_{0:T}
        Returns (z_mu, R_t) for reuse in the regression update.
        """
        offset = self.alpha0 + self.beta0 * self._t1
        if self._S.size:
            offset = offset + (self._S @ self.gamma0)

        z, z_mu, R_t = ffbs_dgev_ncp_laplace_ekf(
            y=self.y,
            offset=offset,
            G_tilde=self._G_tilde,
            Q_tilde=self._Q_tilde,
            sigma=float(self.sigma),
            xi=float(self.xi),
            s_alpha=float(self.s_alpha),
            s_beta=float(self.s_beta),
            s_gamma=float(self.s_gamma),
            idx_tilde_alpha=self.idx_tilde_alpha,
            idx_A=self.idx_A,
            idx_tilde_g_start=self.idx_tilde_g_start,
            m0=np.zeros(self.dim_ncp),
            rng=self.rng,
            jitter=self._ffbs_jitter,
        )

        self.z = z
        return z_mu, R_t


    # ----------------------------- JOINT regression update (FS style) ----------------------------- #
    def update_theta_fs(self, *, z_mu: np.ndarray, R_t: np.ndarray) -> None:
        """
        Joint Gaussian regression update for:
            (alpha0, beta0, gamma0, s_alpha, s_beta, s_gamma)

        Centred-time parametrisation:
            z_mu_t = α_c + β (t - t̄) + S[t]γ0
                     + s_alpha * tilde_alpha_t + s_beta * A_t + s_gamma * tilde_g1_t
                     + ε_t,  ε_t ~ N(0, R_t)

        After sampling, transform back:
            beta0  = β
            alpha0 = α_c - t̄ β
        """
        T = self.T
        t_c = self._t1 - self._tbar

        # regressors from NCP path
        tilde_alpha = self.z[1:, self.idx_tilde_alpha]
        A_t = self.z[1:, self.idx_A]
        tilde_g1 = self.z[1:, self.idx_tilde_g_start]

        # build design
        # static block: [1, t_c, S]
        if self.K_gamma > 0:
            X_static = np.column_stack([np.ones(T), t_c, self._S])
        else:
            X_static = np.column_stack([np.ones(T), t_c])

        # dyn block: [tilde_alpha, A_t, tilde_g1]
        X_dyn = np.column_stack([tilde_alpha, A_t, tilde_g1])

        X = np.column_stack([X_static, X_dyn])
        d = int(X.shape[1])

        # weights
        w = 1.0 / np.maximum(R_t, 1e-12)
        sw = np.sqrt(w)
        Xw = X * sw[:, None]
        yw = z_mu * sw

        # prior mean for θ_c = (α_c, β, γ0, sα, sβ, sγ)
        m0 = np.zeros(d, float)
        m0[0] = self.priors.m0_alpha + self.priors.m0_beta * self._tbar  # prior on α_c
        m0[1] = self.priors.m0_beta

        if self.K_gamma > 0:
            if self.priors.m0_gamma is None:
                m_gamma = np.zeros(self.K_gamma, float)
            else:
                m_gamma = np.asarray(self.priors.m0_gamma, float)
                if m_gamma.size != self.K_gamma:
                    raise ValueError("Priors.m0_gamma must have length p-1")
            m0[2: 2 + self.K_gamma] = m_gamma

        # prior variances
        eps = 1e-12
        s2 = np.full(d, eps, float)

        # α_c, β, γ0
        s2[0] = max(self.priors.P0_alpha, eps)
        s2[1] = max(self.priors.P0_beta, eps)
        if self.K_gamma > 0:
            s2[2: 2 + self.K_gamma] = max(self.priors.P0_gamma, eps)

        # lasso priors on s_* (use sigma2_eff, NOT GEV sigma²)
        idx_s_alpha = 2 + self.K_gamma
        idx_s_beta = 3 + self.K_gamma
        idx_s_gamma = 4 + self.K_gamma

        s2[idx_s_alpha] = max(self.sigma2_eff * self.tau_alpha, eps)
        s2[idx_s_beta] = max(self.sigma2_eff * self.tau_beta, eps)
        s2[idx_s_gamma] = max(self.sigma2_eff * self.tau_gamma, eps)

        inv_s2 = 1.0 / np.maximum(s2, eps)

        XtX = Xw.T @ Xw
        Xty = Xw.T @ yw
        prec = symmetrize(XtX + np.diag(inv_s2)) + 1e-12 * np.eye(d)

        cov = spd_solve(prec, np.eye(d), jitter=1e-12)
        cov = symmetrize(cov)
        mean = cov @ (Xty + inv_s2 * m0)

        theta = self.rng.multivariate_normal(mean, cov)

        alpha_c = float(theta[0])
        beta = float(theta[1])
        self.beta0 = beta
        self.alpha0 = alpha_c - self._tbar * beta

        if self.K_gamma > 0:
            self.gamma0 = theta[2: 2 + self.K_gamma].copy()

        self.s_alpha = float(theta[idx_s_alpha])
        self.s_beta = float(theta[idx_s_beta])
        self.s_gamma = float(theta[idx_s_gamma])

    # ----------------------------- lasso scale updates ----------------------------- #
    def update_lasso_scales(self) -> None:
        """
        Park–Casella style with sigma2_eff:
          tau_k | s_k, sigma2_eff, lambda2 ~ IG(mu_k, lambda2),
            mu_k = sqrt(lambda2 * sigma2_eff / s_k^2)
          lambda2 | tau ~ Gamma(a_lambda + K, b_lambda + 0.5 sum tau_k)
        """
        lam2 = max(float(self.lambda2), 1e-12)
        eps = 1e-16
        sig2 = self.sigma2_eff

        def upd_tau(sk: float) -> float:
            s2 = sk * sk
            if s2 < eps:
                return 1.0
            mu = math.sqrt(lam2 * sig2 / s2)
            return rand_invgauss(mu, lam2, self.rng)

        self.tau_alpha = upd_tau(self.s_alpha)
        self.tau_beta = upd_tau(self.s_beta)
        self.tau_gamma = upd_tau(self.s_gamma)

        K = 3
        shape = float(self.priors.a_lambda + K)
        rate = float(self.priors.b_lambda + 0.5 * (self.tau_alpha + self.tau_beta + self.tau_gamma))
        self.lambda2 = float(self.rng.gamma(shape=shape, scale=1.0 / max(rate, 1e-12)))

    # ----------------------------- sign switches ----------------------------- #
    def random_sign_switches(self) -> None:
        """
        Likelihood-invariant sign flips:
          - alpha: (s_alpha, tilde_alpha)
          - beta:  (s_beta, tilde_beta, A)
          - gamma: (s_gamma, all tilde_g)
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
            self.z[:, self.idx_tilde_g_start: self.idx_tilde_g_end + 1] *= -1.0

    # ----------------------------- observation parameter updates ----------------------------- #
    def _log_prior_logsigma(self, logsigma: float) -> float:
        a = float(self.priors.a_sigma)
        b = float(self.priors.b_sigma)
        return -2.0 * a * logsigma - b * math.exp(-2.0 * logsigma)

    def update_logsigma(self, step: float = 0.05) -> None:
        cur = float(self.logsigma)
        prop = cur + float(self.rng.normal(0.0, step))

        sigma_cur = float(math.exp(cur))
        sigma_prop = float(math.exp(prop))

        mu = self.mu_vec()
        ll_old = gev_loglike_sum(self.y, mu, sigma_cur, self.xi)
        ll_new = gev_loglike_sum(self.y, mu, sigma_prop, self.xi)
        if ll_new == -np.inf:
            return

        lp_old = self._log_prior_logsigma(cur)
        lp_new = self._log_prior_logsigma(prop)

        logacc = (ll_new + lp_new) - (ll_old + lp_old)
        if math.log(self.rng.random()) < min(0.0, logacc):
            self.logsigma = prop
            self.sigma = sigma_prop

    def update_xi(self, step: float = 0.05) -> None:
        cur = float(self.xi)
        prop = cur + float(self.rng.normal(0.0, step))

        lb, ub = float(self.priors.xi_lower), float(self.priors.xi_upper)
        if not (lb <= prop <= ub):
            return

        mu = self.mu_vec()
        ll_old = gev_loglike_sum(self.y, mu, self.sigma, cur)
        ll_new = gev_loglike_sum(self.y, mu, self.sigma, prop)
        if ll_new == -np.inf:
            return

        logacc = ll_new - ll_old
        if math.log(self.rng.random()) < min(0.0, logacc):
            self.xi = prop

    # ----------------------------- progress ----------------------------- #
    def _progress_line(self, it: int) -> str:
        return (
            f"[it {it+1}/{self.cfg.n_iter}] "
            f"σ={self.sigma:.3f} | ξ={self.xi:.3f} | "
            f"Qα={self.s_alpha**2:.3g} Qβ={self.s_beta**2:.3g} Qγ={self.s_gamma**2:.3g} | "
            f"α0={self.alpha0:.3g} β0={self.beta0:.3g} | "
            f"τ=[{self.tau_alpha:.3g},{self.tau_beta:.3g},{self.tau_gamma:.3g}] λ²={self.lambda2:.3g} | "
            f"logL={self.last_loglike:.2f}"
        )

    # ----------------------------- MCMC ----------------------------- #
    def run(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        save_iters = list(range(cfg.burn, cfg.n_iter, cfg.thin))
        save_set = set(save_iters)
        n_kept = len(save_iters)
        keep_idx = 0

        self.keep = {
            "sigma": np.zeros(n_kept, float),
            "xi": np.zeros(n_kept, float),
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

            "loglike": np.zeros(n_kept, float),
            "x": np.zeros((n_kept, self.T, self.dim), float),
        }

        print_every = cfg.progress_every if cfg.progress_every > 0 else max(1, cfg.n_iter // 50)

        for it in range(cfg.n_iter):
            # (A) One-pass EKF-like Laplace FFBS (also returns pseudo-obs)
            z_mu, R_t = self.ffbs_ncp_laplace_ekf()
            self._refresh_cp_from_ncp()

            # (C) Joint regression update for (alpha0,beta0,gamma0,s_*)
            self.update_theta_fs(z_mu=z_mu, R_t=R_t)
            self._refresh_cp_from_ncp()

            # (D) Random sign switches + refresh
            self.random_sign_switches()
            self._refresh_cp_from_ncp()

            # (E) Lasso scale updates (tau, lambda2)
            self.update_lasso_scales()

            # (F) Exact GEV MH updates
            self.update_logsigma()
            self.update_xi()

            # monitoring loglike
            mu_now = self.mu_vec()
            self.last_loglike = gev_loglike_sum(self.y, mu_now, self.sigma, self.xi)

            if cfg.progress and (((it + 1) % print_every == 0) or (it == cfg.n_iter - 1)):
                print(self._progress_line(it))

            if it in save_set:
                self.keep["sigma"][keep_idx] = self.sigma
                self.keep["xi"][keep_idx] = self.xi
                self.keep["mu"][keep_idx] = mu_now
                self.keep["loglike"][keep_idx] = self.last_loglike

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

    # ----------------------------- persistence ----------------------------- #
    def save_posterior(self, out_npz_path: str, extra_meta: Optional[dict] = None) -> None:
        os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)

        arrays = dict(self.keep)
        arrays["y"] = self.y.copy()

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
            "layout": list(self._layout),
            "cfg": asdict(self.cfg),
            "priors": asdict(self.priors),
            "time_center": float(self._tbar),
            "sigma2_eff": float(self.sigma2_eff),
        }
        if extra_meta:
            meta.update(extra_meta)

        meta_path = out_npz_path.replace(".npz", ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"[save] Posterior -> {out_npz_path}")
        print(f"[save] Metadata  -> {meta_path}")


# =============================================================================
# CLI / Example run (kept, but you likely use wrappers)
# =============================================================================
def _parse_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in ("1", "true", "t", "yes", "y", "on")


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


def _csv_floats_or_none(s: Optional[str]) -> Optional[List[float]]:
    if s is None:
        return None
    ss = str(s).strip()
    if not ss:
        return None
    return [float(z) for z in ss.split(",")]


def main() -> None:
    import sys
    import matplotlib.pyplot as plt

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(base_dir)

    from simulator.extremal_time_series import Extremal_Time_Series  # type: ignore

    p = argparse.ArgumentParser(
        description=(
            "Non-centred structural DGEV with seasonal dummies (newest-first) "
            "and Bayesian lasso prior on process SDs. "
            "FFBS on Laplace pseudo-observations + JOINT FS regression update."
        )
    )

    # Simulation
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--period", type=int, default=4)
    p.add_argument("--start-date", type=str, default="2000-01-01")
    p.add_argument("--sigma", type=float, default=3.0)
    p.add_argument("--xi", type=float, default=-0.1)
    p.add_argument("--q-level", type=float, default=0.0000001)
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
    p.add_argument("--prior-a-lambda", type=float, default=0.001)
    p.add_argument("--prior-b-lambda", type=float, default=0.001)

    # Sampler
    p.add_argument("--n-iter", type=int, default=8000)
    p.add_argument("--burn", type=int, default=4000)
    p.add_argument("--thin", type=int, default=2)
    p.add_argument("--seed", type=int, default=40)
    p.add_argument("--progress", default=True)
    p.add_argument("--progress-every", type=int, default=1)
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

    # knobs
    p.add_argument("--ffbs-C0-scale", type=float, default=1e-6)
    p.add_argument("--ffbs-C0-A", type=float, default=1e-6)
    p.add_argument("--sigma2-eff", type=float, default=1.0)

    args = p.parse_args()
    start_date = _parse_date(args.start_date)

    m0_season = _csv_floats_or_none(args.m0_season)
    v0_season = _csv_floats_or_none(args.v0_season)
    if m0_season is None:
        m0_season = [2.0] * (args.period - 1)
    if v0_season is None:
        v0_season = [0.01] * (args.period - 1)

    ts = Extremal_Time_Series(
        parameters=(float(args.sigma), float(args.xi)),
        level_mode="dynamic",
        trend_mode="dynamic",
        seasonal_mode="dynamic",
        period=int(args.period),
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

    y = []
    for _ in range(int(args.T)):
        ts.move()
        y.append(ts.measure())
    y = np.asarray(y, float)

    truths = ts.get_truth_paths(as_numpy=False)
    mu_T = np.asarray(truths["mu"][1 : 1 + int(args.T)], float)
    dates_T = truths.get("index", np.arange(int(args.T)))

    pri_gamma_vec = _csv_floats_or_none(args.prior_m0_gamma) or ([0.0] * (args.period - 1))
    priors = Priors(
        a_sigma=float(args.prior_a_sigma),
        b_sigma=float(args.prior_b_sigma),
        xi_lower=float(args.prior_xi_lower),
        xi_upper=float(args.prior_xi_upper),
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

    gamma0_init = _csv_floats_or_none(args.gamma0_init)

    sampler = DGEVLaplaceNCP(
        y=y,
        period=int(args.period),
        alpha0=float(args.m0_level),
        beta0=float(args.m0_trend),
        gamma0=gamma0_init,
        sigma_init=float(args.sigma_init),
        xi_init=float(args.xi_init),
        s_alpha_init=float(args.s_alpha_init),
        s_beta_init=float(args.s_beta_init),
        s_gamma_init=float(args.s_gamma_init),
        priors=priors,
        cfg=cfg,
        sigma2_eff=float(args.sigma2_eff),
    )

    sampler.set_truth(sigma=float(args.sigma), xi=float(args.xi), Q=(args.q_level, args.q_trend, args.q_season))
    sampler.set_truth_paths(mu=mu_T)

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
        extra_meta={"elapsed_seconds": float(elapsed)},
    )

    if _parse_bool(args.plot):
        mu_hat = post["mu"].mean(axis=0)
        plt.figure(figsize=(10, 4))
        plt.plot(dates_T, y, label="y_t", lw=1)
        plt.plot(dates_T, mu_T, "--", label="μ_t (truth)")
        plt.plot(dates_T, mu_hat, "-.", label="μ̂_t (post mean)")
        plt.title("DGEV (Laplace NCP, Bayesian lasso on process SDs)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
