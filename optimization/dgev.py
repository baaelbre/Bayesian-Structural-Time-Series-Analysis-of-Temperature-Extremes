# optimization/dgev_laplace.py
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
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from utils import (  # type: ignore
    spd_solve,
    symmetrize,
    rand_invgauss,
    gev_loglike_sum,
    gev_score_hess_mu,
)


# =============================================================================
# Robust wrappers
# =============================================================================
def _spd_solve_safe(M: np.ndarray, B: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    try:
        return spd_solve(M, B, jitter=eps)  # type: ignore[arg-type]
    except TypeError:
        return spd_solve(M, B, eps=eps)  # type: ignore[arg-type]


def _log_norm_1d(x: float, m: float, v: float) -> float:
    v = float(max(v, 1e-16))
    z = (x - m) / math.sqrt(v)
    return -0.5 * (z * z + math.log(v))


def _log_norm_vec(x: np.ndarray, m: np.ndarray, v: float) -> float:
    v = float(max(v, 1e-16))
    r = (x - m) / math.sqrt(v)
    return float(-0.5 * (np.sum(r * r) + x.size * math.log(v)))


def _logsumexp(logw: np.ndarray) -> float:
    lw = np.asarray(logw, float)
    m = float(np.max(lw))
    if not np.isfinite(m):
        return -np.inf
    return float(m + math.log(float(np.sum(np.exp(lw - m)))))


def _choice_logweights(rng: np.random.Generator, logw: np.ndarray) -> int:
    lw = np.asarray(logw, float)
    m = float(np.max(lw))
    if not np.isfinite(m):
        return -1
    w = np.exp(lw - m)
    s = float(np.sum(w))
    if s <= 0.0 or not np.isfinite(s):
        return -1
    p = w / s
    return int(rng.choice(len(p), p=p))


# =============================================================================
# Priors & config
# =============================================================================
@dataclass(slots=True)
class Priors:
    # sigma^2 ~ Inv-Gamma(a_sigma, b_sigma) on sigma^2 (shape–rate)
    a_sigma: float = 2.0
    b_sigma: float = 2.0

    # xi ~ Uniform[xi_lower, xi_upper]
    xi_lower: float = -0.5
    xi_upper: float = 0.5

    # Baseline priors (alpha0, beta0, gamma0)
    m0_alpha: float = 0.0
    P0_alpha: float = 10.0
    m0_beta: float = 0.0
    P0_beta: float = 10.0
    m0_gamma: Optional[Sequence[float]] = None  # length p-1 (newest-first)
    P0_gamma: float = 5.0

    # Bayesian lasso hyperparameters (Gamma on lambda^2, shape–rate)
    a_lambda: float = 1.0
    b_lambda: float = 1.0


@dataclass(slots=True)
class SamplerConfig:
    n_iter: int = 5000
    burn: int = 1000
    thin: int = 2
    random_seed: Optional[int] = 40

    # printing
    progress: bool = True
    progress_every: int = 0  # 0 => ~2% of n_iter

    # --- MH tuning / adaptation ---
    adapt: bool = True
    target_acc_1d: float = 0.44
    target_acc_block: float = 0.234
    adapt_rate: float = 0.05

    logsigma_step_init: float = 0.05
    xi_step_init: float = 0.05
    theta_step_init: float = 0.10  # used to init BOTH theta blocks

    # theta scaling for s_* (avoid tau-driven blowups)
    s_scale_floor: float = 1e-3

    # --- z proposal settings ---
    R_inflate: float = 1.0
    R_floor: float = 1e-10
    h_min: float = -1e-6

    # adapt R_inflate during burn-in
    adapt_R_inflate: bool = True
    target_acc_z: float = 0.25
    R_inflate_rate: float = 0.05
    R_inflate_min: float = 0.2
    R_inflate_max: float = 50.0

    # multiple-try MH for z (set >1 to help acceptance)
    z_mtm_M: int = 1

    # kept for CLI compatibility (not used)
    slice_w: float = 1.0
    slice_m: int = 20


# =============================================================================
# Linear-Gaussian FFBS (local copy to avoid circular imports)
# =============================================================================
def _ffbs_gaussian_1d_tvR(
    *,
    y: np.ndarray,      # (T,)
    G: np.ndarray,      # (d,d)
    Q: np.ndarray,      # (d,d)
    H: np.ndarray,      # (1,d)
    R_t: np.ndarray,    # (T,)
    m0: np.ndarray,
    C0: np.ndarray,
    rng: np.random.Generator,
    jitter: float = 1e-12,
    R_floor: float = 1e-12,
) -> np.ndarray:
    y = np.asarray(y, float)
    R_t = np.asarray(R_t, float)
    T = int(y.size)
    d = int(G.shape[0])
    H = np.asarray(H, float).reshape(1, d)

    m = np.zeros((T + 1, d), float)
    C = np.zeros((T + 1, d, d), float)
    a = np.zeros((T + 1, d), float)
    Rm = np.zeros((T + 1, d, d), float)

    m[0] = np.asarray(m0, float)
    C[0] = symmetrize(np.asarray(C0, float)) + jitter * np.eye(d)

    for t in range(1, T + 1):
        Robs = float(R_t[t - 1])
        if (not np.isfinite(Robs)) or Robs <= 0.0:
            Robs = R_floor
        Robs = max(Robs, R_floor)

        a[t] = G @ m[t - 1]
        Rm[t] = symmetrize(G @ C[t - 1] @ G.T + Q) + jitter * np.eye(d)

        F = float((H @ Rm[t] @ H.T).squeeze() + Robs)
        if (not np.isfinite(F)) or (F <= 0.0):
            F = float((H @ (Rm[t] + 1e-10 * np.eye(d)) @ H.T).squeeze() + Robs)

        K = (Rm[t] @ H.T) / F
        v = float(y[t - 1] - float((H @ a[t]).squeeze()))

        m[t] = a[t] + K[:, 0] * v
        C[t] = symmetrize(Rm[t] - K @ (H @ Rm[t])) + jitter * np.eye(d)

    z = np.zeros((T + 1, d), float)
    z[T] = rng.multivariate_normal(m[T], C[T])

    I = np.eye(d)
    for t in range(T - 1, -1, -1):
        Rinv = _spd_solve_safe(Rm[t + 1], I, eps=jitter)
        J = C[t] @ G.T @ Rinv

        mean = m[t] + J @ (z[t + 1] - (G @ m[t]))
        cov = symmetrize(C[t] - J @ Rm[t + 1] @ J.T)

        eigmin = float(np.linalg.eigvalsh(cov).min())
        if (not np.isfinite(eigmin)) or (eigmin < jitter):
            cov = cov + (jitter - (eigmin if np.isfinite(eigmin) else 0.0)) * np.eye(d)

        z[t] = rng.multivariate_normal(mean, cov)

    return z


# =============================================================================
# Sampler
# =============================================================================
class DGEVLaplaceNCP:
    """
    TRUE DGEV posterior sampler with MH-corrected latent-path proposals.

    Requested changes:
      - theta update split into TWO blocks:
          (alpha0, beta0, gamma0) and (s_alpha, s_beta, s_gamma),
        each with its own step size and adaptation.
      - progress line printed ONLY in the old single-line format.
      - R_inflate adapted during burn-in to target z acceptance (Robbins–Monro on log R).
    """

    def __init__(
        self,
        y: np.ndarray,
        period: int,
        *,
        alpha0: float = 0.0,
        beta0: float = 0.0,
        gamma0: Optional[Sequence[float]] = None,  # length p-1 (newest-first)
        sigma_init: float = 1.0,
        xi_init: float = 0.0,
        s_alpha_init: float = 1e-2,
        s_beta_init: float = 1e-3,
        s_gamma_init: float = 1e-3,
        priors: Priors = Priors(),
        cfg: SamplerConfig = SamplerConfig(),
        # FFBS initial covariance knobs
        ffbs_C0_scale: float = 1e-6,
        ffbs_C0_A: float = 1e-6,
        ffbs_jitter: float = 1e-12,
        sigma2_eff: float = 1.0,  # lasso prior scale (decoupled from GEV sigma^2)
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
        self._t1 = np.arange(1, self.T + 1, dtype=float)
        self._tbar = float(self._t1.mean())
        self._t0 = np.arange(0, self.T + 1, dtype=float)

        # CP layout: [alpha, beta, g1, ..., g_{p-1}]
        self._layout: List[str] = ["alpha", "beta"] + [f"g{k}" for k in range(1, self.period)]
        self.dim = len(self._layout)
        self.idx_alpha = 0
        self.idx_beta = 1
        self.idx_g1 = 2
        self.idx_g_start = 2
        self.idx_g_end = self.idx_g_start + (self.period - 2)  # inclusive
        self.K_gamma = self.period - 1  # baseline gamma0 length

        # baseline parameters
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
        if float(sigma_init) <= 0:
            raise ValueError("sigma_init must be > 0")
        self.logsigma = float(math.log(max(float(sigma_init), 1e-12)))
        self.sigma = float(math.exp(self.logsigma))
        self.xi = float(xi_init)
        if not (self.priors.xi_lower <= self.xi <= self.priors.xi_upper):
            raise ValueError("xi_init outside prior support")

        # signed process SDs (lasso shrinkage acts on these)
        self.s_alpha = float(s_alpha_init)
        self.s_beta = float(s_beta_init)
        self.s_gamma = float(s_gamma_init)

        # lasso latent scales and global shrinkage
        self.tau_alpha = 1.0
        self.tau_beta = 1.0
        self.tau_gamma = 1.0
        self.lambda2 = 1.0
        self.sigma2_eff = float(max(sigma2_eff, 1e-12))

        # baseline seasonal design S for gamma0 (sum-to-zero with last season = -sum others)
        self._S = self._build_season_design()

        # NCP layout: [tilde_alpha, tilde_beta, A, tilde_g1..tilde_g_{p-1}]
        self.dim_ncp = self.dim + 1  # adds accumulator A
        self.idx_tilde_alpha = 0
        self.idx_tilde_beta = 1
        self.idx_A = 2
        self.idx_tilde_g_start = 3
        self.idx_tilde_g1 = self.idx_tilde_g_start
        self.idx_tilde_g_end = self.idx_tilde_g_start + (self.period - 2)  # inclusive

        # paths
        self.z = np.zeros((self.T + 1, self.dim_ncp), float)  # NCP path
        self.x = np.zeros((self.T + 1, self.dim), float)      # CP path (derived)

        # fixed NCP system matrices
        self._G_tilde = self._build_G_tilde()
        self._Q_tilde = self._build_Q_tilde()

        # FFBS initial covariance knobs
        self._ffbs_C0_scale = float(ffbs_C0_scale)
        self._ffbs_C0_A = float(ffbs_C0_A)
        self._ffbs_jitter = float(ffbs_jitter)

        # MH tuning (two theta-block step sizes)
        self._logsigma_step = float(cfg.logsigma_step_init)
        self._xi_step = float(cfg.xi_step_init)
        self._theta_step_abg = float(cfg.theta_step_init)
        self._theta_step_s = float(cfg.theta_step_init)

        # adaptive R_inflate (live)
        self._R_inflate = float(cfg.R_inflate)

        # bookkeeping
        self.keep: Dict[str, np.ndarray] = {}
        self.last_loglike: float = float("nan")

        self.acc_z = 0
        self.acc_theta = 0          # total accepted across BOTH theta blocks (0..2 per iter)
        self.acc_theta_abg = 0
        self.acc_theta_s = 0
        self.acc_obs = 0            # counts BOTH logsigma and xi acceptances (0..2 per iter)

        self._refresh_cp_from_ncp()

    # ----------------------------- seasonal baseline design ----------------------------- #
    def _build_season_design(self) -> np.ndarray:
        T, p = self.T, self.period
        K = p - 1
        S = np.zeros((T, K), float)
        for i in range(T):
            season = i % p
            if season < K:
                S[i, season] = 1.0
            else:
                S[i, :] = -1.0
        return S

    # ----------------------------- NCP system matrices ----------------------------- #
    def _build_G_tilde(self) -> np.ndarray:
        d = self.dim_ncp
        G = np.zeros((d, d), float)

        ia, ib, iA = self.idx_tilde_alpha, self.idx_tilde_beta, self.idx_A
        gs, ge = self.idx_tilde_g_start, self.idx_tilde_g_end
        K = ge - gs + 1  # = p-1

        # RW1 for tilde_alpha, tilde_beta
        G[ia, ia] = 1.0
        G[ib, ib] = 1.0

        # accumulator A_t = A_{t-1} + tilde_beta_{t-1}
        G[iA, iA] = 1.0
        G[iA, ib] = 1.0

        # seasonal dummy rotation
        if K > 0:
            R = np.zeros((K, K), float)
            R[0, :] = -1.0
            if K > 1:
                R[1:, :-1] = np.eye(K - 1)
            G[gs:ge + 1, gs:ge + 1] = R

        return G

    def _build_Q_tilde(self) -> np.ndarray:
        """
        Standardised innovation covariance.
        Only tilde_g1 gets innovation noise for seasonality.
        """
        d = self.dim_ncp
        Q = np.zeros((d, d), float)
        Q[self.idx_tilde_alpha, self.idx_tilde_alpha] = 1.0
        Q[self.idx_tilde_beta, self.idx_tilde_beta] = 1.0
        if self.K_gamma > 0:
            Q[self.idx_tilde_g1, self.idx_tilde_g1] = 1.0
        return Q

    # ----------------------------- map NCP -> CP ----------------------------- #
    def _refresh_cp_from_ncp(self) -> None:
        z = self.z
        tilde_alpha = z[:, self.idx_tilde_alpha]
        tilde_beta = z[:, self.idx_tilde_beta]
        A_t = z[:, self.idx_A]
        tilde_g = z[:, self.idx_tilde_g_start:self.idx_tilde_g_end + 1]

        beta_cp = self.beta0 + self.s_beta * tilde_beta
        alpha_cp = self.alpha0 + self._t0 * self.beta0 + self.s_alpha * tilde_alpha + self.s_beta * A_t
        g_cp = self.s_gamma * tilde_g

        self.x[:, self.idx_alpha] = alpha_cp
        self.x[:, self.idx_beta] = beta_cp
        self.x[:, self.idx_g_start:self.idx_g_end + 1] = g_cp

    # ----------------------------- helpers ----------------------------- #
    def _ffbs_C0(self) -> np.ndarray:
        d = self.dim_ncp
        C0 = self._ffbs_C0_scale * np.eye(d)
        C0[self.idx_A, self.idx_A] = self._ffbs_C0_A
        return C0

    def _offset_vec(self, *, alpha0: float, beta0: float, gamma0: np.ndarray) -> np.ndarray:
        off = alpha0 + beta0 * self._t1
        if self._S.size:
            off = off + (self._S @ gamma0)
        return np.asarray(off, float)

    def _mu_from(
        self,
        *,
        z_path: np.ndarray,
        alpha0: float,
        beta0: float,
        gamma0: np.ndarray,
        s_alpha: float,
        s_beta: float,
        s_gamma: float,
    ) -> np.ndarray:
        tilde_alpha = z_path[1:, self.idx_tilde_alpha]
        A_t = z_path[1:, self.idx_A]
        tilde_g1 = z_path[1:, self.idx_tilde_g1]

        alpha_t = alpha0 + beta0 * self._t1 + s_alpha * tilde_alpha + s_beta * A_t
        mu = alpha_t + s_gamma * tilde_g1

        if self._S.size:
            mu = mu + (self._S @ gamma0)
        return np.asarray(mu, float)

    def mu_vec(self) -> np.ndarray:
        return self._mu_from(
            z_path=self.z,
            alpha0=self.alpha0,
            beta0=self.beta0,
            gamma0=self.gamma0,
            s_alpha=self.s_alpha,
            s_beta=self.s_beta,
            s_gamma=self.s_gamma,
        )

    def _H_row(self, *, s_alpha: float, s_beta: float, s_gamma: float) -> np.ndarray:
        d = self.dim_ncp
        H = np.zeros((1, d), float)
        H[0, self.idx_tilde_alpha] = float(s_alpha)
        H[0, self.idx_A] = float(s_beta)
        H[0, self.idx_tilde_g_start] = float(s_gamma)
        return H

    def _loglike_pseudo(
        self,
        *,
        z_path: np.ndarray,   # (T+1,d)
        z_mu: np.ndarray,     # (T,)
        R_t: np.ndarray,      # (T,)
        offset: np.ndarray,   # (T,)
        H: np.ndarray,        # (1,d)
    ) -> float:
        Rt = np.maximum(np.asarray(R_t, float), 1e-12)
        y_lin = np.asarray(z_mu, float) - np.asarray(offset, float)
        pred = (z_path[1:, :] @ H.T).reshape(-1)
        resid = y_lin - pred
        return float(-0.5 * np.sum((resid * resid) / Rt + np.log(Rt)))

    # ----------------------------- priors ----------------------------- #
    def _log_prior_theta(
        self,
        *,
        alpha0: float,
        beta0: float,
        gamma0: np.ndarray,
        s_alpha: float,
        s_beta: float,
        s_gamma: float,
        tau_alpha: float,
        tau_beta: float,
        tau_gamma: float,
    ) -> float:
        pr = self.priors
        lp = 0.0

        lp += _log_norm_1d(alpha0, pr.m0_alpha, pr.P0_alpha)
        lp += _log_norm_1d(beta0, pr.m0_beta, pr.P0_beta)

        if self.K_gamma > 0:
            if pr.m0_gamma is None:
                m = np.zeros(self.K_gamma, float)
            else:
                m = np.asarray(pr.m0_gamma, float)
            lp += _log_norm_vec(gamma0, m, pr.P0_gamma)

        # lasso prior: s_k | tau_k ~ N(0, sigma2_eff * tau_k)
        lp += _log_norm_1d(s_alpha, 0.0, self.sigma2_eff * max(float(tau_alpha), 1e-16))
        lp += _log_norm_1d(s_beta, 0.0, self.sigma2_eff * max(float(tau_beta), 1e-16))
        lp += _log_norm_1d(s_gamma, 0.0, self.sigma2_eff * max(float(tau_gamma), 1e-16))

        return float(lp)

    def _log_prior_logsigma(self, logsigma: float) -> float:
        a = float(self.priors.a_sigma)
        b = float(self.priors.b_sigma)
        # sigma^2 ~ IG(a,b) (shape–rate). Up to constant on logsigma:
        return -2.0 * a * logsigma - b * math.exp(-2.0 * logsigma)

    # ----------------------------- Gibbs: lasso scales ----------------------------- #
    def update_lasso_scales(self) -> None:
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
        if self.rng.random() < 0.5:
            self.s_alpha *= -1.0
            self.z[:, self.idx_tilde_alpha] *= -1.0
        if self.rng.random() < 0.5:
            self.s_beta *= -1.0
            self.z[:, self.idx_tilde_beta] *= -1.0
            self.z[:, self.idx_A] *= -1.0
        if self.K_gamma > 0 and self.rng.random() < 0.5:
            self.s_gamma *= -1.0
            self.z[:, self.idx_tilde_g_start:self.idx_tilde_g_end + 1] *= -1.0

    # ----------------------------- Laplace pseudo-observations ----------------------------- #
    def _laplace_pseudo_obs(
        self,
        *,
        y: np.ndarray,
        offset: np.ndarray,
        H: np.ndarray,
        R_inflate: float,
        m0: np.ndarray,
        C0: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build EKF-style Laplace pseudo observations for mu_t = offset_t + H z_t.

        Returns:
          z_mu : (T,) pseudo observations on mu_t
          R_t  : (T,) pseudo variances (already inflated by R_inflate)
        """
        T = int(y.size)
        d = int(self._G_tilde.shape[0])

        z_mu = np.zeros(T, float)
        R_t = np.zeros(T, float)

        m = np.zeros((T + 1, d), float)
        C = np.zeros((T + 1, d, d), float)

        m[0] = np.asarray(m0, float)
        C[0] = symmetrize(np.asarray(C0, float)) + self._ffbs_jitter * np.eye(d)

        h_min = float(self.cfg.h_min)
        R_floor = float(self.cfg.R_floor)

        for t in range(1, T + 1):
            a = self._G_tilde @ m[t - 1]
            Rm = symmetrize(self._G_tilde @ C[t - 1] @ self._G_tilde.T + self._Q_tilde) + self._ffbs_jitter * np.eye(d)

            mu0 = float(offset[t - 1] + float((H @ a).squeeze()))
            g, h = gev_score_hess_mu(float(y[t - 1]), mu0, float(self.sigma), float(self.xi))

            if (not np.isfinite(g)) or (not np.isfinite(h)) or (h >= h_min):
                g = 0.0
                h = h_min

            w = max(-float(h), 1e-12)
            Robs = 1.0 / w

            Robs *= float(max(R_inflate, 1e-12))
            Robs = float(np.clip(Robs, R_floor, 1e12))

            z_mu[t - 1] = mu0 - float(g) / float(h)
            R_t[t - 1] = Robs

            y_lin = float(z_mu[t - 1] - offset[t - 1])

            F = float((H @ Rm @ H.T).squeeze() + Robs)
            if (not np.isfinite(F)) or (F <= 0.0):
                F = float((H @ (Rm + 1e-10 * np.eye(d)) @ H.T).squeeze() + Robs)

            K = (Rm @ H.T) / F
            v = float(y_lin - float((H @ a).squeeze()))

            m[t] = a + K[:, 0] * v
            C[t] = symmetrize(Rm - K @ (H @ Rm)) + self._ffbs_jitter * np.eye(d)

        return z_mu, R_t

    # ----------------------------- z update: (multiple-try) independence MH ----------------------------- #
    def _z_log_weight(
        self,
        *,
        z_path: np.ndarray,
        z_mu: np.ndarray,
        R_t: np.ndarray,
        offset: np.ndarray,
        H: np.ndarray,
    ) -> float:
        mu = self._mu_from(
            z_path=z_path,
            alpha0=self.alpha0,
            beta0=self.beta0,
            gamma0=self.gamma0,
            s_alpha=self.s_alpha,
            s_beta=self.s_beta,
            s_gamma=self.s_gamma,
        )
        ll_true = gev_loglike_sum(self.y, mu, self.sigma, self.xi)
        if ll_true == -np.inf:
            return -np.inf
        ll_ps = self._loglike_pseudo(z_path=z_path, z_mu=z_mu, R_t=R_t, offset=offset, H=H)
        return float(ll_true - ll_ps)

    def _adapt_R_inflate(self, *, it: int, accepted: int) -> None:
        cfg = self.cfg
        if (not cfg.adapt) or (not cfg.adapt_R_inflate) or (it >= cfg.burn):
            return

        t = max(1.0, float(it + 1))
        gain = float(cfg.R_inflate_rate) / math.sqrt(t)

        # Robbins–Monro on log R: increase R if accepted < target
        logR = math.log(max(self._R_inflate, 1e-12))
        logR += gain * (float(cfg.target_acc_z) - float(accepted))
        self._R_inflate = float(
            np.clip(math.exp(logR), float(cfg.R_inflate_min), float(cfg.R_inflate_max))
        )

    def update_z_mh(self, it: int) -> None:
        cfg = self.cfg
        M = int(max(1, cfg.z_mtm_M))

        offset = self._offset_vec(alpha0=self.alpha0, beta0=self.beta0, gamma0=self.gamma0)
        H = self._H_row(s_alpha=self.s_alpha, s_beta=self.s_beta, s_gamma=self.s_gamma)

        # build ONE pseudo model for this iteration
        z_mu, R_used = self._laplace_pseudo_obs(
            y=self.y,
            offset=offset,
            H=H,
            R_inflate=float(self._R_inflate),
            m0=np.zeros(self.dim_ncp),
            C0=self._ffbs_C0(),
        )
        y_lin = z_mu - offset

        def draw_from_q() -> np.ndarray:
            return _ffbs_gaussian_1d_tvR(
                y=y_lin,
                G=self._G_tilde,
                Q=self._Q_tilde,
                H=H,
                R_t=R_used,
                m0=np.zeros(self.dim_ncp),
                C0=self._ffbs_C0(),
                rng=self.rng,
                jitter=self._ffbs_jitter,
                R_floor=float(cfg.R_floor),
            )

        logw_cur = self._z_log_weight(z_path=self.z, z_mu=z_mu, R_t=R_used, offset=offset, H=H)

        accepted = 0

        if M == 1:
            z_prop = draw_from_q()
            logw_prop = self._z_log_weight(z_path=z_prop, z_mu=z_mu, R_t=R_used, offset=offset, H=H)
            if np.isfinite(logw_prop):
                logacc = float(logw_prop - logw_cur)
                if math.log(self.rng.random()) < min(0.0, logacc):
                    self.z = z_prop
                    self.acc_z += 1
                    self._refresh_cp_from_ncp()
                    accepted = 1

            self._adapt_R_inflate(it=it, accepted=accepted)
            return

        # MTM
        Zcand: List[np.ndarray] = []
        logw = np.zeros(M, float)
        for i in range(M):
            zi = draw_from_q()
            Zcand.append(zi)
            logw[i] = self._z_log_weight(z_path=zi, z_mu=z_mu, R_t=R_used, offset=offset, H=H)

        k = _choice_logweights(self.rng, logw)
        if k < 0:
            self._adapt_R_inflate(it=it, accepted=0)
            return

        z_star = Zcand[k]

        # reverse set
        logw_rev = np.zeros(M, float)
        logw_rev[0] = logw_cur
        for j in range(1, M):
            zj = draw_from_q()
            logw_rev[j] = self._z_log_weight(z_path=zj, z_mu=z_mu, R_t=R_used, offset=offset, H=H)

        logsum_fwd = _logsumexp(logw)
        logsum_rev = _logsumexp(logw_rev)
        if (not np.isfinite(logsum_fwd)) or (not np.isfinite(logsum_rev)):
            self._adapt_R_inflate(it=it, accepted=0)
            return

        logacc = float(logsum_fwd - logsum_rev)
        if math.log(self.rng.random()) < min(0.0, logacc):
            self.z = z_star
            self.acc_z += 1
            self._refresh_cp_from_ncp()
            accepted = 1

        self._adapt_R_inflate(it=it, accepted=accepted)

    # ----------------------------- theta update: TWO blocks ----------------------------- #
    def _theta_block1_abg_mh(self, it: int) -> int:
        """
        Block 1: (alpha0, beta0, gamma0)
        """
        pr = self.priors

        # current
        alpha0_c = float(self.alpha0)
        beta0_c = float(self.beta0)
        gamma0_c = np.asarray(self.gamma0, float) if self.K_gamma > 0 else np.zeros(0, float)

        # proposal scaling: prior scales (works well as a default)
        d1 = 2 + self.K_gamma
        scales = np.ones(d1, float)
        scales[0] = math.sqrt(max(pr.P0_alpha, 1e-12))
        scales[1] = math.sqrt(max(pr.P0_beta, 1e-12))
        if self.K_gamma > 0:
            scales[2:] = math.sqrt(max(pr.P0_gamma, 1e-12))

        vec_c = np.concatenate([np.array([alpha0_c, beta0_c], float), gamma0_c])
        vec_p = vec_c + self._theta_step_abg * scales * self.rng.normal(size=d1)

        alpha0_p = float(vec_p[0])
        beta0_p = float(vec_p[1])
        gamma0_p = np.asarray(vec_p[2:], float) if self.K_gamma > 0 else np.zeros(0, float)

        # log-post current
        mu_c = self._mu_from(
            z_path=self.z,
            alpha0=alpha0_c,
            beta0=beta0_c,
            gamma0=gamma0_c,
            s_alpha=self.s_alpha,
            s_beta=self.s_beta,
            s_gamma=self.s_gamma,
        )
        ll_c = gev_loglike_sum(self.y, mu_c, self.sigma, self.xi)
        lp_c = self._log_prior_theta(
            alpha0=alpha0_c,
            beta0=beta0_c,
            gamma0=gamma0_c,
            s_alpha=self.s_alpha,
            s_beta=self.s_beta,
            s_gamma=self.s_gamma,
            tau_alpha=self.tau_alpha,
            tau_beta=self.tau_beta,
            tau_gamma=self.tau_gamma,
        )

        # log-post prop
        mu_p = self._mu_from(
            z_path=self.z,
            alpha0=alpha0_p,
            beta0=beta0_p,
            gamma0=gamma0_p,
            s_alpha=self.s_alpha,
            s_beta=self.s_beta,
            s_gamma=self.s_gamma,
        )
        ll_p = gev_loglike_sum(self.y, mu_p, self.sigma, self.xi)

        acc = 0
        if ll_p != -np.inf:
            lp_p = self._log_prior_theta(
                alpha0=alpha0_p,
                beta0=beta0_p,
                gamma0=gamma0_p,
                s_alpha=self.s_alpha,
                s_beta=self.s_beta,
                s_gamma=self.s_gamma,
                tau_alpha=self.tau_alpha,
                tau_beta=self.tau_beta,
                tau_gamma=self.tau_gamma,
            )
            logacc = float((ll_p + lp_p) - (ll_c + lp_c))
            acc = 1 if (math.log(self.rng.random()) < min(0.0, logacc)) else 0

        if acc == 1:
            self.alpha0 = alpha0_p
            self.beta0 = beta0_p
            if self.K_gamma > 0:
                self.gamma0 = gamma0_p.copy()
            self._refresh_cp_from_ncp()
            self.acc_theta_abg += 1

        if self.cfg.adapt and it < self.cfg.burn:
            t = max(1.0, float(it + 1))
            gain = float(self.cfg.adapt_rate) / math.sqrt(t)
            self._theta_step_abg *= math.exp(gain * (acc - float(self.cfg.target_acc_block)))
            self._theta_step_abg = float(np.clip(self._theta_step_abg, 1e-6, 10.0))

        return acc

    def _theta_block2_s_mh(self, it: int) -> int:
        """
        Block 2: (s_alpha, s_beta, s_gamma)
        """
        floor = float(self.cfg.s_scale_floor)

        s_c = np.array([self.s_alpha, self.s_beta, self.s_gamma], float)
        scales = np.array([max(abs(self.s_alpha), floor), max(abs(self.s_beta), floor), max(abs(self.s_gamma), floor)], float)
        s_p = s_c + self._theta_step_s * scales * self.rng.normal(size=3)

        s_alpha_p, s_beta_p, s_gamma_p = float(s_p[0]), float(s_p[1]), float(s_p[2])

        # current
        mu_c = self._mu_from(
            z_path=self.z,
            alpha0=self.alpha0,
            beta0=self.beta0,
            gamma0=self.gamma0,
            s_alpha=self.s_alpha,
            s_beta=self.s_beta,
            s_gamma=self.s_gamma,
        )
        ll_c = gev_loglike_sum(self.y, mu_c, self.sigma, self.xi)
        lp_c = self._log_prior_theta(
            alpha0=self.alpha0,
            beta0=self.beta0,
            gamma0=self.gamma0,
            s_alpha=self.s_alpha,
            s_beta=self.s_beta,
            s_gamma=self.s_gamma,
            tau_alpha=self.tau_alpha,
            tau_beta=self.tau_beta,
            tau_gamma=self.tau_gamma,
        )

        # proposed
        mu_p = self._mu_from(
            z_path=self.z,
            alpha0=self.alpha0,
            beta0=self.beta0,
            gamma0=self.gamma0,
            s_alpha=s_alpha_p,
            s_beta=s_beta_p,
            s_gamma=s_gamma_p,
        )
        ll_p = gev_loglike_sum(self.y, mu_p, self.sigma, self.xi)

        acc = 0
        if ll_p != -np.inf:
            lp_p = self._log_prior_theta(
                alpha0=self.alpha0,
                beta0=self.beta0,
                gamma0=self.gamma0,
                s_alpha=s_alpha_p,
                s_beta=s_beta_p,
                s_gamma=s_gamma_p,
                tau_alpha=self.tau_alpha,
                tau_beta=self.tau_beta,
                tau_gamma=self.tau_gamma,
            )
            logacc = float((ll_p + lp_p) - (ll_c + lp_c))
            acc = 1 if (math.log(self.rng.random()) < min(0.0, logacc)) else 0

        if acc == 1:
            self.s_alpha = s_alpha_p
            self.s_beta = s_beta_p
            self.s_gamma = s_gamma_p
            self._refresh_cp_from_ncp()
            self.acc_theta_s += 1

        if self.cfg.adapt and it < self.cfg.burn:
            t = max(1.0, float(it + 1))
            gain = float(self.cfg.adapt_rate) / math.sqrt(t)
            self._theta_step_s *= math.exp(gain * (acc - float(self.cfg.target_acc_block)))
            self._theta_step_s = float(np.clip(self._theta_step_s, 1e-6, 10.0))

        return acc

    def update_theta_mh(self, it: int) -> None:
        """
        Two-block theta update:
          1) (alpha0, beta0, gamma0)
          2) (s_alpha, s_beta, s_gamma)
        """
        acc1 = self._theta_block1_abg_mh(it)
        acc2 = self._theta_block2_s_mh(it)
        self.acc_theta += int(acc1) + int(acc2)

    # ----------------------------- sigma, xi MH ----------------------------- #
    def update_logsigma_mh(self, it: int) -> None:
        cur = float(self.logsigma)
        prop = cur + float(self.rng.normal(0.0, self._logsigma_step))

        sigma_cur = float(math.exp(cur))
        sigma_prop = float(math.exp(prop))

        mu = self.mu_vec()
        ll_old = gev_loglike_sum(self.y, mu, sigma_cur, self.xi)
        ll_new = gev_loglike_sum(self.y, mu, sigma_prop, self.xi)

        acc = 0
        if ll_new != -np.inf:
            lp_old = self._log_prior_logsigma(cur)
            lp_new = self._log_prior_logsigma(prop)
            logacc = float((ll_new + lp_new) - (ll_old + lp_old))
            acc = 1 if (math.log(self.rng.random()) < min(0.0, logacc)) else 0

        if acc == 1:
            self.logsigma = prop
            self.sigma = sigma_prop
            self.acc_obs += 1

        if self.cfg.adapt and it < self.cfg.burn:
            t = max(1.0, float(it + 1))
            gain = float(self.cfg.adapt_rate) / math.sqrt(t)
            self._logsigma_step *= math.exp(gain * (acc - float(self.cfg.target_acc_1d)))
            self._logsigma_step = float(np.clip(self._logsigma_step, 1e-6, 2.0))

    def update_xi_mh(self, it: int) -> None:
        cur = float(self.xi)
        prop = cur + float(self.rng.normal(0.0, self._xi_step))

        lb, ub = float(self.priors.xi_lower), float(self.priors.xi_upper)
        acc = 0
        if lb <= prop <= ub:
            mu = self.mu_vec()
            ll_old = gev_loglike_sum(self.y, mu, self.sigma, cur)
            ll_new = gev_loglike_sum(self.y, mu, self.sigma, prop)
            if ll_new != -np.inf:
                logacc = float(ll_new - ll_old)  # uniform prior
                acc = 1 if (math.log(self.rng.random()) < min(0.0, logacc)) else 0

        if acc == 1:
            self.xi = prop
            self.acc_obs += 1

        if self.cfg.adapt and it < self.cfg.burn:
            t = max(1.0, float(it + 1))
            gain = float(self.cfg.adapt_rate) / math.sqrt(t)
            self._xi_step *= math.exp(gain * (acc - float(self.cfg.target_acc_1d)))
            self._xi_step = float(np.clip(self._xi_step, 1e-6, 1.0))

    # ----------------------------- progress ----------------------------- #
    def _progress_line_old(self, it: int) -> str:
        # EXACT old style single line
        return (
            f"[it {it+1}/{self.cfg.n_iter}] σ={self.sigma:.3f} | ξ={self.xi:.3f} | "
            f"Qα={self.s_alpha**2:.3g} Qβ={self.s_beta**2:.3g} Qγ={self.s_gamma**2:.3g} | "
            f"α0={self.alpha0:.3g} β0={self.beta0:.3g} | "
            f"τ=[{self.tau_alpha:.3g},{self.tau_beta:.3g},{self.tau_gamma:.3g}] λ²={self.lambda2:.3g} | "
            f"acc(z,θ,obs)=[{self.acc_z},{self.acc_theta},{self.acc_obs}] | "
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
            # keep old mh_steps shape (3) for backward compatibility:
            #   [theta_step_abg, logsigma_step, xi_step]
            "mh_steps": np.zeros((n_kept, 3), float),
            # new: second theta step
            "theta_step_s": np.zeros(n_kept, float),
            "R_inflate": np.zeros(n_kept, float),
        }

        print_every = cfg.progress_every if cfg.progress_every > 0 else max(1, cfg.n_iter // 50)

        for it in range(cfg.n_iter):
            # (A) z update
            self.update_z_mh(it)

            # (B) theta update (two blocks)
            self.update_theta_mh(it)

            # (C) sign switches (exact invariances)
            self.random_sign_switches()
            self._refresh_cp_from_ncp()

            # (D) lasso scales
            self.update_lasso_scales()

            # (E) obs params
            self.update_logsigma_mh(it)
            self.update_xi_mh(it)

            # monitor
            mu_now = self.mu_vec()
            self.last_loglike = gev_loglike_sum(self.y, mu_now, self.sigma, self.xi)

            if cfg.progress and (((it + 1) % print_every == 0) or (it == cfg.n_iter - 1)):
                print(self._progress_line_old(it))

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
                self.keep["mh_steps"][keep_idx] = np.array(
                    [self._theta_step_abg, self._logsigma_step, self._xi_step], float
                )
                self.keep["theta_step_s"][keep_idx] = float(self._theta_step_s)
                self.keep["R_inflate"][keep_idx] = float(self._R_inflate)
                keep_idx += 1

        return self.keep

    # ----------------------------- persistence ----------------------------- #
    def save_posterior(self, out_npz_path: str, extra_meta: Optional[dict] = None) -> None:
        os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)

        arrays = dict(self.keep)
        arrays["y"] = self.y.copy()
        np.savez_compressed(out_npz_path, **arrays)

        meta = {
            "T": int(self.T),
            "dim": int(self.dim),
            "period": int(self.period),
            "layout": list(self._layout),
            "cfg": asdict(self.cfg),
            "priors": asdict(self.priors),
            "time_center": float(self._tbar),
            "ffbs_C0_scale": float(self._ffbs_C0_scale),
            "ffbs_C0_A": float(self._ffbs_C0_A),
            "sigma2_eff": float(self.sigma2_eff),
            "sampler": "TRUE_DGEV__MH_corrected_z_proposal__RW_MH_params__TWO_BLOCK_THETA",
            "z_proposal": "Laplace EKF pseudo-obs + Gaussian FFBS",
            "theta_blocks": {
                "block1": "(alpha0, beta0, gamma0)",
                "block2": "(s_alpha, s_beta, s_gamma)",
                "steps": "theta_step_abg in mh_steps[:,0], theta_step_s in theta_step_s",
            },
            "R_inflate_adaptation": {
                "enabled": bool(self.cfg.adapt_R_inflate),
                "target_acc_z": float(self.cfg.target_acc_z),
                "rate": float(self.cfg.R_inflate_rate),
                "min": float(self.cfg.R_inflate_min),
                "max": float(self.cfg.R_inflate_max),
            },
        }
        if extra_meta:
            meta.update(extra_meta)

        meta_path = out_npz_path.replace(".npz", ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"[save] Posterior -> {out_npz_path}")
        print(f"[save] Metadata  -> {meta_path}")


# =============================================================================
# CLI / Example run (kept)
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


def _parse_csv_floats(s: Optional[str], expected_len: Optional[int] = None) -> Optional[List[float]]:
    if s is None:
        return None
    ss = str(s).strip()
    if ss == "":
        return None
    vals = [float(z) for z in ss.split(",")]
    if expected_len is not None and len(vals) != expected_len:
        raise ValueError(f"Expected {expected_len} comma-separated floats, got {len(vals)}")
    return vals


def main() -> None:
    import sys
    import matplotlib.pyplot as plt

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(base_dir)

    from simulator.extremal_time_series import Extremal_Time_Series  # type: ignore

    p = argparse.ArgumentParser(
        description=(
            "TRUE structural DGEV sampler: MH-corrected latent-path proposals via Laplace EKF+FFBS, "
            "RW-MH for parameters using true GEV likelihood, Bayesian lasso on signed process SDs. "
            "Includes optional multiple-try MH for z and adaptive R_inflate. "
            "Theta is updated in TWO MH blocks."
        )
    )

    # Simulation
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--period", type=int, default=4)
    p.add_argument("--start-date", type=str, default="2000-01-01")
    p.add_argument("--sigma", type=float, default=3.0)
    p.add_argument("--xi", type=float, default=-0.1)
    p.add_argument("--q-level", type=float, default=0.001)
    p.add_argument("--q-trend", type=float, default=0.0000002)
    p.add_argument("--q-season", type=float, default=0.00005)
    p.add_argument("--m0-level", type=float, default=3.0)
    p.add_argument("--v0-level", type=float, default=0.05)
    p.add_argument("--m0-trend", type=float, default=0.01)
    p.add_argument("--v0-trend", type=float, default=0.0005)
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
    p.add_argument("--progress-every", type=int, default=20)  # set to 1 for per-iter lines
    p.add_argument("--out-dir", type=str, default="results/simulations/DGEV_TRUE_MH")
    p.add_argument("--plot", default=True)
    p.add_argument("--print-summary", default=True)

    # Initial inference values
    p.add_argument("--sigma-init", type=float, default=1.0)
    p.add_argument("--xi-init", type=float, default=-0.05)
    p.add_argument("--s-alpha-init", type=float, default=1e-1)
    p.add_argument("--s-beta-init", type=float, default=1e-1)
    p.add_argument("--s-gamma-init", type=float, default=1e-1)
    p.add_argument("--gamma0-init", type=str, default=None)

    # knobs
    p.add_argument("--ffbs-C0-scale", type=float, default=1e-6)
    p.add_argument("--ffbs-C0-A", type=float, default=1e-6)
    p.add_argument("--sigma2-eff", type=float, default=1.0)

    p.add_argument("--R-inflate", type=float, default=3.0)
    p.add_argument("--R-floor", type=float, default=1e-10)
    p.add_argument("--h-min", type=float, default=-1e-6)
    p.add_argument("--adapt-R-inflate", default=True)
    p.add_argument("--target-acc-z", type=float, default=0.25)
    p.add_argument("--R-inflate-rate", type=float, default=0.05)
    p.add_argument("--R-inflate-min", type=float, default=0.2)
    p.add_argument("--R-inflate-max", type=float, default=50)

    p.add_argument("--z-mtm-M", type=int, default=1)

    args = p.parse_args()
    start_date = _parse_date(args.start_date)

    K = int(args.period) - 1
    m0_season = _parse_csv_floats(args.m0_season, expected_len=K) or ([2.0] * K)
    v0_season = _parse_csv_floats(args.v0_season, expected_len=K) or ([0.01] * K)

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
    mu_T = np.asarray(truths["mu"][1:1 + int(args.T)], float)
    dates_T = truths.get("index", np.arange(int(args.T)))

    pri_gamma_vec = _parse_csv_floats(args.prior_m0_gamma, expected_len=K) or ([0.0] * K)
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
        R_inflate=float(args.R_inflate),
        R_floor=float(args.R_floor),
        h_min=float(args.h_min),
        adapt_R_inflate=_parse_bool(args.adapt_R_inflate),
        target_acc_z=float(args.target_acc_z),
        R_inflate_rate=float(args.R_inflate_rate),
        R_inflate_min=float(args.R_inflate_min),
        R_inflate_max=float(args.R_inflate_max),
        z_mtm_M=int(args.z_mtm_M),
    )

    gamma0_init = _parse_csv_floats(args.gamma0_init, expected_len=K)

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
        ffbs_C0_scale=float(args.ffbs_C0_scale),
        ffbs_C0_A=float(args.ffbs_C0_A),
        sigma2_eff=float(args.sigma2_eff),
    )

    if _parse_bool(args.print_summary):
        print(
            f"\nSimulated {int(args.T)} observations (σ={float(args.sigma):.4g}, ξ={float(args.xi):.4g}) "
            "with modes dynamic/dynamic/dynamic.\n"
        )
        print(
            f"Sampler: TRUE DGEV with MH-corrected Laplace-FFBS z proposals; "
            f"z_mtm_M={cfg.z_mtm_M}; adaptive R_inflate={cfg.adapt_R_inflate} (init={cfg.R_inflate}).\n"
        )

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
        plt.title("TRUE DGEV via MH-corrected Laplace-FFBS proposals")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
