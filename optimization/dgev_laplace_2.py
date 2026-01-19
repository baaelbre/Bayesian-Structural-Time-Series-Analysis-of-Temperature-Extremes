# optimization/dgev_laplace.py
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    from .ffbs import ffbs_dgev_ncp_laplace  # type: ignore
    from .utils_2 import (  # type: ignore
        spd_solve,
        symmetrize,
        rand_invgauss,
        gev_loglike_sum,
        gev_score_hess_mu,
        parse_bool,
        parse_date,
        parse_csv_floats,
    )
except ImportError:
    from ffbs import ffbs_dgev_ncp_laplace  # type: ignore
    from utils_2 import (  # type: ignore
        spd_solve,
        symmetrize,
        rand_invgauss,
        gev_loglike_sum,
        gev_score_hess_mu,
        parse_bool,
        parse_date,
        parse_csv_floats,
    )


# =============================================================================
# Small metadata helpers
# =============================================================================
def _iso_or_str(x: Any) -> str:
    """
    Best-effort conversion for dates to an ISO-like string for metadata.
    Works with datetime/date-like objects from parse_date(), Periods, strings, etc.
    """
    if x is None:
        return ""
    try:
        return str(x.isoformat())  # datetime/date
    except Exception:
        return str(x)


# =============================================================================
# Priors & config
# =============================================================================
@dataclass()
class Priors:
    # sigma^2 ~ Inv-Gamma(a_sigma, b_sigma) with shape–rate on sigma^2
    a_sigma: float = 2.0
    b_sigma: float = 2.0

    # xi ~ Uniform[xi_lower, xi_upper]
    xi_lower: float = -0.5
    xi_upper: float = 0.5

    # Baseline priors (alpha0, beta0, gamma0 fixed seasonal means)
    m0_alpha: float = 0.0
    P0_alpha: float = 10.0
    m0_beta: float = 0.0
    P0_beta: float = 10.0
    m0_gamma: Optional[Sequence[float]] = None  # length p-1 (newest-first)
    P0_gamma: float = 5.0

    # Hierarchical Bayesian lasso hyperparameters (shape–rate for lambda^2)
    a_lambda: float = 1.0
    b_lambda: float = 1.0


@dataclass()
class SamplerConfig:
    n_iter: int = 5000
    burn: int = 1000
    thin: int = 2
    random_seed: Optional[int] = 40
    progress: bool = True
    progress_every: int = 0  # 0 => ~2% of n_iter

    # Safety knobs
    max_tries_block: int = 25         # retry draws within a block
    s_cap: float = 1e8                # reject if |s_*| exceeds this (pragmatic truncation)
    laplace_z_clip: float = 1e6       # clip pseudo-obs shifts to avoid insane FFBS targets

    # kept for CLI compatibility (not used)
    slice_w: float = 1.0
    slice_m: int = 20


# =============================================================================
# Small safe helpers
# =============================================================================
def _finite(x: float) -> bool:
    return bool(np.isfinite(float(x)))


def _safe_square(x: float) -> float:
    # never raises; returns +inf if overflow
    xf = float(x)
    y = xf * xf
    return float(y)


def _fmt_g(x: float, prec: int = 3) -> str:
    xf = float(x)
    if not np.isfinite(xf):
        return "inf" if xf > 0 else "-inf"
    try:
        return f"{xf:.{prec}g}"
    except Exception:
        # extremely rare formatting edge-cases
        return str(xf)


# =============================================================================
# DGEV Laplace Approximate Gibbs (NCP + dummies + Bayesian lasso)
# =============================================================================
class DGEVLaplaceNCP:
    """
    Structural DGEV with Laplace pseudo-obs + NCP FFBS + FS joint regression step,
    plus exact MH for (logsigma, xi).

    """

    def __init__(
        self,
        y: np.ndarray,
        period: int,
        *,
        start_date: Optional[Any] = None,
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
        # knobs
        ffbs_C0_scale: float = 1e-6,
        ffbs_C0_A: float = 1e-6,
        ffbs_jitter: float = 1e-12,
        sigma2_eff: float = 1.0,
    ):
        self.y = np.asarray(y, float)
        self.T = int(self.y.size)
        self.period = int(period)
        if self.T < 1:
            raise ValueError("y must have length >= 1")
        if self.period < 2:
            raise ValueError("period must be >= 2")

        # optional metadata (e.g. simulation start date or data start date)
        self.start_date = None if start_date is None else _iso_or_str(start_date)

        self.priors = priors
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.random_seed)

        # time indices: t=1..T for obs; t=0..T for states
        self._t1 = np.arange(1, self.T + 1, dtype=float)
        self._tbar = float(self._t1.mean())
        self._t0 = np.arange(0, self.T + 1, dtype=float)

        # CP layout: [alpha, beta, g1..g_{p-1}]
        self._layout: List[str] = ["alpha", "beta"] + [f"g{k}" for k in range(1, self.period)]
        self.dim = len(self._layout)
        self.idx_alpha = 0
        self.idx_beta = 1
        self.idx_g1 = 2
        self.idx_g_start = 2
        self.idx_g_end = self.idx_g_start + (self.period - 2)  # inclusive
        self.K_gamma = self.period - 1  # baseline gamma0 length

        # baseline params
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

        # signed process SDs
        self.s_alpha = float(s_alpha_init)
        self.s_beta = float(s_beta_init)
        self.s_gamma = float(s_gamma_init)

        # lasso latent scales and global shrinkage
        self.tau_alpha = 1.0
        self.tau_beta = 1.0
        self.tau_gamma = 1.0
        self.lambda2 = 1.0

        self.sigma2_eff = float(max(sigma2_eff, 1e-12))

        # baseline seasonal design
        self._S = self._build_season_design()

        # NCP layout: [tilde_alpha, tilde_beta, A, tilde_g1..tilde_g_{p-1}]
        self.dim_ncp = self.dim + 1
        self.idx_tilde_alpha = 0
        self.idx_tilde_beta = 1
        self.idx_A = 2
        self.idx_tilde_g_start = 3
        self.idx_tilde_g1 = self.idx_tilde_g_start
        self.idx_tilde_g_end = self.idx_tilde_g_start + (self.period - 2)  # inclusive

        # paths
        self.z = np.zeros((self.T + 1, self.dim_ncp), float)
        self.x = np.zeros((self.T + 1, self.dim), float)

        # fixed NCP system matrices
        self._G_tilde = self._build_G_tilde()
        self._Q_tilde = self._build_Q_tilde()

        # FFBS knobs
        self._ffbs_C0_scale = float(ffbs_C0_scale)
        self._ffbs_C0_A = float(ffbs_C0_A)
        self._ffbs_jitter = float(ffbs_jitter)

        self._refresh_cp_from_ncp()

        # storage
        self.keep: Dict[str, np.ndarray] = {}
        self.last_loglike: float = float("nan")

        # optional truth overlays
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

        G[ia, ia] = 1.0
        G[ib, ib] = 1.0

        G[iA, iA] = 1.0
        G[iA, ib] = 1.0

        if K > 0:
            R = np.zeros((K, K), float)
            R[0, :] = -1.0
            if K > 1:
                R[1:, :-1] = np.eye(K - 1)
            G[gs : ge + 1, gs : ge + 1] = R

        return G

    def _build_Q_tilde(self) -> np.ndarray:
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
        tilde_g = z[:, self.idx_tilde_g_start : self.idx_tilde_g_end + 1]

        beta_cp = self.beta0 + self.s_beta * tilde_beta
        alpha_cp = self.alpha0 + self._t0 * self.beta0 + self.s_alpha * tilde_alpha + self.s_beta * A_t
        g_cp = self.s_gamma * tilde_g

        self.x[:, self.idx_alpha] = alpha_cp
        self.x[:, self.idx_beta] = beta_cp
        self.x[:, self.idx_g_start : self.idx_g_end + 1] = g_cp

    # ----------------------------- mean vector ----------------------------- #
    def mu_vec(self) -> np.ndarray:
        mu_dyn = self.x[1:, self.idx_alpha] + self.x[1:, self.idx_g1]
        mu_base = (self._S @ self.gamma0) if self._S.size else 0.0
        return mu_dyn + mu_base

    # ----------------------------- loglike + validity ----------------------------- #
    def _compute_loglike(self) -> float:
        mu = self.mu_vec()
        ll = gev_loglike_sum(self.y, mu, self.sigma, self.xi)
        return float(ll)

    def _ok(self) -> bool:
        if not _finite(self.sigma) or self.sigma <= 0:
            return False
        if not (self.priors.xi_lower <= float(self.xi) <= self.priors.xi_upper):
            return False
        if not np.all(np.isfinite(self.z)):
            return False
        if not np.all(np.isfinite(self.gamma0)):
            return False
        # pragmatic caps on signed SDs
        sc = float(self.cfg.s_cap)
        if (abs(float(self.s_alpha)) > sc) or (abs(float(self.s_beta)) > sc) or (abs(float(self.s_gamma)) > sc):
            return False
        ll = self._compute_loglike()
        return np.isfinite(ll)

    def _pack(self) -> dict:
        return {
            "z": self.z.copy(),
            "alpha0": float(self.alpha0),
            "beta0": float(self.beta0),
            "gamma0": self.gamma0.copy(),
            "s_alpha": float(self.s_alpha),
            "s_beta": float(self.s_beta),
            "s_gamma": float(self.s_gamma),
            "tau_alpha": float(self.tau_alpha),
            "tau_beta": float(self.tau_beta),
            "tau_gamma": float(self.tau_gamma),
            "lambda2": float(self.lambda2),
            "logsigma": float(self.logsigma),
            "sigma": float(self.sigma),
            "xi": float(self.xi),
            "last_loglike": float(self.last_loglike),
        }

    def _restore(self, snap: dict) -> None:
        self.z[:] = snap["z"]
        self.alpha0 = float(snap["alpha0"])
        self.beta0 = float(snap["beta0"])
        self.gamma0[:] = snap["gamma0"]
        self.s_alpha = float(snap["s_alpha"])
        self.s_beta = float(snap["s_beta"])
        self.s_gamma = float(snap["s_gamma"])
        self.tau_alpha = float(snap["tau_alpha"])
        self.tau_beta = float(snap["tau_beta"])
        self.tau_gamma = float(snap["tau_gamma"])
        self.lambda2 = float(snap["lambda2"])
        self.logsigma = float(snap["logsigma"])
        self.sigma = float(snap["sigma"])
        self.xi = float(snap["xi"])
        self.last_loglike = float(snap["last_loglike"])
        self._refresh_cp_from_ncp()

    def _try_block(self, draw_fn, max_tries: Optional[int] = None) -> bool:
        tries = int(self.cfg.max_tries_block if max_tries is None else max_tries)
        snap0 = self._pack()
        for _ in range(max(1, tries)):
            draw_fn()
            self._refresh_cp_from_ncp()
            if self._ok():
                return True
            self._restore(snap0)
        self._restore(snap0)
        return False

    # ----------------------------- Laplace pseudo-observations ----------------------------- #
    def laplace_pseudo_mu(self) -> Tuple[np.ndarray, np.ndarray]:
        mu = self.mu_vec()
        z_mu = np.zeros(self.T, float)
        R_t = np.zeros(self.T, float)

        z_clip = float(self.cfg.laplace_z_clip)

        for t in range(self.T):
            g, h = gev_score_hess_mu(self.y[t], mu[t], self.sigma, self.xi)

            # extra safety: if g is huge (usually from near-overflow), damp it
            if (not np.isfinite(g)) or (abs(float(g)) > 1e6):
                g = 0.0
            if (not np.isfinite(h)) or (h >= -1e-10):
                g, h = 0.0, -1e-10

            w = max(-h, 1e-12)
            # z_mu = mu - g/h
            shift = -g / h
            if not np.isfinite(shift):
                shift = 0.0
            if abs(float(shift)) > z_clip:
                shift = math.copysign(z_clip, float(shift))

            z_mu[t] = float(mu[t] + shift)
            R_t[t] = 1.0 / w

        return z_mu, np.clip(R_t, 1e-12, 1e12)

    # ----------------------------- FFBS in NCP (Laplace) ----------------------------- #
    def _ffbs_C0(self) -> np.ndarray:
        d = self.dim_ncp
        C0 = self._ffbs_C0_scale * np.eye(d)
        C0[self.idx_A, self.idx_A] = self._ffbs_C0_A
        return C0

    def ffbs_ncp_laplace(self, *, z_mu: np.ndarray, R_t: np.ndarray) -> None:
        if self._S.size:
            z_star = z_mu - (self._S @ self.gamma0)
        else:
            z_star = z_mu

        self.z = ffbs_dgev_ncp_laplace(
            z_star=z_star,
            R_t=R_t,
            G_tilde=self._G_tilde,
            Q_tilde=self._Q_tilde,
            alpha0=float(self.alpha0),
            beta0=float(self.beta0),
            t1=self._t1,
            s_alpha=float(self.s_alpha),
            s_beta=float(self.s_beta),
            s_gamma=float(self.s_gamma),
            idx_tilde_alpha=self.idx_tilde_alpha,
            idx_A=self.idx_A,
            idx_tilde_g_start=self.idx_tilde_g_start,
            m0=np.zeros(self.dim_ncp),
            C0=self._ffbs_C0(),
            rng=self.rng,
            jitter=self._ffbs_jitter,
        )

    # ----------------------------- JOINT regression update (FS style) ----------------------------- #
    def update_delta(self, *, z_mu: np.ndarray, R_t: np.ndarray) -> None:
        T = self.T
        t_c = self._t1 - self._tbar

        tilde_alpha = self.z[1:, self.idx_tilde_alpha]
        A_t = self.z[1:, self.idx_A]
        tilde_g1 = self.z[1:, self.idx_tilde_g1]

        if self.K_gamma > 0:
            X_static = np.column_stack([np.ones(T), t_c, self._S])
        else:
            X_static = np.column_stack([np.ones(T), t_c])

        X_dyn = np.column_stack([tilde_alpha, A_t, tilde_g1])
        X = np.column_stack([X_static, X_dyn])
        d = int(X.shape[1])

        w = 1.0 / np.maximum(R_t, 1e-12)
        sw = np.sqrt(w)
        Xw = X * sw[:, None]
        yw = z_mu * sw

        idx_alpha_c = 0
        idx_beta = 1
        idx_gamma0 = 2
        idx_s_alpha = idx_gamma0 + self.K_gamma
        idx_s_beta = idx_s_alpha + 1
        idx_s_gamma = idx_s_alpha + 2

        # prior mean
        m0 = np.zeros(d, float)
        m0[idx_alpha_c] = float(self.priors.m0_alpha + self._tbar * self.priors.m0_beta)
        m0[idx_beta] = float(self.priors.m0_beta)

        if self.K_gamma > 0:
            if self.priors.m0_gamma is None:
                m_gamma = np.zeros(self.K_gamma, float)
            else:
                m_gamma = np.asarray(self.priors.m0_gamma, float)
                if m_gamma.size != self.K_gamma:
                    raise ValueError("Priors.m0_gamma must have length p-1")
            m0[idx_gamma0 : idx_gamma0 + self.K_gamma] = m_gamma

        # prior precision
        prior_prec = np.zeros((d, d), float)

        P0a = max(float(self.priors.P0_alpha), 1e-12)
        P0b = max(float(self.priors.P0_beta), 1e-12)
        tb = float(self._tbar)

        Sigma_ab = np.array(
            [[P0a + (tb * tb) * P0b, tb * P0b], [tb * P0b, P0b]],
            dtype=float,
        )
        Sigma_ab = symmetrize(Sigma_ab) + 1e-15 * np.eye(2)
        Prec_ab = spd_solve(Sigma_ab, np.eye(2))
        prior_prec[idx_alpha_c : idx_beta + 1, idx_alpha_c : idx_beta + 1] = symmetrize(Prec_ab)

        if self.K_gamma > 0:
            Pg = max(float(self.priors.P0_gamma), 1e-12)
            prior_prec[
                idx_gamma0 : idx_gamma0 + self.K_gamma,
                idx_gamma0 : idx_gamma0 + self.K_gamma,
            ] = (1.0 / Pg) * np.eye(self.K_gamma)

        eps = 1e-16
        prior_prec[idx_s_alpha, idx_s_alpha] = 1.0 / max(self.sigma2_eff * self.tau_alpha, eps)
        prior_prec[idx_s_beta, idx_s_beta] = 1.0 / max(self.sigma2_eff * self.tau_beta, eps)
        prior_prec[idx_s_gamma, idx_s_gamma] = 1.0 / max(self.sigma2_eff * self.tau_gamma, eps)

        XtX = Xw.T @ Xw
        Xty = Xw.T @ yw

        post_prec = symmetrize(XtX + prior_prec) + 1e-12 * np.eye(d)
        post_cov = spd_solve(post_prec, np.eye(d))
        post_cov = symmetrize(post_cov)

        # guard
        eigmin = float(np.linalg.eigvalsh(post_cov).min())
        if eigmin < 1e-12:
            post_cov = post_cov + (1e-12 - eigmin) * np.eye(d)

        post_mean = post_cov @ (Xty + prior_prec @ m0)
        theta = self.rng.multivariate_normal(post_mean, post_cov)

        alpha_c = float(theta[idx_alpha_c])
        beta = float(theta[idx_beta])
        self.beta0 = beta
        self.alpha0 = alpha_c - self._tbar * beta

        if self.K_gamma > 0:
            self.gamma0 = theta[idx_gamma0 : idx_gamma0 + self.K_gamma].copy()

        self.s_alpha = float(theta[idx_s_alpha])
        self.s_beta = float(theta[idx_s_beta])
        self.s_gamma = float(theta[idx_s_gamma])

    # ----------------------------- lasso scale updates ----------------------------- #
    def update_lasso_scales(self) -> None:
        """
        Park–Casella conditional updates using sigma2_eff:
          tau_k | s_k, sigma2_eff, lambda2 ~ InvGaussian(mu_k, lambda2),
            mu_k = sqrt(lambda2 * sigma2_eff / s_k^2)
          lambda2 | tau ~ Gamma(a_lambda + K, b_lambda + 0.5 * sum tau_k)
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
        if not np.isfinite(ll_old) or not np.isfinite(ll_new):
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
        if not np.isfinite(ll_old) or not np.isfinite(ll_new):
            return

        logacc = ll_new - ll_old  # uniform prior
        if math.log(self.rng.random()) < min(0.0, logacc):
            self.xi = prop

    # ----------------------------- progress ----------------------------- #
    def _progress_line(self, it: int) -> str:
        q_a = _safe_square(self.s_alpha)
        q_b = _safe_square(self.s_beta)
        q_g = _safe_square(self.s_gamma)
        return (
            f"[it {it+1}/{self.cfg.n_iter}] "
            f"σ={float(self.sigma):.3f} | ξ={float(self.xi):.3f} | "
            f"Qα={_fmt_g(q_a)} Qβ={_fmt_g(q_b)} Qγ={_fmt_g(q_g)} | "
            f"α0={_fmt_g(self.alpha0)} β0={_fmt_g(self.beta0)} | "
            f"τ=[{_fmt_g(self.tau_alpha)},{_fmt_g(self.tau_beta)},{_fmt_g(self.tau_gamma)}] "
            f"λ²={_fmt_g(self.lambda2)} | "
            f"logL={float(self.last_loglike):.2f}"
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

        # initialize last-good snapshot
        self.last_loglike = self._compute_loglike()
        last_good = self._pack()

        for it in range(cfg.n_iter):
            # Build Laplace pseudo-obs
            z_mu, R_t = self.laplace_pseudo_mu()

            # (B) FFBS state draw: retry/restore if it creates invalid true loglike
            ok_states = self._try_block(lambda: self.ffbs_ncp_laplace(z_mu=z_mu, R_t=R_t))
            if not ok_states:
                # hard rescue: restore last good
                self._restore(last_good)

            # (C) delta regression draw
            ok_theta = self._try_block(lambda: self.update_delta(z_mu=z_mu, R_t=R_t))
            if not ok_theta:
                self._restore(last_good)

            # (D) sign switches
            snap_before_sign = self._pack()
            self.random_sign_switches()
            self._refresh_cp_from_ncp()
            if not self._ok():
                self._restore(snap_before_sign)

            # (E) lasso scales (should be safe now, but keep a gate anyway)
            snap_before_lasso = self._pack()
            self.update_lasso_scales()
            if not self._ok():
                self._restore(snap_before_lasso)

            # (F) MH updates for obs params (exact GEV likelihood)
            snap_before_mh = self._pack()
            self.update_logsigma()
            self.update_xi()
            self._refresh_cp_from_ncp()
            if not self._ok():
                self._restore(snap_before_mh)

            # monitoring
            self.last_loglike = self._compute_loglike()

            # update last-good if valid
            if np.isfinite(self.last_loglike) and self._ok():
                last_good = self._pack()
            else:
                self._restore(last_good)
                self.last_loglike = self._compute_loglike()

            if cfg.progress and (((it + 1) % print_every == 0) or (it == cfg.n_iter - 1)):
                print(self._progress_line(it))

            if it in save_set:
                mu_now = self.mu_vec()

                self.keep["sigma"][keep_idx] = self.sigma
                self.keep["xi"][keep_idx] = self.xi
                self.keep["mu"][keep_idx] = mu_now
                self.keep["loglike"][keep_idx] = self.last_loglike

                self.keep["Q_alpha"][keep_idx] = _safe_square(self.s_alpha)
                self.keep["s_alpha"][keep_idx] = self.s_alpha
                self.keep["alpha0"][keep_idx] = self.alpha0

                self.keep["Q_beta"][keep_idx] = _safe_square(self.s_beta)
                self.keep["s_beta"][keep_idx] = self.s_beta
                self.keep["beta0"][keep_idx] = self.beta0

                self.keep["Q_gamma"][keep_idx] = _safe_square(self.s_gamma)
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
            "ffbs_C0_scale": float(self._ffbs_C0_scale),
            "ffbs_C0_A": float(self._ffbs_C0_A),
            "sigma2_eff": float(self.sigma2_eff),
            # NEW: persist start_date (if provided)
            "start_date": self.start_date,
            "safety": {
                "max_tries_block": int(self.cfg.max_tries_block),
                "s_cap": float(self.cfg.s_cap),
                "laplace_z_clip": float(self.cfg.laplace_z_clip),
                "reject_restore_enabled": True,
            },
        }
        if extra_meta:
            meta.update(extra_meta)

        meta_path = out_npz_path.replace(".npz", ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"[save] Posterior -> {out_npz_path}")
        print(f"[save] Metadata  -> {meta_path}")


def main() -> None:
    import sys
    import matplotlib.pyplot as plt

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(base_dir)

    from simulator.extremal_time_series import Extremal_Time_Series  # type: ignore

    p = argparse.ArgumentParser(
        description=(
            "Non-centred structural DGEV with baseline seasonal fixed effects (sum-to-zero), "
            "dynamic seasonal deviations with innovations only in the first dummy, "
            "Bayesian lasso on signed process SDs, "
            "FFBS on Laplace pseudo-observations + JOINT FS regression update, "
            "with reject/restore safety."
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

    # safety
    p.add_argument("--max-tries-block", type=int, default=25)
    p.add_argument("--s-cap", type=float, default=1e8)
    p.add_argument("--laplace-z-clip", type=float, default=1e6)

    args = p.parse_args()
    start_date = parse_date(args.start_date)
    start_date_meta = _iso_or_str(start_date)

    K = int(args.period) - 1
    m0_season = parse_csv_floats(args.m0_season, expected_len=K) or ([2.0] * K)
    v0_season = parse_csv_floats(args.v0_season, expected_len=K) or ([0.01] * K)

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

    pri_gamma_vec = parse_csv_floats(args.prior_m0_gamma, expected_len=K) or ([0.0] * K)
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
        progress=parse_bool(args.progress),
        progress_every=int(args.progress_every),
        max_tries_block=int(args.max_tries_block),
        s_cap=float(args.s_cap),
        laplace_z_clip=float(args.laplace_z_clip),
    )

    gamma0_init = parse_csv_floats(args.gamma0_init, expected_len=K)
    sampler = DGEVLaplaceNCP(
        y=y,
        period=int(args.period),
        start_date=start_date,  # NEW: pass through so save_posterior writes it
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

    sampler.set_truth(sigma=float(args.sigma), xi=float(args.xi), Q=(args.q_level, args.q_trend, args.q_season))
    sampler.set_truth_paths(mu=mu_T)

    if parse_bool(args.print_summary):
        print(
            f"\nSimulated {int(args.T)} observations (σ={float(args.sigma):.4g}, ξ={float(args.xi):.4g}) "
            "with modes dynamic/dynamic/dynamic.\n"
        )
        print("Hierarchical Bayesian lasso prior for process SDs (signed s_k):")
        print("  s_k | τ_k, sigma2_eff ~ N(0, sigma2_eff τ_k)")
        print("  τ_k | λ²             ~ Exp(λ²/2)")
        print(f"  λ²  ~ Gamma(a_λ={priors.a_lambda:.3g}, b_λ={priors.b_lambda:.3g})\n")
        print("Safety enabled:")
        print(f"  - retry/restore blocks: max_tries_block={cfg.max_tries_block}")
        print(f"  - cap |s_*| at {cfg.s_cap:g}")
        print(f"  - clip Laplace pseudo shifts at {cfg.laplace_z_clip:g}\n")

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
            # also store explicitly in run meta (handy if you later change sampler fields)
            "start_date": start_date_meta,
            "lasso_priors": {"a_lambda": priors.a_lambda, "b_lambda": priors.b_lambda},
        },
    )

    if parse_bool(args.plot):
        mu_hat = post["mu"].mean(axis=0)
        plt.figure(figsize=(10, 4))
        plt.plot(dates_T, y, label="y_t", lw=1)
        plt.plot(dates_T, mu_T, "--", label="μ_t (truth)")
        plt.plot(dates_T, mu_hat, "-.", label="μ̂_t (post mean)")
        plt.title("DGEV (Laplace NCP, Bayesian lasso; with reject/restore safety)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
