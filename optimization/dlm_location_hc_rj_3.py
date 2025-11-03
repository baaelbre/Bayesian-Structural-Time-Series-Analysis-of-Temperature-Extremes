from __future__ import annotations

"""
Refactored Gaussian structural time-series model with RJ–MCMC + conjugate Gibbs.

Major changes vs. the original script
-------------------------------------
- Clear separation of concerns: utilities, priors/config, model structure, Kalman/FFBS,
  parameter updates, and the RJ kernel live in focused helpers/classes.
- Deterministic vs dynamic blocks are handled by a compact descriptor that builds the
  state dimension, system matrices, and deterministic contribution on demand.
- Numerics hardened: stable SPD solves, jittered Cholesky with graceful pseudo-inverse
  fallback, symmetric guards, and tiny-eigenvalue fixes in backward sampling.
- RNG plumbing: a numpy Generator is threaded throughout (reproducibility + testability).
- Storage simplified but flexible: auto-allocates only for active blocks; saves modes,
  means, variances, and states with dynamic width; metadata JSON kept.
- CLI kept but tightened; argument names unchanged where possible for drop-in use.

Notes
-----
- Half-Cauchy priors on process standard deviations implemented via IG mixture.
- Model priors enter RJ acceptance via log weights (Occam tilt ready).
- Identifiability preserved: trend=dynamic ⇒ level=dynamic.
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple
from collections import deque

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

# =============================================================================
# Utilities
# =============================================================================

EPS = 1e-12


def _mad(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    if v.size == 0:
        return 0.0
    m = np.median(v)
    return float(np.median(np.abs(v - m)))


def _robust_sd(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    return _mad(v) / 1.4826 if v.size else 0.0


def _spd_solve(M: np.ndarray, B: np.ndarray, jitter: float = 1e-12) -> np.ndarray:
    """Solve M X = B for SPD-like M with escalating jitter; pseudo-inverse fallback."""
    n = M.shape[0]
    I = np.eye(n)
    S = 0.5 * (M + M.T)
    for k in range(4):
        try:
            L = np.linalg.cholesky(S + (10**k) * jitter * I)
            Y = np.linalg.solve(L, B)
            return np.linalg.solve(L.T, Y)
        except np.linalg.LinAlgError:
            pass
    return np.linalg.pinv(M) @ B


def _fmt_list(vals, max_elems: int = 6, fmt: str = ".4g", sep: str = ", ", mode: str = "head") -> str:
    if vals is None:
        return "-"
    try:
        v = np.asarray(vals, dtype=float).ravel()
    except Exception:
        v = np.atleast_1d(vals)
    n = v.size
    if n == 0:
        return "[]"
    def _one(x):
        if isinstance(x, (float, np.floating)):
            if np.isnan(x):
                return "nan"
            if np.isposinf(x):
                return "inf"
            if np.isneginf(x):
                return "-inf"
        return f"{x:{fmt}}"
    if n <= max_elems:
        return "[" + sep.join(_one(x) for x in v) + "]"
    ell = "…"
    if mode == "both" and max_elems >= 3:
        k_head = max_elems // 2
        k_tail = max_elems - k_head
        head = sep.join(_one(x) for x in v[:k_head])
        tail = sep.join(_one(x) for x in v[-k_tail:])
        return f"[{head}{sep}{ell}{sep}{tail}]"
    head = sep.join(_one(x) for x in v[:max_elems])
    return f"[{head}{sep}{ell}]"


# =============================================================================
# Priors & Config
# =============================================================================

@dataclass
class Priors:
    # observation precision tau ~ Gamma(a, b) (shape–rate)
    a_sigma: float = 2.0
    b_sigma: float = 1.0

    # means for initial states / deterministic parameters
    m_m0_alpha: float = 0.0
    s_m0_alpha: float = 10.0
    m_m0_beta: float = 0.0
    s_m0_beta: float = 10.0

    # seasonal deterministic prior (newest-first, length p−1)
    m_m0_gamma: Optional[Sequence[float]] = None
    s_m0_gamma: float = 5.0

    # initial state variances: InvGamma on variance
    a_P0_alpha: float = 2.0
    b_P0_alpha: float = 1.0
    a_P0_beta: float = 2.0
    b_P0_beta: float = 1.0
    a_P0_gamma: float = 2.0
    b_P0_gamma: float = 1.0

    # Half-Cauchy scales (via IG mixture) for process SDs
    hc_scale_alpha: float = 0.5
    hc_scale_beta: float = 0.5
    hc_scale_gamma: float = 0.5


@dataclass
class SamplerConfig:
    n_iter: int = 20000
    burn: int = 5000
    thin: int = 2
    random_seed: Optional[int] = 40
    progress: bool = True
    progress_every: int = 0  # if 0, ~2% auto

    # RJ tuning
    rj_moves_per_iter: int = 2
    allow_none_level: bool = False
    allow_none_trend: bool = True
    allow_none_season: bool = True

    # RJ acceptance reporting
    rj_window: int = 500


# =============================================================================
# Model structure (layout + system matrices + deterministic mean)
# =============================================================================

class _Layout:
    """Keeps index bookkeeping for dynamic blocks and constructs system pieces."""

    def __init__(self, period: int, level: str, trend: str, season: str):
        ok = {"dynamic", "deterministic", "none"}
        if level not in ok or trend not in ok or season not in ok:
            raise ValueError("invalid mode")
        if trend == "dynamic" and level != "dynamic":
            raise ValueError("trend=dynamic requires level=dynamic")
        self.period = int(period)
        self.level_mode = level
        self.trend_mode = trend
        self.season_mode = season

        layout: List[str] = []
        self.idx_alpha = self.idx_beta = None
        self.idx_g_start = self.idx_g_end = None

        if level == "dynamic":
            self.idx_alpha = len(layout)
            layout.append("alpha")
        if trend == "dynamic":
            self.idx_beta = len(layout)
            layout.append("beta")
        if season == "dynamic":
            for k in range(1, period):
                layout.append(f"g{k}")
            if period > 1:
                self.idx_g_start = layout.index("g1")
                self.idx_g_end = self.idx_g_start + (period - 2)

        self._labels = layout
        self.dim = len(layout)

    # ---------- matrices ----------
    def H(self) -> np.ndarray:
        if self.dim == 0:
            return np.zeros((1, 0))
        h = np.zeros(self.dim)
        if self.idx_alpha is not None:
            h[self.idx_alpha] = 1.0
        if self.season_mode == "dynamic":
            h[self.idx_g_start] = 1.0
        return h.reshape(1, -1)

    def A(self) -> np.ndarray:
        if self.dim == 0:
            return np.zeros((0, 0))
        A = np.eye(self.dim)
        if (self.idx_alpha is not None) and (self.idx_beta is not None):
            A[self.idx_alpha, self.idx_beta] = 1.0
        if self.season_mode == "dynamic":
            gs, ge = self.idx_g_start, self.idx_g_end
            K = ge - gs + 1
            A[gs, gs:ge + 1] = -1.0
            if K > 1:
                A[gs + 1 : ge + 1, gs:ge] = np.eye(K - 1)
                A[gs + 1 : ge + 1, ge] = 0.0
        return A

    def u(self, m0_beta: float) -> np.ndarray:
        if self.dim == 0:
            return np.zeros(0)
        u = np.zeros(self.dim)
        if (self.idx_alpha is not None) and (self.trend_mode == "deterministic"):
            u[self.idx_alpha] = float(m0_beta)
        return u

    def Q(self, s_alpha: float, s_beta: float, s_gamma: float) -> np.ndarray:
        if self.dim == 0:
            return np.zeros((0, 0))
        Q = np.zeros((self.dim, self.dim))
        if (self.idx_alpha is not None) and (s_alpha > 0):
            Q[self.idx_alpha, self.idx_alpha] = s_alpha**2
        if (self.idx_beta is not None) and (s_beta > 0):
            Q[self.idx_beta, self.idx_beta] = s_beta**2
        if (self.season_mode == "dynamic") and (s_gamma > 0):
            Q[self.idx_g_start, self.idx_g_start] = s_gamma**2
        return Q


# =============================================================================
# Sampler
# =============================================================================

class DLMRJGibbs:
    """Gaussian structural DLM with level/trend/season blocks and RJ–MCMC."""

    # ------------------------------ init ---------------------------------- #
    def __init__(
        self,
        y: np.ndarray,
        period: int,
        level_mode: str = "deterministic",
        trend_mode: str = "deterministic",
        seasonal_mode: str = "deterministic",
        sigma2_init: float = 1.0,
        # dynamic initials
        m0_alpha_init: float = 0.0,
        P0_alpha_init: float = 1.0,
        m0_beta_init: float = 0.0,
        P0_beta_init: float = 1.0,
        m0_gamma_init: Optional[Sequence[float]] = None,
        P0_gamma_init: float = 1.0,
        s_alpha_init: float = 1e-2,
        s_beta_init: float = 1e-3,
        s_gamma_init: float = 1e-3,
        priors: Priors = Priors(),
        cfg: SamplerConfig = SamplerConfig(),
        model_prior: Optional[Dict[str, Dict[str, float]]] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        # data
        self.y = np.asarray(y, float)
        self.T = int(self.y.size)
        self.period = int(period)
        if self.period < 2:
            raise ValueError("period must be >= 2")

        self.priors, self.cfg = priors, cfg
        self.rng = rng or np.random.default_rng(cfg.random_seed)

        # modes & layout
        self.level_mode = level_mode
        self.trend_mode = trend_mode
        self.seasonal_mode = seasonal_mode
        self._layout = _Layout(self.period, level_mode, trend_mode, seasonal_mode)

        # model priors (Occam tilt defaults)
        self.model_prior = model_prior or {
            "level": {"dynamic": 0.5, "deterministic": 0.5, "none": 1e-12},
            "trend": {
                "dynamic": 0.5,
                "deterministic": 0.5,
                "none": 0.4 if cfg.allow_none_trend else 1e-12,
            },
            "season": {
                "dynamic": 0.3,
                "deterministic": 0.7,
                "none": 0.2 if cfg.allow_none_season else 1e-12,
            },
        }

        # observation variance
        self.sigma2 = float(sigma2_init)

        # process SDs + Half-Cauchy auxiliaries for dynamic blocks
        self.s_alpha = float(s_alpha_init) if self._layout.idx_alpha is not None else 0.0
        self.s_beta = float(s_beta_init) if self._layout.idx_beta is not None else 0.0
        self.s_gamma = float(s_gamma_init) if self.seasonal_mode == "dynamic" else 0.0
        self._a_alpha = 1.0
        self._a_beta = 1.0
        self._a_gamma = 1.0

        # initial means/vars for dynamic blocks (or deterministic parameters)
        self.m0_alpha = (
            float(m0_alpha_init)
            if self._layout.idx_alpha is not None
            else float(self.priors.m_m0_alpha)
        )
        self.P0_alpha = float(P0_alpha_init) if self._layout.idx_alpha is not None else 0.0

        self.m0_beta = (
            float(m0_beta_init) if self._layout.idx_beta is not None else float(self.priors.m_m0_beta)
        )
        self.P0_beta = float(P0_beta_init) if self._layout.idx_beta is not None else 0.0

        if self.seasonal_mode == "dynamic":
            if m0_gamma_init is None:
                self.m0_gamma = np.zeros(self.period - 1)
            else:
                g = np.asarray(m0_gamma_init, float)
                if g.size != self.period - 1:
                    raise ValueError("m0_gamma_init must have length p-1 (newest-first)")
                self.m0_gamma = g
            self.P0_gamma = float(P0_gamma_init)
            self._m0_gamma_full = None
        else:
            base = (
                np.zeros(self.period - 1)
                if self.priors.m_m0_gamma is None
                else np.asarray(self.priors.m_m0_gamma, float)
            )
            self.m0_gamma = base.astype(float)
            self.P0_gamma = 0.0
            self._m0_gamma_full = np.r_[self.m0_gamma, -float(np.sum(self.m0_gamma))]

        # latent states holder
        self._alloc_state_holder()
        if self._layout.dim > 0:
            m0_vec, P0_diag = self._current_m0_P0()
            self.x[0] = self.rng.multivariate_normal(
                m0_vec, np.diag(P0_diag) + EPS * np.eye(self._layout.dim)
            )
            self._seed_forward(Q_diag=np.full(self._layout.dim, 1e-6))

        # storage & RJ stats
        self.keep: Dict[str, np.ndarray] = {}
        self._mode_counts = {
            "level": {"dynamic": 0, "deterministic": 0, "none": 0},
            "trend": {"dynamic": 0, "deterministic": 0, "none": 0},
            "season": {"dynamic": 0, "deterministic": 0, "none": 0},
        }
        self.rj_stats = {
            "level": {"proposed": 0, "accepted": 0, "win": deque(maxlen=self.cfg.rj_window)},
            "trend": {"proposed": 0, "accepted": 0, "win": deque(maxlen=self.cfg.rj_window)},
            "season": {"proposed": 0, "accepted": 0, "win": deque(maxlen=self.cfg.rj_window)},
        }

        # progress line
        if self.cfg.progress:
            sd1 = _robust_sd(np.diff(self.y)) if self.T >= 2 else 0.0
            sd2 = _robust_sd(np.diff(self.y, n=2)) if self.T >= 3 else 0.0
            print(
                f"[init] modes L/T/S={self.level_mode[:3]}/{self.trend_mode[:3]}/{self.seasonal_mode[:3]} "
                f"| sd1={sd1:.4g} sd2={sd2:.4g}"
            )

    # --------------------------- structure helpers ------------------------- #

    def _alloc_state_holder(self) -> None:
        self.x = np.zeros((self.T + 1, self._layout.dim), float)

    def _current_m0_P0(self) -> Tuple[np.ndarray, np.ndarray]:
        m0, P0 = [], []
        if self._layout.idx_alpha is not None:
            m0.append(self.m0_alpha)
            P0.append(self.P0_alpha)
        if self._layout.idx_beta is not None:
            m0.append(self.m0_beta)
            P0.append(self.P0_beta)
        if self.seasonal_mode == "dynamic":
            m0.extend(list(self.m0_gamma))
            P0.extend([self.P0_gamma] * (self.period - 1))
        return np.asarray(m0, float), np.asarray(P0, float)

    # deterministic contribution to y_t
    def _mu_det_t(self, t: int) -> float:
        out = 0.0
        if self.level_mode == "deterministic":
            out += float(self.m0_alpha)
        if (self.trend_mode == "deterministic") and (self._layout.idx_alpha is None):
            out += float(self.m0_beta) * t
        if self.seasonal_mode == "deterministic":
            if getattr(self, "_m0_gamma_full", None) is None:
                mg = np.asarray(self.m0_gamma, float)
                self._m0_gamma_full = np.r_[mg, -mg.sum()]
            out += float(self._m0_gamma_full[t % self.period])
        return out

    # --------------------------- likelihood (Kalman) ------------------------ #

    def _kalman_loglik(self) -> float:
        H = self._layout.H()
        A = self._layout.A()
        Q = self._layout.Q(self.s_alpha, self.s_beta, self.s_gamma)
        R = float(self.sigma2)

        if self._layout.dim == 0:
            e = np.array([self.y[t] - self._mu_det_t(t) for t in range(self.T)], float)
            return -0.5 * np.sum(np.log(2 * np.pi * R) + (e * e) / R)

        m0_vec, P0_diag = self._current_m0_P0()
        m = m0_vec.copy()
        C = np.diag(P0_diag) + EPS * np.eye(self._layout.dim)
        ll = 0.0
        u = self._layout.u(self.m0_beta)

        for t in range(self.T):
            a = A @ m + u
            Rm = A @ C @ A.T + Q
            Rm = 0.5 * (Rm + Rm.T) + EPS * np.eye(self._layout.dim)

            y_det = self._mu_det_t(t)
            S = float(H @ Rm @ H.T + R)
            v = float(self.y[t] - y_det - H @ a)
            ll += -0.5 * (math.log(2 * math.pi) + math.log(S) + (v * v) / S)

            K = (Rm @ H.T) / S
            m = a + K.flatten() * v
            C = Rm - K @ (H @ Rm)
            C = 0.5 * (C + C.T) + EPS * np.eye(self._layout.dim)
        return float(ll)

    # ------------------------------ priors --------------------------------- #

    def _log_prior_current(self) -> float:
        lp = 0.0

        def lIG(x, shape, scale):
            x = max(float(x), 1e-300)
            return -(shape + 1.0) * math.log(x) - (scale / x)

        # obs variance via tau ~ Gamma(a,b)
        a, b = self.priors.a_sigma, self.priors.b_sigma
        tau = 1.0 / max(self.sigma2, 1e-300)
        lp += (a - 1.0) * math.log(tau) - b * tau

        # dynamic level
        if self._layout.idx_alpha is not None:
            Qa = max(self.s_alpha**2, 1e-300)
            aa = max(self._a_alpha, 1e-300)
            lp += lIG(Qa, 0.5, 1.0 / aa) + lIG(aa, 1.0, 1.0 / (self.priors.hc_scale_alpha**2))
            lp += -0.5 * ((self.m0_alpha - self.priors.m_m0_alpha) ** 2) / (self.priors.s_m0_alpha**2)
            lp += lIG(self.P0_alpha, self.priors.a_P0_alpha, self.priors.b_P0_alpha)

        # dynamic trend
        if self._layout.idx_beta is not None:
            Qb = max(self.s_beta**2, 1e-300)
            ab = max(self._a_beta, 1e-300)
            lp += lIG(Qb, 0.5, 1.0 / ab) + lIG(ab, 1.0, 1.0 / (self.priors.hc_scale_beta**2))
            lp += -0.5 * ((self.m0_beta - self.priors.m_m0_beta) ** 2) / (self.priors.s_m0_beta**2)
            lp += lIG(self.P0_beta, self.priors.a_P0_beta, self.priors.b_P0_beta)

        # dynamic seasonal
        if self.seasonal_mode == "dynamic":
            Qg = max(self.s_gamma**2, 1e-300)
            ag = max(self._a_gamma, 1e-300)
            lp += lIG(Qg, 0.5, 1.0 / ag) + lIG(ag, 1.0, 1.0 / (self.priors.hc_scale_gamma**2))
            base = (
                np.zeros(self.period - 1)
                if self.priors.m_m0_gamma is None
                else np.asarray(self.priors.m_m0_gamma, float)
            )
            s2 = self.priors.s_m0_gamma**2
            lp += -0.5 * float(np.sum((self.m0_gamma - base) ** 2)) / s2
            lp += lIG(self.P0_gamma, self.priors.a_P0_gamma, self.priors.b_P0_gamma)

        # deterministic params
        if self.level_mode == "deterministic":
            lp += -0.5 * ((self.m0_alpha - self.priors.m_m0_alpha) ** 2) / (
                self.priors.s_m0_alpha**2
            )
        if self.trend_mode == "deterministic":
            lp += -0.5 * ((self.m0_beta - self.priors.m_m0_beta) ** 2) / (self.priors.s_m0_beta**2)
        if self.seasonal_mode == "deterministic":
            K = self.period - 1
            mu = (
                np.zeros(K)
                if self.priors.m_m0_gamma is None
                else np.asarray(self.priors.m_m0_gamma, float).reshape(-1)
            )
            s2 = self.priors.s_m0_gamma**2
            lp += -0.5 * float(np.sum((np.asarray(self.m0_gamma) - mu) ** 2)) / s2

        # model priors
        lp += math.log(self.model_prior["level"][self.level_mode] + 1e-300)
        lp += math.log(self.model_prior["trend"][self.trend_mode] + 1e-300)
        lp += math.log(self.model_prior["season"][self.seasonal_mode] + 1e-300)
        return float(lp)

    # ------------------------------- FFBS ---------------------------------- #

    def _ffbs(self) -> np.ndarray:
        if self._layout.dim == 0:
            return np.zeros((self.T + 1, 0), float)

        H = self._layout.H()
        A = self._layout.A()
        Q = self._layout.Q(self.s_alpha, self.s_beta, self.s_gamma)
        R = float(self.sigma2)
        m0_vec, P0_diag = self._current_m0_P0()

        m = np.zeros((self.T + 1, self._layout.dim))
        C = np.zeros((self.T + 1, self._layout.dim, self._layout.dim))
        a = np.zeros_like(m)
        Rm = np.zeros_like(C)
        m[0] = m0_vec
        C[0] = np.diag(P0_diag) + EPS * np.eye(self._layout.dim)
        u = self._layout.u(self.m0_beta)

        for t in range(1, self.T + 1):
            a[t] = A @ m[t - 1] + u
            Rm[t] = A @ C[t - 1] @ A.T + Q
            Rm[t] = 0.5 * (Rm[t] + Rm[t].T) + EPS * np.eye(self._layout.dim)
            y_det = self._mu_det_t(t - 1)
            S = float(H @ Rm[t] @ H.T + R)
            v = float(self.y[t - 1] - y_det - H @ a[t])
            K = (Rm[t] @ H.T) / S
            m[t] = a[t] + K.flatten() * v
            C[t] = Rm[t] - K @ (H @ Rm[t])
            C[t] = 0.5 * (C[t] + C[t].T) + EPS * np.eye(self._layout.dim)

        x = np.zeros_like(m)
        x[self.T] = self.rng.multivariate_normal(m[self.T], C[self.T])
        for t in range(self.T - 1, -1, -1):
            J = C[t] @ A.T
            J = _spd_solve(Rm[t + 1], J.T).T
            mean = m[t] + J @ (x[t + 1] - (A @ m[t] + u))
            cov = C[t] - J @ Rm[t + 1] @ J.T
            cov = 0.5 * (cov + cov.T)
            mineig = float(np.linalg.eigvalsh(cov).min())
            if mineig < 1e-12:
                cov += (1e-12 - mineig) * np.eye(cov.shape[0])
            x[t] = self.rng.multivariate_normal(mean, cov)
        return x

    def _seed_forward(self, Q_diag: np.ndarray) -> None:
        if self._layout.dim == 0:
            return
        A = self._layout.A()
        u = self._layout.u(self.m0_beta)
        for t in range(1, self.T + 1):
            self.x[t] = A @ self.x[t - 1] + u + self.rng.normal(0.0, np.sqrt(Q_diag), size=self._layout.dim)

    # ----------------------- conjugate parameter updates ------------------- #

    @staticmethod
    def _rinvgamma(rng: np.random.Generator, shape: float, scale: float) -> float:
        return 1.0 / rng.gamma(shape, 1.0 / scale)

    def _mu_vec(self) -> np.ndarray:
        if self._layout.dim == 0:
            return np.array([self._mu_det_t(t) for t in range(self.T)], float)
        H = self._layout.H()
        mu = np.zeros(self.T)
        for t in range(1, self.T + 1):
            mu[t - 1] = self._mu_det_t(t - 1) + float(H @ self.x[t])
        return mu

    def update_sigma2(self) -> None:
        e = self.y - self._mu_vec()
        a = self.priors.a_sigma + 0.5 * self.T
        b = self.priors.b_sigma + 0.5 * float(e @ e)
        tau = self.rng.gamma(shape=a, scale=1.0 / b)
        self.sigma2 = 1.0 / max(tau, 1e-300)

    # innovation sums of squares (Half-Cauchy mixture)
    def _innovation_ss_alpha(self) -> Tuple[float, int]:
        li = self._layout.idx_alpha
        if li is None:
            return 0.0, 0
        ss = 0.0
        for t in range(1, self.T + 1):
            if self._layout.idx_beta is not None:
                drift = self.x[t - 1, self._layout.idx_beta]
            elif self.trend_mode == "deterministic":
                drift = float(self.m0_beta)
            else:
                drift = 0.0
            mean = self.x[t - 1, li] + drift
            ss += (self.x[t, li] - mean) ** 2
        return float(ss), self.T

    def _innovation_ss_beta(self) -> Tuple[float, int]:
        bi = self._layout.idx_beta
        if bi is None:
            return 0.0, 0
        d = self.x[1:, bi] - self.x[:-1, bi]
        return float(np.sum(d * d)), self.T

    def _innovation_ss_gamma(self) -> Tuple[float, int]:
        if self.seasonal_mode != "dynamic":
            return 0.0, 0
        gs, ge = self._layout.idx_g_start, self._layout.idx_g_end
        ss = 0.0
        for t in range(1, self.T + 1):
            prev = self.x[t - 1, gs : ge + 1]
            mean_new = -float(np.sum(prev))
            ss += (self.x[t, gs] - mean_new) ** 2
        return float(ss), self.T

    def update_process_Q_halfcauchy(self) -> None:
        # alpha
        if self._layout.idx_alpha is not None:
            SS, T_eff = self._innovation_ss_alpha()
            A = float(self.priors.hc_scale_alpha)
            Q_alpha = self._rinvgamma(self.rng, 0.5 * T_eff + 0.5, 0.5 * SS + 1.0 / max(self._a_alpha, 1e-300))
            self.s_alpha = math.sqrt(max(Q_alpha, 0.0))
            self._a_alpha = self._rinvgamma(self.rng, 1.0, (1.0 / (A * A)) + 1.0 / max(Q_alpha, 1e-300))
        # beta
        if self._layout.idx_beta is not None:
            SS, T_eff = self._innovation_ss_beta()
            A = float(self.priors.hc_scale_beta)
            Q_beta = self._rinvgamma(self.rng, 0.5 * T_eff + 0.5, 0.5 * SS + 1.0 / max(self._a_beta, 1e-300))
            self.s_beta = math.sqrt(max(Q_beta, 0.0))
            self._a_beta = self._rinvgamma(self.rng, 1.0, (1.0 / (A * A)) + 1.0 / max(Q_beta, 1e-300))
        # gamma
        if self.seasonal_mode == "dynamic":
            SS, T_eff = self._innovation_ss_gamma()
            A = float(self.priors.hc_scale_gamma)
            Q_gamma = self._rinvgamma(self.rng, 0.5 * T_eff + 0.5, 0.5 * SS + 1.0 / max(self._a_gamma, 1e-300))
            self.s_gamma = math.sqrt(max(Q_gamma, 0.0))
            self._a_gamma = self._rinvgamma(self.rng, 1.0, (1.0 / (A * A)) + 1.0 / max(Q_gamma, 1e-300))

    @staticmethod
    def _gibbs_m0_scalar(rng: np.random.Generator, x0: float, m_prior: float, s_prior: float, P0: float) -> float:
        prec = 1.0 / (s_prior**2) + 1.0 / max(P0, 1e-18)
        var = 1.0 / prec
        mean = var * (m_prior / (s_prior**2) + x0 / max(P0, 1e-18))
        return float(rng.normal(mean, math.sqrt(var)))

    def update_m0(self) -> None:
        if self._layout.dim == 0:
            return
        pos = 0
        if self._layout.idx_alpha is not None:
            self.m0_alpha = self._gibbs_m0_scalar(
                self.rng, float(self.x[0, pos]), self.priors.m_m0_alpha, self.priors.s_m0_alpha, self.P0_alpha
            )
            pos += 1
        if self._layout.idx_beta is not None:
            self.m0_beta = self._gibbs_m0_scalar(
                self.rng, float(self.x[0, pos]), self.priors.m_m0_beta, self.priors.s_m0_beta, self.P0_beta
            )
            pos += 1
        if self.seasonal_mode == "dynamic":
            base = (
                np.zeros(self.period - 1)
                if self.priors.m_m0_gamma is None
                else np.asarray(self.priors.m_m0_gamma, float)
            )
            s = float(self.priors.s_m0_gamma)
            for k in range(self.period - 1):
                self.m0_gamma[k] = self._gibbs_m0_scalar(
                    self.rng, float(self.x[0, pos + k]), float(base[k]), s, self.P0_gamma
                )

    def update_P0(self) -> None:
        if self._layout.dim == 0:
            return
        pos = 0
        if self._layout.idx_alpha is not None:
            a = self.priors.a_P0_alpha + 0.5
            b = self.priors.b_P0_alpha + 0.5 * (float(self.x[0, pos]) - self.m0_alpha) ** 2
            self.P0_alpha = 1.0 / self.rng.gamma(a, 1.0 / b)
            pos += 1
        if self._layout.idx_beta is not None:
            a = self.priors.a_P0_beta + 0.5
            b = self.priors.b_P0_beta + 0.5 * (float(self.x[0, pos]) - self.m0_beta) ** 2
            self.P0_beta = 1.0 / self.rng.gamma(a, 1.0 / b)
            pos += 1
        if self.seasonal_mode == "dynamic":
            diffsq = 0.0
            for k in range(self.period - 1):
                diffsq += (float(self.x[0, pos + k]) - float(self.m0_gamma[k])) ** 2
            a = self.priors.a_P0_gamma + 0.5 * (self.period - 1)
            b = self.priors.b_P0_gamma + 0.5 * diffsq
            self.P0_gamma = 1.0 / self.rng.gamma(a, 1.0 / b)

    def update_deterministic_params(self) -> None:
        sig2 = float(self.sigma2)

        # LEVEL: m0_alpha
        if self.level_mode == "deterministic":
            r = self.y.copy()
            if self._layout.dim > 0:
                H = self._layout.H()
                for t in range(1, self.T + 1):
                    r[t - 1] -= float(H @ self.x[t])
            if (self.trend_mode == "deterministic") and (self._layout.idx_alpha is None):
                r -= float(self.m0_beta) * np.arange(self.T, dtype=float)
            if self.seasonal_mode == "deterministic":
                if getattr(self, "_m0_gamma_full", None) is None:
                    mg = np.asarray(self.m0_gamma, float)
                    self._m0_gamma_full = np.r_[mg, -mg.sum()]
                r -= self._m0_gamma_full[np.arange(self.T) % self.period]
            m0, s0 = float(self.priors.m_m0_alpha), float(self.priors.s_m0_alpha)
            prec = self.T / sig2 + 1.0 / (s0 * s0)
            mean = ((r.sum() / sig2) + m0 / (s0 * s0)) / prec
            self.m0_alpha = float(self.rng.normal(mean, math.sqrt(1.0 / prec)))

        # TREND: m0_beta
        if self.trend_mode == "deterministic":
            if self._layout.idx_alpha is not None:
                d = self.x[1:, self._layout.idx_alpha] - self.x[:-1, self._layout.idx_alpha]
                q = float(self.s_alpha**2) if self.s_alpha > 0 else 1e-12
                m0, s0 = float(self.priors.m_m0_beta), float(self.priors.s_m0_beta)
                prec = self.T / q + 1.0 / (s0 * s0)
                mean = ((float(np.sum(d)) / q) + m0 / (s0 * s0)) / prec
                self.m0_beta = float(self.rng.normal(mean, math.sqrt(1.0 / prec)))
            else:
                tvec = np.arange(self.T, dtype=float)
                r = self.y.copy()
                if self._layout.dim > 0:
                    H = self._layout.H()
                    for k in range(1, self.T + 1):
                        r[k - 1] -= float(H @ self.x[k])
                if self.level_mode == "deterministic":
                    r -= float(self.m0_alpha)
                if self.seasonal_mode == "deterministic":
                    if getattr(self, "_m0_gamma_full", None) is None:
                        mg = np.asarray(self.m0_gamma, float)
                        self._m0_gamma_full = np.r_[mg, -mg.sum()]
                    r -= self._m0_gamma_full[np.arange(self.T) % self.period]
                m0, s0 = float(self.priors.m_m0_beta), float(self.priors.s_m0_beta)
                prec = (tvec @ tvec) / sig2 + 1.0 / (s0 * s0)
                mean = ((tvec @ r) / sig2 + m0 / (s0 * s0)) / prec
                self.m0_beta = float(self.rng.normal(mean, math.sqrt(1.0 / prec)))

        # SEASON (deterministic): m0_gamma (K = p-1 with sum-to-zero)
        if self.seasonal_mode == "deterministic":
            K = self.period - 1
            midx = np.arange(self.T) % self.period
            Z = np.zeros((self.T, K))
            for k in range(K):
                Z[:, k] = (midx == k).astype(float) - (midx == K).astype(float)
            r = self.y.copy()
            if self._layout.dim > 0:
                H = self._layout.H()
                for t in range(1, self.T + 1):
                    r[t - 1] -= float(H @ self.x[t])
            if self.level_mode == "deterministic":
                r -= float(self.m0_alpha)
            if (self._layout.idx_alpha is None) and (self.trend_mode == "deterministic"):
                r -= float(self.m0_beta) * np.arange(self.T, dtype=float)
            mu_prior = (
                np.zeros(K)
                if self.priors.m_m0_gamma is None
                else np.asarray(self.priors.m_m0_gamma, float).reshape(-1)
            )
            s2p = float(self.priors.s_m0_gamma) ** 2
            Prec = (Z.T @ Z) / sig2 + np.eye(K) / s2p
            b = (Z.T @ r) / sig2 + mu_prior / s2p
            L = np.linalg.cholesky(Prec)
            mu = np.linalg.solve(L.T, np.linalg.solve(L, b))
            theta = mu + np.linalg.solve(L.T, self.rng.standard_normal(K))
            self.m0_gamma = theta
            self._m0_gamma_full = np.r_[theta, -theta.sum()]

    # ------------------------------- RJ moves ------------------------------- #

    def _snapshot(self) -> dict:
        return {
            "level": self.level_mode,
            "trend": self.trend_mode,
            "season": self.seasonal_mode,
            "sigma2": self.sigma2,
            "s_alpha": self.s_alpha,
            "s_beta": self.s_beta,
            "s_gamma": self.s_gamma,
            "a_alpha": self._a_alpha,
            "a_beta": self._a_beta,
            "a_gamma": self._a_gamma,
            "m0_alpha": self.m0_alpha,
            "P0_alpha": self.P0_alpha,
            "m0_beta": self.m0_beta,
            "P0_beta": self.P0_beta,
            "m0_gamma": None if self.m0_gamma is None else self.m0_gamma.copy(),
            "P0_gamma": self.P0_gamma,
        }

    def _load_snapshot(self, S: dict) -> None:
        self.level_mode = S["level"]
        self.trend_mode = S["trend"]
        self.seasonal_mode = S["season"]
        self.sigma2 = float(S["sigma2"])
        self.s_alpha = float(S["s_alpha"])  # 0 if non-dynamic afterwards
        self.s_beta = float(S["s_beta"])   # 0 if non-dynamic afterwards
        self.s_gamma = float(S["s_gamma"])  # 0 if non-dynamic afterwards
        self._a_alpha = float(S["a_alpha"])
        self._a_beta = float(S["a_beta"])
        self._a_gamma = float(S["a_gamma"])
        self.m0_alpha = float(S["m0_alpha"])
        self.P0_alpha = float(S["P0_alpha"])
        self.m0_beta = float(S["m0_beta"])
        self.P0_beta = float(S["P0_beta"])
        self.m0_gamma = None if S["m0_gamma"] is None else np.asarray(S["m0_gamma"], float).copy()
        self.P0_gamma = float(S["P0_gamma"])
        if self.seasonal_mode == "deterministic":
            mg = np.asarray(self.m0_gamma, float)
            self._m0_gamma_full = np.r_[mg, -mg.sum()]
        else:
            self._m0_gamma_full = None
        self._layout = _Layout(self.period, self.level_mode, self.trend_mode, self.seasonal_mode)

    def _rj_record(self, block: str, accepted: bool) -> None:
        s = self.rj_stats[block]
        s["proposed"] += 1
        if accepted:
            s["accepted"] += 1
        s["win"].append(1 if accepted else 0)

    def _fmt_rj_block(self, block: str) -> str:
        s = self.rj_stats[block]
        prop = max(1, int(s["proposed"]))
        acc = int(s["accepted"]) 
        return f"{100.0 * acc / prop:.1f}%"

    def _fmt_rj_all(self) -> str:
        return f"{self._fmt_rj_block('level')}|{self._fmt_rj_block('trend')}|{self._fmt_rj_block('season')}"

    def _propose_mode(self, comp: str, cur: str) -> str:
        if comp == "level":
            cand = ["dynamic", "deterministic"] + (["none"] if self.cfg.allow_none_level else [])
        elif comp == "trend":
            cand = ["dynamic", "deterministic"] + (["none"] if self.cfg.allow_none_trend else [])
        else:
            cand = ["dynamic", "deterministic"] + (["none"] if self.cfg.allow_none_season else [])
        choices = [z for z in cand if z != cur]
        return self.rng.choice(choices)

    def _legal_modes(self, level: str, trend: str, season: str) -> bool:
        if trend == "dynamic" and level != "dynamic":
            return False
        if (not self.cfg.allow_none_level) and level == "none":
            return False
        return True

    def _draw_prior_dyn_block(self, which: str) -> None:
        # Half-Cauchy via IG mixture: a ~ InvGamma(1, 1/A^2), Q ~ InvGamma(1/2, 1/a)
        if which == "level":
            A = float(self.priors.hc_scale_alpha)
            a_aux = self._rinvgamma(self.rng, 1.0, 1.0 / (A * A))
            Q = self._rinvgamma(self.rng, 0.5, 1.0 / max(a_aux, 1e-300))
            self._a_alpha = a_aux
            self.s_alpha = math.sqrt(max(Q, 1e-18))
            self.m0_alpha = float(self.rng.normal(self.priors.m_m0_alpha, self.priors.s_m0_alpha))
            self.P0_alpha = 1.0 / self.rng.gamma(self.priors.a_P0_alpha, 1.0 / self.priors.b_P0_alpha)
        elif which == "trend":
            A = float(self.priors.hc_scale_beta)
            a_aux = self._rinvgamma(self.rng, 1.0, 1.0 / (A * A))
            Q = self._rinvgamma(self.rng, 0.5, 1.0 / max(a_aux, 1e-300))
            self._a_beta = a_aux
            self.s_beta = math.sqrt(max(Q, 1e-18))
            self.m0_beta = float(self.rng.normal(self.priors.m_m0_beta, self.priors.s_m0_beta))
            self.P0_beta = 1.0 / self.rng.gamma(self.priors.a_P0_beta, 1.0 / self.priors.b_P0_beta)
        elif which == "season":
            A = float(self.priors.hc_scale_gamma)
            a_aux = self._rinvgamma(self.rng, 1.0, 1.0 / (A * A))
            Q = self._rinvgamma(self.rng, 0.5, 1.0 / max(a_aux, 1e-300))
            self._a_gamma = a_aux
            self.s_gamma = math.sqrt(max(Q, 1e-18))
            base = (
                np.zeros(self.period - 1)
                if self.priors.m_m0_gamma is None
                else np.asarray(self.priors.m_m0_gamma, float)
            )
            s = float(self.priors.s_m0_gamma)
            self.m0_gamma = self.rng.normal(base, s, size=self.period - 1)
            self.P0_gamma = 1.0 / self.rng.gamma(self.priors.a_P0_gamma, 1.0 / self.priors.b_P0_gamma)
            self._m0_gamma_full = None

    def _ensure_det_season_defaults(self) -> None:
        if self.seasonal_mode != "deterministic":
            return
        if self.m0_gamma is None or self.m0_gamma.size != (self.period - 1):
            base = (
                np.zeros(self.period - 1)
                if self.priors.m_m0_gamma is None
                else np.asarray(self.priors.m_m0_gamma, float)
            )
            self.m0_gamma = base.astype(float)
        self._m0_gamma_full = np.r_[self.m0_gamma, -float(np.sum(self.m0_gamma))]

    def _rj_move_one(self) -> None:
        comps = ["level", "trend", "season"]
        comp = self.rng.choice(comps)

        level, trend, season = self.level_mode, self.trend_mode, self.seasonal_mode
        cur = {"level": level, "trend": trend, "season": season}[comp]
        prop = self._propose_mode(comp, cur)

        new_level, new_trend, new_season = level, trend, season
        if comp == "level":
            new_level = prop
            if new_trend == "dynamic" and new_level != "dynamic":
                new_trend = "deterministic"
        elif comp == "trend":
            new_trend = prop
            if new_trend == "dynamic" and new_level != "dynamic":
                self._rj_record(comp, False)
                return
        else:
            new_season = prop

        if not self._legal_modes(new_level, new_trend, new_season):
            self._rj_record(comp, False)
            return

        cur_snap = self._snapshot()

        # births: draw proposed dynamic block params
        if comp == "level" and new_level == "dynamic" and level != "dynamic":
            self._draw_prior_dyn_block("level")
        if comp == "trend" and new_trend == "dynamic" and trend != "dynamic":
            self._draw_prior_dyn_block("trend")
        if comp == "season" and new_season == "dynamic" and season != "dynamic":
            self._draw_prior_dyn_block("season")

        if comp == "season" and new_season == "deterministic":
            base = (
                np.zeros(self.period - 1)
                if self.priors.m_m0_gamma is None
                else np.asarray(self.priors.m_m0_gamma, float)
            )
            self.m0_gamma = base.astype(float)
            self._m0_gamma_full = np.r_[self.m0_gamma, -float(np.sum(self.m0_gamma))]
            self.P0_gamma = 0.0

        # apply proposed modes
        self.level_mode, self.trend_mode, self.seasonal_mode = new_level, new_trend, new_season
        self._layout = _Layout(self.period, self.level_mode, self.trend_mode, self.seasonal_mode)
        if self.level_mode != "dynamic":
            self.s_alpha = 0.0
        if self.trend_mode != "dynamic":
            self.s_beta = 0.0
        if self.seasonal_mode != "dynamic":
            self.s_gamma = 0.0
        if self.seasonal_mode == "deterministic":
            self._ensure_det_season_defaults()
        else:
            self._m0_gamma_full = None

        # proposed log posterior
        logpost_prop = self._kalman_loglik() + self._log_prior_current()

        # current log posterior
        self._load_snapshot(cur_snap)
        logpost_cur = self._kalman_loglik() + self._log_prior_current()

        dlogpost = logpost_prop - logpost_cur
        accept = (self.rng.uniform() < math.exp(min(0.0, -(-dlogpost)))) if dlogpost < 0 else True
        self._rj_record(comp, bool(accept))

        if accept:
            # adopt proposal
            self._load_snapshot(cur_snap)
            before = (self.level_mode, self.trend_mode, self.seasonal_mode)

            self.level_mode, self.trend_mode, self.seasonal_mode = new_level, new_trend, new_season
            if self.level_mode != "dynamic":
                self.s_alpha = 0.0
            if self.trend_mode != "dynamic":
                self.s_beta = 0.0
            if self.seasonal_mode != "dynamic":
                self.s_gamma = 0.0
            if self.seasonal_mode == "deterministic":
                self._ensure_det_season_defaults()
            else:
                self._m0_gamma_full = None

            self._layout = _Layout(self.period, self.level_mode, self.trend_mode, self.seasonal_mode)
            self._alloc_state_holder()

            after = (self.level_mode, self.trend_mode, self.seasonal_mode)
            print(
                f"[switch] block={comp} | "
                f"L:{before[0][:3]}→{after[0][:3]} T:{before[1][:3]}→{after[1][:3]} S:{before[2][:3]}→{after[2][:3]} | "
                f"Δlogpost={dlogpost:+.4f} | RJ {self._fmt_rj_all()}"
            )

    # ------------------------------ bookkeeping ---------------------------- #

    def _tally_modes(self) -> None:
        self._mode_counts["level"][self.level_mode] += 1
        self._mode_counts["trend"][self.trend_mode] += 1
        self._mode_counts["season"][self.seasonal_mode] += 1

    def inclusion_probabilities(self) -> Dict[str, Dict[str, float]]:
        total = sum(self._mode_counts["level"].values())
        if total == 0:
            return {k: {m: 0.0 for m in v} for k, v in self._mode_counts.items()}
        out = {}
        for comp, cnt in self._mode_counts.items():
            out[comp] = {m: c / total for m, c in cnt.items()}
        return out

    # ------------------------------- progress ------------------------------- #

    def _progress_line(self, it: int) -> str:
        parts = [
            f"[it {it + 1}/{self.cfg.n_iter}]",
            f"σ={math.sqrt(self.sigma2):.3f}",
            f"L/T/S={self.level_mode[:3]}/{self.trend_mode[:3]}/{self.seasonal_mode[:3]}",
        ]
        if self._layout.idx_alpha is not None:
            parts.append(f"Qα={self.s_alpha**2:.4g}")
        if self._layout.idx_beta is not None:
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
            parts.append(f"m0γ={_fmt_list(self.m0_gamma)} P0γ={self.P0_gamma:.4g}")
        elif self.seasonal_mode == "deterministic":
            parts.append(f"m0γ(det)={_fmt_list(self.m0_gamma)}")
        parts.append(f"RJ {self._fmt_rj_all()}")
        return " | ".join(parts)

    # ---------------------------------- run --------------------------------- #

    def run(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        save_iters = list(range(cfg.burn, cfg.n_iter, cfg.thin))
        n_kept, keep_idx = len(save_iters), 0

        self.keep = {
            "sigma": np.zeros(n_kept, float),
            "mu": np.zeros((n_kept, self.T), float),
            "modes": np.zeros((n_kept, 3), int),  # 0:dyn,1:det,2:none
        }
        x_width = max(self._layout.dim, 1)
        self.keep["x"] = np.zeros((n_kept, self.T, x_width))

        if self._layout.idx_alpha is not None:
            self.keep.update({"Q_alpha": np.zeros(n_kept), "m0_alpha": np.zeros(n_kept), "P0_alpha": np.zeros(n_kept)})
        if self._layout.idx_beta is not None:
            self.keep.update({"Q_beta": np.zeros(n_kept), "m0_beta": np.zeros(n_kept), "P0_beta": np.zeros(n_kept)})
        if self.seasonal_mode == "dynamic":
            self.keep.update(
                {"Q_gamma": np.zeros(n_kept), "m0_gamma": np.zeros((n_kept, self.period - 1)), "P0_gamma": np.zeros(n_kept)}
            )

        print_every = cfg.progress_every if cfg.progress_every > 0 else max(1, cfg.n_iter // 50) or 1

        for it in range(cfg.n_iter):
            if self._layout.dim > 0:
                self.x = self._ffbs()
                self.update_process_Q_halfcauchy()
                self.update_m0()
                self.update_P0()
            self.update_deterministic_params()
            self.update_sigma2()

            for _ in range(cfg.rj_moves_per_iter):
                self._rj_move_one()

            self._tally_modes()

            if cfg.progress and ((it + 1) % print_every == 0 or it == cfg.n_iter - 1):
                print(self._progress_line(it))

            if it in save_iters:
                mu = self._mu_vec()
                self.keep["mu"][keep_idx, :] = mu
                self.keep["sigma"][keep_idx] = math.sqrt(self.sigma2)
                enc = lambda s: 0 if s == "dynamic" else (1 if s == "deterministic" else 2)
                self.keep["modes"][keep_idx, :] = np.array(
                    [enc(self.level_mode), enc(self.trend_mode), enc(self.seasonal_mode)], int
                )
                if self._layout.idx_alpha is not None and "Q_alpha" in self.keep:
                    self.keep["Q_alpha"][keep_idx] = self.s_alpha**2
                    self.keep["m0_alpha"][keep_idx] = self.m0_alpha
                    self.keep["P0_alpha"][keep_idx] = self.P0_alpha
                if self._layout.idx_beta is not None and "Q_beta" in self.keep:
                    self.keep["Q_beta"][keep_idx] = self.s_beta**2
                    self.keep["m0_beta"][keep_idx] = self.m0_beta
                    self.keep["P0_beta"][keep_idx] = self.P0_beta
                if self.seasonal_mode == "dynamic" and "Q_gamma" in self.keep:
                    self.keep["Q_gamma"][keep_idx] = self.s_gamma**2
                    self.keep["m0_gamma"][keep_idx, :] = self.m0_gamma
                    self.keep["P0_gamma"][keep_idx] = self.P0_gamma
                if self._layout.dim > 0:
                    w = min(self.keep["x"].shape[2], self.x.shape[1])
                    self.keep["x"][keep_idx, :, :w] = self.x[1 : self.T + 1, :w]
                keep_idx += 1

        return self.keep

    # ----------------------------------- I/O -------------------------------- #

    def save_posterior(self, out_npz_path: str, extra_meta: Optional[dict] = None) -> None:
        os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)
        arrays = dict(self.keep)
        arrays["y"] = self.y.copy()
        np.savez_compressed(out_npz_path, **arrays)
        meta = {
            "T": int(self.T),
            "period": int(self.period),
            "cfg": asdict(self.cfg),
            "priors": asdict(self.priors),
            "model_prior": self.model_prior,
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
        }
        if extra_meta:
            meta.update(extra_meta)
        with open(out_npz_path.replace(".npz", ".meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


# =============================================================================
# CLI / Example run
# =============================================================================

# We keep the CLI surface compatible with the original for easy swapping.

def _parse_date(s: str | None):
    from datetime import datetime
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


def _csv_model_prior_block(s: str | None, allow_none: bool, defaults: dict) -> dict:
    out = dict(defaults)
    if s:
        pieces = [p.strip() for p in s.split(",") if p.strip()]
        for p in pieces:
            k, v = p.split(":")
            out[k.strip()] = float(v)
    if not allow_none:
        out["none"] = min(out.get("none", 1e-12), 1e-12)
    ssum = sum(out.values())
    if ssum <= 0:
        dsum = sum(defaults.values())
        out = {k: v / dsum for k, v in defaults.items()}
    else:
        out = {k: v / ssum for k, v in out.items()}
    return out


   


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(base_dir)
    from simulator.mean_time_series import Mean_Time_Series  # newest-first convention


    p = argparse.ArgumentParser(
        description=(
            "Gaussian structural TS (level/trend/season) with RJ-MCMC (Δ log posterior) + conjugate Gibbs.\n"
            "If available, data are simulated via simulator.mean_time_series.Mean_Time_Series.\n"
            "When a block is non-dynamic, its m0_* acts as the deterministic parameter."
        )
    )
    # Simulation (data-generating truth)
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--period", type=int, default=4)
    p.add_argument("--start-date", type=str, default="2000-01-01")
    p.add_argument("--level-mode", choices=["dynamic", "deterministic"], default="dynamic")
    p.add_argument("--trend-mode", choices=["dynamic", "deterministic", "none"], default="dynamic")
    p.add_argument("--seasonal-mode", choices=["dynamic", "deterministic", "none"], default="dynamic")
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--q-level", type=float, default=0.001)
    p.add_argument("--q-trend", type=float, default=0.00002)
    p.add_argument("--q-season", type=float, default=0.005)
    p.add_argument("--m0-level", type=float, default=3.0)
    p.add_argument("--v0-level", type=float, default=0.05)
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

    # Half-Cauchy scales
    p.add_argument("--hc-scale-alpha", type=float, default=0.5)
    p.add_argument("--hc-scale-beta", type=float, default=0.5)
    p.add_argument("--hc-scale-gamma", type=float, default=0.5)

    # Model priors (Occam tilt for RJ)
    p.add_argument("--prior-model-level", type=str, default=None)
    p.add_argument("--prior-model-trend", type=str, default=None)
    p.add_argument("--prior-model-season", type=str, default=None)

    # Sampler configuration
    p.add_argument("--n-iter", type=int, default=30000)
    p.add_argument("--burn", type=int, default=10000)
    p.add_argument("--thin", type=int, default=2)
    p.add_argument("--seed", type=int, default=40)
    p.add_argument("--progress", default=True)
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--out-dir", type=str, default="results/simulations/DLM_RJ")
    p.add_argument("--plot", default=True)
    p.add_argument("--print-summary", default=True)

    # Initial inference values
    p.add_argument("--sigma-init", type=float, default=1.0)
    p.add_argument("--s-alpha-init", type=float, default=1)
    p.add_argument("--s-beta-init", type=float, default=1)
    p.add_argument("--s-gamma-init", type=float, default=1)
    p.add_argument("--m0-gamma-init", type=str, default=None)
    p.add_argument("--P0-alpha-init", type=float, default=0.05)
    p.add_argument("--P0-beta-init", type=float, default=0.05)
    p.add_argument("--P0-gamma-init", type=float, default=0.05)

    # RJ options
    p.add_argument("--rj-moves-per-iter", type=int, default=2)
    p.add_argument("--allow-none-level", default=False)
    p.add_argument("--allow-none-trend", default=False)
    p.add_argument("--allow-none-season", default=False)

    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    # Simulate data if simulator present
    if Mean_Time_Series is not None:
        start_date = _parse_date(args.start_date)
        m0_season = _csv_floats_or_none(args.m0_season) or [1] * (args.period - 1)
        v0_season = _csv_floats_or_none(args.v0_season) or [0.05] * (args.period - 1)
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
        mu_T = truths["mu_t"][1 : 1 + args.T]
        dates_T = truths["index"][: args.T]
    else:
        # Fallback: AR(1)-ish synthetic series with seasonality
        t = np.arange(args.T)
        seas = np.sin(2 * np.pi * t / max(2, args.period))
        y = 0.1 * t + 2 * seas + rng.normal(0, args.sigma, size=args.T)
        mu_T = 0.1 * t + 2 * seas
        dates_T = np.arange(args.T)

    # Priors
    pri_gamma_vec = _csv_floats_or_none(args.prior_m_m0_gamma)
    priors = Priors(
        a_sigma=args.prior_a_sigma,
        b_sigma=args.prior_b_sigma,
        m_m0_alpha=args.prior_m_m0_alpha,
        s_m0_alpha=args.prior_s_m0_alpha,
        m_m0_beta=args.prior_m_m0_beta,
        s_m0_beta=args.prior_s_m0_beta,
        m_m0_gamma=None if pri_gamma_vec is None else pri_gamma_vec,
        s_m0_gamma=args.prior_s_m0_gamma,
        a_P0_alpha=args.prior_a_P0_alpha,
        b_P0_alpha=args.prior_b_P0_alpha,
        a_P0_beta=args.prior_a_P0_beta,
        b_P0_beta=args.prior_b_P0_beta,
        a_P0_gamma=args.prior_a_P0_gamma,
        b_P0_gamma=args.prior_b_P0_gamma,
        hc_scale_alpha=args.hc_scale_alpha,
        hc_scale_beta=args.hc_scale_beta,
        hc_scale_gamma=args.hc_scale_gamma,
    )

    # Model prior (for Δ log posterior)
    default_model_prior = {
        "level": {"dynamic": 0.5, "deterministic": 0.5, "none": 1e-12},
        "trend": {"dynamic": 0.5, "deterministic": 0.5, "none": 0.4 if args.allow_none_trend else 1e-12},
        "season": {"dynamic": 0.5, "deterministic": 0.5, "none": 0.2 if args.allow_none_season else 1e-12},
    }
    model_prior = {
        "level": _csv_model_prior_block(args.prior_model_level, args.allow_none_level, default_model_prior["level"]),
        "trend": _csv_model_prior_block(args.prior_model_trend, args.allow_none_trend, default_model_prior["trend"]),
        "season": _csv_model_prior_block(args.prior_model_season, args.allow_none_season, default_model_prior["season"]),
    }

    cfg = SamplerConfig(
        n_iter=int(args.n_iter),
        burn=int(args.burn),
        thin=int(args.thin),
        random_seed=int(args.seed),
        progress=bool(args.progress),
        progress_every=int(args.progress_every),
        rj_moves_per_iter=int(args.rj_moves_per_iter),
        allow_none_level=bool(args.allow_none_level),
        allow_none_trend=bool(args.allow_none_trend),
        allow_none_season=bool(args.allow_none_season),
    )

    m0_gamma_init = _csv_floats_or_none(args.m0_gamma_init)

    sampler = DLMRJGibbs(
        y=y,
        period=args.period,
        level_mode="deterministic",
        trend_mode="deterministic",
        seasonal_mode="deterministic",
        sigma2_init=args.sigma_init**2,
        s_alpha_init=args.s_alpha_init,
        s_beta_init=args.s_beta_init,
        s_gamma_init=args.s_gamma_init,
        m0_alpha_init=args.m0_level,
        m0_beta_init=args.m0_trend,
        m0_gamma_init=m0_gamma_init,
        P0_alpha_init=args.P0_alpha_init,
        P0_beta_init=args.P0_beta_init,
        P0_gamma_init=args.P0_gamma_init,
        priors=priors,
        cfg=cfg,
        model_prior=model_prior,
        rng=rng,
    )

    if args.print_summary:
        if Mean_Time_Series is not None:
            print(
                f"\nSimulated {args.T} observations (σ={args.sigma}) with TRUE modes "
                f"{args.level_mode}/{args.trend_mode}/{args.seasonal_mode}."
            )
        else:
            print(f"\nSimulated fallback synthetic series of length {args.T}.")
        print("Sampler START modes: dynamic/dynamic/dynamic\n")
        print("Half-Cauchy scales (A_k) for process SDs:")
        print(
            f"  A_alpha={priors.hc_scale_alpha:.3f}, A_beta={priors.hc_scale_beta:.3f}, "
            f"A_gamma={priors.hc_scale_gamma:.3f}"
        )
        print("\nModel priors (normalized):")
        for k in ("level", "trend", "season"):
            mp = model_prior[k]
            print(f"  {k:6s}: dyn={mp['dynamic']:.3f}, det={mp['deterministic']:.3f}, none={mp['none']:.3f}")
        print(
            "\nConvention: when a block is non-dynamic, its m0_* acts as the deterministic parameter."
        )

    t0 = time.time()
    post = sampler.run()
    elapsed = time.time() - t0
    print(f"\n[Run completed in {elapsed:.1f}s]")

    out_dir = os.path.join(
        args.out_dir,
        f"CC_{args.level_mode}-{args.trend_mode}-{args.seasonal_mode}__"
        f"{time.strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(out_dir, exist_ok=True)

    sampler.save_posterior(
        out_npz_path=os.path.join(out_dir, "posterior.npz"),
        extra_meta={
            "elapsed_seconds": float(elapsed),
            "sim_truth_modes": {
                "level": args.level_mode,
                "trend": args.trend_mode,
                "season": args.seasonal_mode,
            },
            "start_modes": {"level": "dynamic", "trend": "dynamic", "season": "dynamic"},
            "hc_scales": {
                "alpha": priors.hc_scale_alpha,
                "beta": priors.hc_scale_beta,
                "gamma": priors.hc_scale_gamma,
            },
            "model_prior": model_prior,
        },
    )

    if args.plot:
        mu_hat = post["mu"].mean(axis=0)
        plt.figure(figsize=(10, 4))
        plt.plot(dates_T, y, label="y_t", lw=1)
        plt.plot(dates_T, mu_T, "--", label="μ_t (truth)")
        plt.plot(dates_T, mu_hat, "-.", label="μ̂_t (post mean)")
        plt.title(
            "DLM RJ (Δ log posterior) — current modes: "
            f"{sampler.level_mode}/{sampler.trend_mode}/{sampler.seasonal_mode}"
        )
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fit.png"), dpi=160)
        plt.show()

    inc = sampler.inclusion_probabilities()
    with open(os.path.join(out_dir, "inclusion_probs.json"), "w", encoding="utf-8") as f:
        json.dump(inc, f, indent=2)

    rj_dump = {
        b: {
            "proposed": int(sampler.rj_stats[b]["proposed"]),
            "accepted": int(sampler.rj_stats[b]["accepted"]),
            "acc_rate": (sampler.rj_stats[b]["accepted"] / max(1, sampler.rj_stats[b]["proposed"]))
        }
        for b in ("level", "trend", "season")
    }
    with open(os.path.join(out_dir, "rj_stats.json"), "w", encoding="utf-8") as f:
        json.dump(rj_dump, f, indent=2)

    print(f"[save] Outputs written to: {out_dir}")
