from __future__ import annotations

"""
Numerically safe Gaussian structural DLM with RJ–MCMC + conjugate Gibbs
+ optional ASIS interweaving (non-centered) for process variances.

This rewrite hardens the original implementation against the observed
"variance blow-up + RJ ping-pong" failure mode by adding:
  • Guarded log-posterior (reject if non-finite on either side)
  • RJ warmup gate (delay model switches until chains stabilize)
  • Floors/caps for variances (sigma^2 and Q's)
  • Hardened SPD sampling and Kalman math
  • Safer seasonal innovation SS and birth initializations
  • Larger innovation-variance floor in Kalman S during early iterations
  • Tamed progress printing (clamped Δlogpost)

CLI remains compatible with the previous version. See __main__ block.
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

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

# =========================
# Numerics & small helpers
# =========================
EPS: float = 1e-12       # generic positive jitter
TINY: float = 1e-18      # strict floor for scale/variance
HUGE: float = 1e18       # hard cap for large magnitudes
GAMMA_SCALE_CAP: float = 1e12
SIG2_MIN, SIG2_MAX = 1e-6, 1e6
Q_MIN, Q_MAX = 1e-10, 1e6


def _pos(x: float, floor: float = TINY) -> float:
    x = float(x)
    if not np.isfinite(x) or x <= floor:
        return floor
    return x


def _cap(x: float, hi: float = HUGE) -> float:
    x = float(x)
    if not np.isfinite(x):
        return hi
    return min(x, hi)


def _mad(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    if v.size == 0:
        return 0.0
    m = np.median(v)
    return float(np.median(np.abs(v - m)))


def _robust_sd(v: np.ndarray) -> float:
    return _mad(np.asarray(v, float)) / 1.4826 if v.size else 0.0


def _spd_solve(M: np.ndarray, B: np.ndarray, jitter: float = EPS) -> np.ndarray:
    """Solve M X = B for SPD-like M with jitter escalation; fallback to pinv."""
    n = M.shape[0]
    I = np.eye(n)
    S = 0.5 * (M + M.T)
    for k in range(6):
        try:
            L = np.linalg.cholesky(S + (10.0 ** k) * jitter * I)
            Y = np.linalg.solve(L, B)
            return np.linalg.solve(L.T, Y)
        except np.linalg.LinAlgError:
            pass
    return np.linalg.pinv(M) @ B


def _mvnorm(rng: np.random.Generator, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    cov = 0.5 * (cov + cov.T)
    mine = float(np.linalg.eigvalsh(cov).min())
    if not np.isfinite(mine):
        cov = cov + 1e-8 * np.eye(cov.shape[0])
        mine = float(np.linalg.eigvalsh(cov).min())
    if mine < 1e-10:
        cov = cov + (1e-10 - mine) * np.eye(cov.shape[0])
    return rng.multivariate_normal(mean, cov)


# =========================
# Priors & configuration
# =========================
@dataclass
class Priors:
    a_sigma: float = 2.0
    b_sigma: float = 1.0
    m_m0_alpha: float = 0.0
    s_m0_alpha: float = 10.0
    m_m0_beta: float = 0.0
    s_m0_beta: float = 10.0
    m_m0_gamma: Optional[Sequence[float]] = None
    s_m0_gamma: float = 5.0
    a_P0_alpha: float = 2.0
    b_P0_alpha: float = 1.0
    a_P0_beta: float = 2.0
    b_P0_beta: float = 1.0
    a_P0_gamma: float = 2.0
    b_P0_gamma: float = 1.0
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
    progress_every: int = 0
    # RJ
    rj_moves_per_iter: int = 2
    allow_none_level: bool = False
    allow_none_trend: bool = True
    allow_none_season: bool = True
    rj_warmup: int = 1000  # NEW: gate RJ until after warmup
    # ASIS
    asis: bool = False


# =========================
# Model layout (matrices)
# =========================
class _Layout:
    """Constructs time-invariant DLM matrices for a chosen mode triple."""

    def __init__(self, period: int, level: str, trend: str, season: str):
        ok = {"dynamic", "deterministic", "none"}
        if level not in ok or trend not in ok or season not in ok:
            raise ValueError("invalid mode")
        if trend == "dynamic" and level != "dynamic":
            raise ValueError("trend=dynamic requires level=dynamic")
        self.period = int(period)
        self.level_mode, self.trend_mode, self.season_mode = level, trend, season
        # indices
        labels: List[str] = []
        self.idx_alpha = self.idx_beta = None
        self.idx_g_start = self.idx_g_end = None
        if level == "dynamic":
            self.idx_alpha = len(labels); labels.append("alpha")
        if trend == "dynamic":
            self.idx_beta = len(labels); labels.append("beta")
        if season == "dynamic":
            for k in range(1, period):
                labels.append(f"g{k}")
            if period > 1:
                self.idx_g_start = labels.index("g1")
                self.idx_g_end = self.idx_g_start + (period - 2)
        self._labels = labels
        self.dim = len(labels)
        # cache matrices
        self._H = self._build_H()
        self._A = self._build_A()

    def _build_H(self) -> np.ndarray:
        if self.dim == 0:
            return np.zeros((1, 0))
        h = np.zeros(self.dim)
        if self.idx_alpha is not None:
            h[self.idx_alpha] = 1.0
        if self.season_mode == "dynamic":
            h[self.idx_g_start] = 1.0
        return h.reshape(1, -1)

    def _build_A(self) -> np.ndarray:
        if self.dim == 0:
            return np.zeros((0, 0))
        A = np.eye(self.dim)
        # local linear trend coupling
        if (self.idx_alpha is not None) and (self.idx_beta is not None):
            A[self.idx_alpha, self.idx_beta] = 1.0
        # seasonal rotation (sum-to-zero newest-first)
        if self.season_mode == "dynamic":
            gs, ge = self.idx_g_start, self.idx_g_end
            K = ge - gs + 1
            A[gs, gs: ge + 1] = -1.0
            if K > 1:
                A[gs + 1: ge + 1, gs: ge] = np.eye(K - 1)
                A[gs + 1: ge + 1, ge] = 0.0
        return A

    # public accessors
    def H(self) -> np.ndarray: return self._H
    def A(self) -> np.ndarray: return self._A

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
            Q[self.idx_alpha, self.idx_alpha] = s_alpha ** 2
        if (self.idx_beta is not None) and (s_beta > 0):
            Q[self.idx_beta, self.idx_beta] = s_beta ** 2
        if (self.season_mode == "dynamic") and (s_gamma > 0):
            Q[self.idx_g_start, self.idx_g_start] = s_gamma ** 2
        return Q


# =========================
# Sampler
# =========================
class DLMRJGibbs:
    def __init__(
        self,
        y: np.ndarray,
        period: int,
        level_mode: str = "deterministic",
        trend_mode: str = "deterministic",
        seasonal_mode: str = "deterministic",
        sigma2_init: float = 1.0,
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
        self.y = np.asarray(y, float)
        self.T = int(self.y.size)
        self.period = int(period)
        self.priors, self.cfg = priors, cfg
        self.rng = rng or np.random.default_rng(cfg.random_seed)

        # modes & layout
        self.level_mode, self.trend_mode, self.seasonal_mode = level_mode, trend_mode, seasonal_mode
        self._layout = _Layout(self.period, level_mode, trend_mode, seasonal_mode)

        self.model_prior = model_prior or {
            "level": {"dynamic": 0.5, "deterministic": 0.5, "none": 1e-12},
            "trend": {"dynamic": 0.5, "deterministic": 0.5, "none": 0.4 if cfg.allow_none_trend else 1e-12},
            "season": {"dynamic": 0.3, "deterministic": 0.7, "none": 0.2 if cfg.allow_none_season else 1e-12},
        }

        # observation variance
        self.sigma2 = float(np.clip(sigma2_init, SIG2_MIN, SIG2_MAX))

        # process SDs and their IG-mixers
        self.s_alpha = float(s_alpha_init) if self._layout.idx_alpha is not None else 0.0
        self.s_beta = float(s_beta_init) if self._layout.idx_beta is not None else 0.0
        self.s_gamma = float(s_gamma_init) if self.seasonal_mode == "dynamic" else 0.0
        self._a_alpha = 1.0; self._a_beta = 1.0; self._a_gamma = 1.0

        # initial means/variances (act as deterministic parameters when block non-dynamic)
        self.m0_alpha = float(m0_alpha_init) if self._layout.idx_alpha is not None else float(self.priors.m_m0_alpha)
        self.P0_alpha = float(P0_alpha_init) if self._layout.idx_alpha is not None else 0.0
        self.m0_beta = float(m0_beta_init) if self._layout.idx_beta is not None else float(self.priors.m_m0_beta)
        self.P0_beta = float(P0_beta_init) if self._layout.idx_beta is not None else 0.0
        if self.seasonal_mode == "dynamic":
            if m0_gamma_init is None:
                self.m0_gamma = np.zeros(self.period - 1)
            else:
                m0g = np.asarray(m0_gamma_init, float)
                if m0g.size != self.period - 1:
                    raise ValueError("m0_gamma_init must have length p-1")
                self.m0_gamma = m0g
            self.P0_gamma = float(P0_gamma_init)
            self._m0_gamma_full = None
        else:
            base = np.zeros(self.period - 1) if self.priors.m_m0_gamma is None else np.asarray(self.priors.m_m0_gamma, float)
            self.m0_gamma = base.astype(float)
            self.P0_gamma = 0.0
            self._m0_gamma_full = np.r_[self.m0_gamma, -float(np.sum(self.m0_gamma))]

        # state storage
        self._alloc_state_holder()
        if self._layout.dim > 0:
            m0_vec, P0_diag = self._current_m0_P0()
            C0 = np.diag(P0_diag) + EPS * np.eye(self._layout.dim)
            self.x[0] = _mvnorm(self.rng, m0_vec, C0)
            self._seed_forward(Q_diag=np.full(self._layout.dim, 1e-6))

        # bookkeeping
        self.keep: Dict[str, np.ndarray] = {}
        self._mode_counts = {k: {m: 0 for m in ("dynamic", "deterministic", "none")} for k in ("level", "trend", "season")}
        self.rj_stats = {b: {"proposed": 0, "accepted": 0} for b in ("level", "trend", "season")}

        if self.cfg.progress:
            sd1 = _robust_sd(np.diff(self.y)) if self.T >= 2 else 0.0
            sd2 = _robust_sd(np.diff(self.y, n=2)) if self.T >= 3 else 0.0
            print(f"[init] modes L/T/S={self.level_mode[:3]}/{self.trend_mode[:3]}/{self.seasonal_mode[:3]} | sd1={sd1:.4g} sd2={sd2:.4g}")

    # ----- layout helpers -----
    def _alloc_state_holder(self) -> None:
        self.x = np.zeros((self.T + 1, self._layout.dim), float)

    def _max_state_dim(self) -> int:  # for saved array size
        return 2 + max(0, self.period - 1)

    def _current_m0_P0(self) -> Tuple[np.ndarray, np.ndarray]:
        m0, P0 = [], []
        if self._layout.idx_alpha is not None:
            m0.append(self.m0_alpha); P0.append(self.P0_alpha)
        if self._layout.idx_beta is not None:
            m0.append(self.m0_beta); P0.append(self.P0_beta)
        if self.seasonal_mode == "dynamic":
            m0.extend(list(self.m0_gamma)); P0.extend([self.P0_gamma] * (self.period - 1))
        return np.asarray(m0, float), np.asarray(P0, float)

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

    # =========================
    # Likelihood (Kalman) for RJ
    # =========================
    def _kalman_loglik(self, s_floor: float = EPS) -> float:
        H = self._layout.H(); A = self._layout.A()
        Q = self._layout.Q(self.s_alpha, self.s_beta, self.s_gamma)
        R = _pos(self.sigma2, EPS)
        if self._layout.dim == 0:
            e = np.array([self.y[t] - self._mu_det_t(t) for t in range(self.T)], float)
            return -0.5 * np.sum(np.log(2 * math.pi * R) + (e * e) / R)
        m0_vec, P0_diag = self._current_m0_P0()
        m = m0_vec.copy(); C = np.diag(P0_diag) + EPS * np.eye(self._layout.dim)
        ll = 0.0; u = self._layout.u(self.m0_beta)
        for t in range(self.T):
            a = A @ m + u
            Rm = A @ C @ A.T + Q
            Rm = 0.5 * (Rm + Rm.T) + EPS * np.eye(self._layout.dim)
            ydet = self._mu_det_t(t)
            S = float(H @ Rm @ H.T + R)
            S = max(S, s_floor)
            v = float(self.y[t] - ydet - H @ a)
            ll += -0.5 * (math.log(2 * math.pi) + math.log(S) + (v * v) / S)
            K = (Rm @ H.T) / S
            m = a + K.flatten() * v
            C = Rm - K @ (H @ Rm)
            C = 0.5 * (C + C.T) + EPS * np.eye(self._layout.dim)
        return float(ll)

    # =========================
    # Priors log-density
    # =========================
    def _log_prior_current(self) -> float:
        lp = 0.0
        def lIG(x, shape, scale):
            x = _pos(float(x)); return -((shape + 1.0) * math.log(x) + (scale / x))
        # sigma^2
        a, b = self.priors.a_sigma, self.priors.b_sigma
        tau = 1.0 / _pos(self.sigma2)
        lp += (a - 1.0) * math.log(tau) - b * tau
        # dynamic blocks
        if self._layout.idx_alpha is not None:
            Qa = _pos(self.s_alpha ** 2); aa = _pos(self._a_alpha)
            lp += lIG(Qa, 0.5, 1.0 / aa) + lIG(aa, 1.0, 1.0 / (self.priors.hc_scale_alpha ** 2))
            lp += -0.5 * ((self.m0_alpha - self.priors.m_m0_alpha) ** 2) / (self.priors.s_m0_alpha ** 2)
            lp += lIG(self.P0_alpha, self.priors.a_P0_alpha, self.priors.b_P0_alpha)
        if self._layout.idx_beta is not None:
            Qb = _pos(self.s_beta ** 2); ab = _pos(self._a_beta)
            lp += lIG(Qb, 0.5, 1.0 / ab) + lIG(ab, 1.0, 1.0 / (self.priors.hc_scale_beta ** 2))
            lp += -0.5 * ((self.m0_beta - self.priors.m_m0_beta) ** 2) / (self.priors.s_m0_beta ** 2)
            lp += lIG(self.P0_beta, self.priors.a_P0_beta, self.priors.b_P0_beta)
        if self.seasonal_mode == "dynamic":
            Qg = _pos(self.s_gamma ** 2); ag = _pos(self._a_gamma)
            lp += lIG(Qg, 0.5, 1.0 / ag) + lIG(ag, 1.0, 1.0 / (self.priors.hc_scale_gamma ** 2))
            base = np.zeros(self.period - 1) if self.priors.m_m0_gamma is None else np.asarray(self.priors.m_m0_gamma, float)
            s2 = self.priors.s_m0_gamma ** 2
            lp += -0.5 * float(np.sum((self.m0_gamma - base) ** 2)) / s2
            lp += lIG(self.P0_gamma, self.priors.a_P0_gamma, self.priors.b_P0_gamma)
        # deterministic blocks
        if self.level_mode == "deterministic":
            lp += -0.5 * ((self.m0_alpha - self.priors.m_m0_alpha) ** 2) / (self.priors.s_m0_alpha ** 2)
        if self.trend_mode == "deterministic":
            lp += -0.5 * ((self.m0_beta - self.priors.m_m0_beta) ** 2) / (self.priors.s_m0_beta ** 2)
        if self.seasonal_mode == "deterministic":
            K = self.period - 1
            mu = np.zeros(K) if self.priors.m_m0_gamma is None else np.asarray(self.priors.m_m0_gamma, float)
            s2 = self.priors.s_m0_gamma ** 2
            lp += -0.5 * float(np.sum((np.asarray(self.m0_gamma) - mu) ** 2)) / s2
        # model priors
        lp += math.log(self.model_prior["level"][self.level_mode] + 1e-300)
        lp += math.log(self.model_prior["trend"][self.trend_mode] + 1e-300)
        lp += math.log(self.model_prior["season"][self.seasonal_mode] + 1e-300)
        return float(lp)

    # =========================
    # FFBS
    # =========================
    def _ffbs(self) -> np.ndarray:
        if self._layout.dim == 0:
            return np.zeros((self.T + 1, 0), float)
        H = self._layout.H(); A = self._layout.A()
        Q = self._layout.Q(self.s_alpha, self.s_beta, self.s_gamma)
        R = _pos(self.sigma2, EPS)
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
            ydet = self._mu_det_t(t - 1)
            S = float(H @ Rm[t] @ H.T + R); S = _pos(S, EPS)
            v = float(self.y[t - 1] - ydet - H @ a[t])
            K = (Rm[t] @ H.T) / S
            m[t] = a[t] + K.flatten() * v
            C[t] = Rm[t] - K @ (H @ Rm[t])
            C[t] = 0.5 * (C[t] + C[t].T) + EPS * np.eye(self._layout.dim)
        x = np.zeros_like(m)
        x[self.T] = _mvnorm(self.rng, m[self.T], C[self.T])
        for t in range(self.T - 1, -1, -1):
            J = C[t] @ A.T
            J = _spd_solve(Rm[t + 1], J.T).T
            mean = m[t] + J @ (x[t + 1] - (A @ m[t] + u))
            cov = C[t] - J @ Rm[t + 1] @ J.T
            x[t] = _mvnorm(self.rng, mean, cov)
        return x

    def _seed_forward(self, Q_diag: np.ndarray) -> None:
        if self._layout.dim == 0:
            return
        A = self._layout.A(); u = self._layout.u(self.m0_beta)
        sd = np.sqrt(np.maximum(Q_diag, TINY))
        for t in range(1, self.T + 1):
            self.x[t] = A @ self.x[t - 1] + u + self.rng.normal(0.0, sd, size=self._layout.dim)

    # =========================
    # Conjugate updates
    # =========================
    @staticmethod
    def _rinvgamma(rng: np.random.Generator, shape: float, scale: float) -> float:
        k = _pos(shape); sc = _pos(scale)
        theta = 1.0 / sc
        if not np.isfinite(theta) or theta > GAMMA_SCALE_CAP:
            theta = GAMMA_SCALE_CAP
        g = rng.gamma(k, theta)
        if not np.isfinite(g) or g <= 0.0:
            g = _pos(g)
        return 1.0 / g

    def _mu_vec(self) -> np.ndarray:
        if self._layout.dim == 0:
            return np.array([self._mu_det_t(t) for t in range(self.T)], float)
        H = self._layout.H()
        mu = np.zeros(self.T)
        for t in range(1, self.T + 1):
            mu[t - 1] = self._mu_det_t(t - 1) + float(H @ self.x[t])
        return np.nan_to_num(mu)

    def update_sigma2(self) -> None:
        e = self.y - self._mu_vec()
        a = self.priors.a_sigma + 0.5 * self.T
        b = self.priors.b_sigma + 0.5 * float(e @ e)
        tau = self.rng.gamma(_pos(a), 1.0 / _pos(b))
        s2 = 1.0 / _pos(tau)
        self.sigma2 = float(np.clip(s2, SIG2_MIN, SIG2_MAX))

    # centered innovation sums of squares
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
            prev = self.x[t - 1, gs: ge + 1]
            mean_new = -float(np.sum(prev))
            ss += (self.x[t, gs] - mean_new) ** 2
        return float(ss), self.T

    def update_process_Q_halfcauchy_centered(self) -> None:
        def _safe_update(SS, Te, a_aux, A_scale):
            SS = float(np.nan_to_num(SS, nan=0.0, posinf=HUGE, neginf=0.0))
            SS = _cap(SS, hi=HUGE)
            a_aux = _pos(a_aux)
            Q = self._rinvgamma(self.rng, 0.5 * Te + 0.5, 0.5 * SS + 1.0 / a_aux)
            a_new = self._rinvgamma(self.rng, 1.0, (1.0 / (_pos(A_scale) ** 2)) + 1.0 / _pos(Q))
            Q = float(np.clip(Q, Q_MIN, Q_MAX))
            return Q, max(a_new, TINY)

        if self._layout.idx_alpha is not None:
            SS, Te = self._innovation_ss_alpha(); Q, a = _safe_update(SS, Te, self._a_alpha, self.priors.hc_scale_alpha)
            self.s_alpha = math.sqrt(Q); self._a_alpha = a
        if self._layout.idx_beta is not None:
            SS, Te = self._innovation_ss_beta();  Q, a = _safe_update(SS, Te, self._a_beta,  self.priors.hc_scale_beta)
            self.s_beta  = math.sqrt(Q); self._a_beta  = a
        if self.seasonal_mode == "dynamic":
            SS, Te = self._innovation_ss_gamma(); Q, a = _safe_update(SS, Te, self._a_gamma, self.priors.hc_scale_gamma)
            self.s_gamma = math.sqrt(Q); self._a_gamma = a

    # ===============
    # ASIS
    # ===============
    def _asis_compute_z(self) -> Dict[str, np.ndarray]:
        z: Dict[str, np.ndarray] = {}
        if self._layout.idx_alpha is not None:
            li = self._layout.idx_alpha
            za = np.zeros(self.T)
            denom = _pos(self.s_alpha)
            for t in range(1, self.T + 1):
                if self._layout.idx_beta is not None:
                    drift = self.x[t - 1, self._layout.idx_beta]
                elif self.trend_mode == "deterministic":
                    drift = float(self.m0_beta)
                else:
                    drift = 0.0
                mean = self.x[t - 1, li] + drift
                za[t - 1] = (self.x[t, li] - mean) / denom
            z["alpha"] = np.clip(np.nan_to_num(za, nan=0.0, posinf=0.0, neginf=0.0), -1e9, 1e9)
        if self._layout.idx_beta is not None:
            bi = self._layout.idx_beta
            zb = (self.x[1:, bi] - self.x[:-1, bi]) / _pos(self.s_beta)
            z["beta"] = np.clip(np.nan_to_num(zb, nan=0.0, posinf=0.0, neginf=0.0), -1e9, 1e9)
        if self.seasonal_mode == "dynamic":
            gs, ge = self._layout.idx_g_start, self._layout.idx_g_end
            zg = np.zeros(self.T)
            denom = _pos(self.s_gamma)
            for t in range(1, self.T + 1):
                prev = self.x[t - 1, gs: ge + 1]
                mean_new = -float(np.sum(prev))
                zg[t - 1] = (self.x[t, gs] - mean_new) / denom
            z["gamma"] = np.clip(np.nan_to_num(zg, nan=0.0, posinf=0.0, neginf=0.0), -1e9, 1e9)
        return z

    def _asis_sample_Q_from_z(self, z: Dict[str, np.ndarray]) -> None:
        def _safe_Q_draw(nz, SS, a_aux, A_scale):
            nz = int(max(0, nz))
            SS = _cap(float(np.nan_to_num(SS, nan=0.0, posinf=HUGE, neginf=0.0)), hi=HUGE)
            a_aux = _pos(a_aux)
            Q = self._rinvgamma(self.rng, 0.5 * nz + 0.5, 0.5 * SS + 1.0 / a_aux)
            a_new = self._rinvgamma(self.rng, 1.0, (1.0 / (_pos(A_scale) ** 2)) + 1.0 / _pos(Q))
            Q = float(np.clip(Q, Q_MIN, Q_MAX))
            return Q, max(a_new, TINY)

        if "alpha" in z:
            SS = float(z["alpha"] @ z["alpha"])
            Q, a = _safe_Q_draw(z["alpha"].size, SS, self._a_alpha, self.priors.hc_scale_alpha)
            self.s_alpha = math.sqrt(Q); self._a_alpha = a
        if "beta" in z:
            SS = float(z["beta"] @ z["beta"])
            Q, a = _safe_Q_draw(z["beta"].size, SS, self._a_beta,  self.priors.hc_scale_beta)
            self.s_beta  = math.sqrt(Q); self._a_beta  = a
        if "gamma" in z:
            SS = float(z["gamma"] @ z["gamma"])
            Q, a = _safe_Q_draw(z["gamma"].size, SS, self._a_gamma, self.priors.hc_scale_gamma)
            self.s_gamma = math.sqrt(Q); self._a_gamma = a

    def _asis_rebuild_states_from_z(self, z: Dict[str, np.ndarray]) -> None:
        if self._layout.dim == 0:
            return
        gs, ge = self._layout.idx_g_start, self._layout.idx_g_end
        x_new = self.x.copy()
        for t in range(1, self.T + 1):
            if self._layout.idx_beta is not None:
                bi = self._layout.idx_beta
                x_new[t, bi] = x_new[t - 1, bi] + self.s_beta * z["beta"][t - 1]
            if self._layout.idx_alpha is not None:
                li = self._layout.idx_alpha
                if self._layout.idx_beta is not None:
                    drift = x_new[t - 1, self._layout.idx_beta]
                elif self.trend_mode == "deterministic":
                    drift = float(self.m0_beta)
                else:
                    drift = 0.0
                mean = x_new[t - 1, li] + drift
                x_new[t, li] = mean + self.s_alpha * z["alpha"][t - 1]
            if self.seasonal_mode == "dynamic":
                prev = x_new[t - 1, gs: ge + 1]
                mean_new = -float(np.sum(prev))
                x_new[t, gs] = mean_new + self.s_gamma * z["gamma"][t - 1]
                if (ge - gs + 1) > 1:
                    x_new[t, gs + 1: ge + 1] = prev[: (ge - gs)]
        self.x = x_new

    def asis_step(self) -> None:
        z = self._asis_compute_z()
        if not z:
            return
        self._asis_sample_Q_from_z(z)
        self._asis_rebuild_states_from_z(z)

    # =========================
    # m0 / P0 and deterministic params
    # =========================
    @staticmethod
    def _gibbs_m0_scalar(rng, x0, m_prior, s_prior, P0):
        prec = 1.0 / (s_prior ** 2) + 1.0 / _pos(P0, 1e-18)
        var = 1.0 / prec
        mean = var * (m_prior / (s_prior ** 2) + x0 / _pos(P0, 1e-18))
        return float(rng.normal(mean, math.sqrt(var)))

    def update_m0(self) -> None:
        if self._layout.dim == 0:
            return
        pos = 0
        if self._layout.idx_alpha is not None:
            self.m0_alpha = self._gibbs_m0_scalar(self.rng, float(self.x[0, pos]), self.priors.m_m0_alpha, self.priors.s_m0_alpha, self.P0_alpha); pos += 1
        if self._layout.idx_beta is not None:
            self.m0_beta  = self._gibbs_m0_scalar(self.rng, float(self.x[0, pos]), self.priors.m_m0_beta,  self.priors.s_m0_beta,  self.P0_beta ); pos += 1
        if self.seasonal_mode == "dynamic":
            base = np.zeros(self.period - 1) if self.priors.m_m0_gamma is None else np.asarray(self.priors.m_m0_gamma, float)
            s = float(self.priors.s_m0_gamma)
            for k in range(self.period - 1):
                self.m0_gamma[k] = self._gibbs_m0_scalar(self.rng, float(self.x[0, pos + k]), float(base[k]), s, self.P0_gamma)

    def update_P0(self) -> None:
        if self._layout.dim == 0:
            return
        pos = 0
        if self._layout.idx_alpha is not None:
            a = self.priors.a_P0_alpha + 0.5; b = self.priors.b_P0_alpha + 0.5 * (float(self.x[0, pos]) - self.m0_alpha) ** 2
            self.P0_alpha = 1.0 / self.rng.gamma(_pos(a), 1.0 / _pos(b)); pos += 1
        if self._layout.idx_beta is not None:
            a = self.priors.a_P0_beta + 0.5;  b = self.priors.b_P0_beta  + 0.5 * (float(self.x[0, pos]) - self.m0_beta)  ** 2
            self.P0_beta  = 1.0 / self.rng.gamma(_pos(a), 1.0 / _pos(b)); pos += 1
        if self.seasonal_mode == "dynamic":
            diffsq = 0.0
            for k in range(self.period - 1):
                diffsq += (float(self.x[0, pos + k]) - float(self.m0_gamma[k])) ** 2
            a = self.priors.a_P0_gamma + 0.5 * (self.period - 1); b = self.priors.b_P0_gamma + 0.5 * diffsq
            self.P0_gamma = 1.0 / self.rng.gamma(_pos(a), 1.0 / _pos(b))

    def update_deterministic_params(self) -> None:
        sig2 = float(_pos(self.sigma2, EPS))
        # LEVEL det
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
                    mg = np.asarray(self.m0_gamma, float); self._m0_gamma_full = np.r_[mg, -mg.sum()]
                r -= self._m0_gamma_full[np.arange(self.T) % self.period]
            m0, s0 = float(self.priors.m_m0_alpha), float(self.priors.s_m0_alpha)
            prec = self.T / sig2 + 1.0 / (s0 * s0)
            mean = ((r.sum() / sig2) + m0 / (s0 * s0)) / prec
            self.m0_alpha = float(self.rng.normal(mean, math.sqrt(1.0 / prec)))
        # TREND det
        if self.trend_mode == "deterministic":
            if self._layout.idx_alpha is not None:
                d = self.x[1:, self._layout.idx_alpha] - self.x[:-1, self._layout.idx_alpha]
                q = float(self.s_alpha ** 2) if self.s_alpha > 0 else EPS
                m0, s0 = float(self.priors.m_m0_beta), float(self.priors.s_m0_beta)
                prec = self.T / q + 1.0 / (s0 * s0)
                mean = ((float(np.sum(d)) / q) + m0 / (s0 * s0)) / prec
                self.m0_beta = float(self.rng.normal(mean, math.sqrt(1.0 / prec)))
            else:
                tvec = np.arange(self.T, dtype=float); r = self.y.copy()
                if self._layout.dim > 0:
                    H = self._layout.H()
                    for k in range(1, self.T + 1):
                        r[k - 1] -= float(H @ self.x[k])
                if self.level_mode == "deterministic":
                    r -= float(self.m0_alpha)
                if self.seasonal_mode == "deterministic":
                    if getattr(self, "_m0_gamma_full", None) is None:
                        mg = np.asarray(self.m0_gamma, float); self._m0_gamma_full = np.r_[mg, -mg.sum()]
                    r -= self._m0_gamma_full[np.arange(self.T) % self.period]
                m0, s0 = float(self.priors.m_m0_beta), float(self.priors.s_m0_beta)
                prec = (tvec @ tvec) / sig2 + 1.0 / (s0 * s0)
                mean = ((tvec @ r) / sig2 + m0 / (s0 * s0)) / prec
                self.m0_beta = float(self.rng.normal(mean, math.sqrt(1.0 / prec)))
        # SEASON det
        if self.seasonal_mode == "deterministic":
            K = self.period - 1; midx = np.arange(self.T) % self.period
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
            mu_prior = np.zeros(K) if self.priors.m_m0_gamma is None else np.asarray(self.priors.m_m0_gamma, float)
            s2p = float(self.priors.s_m0_gamma) ** 2
            Prec = (Z.T @ Z) / sig2 + np.eye(K) / s2p
            b = (Z.T @ r) / sig2 + mu_prior / s2p
            L = np.linalg.cholesky(Prec)
            mu = np.linalg.solve(L.T, np.linalg.solve(L, b))
            theta = mu + np.linalg.solve(L.T, self.rng.standard_normal(K))
            self.m0_gamma = theta; self._m0_gamma_full = np.r_[theta, -theta.sum()]

    # =========================
    # RJ moves
    # =========================
    def _snapshot(self) -> dict:
        return {
            "level": self.level_mode, "trend": self.trend_mode, "season": self.seasonal_mode,
            "sigma2": self.sigma2,
            "s_alpha": self.s_alpha, "s_beta": self.s_beta, "s_gamma": self.s_gamma,
            "a_alpha": self._a_alpha, "a_beta": self._a_beta, "a_gamma": self._a_gamma,
            "m0_alpha": self.m0_alpha, "P0_alpha": self.P0_alpha,
            "m0_beta":  self.m0_beta,  "P0_beta":  self.P0_beta,
            "m0_gamma": None if self.m0_gamma is None else self.m0_gamma.copy(),
            "P0_gamma": self.P0_gamma,
        }

    def _load_snapshot(self, S: dict) -> None:
        self.level_mode = S["level"]; self.trend_mode = S["trend"]; self.seasonal_mode = S["season"]
        self.sigma2 = float(np.clip(S["sigma2"], SIG2_MIN, SIG2_MAX))  # keep obs variance sane
        self.s_alpha = float(S["s_alpha"]); self.s_beta = float(S["s_beta"]); self.s_gamma = float(S["s_gamma"])
        self._a_alpha = float(S["a_alpha"]); self._a_beta = float(S["a_beta"]); self._a_gamma = float(S["a_gamma"])
        self.m0_alpha = float(S["m0_alpha"]); self.P0_alpha = float(S["P0_alpha"])
        self.m0_beta  = float(S["m0_beta"]);  self.P0_beta  = float(S["P0_beta"]) 
        self.m0_gamma = None if S["m0_gamma"] is None else np.asarray(S["m0_gamma"], float).copy()
        self.P0_gamma = float(S["P0_gamma"])
        if self.seasonal_mode == "deterministic":
            mg = np.asarray(self.m0_gamma, float); self._m0_gamma_full = np.r_[mg, -mg.sum()]
        else:
            self._m0_gamma_full = None
        self._layout = _Layout(self.period, self.level_mode, self.trend_mode, self.seasonal_mode)

    @staticmethod
    def _mh_decide(prop_lp: float, cur_lp: float, rng: np.random.Generator) -> bool:
        # Hard reject if either side non-finite
        if not np.isfinite(prop_lp) or not np.isfinite(cur_lp):
            return False
        d = prop_lp - cur_lp
        if d >= 0.0:
            return True
        return (math.log(rng.uniform()) < d)

    def _safe_logpost(self, s_floor: float = EPS) -> float:
        ll = self._kalman_loglik(s_floor=s_floor)
        lp = self._log_prior_current()
        val = ll + lp
        return float(val) if np.isfinite(val) else float("nan")

    def _draw_prior_dyn_block(self, which: str) -> None:
        def _birth(A_scale: float) -> Tuple[float, float]:
            a_aux = self._rinvgamma(self.rng, 1.0, 1.0 / _pos(A_scale ** 2))
            Q = self._rinvgamma(self.rng, 0.5, 1.0 / _pos(a_aux))
            Q = float(np.clip(Q, 1e-4, 10.0))  # pragmatic floor/ceiling for new block
            return max(a_aux, TINY), math.sqrt(Q)
        if which == "level":
            a_aux, s = _birth(self.priors.hc_scale_alpha)
            self._a_alpha = a_aux; self.s_alpha = s
            self.m0_alpha = float(self.rng.normal(self.priors.m_m0_alpha, self.priors.s_m0_alpha))
            self.P0_alpha = 1.0 / self.rng.gamma(_pos(self.priors.a_P0_alpha), 1.0 / _pos(self.priors.b_P0_alpha))
        elif which == "trend":
            a_aux, s = _birth(self.priors.hc_scale_beta)
            self._a_beta = a_aux; self.s_beta = s
            self.m0_beta = float(self.rng.normal(self.priors.m_m0_beta, self.priors.s_m0_beta))
            self.P0_beta = 1.0 / self.rng.gamma(_pos(self.priors.a_P0_beta), 1.0 / _pos(self.priors.b_P0_beta))
        elif which == "season":
            a_aux, s = _birth(self.priors.hc_scale_gamma)
            self._a_gamma = a_aux; self.s_gamma = s
            base = np.zeros(self.period - 1) if self.priors.m_m0_gamma is None else np.asarray(self.priors.m_m0_gamma, float)
            self.m0_gamma = self.rng.normal(base, float(self.priors.s_m0_gamma), size=self.period - 1)
            self.P0_gamma = 1.0 / self.rng.gamma(_pos(self.priors.a_P0_gamma), 1.0 / _pos(self.priors.b_P0_gamma))
            self._m0_gamma_full = None

    def _propose_mode(self, comp: str, cur: str) -> str:
        if comp == "level":
            cand = ["dynamic", "deterministic"] + (["none"] if self.cfg.allow_none_level else [])
        elif comp == "trend":
            cand = ["dynamic", "deterministic"] + (["none"] if self.cfg.allow_none_trend else [])
        else:
            cand = ["dynamic", "deterministic"] + (["none"] if self.cfg.allow_none_season else [])
        choices = [c for c in cand if c != cur]
        return str(self.rng.choice(choices))

    def _legal_modes(self, L: str, T: str, S: str) -> bool:
        if T == "dynamic" and L != "dynamic":
            return False
        if (not self.cfg.allow_none_level) and L == "none":
            return False
        return True

    def _rj_move_one(self, s_floor: float) -> None:
        comp = str(self.rng.choice(["level", "trend", "season"]))
        L, T, S = self.level_mode, self.trend_mode, self.seasonal_mode
        cur = {"level": L, "trend": T, "season": S}[comp]
        prop = self._propose_mode(comp, cur)
        L2, T2, S2 = L, T, S
        if comp == "level":
            L2 = prop
            if T2 == "dynamic" and L2 != "dynamic":
                T2 = "deterministic"
        elif comp == "trend":
            T2 = prop
            if T2 == "dynamic" and L2 != "dynamic":
                L2 = "dynamic"  # escape move
        else:
            S2 = prop
        if not self._legal_modes(L2, T2, S2):
            return

        snap = self._snapshot()
        # births draw priors
        if (L != "dynamic") and (L2 == "dynamic"): self._draw_prior_dyn_block("level")
        if (T != "dynamic") and (T2 == "dynamic"): self._draw_prior_dyn_block("trend")
        if (S != "dynamic") and (S2 == "dynamic"): self._draw_prior_dyn_block("season")
        if (S != "deterministic") and (S2 == "deterministic"):
            base = np.zeros(self.period - 1) if self.priors.m_m0_gamma is None else np.asarray(self.priors.m_m0_gamma, float)
            self.m0_gamma = base.astype(float)
            self._m0_gamma_full = np.r_[self.m0_gamma, -float(np.sum(self.m0_gamma))]
            self.P0_gamma = 0.0

        # set proposed modes temporarily
        self.level_mode, self.trend_mode, self.seasonal_mode = L2, T2, S2
        self._layout = _Layout(self.period, L2, T2, S2)
        if self.level_mode != "dynamic": self.s_alpha = 0.0
        if self.trend_mode != "dynamic": self.s_beta = 0.0
        if self.seasonal_mode != "dynamic": self.s_gamma = 0.0
        if self.seasonal_mode == "deterministic":
            if self.m0_gamma is None or self.m0_gamma.size != (self.period - 1):
                base = np.zeros(self.period - 1) if self.priors.m_m0_gamma is None else np.asarray(self.priors.m_m0_gamma, float)
                self.m0_gamma = base.astype(float)
            self._m0_gamma_full = np.r_[self.m0_gamma, -float(np.sum(self.m0_gamma))]
        else:
            self._m0_gamma_full = None

        lp_prop = self._safe_logpost(s_floor=s_floor)
        self._load_snapshot(snap)
        lp_cur = self._safe_logpost(s_floor=s_floor)
        acc = self._mh_decide(lp_prop, lp_cur, self.rng)
        self.rj_stats[comp]["proposed"] += 1
        if acc:
            self._load_snapshot(snap)  # reset
            self.level_mode, self.trend_mode, self.seasonal_mode = L2, T2, S2
            if self.level_mode != "dynamic": self.s_alpha = 0.0
            if self.trend_mode != "dynamic": self.s_beta = 0.0
            if self.seasonal_mode != "dynamic": self.s_gamma = 0.0
            if self.seasonal_mode == "deterministic":
                base = np.zeros(self.period - 1) if self.priors.m_m0_gamma is None else np.asarray(self.priors.m_m0_gamma, float)
                self.m0_gamma = base.astype(float)
                self._m0_gamma_full = np.r_[self.m0_gamma, -float(np.sum(self.m0_gamma))]
            else:
                self._m0_gamma_full = None
            self._layout = _Layout(self.period, self.level_mode, self.trend_mode, self.seasonal_mode)
            self._alloc_state_holder()
            self.rj_stats[comp]["accepted"] += 1
            if self.cfg.progress:
                dclip = float(np.clip(lp_prop - lp_cur, -1e3, 1e3)) if (np.isfinite(lp_prop) and np.isfinite(lp_cur)) else 0.0
                print(f"[switch] block={comp} | L:{L[:3]}→{L2[:3]} T:{T[:3]}→{T2[:3]} S:{S[:3]}→{S2[:3]} | Δlogpost={dclip:+.4f}")

    # =========================
    # Bookkeeping & I/O
    # =========================
    def _tally_modes(self) -> None:
        self._mode_counts["level"][self.level_mode] += 1
        self._mode_counts["trend"][self.trend_mode] += 1
        self._mode_counts["season"][self.seasonal_mode] += 1

    def inclusion_probabilities(self) -> Dict[str, Dict[str, float]]:
        total = sum(self._mode_counts["level"].values()) or 1
        return {comp: {m: c / total for m, c in cnt.items()} for comp, cnt in self._mode_counts.items()}

    def _progress_line(self, it: int) -> str:
        parts = [
            f"[it {it + 1}/{self.cfg.n_iter}]",
            f"σ={math.sqrt(self.sigma2):.3f}",
            f"L/T/S={self.level_mode[:3]}/{self.trend_mode[:3]}/{self.seasonal_mode[:3]}",
        ]
        if self._layout.idx_alpha is not None: parts.append(f"Qα={self.s_alpha ** 2:.4g}")
        if self._layout.idx_beta  is not None: parts.append(f"Qβ={self.s_beta  ** 2:.4g}")
        if self.seasonal_mode == "dynamic": parts.append(f"Qγ={self.s_gamma ** 2:.4g}")
        return " | ".join(parts)

    def run(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        save_iters = list(range(cfg.burn, cfg.n_iter, cfg.thin))
        n_kept, keep_idx = len(save_iters), 0
        max_dim = self._max_state_dim()
        self.keep = {
            "sigma": np.full(n_kept, np.nan),
            "mu":    np.full((n_kept, self.T), np.nan),
            "modes": np.zeros((n_kept, 3), int),
            "x":     np.zeros((n_kept, self.T, max_dim)),
            "Q_alpha":  np.full(n_kept, np.nan),
            "Q_beta":   np.full(n_kept, np.nan),
            "Q_gamma":  np.full(n_kept, np.nan),
            "m0_alpha": np.full(n_kept, np.nan),
            "m0_beta":  np.full(n_kept, np.nan),
            "m0_gamma": np.full((n_kept, self.period - 1), np.nan),
            "P0_alpha": np.full(n_kept, np.nan),
            "P0_beta":  np.full(n_kept, np.nan),
            "P0_gamma": np.full(n_kept, np.nan),
            "m0_alpha_det": np.full(n_kept, np.nan),
            "m0_beta_det":  np.full(n_kept, np.nan),
            "season_det":   np.full((n_kept, self.period - 1), np.nan),
        }
        print_every = cfg.progress_every if cfg.progress_every > 0 else max(1, cfg.n_iter // 50) or 1

        for it in range(cfg.n_iter):
            # ASIS during early iters even if disabled, to stabilize chains
            asis_on = (it < max(1000, cfg.rj_warmup)) or cfg.asis
            s_floor = 1e-8 if it < 500 else EPS

            if self._layout.dim > 0:
                self.x = self._ffbs()
                if asis_on:
                    self.asis_step()
                else:
                    self.update_process_Q_halfcauchy_centered()
                self.update_m0(); self.update_P0()
            self.update_deterministic_params()
            self.update_sigma2()

            # RJ only after warmup
            if it >= cfg.rj_warmup:
                for _ in range(cfg.rj_moves_per_iter):
                    self._rj_move_one(s_floor=s_floor)

            self._tally_modes()
            if cfg.progress and ((it + 1) % print_every == 0 or it == cfg.n_iter - 1):
                print(self._progress_line(it))
            if it in save_iters:
                k = keep_idx; keep_idx += 1
                mu = self._mu_vec()
                self.keep["mu"][k, :] = mu
                self.keep["sigma"][k] = math.sqrt(self.sigma2)
                enc = lambda s: 0 if s == "dynamic" else (1 if s == "deterministic" else 2)
                self.keep["modes"][k, :] = np.array([enc(self.level_mode), enc(self.trend_mode), enc(self.seasonal_mode)], int)
                self.keep["Q_alpha"][k] = (self.s_alpha ** 2) if (self.level_mode == "dynamic") else np.nan
                self.keep["Q_beta"][k]  = (self.s_beta  ** 2) if (self.trend_mode == "dynamic") else np.nan
                self.keep["Q_gamma"][k] = (self.s_gamma ** 2) if (self.seasonal_mode == "dynamic") else np.nan
                if self.level_mode == "dynamic":
                    self.keep["m0_alpha"][k] = self.m0_alpha; self.keep["P0_alpha"][k] = self.P0_alpha
                else:
                    self.keep["m0_alpha_det"][k] = self.m0_alpha
                if self.trend_mode == "dynamic":
                    self.keep["m0_beta"][k] = self.m0_beta; self.keep["P0_beta"][k] = self.P0_beta
                else:
                    self.keep["m0_beta_det"][k] = self.m0_beta
                if self.seasonal_mode == "dynamic":
                    self.keep["m0_gamma"][k, :] = np.asarray(self.m0_gamma, float); self.keep["P0_gamma"][k] = self.P0_gamma
                else:
                    self.keep["season_det"][k, :] = np.asarray(self.m0_gamma, float)
                if self._layout.dim > 0:
                    w = min(self.x.shape[1], self.keep["x"].shape[2])
                    self.keep["x"][k, :, :w] = self.x[1: self.T + 1, :w]
        return self.keep

    def save_posterior(self, out_npz_path: str, extra_meta: Optional[dict] = None) -> None:
        os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)
        arrays = dict(self.keep); arrays["y"] = self.y.copy()
        np.savez_compressed(out_npz_path, **arrays)
        meta = {
            "T": int(self.T),
            "period": int(self.period),
            "cfg": asdict(self.cfg),
            "priors": asdict(self.priors),
            "model_prior": self.model_prior,
            "current_modes": {"level": self.level_mode, "trend": self.trend_mode, "season": self.seasonal_mode},
            "rj_accept": {b: {"proposed": int(self.rj_stats[b]["proposed"]), "accepted": int(self.rj_stats[b]["accepted"]), "acc_rate": (self.rj_stats[b]["accepted"] / max(1, self.rj_stats[b]["proposed"]))} for b in ("level", "trend", "season")},
            "inclusion_probs": self.inclusion_probabilities(),
            "modes_encoding": {"dynamic": 0, "deterministic": 1, "none": 2},
        }
        if extra_meta: meta.update(extra_meta)
        with open(out_npz_path.replace(".npz", ".meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, indent=2, fp=f)
        print(f"[save] Posterior -> {out_npz_path}")


# =========================
# CLI
# =========================

def _csv_floats_or_none(s: str | None):
    if s is None: return None
    s = s.strip()
    if s == "": return None
    return [float(z) for z in s.split(",") if z.strip()]


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(base_dir)
    try:
        from simulator.mean_time_series import Mean_Time_Series  # newest-first convention
    except Exception:
        Mean_Time_Series = None

    p = argparse.ArgumentParser(description="DLM with RJ–MCMC + Half-Cauchy priors. Optional ASIS interweaving.")
    # Simulation
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--period", type=int, default=4)
    p.add_argument("--level-mode", choices=["dynamic", "deterministic"], default="dynamic")
    p.add_argument("--trend-mode", choices=["dynamic", "deterministic", "none"], default="dynamic")
    p.add_argument("--seasonal-mode", choices=["dynamic", "deterministic", "none"], default="dynamic")
    p.add_argument("--sigma", type=float, default=3.0)
    p.add_argument("--q-level", type=float, default=0.01)
    p.add_argument("--q-trend", type=float, default=2e-6)
    p.add_argument("--q-season", type=float, default=0.001)
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
    p.add_argument("--hc-scale-alpha", type=float, default=1.0)
    p.add_argument("--hc-scale-beta",  type=float, default=1.0)
    p.add_argument("--hc-scale-gamma", type=float, default=1.0)
    # Model priors (Occam tilt)
    p.add_argument("--prior-model-level", type=str, default=None)
    p.add_argument("--prior-model-trend", type=str, default=None)
    p.add_argument("--prior-model-season", type=str, default=None)
    # Sampler
    p.add_argument("--n-iter", type=int, default=10000)
    p.add_argument("--burn", type=int, default=5000)
    p.add_argument("--thin", type=int, default=1)
    p.add_argument("--seed", type=int, default=40)
    p.add_argument("--progress", default=True)
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--asis", default=True, help="Enable ASIS interweaving")
    # RJ options
    p.add_argument("--rj-moves-per-iter", type=int, default=10)
    p.add_argument("--rj-warmup", type=int, default=1000, help="Iterations to wait before RJ moves")
    p.add_argument("--allow-none-level", action="store_true", default=False)
    p.add_argument("--allow-none-trend", action="store_true", default=False)
    p.add_argument("--allow-none-season", action="store_true", default=False)
    # IO
    p.add_argument("--out-dir", type=str, default="results/simulations/DLM")
    p.add_argument("--plot", action="store_true", default=True)

    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    # Simulate or fallback
    if Mean_Time_Series is not None:
        m0_season = _csv_floats_or_none(args.m0_season) or [3] * (args.period - 1)
        v0_season = _csv_floats_or_none(args.v0_season) or [0.05] * (args.period - 1)
        mts = Mean_Time_Series(
            sigma=args.sigma, period=args.period,
            level_mode=args.level_mode, trend_mode=args.trend_mode, seasonal_mode=args.seasonal_mode,
            q_level=args.q_level, q_trend=args.q_trend, q_season=args.q_season,
            m0_level=args.m0_level, v0_level=args.v0_level,
            m0_trend=args.m0_trend, v0_trend=args.v0_trend,
            m0_season=m0_season, v0_season=v0_season,
        )
        y = np.array([mts.move() or mts.measure() for _ in range(args.T)], float)
        mu_T = mts.get_truth_paths(as_numpy=True)["mu_t"][1: 1 + args.T]
        dates_T = mts.get_truth_paths(as_numpy=True)["index"][: args.T]
    else:
        t = np.arange(args.T)
        seas = np.sin(2 * np.pi * t / max(2, args.period))
        y = 0.1 * t + 2 * seas + rng.normal(0, args.sigma, size=args.T)
        mu_T = 0.1 * t + 2 * seas
        dates_T = np.arange(args.T)

    pri_gamma_vec = _csv_floats_or_none(args.prior_m_m0_gamma)
    priors = Priors(
        a_sigma=args.prior_a_sigma, b_sigma=args.prior_b_sigma,
        m_m0_alpha=args.prior_m_m0_alpha, s_m0_alpha=args.prior_s_m0_alpha,
        m_m0_beta=args.prior_m_m0_beta,   s_m0_beta=args.prior_s_m0_beta,
        m_m0_gamma=None if pri_gamma_vec is None else pri_gamma_vec, s_m0_gamma=args.prior_s_m0_gamma,
        a_P0_alpha=args.prior_a_P0_alpha, b_P0_alpha=args.prior_b_P0_alpha,
        a_P0_beta=args.prior_a_P0_beta,   b_P0_beta=args.prior_b_P0_beta,
        a_P0_gamma=args.prior_a_P0_gamma, b_P0_gamma=args.prior_b_P0_gamma,
        hc_scale_alpha=args.hc_scale_alpha, hc_scale_beta=args.hc_scale_beta, hc_scale_gamma=args.hc_scale_gamma,
    )

    def _csv_model_prior_block(s: str | None, allow_none: bool, defaults: dict) -> dict:
        out = dict(defaults)
        if s:
            for pz in [p.strip() for p in s.split(",") if p.strip()]:
                k, v = pz.split(":"); out[k.strip()] = float(v)
        if not allow_none:
            out["none"] = min(out.get("none", 1e-12), 1e-12)
        ssum = sum(out.values()); out = {k: v / (ssum if ssum > 0 else sum(defaults.values())) for k, v in out.items()}
        return out

    default_model_prior = {
        "level": {"dynamic": 0.9, "deterministic": 0.1, "none": 1e-12},
        "trend": {"dynamic": 0.9, "deterministic": 0.1, "none": 0.4 if args.allow_none_trend else 1e-12},
        "season": {"dynamic": 0.9, "deterministic": 0.1, "none": 0.2 if args.allow_none_season else 1e-12},
    }
    model_prior = {
        "level":  _csv_model_prior_block(None, args.allow_none_level,  default_model_prior["level"]),
        "trend":  _csv_model_prior_block(None, args.allow_none_trend,  default_model_prior["trend"]),
        "season": _csv_model_prior_block(None, args.allow_none_season, default_model_prior["season"]),
    }

    cfg = SamplerConfig(
        n_iter=int(args.n_iter), burn=int(args.burn), thin=int(args.thin),
        random_seed=int(args.seed), progress=bool(args.progress), progress_every=int(args.progress_every),
        rj_moves_per_iter=int(args.rj_moves_per_iter), rj_warmup=int(args.rj_warmup),
        allow_none_level=bool(args.allow_none_level), allow_none_trend=bool(args.allow_none_trend), allow_none_season=bool(args.allow_none_season),
        asis=bool(args.asis),
    )

    sampler = DLMRJGibbs(
        y=y, period=args.period,
        level_mode="deterministic", trend_mode="deterministic", seasonal_mode="deterministic",
        sigma2_init=1.0,
        s_alpha_init=0.1, s_beta_init=0.1, s_gamma_init=0.1,
        m0_alpha_init=args.m0_level, m0_beta_init=args.m0_trend,
        m0_gamma_init=None, P0_alpha_init=0.05, P0_beta_init=0.05, P0_gamma_init=0.05,
        priors=priors, cfg=cfg, model_prior=model_prior, rng=rng,
    )

    t0 = time.time()
    post = sampler.run()
    print(f"\n[Run completed in {time.time()-t0:.1f}s] (ASIS={cfg.asis}, RJ-warmup={cfg.rj_warmup})")

    out_dir = os.path.join(args.out_dir, f"DLM_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    sampler.save_posterior(os.path.join(out_dir, "posterior.npz"), extra_meta={"asis": bool(cfg.asis)})

    if args.plot:
        import matplotlib.pyplot as plt
        mu_hat = post["mu"].mean(axis=0)
        plt.figure(figsize=(10, 4))
        plt.plot(dates_T, y, label="y_t", lw=1)
        if Mean_Time_Series is not None:
            plt.plot(dates_T, mu_T, "--", label="μ_t (truth)")
        plt.plot(dates_T, mu_hat, "-.", label="μ̂_t (post mean)")
        plt.title(f"DLM (ASIS={cfg.asis}, RJ warmup={cfg.rj_warmup}) — modes now: {sampler.level_mode}/{sampler.trend_mode}/{sampler.seasonal_mode}")
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fit.png"), dpi=160)
        plt.show()
