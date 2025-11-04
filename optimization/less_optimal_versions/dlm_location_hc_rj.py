from __future__ import annotations

import math, json, os, warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple
from collections import deque

import numpy as np
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ============================== small utilities ==============================

def _mad(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    if v.size == 0: return 0.0
    m = np.median(v)
    return float(np.median(np.abs(v - m)))

def _robust_sd(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    return _mad(v) / 1.4826 if v.size else 0.0

def _spd_solve(M: np.ndarray, B: np.ndarray, jitter: float = 1e-12) -> np.ndarray:
    """Solve M X = B for SPD M with escalating jitter; pseudo-inverse fallback."""
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

def _fmt_list(vals,
              max_elems: int = 6,
              fmt: str = ".4g",
              sep: str = ", ",
              brackets: tuple[str, str] = ("[", "]"),
              mode: str = "head") -> str:
    if vals is None: return "-"
    try:
        v = np.asarray(vals, dtype=float).ravel()
    except Exception:
        v = np.atleast_1d(vals)
    L, R = brackets
    n = v.size
    if n == 0: return f"{L}{R}"
    def _one(x):
        if isinstance(x, (float, np.floating)):
            if np.isnan(x): return "nan"
            if np.isposinf(x): return "inf"
            if np.isneginf(x): return "-inf"
        return f"{x:{fmt}}"
    if n <= max_elems:
        return f"{L}{sep.join(_one(x) for x in v)}{R}"
    ell = "…"
    if mode == "both" and max_elems >= 3:
        k_head = max_elems // 2
        k_tail = max_elems - k_head
        head = sep.join(_one(x) for x in v[:k_head])
        tail = sep.join(_one(x) for x in v[-k_tail:])
        return f"{L}{head}{sep}{ell}{sep}{tail}{R}"
    head = sep.join(_one(x) for x in v[:max_elems])
    return f"{L}{head}{sep}{ell}{R}"

# =============================== priors & config =============================

@dataclass
class Priors:
    # obs variance: τ = 1/σ² ~ Gamma(a_sigma, b_sigma) (shape–rate)
    a_sigma: float = 2.0
    b_sigma: float = 1.0

    # means for initial states / deterministic parameters
    m_m0_alpha: float = 0.0; s_m0_alpha: float = 10.0
    m_m0_beta:  float = 0.0; s_m0_beta:  float = 10.0

    # seasonal deterministic prior (newest-first, length p−1)
    m_m0_gamma: Optional[Sequence[float]] = None
    s_m0_gamma: float = 5.0

    # initial state variances (InvGamma)
    a_P0_alpha: float = 2.0; b_P0_alpha: float = 1.0
    a_P0_beta:  float = 2.0; b_P0_beta:  float = 1.0
    a_P0_gamma: float = 2.0; b_P0_gamma: float = 1.0

    # Half-Cauchy scales for process SDs (via IG mixture)
    hc_scale_alpha: float = 0.5
    hc_scale_beta:  float = 0.5
    hc_scale_gamma: float = 0.5

@dataclass
class SamplerConfig:
    n_iter: int = 20000
    burn: int = 5000
    thin: int = 2
    random_seed: Optional[int] = 40
    progress: bool = True
    progress_every: int = 0        # ~2% auto if 0

    # RJ tuning
    rj_moves_per_iter: int = 2
    allow_none_level: bool = False   # level usually always present
    allow_none_trend: bool = True
    allow_none_season: bool = True

    # RJ acceptance reporting
    rj_window: int = 500            # window for recent acc% (per block)

# =============================== main sampler ================================

class DLMRJGibbs:
    """
    Gaussian structural DLM with level / trend / seasonal blocks.
    Each block mode ∈ {dynamic, deterministic, none}.
    RJ–MCMC toggles block modes; acceptance via marginal Kalman loglik.

    Identifiability: trend='dynamic' ⇒ level='dynamic'.
    """

    # --------------------------------- init --------------------------------- #
    def __init__(
        self,
        y: np.ndarray,
        period: int,
        level_mode: str = "dynamic",
        trend_mode: str = "dynamic",
        seasonal_mode: str = "dynamic",
        sigma2_init: float = 1.0,
        # dynamic initials
        m0_alpha_init: float = 0.0, P0_alpha_init: float = 1.0,
        m0_beta_init:  float = 0.0, P0_beta_init:  float = 1.0,
        m0_gamma_init: Optional[Sequence[float]] = None, P0_gamma_init: float = 1.0,
        s_alpha_init: float = 1e-2, s_beta_init: float = 1e-3, s_gamma_init: float = 1e-3,
        priors: Priors = Priors(),
        cfg: SamplerConfig = SamplerConfig(),
    ):
        # data
        self.y = np.asarray(y, float)
        self.T = int(self.y.size)
        self.period = int(period)
        if self.period < 2:
            raise ValueError("period must be >= 2")

        # cfg & seed
        self.priors, self.cfg = priors, cfg
        if cfg.random_seed is not None:
            np.random.seed(int(cfg.random_seed))

        # modes
        ok = {"dynamic", "deterministic", "none"}
        if level_mode not in ok or trend_mode not in ok or seasonal_mode not in ok:
            raise ValueError("invalid mode")
        if trend_mode == "dynamic" and level_mode != "dynamic":
            raise ValueError("trend=dynamic requires level=dynamic")
        if not cfg.allow_none_level and level_mode == "none":
            raise ValueError("level='none' not allowed")

        self.level_mode   = level_mode
        self.trend_mode   = trend_mode
        self.seasonal_mode= seasonal_mode

        # observation variance
        self.sigma2 = float(sigma2_init)

        # process SDs (0 if not dynamic) + Half-Cauchy auxiliaries
        self.s_alpha = float(s_alpha_init) if self.level_mode   == "dynamic" else 0.0
        self.s_beta  = float(s_beta_init)  if self.trend_mode   == "dynamic" else 0.0
        self.s_gamma = float(s_gamma_init) if self.seasonal_mode== "dynamic" else 0.0
        self._a_alpha = 1.0; self._a_beta = 1.0; self._a_gamma = 1.0

        # initial state means/vars for dynamic blocks
        self.m0_alpha = float(m0_alpha_init) if self.level_mode == "dynamic" else 0.0
        self.P0_alpha = float(P0_alpha_init) if self.level_mode == "dynamic" else 0.0
        self.m0_beta  = float(m0_beta_init)  if self.trend_mode == "dynamic" else 0.0
        self.P0_beta  = float(P0_beta_init)  if self.trend_mode == "dynamic" else 0.0

        if self.seasonal_mode == "dynamic":
            if m0_gamma_init is None:
                self.m0_gamma = np.zeros(self.period - 1, float)
            else:
                g = np.asarray(m0_gamma_init, float)
                if g.size != self.period - 1:
                    raise ValueError("m0_gamma_init must have length p-1 (newest-first)")
                self.m0_gamma = g
            self.P0_gamma = float(P0_gamma_init)
        else:
            base = np.zeros(self.period - 1) if self.priors.m_m0_gamma is None \
                   else np.asarray(self.priors.m_m0_gamma, float)
            self.det_gamma = np.r_[base, -float(np.sum(base))].astype(float)
            self.m0_gamma = None
            self.P0_gamma = 0.0

        # deterministic coefficients (used if corresponding mode='deterministic')
        self.det_alpha = float(self.priors.m_m0_alpha)
        self.det_beta  = float(self.priors.m_m0_beta)
        if self.seasonal_mode == "deterministic":
            self.det_gamma = self.det_gamma.copy()

        # build layout + seed latent path
        self._rebuild_layout()
        self._alloc_state_holder()   # sets self.x with current dim
        if self.dim > 0:
            m0_vec, P0_diag = self._current_m0_P0()
            self.x[0] = np.random.multivariate_normal(m0_vec, np.diag(P0_diag) + 1e-10 * np.eye(self.dim))
            self._seed_forward(Q_diag=np.full(self.dim, 1e-6))

        # storage & mode tallies
        self.keep: Dict[str, np.ndarray] = {}
        self._mode_counts = {
            "level":  {"dynamic": 0, "deterministic": 0, "none": 0},
            "trend":  {"dynamic": 0, "deterministic": 0, "none": 0},
            "season": {"dynamic": 0, "deterministic": 0, "none": 0},
        }

        # RJ acceptance tracking (cumulative + windowed)
        self.rj_stats = {
            "level":  {"proposed": 0, "accepted": 0, "win": deque(maxlen=self.cfg.rj_window)},
            "trend":  {"proposed": 0, "accepted": 0, "win": deque(maxlen=self.cfg.rj_window)},
            "season": {"proposed": 0, "accepted": 0, "win": deque(maxlen=self.cfg.rj_window)},
        }

        # for emitting switch logs
        self._last_modes = (self.level_mode, self.trend_mode, self.seasonal_mode)

        # progress scales (robust)
        if self.cfg.progress:
            sd1 = _robust_sd(np.diff(self.y)) if self.T >= 2 else 0.0
            sd2 = _robust_sd(np.diff(self.y, n=2)) if self.T >= 3 else 0.0
            print(f"[init] modes L/T/S={self.level_mode[:3]}/{self.trend_mode[:3]}/{self.seasonal_mode[:3]} | sd1={sd1:.4g} sd2={sd2:.4g}")

    # ------------------------------ structure -------------------------------- #

    def _alloc_state_holder(self) -> None:
        """Allocate state holder self.x with the *current* dimension."""
        self.x = np.zeros((self.T + 1, self.dim), float)

    def _rebuild_layout(self) -> None:
        layout: List[str] = []
        self.idx_alpha = self.idx_beta = None
        self.idx_g_start = self.idx_g_end = None

        if self.level_mode == "dynamic":
            self.idx_alpha = len(layout); layout.append("alpha")
        if self.trend_mode == "dynamic":
            self.idx_beta = len(layout); layout.append("beta")
        if self.seasonal_mode == "dynamic":
            for k in range(1, self.period):
                layout.append(f"g{k}")
            if self.period > 1:
                self.idx_g_start = layout.index("g1")
                self.idx_g_end   = self.idx_g_start + (self.period - 2)

        self._layout = layout
        self.dim = len(layout)

    def _current_m0_P0(self) -> Tuple[np.ndarray, np.ndarray]:
        m0, P0 = [], []
        if self.idx_alpha is not None:
            m0.append(self.m0_alpha); P0.append(self.P0_alpha)
        if self.idx_beta is not None:
            m0.append(self.m0_beta);  P0.append(self.P0_beta)
        if self.seasonal_mode == "dynamic":
            m0.extend(list(self.m0_gamma))
            P0.extend([self.P0_gamma] * (self.period - 1))
        return np.asarray(m0, float), np.asarray(P0, float)

    # --------------------------------- system -------------------------------- #

    def _H(self) -> np.ndarray:
        if self.dim == 0: return np.zeros((1, 0))
        h = np.zeros(self.dim, float)
        if self.idx_alpha is not None:
            h[self.idx_alpha] = 1.0
        if self.seasonal_mode == "dynamic":
            h[self.idx_g_start] = 1.0
        return h.reshape(1, -1)

    def _A(self) -> np.ndarray:
        if self.dim == 0: return np.zeros((0, 0))
        A = np.eye(self.dim)
        if (self.idx_alpha is not None) and (self.idx_beta is not None):
            A[self.idx_alpha, self.idx_beta] = 1.0
        if self.seasonal_mode == "dynamic":
            gs, ge = self.idx_g_start, self.idx_g_end
            K = ge - gs + 1
            A[gs, gs:ge+1] = -1.0
            if K > 1:
                A[gs+1:ge+1, gs:ge] = np.eye(K - 1)
                A[gs+1:ge+1, ge]    = 0.0
        return A

    def _u(self) -> np.ndarray:
        if self.dim == 0: return np.zeros(0)
        u = np.zeros(self.dim, float)
        if (self.idx_alpha is not None) and (self.trend_mode == "deterministic"):
            u[self.idx_alpha] = float(self.det_beta)
        return u

    def _Q(self) -> np.ndarray:
        if self.dim == 0: return np.zeros((0, 0))
        Q = np.zeros((self.dim, self.dim))
        if (self.idx_alpha is not None) and (self.s_alpha > 0): Q[self.idx_alpha, self.idx_alpha] = self.s_alpha**2
        if (self.idx_beta  is not None) and (self.s_beta  > 0): Q[self.idx_beta,  self.idx_beta]  = self.s_beta**2
        if (self.seasonal_mode == "dynamic") and (self.s_gamma > 0): Q[self.idx_g_start, self.idx_g_start] = self.s_gamma**2
        return Q

    def _mu_det_t(self, t: int) -> float:
        out = 0.0
        if self.level_mode == "deterministic":
            out += self.det_alpha
        if (self.trend_mode == "deterministic") and (self.idx_alpha is None):
            out += self.det_beta * t
        if self.seasonal_mode == "deterministic":
            out += float(self.det_gamma[t % self.period])
        return out

    # --------------------------- marginal log-likelihood ---------------------- #

    def _kalman_loglik(self) -> float:
        """p(y|params,model) integrating out states."""
        H, A, Q = self._H(), self._A(), self._Q()
        R = float(self.sigma2)
        if self.dim == 0:
            e = np.array([self.y[t] - self._mu_det_t(t) for t in range(self.T)], float)
            return -0.5 * np.sum(np.log(2 * np.pi * R) + (e * e) / R)

        m0_vec, P0_diag = self._current_m0_P0()
        m = m0_vec.copy()
        C = np.diag(P0_diag) + 1e-12 * np.eye(self.dim)
        ll = 0.0
        u = self._u()
        for t in range(self.T):
            a = A @ m + u
            Rm = A @ C @ A.T + Q
            Rm = 0.5 * (Rm + Rm.T) + 1e-12 * np.eye(self.dim)

            y_det = self._mu_det_t(t)
            S = float(H @ Rm @ H.T + R)
            v = float(self.y[t] - y_det - H @ a)
            ll += -0.5 * (math.log(2 * math.pi) + math.log(S) + (v * v) / S)

            K = (Rm @ H.T) / S
            m = a + K.flatten() * v
            C = Rm - K @ (H @ Rm)
            C = 0.5 * (C + C.T) + 1e-12 * np.eye(self.dim)
        return float(ll)

    # ---------------------------------- FFBS ---------------------------------- #

    def _ffbs(self) -> np.ndarray:
        """Draw latent states with *current* dimension; independent of old holder."""
        if self.dim == 0:
            return np.zeros((self.T + 1, 0), float)

        H, A, Q = self._H(), self._A(), self._Q()
        R = float(self.sigma2)
        m0_vec, P0_diag = self._current_m0_P0()

        m  = np.zeros((self.T + 1, self.dim)); m[0] = m0_vec
        C  = np.zeros((self.T + 1, self.dim, self.dim)); C[0] = np.diag(P0_diag) + 1e-12 * np.eye(self.dim)
        a  = np.zeros((self.T + 1, self.dim))
        Rm = np.zeros((self.T + 1, self.dim, self.dim))
        u  = self._u()

        for t in range(1, self.T + 1):
            a[t]  = A @ m[t - 1] + u
            Rm[t] = A @ C[t - 1] @ A.T + Q
            Rm[t] = 0.5 * (Rm[t] + Rm[t].T) + 1e-12 * np.eye(self.dim)
            y_det = self._mu_det_t(t - 1)
            S     = float(H @ Rm[t] @ H.T + R)
            v     = float(self.y[t - 1] - y_det - H @ a[t])
            K     = (Rm[t] @ H.T) / S
            m[t]  = a[t] + K.flatten() * v
            C[t]  = Rm[t] - K @ (H @ Rm[t])
            C[t]  = 0.5 * (C[t] + C[t].T) + 1e-12 * np.eye(self.dim)

        x = np.zeros((self.T + 1, self.dim), float)
        x[self.T] = np.random.multivariate_normal(m[self.T], C[self.T])
        for t in range(self.T - 1, -1, -1):
            J = C[t] @ A.T
            J = _spd_solve(Rm[t + 1], J.T).T
            mean = m[t] + J @ (x[t + 1] - (A @ m[t] + u))
            cov  = C[t] - J @ Rm[t + 1] @ J.T
            cov  = 0.5 * (cov + cov.T)
            mineig = float(np.linalg.eigvalsh(cov).min())
            if mineig < 1e-12:
                cov += (1e-12 - mineig) * np.eye(cov.shape[0])
            x[t] = np.random.multivariate_normal(mean, cov)
        return x

    def _seed_forward(self, Q_diag: np.ndarray) -> None:
        if self.dim == 0: return
        A, u = self._A(), self._u()
        for t in range(1, self.T + 1):
            self.x[t] = A @ self.x[t - 1] + u + np.random.normal(0.0, np.sqrt(Q_diag), size=self.dim)

    # -------------------------- conjugate parameter steps --------------------- #

    @staticmethod
    def _rinvgamma(shape: float, scale: float) -> float:
        return 1.0 / np.random.gamma(shape, 1.0 / scale)

    def _mu_vec(self) -> np.ndarray:
        if self.dim == 0:
            return np.array([self._mu_det_t(t) for t in range(self.T)], float)
        H = self._H()
        mu = np.zeros(self.T, float)
        for t in range(1, self.T + 1):
            mu[t - 1] = self._mu_det_t(t - 1) + float(H @ self.x[t])
        return mu

    def update_sigma2(self) -> None:
        e = self.y - self._mu_vec()
        a = self.priors.a_sigma + 0.5 * self.T
        b = self.priors.b_sigma + 0.5 * float(e @ e)
        tau = np.random.gamma(shape=a, scale=1.0 / b)  # precision
        self.sigma2 = 1.0 / max(tau, 1e-300)

    # innovation sums of squares (for Half-Cauchy mixture updates)
    def _innovation_ss_alpha(self) -> Tuple[float, int]:
        if self.idx_alpha is None: return 0.0, 0
        ss = 0.0
        for t in range(1, self.T + 1):
            drift = 0.0
            if self.idx_beta is not None:
                drift = self.x[t - 1, self.idx_beta]
            elif self.trend_mode == "deterministic":
                drift = float(self.det_beta)
            mean = self.x[t - 1, self.idx_alpha] + drift
            ss += (self.x[t, self.idx_alpha] - mean) ** 2
        return float(ss), self.T

    def _innovation_ss_beta(self) -> Tuple[float, int]:
        if self.idx_beta is None: return 0.0, 0
        d = self.x[1:, self.idx_beta] - self.x[:-1, self.idx_beta]
        return float(np.sum(d * d)), self.T

    def _innovation_ss_gamma(self) -> Tuple[float, int]:
        if self.seasonal_mode != "dynamic": return 0.0, 0
        gs, ge = self.idx_g_start, self.idx_g_end
        ss = 0.0
        for t in range(1, self.T + 1):
            prev = self.x[t - 1, gs:ge + 1]
            mean_new = -float(np.sum(prev))
            ss += (self.x[t, gs] - mean_new) ** 2
        return float(ss), self.T

    def update_process_Q_halfcauchy(self) -> None:
        # alpha
        if self.idx_alpha is not None:
            SS, T_eff = self._innovation_ss_alpha()
            A = float(self.priors.hc_scale_alpha)
            Q_alpha = self._rinvgamma(0.5 * T_eff + 0.5, 0.5 * SS + 1.0 / max(self._a_alpha, 1e-300))
            self.s_alpha = math.sqrt(max(Q_alpha, 0.0))
            self._a_alpha = self._rinvgamma(1.0, (1.0 / (A * A)) + 1.0 / max(Q_alpha, 1e-300))
        # beta
        if self.idx_beta is not None:
            SS, T_eff = self._innovation_ss_beta()
            A = float(self.priors.hc_scale_beta)
            Q_beta = self._rinvgamma(0.5 * T_eff + 0.5, 0.5 * SS + 1.0 / max(self._a_beta, 1e-300))
            self.s_beta = math.sqrt(max(Q_beta, 0.0))
            self._a_beta = self._rinvgamma(1.0, (1.0 / (A * A)) + 1.0 / max(Q_beta, 1e-300))
        # gamma
        if self.seasonal_mode == "dynamic":
            SS, T_eff = self._innovation_ss_gamma()
            A = float(self.priors.hc_scale_gamma)
            Q_gamma = self._rinvgamma(0.5 * T_eff + 0.5, 0.5 * SS + 1.0 / max(self._a_gamma, 1e-300))
            self.s_gamma = math.sqrt(max(Q_gamma, 0.0))
            self._a_gamma = self._rinvgamma(1.0, (1.0 / (A * A)) + 1.0 / max(Q_gamma, 1e-300))

    @staticmethod
    def _gibbs_m0_scalar(x0: float, m_prior: float, s_prior: float, P0: float) -> float:
        prec = 1.0 / (s_prior**2) + 1.0 / max(P0, 1e-18)
        var = 1.0 / prec
        mean = var * (m_prior / (s_prior**2) + x0 / max(P0, 1e-18))
        return float(np.random.normal(mean, math.sqrt(var)))

    def update_m0(self) -> None:
        if self.dim == 0: return
        pos = 0
        if self.idx_alpha is not None:
            self.m0_alpha = self._gibbs_m0_scalar(
                float(self.x[0, pos]), self.priors.m_m0_alpha, self.priors.s_m0_alpha, self.P0_alpha
            )
            pos += 1
        if self.idx_beta is not None:
            self.m0_beta = self._gibbs_m0_scalar(
                float(self.x[0, pos]), self.priors.m_m0_beta, self.priors.s_m0_beta, self.P0_beta
            )
            pos += 1
        if self.seasonal_mode == "dynamic":
            base = np.zeros(self.period - 1) if self.priors.m_m0_gamma is None \
                   else np.asarray(self.priors.m_m0_gamma, float)
            s = float(self.priors.s_m0_gamma)
            for k in range(self.period - 1):
                self.m0_gamma[k] = self._gibbs_m0_scalar(float(self.x[0, pos + k]), float(base[k]), s, self.P0_gamma)

    def update_P0(self) -> None:
        if self.dim == 0: return
        pos = 0
        if self.idx_alpha is not None:
            a = self.priors.a_P0_alpha + 0.5
            b = self.priors.b_P0_alpha + 0.5 * (float(self.x[0, pos]) - self.m0_alpha) ** 2
            self.P0_alpha = 1.0 / np.random.gamma(a, 1.0 / b); pos += 1
        if self.idx_beta is not None:
            a = self.priors.a_P0_beta + 0.5
            b = self.priors.b_P0_beta + 0.5 * (float(self.x[0, pos]) - self.m0_beta) ** 2
            self.P0_beta  = 1.0 / np.random.gamma(a, 1.0 / b); pos += 1
        if self.seasonal_mode == "dynamic":
            diffsq = 0.0
            for k in range(self.period - 1):
                diffsq += (float(self.x[0, pos + k]) - float(self.m0_gamma[k])) ** 2
            a = self.priors.a_P0_gamma + 0.5 * (self.period - 1)
            b = self.priors.b_P0_gamma + 0.5 * diffsq
            self.P0_gamma = 1.0 / np.random.gamma(a, 1.0 / b)

    def update_deterministic_params(self) -> None:
        # level intercept
        if self.level_mode == "deterministic":
            r = self.y.copy()
            if self.dim > 0:
                H = self._H()
                for t in range(1, self.T + 1):
                    r[t - 1] -= float(H @ self.x[t])
            if self.seasonal_mode == "deterministic":
                r -= self.det_gamma[np.arange(self.T) % self.period]
            s2 = float(self.sigma2); m0 = float(self.priors.m_m0_alpha); s0 = float(self.priors.s_m0_alpha)
            prec = self.T / s2 + 1.0 / (s0 * s0)
            mean = ((r.sum() / s2) + m0 / (s0 * s0)) / prec
            self.det_alpha = float(np.random.normal(mean, math.sqrt(1.0 / prec)))

        # trend slope
        if self.trend_mode == "deterministic":
            if self.idx_alpha is not None:
                d = self.x[1:, self.idx_alpha] - self.x[:-1, self.idx_alpha]
                s2 = float(self.s_alpha**2) if self.s_alpha > 0 else 1e-12
                m0 = float(self.priors.m_m0_beta); s0 = float(self.priors.s_m0_beta)
                prec = self.T / s2 + 1.0 / (s0 * s0)
                mean = ((float(np.sum(d)) / s2) + m0 / (s0 * s0)) / prec
                self.det_beta = float(np.random.normal(mean, math.sqrt(1.0 / prec)))
            else:
                tvec = np.arange(self.T, dtype=float)
                r = self.y.copy()
                if self.dim > 0:
                    H = self._H()
                    for k in range(1, self.T + 1):
                        r[k - 1] -= float(H @ self.x[k])
                if self.level_mode == "deterministic":
                    r -= self.det_alpha
                if self.seasonal_mode == "deterministic":
                    r -= self.det_gamma[np.arange(self.T) % self.period]
                s2 = float(self.sigma2); m0 = float(self.priors.m_m0_beta); s0 = float(self.priors.s_m0_beta)
                prec = (tvec @ tvec) / s2 + 1.0 / (s0 * s0)
                mean = ((tvec @ r) / s2 + m0 / (s0 * s0)) / prec
                self.det_beta = float(np.random.normal(mean, math.sqrt(1.0 / prec)))

        # seasonal (sum-to-zero coding with K=p−1 free dummies)
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
                r -= self.det_alpha
            if (self.idx_alpha is None) and (self.trend_mode == "deterministic"):
                r -= self.det_beta * np.arange(self.T, dtype=float)

            K = self.period - 1
            mu_prior = np.zeros(K) if self.priors.m_m0_gamma is None \
                       else np.asarray(self.priors.m_m0_gamma, float).reshape(-1)
            s2p = float(self.priors.s_m0_gamma) ** 2
            Z = self._Z_season; sig2 = float(self.sigma2)
            Prec = (Z.T @ Z) / sig2 + np.eye(K) / s2p
            b = (Z.T @ r) / sig2 + mu_prior / s2p
            mu = np.linalg.solve(Prec, b)
            L = np.linalg.cholesky(Prec)
            theta = mu + np.linalg.solve(L.T, np.random.randn(K))
            self.det_gamma = np.r_[theta, -theta.sum()]

    # --------------------------------- RJ moves -------------------------------- #

    def _snapshot(self) -> dict:
        return {
            "level": self.level_mode, "trend": self.trend_mode, "season": self.seasonal_mode,
            "sigma2": self.sigma2,
            "s_alpha": self.s_alpha, "s_beta": self.s_beta, "s_gamma": self.s_gamma,
            "a_alpha": self._a_alpha, "a_beta": self._a_beta, "a_gamma": self._a_gamma,
            "m0_alpha": self.m0_alpha, "P0_alpha": self.P0_alpha,
            "m0_beta": self.m0_beta, "P0_beta": self.P0_beta,
            "m0_gamma": None if self.m0_gamma is None else self.m0_gamma.copy(),
            "P0_gamma": self.P0_gamma,
            "det_alpha": self.det_alpha, "det_beta": self.det_beta,
            "det_gamma": None if getattr(self, "det_gamma", None) is None else self.det_gamma.copy(),
        }

    def _load_snapshot(self, S: dict) -> None:
        self.level_mode = S["level"]; self.trend_mode = S["trend"]; self.seasonal_mode = S["season"]
        self.sigma2 = float(S["sigma2"])
        self.s_alpha = float(S["s_alpha"]); self.s_beta = float(S["s_beta"]); self.s_gamma = float(S["s_gamma"])
        self._a_alpha = float(S["a_alpha"]); self._a_beta = float(S["a_beta"]); self._a_gamma = float(S["a_gamma"])
        self.m0_alpha = float(S["m0_alpha"]); self.P0_alpha = float(S["P0_alpha"])
        self.m0_beta  = float(S["m0_beta"]);  self.P0_beta  = float(S["P0_beta"])
        self.m0_gamma = None if S["m0_gamma"] is None else np.asarray(S["m0_gamma"], float).copy()
        self.P0_gamma = float(S["P0_gamma"])
        self.det_alpha = float(S["det_alpha"]); self.det_beta = float(S["det_beta"])
        dg = S["det_gamma"]; self.det_gamma = None if dg is None else np.asarray(dg, float).copy()
        self._rebuild_layout()

    def _rj_record(self, block: str, accepted: bool) -> None:
        s = self.rj_stats[block]
        s["proposed"] += 1
        if accepted: s["accepted"] += 1
        s["win"].append(1 if accepted else 0)

    def _fmt_rj_block(self, block: str) -> str:
        s = self.rj_stats[block]
        prop = max(1, int(s["proposed"]))
        acc  = int(s["accepted"])
        rate = 100.0 * acc / prop
        wrate = 100.0 * (sum(s["win"]) / max(1, len(s["win"])))
        return f"{rate:.1f}%"

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
        return np.random.choice(choices)

    def _legal_modes(self, level: str, trend: str, season: str) -> bool:
        if trend == "dynamic" and level != "dynamic": return False
        if (not self.cfg.allow_none_level) and level == "none": return False
        return True

    def _draw_prior_dyn_block(self, which: str) -> None:
        # Half-Cauchy via IG mixture: a ~ InvGamma(1, 1/A^2); Q ~ InvGamma(1/2, 1/a)
        if which == "level":
            A = float(self.priors.hc_scale_alpha)
            a_aux = self._rinvgamma(1.0, 1.0 / (A * A))
            Q = self._rinvgamma(0.5, 1.0 / max(a_aux, 1e-300))
            self._a_alpha = a_aux; self.s_alpha = math.sqrt(max(Q, 1e-18))
            self.m0_alpha = float(np.random.normal(self.priors.m_m0_alpha, self.priors.s_m0_alpha))
            self.P0_alpha = 1.0 / np.random.gamma(self.priors.a_P0_alpha, 1.0 / self.priors.b_P0_alpha)
        elif which == "trend":
            A = float(self.priors.hc_scale_beta)
            a_aux = self._rinvgamma(1.0, 1.0 / (A * A))
            Q = self._rinvgamma(0.5, 1.0 / max(a_aux, 1e-300))
            self._a_beta = a_aux; self.s_beta = math.sqrt(max(Q, 1e-18))
            self.m0_beta = float(np.random.normal(self.priors.m_m0_beta, self.priors.s_m0_beta))
            self.P0_beta = 1.0 / np.random.gamma(self.priors.a_P0_beta, 1.0 / self.priors.b_P0_beta)
        elif which == "season":
            A = float(self.priors.hc_scale_gamma)
            a_aux = self._rinvgamma(1.0, 1.0 / (A * A))
            Q = self._rinvgamma(0.5, 1.0 / max(a_aux, 1e-300))
            self._a_gamma = a_aux; self.s_gamma = math.sqrt(max(Q, 1e-18))
            base = np.zeros(self.period - 1) if self.priors.m_m0_gamma is None \
                   else np.asarray(self.priors.m_m0_gamma, float)
            s = float(self.priors.s_m0_gamma)
            self.m0_gamma = np.random.normal(base, s, size=self.period - 1)
            self.P0_gamma = 1.0 / np.random.gamma(self.priors.a_P0_gamma, 1.0 / self.priors.b_P0_gamma)

    def _maybe_print_switch(self, it: int, before: Tuple[str,str,str], after: Tuple[str,str,str],
                            block: str, dll: float, accepted: bool) -> None:
        if not accepted or before == after: return
        bL,bT,bS = before; aL,aT,aS = after
        changed = []
        if bL != aL: changed.append(f"L:{bL[:3]}→{aL[:3]}")
        if bT != aT: changed.append(f"T:{bT[:3]}→{aT[:3]}")
        if bS != aS: changed.append(f"S:{bS[:3]}→{aS[:3]}")
        msg = (f"[switch @ it {it+1}] block={block} | "
               f"{' '.join(changed)} | Δloglik={dll:+.4f} | RJ {self._fmt_rj_all()}")
        print(msg)
        self._last_modes = after

    def _rj_move_one(self, it: int) -> None:
        comps = ["level", "trend", "season"]
        comp = np.random.choice(comps)

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
        # propose parameters needed for births
        if comp == "level" and new_level == "dynamic" and level != "dynamic":
            self._draw_prior_dyn_block("level")
        if comp == "trend" and new_trend == "dynamic" and trend != "dynamic":
            self._draw_prior_dyn_block("trend")
        if comp == "season" and new_season == "dynamic" and season != "dynamic":
            self._draw_prior_dyn_block("season")

        # apply proposed modes
        self.level_mode, self.trend_mode, self.seasonal_mode = new_level, new_trend, new_season
        if self.level_mode != "dynamic": self.s_alpha = 0.0
        if self.trend_mode != "dynamic": self.s_beta  = 0.0
        if self.seasonal_mode != "dynamic": self.s_gamma = 0.0
        if self.seasonal_mode == "deterministic" and getattr(self, "det_gamma", None) is None:
            base = np.zeros(self.period - 1) if self.priors.m_m0_gamma is None \
                   else np.asarray(self.priors.m_m0_gamma, float)
            self.det_gamma = np.r_[base, -float(np.sum(base))].astype(float)
        self._rebuild_layout()

        ll_prop = self._kalman_loglik()
        self._load_snapshot(cur_snap)
        ll_cur  = self._kalman_loglik()
        dll = ll_prop - ll_cur

        accept = (np.log(np.random.rand()) < dll)
        self._rj_record(comp, bool(accept))

        if accept:
            # adopt proposal
            self._load_snapshot(cur_snap)
            before = (self.level_mode, self.trend_mode, self.seasonal_mode)

            self.level_mode, self.trend_mode, self.seasonal_mode = new_level, new_trend, new_season
            if self.level_mode != "dynamic": self.s_alpha = 0.0
            if self.trend_mode != "dynamic": self.s_beta  = 0.0
            if self.seasonal_mode != "dynamic": self.s_gamma = 0.0
            if self.seasonal_mode == "deterministic" and getattr(self, "det_gamma", None) is None:
                base = np.zeros(self.period - 1) if self.priors.m_m0_gamma is None \
                       else np.asarray(self.priors.m_m0_gamma, float)
                self.det_gamma = np.r_[base, -float(np.sum(base))].astype(float)

            self._rebuild_layout()
            # IMPORTANT: re-allocate latent holder with current dim after an accepted RJ move
            self._alloc_state_holder()

            after = (self.level_mode, self.trend_mode, self.seasonal_mode)
            self._maybe_print_switch(it, before, after, comp, dll, accepted=True)

    # ------------------------------- bookkeeping ----------------------------- #

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

    # -------------------------------- progress -------------------------------- #

    def _progress_line(self, it: int) -> str:
        parts = [
            f"[it {it + 1}/{self.cfg.n_iter}]",
            f"σ={math.sqrt(self.sigma2):.3f}",
            f"L/T/S={self.level_mode[:3]}/{self.trend_mode[:3]}/{self.seasonal_mode[:3]}",
        ]

        # Q's (process variances) only for dynamic blocks
        if self.idx_alpha is not None:
            parts.append(f"Qα={self.s_alpha**2:.4g}")
        if self.idx_beta is not None:
            parts.append(f"Qβ={self.s_beta**2:.4g}")
        if self.seasonal_mode == "dynamic":
            parts.append(f"Qγ={self.s_gamma**2:.4g}")

        # LEVEL block
        if self.level_mode == "dynamic":
            parts.append(f"m0α={self.m0_alpha:.4g} P0α={self.P0_alpha:.4g}")
        elif self.level_mode == "deterministic":
            parts.append(f"α(det)={self.det_alpha:.4g}")

        # TREND block
        if self.trend_mode == "dynamic":
            parts.append(f"m0β={self.m0_beta:.4g} P0β={self.P0_beta:.4g}")
        elif self.trend_mode == "deterministic":
            parts.append(f"β(det)={self.det_beta:.4g}")

        # SEASONAL block
        if self.seasonal_mode == "dynamic":
            gtxt = _fmt_list(self.m0_gamma, 6, ".4g")
            parts.append(f"m0γ={gtxt} P0γ={self.P0_gamma:.4g}")
        elif self.seasonal_mode == "deterministic":
            gtxt = _fmt_list(self.det_gamma[:-1], 6, ".4g")
            parts.append(f"γ(det)={gtxt}")

        parts.append(f"RJ {self._fmt_rj_all()}")
        return " | ".join(parts)

    # ----------------------------------- run ---------------------------------- #

    def run(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        save_iters = list(range(cfg.burn, cfg.n_iter, cfg.thin))
        n_kept, keep_idx = len(save_iters), 0

        # allocate storage
        self.keep = {
            "sigma": np.zeros(n_kept, float),
            "mu": np.zeros((n_kept, self.T), float),
            "modes": np.zeros((n_kept, 3), int)  # 0:dyn,1:det,2:none
        }
        # Note: we start from current dim; we always start at maximal width in most runs.
        x_width = max(self.dim, 1)
        self.keep["x"] = np.zeros((n_kept, self.T, x_width))

        if self.level_mode == "dynamic":
            self.keep.update({"Q_alpha": np.zeros(n_kept),
                              "m0_alpha": np.zeros(n_kept),
                              "P0_alpha": np.zeros(n_kept)})
        if self.trend_mode == "dynamic":
            self.keep.update({"Q_beta": np.zeros(n_kept),
                              "m0_beta": np.zeros(n_kept),
                              "P0_beta": np.zeros(n_kept)})
        if self.seasonal_mode == "dynamic":
            self.keep.update({"Q_gamma": np.zeros(n_kept),
                              "m0_gamma": np.zeros((n_kept, self.period - 1)),
                              "P0_gamma": np.zeros(n_kept)})

        print_every = cfg.progress_every if cfg.progress_every > 0 else max(1, cfg.n_iter // 50) or 1

        for it in range(cfg.n_iter):
            # states
            if self.dim > 0:
                self.x = self._ffbs()

            # process variances
            if self.dim > 0:
                self.update_process_Q_halfcauchy()

            # initial means/vars
            if self.dim > 0:
                self.update_m0()
                self.update_P0()

            # deterministic params
            self.update_deterministic_params()
            # observation variance
            self.update_sigma2()

            # RJ toggles
            for _ in range(cfg.rj_moves_per_iter):
                self._rj_move_one(it)

            # mode tally
            self._tally_modes()

            # periodic progress
            if cfg.progress and ((it + 1) % print_every == 0 or it == cfg.n_iter - 1):
                print(self._progress_line(it))

            # save draws
            if it in save_iters:
                mu = self._mu_vec()
                self.keep["mu"][keep_idx, :] = mu
                self.keep["sigma"][keep_idx] = math.sqrt(self.sigma2)
                enc = lambda s: 0 if s == "dynamic" else (1 if s == "deterministic" else 2)
                self.keep["modes"][keep_idx, :] = np.array(
                    [enc(self.level_mode), enc(self.trend_mode), enc(self.seasonal_mode)], int
                )
                if self.idx_alpha is not None and "Q_alpha" in self.keep:
                    self.keep["Q_alpha"][keep_idx] = self.s_alpha**2
                    self.keep["m0_alpha"][keep_idx] = self.m0_alpha
                    self.keep["P0_alpha"][keep_idx] = self.P0_alpha
                if self.idx_beta is not None and "Q_beta" in self.keep:
                    self.keep["Q_beta"][keep_idx] = self.s_beta**2
                    self.keep["m0_beta"][keep_idx] = self.m0_beta
                    self.keep["P0_beta"][keep_idx] = self.P0_beta
                if self.seasonal_mode == "dynamic" and "Q_gamma" in self.keep:
                    self.keep["Q_gamma"][keep_idx] = self.s_gamma**2
                    self.keep["m0_gamma"][keep_idx, :] = self.m0_gamma
                    self.keep["P0_gamma"][keep_idx] = self.P0_gamma
                if self.dim > 0:
                    w = min(self.keep["x"].shape[2], self.x.shape[1])
                    self.keep["x"][keep_idx, :, :w] = self.x[1:self.T + 1, :w]
                keep_idx += 1

        return self.keep

    # ----------------------------------- I/O ---------------------------------- #

    def save_posterior(self, out_npz_path: str, extra_meta: Optional[dict] = None) -> None:
        os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)
        arrays = dict(self.keep); arrays["y"] = self.y.copy()
        np.savez_compressed(out_npz_path, **arrays)
        meta = {
            "T": int(self.T), "period": int(self.period),
            "cfg": asdict(self.cfg), "priors": asdict(self.priors),
            "rj_accept": {
                b: {
                    "proposed": int(self.rj_stats[b]["proposed"]),
                    "accepted": int(self.rj_stats[b]["accepted"]),
                    "acc_rate": (self.rj_stats[b]["accepted"] / max(1, self.rj_stats[b]["proposed"]))
                } for b in ("level", "trend", "season")
            }
        }
        if extra_meta: meta.update(extra_meta)
        with open(out_npz_path.replace(".npz", ".meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

# ------------------------- CLI / Example run (RJ + Gibbs) ------------------ #
if __name__ == "__main__":
    import argparse, os, sys, time
    from datetime import datetime
    import numpy as np
    import matplotlib.pyplot as plt

    # local imports
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(base_dir)
    from simulator.mean_time_series import Mean_Time_Series  # newest-first convention

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

    # --------------------------------------------------------------------------
    # Command-line arguments
    # --------------------------------------------------------------------------
    p = argparse.ArgumentParser(
        description=(
            "Gaussian structural TS with RJ-MCMC + conjugate Gibbs. "
            "Components for SIMULATOR: level/trend/season with dynamic/deterministic/(optional none). "
            "Sampler ALWAYS starts at dynamic/dynamic/dynamic."
        )
    )

    # ----- Simulation (data-generating truth) -----
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--period", type=int, default=4)
    p.add_argument("--start-date", type=str, default="2000-01-01")
    p.add_argument("--level-mode", choices=["dynamic", "deterministic"], default="dynamic")
    p.add_argument("--trend-mode", choices=["dynamic", "deterministic", "none"], default="deterministic")
    p.add_argument("--seasonal-mode", choices=["dynamic", "deterministic", "none"], default="deterministic")
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--q-level", type=float, default=0.001)
    p.add_argument("--q-trend", type=float, default=0.00002)
    p.add_argument("--q-season", type=float, default=0.00005)
    p.add_argument("--m0-level", type=float, default=3.0)
    p.add_argument("--v0-level", type=float, default=0.25)
    p.add_argument("--m0-trend", type=float, default=0.015)
    p.add_argument("--v0-trend", type=float, default=0.05)
    p.add_argument("--m0-season", type=str, default=None)
    p.add_argument("--v0-season", type=str, default=None)

    # ----- Priors -----
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

    # Half-Cauchy scales for process SDs (A_k). Tip: 0.5 is a good default.
    p.add_argument("--hc-scale-alpha", type=float, default=1)
    p.add_argument("--hc-scale-beta",  type=float, default=1)
    p.add_argument("--hc-scale-gamma", type=float, default=1)

    # ----- Sampler configuration -----
    p.add_argument("--n-iter", type=int, default=30000)
    p.add_argument("--burn", type=int, default=10000)
    p.add_argument("--thin", type=int, default=2)
    p.add_argument("--seed", type=int, default=40)
    p.add_argument("--progress", default=True)
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--out-dir", type=str, default="results/simulations/DLM_RJ")
    p.add_argument("--plot", default=True)
    p.add_argument("--print-summary", default=True)

    # Initial inference values (starting point of MCMC)
    p.add_argument("--sigma-init", type=float, default=1.0)
    p.add_argument("--s-alpha-init", type=float, default=1e-1)
    p.add_argument("--s-beta-init",  type=float, default=1e-1)
    p.add_argument("--s-gamma-init", type=float, default=1e-1)
    p.add_argument("--m0-gamma-init", type=str, default=None)
    p.add_argument("--P0-alpha-init", type=float, default=0.25)
    p.add_argument("--P0-beta-init",  type=float, default=0.05)
    p.add_argument("--P0-gamma-init", type=float, default=0.25)

    # RJ options
    p.add_argument("--rj-moves-per-iter", type=int, default=2)
    p.add_argument("--allow-none-level", default=False, help="Allow level='none' (usually False).")
    p.add_argument("--allow-none-trend", default=True, help="Allow trend='none'.")
    p.add_argument("--allow-none-season", default=False, help="Allow season='none'.")

    args = p.parse_args()
    np.random.seed(args.seed)

    # ---- Simulate data from Mean_Time_Series ----
    start_date = _parse_date(args.start_date)
    m0_season = _csv_floats_or_none(args.m0_season) or [1.0] * (args.period - 1)
    v0_season = _csv_floats_or_none(args.v0_season) or [0.25] * (args.period - 1)

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
    mu_T   = truths["mu_t"][1:1 + args.T]
    dates_T= truths["index"][:args.T]

    # ---- Priors ----
    pri_gamma_vec = _csv_floats_or_none(args.prior_m_m0_gamma)
    priors = Priors(
        a_sigma=args.prior_a_sigma, b_sigma=args.prior_b_sigma,
        m_m0_alpha=args.prior_m_m0_alpha, s_m0_alpha=args.prior_s_m0_alpha,
        m_m0_beta=args.prior_m_m0_beta,   s_m0_beta=args.prior_s_m0_beta,
        m_m0_gamma=None if pri_gamma_vec is None else pri_gamma_vec,
        s_m0_gamma=args.prior_s_m0_gamma,
        a_P0_alpha=args.prior_a_P0_alpha, b_P0_alpha=args.prior_b_P0_alpha,
        a_P0_beta=args.prior_a_P0_beta,   b_P0_beta=args.prior_b_P0_beta,
        a_P0_gamma=args.prior_a_P0_gamma, b_P0_gamma=args.prior_b_P0_gamma,
        hc_scale_alpha=args.hc_scale_alpha,
        hc_scale_beta=args.hc_scale_beta,
        hc_scale_gamma=args.hc_scale_gamma,
    )

    # ---- Sampler config (RJ + Gibbs) ----
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

    # Initial values for dynamic blocks (seasonal m0 if needed)
    m0_gamma_init = (
        [float(z) for z in args.m0_gamma_init.split(",")] if args.m0_gamma_init else None
    )

    # ---- Build sampler: ALWAYS start dyn/dyn/dyn ----
    sampler = DLMRJGibbs(
        y=y, period=args.period,
        level_mode="dynamic", trend_mode="dynamic", seasonal_mode="dynamic",
        sigma2_init=args.sigma_init ** 2,
        s_alpha_init=args.s_alpha_init, s_beta_init=args.s_beta_init, s_gamma_init=args.s_gamma_init,
        m0_alpha_init=args.m0_level, m0_beta_init=args.m0_trend,
        m0_gamma_init=m0_gamma_init, P0_alpha_init=args.P0_alpha_init,
        P0_beta_init=args.P0_beta_init, P0_gamma_init=args.P0_gamma_init,
        priors=priors, cfg=cfg,
    )

    if args.print_summary:
        print(f"\nSimulated {args.T} observations (σ={mts.sigma}) with TRUE modes "
              f"{args.level_mode}/{args.trend_mode}/{args.seasonal_mode}.")
        print("Sampler START modes: dynamic/dynamic/dynamic\n")
        print("Half-Cauchy scales (A_k) for process SDs:")
        print(f"  A_alpha={priors.hc_scale_alpha:.3f}, A_beta={priors.hc_scale_beta:.3f}, "
              f"A_gamma={priors.hc_scale_gamma:.3f}\n")

    # ---- Run MCMC ----
    t0 = time.time()
    post = sampler.run()
    elapsed = time.time() - t0
    print(f"\n[Run completed in {elapsed:.1f}s]")

    # ---- Output directory ----
    out_dir = os.path.join(
        args.out_dir,
        f"RJ-start_dyn-dyn-dyn__"
        f"truth_{args.level_mode}-{args.trend_mode}-{args.seasonal_mode}__"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(out_dir, exist_ok=True)

    # Save posterior arrays + metadata
    sampler.save_posterior(
        out_npz_path=os.path.join(out_dir, "posterior.npz"),
        extra_meta={
            "elapsed_seconds": float(elapsed),
            "sim_truth_modes": {
                "level": args.level_mode, "trend": args.trend_mode, "season": args.seasonal_mode
            },
            "start_modes": {
                "level": "dynamic", "trend": "dynamic", "season": "dynamic"
            },
            "hc_scales": {
                "alpha": priors.hc_scale_alpha,
                "beta": priors.hc_scale_beta,
                "gamma": priors.hc_scale_gamma,
            },
        },
    )

    # ---- Summary ----
    if args.print_summary:
        print("\n--- Posterior means ---")
        print(f"σ = {np.mean(post['sigma']):.4f}")
        for k in ["alpha", "beta", "gamma"]:
            key = f"Q_{k}"
            if key in post:
                mQ = np.mean(post[key])
                print(f"Q_{k} = {mQ:.4g} (√Q ≈ {math.sqrt(mQ):.4g})")

        # Posterior inclusion probabilities from RJ visit frequencies
        inc = sampler.inclusion_probabilities()
        def _fmt_probs(d):
            return ", ".join([f"{k}={v:.3f}" for k,v in d.items()])
        print("\n--- Posterior inclusion probabilities (visit frequencies) ---")
        print("Level:  " + _fmt_probs(inc["level"]))
        print("Trend:  " + _fmt_probs(inc["trend"]))
        print("Season: " + _fmt_probs(inc["season"]))

    # ---- Plot ----
    if args.plot:
        mu_hat = post["mu"].mean(axis=0)
        plt.figure(figsize=(10, 4))
        plt.plot(dates_T, y, label="y_t", lw=1)
        plt.plot(dates_T, mu_T, "--", label="μ_t (truth)")
        plt.plot(dates_T, mu_hat, "-.", label="μ̂_t (post mean)")
        plt.title(
            "DLM RJ (current modes: "
            f"{sampler.level_mode}/{sampler.trend_mode}/{sampler.seasonal_mode})"
        )
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fit.png"), dpi=160)
        plt.show()

    # Also save inclusion probabilities to JSON
    inc = sampler.inclusion_probabilities()
    with open(os.path.join(out_dir, "inclusion_probs.json"), "w", encoding="utf-8") as f:
        json.dump(inc, f, indent=2)
    print(f"[save] Outputs written to: {out_dir}")
