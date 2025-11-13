from __future__ import annotations

import json, math, os, time, warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple
from harmonic_helpers import (
        center_and_report_dummies_full,
        dummies_full_to_harmonics_fft,
        harmonics_to_dummies_full_fft,
    )
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
EPS = 1e-12

# =============================================================================
# Small utils
# =============================================================================

def _mad(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    if v.size == 0: return 0.0
    m = np.median(v)
    return float(np.median(np.abs(v - m)))

def _robust_sd(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    return _mad(v) / 1.4826 if v.size else 0.0

def _spd_solve(M: np.ndarray, B: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    n = M.shape[0]
    I = np.eye(n)
    for k in range(3):
        try:
            L = np.linalg.cholesky(0.5*(M+M.T) + (eps * (10**k)) * I)
            Y = np.linalg.solve(L, B)
            return np.linalg.solve(L.T, Y)
        except np.linalg.LinAlgError:
            continue
    return np.linalg.pinv(M) @ B

def _fmt_list(vals, max_elems: int = 6, fmt: str = ".4g", sep: str = ", ", mode: str = "head") -> str:
    if vals is None: return "-"
    v = np.asarray(vals, float).ravel()
    n = v.size
    if n == 0: return "[]"
    def _one(x):
        if isinstance(x, (float, np.floating)):
            if np.isnan(x): return "nan"
            if np.isposinf(x): return "inf"
            if np.isneginf(x): return "-inf"
        return f"{x:{fmt}}"
    if n <= max_elems: return "[" + sep.join(_one(x) for x in v) + "]"
    if mode == "both" and max_elems >= 3:
        k_head = max_elems // 2
        k_tail = max_elems - k_head
        head = sep.join(_one(x) for x in v[:k_head])
        tail = sep.join(_one(x) for x in v[-k_tail:])
        return f"[{head}{sep}…{sep}{tail}]"
    head = sep.join(_one(x) for x in v[:max_elems])
    return f"[{head}{sep}…]"
# =============================================================================
# Priors & Config
# =============================================================================

@dataclass
class Priors:
    # observation precision tau ~ Gamma(a, b) (shape–rate)  ⇒  σ² = 1/τ
    a_sigma: float = 2.0
    b_sigma: float = 1.0

    # m0 priors (Normal)
    m_m0_alpha: float = 0.0
    s_m0_alpha: float = 10.0
    m_m0_beta: float  = 0.0
    s_m0_beta: float  = 10.0
    m_m0_nyq:  float  = 0.0
    # harmonic means (cos/sin) optional; sd shared
    m_m0_cos: Optional[Sequence[float]] = None
    m_m0_sin: Optional[Sequence[float]] = None
    s_m0_harm: float = 5.0

    # P0 priors (Inv-Gamma on variance)
    a_P0_alpha: float = 2.0
    b_P0_alpha: float = 1.0
    a_P0_beta:  float = 2.0
    b_P0_beta:  float = 1.0
    a_P0_harm:  float = 2.0
    b_P0_harm:  float = 1.0

    # Half-Cauchy scales (IG mixture) for process SDs
    hc_scale_alpha: float = 0.5
    hc_scale_beta:  float = 0.5
    hc_scale_gamma: float = 0.5

@dataclass
class SamplerConfig:
    n_iter: int = 20000
    burn:   int = 5000
    thin:   int = 2
    random_seed: Optional[int] = 42
    progress: bool = True
    progress_every: int = 0  # if 0, ~2%
    print_dummies_every: int = 0  # if >0, print seasonal dummies every N iters

    # RJ tuning
    rj_moves_per_iter: int = 2
    allow_none_level:  bool = False
    allow_none_trend:  bool = True
    allow_none_season: bool = True
    rj_window: int = 500  # rolling window for printed acc-rates

# =============================================================================
# RJ–Gibbs DLM with harmonic seasonality
# =============================================================================

class DLMRJGibbsHarmonic:
    """
    Gaussian structural DLM with level/trend/season and harmonic seasonal states,
    equipped with RJ–MCMC over block modes: {dynamic, deterministic, none}.
    Seasonals use K cosine/sine pairs (+ optional Nyquist for even period).

    State (when dynamic):
        [alpha?] [beta?] [c1 s1 | c2 s2 | ... | cK sK | nyq?]

    Observation:
        y_t = μ_det(t) + H x_t + ε_t,   ε_t ~ N(0, σ²)
        H loads alpha and all cosine states (and Nyquist if used).
    """

    # --------------------------- Construction --------------------------- #
    def __init__(
        self,
        y: np.ndarray,
        period: int,
        harmonics: Optional[int] = None,
        use_nyquist: Optional[bool] = None,
        level_mode: str = "deterministic",
        trend_mode: str = "deterministic",
        seasonal_mode: str = "deterministic",
        # Initial values (used as priors if dynamic; fixed params if deterministic)
        m0_alpha_init: float = 0.0, P0_alpha_init: float = 1.0,
        m0_beta_init: float  = 0.0, P0_beta_init:  float = 1.0,
        m0_cos_init: Optional[Sequence[float]] = None,
        m0_sin_init: Optional[Sequence[float]] = None,
        m0_nyq_init: Optional[float] = 0.0,
        P0_harm_init: float = 1.0,
        # observation variance + process SD inits
        sigma2_init: float = 1.0,
        s_alpha_init: float = 1e-2,
        s_beta_init:  float = 1e-3,
        s_gamma_init: float = 1e-3,
        # priors / config
        priors: Priors = Priors(),
        cfg: SamplerConfig = SamplerConfig(),
        model_prior: Optional[Dict[str, Dict[str, float]]] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.y = np.asarray(y, float)
        self.T = int(self.y.size)
        self.s = int(period)
        if self.s < 2:
            raise ValueError("period must be >= 2")
        self.priors, self.cfg = priors, cfg
        self.rng = rng or np.random.default_rng(cfg.random_seed)

        # Harmonic resolution
        K_full = (self.s - 1) // 2
        self.K = K_full if harmonics is None else int(harmonics)
        if not (0 <= self.K <= K_full):
            raise ValueError(f"harmonics K must be in [0, {K_full}] for s={self.s}")
        even = (self.s % 2) == 0
        if use_nyquist is None:
            self.use_nyq = bool(even and (self.K >= (self.s // 2 - 1)))
        else:
            self.use_nyq = bool(use_nyquist and even)

        # Modes
        ok = {"dynamic", "deterministic", "none"}
        if level_mode not in ok or trend_mode not in ok or seasonal_mode not in ok:
            raise ValueError("invalid mode")
        if trend_mode == "dynamic" and level_mode != "dynamic":
            raise ValueError("trend=dynamic requires level=dynamic")
        self.level_mode, self.trend_mode, self.seasonal_mode = level_mode, trend_mode, seasonal_mode

        # RJ model prior (Occam tilt)
        self.model_prior = model_prior or {
            "level":  {"dynamic": 0.5, "deterministic": 0.5, "none": 1e-12},
            "trend":  {"dynamic": 0.5, "deterministic": 0.5, "none": (0.4 if cfg.allow_none_trend else 1e-12)},
            "season": {"dynamic": 0.3, "deterministic": 0.7, "none": (0.2 if cfg.allow_none_season else 1e-12)},
        }

        # Precompute seasonal frequencies
        self._omegas = 2.0 * np.pi * (np.arange(1, self.K + 1, dtype=float)) / float(self.s)
        self._cosw = np.cos(self._omegas)
        self._sinw = np.sin(self._omegas)

        # Layout
        self._rebuild_layout()

        # Parameters / inits
        self.sigma2 = float(sigma2_init)
        self.s_alpha = float(s_alpha_init) if self.idx_alpha is not None else 0.0
        self.s_beta  = float(s_beta_init)  if self.idx_beta  is not None else 0.0
        self.s_gamma = float(s_gamma_init) if self.seasonal_mode == "dynamic" else 0.0
        # IG mixture auxiliaries for Half-Cauchy
        self.a_alpha = 1.0; self.a_beta = 1.0; self.a_gamma = 1.0

        self.m0_alpha = float(m0_alpha_init) if self.idx_alpha is not None or self.level_mode=="deterministic" else 0.0
        self.P0_alpha = float(P0_alpha_init) if self.idx_alpha is not None else 0.0
        self.m0_beta  = float(m0_beta_init)  if self.idx_beta  is not None or self.trend_mode=="deterministic" else 0.0
        self.P0_beta  = float(P0_beta_init)  if self.idx_beta  is not None else 0.0

        if self.K > 0:
            if m0_cos_init is None: m0_cos_init = np.zeros(self.K, float)
            if m0_sin_init is None: m0_sin_init = np.zeros(self.K, float)
            if len(m0_cos_init) != self.K or len(m0_sin_init) != self.K:
                raise ValueError("m0_cos_init and m0_sin_init must have length K")
            self.m0_cos = np.asarray(m0_cos_init, float)
            self.m0_sin = np.asarray(m0_sin_init, float)
        else:
            self.m0_cos = np.zeros(0, float)
            self.m0_sin = np.zeros(0, float)
        self.m0_nyq = (float(m0_nyq_init) if (self.use_nyq and m0_nyq_init is not None) else (0.0 if self.use_nyq else None))
        self.P0_harm = float(P0_harm_init)

        # Latent path
        self._alloc_state_holder()
        if self.dim > 0:
            m0_vec, P0_diag = self._current_m0_P0()
            self.x[0] = self.rng.multivariate_normal(m0_vec, np.diag(P0_diag) + EPS*np.eye(self.dim))
            self._propagate_initial_path(Q_init=np.full(self.dim, 1e-6))

        # Storage
        self.keep: Dict[str, np.ndarray] = {}
        self._mode_counts = {
            "level": {"dynamic": 0, "deterministic": 0, "none": 0},
            "trend": {"dynamic": 0, "deterministic": 0, "none": 0},
            "season":{"dynamic": 0, "deterministic": 0, "none": 0},
        }
        from collections import deque
        self.rj_stats = {
            "level":  {"proposed": 0, "accepted": 0, "win": deque(maxlen=self.cfg.rj_window)},
            "trend":  {"proposed": 0, "accepted": 0, "win": deque(maxlen=self.cfg.rj_window)},
            "season": {"proposed": 0, "accepted": 0, "win": deque(maxlen=self.cfg.rj_window)},
        }

        # Optional truth overlays
        self.true_sigma = None; self.true_Q = None; self.true_mu_t = None
        self.true_alpha_t = None; self.true_beta_t = None; self.true_gamma_t = None

        if self.cfg.progress:
            sd1 = _robust_sd(np.diff(self.y)) if self.T >= 2 else 0.0
            sd2 = _robust_sd(np.diff(self.y, n=2)) if self.T >= 3 else 0.0
            print(f"[init] modes L/T/S={self.level_mode[:3]}/{self.trend_mode[:3]}/{self.seasonal_mode[:3]} | sd1={sd1:.4g} sd2={sd2:.4g} | K={self.K} nyq={self.use_nyq}")

    def set_truth_paths(
    self,
    mu: Optional[Sequence[float]] = None,
    sigma: Optional[float] = None,
    Q: Optional[Sequence[float]] = None,
    alpha: Optional[Sequence[float]] = None,
    beta: Optional[Sequence[float]] = None,
    gamma: Optional[Sequence[float]] = None,
) -> None:
        if mu is not None:
            self.true_mu_t = np.asarray(mu, float)
        if sigma is not None:
            self.true_sigma = float(sigma)
        if Q is not None:
            self.true_Q = np.asarray(Q, float)
        self.true_alpha_t = None if alpha is None else np.asarray(alpha, float)
        self.true_beta_t  = None if beta  is None else np.asarray(beta,  float)
        self.true_gamma_t = None if gamma is None else np.asarray(gamma, float)

    # --------------------------- Layout & matrices ---------------------------

    def _rebuild_layout(self) -> None:
        layout: List[str] = []
        if self.level_mode == "dynamic": layout.append("alpha")
        if self.trend_mode == "dynamic": layout.append("beta")
        if self.seasonal_mode == "dynamic":
            for k in range(1, self.K + 1): layout += [f"c{k}", f"s{k}"]
            if self.use_nyq: layout.append("nyq")
        self._layout = layout
        self.dim = len(layout)
        self.idx_alpha = (layout.index("alpha") if "alpha" in layout else None)
        self.idx_beta  = (layout.index("beta")  if "beta"  in layout else None)

    def _idx_pair(self, k: int) -> int:
        pos = 0
        if self.idx_alpha is not None: pos += 1
        if self.idx_beta  is not None: pos += 1
        pos += 2*(k-1)
        return pos

    def _H(self) -> np.ndarray:
        if self.dim == 0: return np.zeros((1,0))
        h = np.zeros(self.dim, float)
        if self.idx_alpha is not None: h[self.idx_alpha] = 1.0
        if self.seasonal_mode == "dynamic" and self.K > 0:
            for k in range(1, self.K + 1):
                i = self._idx_pair(k)
                h[i] = 1.0  # load cos only
            if self.use_nyq: h[self._nyq_index()] = 1.0
        return h.reshape(1,-1)

    def _A(self) -> np.ndarray:
        if self.dim == 0: return np.zeros((0,0))
        A = np.eye(self.dim)
        if self.idx_alpha is not None and self.idx_beta is not None:
            A[self.idx_alpha, self.idx_beta] = 1.0
        if self.seasonal_mode == "dynamic" and self.K > 0:
            for k in range(1, self.K + 1):
                i = self._idx_pair(k)
                co, si = float(self._cosw[k - 1]), float(self._sinw[k - 1])
                A[i,   i  ] =  co; A[i,   i+1] =  si
                A[i+1, i  ] = -si; A[i+1, i+1] =  co
            if self.use_nyq:
                A[self._nyq_index(), self._nyq_index()] = -1.0
        return A

    def _nyq_index(self) -> int:
        assert self.use_nyq
        pos = 0
        if self.idx_alpha is not None: pos += 1
        if self.idx_beta  is not None: pos += 1
        pos += 2*self.K
        return pos

    def _u(self) -> np.ndarray:
        if self.dim == 0: return np.zeros(0, float)
        u = np.zeros(self.dim, float)
        if (self.idx_alpha is not None) and (self.trend_mode == "deterministic"):
            u[self.idx_alpha] = float(self.m0_beta)
        return u

    def _Q(self) -> np.ndarray:
        if self.dim == 0: return np.zeros((0,0))
        Q = np.zeros((self.dim, self.dim))
        if self.idx_alpha is not None and self.s_alpha > 0: Q[self.idx_alpha, self.idx_alpha] = self.s_alpha**2
        if self.idx_beta  is not None and self.s_beta  > 0: Q[self.idx_beta,  self.idx_beta ] = self.s_beta**2
        if self.seasonal_mode == "dynamic" and self.s_gamma > 0:
            for k in range(1, self.K + 1):
                i = self._idx_pair(k)
                Q[i, i] = self.s_gamma**2; Q[i+1, i+1] = self.s_gamma**2
            if self.use_nyq:
                Q[self._nyq_index(), self._nyq_index()] = self.s_gamma**2
        return Q

    # ---------------- deterministic pieces ----------------

    def _season_det(self, t: int) -> float:
        if self.seasonal_mode != "deterministic": return 0.0
        val = 0.0
        for k in range(1, self.K + 1):
            w = self._omegas[k - 1]
            val += self.m0_cos[k - 1]*math.cos(w*t) + self.m0_sin[k - 1]*math.sin(w*t)
        if self.use_nyq and (self.m0_nyq is not None):
            val += float(self.m0_nyq) * ((-1.0)**t)
        return float(val)

    def _mu_det(self, t: int) -> float:
        out = 0.0
        if self.level_mode == "deterministic": out += self.m0_alpha
        if (self.trend_mode == "deterministic") and (self.idx_alpha is None): out += self.m0_beta * t
        out += self._season_det(t)
        return out

    def _current_m0_P0(self) -> Tuple[np.ndarray, np.ndarray]:
        m0, P0 = [], []
        if self.idx_alpha is not None: m0.append(self.m0_alpha); P0.append(self.P0_alpha)
        if self.idx_beta  is not None: m0.append(self.m0_beta ); P0.append(self.P0_beta )
        if self.seasonal_mode == "dynamic":
            for k in range(self.K):
                m0 += [self.m0_cos[k], self.m0_sin[k]]
                P0 += [self.P0_harm,   self.P0_harm]
            if self.use_nyq:
                m0.append(float(0.0 if self.m0_nyq is None else self.m0_nyq)); P0.append(self.P0_harm)
        return np.asarray(m0, float), np.asarray(P0, float)

    # ----------------------- FFBS (Kalman + CK) -----------------------

    def _ffbs(self) -> np.ndarray:
        if self.dim == 0: return np.zeros_like(self.x)
        H, A, Q, R = self._H(), self._A(), self._Q(), float(self.sigma2)
        m0_vec, P0_diag = self._current_m0_P0()
        m = np.zeros((self.T + 1, self.dim))
        C = np.zeros((self.T + 1, self.dim, self.dim))
        a = np.zeros_like(m); Rm = np.zeros_like(C)
        m[0] = m0_vec; C[0] = np.diag(P0_diag) + EPS*np.eye(self.dim)
        u = self._u()

        for t in range(1, self.T + 1):
            a[t]  = A @ m[t-1] + u
            Rm[t] = A @ C[t-1] @ A.T + Q
            Rm[t] = 0.5*(Rm[t] + Rm[t].T) + EPS*np.eye(self.dim)
            y_det = self._mu_det(t-1)
            S = float(H @ Rm[t] @ H.T + R)
            v = float(self.y[t-1] - y_det - H @ a[t])
            K = (Rm[t] @ H.T) / S
            m[t] = a[t] + K.flatten()*v
            C[t] = Rm[t] - K @ (H @ Rm[t])
            C[t] = 0.5*(C[t] + C[t].T) + EPS*np.eye(self.dim)

        x = np.zeros_like(self.x)
        x[self.T] = self.rng.multivariate_normal(m[self.T], C[self.T])
        for t in range(self.T - 1, -1, -1):
            J = C[t] @ A.T
            J = _spd_solve(Rm[t+1], J.T).T
            mean = m[t] + J @ (x[t+1] - (A @ m[t] + u))
            cov  = C[t] - J @ Rm[t+1] @ J.T
            cov  = 0.5*(cov + cov.T)
            mineig = float(np.linalg.eigvalsh(cov).min())
            if mineig < 1e-12:
                cov += (1e-12 - mineig)*np.eye(cov.shape[0])
            x[t] = self.rng.multivariate_normal(mean, cov)
        return x

    def _propagate_initial_path(self, Q_init: np.ndarray) -> None:
        if self.dim == 0: return
        A, u = self._A(), self._u()
        for t in range(1, self.T + 1):
            self.x[t] = A @ self.x[t-1] + u + self.rng.normal(0.0, np.sqrt(Q_init), size=self.dim)

    # ------------------ Helpers: μ and σ² update ------------------

    def _mu_vec(self) -> np.ndarray:
        H = self._H()
        mu = np.zeros(self.T, float)
        for t in range(1, self.T + 1):
            dyn = float(H @ self.x[t]) if self.dim > 0 else 0.0
            mu[t - 1] = self._mu_det(t - 1) + dyn
        return mu

    def update_sigma2(self) -> None:
        e = self.y - self._mu_vec()
        a = self.priors.a_sigma + 0.5*self.T
        b = self.priors.b_sigma + 0.5*float(e @ e)
        tau = self.rng.gamma(shape=a, scale=1.0/b)
        self.sigma2 = 1.0 / max(tau, 1e-300)

    # ------------- Innovation SS (for Q updates) -------------

    def _innovation_ss_alpha(self) -> Tuple[float, int]:
        if self.idx_alpha is None: return 0.0, 0
        ss = 0.0
        for t in range(1, self.T + 1):
            drift = 0.0
            if self.idx_beta is not None: drift = self.x[t-1, self.idx_beta]
            elif self.trend_mode == "deterministic": drift = float(self.m0_beta)
            mean = self.x[t-1, self.idx_alpha] + drift
            ss += (self.x[t, self.idx_alpha] - mean)**2
        return float(ss), self.T

    def _innovation_ss_beta(self) -> Tuple[float, int]:
        if self.idx_beta is None: return 0.0, 0
        d = self.x[1:, self.idx_beta] - self.x[:-1, self.idx_beta]
        return float(np.sum(d*d)), self.T

    def _innovation_ss_gamma(self) -> Tuple[float, int]:
        if self.seasonal_mode != "dynamic": return 0.0, 0
        ss = 0.0; per_step = 0
        for k in range(1, self.K + 1):
            i = self._idx_pair(k)
            co, si = float(self._cosw[k - 1]), float(self._sinw[k - 1])
            R = np.array([[co, si], [-si, co]], float)
            for t in range(1, self.T + 1):
                prev = self.x[t-1, i:i+2]
                mean = R @ prev
                err  = self.x[t, i:i+2] - mean
                ss += float(err @ err)
        per_step += 2*self.K
        if self.use_nyq:
            j = self._nyq_index()
            for t in range(1, self.T + 1):
                mean = -self.x[t-1, j]
                err  = self.x[t, j] - mean
                ss += float(err*err)
            per_step += 1
        T_eff = self.T * per_step
        return float(ss), int(T_eff)

    # --------- Half-Cauchy IG-mixture updates for Q ---------

    @staticmethod
    def _rinvgamma(rng: np.random.Generator, shape: float, scale: float) -> float:
        return 1.0 / rng.gamma(shape, 1.0/scale)

    def update_process_Q_halfcauchy(self) -> None:
        if self.idx_alpha is not None:
            SS, T_eff = self._innovation_ss_alpha()
            A = float(self.priors.hc_scale_alpha)
            Q_alpha = self._rinvgamma(self.rng, 0.5*T_eff + 0.5, 0.5*SS + 1.0/max(self.a_alpha,1e-300))
            self.s_alpha = math.sqrt(max(Q_alpha, 0.0))
            self.a_alpha = self._rinvgamma(self.rng, 1.0, (1.0/(A*A)) + 1.0/max(Q_alpha,1e-300))
        if self.idx_beta is not None:
            SS, T_eff = self._innovation_ss_beta()
            A = float(self.priors.hc_scale_beta)
            Q_beta  = self._rinvgamma(self.rng, 0.5*T_eff + 0.5, 0.5*SS + 1.0/max(self.a_beta,1e-300))
            self.s_beta = math.sqrt(max(Q_beta, 0.0))
            self.a_beta = self._rinvgamma(self.rng, 1.0, (1.0/(A*A)) + 1.0/max(Q_beta,1e-300))
        if self.seasonal_mode == "dynamic":
            SS, T_eff = self._innovation_ss_gamma()
            A = float(self.priors.hc_scale_gamma)
            Q_gamma = self._rinvgamma(self.rng, 0.5*T_eff + 0.5, 0.5*SS + 1.0/max(self.a_gamma,1e-300))
            self.s_gamma = math.sqrt(max(Q_gamma, 0.0))
            self.a_gamma = self._rinvgamma(self.rng, 1.0, (1.0/(A*A)) + 1.0/max(Q_gamma,1e-300))

    # --------- m0 / P0 dynamic updates; deterministic params ---------

    @staticmethod
    def _gibbs_m0_scalar(rng: np.random.Generator, x0: float, m_prior: float, s_prior: float, P0: float) -> float:
        prec = 1.0/(s_prior**2) + 1.0/max(P0, 1e-18)
        var  = 1.0/prec
        mean = var * (m_prior/(s_prior**2) + x0/max(P0, 1e-18))
        return float(rng.normal(mean, math.sqrt(var)))

    def update_m0(self) -> None:
        if self.dim == 0: return
        pos = 0
        if self.idx_alpha is not None:
            self.m0_alpha = self._gibbs_m0_scalar(self.rng, float(self.x[0, pos]), self.priors.m_m0_alpha, self.priors.s_m0_alpha, self.P0_alpha); pos += 1
        if self.idx_beta is not None:
            self.m0_beta  = self._gibbs_m0_scalar(self.rng, float(self.x[0, pos]), self.priors.m_m0_beta,  self.priors.s_m0_beta,  self.P0_beta ); pos += 1
        if self.seasonal_mode == "dynamic":
            s0 = float(self.priors.s_m0_harm)
            m_cos = (np.zeros(self.K) if self.priors.m_m0_cos is None else np.asarray(self.priors.m_m0_cos, float))
            m_sin = (np.zeros(self.K) if self.priors.m_m0_sin is None else np.asarray(self.priors.m_m0_sin, float))
            if m_cos.size != self.K or m_sin.size != self.K:
                raise ValueError("priors.m_m0_cos/m_m0_sin must have length K")
            for k in range(self.K):
                self.m0_cos[k] = self._gibbs_m0_scalar(self.rng, float(self.x[0, pos + 2*k    ]), float(m_cos[k]), s0, self.P0_harm)
                self.m0_sin[k] = self._gibbs_m0_scalar(self.rng, float(self.x[0, pos + 2*k + 1]), float(m_sin[k]), s0, self.P0_harm)
            if self.use_nyq:
                j = pos + 2*self.K
                self.m0_nyq = self._gibbs_m0_scalar(self.rng, float(self.x[0, j]), float(self.priors.m_m0_nyq), s0, self.P0_harm)

    def update_P0(self) -> None:
        if self.dim == 0: return
        pos = 0
        if self.idx_alpha is not None:
            a = self.priors.a_P0_alpha + 0.5
            b = self.priors.b_P0_alpha + 0.5*(float(self.x[0, pos]) - self.m0_alpha)**2
            self.P0_alpha = 1.0/self.rng.gamma(a, 1.0/b); pos += 1
        if self.idx_beta is not None:
            a = self.priors.a_P0_beta + 0.5
            b = self.priors.b_P0_beta + 0.5*(float(self.x[0, pos]) - self.m0_beta)**2
            self.P0_beta = 1.0/self.rng.gamma(a, 1.0/b); pos += 1
        if self.seasonal_mode == "dynamic":
            diffsq = 0.0
            Ktot = 2*self.K + (1 if self.use_nyq else 0)
            target = [*self.m0_cos, *self.m0_sin] + ([float(self.m0_nyq)] if self.use_nyq else [])
            for k in range(Ktot):
                diffsq += (float(self.x[0, pos + k]) - float(target[k]))**2
            a = self.priors.a_P0_harm + 0.5*Ktot
            b = self.priors.b_P0_harm + 0.5*diffsq
            self.P0_harm = 1.0/self.rng.gamma(a, 1.0/b)

    def update_deterministic_params(self) -> None:
        # Level (det)
        if self.level_mode == "deterministic":
            r = self.y.copy()
            if self.dim > 0:
                H = self._H()
                for t in range(1, self.T + 1): r[t-1] -= float(H @ self.x[t])
            r -= np.array([self._season_det(t) for t in range(self.T)], float)
            if (self.idx_alpha is None) and (self.trend_mode == "deterministic"):
                r -= self.m0_beta * np.arange(self.T, dtype=float)
            s2 = float(self.sigma2); m0, s0 = float(self.priors.m_m0_alpha), float(self.priors.s_m0_alpha)
            prec = self.T/s2 + 1.0/(s0*s0); mean = ((r.sum()/s2) + m0/(s0*s0)) / prec
            self.m0_alpha = float(self.rng.normal(mean, math.sqrt(1.0/prec)))
        # Trend (det)
        if self.trend_mode == "deterministic":
            if self.idx_alpha is not None:
                d = self.x[1:, self.idx_alpha] - self.x[:-1, self.idx_alpha]
                q = float(self.s_alpha**2) if self.s_alpha > 0 else 1e-12
                m0, s0 = float(self.priors.m_m0_beta), float(self.priors.s_m0_beta)
                prec = self.T/q + 1.0/(s0*s0); mean = ((float(np.sum(d))/q) + m0/(s0*s0)) / prec
                self.m0_beta = float(self.rng.normal(mean, math.sqrt(1.0/prec)))
            else:
                t = np.arange(self.T, dtype=float); r = self.y.copy()
                if self.dim > 0:
                    H = self._H()
                    for k in range(1, self.T + 1): r[k-1] -= float(H @ self.x[k])
                if self.level_mode == "deterministic": r -= self.m0_alpha
                r -= np.array([self._season_det(tt) for tt in range(self.T)], float)
                m0, s0 = float(self.priors.m_m0_beta), float(self.priors.s_m0_beta)
                sig2 = float(self.sigma2); prec = (t@t)/sig2 + 1.0/(s0*s0)
                mean = ((t@r)/sig2 + m0/(s0*s0))/prec
                self.m0_beta = float(self.rng.normal(mean, math.sqrt(1.0/prec)))
        # Season (det): regress on cos/sin(+nyq)
        if self.seasonal_mode == "deterministic":
            t = np.arange(self.T, dtype=float); Zcols=[]
            for k in range(1, self.K + 1):
                w = self._omegas[k-1]
                Zcols += [np.cos(w*t), np.sin(w*t)]
            if self.use_nyq: Zcols.append(((-1.0)**t))
            Z = np.column_stack(Zcols) if Zcols else np.zeros((self.T, 0))
            r = self.y.copy()
            if self.dim > 0:
                H = self._H()
                for k in range(1, self.T + 1): r[k-1] -= float(H @ self.x[k])
            if self.level_mode == "deterministic": r -= self.m0_alpha
            if (self.idx_alpha is None) and (self.trend_mode == "deterministic"):
                r -= self.m0_beta * t
            p = Z.shape[1]; s2p = float(self.priors.s_m0_harm)**2
            m_prior = np.zeros(p)
            if (self.priors.m_m0_cos is not None) and (self.priors.m_m0_sin is not None):
                if (len(self.priors.m_m0_cos) == self.K) and (len(self.priors.m_m0_sin) == self.K):
                    m_prior[:2*self.K:2] = np.asarray(self.priors.m_m0_cos, float)
                    m_prior[1:2*self.K:2] = np.asarray(self.priors.m_m0_sin, float)
            if self.use_nyq and (p > 2*self.K): m_prior[-1] = float(self.priors.m_m0_nyq)
            sig2 = float(self.sigma2)
            Prec = (Z.T@Z)/sig2 + np.eye(p)/s2p
            b    = (Z.T@r)/sig2 + m_prior/s2p
            L = np.linalg.cholesky(Prec)
            mu = np.linalg.solve(L.T, np.linalg.solve(L, b))
            theta = mu + np.linalg.solve(L.T, self.rng.standard_normal(p))
            if self.K > 0:
                self.m0_cos = theta[:2*self.K:2].copy(); self.m0_sin = theta[1:2*self.K:2].copy()
            if self.use_nyq: self.m0_nyq = float(theta[-1])

    # ------------------------- Log posterior pieces -------------------------

    def _kalman_loglik(self) -> float:
        H, A, Q, R = self._H(), self._A(), self._Q(), float(self.sigma2)
        if self.dim == 0:
            e = np.array([self.y[t] - self._mu_det(t) for t in range(self.T)], float)
            return -0.5 * np.sum(np.log(2*np.pi*R) + (e*e)/R)
        m0_vec, P0_diag = self._current_m0_P0()
        m = m0_vec.copy(); C = np.diag(P0_diag) + EPS*np.eye(self.dim)
        ll = 0.0; u = self._u()
        for t in range(self.T):
            a = A @ m + u
            Rm = A @ C @ A.T + Q
            Rm = 0.5*(Rm + Rm.T) + EPS*np.eye(self.dim)
            y_det = self._mu_det(t)
            S = float(H @ Rm @ H.T + R)
            v = float(self.y[t] - y_det - H @ a)
            ll += -0.5*(math.log(2*math.pi) + math.log(S) + (v*v)/S)
            K = (Rm @ H.T) / S
            m = a + K.flatten()*v
            C = Rm - K @ (H @ Rm)
            C = 0.5*(C + C.T) + EPS*np.eye(self.dim)
        return float(ll)

    def _log_prior_current(self) -> float:
        lp = 0.0
        def lIG(x, shape, scale):
            x = max(float(x), 1e-300)
            return -(shape + 1.0)*math.log(x) - (scale/x)
        # σ² via τ~Gamma(a,b) on precision
        a, b = self.priors.a_sigma, self.priors.b_sigma
        tau = 1.0/max(self.sigma2, 1e-300)
        lp += (a - 1.0)*math.log(tau) - b*tau

        # dynamic level
        if self.idx_alpha is not None:
            Qa = max(self.s_alpha**2, 1e-300); aa = max(self.a_alpha, 1e-300)
            lp += lIG(Qa, 0.5, 1.0/aa) + lIG(aa, 1.0, 1.0/(self.priors.hc_scale_alpha**2))
            lp += -0.5*((self.m0_alpha - self.priors.m_m0_alpha)**2)/(self.priors.s_m0_alpha**2)
            lp += lIG(self.P0_alpha, self.priors.a_P0_alpha, self.priors.b_P0_alpha)

        # dynamic trend
        if self.idx_beta is not None:
            Qb = max(self.s_beta**2, 1e-300); ab = max(self.a_beta, 1e-300)
            lp += lIG(Qb, 0.5, 1.0/ab) + lIG(ab, 1.0, 1.0/(self.priors.hc_scale_beta**2))
            lp += -0.5*((self.m0_beta - self.priors.m_m0_beta)**2)/(self.priors.s_m0_beta**2)
            lp += lIG(self.P0_beta, self.priors.a_P0_beta, self.priors.b_P0_beta)

        # dynamic season
        if self.seasonal_mode == "dynamic":
            Qg = max(self.s_gamma**2, 1e-300); ag = max(self.a_gamma, 1e-300)
            lp += lIG(Qg, 0.5, 1.0/ag) + lIG(ag, 1.0, 1.0/(self.priors.hc_scale_gamma**2))
            s2 = self.priors.s_m0_harm**2
            base_cos = (np.zeros(self.K) if self.priors.m_m0_cos is None else np.asarray(self.priors.m_m0_cos, float))
            base_sin = (np.zeros(self.K) if self.priors.m_m0_sin is None else np.asarray(self.priors.m_m0_sin, float))
            lp += -0.5*float(np.sum((self.m0_cos - base_cos)**2))/s2
            lp += -0.5*float(np.sum((self.m0_sin - base_sin)**2))/s2
            if self.use_nyq:
                lp += -0.5*((float(self.m0_nyq) - float(self.priors.m_m0_nyq))**2)/s2
            lp += lIG(self.P0_harm, self.priors.a_P0_harm, self.priors.b_P0_harm)

        # deterministic params priors
        if self.level_mode == "deterministic":
            lp += -0.5*((self.m0_alpha - self.priors.m_m0_alpha)**2)/(self.priors.s_m0_alpha**2)
        if self.trend_mode == "deterministic":
            lp += -0.5*((self.m0_beta  - self.priors.m_m0_beta )**2)/(self.priors.s_m0_beta **2)
        if self.seasonal_mode == "deterministic":
            s2 = self.priors.s_m0_harm**2
            base_cos = (np.zeros(self.K) if self.priors.m_m0_cos is None else np.asarray(self.priors.m_m0_cos, float))
            base_sin = (np.zeros(self.K) if self.priors.m_m0_sin is None else np.asarray(self.priors.m_m0_sin, float))
            lp += -0.5*float(np.sum((self.m0_cos - base_cos)**2))/s2
            lp += -0.5*float(np.sum((self.m0_sin - base_sin)**2))/s2
            if self.use_nyq: lp += -0.5*((float(self.m0_nyq) - float(self.priors.m_m0_nyq))**2)/s2

        # model priors
        lp += math.log(self.model_prior["level"][self.level_mode] + 1e-300)
        lp += math.log(self.model_prior["trend"][self.trend_mode] + 1e-300)
        lp += math.log(self.model_prior["season"][self.seasonal_mode] + 1e-300)
        return float(lp)

    # ------------------------------- RJ moves -------------------------------

    def _alloc_state_holder(self) -> None:
        self.x = np.zeros((self.T + 1, self.dim), float)

    def _snapshot(self) -> dict:
        return {
            "level": self.level_mode, "trend": self.trend_mode, "season": self.seasonal_mode,
            "sigma2": self.sigma2,
            "s_alpha": self.s_alpha, "s_beta": self.s_beta, "s_gamma": self.s_gamma,
            "a_alpha": self.a_alpha, "a_beta": self.a_beta, "a_gamma": self.a_gamma,
            "m0_alpha": self.m0_alpha, "P0_alpha": self.P0_alpha,
            "m0_beta":  self.m0_beta,  "P0_beta":  self.P0_beta,
            "m0_cos": None if self.m0_cos is None else self.m0_cos.copy(),
            "m0_sin": None if self.m0_sin is None else self.m0_sin.copy(),
            "m0_nyq": self.m0_nyq,
            "P0_harm": self.P0_harm,
        }

    def _load_snapshot(self, S: dict) -> None:
        self.level_mode = S["level"]; self.trend_mode = S["trend"]; self.seasonal_mode = S["season"]
        self.sigma2 = float(S["sigma2"])
        self.s_alpha = float(S["s_alpha"]); self.s_beta = float(S["s_beta"]); self.s_gamma = float(S["s_gamma"])
        self.a_alpha = float(S["a_alpha"]); self.a_beta = float(S["a_beta"]); self.a_gamma = float(S["a_gamma"])
        self.m0_alpha = float(S["m0_alpha"]); self.P0_alpha = float(S["P0_alpha"])
        self.m0_beta  = float(S["m0_beta"]);  self.P0_beta  = float(S["P0_beta"])
        self.m0_cos = None if S["m0_cos"] is None else np.asarray(S["m0_cos"], float).copy()
        self.m0_sin = None if S["m0_sin"] is None else np.asarray(S["m0_sin"], float).copy()
        self.m0_nyq = S["m0_nyq"]; self.P0_harm = float(S["P0_harm"])
        self._rebuild_layout()

    def _legal_modes(self, level: str, trend: str, season: str) -> bool:
        if trend == "dynamic" and level != "dynamic": return False
        if (not self.cfg.allow_none_level) and (level == "none"): return False
        return True

    def _rj_record(self, block: str, accepted: bool) -> None:
        s = self.rj_stats[block]; s["proposed"] += 1
        if accepted: s["accepted"] += 1
        s["win"].append(1 if accepted else 0)

    def _fmt_rj_block(self, block: str) -> str:
        s = self.rj_stats[block]
        prop = max(1, int(s["proposed"])); acc = int(s["accepted"])
        return f"{100.0*acc/prop:.1f}%"

    def _fmt_rj_all(self) -> str:
        return f"{self._fmt_rj_block('level')}|{self._fmt_rj_block('trend')}|{self._fmt_rj_block('season')}"

    def _mh_accept(self, dlogpost: float) -> bool:
        if not np.isfinite(dlogpost): return False
        if dlogpost >= 0: return True
        return (math.log(self.rng.uniform()) < dlogpost)

    def _safe_logpost(self) -> float:
        try:
            s = self._kalman_loglik() + self._log_prior_current()
            return float(s) if np.isfinite(s) else float("-inf")
        except Exception:
            return float("-inf")

    def _draw_prior_dyn_block(self, which: str) -> None:
        # Draw reasonable priors for a newly-dynamic block (Half-Cauchy via IG mixture)
        if which == "level":
            A = float(self.priors.hc_scale_alpha)
            a_aux = self._rinvgamma(self.rng, 1.0, 1.0/(A*A))
            Q = self._rinvgamma(self.rng, 0.5, 1.0/max(a_aux,1e-300))
            self.a_alpha = a_aux; self.s_alpha = math.sqrt(max(Q, 1e-18))
            self.m0_alpha = float(self.rng.normal(self.priors.m_m0_alpha, self.priors.s_m0_alpha))
            self.P0_alpha = 1.0/self.rng.gamma(self.priors.a_P0_alpha, 1.0/self.priors.b_P0_alpha)
        elif which == "trend":
            A = float(self.priors.hc_scale_beta)
            a_aux = self._rinvgamma(self.rng, 1.0, 1.0/(A*A))
            Q = self._rinvgamma(self.rng, 0.5, 1.0/max(a_aux,1e-300))
            self.a_beta = a_aux; self.s_beta = math.sqrt(max(Q, 1e-18))
            self.m0_beta = float(self.rng.normal(self.priors.m_m0_beta, self.priors.s_m0_beta))
            self.P0_beta = 1.0/self.rng.gamma(self.priors.a_P0_beta, 1.0/self.priors.b_P0_beta)
        elif which == "season":
            A = float(self.priors.hc_scale_gamma)
            a_aux = self._rinvgamma(self.rng, 1.0, 1.0/(A*A))
            Q = self._rinvgamma(self.rng, 0.5, 1.0/max(a_aux,1e-300))
            self.a_gamma = a_aux; self.s_gamma = math.sqrt(max(Q, 1e-18))
            base_cos = (np.zeros(self.K) if self.priors.m_m0_cos is None else np.asarray(self.priors.m_m0_cos, float))
            base_sin = (np.zeros(self.K) if self.priors.m_m0_sin is None else np.asarray(self.priors.m_m0_sin, float))
            s = float(self.priors.s_m0_harm)
            self.m0_cos = self.rng.normal(base_cos, s, size=self.K)
            self.m0_sin = self.rng.normal(base_sin, s, size=self.K)
            if self.use_nyq: self.m0_nyq = float(self.rng.normal(self.priors.m_m0_nyq, s))
            self.P0_harm = 1.0/self.rng.gamma(self.priors.a_P0_harm, 1.0/self.priors.b_P0_harm)

    def _propose_mode(self, comp: str, cur: str) -> str:
        if comp == "level":
            cand = ["dynamic", "deterministic"] + (["none"] if self.cfg.allow_none_level else [])
        elif comp == "trend":
            cand = ["dynamic", "deterministic"] + (["none"] if self.cfg.allow_none_trend else [])
        else:
            cand = ["dynamic", "deterministic"] + (["none"] if self.cfg.allow_none_season else [])
        choices = [z for z in cand if z != cur]
        return self.rng.choice(choices)

    def _ensure_det_season_defaults(self) -> None:
        if self.seasonal_mode != "deterministic": return
        if self.m0_cos is None or self.m0_cos.size != self.K: self.m0_cos = np.zeros(self.K)
        if self.m0_sin is None or self.m0_sin.size != self.K: self.m0_sin = np.zeros(self.K)
        if self.use_nyq and (self.m0_nyq is None): self.m0_nyq = 0.0

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
            # escape: if proposing trend=dynamic while level!=dynamic, jointly flip level→dynamic
            if new_trend == "dynamic" and new_level != "dynamic":
                new_level = "dynamic"
        else:
            new_season = prop

        if not self._legal_modes(new_level, new_trend, new_season):
            self._rj_record(comp, False); return

        cur_snap = self._snapshot()

        # births
        if (level != "dynamic") and (new_level == "dynamic"):  self._draw_prior_dyn_block("level")
        if (trend != "dynamic") and (new_trend == "dynamic"):  self._draw_prior_dyn_block("trend")
        if (season != "dynamic") and (new_season == "dynamic"): self._draw_prior_dyn_block("season")
        # becoming deterministic season
        if (season != "deterministic") and (new_season == "deterministic"):
            self._ensure_det_season_defaults()
            self.P0_harm = 0.0

        # temporarily apply proposed modes for logpost
        self.level_mode, self.trend_mode, self.seasonal_mode = new_level, new_trend, new_season
        self._rebuild_layout()
        if self.level_mode != "dynamic":   self.s_alpha = 0.0
        if self.trend_mode != "dynamic":   self.s_beta  = 0.0
        if self.seasonal_mode != "dynamic":self.s_gamma = 0.0
        if self.seasonal_mode == "deterministic": self._ensure_det_season_defaults()

        logpost_prop = self._safe_logpost()
        # restore current
        self._load_snapshot(cur_snap)
        logpost_cur = self._safe_logpost()
        if not (np.isfinite(logpost_prop) and np.isfinite(logpost_cur)):
            self._rj_record(comp, False); return

        dlogpost = logpost_prop - logpost_cur
        acc = self._mh_accept(dlogpost)
        self._rj_record(comp, bool(acc))
        disp_dlog = float(np.clip(dlogpost, -1e3, 1e3)) if np.isfinite(dlogpost) else float('nan')

        if acc:
            before = (self.level_mode, self.trend_mode, self.seasonal_mode)
            # load then set proposed
            self._load_snapshot(cur_snap)
            self.level_mode, self.trend_mode, self.seasonal_mode = new_level, new_trend, new_season
            if self.level_mode != "dynamic":   self.s_alpha = 0.0
            if self.trend_mode != "dynamic":   self.s_beta  = 0.0
            if self.seasonal_mode != "dynamic":self.s_gamma = 0.0
            if self.seasonal_mode == "deterministic": self._ensure_det_season_defaults()
            self._rebuild_layout()
            self._alloc_state_holder()
            after = (self.level_mode, self.trend_mode, self.seasonal_mode)
            print(f"[switch] block={comp} | L:{before[0][:3]}→{after[0][:3]} T:{before[1][:3]}→{after[1][:3]} S:{before[2][:3]}→{after[2][:3]} | Δlogpost={disp_dlog:+.4f} | RJ {self._fmt_rj_all()}")

    # ------------------------------ bookkeeping ----------------------------

    def _tally_modes(self) -> None:
        self._mode_counts["level"][self.level_mode]   += 1
        self._mode_counts["trend"][self.trend_mode]   += 1
        self._mode_counts["season"][self.seasonal_mode]+= 1

    def inclusion_probabilities(self) -> Dict[str, Dict[str, float]]:
        total = sum(self._mode_counts["level"].values())
        if total == 0:
            return {k: {m: 0.0 for m in v} for k, v in self._mode_counts.items()}
        out = {}
        for comp, cnt in self._mode_counts.items():
            out[comp] = {m: c/total for m, c in cnt.items()}
        return out

    # ------------------------------- progress -------------------------------

    def _progress_line(self, it: int) -> str:
        parts = [f"[it {it+1}/{self.cfg.n_iter}]",
                 f"σ={math.sqrt(self.sigma2):.3f}",
                 f"L/T/S={self.level_mode[:3]}/{self.trend_mode[:3]}/{self.seasonal_mode[:3]}"]
        if self.idx_alpha is not None: parts.append(f"Qα={self.s_alpha**2:.4g}")
        if self.idx_beta  is not None: parts.append(f"Qβ={self.s_beta**2:.4g}")
        if self.seasonal_mode == "dynamic": parts.append(f"Qγ={self.s_gamma**2:.4g}")
        if self.level_mode == "dynamic": parts.append(f"m0α={self.m0_alpha:.4g} P0α={self.P0_alpha:.4g}")
        elif self.level_mode == "deterministic": parts.append(f"m0α(det)={self.m0_alpha:.4g}")
        if self.trend_mode == "dynamic": parts.append(f"m0β={self.m0_beta:.4g} P0β={self.P0_beta:.4g}")
        elif self.trend_mode == "deterministic": parts.append(f"m0β(det)={self.m0_beta:.4g}")
        if self.seasonal_mode == "dynamic":
            parts.append(f"m0cos={_fmt_list(self.m0_cos)} m0sin={_fmt_list(self.m0_sin)} P0harm={self.P0_harm:.4g}"
                         + (f" nyq={0.0 if (self.m0_nyq is None) else float(self.m0_nyq):.4g}" if self.use_nyq else ""))
        elif self.seasonal_mode == "deterministic":
            parts.append(f"m0cos(det)={_fmt_list(self.m0_cos)} m0sin(det)={_fmt_list(self.m0_sin)}"
                         + (f" nyq(det)={0.0 if (self.m0_nyq is None) else float(self.m0_nyq):.4g}" if self.use_nyq else ""))
        parts.append(f"RJ {self._fmt_rj_all()}")
        return " | ".join(parts)

    # --------------------------------- MCMC ---------------------------------

    def run(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        save_iters = list(range(cfg.burn, cfg.n_iter, cfg.thin))
        n_kept, keep_idx = len(save_iters), 0

        # pre-allocate with maximal possible dim (safe for RJ growth)
        max_dim = (1 if True else 0) + (1 if True else 0) + (2*self.K + (1 if self.use_nyq else 0))

        self.keep = {
            "sigma": np.full(n_kept, np.nan, float),
            "mu":    np.full((n_kept, self.T), np.nan, float),
            "modes": np.zeros((n_kept, 3), int),  # 0:dyn,1:det,2:none
            "x":     np.zeros((n_kept, self.T, max_dim), float),
            "Q_alpha": np.full(n_kept, np.nan, float),
            "Q_beta":  np.full(n_kept, np.nan, float),
            "Q_gamma": np.full(n_kept, np.nan, float),
            "m0_alpha": np.full(n_kept, np.nan, float),
            "m0_beta":  np.full(n_kept, np.nan, float),
            "m0_cos":   np.full((n_kept, self.K), np.nan, float),
            "m0_sin":   np.full((n_kept, self.K), np.nan, float),
            "m0_nyq":   np.full(n_kept, np.nan, float) if self.use_nyq else np.zeros(0),
            "P0_alpha": np.full(n_kept, np.nan, float),
            "P0_beta":  np.full(n_kept, np.nan, float),
            "P0_harm":  np.full(n_kept, np.nan, float),
        }

        print_every = cfg.progress_every if cfg.progress_every > 0 else max(1, cfg.n_iter // 50) or 1

        for it in range(cfg.n_iter):
            if self.dim > 0:
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
            if (self.cfg.print_dummies_every > 0) and ((it + 1) % self.cfg.print_dummies_every == 0):
                if self.K > 0:
                    d = harmonics_to_dummies_full_fft(
                        s=self.s,
                        cos_coefs=(self.m0_cos if self.seasonal_mode == "deterministic" else self.x[it % (self.T+1), self._idx_pair(1)::2][:self.K]),
                        sin_coefs=(self.m0_sin if self.seasonal_mode == "deterministic" else self.x[it % (self.T+1), self._idx_pair(1)+1::2][:self.K]),
                        use_nyquist=self.use_nyq,
                        nyq_coef=(self.m0_nyq if self.seasonal_mode == "deterministic" else (0.0 if not self.use_nyq else self.x[it % (self.T+1), self._nyq_index()])),
                    )
                    print(f"[season][iter {it+1}] dummies={_fmt_list(d, max_elems=8, mode='both')}")


            if it in save_iters:
                k = keep_idx
                mu = self._mu_vec()
                self.keep["mu"][k, :] = mu
                self.keep["sigma"][k] = math.sqrt(self.sigma2)
                enc = lambda s: 0 if s == "dynamic" else (1 if s == "deterministic" else 2)
                self.keep["modes"][k, :] = np.array([enc(self.level_mode), enc(self.trend_mode), enc(self.seasonal_mode)], int)

                if self.level_mode == "dynamic":
                    self.keep["Q_alpha"][k] = self.s_alpha**2
                    self.keep["m0_alpha"][k] = self.m0_alpha
                    self.keep["P0_alpha"][k] = self.P0_alpha
                if self.trend_mode == "dynamic":
                    self.keep["Q_beta"][k]  = self.s_beta**2
                    self.keep["m0_beta"][k] = self.m0_beta
                    self.keep["P0_beta"][k] = self.P0_beta
                if self.seasonal_mode == "dynamic":
                    self.keep["Q_gamma"][k] = self.s_gamma**2
                    if self.K > 0:
                        self.keep["m0_cos"][k, :] = self.m0_cos
                        self.keep["m0_sin"][k, :] = self.m0_sin
                    if self.use_nyq: self.keep["m0_nyq"][k] = 0.0 if (self.m0_nyq is None) else float(self.m0_nyq)
                    self.keep["P0_harm"][k] = self.P0_harm
                elif self.seasonal_mode == "deterministic":
                    if self.K > 0:
                        self.keep["m0_cos"][k, :] = self.m0_cos
                        self.keep["m0_sin"][k, :] = self.m0_sin
                    if self.use_nyq: self.keep["m0_nyq"][k] = 0.0 if (self.m0_nyq is None) else float(self.m0_nyq)

                if self.dim > 0:
                    w = min(self.x.shape[1], self.keep["x"].shape[2])
                    self.keep["x"][k, :, :w] = self.x[1:self.T+1, :w]
                keep_idx += 1

        return self.keep

    # ------------------------------- Persistence ----------------------------

    def save_posterior(self, out_npz_path: str, extra_meta: Optional[dict] = None) -> None:
        os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)
        arrays = dict(self.keep); arrays["y"] = self.y.copy()
        if "x" not in arrays: arrays["x"] = np.zeros((0,0,0))

        if self.true_sigma is not None: arrays["true_sigma"] = float(self.true_sigma)
        if self.true_Q is not None:     arrays["true_Q"] = np.asarray(self.true_Q, float)
        if self.true_mu_t is not None:  arrays["true_mu_t"] = np.asarray(self.true_mu_t, float)

        np.savez_compressed(out_npz_path, **arrays)

        meta = {
            "T": int(self.T),
            "dim": int(self.dim),
            "period": int(self.s),
            "harmonics": int(self.K),
            "use_nyquist": bool(self.use_nyq),
            "modes": {
                "level": self.level_mode, "trend": self.trend_mode, "season": self.seasonal_mode,
            },
            "layout": list(self._layout),
            "cfg": asdict(self.cfg),
            "priors": asdict(self.priors),
            "rj_accept": {
                b: {
                    "proposed": int(self.rj_stats[b]["proposed"]),
                    "accepted": int(self.rj_stats[b]["accepted"]),
                    "acc_rate": (self.rj_stats[b]["accepted"] / max(1, self.rj_stats[b]["proposed"]))
                } for b in ("level","trend","season")
            },
            "inclusion_probs": self.inclusion_probabilities(),
            "modes_encoding": {"dynamic":0,"deterministic":1,"none":2},
        }
        if extra_meta: meta.update(extra_meta)
        meta_path = out_npz_path.replace(".npz", ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f: json.dump(meta, f, indent=2)
        print(f"[save] Posterior -> {out_npz_path}")
        print(f"[save] Metadata  -> {meta_path}")


if __name__ == "__main__":
    import sys, os, argparse, time
    from datetime import datetime
    from typing import Optional, List

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(base_dir)
    from simulator.mean_time_series_harmonic import Mean_Time_Series

    # ---------- helpers ----------
    def _parse_date(s: Optional[str]):
        from datetime import datetime as _dt
        if not s: return _dt.today()
        parts = [int(p) for p in s.split("-")]
        if len(parts) == 1:  return _dt(parts[0], 1, 1)
        if len(parts) == 2:  return _dt(parts[0], parts[1], 1)
        if len(parts) == 3:  return _dt(parts[0], parts[1], parts[2])
        raise ValueError("start-date must be YYYY, YYYY-MM, or YYYY-MM-DD")

    def _csv_floats(s: Optional[str]) -> Optional[List[float]]:
        if s is None: return None
        s = s.strip()
        if s == "": return None
        return [float(z) for z in s.split(",") if z.strip() != ""]

    def _norm_model_prior_block(raw: Optional[str], allow_none: bool, defaults: dict) -> dict:
        out = dict(defaults)
        if raw:
            for p in [q.strip() for q in raw.split(",") if q.strip()]:
                k, v = p.split(":"); out[k.strip()] = float(v)
        if not allow_none: out["none"] = min(out.get("none", 1e-12), 1e-12)
        ssum = sum(out.values())
        return ({k: v/sum(defaults.values()) for k, v in defaults.items()}
                if ssum <= 0 else {k: v/ssum for k, v in out.items()})

    # ---------- CLI ----------
    p = argparse.ArgumentParser(
        description=("Gaussian DLM (level/trend/season) with harmonic seasonality and RJ over "
                     "{dynamic, deterministic, none}. Adds simulator knobs for harmonic means/vars, "
                     "and optional full-length seasonal dummies → FFT projection.")
    )

    # Data / simulation
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--period", type=int, default=4)
    p.add_argument("--start-date", type=str, default="2000-01-01")
    p.add_argument("--sigma", type=float, default=2.0,
                   help="Simulator obs SD (if simulator available).")
    
    p.add_argument("--level-mode", choices=["dynamic", "deterministic"], default="deterministic")
    p.add_argument("--trend-mode", choices=["dynamic", "deterministic", "none"], default="deterministic")
    p.add_argument("--seasonal-mode", choices=["dynamic", "deterministic", "none"], default="deterministic")


    # Simulator process variances (truth)
    p.add_argument("--q-level", type=float, default=0.05)
    p.add_argument("--q-trend", type=float, default=0.000002)
    p.add_argument("--q-season", type=float, default=0.0001,
                   help="One scalar seasonal process variance (simulator)")

    # Simulator INITIAL means/vars for level/trend (truth)
    p.add_argument("--m0-level", type=float, default=5.0)
    p.add_argument("--v0-level", type=float, default=0.25)
    p.add_argument("--m0-trend", type=float, default=0.02)
    p.add_argument("--v0-trend", type=float, default=0.05)

    # Modes at START (sampler; RJ will move)
    p.add_argument("--start-level", choices=["dynamic","deterministic","none"], default="dynamic")
    p.add_argument("--start-trend", choices=["dynamic","deterministic","none"], default="dynamic")
    p.add_argument("--start-season", choices=["dynamic","deterministic","none"], default="deterministic")

    # Harmonic spec (sampler & simulator)
    p.add_argument("--harmonics", type=int, default=None, help="K harmonics; 0..floor((s-1)/2)")
    p.add_argument("--use-nyquist", type=int, default=1,
                   help="1/0; only relevant when period is even.")

    # ---- NEW: harmonic knobs for simulator (truth) ----
    # Provide full-length dummies (length = period) to set harmonic baseline for truth & inits
    p.add_argument("--season-dummies", type=str, default='1,1,1,-3',
                   help="CSV length=period; mean-centered and projected to (cos,sin,nyq)")

    # Direct (cos/sin/nyq) simulator initial means and variances (vectors of length K)
    p.add_argument("--m0-cos", type=str, default=None, help="CSV length K")
    p.add_argument("--m0-sin", type=str, default=None, help="CSV length K")
    p.add_argument("--m0-nyq", type=float, default=None, help="scalar if nyquist active")
    p.add_argument("--v0-cos", type=str, default=None, help="CSV length K (variances)")
    p.add_argument("--v0-sin", type=str, default=None, help="CSV length K (variances)")
    p.add_argument("--v0-nyq", type=float, default=None, help="scalar variance for nyquist")

    # ---- Sampler inits for harmonics (optional; else use FFT from dummies or zeros) ----
    p.add_argument("--m0-cos-init", type=str, default=None, help="CSV length K")
    p.add_argument("--m0-sin-init", type=str, default=None, help="CSV length K")
    p.add_argument("--m0-nyq-init", type=float, default=None, help="scalar if nyquist active")

    # Sampler config
    p.add_argument("--n-iter", type=int, default=20000)
    p.add_argument("--burn", type=int, default=5000)
    p.add_argument("--thin", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--progress", type=int, default=1)
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--print-dummies-every", type=int, default=0)

    # RJ controls
    p.add_argument("--rj-moves-per-iter", type=int, default=4)
    p.add_argument("--allow-none-level", type=int, default=0)
    p.add_argument("--allow-none-trend", type=int, default=0)
    p.add_argument("--allow-none-season", type=int, default=1)

    # Model prior (RJ)
    p.add_argument("--prior-model-level", type=str, default=None,
                   help="CSV 'dynamic:0.5, deterministic:0.5, none:0.0'")
    p.add_argument("--prior-model-trend", type=str, default=None)
    p.add_argument("--prior-model-season", type=str, default=None)

    # Half-Cauchy scales (process SDs)
    p.add_argument("--hc-scale-alpha", type=float, default=0.5)
    p.add_argument("--hc-scale-beta",  type=float, default=0.5)
    p.add_argument("--hc-scale-gamma", type=float, default=0.5)

    # Observation variance prior
    p.add_argument("--prior-a-sigma", type=float, default=2.0)
    p.add_argument("--prior-b-sigma", type=float, default=1.0)

    # m0 priors (sampler)
    p.add_argument("--prior-m-m0-alpha", type=float, default=0.0)
    p.add_argument("--prior-s-m0-alpha", type=float, default=10.0)
    p.add_argument("--prior-m-m0-beta",  type=float, default=0.0)
    p.add_argument("--prior-s-m0-beta",  type=float, default=10.0)
    p.add_argument("--prior-m-m0-cos",   type=str, default=None, help="CSV length K or None")
    p.add_argument("--prior-m-m0-sin",   type=str, default=None, help="CSV length K or None")
    p.add_argument("--prior-m-m0-nyq",   type=float, default=0.0)
    p.add_argument("--prior-s-m0-harm",  type=float, default=5.0)

    # P0 priors (sampler)
    p.add_argument("--prior-a-P0-alpha", type=float, default=2.0)
    p.add_argument("--prior-b-P0-alpha", type=float, default=1.0)
    p.add_argument("--prior-a-P0-beta",  type=float, default=2.0)
    p.add_argument("--prior-b-P0-beta",  type=float, default=1.0)
    p.add_argument("--prior-a-P0-harm",  type=float, default=2.0)
    p.add_argument("--prior-b-P0-harm",  type=float, default=1.0)

    # Initial inference values (sampler)
    p.add_argument("--sigma-init", type=float, default=1.0)
    p.add_argument("--s-alpha-init", type=float, default=1e-2)
    p.add_argument("--s-beta-init",  type=float, default=1e-3)
    p.add_argument("--s-gamma-init", type=float, default=1e-3)
    p.add_argument("--P0-alpha-init", type=float, default=0.25)
    p.add_argument("--P0-beta-init",  type=float, default=0.05)
    p.add_argument("--P0-harm-init",  type=float, default=0.25)

    # Output / UX
    p.add_argument("--out-dir", type=str, default="results/simulations/DLM_RJ_harmonic")
    p.add_argument("--plot", type=int, default=1)
    p.add_argument("--print-summary", type=int, default=1)

    args = p.parse_args()
    np.random.seed(args.seed)

    # Nyquist logic
    even = (args.period % 2 == 0)
    use_nyq = bool(int(args.use_nyquist) and even)
    if args.harmonics is None:
        args.harmonics = (args.period-1)//2
    # 
    # Parse arrays
    m0_cos_init = _csv_floats(args.m0_cos_init)
    m0_sin_init = _csv_floats(args.m0_sin_init)
    m0_cos = _csv_floats(args.m0_cos)
    m0_sin = _csv_floats(args.m0_sin)
    v0_cos = _csv_floats(args.v0_cos)
    v0_sin = _csv_floats(args.v0_sin)
    prior_m_cos  = _csv_floats(args.prior_m_m0_cos)
    prior_m_sin  = _csv_floats(args.prior_m_m0_sin)
    season_dummies = _csv_floats(args.season_dummies)

    # If (cos/sin) inits missing but dummies given, project dummies → (cos,sin,nyq)
    if (m0_cos_init is None or m0_sin_init is None or
        m0_cos is None or m0_sin is None or
        (use_nyq and (args.m0_nyq_init is None or args.m0_nyq is None))):
        if season_dummies is not None:
            if len(season_dummies) != args.period:
                raise ValueError(f"--season-dummies must have length period={args.period}")
            centered = center_and_report_dummies_full(season_dummies, tol=1e-12)
            cos_coefs, sin_coefs, nyq_val = dummies_full_to_harmonics_fft(
                centered, K=args.harmonics, use_nyquist=use_nyq
            )
            if m0_cos_init is None:   m0_cos_init = list(cos_coefs)
            if m0_sin_init is None:   m0_sin_init = list(sin_coefs)
            if m0_cos is None:  m0_cos = list(cos_coefs)
            if m0_sin is None:  m0_sin = list(sin_coefs)
            if use_nyq:
                if args.m0_nyq_init  is None: args.m0_nyq_init  = float(0.0 if nyq_val is None else nyq_val)
                if args.m0_nyq is None: args.m0_nyq = float(0.0 if nyq_val is None else nyq_val)

    # Fallbacks to zeros of correct length
    if m0_cos_init   is None: m0_cos_init   = [0.0] * args.harmonics
    if m0_sin_init   is None: m0_sin_init   = [0.0] * args.harmonics
    if m0_cos  is None: m0_cos  = [0.0] * args.harmonics
    if m0_sin  is None: m0_sin  = [0.0] * args.harmonics
    if v0_cos  is None: v0_cos  = [0.25] * args.harmonics
    if v0_sin  is None: v0_sin  = [0.25] * args.harmonics
    if args.m0_nyq_init  is None:  args.m0_nyq_init  = 0.0
    if args.m0_nyq is None:  args.m0_nyq = 0.0
    if args.v0_nyq is None:  args.v0_nyq = 0.25

    # -------- build simulator (truth) with harmonic knobs (if supported) --------
    sim_kwargs = dict(
        sigma=args.sigma,
        period=args.period,
        level_mode=args.level_mode,
        trend_mode=args.trend_mode,
        seasonal_mode=args.seasonal_mode,
        season_harmonics=args.harmonics,
        season_use_nyquist=use_nyq,
        q_level=args.q_level,
        q_trend=args.q_trend,
        q_season=args.q_season,
        m0_level=args.m0_level, v0_level=args.v0_level,
        m0_trend=args.m0_trend, v0_trend=args.v0_trend,
        start_date=_parse_date(args.start_date),
        rng=np.random.default_rng(args.seed),
    )
    # Soft-inject harmonic means/vars for truth if Mean_Time_Series supports them
    # (avoids breaking if the simulator signature doesn’t have these yet).
    sim_harmonic_opts = {
        "m0_cos": np.asarray(m0_cos, float),
        "m0_sin": np.asarray(m0_sin, float),
        "m0_nyq": float(args.m0_nyq) if use_nyq else None,
        "v0_cos": np.asarray(v0_cos, float),
        "v0_sin": np.asarray(v0_sin, float),
        "v0_nyq": float(args.v0_nyq) if use_nyq else None,
    }
    for k, v in sim_harmonic_opts.items():
        if v is not None:
            sim_kwargs[k] = v

    mts = Mean_Time_Series(**sim_kwargs)
    y = np.array([mts.move() or mts.measure() for _ in range(args.T)], float)
    truth = mts.get_truth_paths(as_numpy=True)
    mu_T = truth["mu_t"][1:1 + args.T]
    dates_T = truth["index"][:args.T]

    # -------- Priors --------
    priors = Priors(
        a_sigma=args.prior_a_sigma, b_sigma=args.prior_b_sigma,
        m_m0_alpha=args.prior_m_m0_alpha, s_m0_alpha=args.prior_s_m0_alpha,
        m_m0_beta=args.prior_m_m0_beta,  s_m0_beta=args.prior_s_m0_beta,
        m_m0_cos=None if prior_m_cos is None else prior_m_cos,
        m_m0_sin=None if prior_m_sin is None else prior_m_sin,
        m_m0_nyq=args.prior_m_m0_nyq,
        s_m0_harm=args.prior_s_m0_harm,
        a_P0_alpha=args.prior_a_P0_alpha, b_P0_alpha=args.prior_b_P0_alpha,
        a_P0_beta=args.prior_a_P0_beta,   b_P0_beta=args.prior_b_P0_beta,
        a_P0_harm=args.prior_a_P0_harm,   b_P0_harm=args.prior_b_P0_harm,
        hc_scale_alpha=args.hc_scale_alpha,
        hc_scale_beta=args.hc_scale_beta,
        hc_scale_gamma=args.hc_scale_gamma,
    )

    # -------- Model prior (RJ) --------
    default_model_prior = {
        "level":  {"dynamic": 0.1, "deterministic": 0.9, "none": 1e-12},
        "trend":  {"dynamic": 0.1, "deterministic": 0.9, "none": 0.1},
        "season": {"dynamic": 0.1, "deterministic": 0.9,  "none": 0.1},
    }
    model_prior = {
        "level": _norm_model_prior_block(args.prior_model_level, bool(args.allow_none_level), default_model_prior["level"]),
        "trend": _norm_model_prior_block(args.prior_model_trend, bool(args.allow_none_trend), default_model_prior["trend"]),
        "season": _norm_model_prior_block(args.prior_model_season, bool(args.allow_none_season), default_model_prior["season"]),
    }

    # -------- Sampler config --------
    cfg = SamplerConfig(
        n_iter=int(args.n_iter), burn=int(args.burn), thin=int(args.thin),
        random_seed=int(args.seed),
        progress=bool(args.progress), progress_every=int(args.progress_every),
        print_dummies_every=int(args.print_dummies_every),
        rj_moves_per_iter=int(args.rj_moves_per_iter),
        allow_none_level=bool(args.allow_none_level),
        allow_none_trend=bool(args.allow_none_trend),
        allow_none_season=bool(args.allow_none_season),
    )

    # -------- Build sampler --------
    sampler = DLMRJGibbsHarmonic(
        y=y,
        period=args.period,
        harmonics=args.harmonics,
        use_nyquist=use_nyq,
        level_mode=args.start_level,
        trend_mode=args.start_trend,
        seasonal_mode=args.start_season,
        sigma2_init=args.sigma_init ** 2,
        s_alpha_init=args.s_alpha_init,
        s_beta_init=args.s_beta_init,
        s_gamma_init=args.s_gamma_init,
        P0_alpha_init=args.P0_alpha_init,
        P0_beta_init=args.P0_beta_init,
        P0_harm_init=args.P0_harm_init,
        m0_alpha_init=0.0,
        m0_beta_init=0.0 if args.start_trend != "none" else 0.0,
        m0_cos_init=m0_cos_init,
        m0_sin_init=m0_sin_init,
        m0_nyq_init=args.m0_nyq_init,
        priors=priors,
        cfg=cfg,
        model_prior=model_prior,
        rng=np.random.default_rng(args.seed),
    )
    sampler.set_truth_paths(mu=mu_T)

    if args.print_summary:
        print(f"\nSimulated {args.T} obs with START modes: "
              f"{args.start_level}/{args.start_trend}/{args.start_season}.")
        print(f"Harmonics: K={sampler.K}, Nyquist={sampler.use_nyq}")
        print("Model prior (normalized):")
        for k in ("level","trend","season"):
            mp = model_prior[k]
            print(f"  {k:6s}: dyn={mp['dynamic']:.3f}, det={mp['deterministic']:.3f}, none={mp['none']:.3f}")

    # -------- Run --------
    t0 = time.time()
    post = sampler.run()
    elapsed = time.time() - t0
    print(f"\n[Run completed in {elapsed:.1f}s]")

    # -------- Save --------
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        args.out_dir,
        f"{args.start_level}-{args.start_trend}-{args.start_season}"
        f"_K{sampler.K}_nyq{int(bool(sampler.use_nyq))}_{stamp}"
    )
    os.makedirs(out_dir, exist_ok=True)

    sampler.save_posterior(
        out_npz_path=os.path.join(out_dir, "posterior.npz"),
        extra_meta={
            "elapsed_seconds": float(elapsed),
            "start_modes": {"level": args.start_level, "trend": args.start_trend, "season": args.start_season},
            "hc_scales": {"alpha": priors.hc_scale_alpha, "beta": priors.hc_scale_beta, "gamma": priors.hc_scale_gamma},
            "model_prior": model_prior,
        },
    )

    inc = sampler.inclusion_probabilities()
    with open(os.path.join(out_dir, "inclusion_probs.json"), "w", encoding="utf-8") as f:
        json.dump(inc, f, indent=2)

    rj_dump = {}
    for b in ("level","trend","season"):
        s = sampler.rj_stats[b]
        rj_dump[b] = {"proposed": int(s["proposed"]), "accepted": int(s["accepted"]),
                      "acc_rate": (s["accepted"] / max(1, s["proposed"]))}
    with open(os.path.join(out_dir, "rj_stats.json"), "w", encoding="utf-8") as f:
        json.dump(rj_dump, f, indent=2)

    if args.print_summary:
        def _finite(v):
            vv = np.asarray(v, float).ravel()
            return vv[np.isfinite(vv)]
        def _summ(name, a):
            v = _finite(a)
            if v.size:
                q = np.quantile(v, [0.05, 0.5, 0.95])
                print(f"  {name:12s} mean={v.mean():.4g} sd={v.std(ddof=1):.4g} "
                      f"[{q[0]:.4g}, {q[1]:.4g}, {q[2]:.4g}] n={v.size}")
        print("\n--- Posterior summaries ---")
        _summ("sigma", post.get("sigma", []))
        for key in ("Q_alpha", "Q_beta", "Q_gamma"):
            if key in post: _summ(key, post[key])

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            mu_hat = post["mu"].mean(axis=0)
            plt.figure(figsize=(10, 4))
            plt.plot(dates_T, y, label="y_t", lw=1)
            if sampler.true_mu_t is not None:
                plt.plot(dates_T, mu_T, "--", label="μ_t (truth)")
            plt.plot(dates_T, mu_hat, "-.", label="μ̂_t (post mean)")
            cur = f"{sampler.level_mode}/{sampler.trend_mode}/{sampler.seasonal_mode}"
            plt.title(f"DLM RJ harmonic — current modes: {cur} | K={sampler.K}, nyq={sampler.use_nyq}")
            plt.grid(True); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "fit.png"), dpi=160)
            plt.show()
        except Exception as e:
            print(f"[plot] skipped ({e})")

    print(f"[save] Outputs -> {out_dir}")
