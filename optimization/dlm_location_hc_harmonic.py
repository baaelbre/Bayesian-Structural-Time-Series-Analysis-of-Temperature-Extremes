from __future__ import annotations

import json, math, os, warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
warnings.filterwarnings("ignore", category=DeprecationWarning)

# =============================================================================
# Small utils
# =============================================================================

def _mad(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    if v.size == 0:
        return 0.0
    med = np.median(v)
    return float(np.median(np.abs(v - med)))

def _robust_sd(v: np.ndarray) -> float:
    return _mad(np.asarray(v, float)) / 1.4826 if v.size else 0.0

def _spd_solve(M: np.ndarray, B: np.ndarray, eps: float = 1e-10) -> np.ndarray:
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

# ---- Harmonic <-> dummies helpers -------------------------------------------
# New: dummies input is length s-1; last element is implied to enforce sum-to-zero.

def _extend_sum_to_zero(d_short: Sequence[float], period: int) -> np.ndarray:
    """Given s-1 seasonal dummies, append the s-th so that the sum is zero."""
    d_short = np.asarray(d_short, float)
    if d_short.size != period - 1:
        raise ValueError(f"season_dummies must have length period-1 = {period-1}")
    last = -float(np.sum(d_short))
    return np.concatenate([d_short, np.array([last], float)])

def _dummies_to_harmonics(
    dummies_short: Sequence[float],
    period: int,
    K: int,
    use_nyquist: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    """
    Project a seasonal dummy vector (provided as length s-1; last implied)
    onto the first K real Fourier harmonics. Returns (cos[K], sin[K], nyq_or_None).
    """
    s = int(period)
    if s < 2:
        raise ValueError("period s must be >= 2")
    x = _extend_sum_to_zero(dummies_short, s)  # now length s and sum zero

    K_full = (s - 1) // 2
    if not (0 <= K <= K_full):
        raise ValueError(f"K must be in [0, {(s-1)//2}] for s={s}")

    even = (s % 2) == 0
    use_nyquist = bool(use_nyquist and even)

    t = np.arange(s, dtype=float)
    c = np.zeros(K, float)
    sines = np.zeros(K, float)

    # No mean-centering needed: sum(x)=0 already
    for k in range(1, K + 1):
        w = 2.0 * np.pi * k / s
        c[k-1]     = (2.0 / s) * np.dot(x, np.cos(w * t))
        sines[k-1] = (2.0 / s) * np.dot(x, np.sin(w * t))

    nyq_val: Optional[float] = None
    if use_nyquist:
        nyq_vec = (-1.0) ** t  # cos(pi t), sin(pi t)=0
        nyq_val = (2.0 / s) * float(np.dot(x, nyq_vec))

    return c, sines, nyq_val

def _harmonics_to_dummies(
    period: int,
    cos_coefs: Sequence[float],
    sin_coefs: Sequence[float],
    use_nyquist: bool,
    nyq_coef: Optional[float] = None,
) -> np.ndarray:
    """
    Reconstruct the (length s) seasonal pattern from harmonics; mean is ~0
    (no k=0 term). If even and use_nyquist, include Nyquist. Returns length s.
    """
    s = int(period)
    K = int(len(cos_coefs))
    cos_coefs = np.asarray(cos_coefs, float)
    sin_coefs = np.asarray(sin_coefs, float)
    t = np.arange(s, dtype=float)
    out = np.zeros(s, float)
    for k in range(1, K + 1):
        w = 2.0 * np.pi * k / s
        out += cos_coefs[k-1] * np.cos(w * t) + sin_coefs[k-1] * np.sin(w * t)
    if use_nyquist and (s % 2 == 0) and (nyq_coef is not None):
        out += float(nyq_coef) * ((-1.0) ** t)
    # guard tiny drift due to numerics: enforce exact sum-to-zero
    out -= np.mean(out)
    out[:-1] -= (np.sum(out) / (s - 1.0))  # distribute minutely; last will auto fix to 0-sum
    out[-1] = -np.sum(out[:-1])
    return out

# =============================================================================
# Priors & Config
# =============================================================================

@dataclass
class Priors:
    a_sigma: float = 2.0
    b_sigma: float = 1.0

    m_m0_alpha: float = 0.0
    s_m0_alpha: float = 10.0
    m_m0_beta: float = 0.0
    s_m0_beta: float = 10.0

    m_m0_cos: Optional[Sequence[float]] = None
    m_m0_sin: Optional[Sequence[float]] = None
    m_m0_nyq: float = 0.0
    s_m0_harm: float = 5.0

    a_P0_alpha: float = 2.0
    b_P0_alpha: float = 1.0
    a_P0_beta: float = 2.0
    b_P0_beta: float = 1.0
    a_P0_harm: float = 2.0
    b_P0_harm: float = 1.0

    hc_scale_alpha: float = 0.5
    hc_scale_beta:  float = 0.5
    hc_scale_gamma: float = 0.5

@dataclass
class SamplerConfig:
    n_iter: int = 4000
    burn: int = 1000
    thin: int = 1
    random_seed: Optional[int] = 7
    progress: bool = True
    progress_every: int = 0  # 0 => ~2% of n_iter
    # NEW: print reconstructed dummies (length s-1) every N iters (0=off)
    print_dummies_every: int = 0

# =============================================================================
# DLM Sampler – harmonic seasonality (cos/sin pairs + optional Nyquist)
# =============================================================================

class DLMGibbsHarmonic:
    """
    Gaussian structural DLM with harmonic seasonal states.

    State (dynamic):
      [alpha] [beta] [c1 s1 | c2 s2 | ... | cK sK | nyq?]

    Observation (constant H):
      y_t = alpha_t  + sum_k c_{k,t}  + (nyq_t if used) + ε_t
      (sin states s_{k,t} are not directly loaded in H; the rotation in A
       makes their effect appear over time.)

    Transition:
      alpha_t   = alpha_{t-1} + beta_{t-1} (if trend dynamic) + η_{α,t}
      beta_t    = beta_{t-1} + η_{β,t}
      [c_k; s_k]_t = R(ω_k) [c_k; s_k]_{t-1} + η_{k,t},   η_{k,t} ~ N(0, q_γ I_2)
      nyq_t     = (-1) * nyq_{t-1} + η_{nyq,t},            η_{nyq,t} ~ N(0, q_γ)

    Process noises:
      q_α = s_α^2,  q_β = s_β^2,  q_γ = s_γ^2  (single s_γ shared by all seasonal coords)
    """

    # --------------------------- Construction --------------------------- #
    def __init__(
        self,
        y: np.ndarray,
        period: int,
        harmonics: Optional[int] = None,
        use_nyquist: Optional[bool] = None,
        level_mode: str = "dynamic",
        trend_mode: str = "dynamic",
        seasonal_mode: str = "dynamic",
        m0_alpha_init: float = 0.0,
        P0_alpha_init: float = 1.0,
        m0_beta_init: float = 0.0,
        P0_beta_init: float = 1.0,
        # (harmonic) initial means/vars for dynamic season
        m0_cos_init: Optional[Sequence[float]] = None,
        m0_sin_init: Optional[Sequence[float]] = None,
        m0_nyq_init: float = 0.0,
        P0_harm_init: float = 1.0,
        # observation variance + process SD inits
        sigma2_init: float = 1.0,
        s_alpha_init: float = 1e-2,
        s_beta_init: float = 1e-3,
        s_gamma_init: float = 1e-3,
        # user-provided harmonics OR dummies (harmonics override if both given)
        m0_cos_provided: Optional[Sequence[float]] = None,
        m0_sin_provided: Optional[Sequence[float]] = None,
        m0_nyq_provided: Optional[float] = None,
        season_dummies_short: Optional[Sequence[float]] = None,  # length s-1
        # priors / config
        priors: Priors = Priors(),
        cfg: SamplerConfig = SamplerConfig(),
    ):
        # ---------------- data / spec ----------------
        self.y = np.asarray(y, float)
        self.T = int(self.y.size)
        self.s = int(period)
        if self.s < 2:
            raise ValueError("period must be >= 2")

        # harmonic resolution (K & Nyquist)
        K_full = (self.s - 1) // 2
        self.K = K_full if harmonics is None else int(harmonics)
        if not (0 <= self.K <= K_full):
            raise ValueError(f"harmonics K must be in [0, {K_full}] for s={self.s}")
        even = (self.s % 2) == 0
        if use_nyquist is None:
            self.use_nyq = bool(even and (self.K >= (self.s // 2 - 1)))
        else:
            self.use_nyq = bool(use_nyquist and even)

        # modes
        ok = {"dynamic", "deterministic", "none"}
        if level_mode not in ok or trend_mode not in ok or seasonal_mode not in ok:
            raise ValueError("invalid mode")
        if trend_mode == "dynamic" and level_mode != "dynamic":
            raise ValueError("dynamic trend requires dynamic level")
        self.level_mode, self.trend_mode, self.seasonal_mode = level_mode, trend_mode, seasonal_mode

        # RNG / config
        self.priors, self.cfg = priors, cfg
        if cfg.random_seed is not None:
            np.random.seed(cfg.random_seed)

        # ---------------- harmonics from user / dummies ----------------
        have_harm = (m0_cos_provided is not None) and (m0_sin_provided is not None)
        if have_harm:
            if len(m0_cos_provided) != self.K or len(m0_sin_provided) != self.K:
                raise ValueError(f"provided m0_cos/m0_sin must have length K={self.K}")
            base_cos = np.asarray(m0_cos_provided, float)
            base_sin = np.asarray(m0_sin_provided, float)
            base_nyq = (float(m0_nyq_provided) if (self.use_nyq and m0_nyq_provided is not None) else
                        (0.0 if self.use_nyq else None))
            source = "harmonics(provided)"
        else:
            if season_dummies_short is None:
                # default: flat zero seasonal dummies
                season_dummies_short = np.zeros(self.s - 1, float)
            season_dummies_short = np.asarray(season_dummies_short, float)
            c, sines, nyq_val = _dummies_to_harmonics(
                season_dummies_short, period=self.s, K=self.K, use_nyquist=self.use_nyq
            )
            base_cos, base_sin = c, sines
            base_nyq = float(0.0 if nyq_val is None else nyq_val) if self.use_nyq else None
            source = "dummies(projected s-1)"

        # rotation caches
        self._omegas = 2.0 * np.pi * (np.arange(1, self.K + 1, dtype=float)) / float(self.s)
        self._cos = np.cos(self._omegas)
        self._sin = np.sin(self._omegas)

        # ---------------- layout (dynamic) ----------------
        layout: List[str] = []
        if self.level_mode == "dynamic":
            layout.append("alpha")
        if self.trend_mode == "dynamic":
            layout.append("beta")
        if self.seasonal_mode == "dynamic":
            for k in range(1, self.K + 1):
                layout += [f"c{k}", f"s{k}"]
            if self.use_nyq:
                layout.append("nyq")
        self._layout = layout
        self.dim = len(layout)

        self.idx_alpha = layout.index("alpha") if "alpha" in layout else None
        self.idx_beta  = layout.index("beta")  if "beta"  in layout else None

        # pair indexer
        def _idx_pair(k: int) -> int:
            pos = 0
            if self.idx_alpha is not None: pos += 1
            if self.idx_beta  is not None: pos += 1
            pos += 2 * (k - 1)
            return pos
        self._idx_pair = _idx_pair

        if self.seasonal_mode == "dynamic":
            self.idx_first_season = (_idx_pair(1) if self.K > 0 else None)
            self.idx_nyq = None
            if self.use_nyq:
                self.idx_nyq = (0
                    + (1 if self.idx_alpha is not None else 0)
                    + (1 if self.idx_beta  is not None else 0)
                    + 2*self.K)
        else:
            self.idx_first_season = None
            self.idx_nyq = None

        # ---------------- parameters / inits ----------------
        self.sigma2 = float(sigma2_init)
        self.s_alpha = float(s_alpha_init) if self.idx_alpha is not None else 0.0
        self.s_beta  = float(s_beta_init)  if self.idx_beta  is not None else 0.0
        self.s_gamma = float(s_gamma_init) if self.seasonal_mode == "dynamic" else 0.0

        # Aux variables for Half-Cauchy mixtures
        self.a_alpha = 1.0
        self.a_beta  = 1.0
        self.a_gamma = 1.0

        # Initial m0/P0 for dynamic coords
        self.m0_alpha = float(m0_alpha_init) if self.idx_alpha is not None else 0.0
        self.P0_alpha = float(P0_alpha_init) if self.idx_alpha is not None else 0.0
        self.m0_beta  = float(m0_beta_init)  if self.idx_beta  is not None else 0.0
        self.P0_beta  = float(P0_beta_init)  if self.idx_beta  is not None else 0.0

        if self.seasonal_mode == "dynamic":
            if m0_cos_init is None: m0_cos_init = base_cos
            if m0_sin_init is None: m0_sin_init = base_sin
            if len(m0_cos_init) != self.K or len(m0_sin_init) != self.K:
                raise ValueError("m0_cos_init and m0_sin_init must have length K")
            self.m0_cos = np.asarray(m0_cos_init, float)
            self.m0_sin = np.asarray(m0_sin_init, float)
            self.m0_nyq = (float(m0_nyq_init) if self.use_nyq else None)
            self.P0_harm = float(P0_harm_init)
        else:
            # deterministic seasonal coefficients (fixed parameters to estimate)
            self.m0_cos = np.asarray(base_cos, float)
            self.m0_sin = np.asarray(base_sin, float)
            self.m0_nyq = float(0.0 if base_nyq is None else base_nyq) if self.use_nyq else None
            self.P0_harm = 0.0

        # latent path
        self.x = np.zeros((self.T + 1, self.dim), float)
        if self.dim > 0:
            m0_vec, P0_diag = self._current_m0_P0()
            self.x[0] = np.random.multivariate_normal(
                m0_vec, np.diag(P0_diag) + 1e-12 * np.eye(self.dim)
            )
            self._propagate_initial_path(Q_init=np.full(self.dim, 1e-6, float))

        self._season_note = f"init={source} | K={self.K} nyq={self.use_nyq}"

        # storage
        self.keep: Dict[str, np.ndarray] = {}

        # ---- Truth overlays (optional) ----
        self.true_sigma: Optional[float] = None
        self.true_Q: Optional[np.ndarray] = None
        self.true_mu_t: Optional[np.ndarray] = None
        self.true_alpha_t: Optional[np.ndarray] = None
        self.true_beta_t: Optional[np.ndarray] = None
        self.true_gamma_t: Optional[np.ndarray] = None

        # handy scales
        if self.cfg.progress:
            y = self.y
            sd1 = _robust_sd(np.diff(y)) if y.size >= 2 else 0.0
            sd2 = _robust_sd(np.diff(y, n=2)) if y.size >= 3 else 0.0
            print(f"[init] sd1={sd1:.4g}, sd2={sd2:.4g} | {self._season_note}")

    # --------------------- Truth overlays (optional) --------------------- #
    def set_truth(self, **kwargs) -> None:
        self.true_sigma = float(kwargs["sigma"]) if "sigma" in kwargs and kwargs["sigma"] is not None else None
        self.true_Q = np.asarray(kwargs["Q"], float) if "Q" in kwargs and kwargs["Q"] is not None else None
        self.true_mu_t = np.asarray(kwargs["mu"], float) if "mu" in kwargs and kwargs["mu"] is not None else None
        self.true_alpha_t = np.asarray(kwargs["alpha"], float) if "alpha" in kwargs and kwargs["alpha"] is not None else None
        self.true_beta_t = np.asarray(kwargs["beta"], float) if "beta" in kwargs and kwargs["beta"] is not None else None
        self.true_gamma_t = np.asarray(kwargs["gamma"], float) if "gamma" in kwargs and kwargs["gamma"] is not None else None

    def set_truth_paths(self, mu: Optional[np.ndarray] = None, **_) -> None:
        self.true_mu_t = None if mu is None else np.asarray(mu, float)

    # ----------------------------- Model matrices -----------------------------

    def _H(self) -> np.ndarray:
        if self.dim == 0:
            return np.zeros((1, 0))
        h = np.zeros(self.dim, float)
        if self.idx_alpha is not None:
            h[self.idx_alpha] = 1.0
        if self.seasonal_mode == "dynamic" and self.K > 0:
            for k in range(1, self.K + 1):
                i = self._idx_pair(k)
                h[i] = 1.0  # load cos only
            if self.use_nyq:
                h[self.idx_nyq] = 1.0
        return h.reshape(1, -1)

    def _A(self) -> np.ndarray:
        if self.dim == 0:
            return np.zeros((0, 0))
        A = np.eye(self.dim)
        if self.idx_alpha is not None and self.idx_beta is not None:
            A[self.idx_alpha, self.idx_beta] = 1.0
        if self.seasonal_mode == "dynamic" and self.K > 0:
            for k in range(1, self.K + 1):
                i = self._idx_pair(k)
                co, si = float(self._cos[k - 1]), float(self._sin[k - 1])
                A[i,   i  ] =  co
                A[i,   i+1] =  si
                A[i+1, i  ] = -si
                A[i+1, i+1] =  co
            if self.use_nyq:
                A[self.idx_nyq, self.idx_nyq] = -1.0
        return A

    def _u(self) -> np.ndarray:
        if self.dim == 0:
            return np.zeros(0, float)
        u = np.zeros(self.dim, float)
        if (self.idx_alpha is not None) and (self.trend_mode == "deterministic"):
            u[self.idx_alpha] = float(self.m0_beta)
        return u

    def _Q(self) -> np.ndarray:
        if self.dim == 0:
            return np.zeros((0, 0))
        Q = np.zeros((self.dim, self.dim))
        if self.idx_alpha is not None and self.s_alpha > 0:
            Q[self.idx_alpha, self.idx_alpha] = self.s_alpha**2
        if self.idx_beta is not None and self.s_beta > 0:
            Q[self.idx_beta, self.idx_beta] = self.s_beta**2
        if self.seasonal_mode == "dynamic" and self.s_gamma > 0:
            for k in range(1, self.K + 1):
                i = self._idx_pair(k)
                Q[i, i] = self.s_gamma**2
                Q[i+1, i+1] = self.s_gamma**2
            if self.use_nyq:
                Q[self.idx_nyq, self.idx_nyq] = self.s_gamma**2
        return Q

    # ---------------- deterministic mean pieces ----------------

    def _season_det(self, t: int) -> float:
        if self.seasonal_mode != "deterministic":
            return 0.0
        val = 0.0
        for k in range(1, self.K + 1):
            w = self._omegas[k - 1]
            val += self.m0_cos[k - 1] * math.cos(w * t) + self.m0_sin[k - 1] * math.sin(w * t)
        if self.use_nyq and (self.m0_nyq is not None):
            val += float(self.m0_nyq) * ((-1.0) ** t)
        return float(val)

    def _mu_det(self, t: int) -> float:
        out = 0.0
        if self.level_mode == "deterministic":
            out += self.m0_alpha
        if (self.trend_mode == "deterministic") and (self.idx_alpha is None):
            out += self.m0_beta * t
        out += self._season_det(t)
        return out

    def _current_m0_P0(self) -> Tuple[np.ndarray, np.ndarray]:
        m0, P0 = [], []
        if self.idx_alpha is not None:
            m0.append(self.m0_alpha); P0.append(self.P0_alpha)
        if self.idx_beta is not None:
            m0.append(self.m0_beta);  P0.append(self.P0_beta)
        if self.seasonal_mode == "dynamic":
            for k in range(self.K):
                m0 += [self.m0_cos[k], self.m0_sin[k]]
                P0 += [self.P0_harm,   self.P0_harm]
            if self.use_nyq:
                m0.append(float(0.0 if self.m0_nyq is None else self.m0_nyq))
                P0.append(self.P0_harm)
        return np.asarray(m0, float), np.asarray(P0, float)

    # ------------------------- FFBS (Kalman + Carter–Kohn) -------------------------

    def _ffbs(self) -> np.ndarray:
        if self.dim == 0:
            return self.x.copy()
        H, A, Q, R = self._H(), self._A(), self._Q(), float(self.sigma2)
        m0_vec, P0_diag = self._current_m0_P0()
        m = np.zeros((self.T + 1, self.dim))
        C = np.zeros((self.T + 1, self.dim, self.dim))
        a = np.zeros((self.T + 1, self.dim))
        Rm = np.zeros((self.T + 1, self.dim, self.dim))
        m[0] = m0_vec
        C[0] = np.diag(P0_diag) + 1e-12 * np.eye(self.dim)
        u = self._u()

        # forward
        for t in range(1, self.T + 1):
            a[t]  = A @ m[t - 1] + u
            Rm[t] = A @ C[t - 1] @ A.T + Q
            Rm[t] = 0.5 * (Rm[t] + Rm[t].T) + 1e-12 * np.eye(self.dim)

            resid_mean = float(self.y[t - 1] - self._mu_det(t - 1))
            S = float(H @ Rm[t] @ H.T + R)
            if S <= 0:
                S = float(H @ (Rm[t] + 1e-10 * np.eye(self.dim)) @ H.T + R)
            K = (Rm[t] @ H.T) / S
            v = resid_mean - float(H @ a[t])
            m[t] = a[t] + (K.flatten() * v)
            C[t] = Rm[t] - K @ (H @ Rm[t])
            C[t] = 0.5 * (C[t] + C[t].T) + 1e-12 * np.eye(self.dim)

        # backward
        x = np.zeros_like(self.x)
        x[self.T] = np.random.multivariate_normal(m[self.T], C[self.T])
        for t in range(self.T - 1, -1, -1):
            J = C[t] @ A.T
            J = J @ _spd_solve(Rm[t + 1], np.eye(self.dim))
            mean = m[t] + J @ (x[t + 1] - (A @ m[t] + u))
            cov  = C[t] - J @ Rm[t + 1] @ J.T
            cov  = 0.5 * (cov + cov.T)
            eigmin = float(np.linalg.eigvalsh(cov).min())
            if eigmin < 1e-12:
                cov += (1e-12 - eigmin) * np.eye(cov.shape[0])
            x[t] = np.random.multivariate_normal(mean, cov)
        return x

    def _propagate_initial_path(self, Q_init: np.ndarray) -> None:
        if self.dim == 0:
            return
        A, u = self._A(), self._u()
        for t in range(1, self.T + 1):
            self.x[t] = A @ self.x[t - 1] + u + np.random.normal(0.0, np.sqrt(Q_init), size=self.dim)

    # ------------------ Helpers: μ and residuals ------------------ #
    def _mu_vec(self) -> np.ndarray:
        H = self._H()
        mu = np.zeros(self.T, float)
        for t in range(1, self.T + 1):
            dyn = float(H @ self.x[t]) if self.dim > 0 else 0.0
            mu[t - 1] = self._mu_det(t - 1) + dyn
        return mu

    # ------------------ σ² | rest  (Gamma on precision) ------------------ #
    def update_sigma2(self) -> None:
        e = self.y - self._mu_vec()
        a = self.priors.a_sigma + 0.5 * self.T
        b = self.priors.b_sigma + 0.5 * float(e @ e)
        tau = np.random.gamma(shape=a, scale=1.0 / b)  # precision
        self.sigma2 = 1.0 / max(tau, 1e-300)

    # ------------- Innovation sums of squares (for Q updates) ------------- #
    def _innovation_ss_alpha(self) -> Tuple[float, int]:
        if self.idx_alpha is None:
            return 0.0, 0
        ss = 0.0
        for t in range(1, self.T + 1):
            drift = 0.0
            if self.idx_beta is not None:
                drift = self.x[t - 1, self.idx_beta]
            elif self.trend_mode == "deterministic":
                drift = float(self.m0_beta)
            mean = self.x[t - 1, self.idx_alpha] + drift
            ss += (self.x[t, self.idx_alpha] - mean) ** 2
        return float(ss), self.T

    def _innovation_ss_beta(self) -> Tuple[float, int]:
        if self.idx_beta is None:
            return 0.0, 0
        d = self.x[1:, self.idx_beta] - self.x[:-1, self.idx_beta]
        return float(np.sum(d * d)), self.T

    def _innovation_ss_gamma(self) -> Tuple[float, int]:
        if self.seasonal_mode != "dynamic":
            return 0.0, 0
        ss = 0.0
        per_step = 0
        for k in range(1, self.K + 1):
            i = self._idx_pair(k)
            co, si = float(self._cos[k - 1]), float(self._sin[k - 1])
            R = np.array([[co, si], [-si, co]], float)
            for t in range(1, self.T + 1):
                prev = self.x[t - 1, i:i+2]
                mean = R @ prev
                err = self.x[t, i:i+2] - mean
                ss += float(err @ err)
        per_step += 2 * self.K
        if self.use_nyq:
            j = self.idx_nyq
            for t in range(1, self.T + 1):
                mean = -self.x[t - 1, j]
                err = self.x[t, j] - mean
                ss += float(err * err)
            per_step += 1
        T_eff = self.T * per_step
        return float(ss), int(T_eff)

    # =============================================================================
    # Pure Gibbs for process variances with Half-Cauchy priors
    # =============================================================================
    @staticmethod
    def _sample_invgamma(shape: float, scale: float) -> float:
        return 1.0 / np.random.gamma(shape, 1.0 / scale)

    def update_process_Q_halfcauchy(self) -> None:
        # α
        if self.idx_alpha is not None:
            SS, T_eff = self._innovation_ss_alpha()
            A = float(self.priors.hc_scale_alpha)
            shape_Q = 0.5 * T_eff + 0.5
            scale_Q = 0.5 * SS + 1.0 / max(self.a_alpha, 1e-300)
            Q_alpha = self._sample_invgamma(shape_Q, scale_Q)
            self.s_alpha = math.sqrt(max(Q_alpha, 0.0))
            self.a_alpha = self._sample_invgamma(1.0, (1.0 / (A * A)) + (1.0 / max(Q_alpha, 1e-300)))
        # β
        if self.idx_beta is not None:
            SS, T_eff = self._innovation_ss_beta()
            A = float(self.priors.hc_scale_beta)
            shape_Q = 0.5 * T_eff + 0.5
            scale_Q = 0.5 * SS + 1.0 / max(self.a_beta, 1e-300)
            Q_beta = self._sample_invgamma(shape_Q, scale_Q)
            self.s_beta = math.sqrt(max(Q_beta, 0.0))
            self.a_beta = self._sample_invgamma(1.0, (1.0 / (A * A)) + (1.0 / max(Q_beta, 1e-300)))
        # γ
        if self.seasonal_mode == "dynamic":
            SS, T_eff = self._innovation_ss_gamma()
            A = float(self.priors.hc_scale_gamma)
            shape_Q = 0.5 * T_eff + 0.5
            scale_Q = 0.5 * SS + 1.0 / max(self.a_gamma, 1e-300)
            Q_gamma = self._sample_invgamma(shape_Q, scale_Q)
            self.s_gamma = math.sqrt(max(Q_gamma, 0.0))
            self.a_gamma = self._sample_invgamma(1.0, (1.0 / (A * A)) + (1.0 / max(Q_gamma, 1e-300)))

    # --- m0 | P0, x0 (Normal); P0 | m0, x0 (Inv-Gamma) --- #

    @staticmethod
    def _gibbs_m0_scalar(x0: float, m_prior: float, s_prior: float, P0: float) -> float:
        prec = 1.0 / (s_prior**2) + 1.0 / max(1e-18, P0)
        var = 1.0 / prec
        mean = var * (m_prior / (s_prior**2) + x0 / max(1e-18, P0))
        return float(np.random.normal(mean, math.sqrt(var)))

    def update_m0(self) -> None:
        if self.dim == 0:
            return
        pos = 0
        if self.idx_alpha is not None:
            self.m0_alpha = self._gibbs_m0_scalar(
                float(self.x[0, pos]), self.priors.m_m0_alpha, self.priors.s_m0_alpha, self.P0_alpha
            ); pos += 1
        if self.idx_beta is not None:
            self.m0_beta = self._gibbs_m0_scalar(
                float(self.x[0, pos]), self.priors.m_m0_beta, self.priors.s_m0_beta, self.P0_beta
            ); pos += 1
        if self.seasonal_mode == "dynamic":
            s0 = float(self.priors.s_m0_harm)
            m_cos = (np.zeros(self.K) if self.priors.m_m0_cos is None
                     else np.asarray(self.priors.m_m0_cos, float))
            m_sin = (np.zeros(self.K) if self.priors.m_m0_sin is None
                     else np.asarray(self.priors.m_m0_sin, float))
            if m_cos.size != self.K or m_sin.size != self.K:
                raise ValueError("priors.m_m0_cos/m_m0_sin must have length K")
            for k in range(self.K):
                self.m0_cos[k] = self._gibbs_m0_scalar(float(self.x[0, pos + 2*k    ]), float(m_cos[k]), s0, self.P0_harm)
                self.m0_sin[k] = self._gibbs_m0_scalar(float(self.x[0, pos + 2*k + 1]), float(m_sin[k]), s0, self.P0_harm)
            if self.use_nyq:
                j = pos + 2*self.K
                self.m0_nyq = self._gibbs_m0_scalar(float(self.x[0, j]), float(self.priors.m_m0_nyq), s0, self.P0_harm)

    def update_P0(self) -> None:
        if self.dim == 0:
            return
        pos = 0
        if self.idx_alpha is not None:
            a = self.priors.a_P0_alpha + 0.5
            b = self.priors.b_P0_alpha + 0.5 * (float(self.x[0, pos]) - self.m0_alpha) ** 2
            self.P0_alpha = 1.0 / np.random.gamma(shape=a, scale=1.0 / b); pos += 1
        if self.idx_beta is not None:
            a = self.priors.a_P0_beta + 0.5
            b = self.priors.b_P0_beta + 0.5 * (float(self.x[0, pos]) - self.m0_beta) ** 2
            self.P0_beta = 1.0 / np.random.gamma(shape=a, scale=1.0 / b); pos += 1
        if self.seasonal_mode == "dynamic":
            diffsq = 0.0
            Ktot = 2*self.K + (1 if self.use_nyq else 0)
            target = [*self.m0_cos, *self.m0_sin] + ([float(self.m0_nyq)] if self.use_nyq else [])
            for k in range(Ktot):
                diffsq += (float(self.x[0, pos + k]) - float(target[k])) ** 2
            a = self.priors.a_P0_harm + 0.5 * Ktot
            b = self.priors.b_P0_harm + 0.5 * diffsq
            self.P0_harm = 1.0 / np.random.gamma(shape=a, scale=1.0 / b)

    # --- Deterministic parameter updates (conjugate) --- #

    def update_deterministic_params(self) -> None:
        # deterministic level
        if self.level_mode == "deterministic":
            r = self.y.copy()
            if self.dim > 0:
                H = self._H()
                for t in range(1, self.T + 1):
                    r[t - 1] -= float(H @ self.x[t])
            r -= np.array([self._season_det(t) for t in range(self.T)], float)
            if (self.idx_alpha is None) and (self.trend_mode == "deterministic"):
                r -= self.m0_beta * np.arange(self.T, dtype=float)
            s2 = float(self.sigma2)
            m0, s0 = float(self.priors.m_m0_alpha), float(self.priors.s_m0_alpha)
            prec = self.T / s2 + 1.0 / (s0**2)
            mean = ((r.sum() / s2) + m0 / (s0**2)) / prec
            var = 1.0 / prec
            self.m0_alpha = float(np.random.normal(mean, math.sqrt(var)))

        # deterministic trend
        if self.trend_mode == "deterministic":
            if self.idx_alpha is not None:
                d = self.x[1:, self.idx_alpha] - self.x[:-1, self.idx_alpha]
                s2 = float(self.s_alpha**2) if self.s_alpha > 0 else 1e-12
                m0, s0 = float(self.priors.m_m0_beta), float(self.priors.s_m0_beta)
                prec = (self.T / s2) + 1.0 / (s0**2)
                mean = ((float(np.sum(d)) / s2) + m0 / (s0**2)) / prec
                var = 1.0 / prec
                self.m0_beta = float(np.random.normal(mean, math.sqrt(var)))
            else:
                t = np.arange(self.T, dtype=float)
                r = self.y.copy()
                if self.dim > 0:
                    H = self._H()
                    for k in range(1, self.T + 1):
                        r[k - 1] -= float(H @ self.x[k])
                if self.level_mode == "deterministic":
                    r -= self.m0_alpha
                r -= np.array([self._season_det(tt) for tt in range(self.T)], float)
                m0, s0 = float(self.priors.m_m0_beta), float(self.priors.s_m0_beta)
                sig2 = float(self.sigma2)
                prec = (t @ t) / sig2 + 1.0 / (s0**2)
                mean = ((t @ r) / sig2 + m0 / (s0**2)) / prec
                var = 1.0 / prec
                self.m0_beta = float(np.random.normal(mean, math.sqrt(var)))

        # deterministic season: regress on [cos_k, sin_k, nyq]
        if self.seasonal_mode == "deterministic":
            t = np.arange(self.T, dtype=float)
            Zcols = []
            for k in range(1, self.K + 1):
                w = self._omegas[k - 1]
                Zcols.append(np.cos(w * t))
                Zcols.append(np.sin(w * t))
            if self.use_nyq:
                Zcols.append(((-1.0) ** t))
            Z = np.column_stack(Zcols) if Zcols else np.zeros((self.T, 0))
            r = self.y.copy()
            if self.dim > 0:
                H = self._H()
                for k in range(1, self.T + 1):
                    r[k - 1] -= float(H @ self.x[k])
            if self.level_mode == "deterministic":
                r -= self.m0_alpha
            if (self.idx_alpha is None) and (self.trend_mode == "deterministic"):
                r -= self.m0_beta * t
            p = Z.shape[1]
            s2 = float(self.priors.s_m0_harm) ** 2
            m_prior = np.zeros(p)
            if (self.priors.m_m0_cos is not None) and (self.priors.m_m0_sin is not None):
                if (len(self.priors.m_m0_cos) == self.K) and (len(self.priors.m_m0_sin) == self.K):
                    m_prior[:2*self.K:2] = np.asarray(self.priors.m_m0_cos, float)
                    m_prior[1:2*self.K:2] = np.asarray(self.priors.m_m0_sin, float)
            if self.use_nyq and (p > 2*self.K):
                m_prior[-1] = float(self.priors.m_m0_nyq)
            sig2 = float(self.sigma2)
            Prec = (Z.T @ Z) / sig2 + np.eye(p) / s2
            b = (Z.T @ r) / sig2 + m_prior / s2
            mu = np.linalg.solve(Prec, b)
            L = np.linalg.cholesky(Prec)
            theta = mu + np.linalg.solve(L.T, np.random.randn(p))
            if self.K > 0:
                self.m0_cos = theta[:2*self.K:2].copy()
                self.m0_sin = theta[1:2*self.K:2].copy()
            if self.use_nyq:
                self.m0_nyq = float(theta[-1])

    # ------------------- Progress & printing dummies ------------------- #

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
        parts = [f"[it {it + 1}/{self.cfg.n_iter}]",
                 f"σ={math.sqrt(self.sigma2):.3f}"]
        if self.idx_alpha is not None: parts.append(f"Qα={self.s_alpha**2:.4g}")
        if self.idx_beta  is not None: parts.append(f"Qβ={self.s_beta**2:.4g}")
        if self.seasonal_mode == "dynamic": parts.append(f"Qγ={self.s_gamma**2:.4g}")
        if self.level_mode != "none":
            parts.append(f"m0α={self.m0_alpha:.4g} P0α={(self.P0_alpha if self.level_mode=='dynamic' else 0.0):.4g}")
        if self.trend_mode != "none":
            parts.append(f"m0β={self.m0_beta:.4g} P0β={(self.P0_beta if self.trend_mode=='dynamic' else 0.0):.4g}")
        if self.seasonal_mode != "none":
            parts.append(f"m0cos={self._fmt_list(self.m0_cos,6)} m0sin={self._fmt_list(self.m0_sin,6)}"
                         + (f" nyq={0.0 if (self.m0_nyq is None) else float(self.m0_nyq):.4g}" if self.use_nyq else ""))
            if self.seasonal_mode == "dynamic":
                parts.append(f"P0harm={self.P0_harm:.4g}")
        return " | ".join(parts)

    def _maybe_print_dummies(self, it: int) -> None:
        n = int(self.cfg.print_dummies_every)
        if n <= 0:
            return
        if (it + 1) % n != 0 and it != self.cfg.n_iter - 1:
            return
        # reconstruct length-s dummies, then print first s-1 as requested
        if self.seasonal_mode in ("dynamic", "deterministic"):
            d = _harmonics_to_dummies(
                period=self.s,
                cos_coefs=self.m0_cos if self.K > 0 else np.zeros(0),
                sin_coefs=self.m0_sin if self.K > 0 else np.zeros(0),
                use_nyquist=self.use_nyq,
                nyq_coef=self.m0_nyq,
            )
            tag = "dyn" if self.seasonal_mode == "dynamic" else "det"
            d_short = d[:-1]
            print(f"[it {it + 1}] dummies({tag}) first s-1 = {self._fmt_list(d_short, max_elems=self.s-1, fmt='.4f')}")

    # --------------------------------- MCMC --------------------------------- #

    def run(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        save_iters = list(range(cfg.burn, cfg.n_iter, cfg.thin))
        n_kept, keep_idx = len(save_iters), 0

        # allocate storage
        self.keep = {"sigma": np.zeros(n_kept, float), "mu": np.zeros((n_kept, self.T), float)}
        if self.idx_alpha is not None:
            self.keep.update({"Q_alpha": np.zeros(n_kept),
                              "m0_alpha": np.zeros(n_kept), "P0_alpha": np.zeros(n_kept)})
        if self.idx_beta is not None:
            self.keep.update({"Q_beta": np.zeros(n_kept),
                              "m0_beta": np.zeros(n_kept), "P0_beta": np.zeros(n_kept)})
        if self.seasonal_mode == "dynamic":
            self.keep.update({
                "Q_gamma": np.zeros(n_kept),
                "m0_cos": np.zeros((n_kept, self.K)),
                "m0_sin": np.zeros((n_kept, self.K)),
                "m0_nyq": np.zeros(n_kept) if self.use_nyq else np.zeros(0),
                "P0_harm": np.zeros(n_kept),
                "x": np.zeros((n_kept, self.T, self.dim)) if self.dim > 0 else np.zeros((0,0,0))
            })
        else:
            self.keep.update({
                "m0_cos": np.zeros((n_kept, self.K)),
                "m0_sin": np.zeros((n_kept, self.K)),
                "m0_nyq": np.zeros(n_kept) if self.use_nyq else np.zeros(0)
            })
            if self.dim > 0:
                self.keep["x"] = np.zeros((n_kept, self.T, self.dim))

        print_every = cfg.progress_every if cfg.progress_every > 0 else max(1, cfg.n_iter // 50) or 1

        for it in range(cfg.n_iter):
            # 1) FFBS
            if self.dim > 0:
                self.x = self._ffbs()

            # 2) process Q via Half-Cauchy IG mixtures (pure Gibbs)
            if self.dim > 0:
                self.update_process_Q_halfcauchy()

            # 3) m0 and 4) P0 for dynamic coords
            if self.dim > 0:
                self.update_m0()
                self.update_P0()

            # 5) deterministic params (level/trend/season)
            self.update_deterministic_params()

            # 6) σ² (Gibbs)
            self.update_sigma2()

            # progress
            if cfg.progress and ((it + 1) % print_every == 0 or it == cfg.n_iter - 1):
                print(self._progress_line(it))

            # print dummies (first s-1) per your request
            self._maybe_print_dummies(it)

            # save
            if it in save_iters:
                mu = self._mu_vec()
                self.keep["mu"][keep_idx, :] = mu
                self.keep["sigma"][keep_idx] = math.sqrt(self.sigma2)
                if self.idx_alpha is not None:
                    self.keep["Q_alpha"][keep_idx] = self.s_alpha**2
                    self.keep["m0_alpha"][keep_idx] = self.m0_alpha
                    self.keep["P0_alpha"][keep_idx] = self.P0_alpha
                if self.idx_beta is not None:
                    self.keep["Q_beta"][keep_idx] = self.s_beta**2
                    self.keep["m0_beta"][keep_idx] = self.m0_beta
                    self.keep["P0_beta"][keep_idx] = self.P0_beta
                if self.seasonal_mode == "dynamic":
                    self.keep["Q_gamma"][keep_idx] = self.s_gamma**2
                    self.keep["m0_cos"][keep_idx, :] = self.m0_cos
                    self.keep["m0_sin"][keep_idx, :] = self.m0_sin
                    if self.use_nyq:
                        self.keep["m0_nyq"][keep_idx] = 0.0 if (self.m0_nyq is None) else float(self.m0_nyq)
                    self.keep["P0_harm"][keep_idx] = self.P0_harm
                else:
                    self.keep["m0_cos"][keep_idx, :] = self.m0_cos
                    self.keep["m0_sin"][keep_idx, :] = self.m0_sin
                    if self.use_nyq:
                        self.keep["m0_nyq"][keep_idx] = 0.0 if (self.m0_nyq is None) else float(self.m0_nyq)
                if "x" in self.keep and self.dim > 0:
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

        if self.true_sigma is not None: arrays["true_sigma"] = float(self.true_sigma)
        if self.true_Q is not None:     arrays["true_Q"] = np.asarray(self.true_Q, float)
        if self.true_mu_t is not None:  arrays["true_mu_t"] = np.asarray(self.true_mu_t, float)
        if hasattr(self, "true_alpha_t") and self.true_alpha_t is not None:
            arrays["true_alpha_t"] = np.asarray(self.true_alpha_t, float)
        if hasattr(self, "true_beta_t") and self.true_beta_t is not None:
            arrays["true_beta_t"] = np.asarray(self.true_beta_t, float)
        if hasattr(self, "true_gamma_t") and self.true_gamma_t is not None:
            arrays["true_gamma_t"] = np.asarray(self.true_gamma_t, float)

        np.savez_compressed(out_npz_path, **arrays)

        meta = {
            "T": int(self.T),
            "dim": int(self.dim),
            "period": int(self.s),
            "harmonics": int(self.K),
            "use_nyquist": bool(self.use_nyq),
            "modes": {
                "level_mode": self.level_mode,
                "trend_mode": self.trend_mode,
                "seasonal_mode": self.seasonal_mode,
            },
            "layout": list(self._layout),
            "cfg": asdict(self.cfg),
            "priors": asdict(self.priors),
            "note": self._season_note,
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
    import argparse, os, sys, time
    from datetime import datetime
    import numpy as np
    import matplotlib.pyplot as plt

    # make simulator importable (expects simulator/mean_time_series_harmonic.py)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(base_dir)
    from simulator.mean_time_series_harmonic import Mean_Time_Series  # harmonic simulator

    # ------------------------- small parsers -------------------------
    def _parse_date(s: str | None):
        if not s:
            from datetime import datetime as _dt
            return _dt.today()
        parts = [int(p) for p in s.split("-")]
        if   len(parts) == 1: return datetime(parts[0], 1, 1)
        elif len(parts) == 2: return datetime(parts[0], parts[1], 1)
        elif len(parts) == 3: return datetime(parts[0], parts[1], parts[2])
        raise ValueError("start-date must be YYYY, YYYY-MM, or YYYY-MM-DD")

    # ------------------------- CLI args -------------------------
    p = argparse.ArgumentParser(
        description=(
            "Gaussian DLM with harmonic seasonality (cos/sin pairs + optional Nyquist). "
            "FFBS + conjugate Gibbs; Half-Cauchy (IG-mixture) on process SDs. "
            "Dummies input is length s-1; last implied."
        )
    )

    # Simulation controls
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--period", type=int, default=12)
    p.add_argument("--start-date", type=str, default="2000-01-01")
    p.add_argument("--sigma", type=float, default=2.0)

    p.add_argument("--level-mode", choices=["dynamic", "deterministic"], default="deterministic")
    p.add_argument("--trend-mode", choices=["dynamic", "deterministic", "none"], default="deterministic")
    p.add_argument("--seasonal-mode", choices=["dynamic", "deterministic", "none"], default="deterministic")

    p.add_argument("--q-level", type=float, default=0.05)
    p.add_argument("--q-trend", type=float, default=0.000002)
    p.add_argument("--q-season", type=float, default=0.0001, help="One scalar seasonal process variance (simulator)")

    # Simulator priors (used for truth generation)
    p.add_argument("--m0-level", type=float, default=5.0)
    p.add_argument("--v0-level", type=float, default=0.25)
    p.add_argument("--m0-trend", type=float, default=0.02)
    p.add_argument("--v0-trend", type=float, default=0.05)

    # Harmonic spec for both simulator & sampler
    p.add_argument("--harmonics", type=int, default=5, help="K; None=full floor((s-1)/2)")
    p.add_argument("--use-nyquist", type=int, default=1, help="1/0; None=auto if even period & K allows Nyquist")

    # Option A: provide harmonic coefficients (overrides dummies if given)
    p.add_argument("--m0-cos", type=str, default="1,1,1,1,1", help="CSV length K")
    p.add_argument("--m0-sin", type=str, default="1,1,1,1,1", help="CSV length K")
    p.add_argument("--m0-nyq", type=float, default=0.0)

    # Option B: provide seasonal dummies (length s-1) to be projected to harmonics
    p.add_argument("--season-dummies", type=str, default=None, help="CSV length=period-1 (last is implied)")
    p.add_argument("--center-dummies", type=int, default=1, help="(kept for compatibility; ignored since sum-to-zero)")

    # Sampler configuration
    p.add_argument("--n-iter", type=int, default=20000)
    p.add_argument("--burn", type=int, default=5000)
    p.add_argument("--thin", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--progress", type=int, default=1)
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--print-dummies-every", type=int, default=1, help="Print first s-1 reconstructed dummies every N iters (0=off)")

    # Half-Cauchy scales (process SDs)
    p.add_argument("--hc-scale-alpha", type=float, default=0.5)
    p.add_argument("--hc-scale-beta",  type=float, default=0.5)
    p.add_argument("--hc-scale-gamma", type=float, default=0.5)

    # Observation variance prior
    p.add_argument("--prior-a-sigma", type=float, default=2.0)
    p.add_argument("--prior-b-sigma", type=float, default=1.0)

    # m0 priors
    p.add_argument("--prior-m-m0-alpha", type=float, default=0.0)
    p.add_argument("--prior-s-m0-alpha", type=float, default=10.0)
    p.add_argument("--prior-m-m0-beta", type=float, default=0.0)
    p.add_argument("--prior-s-m0-beta", type=float, default=10.0)
    p.add_argument("--prior-m-m0-cos", type=str, default=None, help="CSV length K or None")
    p.add_argument("--prior-m-m0-sin", type=str, default=None, help="CSV length K or None")
    p.add_argument("--prior-m-m0-nyq", type=float, default=0.0)
    p.add_argument("--prior-s-m0-harm", type=float, default=5.0)

    # P0 priors
    p.add_argument("--prior-a-P0-alpha", type=float, default=2.0)
    p.add_argument("--prior-b-P0-alpha", type=float, default=1.0)
    p.add_argument("--prior-a-P0-beta", type=float, default=2.0)
    p.add_argument("--prior-b-P0-beta", type=float, default=1.0)
    p.add_argument("--prior-a-P0-harm", type=float, default=2.0)
    p.add_argument("--prior-b-P0-harm", type=float, default=1.0)

    # Initial inference values
    p.add_argument("--sigma-init", type=float, default=1.0)
    p.add_argument("--s-alpha-init", type=float, default=1e-2)
    p.add_argument("--s-beta-init", type=float, default=1e-3)
    p.add_argument("--s-gamma-init", type=float, default=1e-3)
    p.add_argument("--P0-alpha-init", type=float, default=0.25)
    p.add_argument("--P0-beta-init", type=float, default=0.05)
    p.add_argument("--P0-harm-init", type=float, default=0.25)
    p.add_argument("--m0-cos-init", type=str, default='1,1,1,1,1', help="CSV length K (dynamic season)")
    p.add_argument("--m0-sin-init", type=str, default='1,1,1,1,1', help="CSV length K (dynamic season)")
    p.add_argument("--m0-nyq-init", type=float, default=0.0)

    # Output / UX
    p.add_argument("--out-dir", type=str, default="results/simulations/DLM_harm")
    p.add_argument("--plot", type=int, default=1)
    p.add_argument("--print-summary", type=int, default=1)

    args = p.parse_args()
    np.random.seed(args.seed)

    if args.harmonics == None:
        args.harmonics = (args.period - 1) // 2

    if args.m0_cos is not None:
        args.m0_cos = [float(z) for z in args.m0_cos.split(",")]
    if args.m0_sin is not None:
        args.m0_sin = [float(z) for z in args.m0_sin.split(",")]
    if args.season_dummies is not None:
        args.season_dummies = [float(z) for z in args.season_dummies.split(",")]

    # ---------- simulate data (truth) ----------

    mts = Mean_Time_Series(
        sigma=args.sigma,
        period=args.period,
        level_mode=args.level_mode,
        trend_mode=args.trend_mode,
        seasonal_mode=args.seasonal_mode,
        season_harmonics=args.harmonics,
        season_use_nyquist=args.use_nyq,
        q_level=args.q_level,
        q_trend=args.q_trend,
        q_season=args.q_season,  # one scalar for all seasonal states, possibly generalize later
        m0_level=args.m0_level, v0_level=args.v0_level,
        m0_trend=(0.0 if args.trend_mode == "none" else args.m0_trend), v0_trend=args.v0_trend,
        m0_cos=args.m0_cos, # if given, it overrides dummies
        m0_sin=args.m0_sin, # if given, it overrides dummies
        m0_nyquist=args.m0_nyq, # if given, it overrides dummies
        season_dummies=args.season_dummies,
        center_dummies=bool(int(args.center_dummies)),
        start_date=_parse_date(args.start_date),
        rng=np.random.default_rng(args.seed),
    )

    y = np.array([mts.move() or mts.measure() for _ in range(args.T)], float)
    truth = mts.get_truth_paths(as_numpy=True)
    mu_T = truth["mu_t"][1:1 + args.T]
    dates_T = truth["index"][:args.T]

    # ---------- priors ----------
    pri_m_cos = _parse_csv_maybe(args.prior_m_m0_cos)
    pri_m_sin = _parse_csv_maybe(args.prior_m_m0_sin)

    priors = Priors(
        a_sigma=args.prior_a_sigma, b_sigma=args.prior_b_sigma,
        m_m0_alpha=args.prior_m_m0_alpha, s_m0_alpha=args.prior_s_m0_alpha,
        m_m0_beta=args.prior_m_m0_beta,  s_m0_beta=args.prior_s_m0_beta,
        m_m0_cos=None if pri_m_cos is None else pri_m_cos,
        m_m0_sin=None if pri_m_sin is None else pri_m_sin,
        m_m0_nyq=args.prior_m_m0_nyq,
        s_m0_harm=args.prior_s_m0_harm,
        a_P0_alpha=args.prior_a_P0_alpha, b_P0_alpha=args.prior_b_P0_alpha,
        a_P0_beta=args.prior_a_P0_beta,   b_P0_beta=args.prior_b_P0_beta,
        a_P0_harm=args.prior_a_P0_harm,   b_P0_harm=args.prior_b_P0_harm,
        hc_scale_alpha=args.hc_scale_alpha,
        hc_scale_beta=args.hc_scale_beta,
        hc_scale_gamma=args.hc_scale_gamma,
    )

    cfg = SamplerConfig(
        n_iter=int(args.n_iter), burn=int(args.burn), thin=int(args.thin),
        random_seed=int(args.seed), progress=bool(args.progress),
        progress_every=int(args.progress_every),
        print_dummies_every=int(args.print_dummies_every),
    )

    # initial seasonal means for dynamic season (optional)
    m0_cos_init = _parse_csv_maybe(args.m0_cos_init)
    m0_sin_init = _parse_csv_maybe(args.m0_sin_init)

    # ---------- sampler ----------
    sampler = DLMGibbsHarmonic(
        y=y, period=args.period,
        harmonics=args.harmonics, use_nyquist=args.use_nyq,
        level_mode=args.level_mode, trend_mode=args.trend_mode, seasonal_mode=args.seasonal_mode,
        sigma2_init=args.sigma_init ** 2,
        s_alpha_init=args.s_alpha_init, s_beta_init=args.s_beta_init, s_gamma_init=args.s_gamma_init,
        m0_alpha_init=args.m0_level, m0_beta_init=(0.0 if args.trend_mode == "none" else args.m0_trend),
        P0_alpha_init=args.P0_alpha_init, P0_beta_init=args.P0_beta_init, P0_harm_init=args.P0_harm_init,
        # preferred harmonics (override dummies if present)
        m0_cos_provided=args.m0_cos, m0_sin_provided=args.m0_sin,
        m0_nyq_provided=(args.m0_nyq if (args.m0_cos and args.m0_sin) else None),
        # or dummies (length s-1) to be projected
        season_dummies_short=args.season_dummies,
        # dynamic initial means
        m0_cos_init=m0_cos_init, m0_sin_init=m0_sin_init, m0_nyq_init=args.m0_nyq_init,
        priors=priors, cfg=cfg,
    )

    # optional truth overlays
    sampler.set_truth_paths(mu=mu_T)

    if args.print_summary:
        print(f"\nSimulated {args.T} observations (σ={mts.sigma:.3g}) with modes "
              f"{args.level_mode}/{args.trend_mode}/{args.seasonal_mode}.")
        print(f"Harmonics: K={sampler.K}, Nyquist={sampler.use_nyq}\n"
              f"HC scales: Aα={priors.hc_scale_alpha}, Aβ={priors.hc_scale_beta}, Aγ={priors.hc_scale_gamma}")

    # ---------- run ----------
    t0 = time.time()
    post = sampler.run()
    elapsed = time.time() - t0
    print(f"\n[Run completed in {elapsed:.1f}s]")

    # ---------- save ----------
    out_dir = os.path.join(
        args.out_dir,
        f"{args.level_mode}-{args.trend_mode}-{args.seasonal_mode}_K{sampler.K}_nyq{int(bool(sampler.use_nyq))}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(out_dir, exist_ok=True)
    sampler.save_posterior(
        out_npz_path=os.path.join(out_dir, "posterior.npz"),
        extra_meta={
            "elapsed_seconds": float(elapsed),
            "hc_scales": {"alpha": priors.hc_scale_alpha, "beta": priors.hc_scale_beta, "gamma": priors.hc_scale_gamma},
        },
    )

    # ---------- quick summary + plot ----------
    if args.print_summary:
        print("\n--- Posterior means ---")
        print(f"σ = {np.mean(post['sigma']):.4f}")
        for k in ["alpha", "beta", "gamma"]:
            key = f"Q_{k}"
            if key in post:
                mQ = np.mean(post[key])
                print(f"Q_{k} = {mQ:.4g} (√Q ≈ {math.sqrt(max(mQ,0.0)):.4g})")

    if args.plot:
        mu_hat = post["mu"].mean(axis=0)
        plt.figure(figsize=(10, 4))
        plt.plot(dates_T, y, label="y_t", lw=1)
        if "true_mu_t" in post or 'true_mu_t' in sampler.__dict__:
            plt.plot(dates_T, mu_T, "--", label="μ_t (truth)")
        plt.plot(dates_T, mu_hat, "-.", label="μ̂_t (post mean)")
        plt.title(f"DLM harmonic: {args.level_mode}/{args.trend_mode}/{args.seasonal_mode} | K={sampler.K}, nyq={sampler.use_nyq}")
        plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()
