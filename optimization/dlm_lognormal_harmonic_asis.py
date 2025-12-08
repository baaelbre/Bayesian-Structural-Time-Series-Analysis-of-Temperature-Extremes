from __future__ import annotations

"""
Gaussian structural time–series model with harmonic seasonality (cos/sin pairs + optional Nyquist)
and log-normal priors on process standard deviations, fitted using FFBS within Gibbs.

Process SDs are updated using an ASIS scheme:
  • Centred parameterisation (CP): log-normal priors on SDs, slice sampling on ln s | x.
  • Non-centred parameterisation (NCP): log-normal priors on SDs, slice sampling on ln s | (x0, eta).
  • Each iteration performs BOTH CP and NCP updates, with deterministic transforms in between.
"""

import json, math, os, sys, time, warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_dir)

from optimization.harmonic_helpers import (
    center_and_report_dummies_full,
    dummies_full_to_harmonics_fft,
    harmonics_to_dummies_full_fft,
)

# =============================================================================
# Small utils
# =============================================================================

def _spd_solve(M: np.ndarray, B: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    n = M.shape[0]
    I = np.eye(n)
    for k in range(3):
        try:
            L = np.linalg.cholesky(M + (eps * (10 ** k)) * I)
            Y = np.linalg.solve(L, B)
            return np.linalg.solve(L.T, Y)
        except np.linalg.LinAlgError:
            continue
    return np.linalg.pinv(M) @ B


# =============================================================================
# Slice sampler (univariate)
# =============================================================================

def _slice_sample(
    logpdf, z0: float, rng: np.random.Generator,
    w: float = 1.0, m: int = 10,
    max_shrink: int = 1000
) -> float:
    """
    Neal (2003) style stepping-out + shrinkage slice sampler for 1D.

    logpdf: callable(z)->log p(z) up to a constant. Must be finite near z0.
    w: initial bracket width; m: max stepping-out steps per side (<=0 means infinite).
    """
    z0 = float(z0)
    logy = float(logpdf(z0)) - rng.exponential(1.0)  # log of uniform slice

    # Step out
    u = rng.uniform(0.0, 1.0)
    L = z0 - u * w
    R = L + w
    J = int(rng.integers(0, m + 1)) if m > 0 else 0
    K = (m - 1 - J) if m > 0 else 0

    while (J > 0) and (logpdf(L) > logy):
        L -= w
        J -= 1
    while (K > 0) and (logpdf(R) > logy):
        R += w
        K -= 1

    # Shrinkage
    it = 0
    while it < max_shrink:
        z1 = rng.uniform(L, R)
        if logpdf(z1) >= logy:
            return float(z1)
        if z1 < z0:
            L = z1
        else:
            R = z1
        it += 1

    # Failsafe
    return float(z0)


# =============================================================================
# Priors & Config
# =============================================================================

@dataclass
class Priors:
    # σ² ~ IG(a_sigma, b_sigma) via Gamma on precision
    a_sigma: float = 2.0
    b_sigma: float = 1.0

    # m0 priors (Normal)
    m_m0_alpha: float = 0.0
    s_m0_alpha: float = 10.0
    m_m0_beta: float = 0.0
    s_m0_beta: float = 10.0
    m_m0_nyq: float = 0.0
    # common sd for harmonic m0 priors (cos/sin)
    s_m0_harm: float = 5.0
    # optional means for harmonics (if desired)
    m_m0_cos: Optional[Sequence[float]] = None
    m_m0_sin: Optional[Sequence[float]] = None

    # P0 priors (Inv-Gamma)
    a_P0_alpha: float = 2.0
    b_P0_alpha: float = 1.0
    a_P0_beta: float = 2.0
    b_P0_beta: float = 1.0
    a_P0_harm: float = 2.0
    b_P0_harm: float = 1.0

    # Log-normal priors on process SDs:  ln s_k ~ Normal(mu, sd^2)
    ln_s_alpha_mu: float = -5.0
    ln_s_alpha_sd: float = 1.0
    ln_s_beta_mu:  float = -7.0
    ln_s_beta_sd:  float = 1.0
    ln_s_gamma_mu: float = -6.0
    ln_s_gamma_sd: float = 1.0


@dataclass
class SamplerConfig:
    n_iter: int = 4000
    burn: int = 1000
    thin: int = 1
    random_seed: Optional[int] = 7
    progress: bool = True
    progress_every: int = 0  # 0 => ~2% of n_iter
    # print reconstructed (sized s) dummies each N iterations (0=off)
    print_dummies_every: int = 0

    # Slice parameters for log SD updates
    slice_w: float = 1.0  # initial bracket width in z-space
    slice_m: int = 10     # max stepping-out steps per side


# =============================================================================
# DLM Sampler – harmonic seasonality (cos/sin pairs + optional Nyquist)
# =============================================================================

class DLMGibbsHarmonic:
    """
    Gaussian structural DLM with harmonic seasonal states

    State (dynamic):
      [alpha] [beta] [c1 s1 | c2 s2 | ... | cK sK | nyq?]

    Observation:
      y_t = (deterministic: level/trend/season pieces) + H x_t + ε_t,
      ε_t ~ N(0, σ²).

    Process disturbances (NCP):
      x_t = A x_{t-1} + u + S eta_t,    eta_t ~ N(0, I_dim),
      S = diag(s_alpha, s_beta, s_gamma, ...).

    • Latent block we keep is (x0, eta_{1:T}).
    • Process SDs s_k have log-normal priors: ln s_k ~ N(μ_k, σ_k²).
    • ASIS: in each iteration we do
        1. FFBS draw of x in centred parametrisation (CP).
        2. CP slice update of ln s | x.
        3. Deterministic CP→NCP transform to eta.
        4. NCP slice update of ln s | (x0, eta).
        5. Rebuild x from (x0, eta, s).
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
        # Initial means/vars (used as priors if dynamic; as fixed params if deterministic)
        m0_alpha_init: float = 0.0,
        P0_alpha_init: float = 1.0,
        m0_beta_init: float = 0.0,
        P0_beta_init: float = 1.0,
        m0_cos_init: Optional[Sequence[float]] = None,
        m0_sin_init: Optional[Sequence[float]] = None,
        m0_nyq_init: float = 0.0,
        P0_harm_init: float = 1.0,

        # observation variance + process SD inits
        sigma2_init: float = 1.0,
        s_alpha_init: float = 1e-2,
        s_beta_init: float = 1e-3,
        s_gamma_init: float = 1e-3,

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

        self.priors, self.cfg = priors, cfg
        if cfg.random_seed is not None:
            np.random.seed(cfg.random_seed)
            self._rng = np.random.default_rng(cfg.random_seed)
        else:
            self._rng = np.random.default_rng()

        # rotation caches for seasonal dynamics
        self._omegas = 2.0 * np.pi * (np.arange(1, self.K + 1, dtype=float)) / float(self.s)
        self._cosw = np.cos(self._omegas)
        self._sinw = np.sin(self._omegas)

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

        def _idx_pair(k: int) -> int:
            pos = 0
            if self.idx_alpha is not None:
                pos += 1
            if self.idx_beta is not None:
                pos += 1
            pos += 2 * (k - 1)
            return pos

        self._idx_pair = _idx_pair

        if self.seasonal_mode == "dynamic":
            self.idx_first_season = (_idx_pair(1) if self.K > 0 else None)
            self.idx_nyq = (None if not self.use_nyq else
                            ((1 if self.idx_alpha is not None else 0) +
                             (1 if self.idx_beta  is not None else 0) +
                             2 * self.K))
            # indices of all gamma-coordinates (cos/sin pairs + optional nyq)
            gamma_idx: List[int] = []
            for k in range(1, self.K + 1):
                i = _idx_pair(k)
                gamma_idx.extend([i, i + 1])
            if self.use_nyq and (self.idx_nyq is not None):
                gamma_idx.append(self.idx_nyq)
            self._gamma_idx: List[int] = gamma_idx
        else:
            self.idx_first_season = None
            self.idx_nyq = None
            self._gamma_idx = []

        # ---------------- parameters / inits ----------------
        self.sigma2 = float(sigma2_init)
        self.s_alpha = float(s_alpha_init) if self.idx_alpha is not None else 0.0
        self.s_beta  = float(s_beta_init)  if self.idx_beta  is not None else 0.0
        self.s_gamma = float(s_gamma_init) if self.seasonal_mode == "dynamic" else 0.0

        self.m0_alpha = float(m0_alpha_init) if self.idx_alpha is not None else 0.0
        self.P0_alpha = float(P0_alpha_init) if self.idx_alpha is not None else 0.0
        self.m0_beta  = float(m0_beta_init)  if self.idx_beta  is not None else 0.0
        self.P0_beta  = float(P0_beta_init)  if self.idx_beta  is not None else 0.0

        # Seasonal (dynamic priors OR deterministic fixed coefficients)
        if self.K > 0:
            if m0_cos_init is None:
                m0_cos_init = np.zeros(self.K, float)
            if m0_sin_init is None:
                m0_sin_init = np.zeros(self.K, float)
            if len(m0_cos_init) != self.K or len(m0_sin_init) != self.K:
                raise ValueError("m0_cos_init and m0_sin_init must have length K")
            self.m0_cos = np.asarray(m0_cos_init, float)
            self.m0_sin = np.asarray(m0_sin_init, float)
        else:
            self.m0_cos = np.zeros(0, float)
            self.m0_sin = np.zeros(0, float)
        self.m0_nyq = float(m0_nyq_init) if self.use_nyq else None
        self.P0_harm = float(P0_harm_init)

        # latent path (states) — always derived from (x0, eta, s)
        self.x = np.zeros((self.T + 1, self.dim), float)
        # disturbance non-centred block: eta_t ~ N(0, I_dim)
        self.eta = np.zeros((self.T, self.dim), float) if self.dim > 0 else np.zeros((0, 0))
        # initial state x0
        self.x0 = np.zeros(self.dim, float) if self.dim > 0 else np.zeros(0, float)

        if self.dim > 0:
            m0_vec, P0_diag = self._current_m0_P0()
            self.x0 = np.random.multivariate_normal(
                m0_vec, np.diag(P0_diag) + 1e-12 * np.eye(self.dim)
            )
            # start with random eta; x from NCP mapping
            self.eta = self._rng.normal(size=(self.T, self.dim))
            S_diag_init = self._S_diag()
            self.x = self._x_from_eta(S_diag=S_diag_init)

        # storage
        self.keep: Dict[str, np.ndarray] = {}

        # Optional truth overlays (placeholders)
        self.true_sigma: Optional[float] = None
        self.true_Q: Optional[np.ndarray] = None
        self.true_mu_t: Optional[np.ndarray] = None
        self.true_alpha_t: Optional[np.ndarray] = None
        self.true_beta_t: Optional[np.ndarray] = None
        self.true_gamma_t: Optional[np.ndarray] = None

        # CP stats cache for ASIS
        self._cp_stats: Dict[str, Tuple[float, int]] = {}

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

    # -------------------- System matrices -------------------- #
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
            if self.use_nyq and self.idx_nyq is not None:
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
                co, si = float(self._cosw[k - 1]), float(self._sinw[k - 1])
                A[i,   i  ] =  co
                A[i,   i+1] =  si
                A[i+1, i  ] = -si
                A[i+1, i+1] =  co
            if self.use_nyq and (self.idx_nyq is not None):
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
        """
        Process covariance for the centred representation, used only in FFBS.
        In the NCP view, this corresponds to w_t = S eta_t, Q = diag(S^2).
        """
        if self.dim == 0:
            return np.zeros((0, 0))
        Q = np.zeros((self.dim, self.dim))
        if self.idx_alpha is not None and self.s_alpha > 0:
            Q[self.idx_alpha, self.idx_alpha] = self.s_alpha ** 2
        if self.idx_beta is not None and self.s_beta > 0:
            Q[self.idx_beta, self.idx_beta] = self.s_beta ** 2
        if self.seasonal_mode == "dynamic" and self.s_gamma > 0:
            for k in range(1, self.K + 1):
                i = self._idx_pair(k)
                Q[i, i] = self.s_gamma ** 2
                Q[i + 1, i + 1] = self.s_gamma ** 2
            if self.use_nyq and (self.idx_nyq is not None):
                Q[self.idx_nyq, self.idx_nyq] = self.s_gamma ** 2
        return Q

    # ---------------- disturbance NCP helpers ---------------- #

    def _S_diag(self) -> np.ndarray:
        """
        Diagonal of S such that w_t = S eta_t (elementwise) and Q = diag(S^2).
        """
        d = np.zeros(self.dim, float)
        if self.idx_alpha is not None:
            d[self.idx_alpha] = self.s_alpha
        if self.idx_beta is not None:
            d[self.idx_beta] = self.s_beta
        if self.seasonal_mode == "dynamic":
            if self.K > 0:
                for k in range(1, self.K + 1):
                    i = self._idx_pair(k)
                    d[i]   = self.s_gamma
                    d[i+1] = self.s_gamma
            if self.use_nyq and (self.idx_nyq is not None):
                d[self.idx_nyq] = self.s_gamma
        return d

    def _x_from_eta(self, S_diag: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reconstruct the full state path x[0:T] from (x0, eta[1:T]) and a given S_diag.
        """
        if self.dim == 0:
            return np.zeros((self.T + 1, 0), float)
        if S_diag is None:
            S_diag = self._S_diag()
        A, u = self._A(), self._u()
        x = np.zeros((self.T + 1, self.dim), float)
        x[0] = self.x0
        for t in range(1, self.T + 1):
            noise = S_diag * self.eta[t - 1]  # elementwise
            x[t] = A @ x[t - 1] + u + noise
        return x

    def _update_eta_from_x(self) -> None:
        """
        Given a state path self.x (from FFBS under Q), compute the implied disturbances
        eta_t = S^{-1}(x_t - A x_{t-1} - u) for the current SDs.

        This is the CP → NCP transform used inside ASIS.
        """
        if self.dim == 0:
            return
        A, u = self._A(), self._u()
        S_diag = self._S_diag()
        S_safe = np.where(S_diag > 0, S_diag, 1e-12)
        eta = np.zeros((self.T, self.dim), float)
        for t in range(1, self.T + 1):
            innov = self.x[t] - (A @ self.x[t - 1] + u)
            eta[t - 1] = innov / S_safe
        self.eta = eta
        self.x0 = self.x[0].copy()

    def _mu_from_x(self, x: np.ndarray) -> np.ndarray:
        """
        Compute μ_t from a given state path x[0:T].
        """
        H = self._H()
        mu = np.zeros(self.T, float)
        for t in range(1, self.T + 1):
            dyn = float(H @ x[t]) if self.dim > 0 else 0.0
            mu[t - 1] = self._mu_det(t - 1) + dyn
        return mu

    def _loglik_y_given_eta_s(self, s_alpha: float, s_beta: float, s_gamma: float) -> float:
        """
        Non-centred log-likelihood log p(y | x0, eta, s_alpha, s_beta, s_gamma, sigma2).
        Rebuilds x from (x0, eta, s) and computes the Gaussian likelihood.
        """
        if self.dim == 0:
            # purely deterministic mean
            mu_det = np.array([self._mu_det(t) for t in range(self.T)], float)
            e = self.y - mu_det
            s2 = float(self.sigma2)
            return -0.5 * (self.T * math.log(2 * math.pi * s2) + float(e @ e) / s2)

        # build S_diag for these candidate SDs
        d = np.zeros(self.dim, float)
        if self.idx_alpha is not None:
            d[self.idx_alpha] = s_alpha
        if self.idx_beta is not None:
            d[self.idx_beta] = s_beta
        if self.seasonal_mode == "dynamic":
            if self.K > 0:
                for k in range(1, self.K + 1):
                    i = self._idx_pair(k)
                    d[i]   = s_gamma
                    d[i+1] = s_gamma
            if self.use_nyq and (self.idx_nyq is not None):
                d[self.idx_nyq] = s_gamma

        x = self._x_from_eta(S_diag=d)
        mu = self._mu_from_x(x)
        e = self.y - mu
        s2 = float(self.sigma2)
        return -0.5 * (self.T * math.log(2 * math.pi * s2) + float(e @ e) / s2)

    # ---------------- deterministic mean pieces ---------------- #

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
            m0.append(self.m0_alpha)
            P0.append(self.P0_alpha)
        if self.idx_beta is not None:
            m0.append(self.m0_beta)
            P0.append(self.P0_beta)
        if self.seasonal_mode == "dynamic":
            for k in range(self.K):
                m0 += [self.m0_cos[k], self.m0_sin[k]]
                P0 += [self.P0_harm,   self.P0_harm]
            if self.use_nyq:
                m0.append(float(0.0 if self.m0_nyq is None else self.m0_nyq))
                P0.append(self.P0_harm)
        return np.asarray(m0, float), np.asarray(P0, float)

    # ------------------------- FFBS (Kalman + Carter–Kohn) ------------------------- #

    def _ffbs(self) -> np.ndarray:
        """
        Centred FFBS on x using Q(s); used for CP draw of x | y, theta.
        """
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

    # ------------------ Helpers: μ and residuals ------------------ #

    def _mu_vec(self) -> np.ndarray:
        """
        Convenience wrapper for μ_t using the *current* x path.
        """
        if self.dim == 0:
            return np.array([self._mu_det(t) for t in range(self.T)], float)
        return self._mu_from_x(self.x)

    # ------------------ σ² | rest  (Gamma on precision) ------------------ #

    def update_sigma2(self) -> None:
        e = self.y - self._mu_vec()
        a = self.priors.a_sigma + 0.5 * self.T
        b = self.priors.b_sigma + 0.5 * float(e @ e)
        tau = np.random.gamma(shape=a, scale=1.0 / b)  # precision
        self.sigma2 = 1.0 / max(tau, 1e-300)

    # =============================================================================
    # Process SDs: ASIS scheme (CP + NCP) with log-normal priors
    # =============================================================================
    # --- CP part: use innovations w_t = x_t - A x_{t-1} - u -------------------- #

    def _compute_cp_stats(self) -> None:
        """
        Precompute sufficient statistics for CP updates of s_alpha, s_beta, s_gamma:
            For each block k:
                ssq_k = sum_{t, dims} w_{t,k}^2
                n_k   = number of scalar innovations used

        Stored in self._cp_stats as:
            {"alpha": (ssq_alpha, n_alpha), "beta": (...), "gamma": (...)}
        """
        if self.dim == 0:
            self._cp_stats = {}
            return

        A, u = self._A(), self._u()
        ssq_alpha = 0.0
        ssq_beta  = 0.0
        ssq_gamma = 0.0
        n_alpha = 0
        n_beta  = 0
        n_gamma = 0

        for t in range(1, self.T + 1):
            innov = self.x[t] - (A @ self.x[t - 1] + u)
            if self.idx_alpha is not None:
                val = float(innov[self.idx_alpha])
                ssq_alpha += val * val
                n_alpha += 1
            if self.idx_beta is not None:
                val = float(innov[self.idx_beta])
                ssq_beta += val * val
                n_beta += 1
            if self.seasonal_mode == "dynamic" and self._gamma_idx:
                for j in self._gamma_idx:
                    val = float(innov[j])
                    ssq_gamma += val * val
                    n_gamma += 1

        stats: Dict[str, Tuple[float, int]] = {}
        if self.idx_alpha is not None:
            stats["alpha"] = (ssq_alpha, n_alpha)
        if self.idx_beta is not None:
            stats["beta"] = (ssq_beta, n_beta)
        if self.seasonal_mode == "dynamic" and n_gamma > 0:
            stats["gamma"] = (ssq_gamma, n_gamma)
        self._cp_stats = stats

    def _logpost_z_block_cp(self, z: float, which: str) -> float:
        """
        ln s_k = z for k in {alpha, beta, gamma}, CP version.

        Uses innovations w and log-likelihood
            log p(w | s) ∝ -n_k log s - ssq_k / (2 s^2)
        plus Normal prior on ln s.
        """
        if which not in self._cp_stats:
            return -np.inf

        z = float(z)
        s = math.exp(z)
        if s <= 0:
            return -np.inf

        ssq, n_eff = self._cp_stats[which]
        # Likelihood (dropping additive constants)
        ll = -n_eff * math.log(s) - ssq / (2.0 * s * s)

        if which == "alpha":
            mu, sd = float(self.priors.ln_s_alpha_mu), float(self.priors.ln_s_alpha_sd)
        elif which == "beta":
            mu, sd = float(self.priors.ln_s_beta_mu), float(self.priors.ln_s_beta_sd)
        elif which == "gamma":
            mu, sd = float(self.priors.ln_s_gamma_mu), float(self.priors.ln_s_gamma_sd)
        else:
            raise ValueError("which must be one of 'alpha', 'beta', 'gamma'")

        lp = -0.5 * ((z - mu) ** 2) / (sd ** 2)
        return ll + lp

    def update_process_Q_lognormal_cp(self) -> None:
        """
        Centred update of process SDs s_alpha, s_beta, s_gamma via slice sampling in z=ln s,
        using innovations w_t from the current centred state path x.

        This is the CP part of ASIS.
        """
        if self.dim == 0:
            return

        rng = self._rng
        w, m = float(self.cfg.slice_w), int(self.cfg.slice_m)

        # compute sufficient stats under current x
        self._compute_cp_stats()

        # α
        if self.idx_alpha is not None and "alpha" in self._cp_stats:
            z0 = math.log(max(self.s_alpha, 1e-16))
            logpdf = lambda z: self._logpost_z_block_cp(z, which="alpha")
            z = _slice_sample(logpdf, z0, rng, w=w, m=m)
            self.s_alpha = float(math.exp(z))

        # β
        if self.idx_beta is not None and "beta" in self._cp_stats:
            z0 = math.log(max(self.s_beta, 1e-16))
            logpdf = lambda z: self._logpost_z_block_cp(z, which="beta")
            z = _slice_sample(logpdf, z0, rng, w=w, m=m)
            self.s_beta = float(math.exp(z))

        # γ
        if self.seasonal_mode == "dynamic" and "gamma" in self._cp_stats:
            z0 = math.log(max(self.s_gamma, 1e-16))
            logpdf = lambda z: self._logpost_z_block_cp(z, which="gamma")
            z = _slice_sample(logpdf, z0, rng, w=w, m=m)
            self.s_gamma = float(math.exp(z))

    # --- NCP part: your previous disturbance-based update --------------------- #

    def _logpost_z_block(self, z: float, which: str) -> float:
        """
        ln s_k = z, k in {alpha, beta, gamma}, NCP version.
        Posterior ∝ likelihood(y | x0, eta, s) * prior(z).
        Other SDs kept fixed at current values.
        """
        z = float(z)
        if which == "alpha":
            s_alpha = math.exp(z)
            s_beta  = self.s_beta
            s_gamma = self.s_gamma
            mu, sd = float(self.priors.ln_s_alpha_mu), float(self.priors.ln_s_alpha_sd)
        elif which == "beta":
            s_alpha = self.s_alpha
            s_beta  = math.exp(z)
            s_gamma = self.s_gamma
            mu, sd = float(self.priors.ln_s_beta_mu), float(self.priors.ln_s_beta_sd)
        elif which == "gamma":
            s_alpha = self.s_alpha
            s_beta  = self.s_beta
            s_gamma = math.exp(z)
            mu, sd = float(self.priors.ln_s_gamma_mu), float(self.priors.ln_s_gamma_sd)
        else:
            raise ValueError("which must be one of 'alpha', 'beta', 'gamma'")

        ll = self._loglik_y_given_eta_s(s_alpha, s_beta, s_gamma)
        lp = -0.5 * ((z - mu) ** 2) / (sd ** 2)
        return ll + lp

    def update_process_Q_lognormal_ncp(self) -> None:
        """
        Non-centred update of process SDs s_alpha, s_beta, s_gamma via slice sampling in z=ln s.
        Uses the likelihood p(y | x0, eta, s) and normal priors on ln s.

        This is the NCP part of ASIS.
        """
        if self.dim == 0:
            return

        rng = self._rng
        w, m = float(self.cfg.slice_w), int(self.cfg.slice_m)

        # α
        if self.idx_alpha is not None:
            z0 = math.log(max(self.s_alpha, 1e-16))
            logpdf = lambda z: self._logpost_z_block(z, which="alpha")
            z = _slice_sample(logpdf, z0, rng, w=w, m=m)
            self.s_alpha = float(math.exp(z))

        # β
        if self.idx_beta is not None:
            z0 = math.log(max(self.s_beta, 1e-16))
            logpdf = lambda z: self._logpost_z_block(z, which="beta")
            z = _slice_sample(logpdf, z0, rng, w=w, m=m)
            self.s_beta = float(math.exp(z))

        # γ
        if self.seasonal_mode == "dynamic":
            z0 = math.log(max(self.s_gamma, 1e-16))
            logpdf = lambda z: self._logpost_z_block(z, which="gamma")
            z = _slice_sample(logpdf, z0, rng, w=w, m=m)
            self.s_gamma = float(math.exp(z))

        # after SD updates, rebuild x from (x0, eta, s)
        if self.dim > 0:
            S_diag = self._S_diag()
            self.x = self._x_from_eta(S_diag=S_diag)

    # --- m0 | P0, x0 (Normal); P0 | m0, x0 (Inv-Gamma) --- #

    @staticmethod
    def _gibbs_m0_scalar(x0: float, m_prior: float, s_prior: float, P0: float) -> float:
        prec = 1.0 / (s_prior ** 2) + 1.0 / max(1e-18, P0)
        var = 1.0 / prec
        mean = var * (m_prior / (s_prior ** 2) + x0 / max(1e-18, P0))
        return float(np.random.normal(mean, math.sqrt(var)))

    def update_m0(self) -> None:
        if self.dim == 0:
            return
        pos = 0
        if self.idx_alpha is not None:
            self.m0_alpha = self._gibbs_m0_scalar(
                float(self.x0[pos]), self.priors.m_m0_alpha, self.priors.s_m0_alpha, self.P0_alpha
            )
            pos += 1
        if self.idx_beta is not None:
            self.m0_beta = self._gibbs_m0_scalar(
                float(self.x0[pos]), self.priors.m_m0_beta, self.priors.s_m0_beta, self.P0_beta
            )
            pos += 1
        if self.seasonal_mode == "dynamic":
            s0 = float(self.priors.s_m0_harm)
            m_cos = (np.zeros(self.K) if self.priors.m_m0_cos is None
                     else np.asarray(self.priors.m_m0_cos, float))
            m_sin = (np.zeros(self.K) if self.priors.m_m0_sin is None
                     else np.asarray(self.priors.m_m0_sin, float))
            if m_cos.size != self.K or m_sin.size != self.K:
                raise ValueError("priors.m_m0_cos/m_m0_sin must have length K")
            for k in range(self.K):
                self.m0_cos[k] = self._gibbs_m0_scalar(
                    float(self.x0[pos + 2 * k]),
                    float(m_cos[k]), s0, self.P0_harm
                )
                self.m0_sin[k] = self._gibbs_m0_scalar(
                    float(self.x0[pos + 2 * k + 1]),
                    float(m_sin[k]), s0, self.P0_harm
                )
            if self.use_nyq:
                j = pos + 2 * self.K
                self.m0_nyq = self._gibbs_m0_scalar(
                    float(self.x0[j]), float(self.priors.m_m0_nyq), s0, self.P0_harm
                )

    def update_P0(self) -> None:
        if self.dim == 0:
            return
        pos = 0
        if self.idx_alpha is not None:
            a = self.priors.a_P0_alpha + 0.5
            b = self.priors.b_P0_alpha + 0.5 * (float(self.x0[pos]) - self.m0_alpha) ** 2
            self.P0_alpha = 1.0 / np.random.gamma(shape=a, scale=1.0 / b)
            pos += 1
        if self.idx_beta is not None:
            a = self.priors.a_P0_beta + 0.5
            b = self.priors.b_P0_beta + 0.5 * (float(self.x0[pos]) - self.m0_beta) ** 2
            self.P0_beta = 1.0 / np.random.gamma(shape=a, scale=1.0 / b)
            pos += 1
        if self.seasonal_mode == "dynamic":
            diffsq = 0.0
            Ktot = 2 * self.K + (1 if self.use_nyq else 0)
            target = [*self.m0_cos, *self.m0_sin] + (
                [float(self.m0_nyq)] if self.use_nyq else []
            )
            for k in range(Ktot):
                diffsq += (float(self.x0[pos + k]) - float(target[k])) ** 2
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
            prec = self.T / s2 + 1.0 / (s0 ** 2)
            mean = ((r.sum() / s2) + m0 / (s0 ** 2)) / prec
            var = 1.0 / prec
            self.m0_alpha = float(np.random.normal(mean, math.sqrt(var)))

        # deterministic trend
        if self.trend_mode == "deterministic":
            if self.idx_alpha is not None:
                d = self.x[1:, self.idx_alpha] - self.x[:-1, self.idx_alpha]
                s2 = float(self.s_alpha ** 2) if self.s_alpha > 0 else 1e-12
                m0, s0 = float(self.priors.m_m0_beta), float(self.priors.s_m0_beta)
                prec = (self.T / s2) + 1.0 / (s0 ** 2)
                mean = ((float(np.sum(d)) / s2) + m0 / (s0 ** 2)) / prec
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
                prec = (t @ t) / sig2 + 1.0 / (s0 ** 2)
                mean = ((t @ r) / sig2 + m0 / (s0 ** 2)) / prec
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
                    m_prior[:2 * self.K:2] = np.asarray(self.priors.m_m0_cos, float)
                    m_prior[1:2 * self.K:2] = np.asarray(self.priors.m_m0_sin, float)
            if self.use_nyq and (p > 2 * self.K):
                m_prior[-1] = float(self.priors.m_m0_nyq)
            sig2 = float(self.sigma2)
            Prec = (Z.T @ Z) / sig2 + np.eye(p) / s2
            b = (Z.T @ r) / sig2 + m_prior / s2
            mu = np.linalg.solve(Prec, b)
            L = np.linalg.cholesky(Prec)
            theta = mu + np.linalg.solve(L.T, np.random.randn(p))
            if self.K > 0:
                self.m0_cos = theta[:2 * self.K:2].copy()
                self.m0_sin = theta[1:2 * self.K:2].copy()
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
        if self.idx_alpha is not None:
            parts.append(f"Qα={self.s_alpha ** 2:.4g}")
        if self.idx_beta is not None:
            parts.append(f"Qβ={self.s_beta ** 2:.4g}")
        if self.seasonal_mode == "dynamic":
            parts.append(f"Qγ={self.s_gamma ** 2:.4g}")
        if self.level_mode != "none":
            parts.append(f"m0α={self.m0_alpha:.4g} P0α={(self.P0_alpha if self.level_mode == 'dynamic' else 0.0):.4g}")
        if self.trend_mode != "none":
            parts.append(f"m0β={self.m0_beta:.4g} P0β={(self.P0_beta if self.trend_mode == 'dynamic' else 0.0):.4g}")
        if self.seasonal_mode != "none":
            parts.append(
                f"m0cos={self._fmt_list(self.m0_cos, 6)} m0sin={self._fmt_list(self.m0_sin, 6)}"
                + (f" nyq={0.0 if (self.m0_nyq is None) else float(self.m0_nyq):.4g}"
                   if self.use_nyq else "")
            )
            if self.seasonal_mode == "dynamic":
                parts.append(f"P0harm={self.P0_harm:.4g}")
        return " | ".join(parts)

    def _maybe_print_dummies(self, it: int) -> None:
        n = int(self.cfg.print_dummies_every)
        if n <= 0:
            return
        if (it + 1) % n != 0 and it != self.cfg.n_iter - 1:
            return
        if self.seasonal_mode in ("dynamic", "deterministic"):
            d = harmonics_to_dummies_full_fft(
                s=self.s,
                cos_coefs=self.m0_cos if self.K > 0 else np.zeros(0),
                sin_coefs=self.m0_sin if self.K > 0 else np.zeros(0),
                use_nyquist=self.use_nyq,
                nyq_coef=self.m0_nyq,
            )
            print(f"seasonal dummies = {self._fmt_list(d, max_elems=self.s, fmt='.4f')}")

    # --------------------------------- MCMC --------------------------------- #

    def run(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        save_iters = list(range(cfg.burn, cfg.n_iter, cfg.thin))
        n_kept, keep_idx = len(save_iters), 0

        # allocate storage
        self.keep = {
            "sigma": np.zeros(n_kept, float),
            "mu": np.zeros((n_kept, self.T), float),
        }
        if self.idx_alpha is not None:
            self.keep.update({
                "Q_alpha": np.zeros(n_kept),
                "m0_alpha": np.zeros(n_kept),
                "P0_alpha": np.zeros(n_kept),
            })
        if self.idx_beta is not None:
            self.keep.update({
                "Q_beta": np.zeros(n_kept),
                "m0_beta": np.zeros(n_kept),
                "P0_beta": np.zeros(n_kept),
            })
        if self.seasonal_mode == "dynamic":
            self.keep.update({
                "Q_gamma": np.zeros(n_kept),
                "m0_cos": np.zeros((n_kept, self.K)),
                "m0_sin": np.zeros((n_kept, self.K)),
                "m0_nyq": np.zeros(n_kept) if self.use_nyq else np.zeros(0),
                "P0_harm": np.zeros(n_kept),
                "x": np.zeros((n_kept, self.T, self.dim)) if self.dim > 0 else np.zeros((0, 0, 0)),
            })
        else:
            self.keep.update({
                "m0_cos": np.zeros((n_kept, self.K)),
                "m0_sin": np.zeros((n_kept, self.K)),
                "m0_nyq": np.zeros(n_kept) if self.use_nyq else np.zeros(0),
            })
            if self.dim > 0:
                self.keep["x"] = np.zeros((n_kept, self.T, self.dim))

        print_every = cfg.progress_every if cfg.progress_every > 0 else max(1, cfg.n_iter // 50) or 1

        for it in range(cfg.n_iter):
            # 1) CENTRED FFBS: x | y, current SDs
            if self.dim > 0:
                self.x = self._ffbs()

                # 2) CP update of SDs via log-normal priors + innovations
                self.update_process_Q_lognormal_cp()

                # 3) CP → NCP transform: compute eta from (x, s)
                self._update_eta_from_x()

                # 4) NCP update of SDs via log-normal priors + (x0, eta)
                self.update_process_Q_lognormal_ncp()
            # After this, self.x has been rebuilt from (x0, eta, s) with updated SDs.

            # 5) m0 and 6) P0 for dynamic coords (use x0)
            if self.dim > 0:
                self.update_m0()
                self.update_P0()

            # 7) deterministic params (level/trend/season)
            self.update_deterministic_params()

            # 8) σ² (Gibbs)
            self.update_sigma2()

            # progress
            if cfg.progress and ((it + 1) % print_every == 0 or it == cfg.n_iter - 1):
                print(self._progress_line(it))

            # print reconstructed full-length dummies
            self._maybe_print_dummies(it)

            # save
            if it in save_iters:
                mu = self._mu_vec()
                self.keep["mu"][keep_idx, :] = mu
                self.keep["sigma"][keep_idx] = math.sqrt(self.sigma2)
                if self.idx_alpha is not None:
                    self.keep["Q_alpha"][keep_idx] = self.s_alpha ** 2
                    self.keep["m0_alpha"][keep_idx] = self.m0_alpha
                    self.keep["P0_alpha"][keep_idx] = self.P0_alpha
                if self.idx_beta is not None:
                    self.keep["Q_beta"][keep_idx] = self.s_beta ** 2
                    self.keep["m0_beta"][keep_idx] = self.m0_beta
                    self.keep["P0_beta"][keep_idx] = self.P0_beta
                if self.seasonal_mode == "dynamic":
                    self.keep["Q_gamma"][keep_idx] = self.s_gamma ** 2
                    self.keep["m0_cos"][keep_idx, :] = self.m0_cos
                    self.keep["m0_sin"][keep_idx, :] = self.m0_sin
                    if self.use_nyq:
                        self.keep["m0_nyq"][keep_idx] = (
                            0.0 if (self.m0_nyq is None) else float(self.m0_nyq)
                        )
                    self.keep["P0_harm"][keep_idx] = self.P0_harm
                else:
                    self.keep["m0_cos"][keep_idx, :] = self.m0_cos
                    self.keep["m0_sin"][keep_idx, :] = self.m0_sin
                    if self.use_nyq:
                        self.keep["m0_nyq"][keep_idx] = (
                            0.0 if (self.m0_nyq is None) else float(self.m0_nyq)
                        )
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

        if self.true_sigma is not None:
            arrays["true_sigma"] = float(self.true_sigma)
        if self.true_Q is not None:
            arrays["true_Q"] = np.asarray(self.true_Q, float)
        if self.true_mu_t is not None:
            arrays["true_mu_t"] = np.asarray(self.true_mu_t, float)
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

    import sys, os
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(base_dir)
    from simulator.mean_time_series_harmonic import Mean_Time_Series

    def _parse_date(s: str | None):
        if not s:
            from datetime import datetime as _dt
            return _dt.today()
        parts = [int(p) for p in s.split("-")]
        if   len(parts) == 1: return datetime(parts[0], 1, 1)
        elif len(parts) == 2: return datetime(parts[0], parts[1], 1)
        elif len(parts) == 3: return datetime(parts[0], parts[1], parts[2])
        raise ValueError("start-date must be YYYY, YYYY-MM, or YYYY-MM-DD")

    def _parse_csv_maybe(s: Optional[str]) -> Optional[List[float]]:
        if s is None: return None
        s = s.strip()
        if s == "": return None
        return [float(z) for z in s.split(",")]

    p = argparse.ArgumentParser(
        description=(
            "Gaussian DLM with harmonic seasonality (cos/sin pairs + optional Nyquist). "
            "FFBS + conjugate Gibbs for Gaussian parts; log-normal priors on process SDs "
            "with ASIS updates (CP + NCP slice sampling on log SD). "
            "Sampler does NOT accept dummies; the CLI can project full-length dummies "
            "to (cos,sin,nyq) if m0_cos_init/m0_sin_init are not supplied."
        )
    )

    # Simulation controls
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--period", type=int, default=4)
    p.add_argument("--start-date", type=str, default="2000-01-01")
    p.add_argument("--sigma", type=float, default=2.0)

    p.add_argument("--level-mode", choices=["dynamic", "deterministic"], default="dynamic")
    p.add_argument("--trend-mode", choices=["dynamic", "deterministic", "none"], default="dynamic")
    p.add_argument("--seasonal-mode", choices=["dynamic", "deterministic", "none"], default="dynamic")

    p.add_argument("--q-level", type=float, default=0.05)
    p.add_argument("--q-trend", type=float, default=0.000002)
    p.add_argument("--q-season", type=float, default=0.0001, help="One scalar seasonal process variance (simulator)")

    # Simulator priors (truth generation)
    p.add_argument("--m0-level", type=float, default=5.0)
    p.add_argument("--v0-level", type=float, default=0.25)
    p.add_argument("--m0-trend", type=float, default=0.02)
    p.add_argument("--v0-trend", type=float, default=0.05)

    # Harmonic spec for both simulator & sampler
    p.add_argument("--harmonics", type=int, default=None, help="K; None=full floor((s-1)/2)")
    p.add_argument("--use-nyquist", type=int, default=1, help="1/0; None=auto if even period & K allows Nyquist")

    # Option A (simulator): provide harmonic coefficients for the simulator only
    p.add_argument("--sim-m0-cos", type=str, default=None, help="CSV length K (simulator only)")
    p.add_argument("--sim-m0-sin", type=str, default=None, help="CSV length K (simulator only)")
    p.add_argument("--sim-m0-nyq", type=float, default=0.0)

    # Option B (simulator & CLI init): provide full-length seasonal dummies (length = period)
    p.add_argument("--season-dummies", type=str, default='1,1,1,-3', help="CSV length=period.")

    # Sampler configuration
    p.add_argument("--n-iter", type=int, default=10000)
    p.add_argument("--burn", type=int, default=5000)
    p.add_argument("--thin", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--progress", type=int, default=1)
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--print-dummies-every", type=int, default=1, help="Print reconstructed full-length dummies every N iters (0=off)")

    # Log-normal prior hyperparameters (process SDs)
    p.add_argument("--ln-s-alpha-mu", type=float, default=-1.0)
    p.add_argument("--ln-s-alpha-sd", type=float, default=2.0)
    p.add_argument("--ln-s-beta-mu",  type=float, default=-2.0)
    p.add_argument("--ln-s-beta-sd",  type=float, default=2.0)
    p.add_argument("--ln-s-gamma-mu", type=float, default=-2.0)
    p.add_argument("--ln-s-gamma-sd", type=float, default=2.0)

    # Slice controls
    p.add_argument("--slice-w", type=float, default=1.0)
    p.add_argument("--slice-m", type=int, default=10)

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
    p.add_argument("--m0-cos-init", type=str, default=None, help="CSV length K (sampler init; if None and dummies given, CLI projects)")
    p.add_argument("--m0-sin-init", type=str, default=None, help="CSV length K (sampler init; if None and dummies given, CLI projects)")
    p.add_argument("--m0-nyq-init", type=float, default=None, help="If None and dummies given (and nyq used), CLI projects")

    # Output / UX
    p.add_argument("--out-dir", type=str, default="results/simulations/DLM_harm")
    p.add_argument("--plot", type=int, default=1)
    p.add_argument("--print-summary", type=int, default=1)

    args = p.parse_args()
    np.random.seed(args.seed)

    if args.harmonics is None:
        args.harmonics = (args.period - 1) // 2
    use_nyq = bool(int(args.use_nyquist))

    # ---------- (optional) parse lists ----------
    sim_m0_cos = _parse_csv_maybe(args.sim_m0_cos)
    sim_m0_sin = _parse_csv_maybe(args.sim_m0_sin)
    pri_m_cos  = _parse_csv_maybe(args.prior_m_m0_cos)
    pri_m_sin  = _parse_csv_maybe(args.prior_m_m0_sin)
    m0_cos_init = _parse_csv_maybe(args.m0_cos_init)
    m0_sin_init = _parse_csv_maybe(args.m0_sin_init)
    season_dummies = _parse_csv_maybe(args.season_dummies)

    # ---------- simulator inputs ----------
    mts = Mean_Time_Series(
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
        m0_trend=(0.0 if args.trend_mode == "none" else args.m0_trend), v0_trend=args.v0_trend,
        # Simulator can use harmonics if provided; else it will project its own dummies logic
        m0_cos=sim_m0_cos, m0_sin=sim_m0_sin, m0_nyq=args.sim_m0_nyq,
        season_dummies=season_dummies,  # simulator still accepts dummies (full length)
        start_date=_parse_date(args.start_date),
        rng=np.random.default_rng(args.seed),
    )

    # simulate
    y = np.array([mts.move() or mts.measure() for _ in range(args.T)], float)
    truth = mts.get_truth_paths(as_numpy=True)
    mu_T = truth["mu_t"][1:1 + args.T]
    dates_T = truth["index"][:args.T]

    # ---------- priors ----------
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
        ln_s_alpha_mu=args.ln_s_alpha_mu, ln_s_alpha_sd=args.ln_s_alpha_sd,
        ln_s_beta_mu=args.ln_s_beta_mu,   ln_s_beta_sd=args.ln_s_beta_sd,
        ln_s_gamma_mu=args.ln_s_gamma_mu, ln_s_gamma_sd=args.ln_s_gamma_sd,
    )

    cfg = SamplerConfig(
        n_iter=int(args.n_iter), burn=int(args.burn), thin=int(args.thin),
        random_seed=int(args.seed), progress=bool(args.progress),
        progress_every=int(args.progress_every),
        print_dummies_every=int(args.print_dummies_every),
        slice_w=float(args.slice_w), slice_m=int(args.slice_m),
    )

    # ---------- derive sampler seasonal inits from full-length dummies if needed ----------
    if (m0_cos_init is None or m0_sin_init is None) and (season_dummies is not None):
        if len(season_dummies) != args.period:
            raise ValueError(f"--season-dummies must have length period={args.period}")
        centered = center_and_report_dummies_full(season_dummies, tol=1e-12)
        # project to (cos,sin,nyq)
        K = args.harmonics
        cos_coefs, sin_coefs, nyq_val = dummies_full_to_harmonics_fft(centered, K=K, use_nyquist=use_nyq)
        if m0_cos_init is None: m0_cos_init = list(cos_coefs)
        if m0_sin_init is None: m0_sin_init = list(sin_coefs)
        if args.m0_nyq_init is None and use_nyq:
            args.m0_nyq_init = float(0.0 if nyq_val is None else nyq_val)

    # Ensure arrays exist if still None
    if m0_cos_init is None: m0_cos_init = [0.0] * args.harmonics
    if m0_sin_init is None: m0_sin_init = [0.0] * args.harmonics
    if args.m0_nyq_init is None:
        args.m0_nyq_init = 0.0

    # ---------- sampler ----------
    sampler = DLMGibbsHarmonic(
        y=y, period=args.period,
        harmonics=args.harmonics, use_nyquist=use_nyq,
        level_mode=args.level_mode, trend_mode=args.trend_mode, seasonal_mode=args.seasonal_mode,
        sigma2_init=args.sigma_init ** 2,
        s_alpha_init=args.s_alpha_init, s_beta_init=args.s_beta_init, s_gamma_init=args.s_gamma_init,
        m0_alpha_init=args.m0_level,
        m0_beta_init=(0.0 if args.trend_mode == "none" else args.m0_trend),
        P0_alpha_init=args.P0_alpha_init,
        P0_beta_init=args.P0_beta_init,
        P0_harm_init=args.P0_harm_init,
        # seasonal initialisation (NO dummies passed into the sampler)
        m0_cos_init=m0_cos_init,
        m0_sin_init=m0_sin_init,
        m0_nyq_init=args.m0_nyq_init,
        priors=priors, cfg=cfg,
    )

    # optional truth overlays
    sampler.set_truth_paths(mu=mu_T)

    if args.print_summary:
        print(f"\nSimulated {args.T} observations (σ={mts.sigma:.3g}) with modes "
              f"{args.level_mode}/{args.trend_mode}/{args.seasonal_mode}.")
        print(f"Harmonics: K={sampler.K}, Nyquist={sampler.use_nyq}\n"
              f"LN priors (ln s): α~N({priors.ln_s_alpha_mu:.2f},{priors.ln_s_alpha_sd:.2f}²), "
              f"β~N({priors.ln_s_beta_mu:.2f},{priors.ln_s_beta_sd:.2f}²), "
              f"γ~N({priors.ln_s_gamma_mu:.2f},{priors.ln_s_gamma_sd:.2f}²)")

    # ---------- run ----------
    t0 = time.time()
    post = sampler.run()
    elapsed = time.time() - t0
    print(f"[Run completed in {elapsed:.1f}s]")

    # ---------- save ----------
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(
        args.out_dir,
        f"{args.level_mode}-{args.trend_mode}-{args.seasonal_mode}_K{sampler.K}_nyq{int(bool(sampler.use_nyq))}_{stamp}",
    )
    os.makedirs(out_dir, exist_ok=True)
    sampler.save_posterior(
        out_npz_path=os.path.join(out_dir, "posterior.npz"),
        extra_meta={
            "elapsed_seconds": float(elapsed),
            "ln_s_priors": {
                "alpha": {"mu": priors.ln_s_alpha_mu, "sd": priors.ln_s_alpha_sd},
                "beta":  {"mu": priors.ln_s_beta_mu,  "sd": priors.ln_s_beta_sd},
                "gamma": {"mu": priors.ln_s_gamma_mu, "sd": priors.ln_s_gamma_sd},
            },
        },
    )

    # ---------- quick summary + plot ----------
    if args.print_summary:
        print("--- Posterior means ---")
        print(f"σ = {np.mean(post['sigma']):.4f}")
        for k in ["alpha", "beta", "gamma"]:
            key = f"Q_{k}"
            if key in post:
                mQ = np.mean(post[key])
                print(f"Q_{k} = {mQ:.4g} (√Q ≈ {math.sqrt(max(mQ, 0.0)):.4g})")

    if args.plot:
        import matplotlib.pyplot as plt
        mu_hat = post["mu"].mean(axis=0)
        plt.figure(figsize=(10, 4))
        plt.plot(dates_T, y, label="y_t", lw=1)
        if 'true_mu_t' in sampler.__dict__ and sampler.true_mu_t is not None:
            plt.plot(dates_T, mu_T, "--", label="μ_t (truth)")
        plt.plot(dates_T, mu_hat, "-.", label="μ̂_t (post mean)")
        plt.title(
            f"DLM harmonic (ASIS): {args.level_mode}/{args.trend_mode}/{args.seasonal_mode} "
            f"| K={sampler.K}, nyq={sampler.use_nyq}"
        )
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
