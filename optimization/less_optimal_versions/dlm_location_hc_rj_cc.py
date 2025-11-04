from __future__ import annotations
"""
Carlin–Chib (CC) sampler for a Gaussian structural DLM (level / trend / season).

- Maintains parameter sets for all modes of each block: {dynamic, deterministic, none}.
- Uses Carlin–Chib pseudo-priors to Gibbs-update the block modes.
- Likelihood integrates out inactive states; active states updated by FFBS.
- Dynamic blocks use conjugate updates; process SDs have Half-Cauchy true prior via IG–IG mixture.

Main class: `DLM_CC` with `.run()` and `.save_posterior(...)`.
"""

import argparse, json, math, os, sys, time, warnings
from collections import deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

EPS = 1e-12


# =============================================================================
# Utilities
# =============================================================================

def _mad(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    if v.size == 0:
        return 0.0
    med = np.median(v)
    return float(np.median(np.abs(v - med)))


def _robust_sd(v: np.ndarray) -> float:
    return _mad(np.asarray(v, float)) / 1.4826 if v.size else 0.0


def _spd_solve(M: np.ndarray, B: np.ndarray, jitter: float = 1e-12) -> np.ndarray:
    """
    Solve M X = B with increasing diagonal jitter; fallback to pinv if needed.
    """
    n = M.shape[0]
    I = np.eye(n)
    S = 0.5 * (M + M.T)
    for k in range(4):
        try:
            L = np.linalg.cholesky(S + (10.0**k) * jitter * I)
            Y = np.linalg.solve(L, B)
            return np.linalg.solve(L.T, Y)
        except np.linalg.LinAlgError:
            pass
    return np.linalg.pinv(M) @ B


def _log_ig(x: float, shape: float, scale: float) -> float:
    """
    Log-density of Inv-Gamma(shape, scale) on a variance parameter x.
    (Up to additive constants; intended for differences.)
    """
    x = max(float(x), 1e-300)
    return -(shape + 1.0) * math.log(x) - (scale / x)


# =============================================================================
# Priors
# =============================================================================

@dataclass
class Priors:
    # Observation precision tau ~ Gamma(a_sigma, b_sigma) (shape–rate), sigma^2 = 1/tau
    a_sigma: float = 2.0
    b_sigma: float = 1.0

    # True priors for initial means (used for dynamic OR deterministic blocks)
    m_m0_alpha: float = 0.0
    s_m0_alpha: float = 10.0
    m_m0_beta: float = 0.0
    s_m0_beta: float = 10.0

    # Seasonal prior (length p-1; "newest first")
    m_m0_gamma: Optional[Sequence[float]] = None
    s_m0_gamma: float = 5.0

    # True priors for initial variances (InvGamma on variance)
    a_P0_alpha: float = 2.0
    b_P0_alpha: float = 1.0
    a_P0_beta: float = 2.0
    b_P0_beta: float = 1.0
    a_P0_gamma: float = 2.0
    b_P0_gamma: float = 1.0

    # Half-Cauchy scales for process SDs (true prior via IG–IG mixture)
    hc_scale_alpha: float = 0.5
    hc_scale_beta: float = 0.5
    hc_scale_gamma: float = 0.5


@dataclass
class PseudoPriors:
    """
    Carlin–Chib pseudo-priors (independent product form per block/mode).
    We set sensible defaults by *deriving them from the true Priors* so they are
    "similar" unless explicitly overridden.

    For dynamic modes we use:
        m0 ~ N(pm, sm^2),  P0 ~ IG(aP, bP),  Q ~ IG(aQ, bQ),  a_aux ~ IG(aA, bA)
    For deterministic modes:
        m0 ~ N(pm, sm^2)  (seasonal m0 is vector with diagonal variance sm^2)
    """
    # Means
    m_m0_alpha: Optional[float] = None
    s_m0_alpha: Optional[float] = None
    m_m0_beta: Optional[float] = None
    s_m0_beta: Optional[float] = None
    m_m0_gamma: Optional[Sequence[float]] = None
    s_m0_gamma: Optional[float] = None

    # P0 hyperparams
    a_P0_alpha: Optional[float] = None
    b_P0_alpha: Optional[float] = None
    a_P0_beta: Optional[float] = None
    b_P0_beta: Optional[float] = None
    a_P0_gamma: Optional[float] = None
    b_P0_gamma: Optional[float] = None

    # Q hyperparams (independent IG for pseudo-prior)
    a_Q_alpha: Optional[float] = None
    b_Q_alpha: Optional[float] = None
    a_Q_beta: Optional[float] = None
    b_Q_beta: Optional[float] = None
    a_Q_gamma: Optional[float] = None
    b_Q_gamma: Optional[float] = None

    # a_aux hyperparams (IG–IG mixture partner for Half-Cauchy mimicry)
    a_A_alpha: Optional[float] = None
    b_A_alpha: Optional[float] = None
    a_A_beta: Optional[float] = None
    b_A_beta: Optional[float] = None
    a_A_gamma: Optional[float] = None
    b_A_gamma: Optional[float] = None


def _derive_pseudo_from_true(period: int, pri: Priors, pp: Optional[PseudoPriors]) -> PseudoPriors:
    """
    Fill missing pseudo-prior hyperparameters from the true priors to keep them "similar".
    This makes CC proposals well-aligned with the actual prior/likelihood geometry.
    """
    pp = pp or PseudoPriors()

    # Means & scales for deterministic/dynamic m0's
    if pp.m_m0_alpha is None: pp.m_m0_alpha = pri.m_m0_alpha
    if pp.s_m0_alpha is None: pp.s_m0_alpha = max(pri.s_m0_alpha / 3.0, 1e-6)  # slightly tighter to help mixing

    if pp.m_m0_beta is None:  pp.m_m0_beta  = pri.m_m0_beta
    if pp.s_m0_beta is None:  pp.s_m0_beta  = max(pri.s_m0_beta / 3.0, 1e-6)

    if pp.m_m0_gamma is None:
        K = max(0, period - 1)
        base = np.zeros(K) if pri.m_m0_gamma is None else np.asarray(pri.m_m0_gamma, float)
        pp.m_m0_gamma = base.tolist()
    if pp.s_m0_gamma is None:
        pp.s_m0_gamma = max(pri.s_m0_gamma / 2.0, 1e-6)

    # P0 IG hyperparams: mirror true prior
    if pp.a_P0_alpha is None: pp.a_P0_alpha = pri.a_P0_alpha
    if pp.b_P0_alpha is None: pp.b_P0_alpha = pri.b_P0_alpha
    if pp.a_P0_beta  is None: pp.a_P0_beta  = pri.a_P0_beta
    if pp.b_P0_beta  is None: pp.b_P0_beta  = pri.b_P0_beta
    if pp.a_P0_gamma is None: pp.a_P0_gamma = pri.a_P0_gamma
    if pp.b_P0_gamma is None: pp.b_P0_gamma = pri.b_P0_gamma

    # Half-Cauchy(a) on SD is represented by: a_aux ~ IG(1, 1/a^2), Q ~ IG(1/2, 1/a_aux)
    # For *pseudo* we use independent IGs that roughly emulate that scale:
    #   set a_A ~ 1.0, b_A = 1/a^2   (same as true auxiliary)
    #   set a_Q ~ 0.6 (near 0.5), b_Q ~ 1.0  (moderately diffuse)
    def fill_A_Q(a_scale: float,
                 a_A: Optional[float], b_A: Optional[float],
                 a_Q: Optional[float], b_Q: Optional[float]) -> Tuple[float, float, float, float]:
        a_A_ = 1.0 if a_A is None else a_A
        b_A_ = (1.0 / (a_scale * a_scale)) if b_A is None else b_A
        a_Q_ = 0.6 if a_Q is None else a_Q  # slightly > 0.5 for stability
        b_Q_ = 1.0 if b_Q is None else b_Q
        return a_A_, b_A_, a_Q_, b_Q_

    (pp.a_A_alpha, pp.b_A_alpha, pp.a_Q_alpha, pp.b_Q_alpha) = fill_A_Q(
        pri.hc_scale_alpha, pp.a_A_alpha, pp.b_A_alpha, pp.a_Q_alpha, pp.b_Q_alpha
    )
    (pp.a_A_beta,  pp.b_A_beta,  pp.a_Q_beta,  pp.b_Q_beta)  = fill_A_Q(
        pri.hc_scale_beta,  pp.a_A_beta,  pp.b_A_beta,  pp.a_Q_beta,  pp.b_Q_beta
    )
    (pp.a_A_gamma, pp.b_A_gamma, pp.a_Q_gamma, pp.b_Q_gamma) = fill_A_Q(
        pri.hc_scale_gamma, pp.a_A_gamma, pp.b_A_gamma, pp.a_Q_gamma, pp.b_Q_gamma
    )
    return pp


# =============================================================================
# Sampler Config
# =============================================================================

@dataclass
class SamplerConfig:
    n_iter: int = 20_000
    burn: int = 5_000
    thin: int = 2
    random_seed: Optional[int] = 42
    progress: bool = True
    progress_every: int = 0  # auto if 0 (~2%)

    allow_none_level: bool = False
    allow_none_trend: bool = True
    allow_none_season: bool = True

    acc_window: int = 500  # window for CC acceptance summaries


# =============================================================================
# State-space layout (for current modes)
# =============================================================================

class _Layout:
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

        labels: List[str] = []
        self.idx_alpha: Optional[int] = None
        self.idx_beta: Optional[int] = None
        self.idx_g_start: Optional[int] = None
        self.idx_g_end: Optional[int] = None

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
                A[gs + 1:ge + 1, gs:ge] = np.eye(K - 1)
                A[gs + 1:ge + 1, ge] = 0.0
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
# DLM with Carlin–Chib
# =============================================================================

class DLM_CC:
    def __init__(
        self,
        y: np.ndarray,
        period: int,
        level_mode: str = "dynamic",
        trend_mode: str = "dynamic",
        seasonal_mode: str = "dynamic",
        sigma2_init: float = 1.0,
        # active block initials
        m0_alpha_init: float = 0.0, P0_alpha_init: float = 1.0,
        m0_beta_init: float = 0.0,  P0_beta_init: float = 1.0,
        m0_gamma_init: Optional[Sequence[float]] = None, P0_gamma_init: float = 1.0,
        s_alpha_init: float = 1e-2, s_beta_init: float = 1e-3, s_gamma_init: float = 1e-3,
        priors: Priors = Priors(),
        pseudo: Optional[PseudoPriors] = None,  # may be None; we derive it "similar" to priors
        cfg: SamplerConfig = SamplerConfig(),
        model_prior: Optional[Dict[str, Dict[str, float]]] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.y = np.asarray(y, float)
        self.T = int(self.y.size)
        self.period = int(period)
        if self.period < 2:
            raise ValueError("period must be >= 2")

        self.priors = priors
        self.pseudo = _derive_pseudo_from_true(self.period, self.priors, pseudo)
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.random_seed)

        self.model_prior = model_prior or {
            "level": {"dynamic": 0.5, "deterministic": 0.5, "none": 1e-12},
            "trend": {"dynamic": 0.5, "deterministic": 0.5, "none": 0.4 if cfg.allow_none_trend else 1e-12},
            "season": {"dynamic": 0.5, "deterministic": 0.5, "none": 0.2 if cfg.allow_none_season else 1e-12},
        }

        # params[block][mode] -> dict of parameters
        self.params: Dict[str, Dict[str, dict]] = {
            "level":  {"dynamic": {}, "deterministic": {}, "none": {}},
            "trend":  {"dynamic": {}, "deterministic": {}, "none": {}},
            "season": {"dynamic": {}, "deterministic": {}, "none": {}},
        }

        self.level_mode = level_mode
        self.trend_mode = trend_mode
        self.seasonal_mode = seasonal_mode

        self.sigma2 = float(sigma2_init)

        self._init_all_params(
            m0_alpha_init, P0_alpha_init, m0_beta_init, P0_beta_init,
            m0_gamma_init, P0_gamma_init, s_alpha_init, s_beta_init, s_gamma_init
        )

        self._layout = _Layout(self.period, self.level_mode, self.trend_mode, self.seasonal_mode)
        self.x = np.zeros((self.T + 1, self._layout.dim))

        self.keep: Dict[str, np.ndarray] = {}
        self._mode_counts = {
            "level": {"dynamic": 0, "deterministic": 0, "none": 0},
            "trend": {"dynamic": 0, "deterministic": 0, "none": 0},
            "season": {"dynamic": 0, "deterministic": 0, "none": 0},
        }
        self.acc_hist = {
            "level": deque(maxlen=cfg.acc_window),
            "trend": deque(maxlen=cfg.acc_window),
            "season": deque(maxlen=cfg.acc_window),
        }

        if self.cfg.progress:
            sd1 = _robust_sd(np.diff(self.y)) if self.T >= 2 else 0.0
            sd2 = _robust_sd(np.diff(self.y, n=2)) if self.T >= 3 else 0.0
            print(f"[init] L/T/S={self.level_mode[:3]}/{self.trend_mode[:3]}/{self.seasonal_mode[:3]} | sd1={sd1:.4g} sd2={sd2:.4g}")

    # ------------------------------ Formatting ------------------------------ #

    def _fmt_list(self, v: Sequence[float], max_len: int = 6) -> str:
        arr = np.asarray(v, float).ravel().tolist()
        if len(arr) <= max_len:
            body = ", ".join(f"{x:.3g}" for x in arr)
        else:
            head = ", ".join(f"{x:.3g}" for x in arr[:max_len-1])
            body = f"{head}, …, {arr[-1]:.3g}"
        return f"[{body}]"

    def _fmt_rj(self, block: str) -> str:
        hist = self.acc_hist.get(block, [])
        if not hist:
            return "0.0%"
        p = 100.0 * (sum(hist) / max(1, len(hist)))
        return f"{p:.1f}%"

    def _fmt_rj_all(self) -> str:
        return f"{self._fmt_rj('level')}|{self._fmt_rj('trend')}|{self._fmt_rj('season')}"

    def _progress_line(self, it: int) -> str:
        parts = [
            f"[it {it + 1}/{self.cfg.n_iter}]",
            f"σ={math.sqrt(self.sigma2):.3f}",
            f"L/T/S={self.level_mode[:3]}/{self.trend_mode[:3]}/{self.seasonal_mode[:3]}",
        ]
        # Q's
        if self._layout.idx_alpha is not None:
            parts.append(f"Qα={float(self.params['level']['dynamic']['Q']):.4g}")
        if self._layout.idx_beta is not None:
            parts.append(f"Qβ={float(self.params['trend']['dynamic']['Q']):.4g}")
        if self.seasonal_mode == "dynamic":
            parts.append(f"Qγ={float(self.params['season']['dynamic']['Q']):.4g}")
        # m0/P0
        if self.level_mode == "dynamic":
            parts.append(f"m0α={float(self.params['level']['dynamic']['m0']):.4g} P0α={float(self.params['level']['dynamic']['P0']):.4g}")
        elif self.level_mode == "deterministic":
            parts.append(f"m0α(det)={float(self.params['level']['deterministic']['m0']):.4g}")
        if self.trend_mode == "dynamic":
            parts.append(f"m0β={float(self.params['trend']['dynamic']['m0']):.4g} P0β={float(self.params['trend']['dynamic']['P0']):.4g}")
        elif self.trend_mode == "deterministic":
            parts.append(f"m0β(det)={float(self.params['trend']['deterministic']['m0']):.4g}")
        if self.seasonal_mode == "dynamic":
            parts.append(f"m0γ={self._fmt_list(self.params['season']['dynamic']['m0'])} P0γ={float(self.params['season']['dynamic']['P0']):.4g}")
        elif self.seasonal_mode == "deterministic":
            parts.append(f"m0γ(det)={self._fmt_list(self.params['season']['deterministic']['m0'])}")
        parts.append(f"RJ {self._fmt_rj_all()}")
        return " | ".join(parts)

    # ------------------------- Initialization helpers ----------------------- #

    def _init_all_params(self, m0a, P0a, m0b, P0b, m0g, P0g, sa, sb, sg):
        # level
        self.params["level"]["dynamic"] = {"m0": float(m0a), "P0": float(P0a), "Q": float(sa**2), "a_aux": 1.0}
        self.params["level"]["deterministic"] = {"m0": self.priors.m_m0_alpha}
        self.params["level"]["none"] = {}

        # trend
        self.params["trend"]["dynamic"] = {"m0": float(m0b), "P0": float(P0b), "Q": float(sb**2), "a_aux": 1.0}
        self.params["trend"]["deterministic"] = {"m0": self.priors.m_m0_beta}
        self.params["trend"]["none"] = {}

        # season (vector length p-1 for deterministic/dynamic m0)
        K = self.period - 1
        if m0g is None:
            m0g = np.zeros(K)
        else:
            m0g = np.asarray(m0g, float)
            if m0g.size != K:
                raise ValueError("m0_gamma_init must have length p-1")
        self.params["season"]["dynamic"] = {"m0": m0g.astype(float).copy(), "P0": float(P0g), "Q": float(sg**2), "a_aux": 1.0}
        base = np.zeros(K) if self.priors.m_m0_gamma is None else np.asarray(self.priors.m_m0_gamma, float)
        self.params["season"]["deterministic"] = {"m0": base.astype(float)}
        self.params["season"]["none"] = {}

    # ----------------------- Deterministic contribution ---------------------- #

    def _mu_det_t(self, t: int, level_mode: str, trend_mode: str, season_mode: str) -> float:
        out = 0.0
        if level_mode == "deterministic":
            out += float(self.params["level"]["deterministic"]["m0"])
        if (trend_mode == "deterministic") and (level_mode != "dynamic"):
            out += float(self.params["trend"]["deterministic"]["m0"]) * t
        if season_mode == "deterministic":
            g = self.params["season"]["deterministic"]["m0"].reshape(-1)
            g_full = np.r_[g, -g.sum()]
            out += float(g_full[t % self.period])
        return out

    # ----------------------------- Log-likelihood ---------------------------- #

    def _kalman_loglik_given_modes(self, L: str, Tm: str, S: str) -> float:
        layout = _Layout(self.period, L, Tm, S)
        H = layout.H()
        A = layout.A()

        s_alpha = math.sqrt(self.params["level"]["dynamic"].get("Q", 0.0)) if layout.idx_alpha is not None else 0.0
        s_beta  = math.sqrt(self.params["trend"]["dynamic"].get("Q", 0.0)) if layout.idx_beta is not None else 0.0
        s_gamma = math.sqrt(self.params["season"]["dynamic"].get("Q", 0.0)) if S == "dynamic" else 0.0
        Q = layout.Q(s_alpha, s_beta, s_gamma)
        R = float(self.sigma2)

        if layout.dim == 0:
            e = np.array([self.y[t] - self._mu_det_t(t, L, Tm, S) for t in range(self.T)], float)
            return float(-0.5 * np.sum(np.log(2 * np.pi * R) + (e * e) / R))

        # initial mean & cov from active dynamic params
        m0_list, P0_list = [], []
        if layout.idx_alpha is not None:
            m0_list.append(self.params["level"]["dynamic"]["m0"]); P0_list.append(self.params["level"]["dynamic"]["P0"])
        if layout.idx_beta is not None:
            m0_list.append(self.params["trend"]["dynamic"]["m0"]); P0_list.append(self.params["trend"]["dynamic"]["P0"])
        if S == "dynamic":
            g0 = self.params["season"]["dynamic"]["m0"].reshape(-1)
            m0_list.extend(list(g0)); P0_list.extend([self.params["season"]["dynamic"]["P0"]] * (self.period - 1))

        m = np.asarray(m0_list, float)
        C = np.diag(np.asarray(P0_list, float)) + EPS * np.eye(layout.dim)
        u = layout.u(self.params["trend"]["deterministic"]["m0"]) if Tm == "deterministic" else layout.u(0.0)

        ll = 0.0
        for t in range(self.T):
            a = A @ m + u
            Rm = A @ C @ A.T + Q
            Rm = 0.5 * (Rm + Rm.T) + EPS * np.eye(layout.dim)
            y_det = self._mu_det_t(t, L, Tm, S)
            Syy = float(H @ Rm @ H.T + R)
            v = float(self.y[t] - y_det - H @ a)
            ll += -0.5 * (math.log(2 * math.pi) + math.log(Syy) + (v * v) / Syy)
            K = (Rm @ H.T) / Syy
            m = a + K.flatten() * v
            C = Rm - K @ (H @ Rm)
            C = 0.5 * (C + C.T) + EPS * np.eye(layout.dim)

        return float(ll)

    # ----------------------- True prior & pseudo-prior ----------------------- #

    def _log_true_prior_block(self, which: str, mode: str) -> float:
        if mode == "none":
            return 0.0

        if which == "level":
            pri_m, s_m = self.priors.m_m0_alpha, self.priors.s_m0_alpha
            aP, bP = self.priors.a_P0_alpha, self.priors.b_P0_alpha
            A = self.priors.hc_scale_alpha
        elif which == "trend":
            pri_m, s_m = self.priors.m_m0_beta, self.priors.s_m0_beta
            aP, bP = self.priors.a_P0_beta, self.priors.b_P0_beta
            A = self.priors.hc_scale_beta
        else:
            base = (np.zeros(self.period - 1) if self.priors.m_m0_gamma is None
                    else np.asarray(self.priors.m_m0_gamma, float))
            s_m = self.priors.s_m0_gamma
            aP, bP = self.priors.a_P0_gamma, self.priors.b_P0_gamma
            A = self.priors.hc_scale_gamma

        par = self.params[which][mode]

        if mode == "deterministic":
            if which == "season":
                g = par["m0"].reshape(-1)
                mu = base if 'base' in locals() else np.zeros_like(g)
                return float(-0.5 * np.sum((g - mu) ** 2) / (s_m**2))
            else:
                m0 = float(par["m0"])
                return float(-0.5 * ((m0 - pri_m) ** 2) / (s_m**2))

        # dynamic
        if which == "season":
            g0 = par["m0"].reshape(-1)
            mu = base if 'base' in locals() else np.zeros_like(g0)
            lp = -0.5 * np.sum((g0 - mu) ** 2) / (s_m**2)
        else:
            m0 = float(par["m0"])
            lp = -0.5 * ((m0 - pri_m) ** 2) / (s_m**2)

        P0, Q, aaux = float(par["P0"]), float(par["Q"]), float(par["a_aux"])
        lp += _log_ig(P0, aP, bP)
        lp += _log_ig(aaux, 1.0, 1.0 / (A * A))
        lp += _log_ig(Q, 0.5, 1.0 / max(aaux, 1e-300))
        return float(lp)

    def _log_pseudo_prior_block(self, which: str, mode: str) -> float:
        if mode == "none":
            return 0.0

        if which == "level":
            pm, sm = self.pseudo.m_m0_alpha, self.pseudo.s_m0_alpha
            aP, bP = self.pseudo.a_P0_alpha, self.pseudo.b_P0_alpha
            aQ, bQ = self.pseudo.a_Q_alpha, self.pseudo.b_Q_alpha
            aA, bA = self.pseudo.a_A_alpha, self.pseudo.b_A_alpha
        elif which == "trend":
            pm, sm = self.pseudo.m_m0_beta, self.pseudo.s_m0_beta
            aP, bP = self.pseudo.a_P0_beta, self.pseudo.b_P0_beta
            aQ, bQ = self.pseudo.a_Q_beta, self.pseudo.b_Q_beta
            aA, bA = self.pseudo.a_A_beta, self.pseudo.b_A_beta
        else:
            base = (None if self.pseudo.m_m0_gamma is None
                    else np.asarray(self.pseudo.m_m0_gamma, float))
            sm = self.pseudo.s_m0_gamma
            aP, bP = self.pseudo.a_P0_gamma, self.pseudo.b_P0_gamma
            aQ, bQ = self.pseudo.a_Q_gamma, self.pseudo.b_Q_gamma
            aA, bA = self.pseudo.a_A_gamma, self.pseudo.b_A_gamma

        par = self.params[which][mode]

        if mode == "deterministic":
            if which == "season":
                g = par["m0"].reshape(-1)
                mu = np.zeros_like(g) if base is None else base
                return float(-0.5 * np.sum((g - mu) ** 2) / (sm**2))
            else:
                m0 = float(par["m0"])
                return float(-0.5 * ((m0 - pm) ** 2) / (sm**2))

        # dynamic
        if which == "season":
            g0 = par["m0"].reshape(-1)
            mu = np.zeros_like(g0) if base is None else base
            lp = -0.5 * np.sum((g0 - mu) ** 2) / (sm**2)
        else:
            m0 = float(par["m0"])
            lp = -0.5 * ((m0 - pm) ** 2) / (sm**2)

        P0, Q, aaux = float(par["P0"]), float(par["Q"]), float(par["a_aux"])
        lp += _log_ig(P0, aP, bP)
        lp += _log_ig(Q, aQ, bQ)
        lp += _log_ig(aaux, aA, bA)
        return float(lp)

    def _sample_from_pseudo(self, which: str, mode: str) -> None:
        """
        Refresh an INACTIVE mode's parameters from its pseudo-prior.
        """
        if mode == "none":
            self.params[which][mode] = {}
            return

        if which == "level":
            pm, sm = self.pseudo.m_m0_alpha, self.pseudo.s_m0_alpha
            aP, bP = self.pseudo.a_P0_alpha, self.pseudo.b_P0_alpha
            aQ, bQ = self.pseudo.a_Q_alpha, self.pseudo.b_Q_alpha
            aA, bA = self.pseudo.a_A_alpha, self.pseudo.b_A_alpha
        elif which == "trend":
            pm, sm = self.pseudo.m_m0_beta, self.pseudo.s_m0_beta
            aP, bP = self.pseudo.a_P0_beta, self.pseudo.b_P0_beta
            aQ, bQ = self.pseudo.a_Q_beta, self.pseudo.b_Q_beta
            aA, bA = self.pseudo.a_A_beta, self.pseudo.b_A_beta
        else:
            pm = 0.0
            sm = self.pseudo.s_m0_gamma
            aP, bP = self.pseudo.a_P0_gamma, self.pseudo.b_P0_gamma
            aQ, bQ = self.pseudo.a_Q_gamma, self.pseudo.b_Q_gamma
            aA, bA = self.pseudo.a_A_gamma, self.pseudo.b_A_gamma

        if mode == "deterministic":
            if which == "season":
                K = self.period - 1
                base = np.zeros(K) if self.pseudo.m_m0_gamma is None else np.asarray(self.pseudo.m_m0_gamma, float)
                g = self.rng.normal(base, sm, size=K)
                self.params[which][mode] = {"m0": g}
            else:
                m0 = float(self.rng.normal(pm, sm))
                self.params[which][mode] = {"m0": m0}
            return

        # dynamic
        rinv = lambda a, b: 1.0 / self.rng.gamma(a, 1.0 / b)  # IG draw
        m0 = float(self.rng.normal(pm, sm))
        P0 = float(rinv(aP, bP))
        Q = float(rinv(aQ, bQ))
        aA = float(rinv(aA, bA))
        if which == "season":
            K = self.period - 1
            m0_vec = self.rng.normal(0.0, sm, size=K)
            self.params[which][mode] = {"m0": m0_vec, "P0": P0, "Q": Q, "a_aux": aA}
        else:
            self.params[which][mode] = {"m0": m0, "P0": P0, "Q": Q, "a_aux": aA}

    # --------------------------- Active block updates ------------------------ #

    def _ffbs_active(self) -> None:
        L, Tm, S = self.level_mode, self.trend_mode, self.seasonal_mode
        layout = _Layout(self.period, L, Tm, S)
        self._layout = layout
        if layout.dim == 0:
            return

        H = layout.H()
        A = layout.A()

        s_alpha = math.sqrt(self.params["level"]["dynamic"].get("Q", 0.0)) if layout.idx_alpha is not None else 0.0
        s_beta  = math.sqrt(self.params["trend"]["dynamic"].get("Q", 0.0)) if layout.idx_beta is not None else 0.0
        s_gamma = math.sqrt(self.params["season"]["dynamic"].get("Q", 0.0)) if S == "dynamic" else 0.0
        Q = layout.Q(s_alpha, s_beta, s_gamma)
        R = float(self.sigma2)

        m0_list, P0_list = [], []
        if layout.idx_alpha is not None:
            m0_list.append(self.params["level"]["dynamic"]["m0"]); P0_list.append(self.params["level"]["dynamic"]["P0"])
        if layout.idx_beta is not None:
            m0_list.append(self.params["trend"]["dynamic"]["m0"]); P0_list.append(self.params["trend"]["dynamic"]["P0"])
        if S == "dynamic":
            g0 = self.params["season"]["dynamic"]["m0"].reshape(-1)
            m0_list.extend(list(g0)); P0_list.extend([self.params["season"]["dynamic"]["P0"]] * (self.period - 1))

        m = np.zeros((self.T + 1, layout.dim))
        C = np.zeros((self.T + 1, layout.dim, layout.dim))
        m[0] = np.asarray(m0_list, float)
        C[0] = np.diag(np.asarray(P0_list, float)) + EPS * np.eye(layout.dim)
        u = layout.u(self.params["trend"]["deterministic"]["m0"]) if Tm == "deterministic" else layout.u(0.0)

        a = np.zeros_like(m)
        Rm = np.zeros_like(C)
        for t in range(1, self.T + 1):
            a[t] = A @ m[t - 1] + u
            Rm[t] = A @ C[t - 1] @ A.T + Q
            Rm[t] = 0.5 * (Rm[t] + Rm[t].T) + EPS * np.eye(layout.dim)
            y_det = self._mu_det_t(t - 1, L, Tm, S)
            Syy = float(H @ Rm[t] @ H.T + R)
            v = float(self.y[t - 1] - y_det - H @ a[t])
            K = (Rm[t] @ H.T) / Syy
            m[t] = a[t] + K.flatten() * v
            C[t] = Rm[t] - K @ (H @ Rm[t])
            C[t] = 0.5 * (C[t] + C[t].T) + EPS * np.eye(layout.dim)

        # backward sampling
        self.x = np.zeros_like(m)
        self.x[self.T] = self.rng.multivariate_normal(m[self.T], C[self.T])
        for t in range(self.T - 1, -1, -1):
            J = C[t] @ A.T
            J = _spd_solve(Rm[t + 1], J.T).T
            mean = m[t] + J @ (self.x[t + 1] - (A @ m[t] + u))
            cov = C[t] - J @ Rm[t + 1] @ J.T
            cov = 0.5 * (cov + cov.T)
            mine = float(np.linalg.eigvalsh(cov).min())
            if mine < 1e-12:
                cov += (1e-12 - mine) * np.eye(cov.shape[0])
            self.x[t] = self.rng.multivariate_normal(mean, cov)

    def _update_active_params(self) -> None:
        """
        Conjugate updates for active dynamic blocks + deterministic coefficients for active det blocks.
        """
        L, Tm, S = self.level_mode, self.trend_mode, self.seasonal_mode
        R = float(self.sigma2)
        layout = self._layout
        rinv = lambda a, b: 1.0 / self.rng.gamma(a, 1.0 / b)  # IG draw

        # alpha (level) dynamic
        if layout.idx_alpha is not None:
            ss = 0.0
            for t in range(1, self.T + 1):
                drift = (self.x[t - 1, layout.idx_beta]
                         if layout.idx_beta is not None
                         else (self.params["trend"]["deterministic"]["m0"] if Tm == "deterministic" else 0.0))
                mean = self.x[t - 1, layout.idx_alpha] + drift
                ss += (self.x[t, layout.idx_alpha] - mean) ** 2
            aaux = self.params["level"]["dynamic"]["a_aux"]
            Q = rinv(0.5 * self.T + 0.5, 0.5 * ss + 1.0 / max(aaux, 1e-300))
            self.params["level"]["dynamic"]["Q"] = float(Q)
            A = self.priors.hc_scale_alpha
            aaux = rinv(1.0, (1.0 / (A * A)) + 1.0 / max(Q, 1e-300))
            self.params["level"]["dynamic"]["a_aux"] = float(aaux)

            # m0, P0
            x0 = float(self.x[0, layout.idx_alpha])
            mpr, spr = self.priors.m_m0_alpha, self.priors.s_m0_alpha
            P0 = self.params["level"]["dynamic"]["P0"]
            prec = 1.0 / (spr**2) + 1.0 / max(P0, 1e-18)
            var = 1.0 / prec
            mean = var * (mpr / (spr**2) + x0 / max(P0, 1e-18))
            self.params["level"]["dynamic"]["m0"] = float(self.rng.normal(mean, math.sqrt(var)))
            aP, bP = self.priors.a_P0_alpha, self.priors.b_P0_alpha
            diff2 = (x0 - self.params["level"]["dynamic"]["m0"]) ** 2
            self.params["level"]["dynamic"]["P0"] = rinv(aP + 0.5, bP + 0.5 * diff2)

        # beta (trend) dynamic
        if layout.idx_beta is not None:
            d = self.x[1:, layout.idx_beta] - self.x[:-1, layout.idx_beta]
            ss = float(np.sum(d * d))
            aaux = self.params["trend"]["dynamic"]["a_aux"]
            Q = rinv(0.5 * self.T + 0.5, 0.5 * ss + 1.0 / max(aaux, 1e-300))
            self.params["trend"]["dynamic"]["Q"] = float(Q)
            A = self.priors.hc_scale_beta
            aaux = rinv(1.0, (1.0 / (A * A)) + 1.0 / max(Q, 1e-300))
            self.params["trend"]["dynamic"]["a_aux"] = float(aaux)

            x0 = float(self.x[0, layout.idx_beta])
            mpr, spr = self.priors.m_m0_beta, self.priors.s_m0_beta
            P0 = self.params["trend"]["dynamic"]["P0"]
            prec = 1.0 / (spr**2) + 1.0 / max(P0, 1e-18)
            var = 1.0 / prec
            mean = var * (mpr / (spr**2) + x0 / max(P0, 1e-18))
            self.params["trend"]["dynamic"]["m0"] = float(self.rng.normal(mean, math.sqrt(var)))
            aP, bP = self.priors.a_P0_beta, self.priors.b_P0_beta
            diff2 = (x0 - self.params["trend"]["dynamic"]["m0"]) ** 2
            self.params["trend"]["dynamic"]["P0"] = rinv(aP + 0.5, bP + 0.5 * diff2)

        # seasonal dynamic
        if S == "dynamic":
            gs = self._layout.idx_g_start
            ge = self._layout.idx_g_end
            ss = 0.0
            for t in range(1, self.T + 1):
                prev = self.x[t - 1, gs:ge + 1]
                mean_new = -float(np.sum(prev))
                ss += (self.x[t, gs] - mean_new) ** 2
            aaux = self.params["season"]["dynamic"]["a_aux"]
            Q = rinv(0.5 * self.T + 0.5, 0.5 * ss + 1.0 / max(aaux, 1e-300))
            self.params["season"]["dynamic"]["Q"] = float(Q)
            A = self.priors.hc_scale_gamma
            aaux = rinv(1.0, (1.0 / (A * A)) + 1.0 / max(Q, 1e-300))
            self.params["season"]["dynamic"]["a_aux"] = float(aaux)

            # initial vector m0_gamma
            g0 = self.params["season"]["dynamic"]["m0"].reshape(-1)
            P0 = self.params["season"]["dynamic"]["P0"]
            base = (np.zeros_like(g0) if self.priors.m_m0_gamma is None
                    else np.asarray(self.priors.m_m0_gamma, float))
            spr = self.priors.s_m0_gamma
            for k in range(self.period - 1):
                x0 = float(self.x[0, gs + k])
                prec = 1.0 / (spr**2) + 1.0 / max(P0, 1e-18)
                var = 1.0 / prec
                mean = var * (base[k] / (spr**2) + x0 / max(P0, 1e-18))
                g0[k] = float(self.rng.normal(mean, math.sqrt(var)))
            self.params["season"]["dynamic"]["m0"] = g0
            aP, bP = self.priors.a_P0_gamma, self.priors.b_P0_gamma
            diff2 = float(np.sum((self.x[0, gs:ge + 1] - g0) ** 2))
            self.params["season"]["dynamic"]["P0"] = rinv(aP + 0.5 * (self.period - 1), bP + 0.5 * diff2)

        # deterministic actives
        if self.level_mode == "deterministic":
            r = self._residual_no_det_level()
            m0, s0 = self.priors.m_m0_alpha, self.priors.s_m0_alpha
            prec = self.T / R + 1.0 / (s0 * s0)
            mean = ((r.sum() / R) + m0 / (s0 * s0)) / prec
            self.params["level"]["deterministic"]["m0"] = float(self.rng.normal(mean, math.sqrt(1.0 / prec)))

        if self.trend_mode == "deterministic":
            if self._layout.idx_alpha is not None:
                d = self.x[1:, self._layout.idx_alpha] - self.x[:-1, self._layout.idx_alpha]
                q = self.params["level"]["dynamic"]["Q"]
                m0, s0 = self.priors.m_m0_beta, self.priors.s_m0_beta
                prec = self.T / q + 1.0 / (s0 * s0)
                mean = ((float(np.sum(d)) / q) + m0 / (s0 * s0)) / prec
                self.params["trend"]["deterministic"]["m0"] = float(self.rng.normal(mean, math.sqrt(1.0 / prec)))
            else:
                tvec = np.arange(self.T, dtype=float)
                r = self._residual_no_det_trend()
                m0, s0 = self.priors.m_m0_beta, self.priors.s_m0_beta
                prec = (tvec @ tvec) / R + 1.0 / (s0 * s0)
                mean = ((tvec @ r) / R + m0 / (s0 * s0)) / prec
                self.params["trend"]["deterministic"]["m0"] = float(self.rng.normal(mean, math.sqrt(1.0 / prec)))

        if self.seasonal_mode == "deterministic":
            K = self.period - 1
            idx = np.arange(self.T) % self.period
            Z = np.zeros((self.T, K))
            for k in range(K):
                Z[:, k] = (idx == k).astype(float) - (idx == K).astype(float)
            r = self._residual_no_det_season()
            mu_prior = np.zeros(K) if self.priors.m_m0_gamma is None else np.asarray(self.priors.m_m0_gamma, float)
            s2p = float(self.priors.s_m0_gamma) ** 2
            Prec = (Z.T @ Z) / R + np.eye(K) / s2p
            b = (Z.T @ r) / R + mu_prior / s2p
            Lc = np.linalg.cholesky(Prec)
            mu = np.linalg.solve(Lc.T, np.linalg.solve(Lc, b))
            theta = mu + np.linalg.solve(Lc.T, self.rng.standard_normal(K))
            self.params["season"]["deterministic"]["m0"] = theta

    # ------------------------------ Residuals -------------------------------- #

    def _residual_no_det_level(self) -> np.ndarray:
        L, Tm, S = self.level_mode, self.trend_mode, self.seasonal_mode
        r = self.y.copy()
        if self._layout.dim > 0:
            H = self._layout.H()
            for t in range(1, self.T + 1):
                r[t - 1] -= float(H @ self.x[t])
        if (Tm == "deterministic") and (L != "dynamic"):
            r -= float(self.params["trend"]["deterministic"]["m0"]) * np.arange(self.T, dtype=float)
        if S == "deterministic":
            g = self.params["season"]["deterministic"]["m0"].reshape(-1)
            g_full = np.r_[g, -g.sum()]
            r -= g_full[np.arange(self.T) % self.period]
        return r

    def _residual_no_det_trend(self) -> np.ndarray:
        L, Tm, S = self.level_mode, self.trend_mode, self.seasonal_mode
        r = self.y.copy()
        if self._layout.dim > 0:
            H = self._layout.H()
            for t in range(1, self.T + 1):
                r[t - 1] -= float(H @ self.x[t])
        if L == "deterministic":
            r -= float(self.params["level"]["deterministic"]["m0"])
        if S == "deterministic":
            g = self.params["season"]["deterministic"]["m0"].reshape(-1)
            g_full = np.r_[g, -g.sum()]
            r -= g_full[np.arange(self.T) % self.period]
        return r

    def _residual_no_det_season(self) -> np.ndarray:
        L, Tm, S = self.level_mode, self.trend_mode, self.seasonal_mode
        r = self.y.copy()
        if self._layout.dim > 0:
            H = self._layout.H()
            for t in range(1, self.T + 1):
                r[t - 1] -= float(H @ self.x[t])
        if L == "deterministic":
            r -= float(self.params["level"]["deterministic"]["m0"])
        if (Tm == "deterministic") and (L != "dynamic"):
            r -= float(self.params["trend"]["deterministic"]["m0"]) * np.arange(self.T, dtype=float)
        return r

    # ------------------------------ sigma^2 ---------------------------------- #

    def _update_sigma2(self) -> None:
        if self._layout.dim == 0:
            mu = np.array([self._mu_det_t(t, self.level_mode, self.trend_mode, self.seasonal_mode) for t in range(self.T)])
        else:
            H = self._layout.H()
            mu = np.zeros(self.T)
            for t in range(1, self.T + 1):
                mu[t - 1] = self._mu_det_t(t - 1, self.level_mode, self.trend_mode, self.seasonal_mode) + float(H @ self.x[t])
        e = self.y - mu
        a = self.priors.a_sigma + 0.5 * self.T
        b = self.priors.b_sigma + 0.5 * float(e @ e)
        tau = self.rng.gamma(a, 1.0 / b)
        self.sigma2 = 1.0 / max(tau, 1e-300)

    # ------------------------------ CC update -------------------------------- #

    def _legal(self, L: str, Tm: str, S: str) -> bool:
        if Tm == "dynamic" and L != "dynamic":
            return False
        if (not self.cfg.allow_none_level) and L == "none":
            return False
        return True

    def _cc_update_block(self, block: str) -> None:
        # refresh inactive modes
        for m in ("dynamic", "deterministic", "none"):
            active = {"level": self.level_mode, "trend": self.trend_mode, "season": self.seasonal_mode}[block]
            if m != active:
                self._sample_from_pseudo(block, m)

        modes = ["dynamic", "deterministic", "none"]
        L, Tm, S = self.level_mode, self.trend_mode, self.seasonal_mode
        weights: List[float] = []
        cand: List[Tuple[str, str, str]] = []

        for m in modes:
            Lc, Tc, Sc = L, Tm, S
            if block == "level":   Lc = m
            elif block == "trend": Tc = m
            else:                  Sc = m
            if not self._legal(Lc, Tc, Sc):
                continue

            ll = self._kalman_loglik_given_modes(Lc, Tc, Sc)
            lp = self._log_true_prior_block(block, m)
            lpp = self._log_pseudo_prior_block(block, m)
            mp = math.log(self.model_prior[block][m] + 1e-300)
            weights.append(float(ll + lp + mp - lpp))
            cand.append((Lc, Tc, Sc))

        if len(cand) == 1:
            newL, newT, newS = cand[0]
        else:
            w = np.asarray(weights, float)
            w -= np.max(w)
            p = np.exp(w)
            p_sum = p.sum()
            if not np.isfinite(p_sum) or p_sum <= 0.0:
                p = np.ones_like(p) / len(p)
            else:
                p /= p_sum
            idx = int(np.searchsorted(np.cumsum(p), self.rng.uniform()))
            newL, newT, newS = cand[idx]

        changed = (newL != L) or (newT != Tm) or (newS != S)
        if changed:
            self.level_mode, self.trend_mode, self.seasonal_mode = newL, newT, newS
            self.acc_hist[block].append(1)
        else:
            self.acc_hist[block].append(0)

    # -------------------------------- run() ---------------------------------- #

    def run(self) -> Dict[str, np.ndarray]:
        save_iters = list(range(self.cfg.burn, self.cfg.n_iter, self.cfg.thin))
        n_kept, keep_idx = len(save_iters), 0
        self.keep = {
            "sigma": np.zeros(n_kept),
            "mu": np.zeros((n_kept, self.T)),
            "modes": np.zeros((n_kept, 3), int),  # 0:dyn, 1:det, 2:none
        }
        print_every = self.cfg.progress_every if self.cfg.progress_every > 0 else max(1, self.cfg.n_iter // 50) or 1

        for it in range(self.cfg.n_iter):
            # 1) FFBS + active parameter updates
            self._ffbs_active()
            self._update_active_params()

            # 2) observation variance
            self._update_sigma2()

            # 3) CC block updates
            self._cc_update_block("level")
            self._cc_update_block("trend")
            self._cc_update_block("season")

            # 4) tally
            self._mode_counts["level"][self.level_mode] += 1
            self._mode_counts["trend"][self.trend_mode] += 1
            self._mode_counts["season"][self.seasonal_mode] += 1

            # 5) progress
            if self.cfg.progress and ((it + 1) % print_every == 0 or it == self.cfg.n_iter - 1):
                print(self._progress_line(it))

            # 6) save
            if it in save_iters:
                if self._layout.dim == 0:
                    mu = np.array([self._mu_det_t(t, self.level_mode, self.trend_mode, self.seasonal_mode) for t in range(self.T)])
                else:
                    H = self._layout.H()
                    mu = np.zeros(self.T)
                    for t in range(1, self.T + 1):
                        mu[t - 1] = self._mu_det_t(t - 1, self.level_mode, self.trend_mode, self.seasonal_mode) + float(H @ self.x[t])
                self.keep["mu"][keep_idx, :] = mu
                self.keep["sigma"][keep_idx] = math.sqrt(self.sigma2)
                enc = lambda s: 0 if s == "dynamic" else (1 if s == "deterministic" else 2)
                self.keep["modes"][keep_idx, :] = np.array(
                    [enc(self.level_mode), enc(self.trend_mode), enc(self.seasonal_mode)], int
                )
                keep_idx += 1

        return self.keep

    # --------------------------------- I/O ----------------------------------- #

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
            "pseudo": asdict(self.pseudo),
            "model_prior": self.model_prior,
            "visit_freq": {k: {m: int(c) for m, c in v.items()} for k, v in self._mode_counts.items()},
        }
        if extra_meta:
            meta.update(extra_meta)
        with open(out_npz_path.replace(".npz", ".meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

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
            "Gaussian structural TS (level/trend/season) with Carlin–Chib (pseudo-priors) + conjugate Gibbs.\n"
            "If available, data are simulated via simulator.mean_time_series.Mean_Time_Series.\n"
            "When a block is non-dynamic, its m0_* acts as the deterministic parameter."
        )
    )
    # ---------------------------------------------------------------------
    # Simulation (data-generating truth)
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # Priors
    # ---------------------------------------------------------------------
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

    # Half-Cauchy scales (for process SDs via IG–IG)
    p.add_argument("--hc-scale-alpha", type=float, default=0.5)
    p.add_argument("--hc-scale-beta", type=float, default=0.5)
    p.add_argument("--hc-scale-gamma", type=float, default=0.5)

    # ---------------------------------------------------------------------
    # Model priors (Occam tilt for CC selection; same interface as RJ)
    # ---------------------------------------------------------------------
    p.add_argument("--prior-model-level", type=str, default=None)
    p.add_argument("--prior-model-trend", type=str, default=None)
    p.add_argument("--prior-model-season", type=str, default=None)

    # ---------------------------------------------------------------------
    # Sampler configuration
    # ---------------------------------------------------------------------
    p.add_argument("--n-iter", type=int, default=30000)
    p.add_argument("--burn", type=int, default=10000)
    p.add_argument("--thin", type=int, default=2)
    p.add_argument("--seed", type=int, default=40)
    p.add_argument("--progress", default=True)
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--out-dir", type=str, default="results/simulations/DLM")
    p.add_argument("--plot", default=True)
    p.add_argument("--print-summary", default=True)

    # ---------------------------------------------------------------------
    # Initial inference values
    # ---------------------------------------------------------------------
    p.add_argument("--sigma-init", type=float, default=1.0)
    p.add_argument("--s-alpha-init", type=float, default=1.0)
    p.add_argument("--s-beta-init", type=float, default=1.0)
    p.add_argument("--s-gamma-init", type=float, default=1.0)
    p.add_argument("--m0-gamma-init", type=str, default=None)
    p.add_argument("--P0-alpha-init", type=float, default=0.05)
    p.add_argument("--P0-beta-init", type=float, default=0.05)
    p.add_argument("--P0-gamma-init", type=float, default=0.05)

    # ---------------------------------------------------------------------
    # CC options (legality of 'none' per block)
    # ---------------------------------------------------------------------
    p.add_argument("--allow-none-level", default=False)
    p.add_argument("--allow-none-trend", default=False)
    p.add_argument("--allow-none-season", default=False)

    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    # ---------------------------------------------------------------------
    # Simulate data via your simulator (fallbacks mirror your RJ runner)
    # ---------------------------------------------------------------------
    if Mean_Time_Series is not None:
        start_date = _parse_date(args.start_date)
        m0_season = _csv_floats_or_none(args.m0_season) or [1.0] * (args.period - 1)
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
        mu_T = truths["mu_t"][1:1 + args.T]
        dates_T = truths["index"][:args.T]
        truth_modes = f"{args.level_mode}/{args.trend_mode}/{args.seasonal_mode}"
    else:
        t = np.arange(args.T)
        seas = np.sin(2 * np.pi * t / max(2, args.period))
        y = 0.1 * t + 2 * seas + rng.normal(0, args.sigma, size=args.T)
        mu_T = 0.1 * t + 2 * seas
        dates_T = np.arange(args.T)
        truth_modes = f"{args.level_mode}/{args.trend_mode}/{args.seasonal_mode}"

    # ---------------------------------------------------------------------
    # Build priors (true) and auto-derived pseudo-priors (similar to true)
    # ---------------------------------------------------------------------
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
    # Pass pseudo=None to let the class derive pseudo-priors similar to priors
    pseudo = None

    # ---------------------------------------------------------------------
    # Model prior (Occam tilt, same shape as your RJ default)
    # ---------------------------------------------------------------------
    default_model_prior = {
        "level":  {"dynamic": 0.1, "deterministic": 0.9, "none": 1e-12},
        "trend":  {"dynamic": 0.1, "deterministic": 0.9, "none": (0.4 if args.allow_none_trend else 1e-12)},
        "season": {"dynamic": 0.1, "deterministic": 0.9, "none": (0.2 if args.allow_none_season else 1e-12)},
    }
    model_prior = {
        "level":  _csv_model_prior_block(args.prior_model_level,  args.allow_none_level,  default_model_prior["level"]),
        "trend":  _csv_model_prior_block(args.prior_model_trend,  args.allow_none_trend,  default_model_prior["trend"]),
        "season": _csv_model_prior_block(args.prior_model_season, args.allow_none_season, default_model_prior["season"]),
    }

    # ---------------------------------------------------------------------
    # Config
    # ---------------------------------------------------------------------
    cfg = SamplerConfig(
        n_iter=int(args.n_iter),
        burn=int(args.burn),
        thin=int(args.thin),
        random_seed=int(args.seed),
        progress=bool(args.progress),
        progress_every=int(args.progress_every),
        allow_none_level=bool(args.allow_none_level),
        allow_none_trend=bool(args.allow_none_trend),
        allow_none_season=bool(args.allow_none_season),
    )

    m0_gamma_init = _csv_floats_or_none(args.m0_gamma_init)

    # ---------------------------------------------------------------------
    # Sampler (start in det/det/det like your previous CC runner)
    # ---------------------------------------------------------------------
    sampler = DLM_CC(
        y=y,
        period=args.period,
        level_mode="deterministic",
        trend_mode="deterministic",
        seasonal_mode="deterministic",
        sigma2_init=args.sigma_init,
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
        pseudo=pseudo,         # derive pseudo-priors ~ similar to true priors
        cfg=cfg,
        model_prior=model_prior,
        rng=rng,
    )

    if args.print_summary:
        if Mean_Time_Series is not None:
            print(
                f"\nSimulated {args.T} observations (σ={args.sigma}) with TRUE modes "
                f"{truth_modes}."
            )
        else:
            print(f"\nSimulated fallback synthetic series of length {args.T}.")
        print("Sampler START modes: deterministic/deterministic/deterministic (CC updates by Gibbs)\n")
        print("Half-Cauchy scales (A_k) for process SDs:")
        print(
            f"  A_alpha={priors.hc_scale_alpha:.3f}, A_beta={priors.hc_scale_beta:.3f}, "
            f"A_gamma={priors.hc_scale_gamma:.3f}"
        )
        print("\nModel priors (normalized):")
        for k in ("level", "trend", "season"):
            mp = model_prior[k]
            print(f"  {k:6s}: dyn={mp['dynamic']:.3f}, det={mp['deterministic']:.3f}, none={mp['none']:.3f}")
        print("\nConvention: when a block is non-dynamic, its m0_* acts as the deterministic parameter.")

    # ---------------------------------------------------------------------
    # Run
    # ---------------------------------------------------------------------
    t0 = time.time()
    post = sampler.run()
    elapsed = time.time() - t0
    print(f"\n[Run completed in {elapsed:.1f}s]")

    # ---------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------
    out_dir = os.path.join(
        args.out_dir,
        f"CC_{args.level_mode}-{args.trend_mode}-{args.seasonal_mode}__{time.strftime('%Y%m%d_%H%M%S')}",
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
            "start_modes": {"level": "deterministic", "trend": "deterministic", "season": "deterministic"},
            "hc_scales": {
                "alpha": priors.hc_scale_alpha,
                "beta":  priors.hc_scale_beta,
                "gamma": priors.hc_scale_gamma,
            },
            "model_prior": model_prior,
        },
    )

    # Dump simple CC stats (visit freq + recent acceptance)
    visit = {
        blk: {m: int(c) for m, c in sampler._mode_counts[blk].items()}
        for blk in ("level", "trend", "season")
    }
    acc = {
        "level": float(sum(sampler.acc_hist["level"]) / max(1, len(sampler.acc_hist["level"]))),
        "trend": float(sum(sampler.acc_hist["trend"]) / max(1, len(sampler.acc_hist["trend"]))),
        "season": float(sum(sampler.acc_hist["season"]) / max(1, len(sampler.acc_hist["season"]))),
    }
    cc_dump = {"visit_counts": visit, "accept_recent": acc}
    with open(os.path.join(out_dir, "cc_stats.json"), "w", encoding="utf-8") as f:
        json.dump(cc_dump, f, indent=2)

    # ---------------------------------------------------------------------
    # Optional plot
    # ---------------------------------------------------------------------
    if args.plot:
        mu_hat = post["mu"].mean(axis=0)
        plt.figure(figsize=(10, 4))
        plt.plot(dates_T, y, label="y_t", lw=1)
        plt.plot(dates_T, mu_T, "--", label="μ_t (truth)")
        plt.plot(dates_T, mu_hat, "-.", label="μ̂_t (post mean)")
        plt.title(
            "DLM CC (Gibbs via pseudo-priors) — current modes: "
            f"{sampler.level_mode}/{sampler.trend_mode}/{sampler.seasonal_mode}"
        )
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fit.png"), dpi=160)
        plt.show()

    print(f"[save] Outputs written to: {out_dir}")


