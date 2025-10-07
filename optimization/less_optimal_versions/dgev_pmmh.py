# %% optimization/dgev_pmmh.py
from __future__ import annotations

import os, math, json, time
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, List, Sequence

import numpy as np
from tqdm import tqdm
from datetime import datetime

# =============================================================================
# Utilities
# =============================================================================

def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

def build_seasonal(period: int) -> np.ndarray:
    """Default smooth seasonal for first (p-1) entries (last is implied by sum-to-zero)."""
    g = np.cos(2 * np.pi * np.arange(period) / period)
    g -= np.mean(g)
    return g[: period - 1].astype(float)

def parse_csv_floats(s: Optional[str]) -> Optional[List[float]]:
    """Parse comma-separated floats like '0,0.2,-0.1' -> [0.0, 0.2, -0.1]."""
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return [float(tok) for tok in s.split(",")]

# =============================================================================
# GEV helpers
# =============================================================================

def gev_logpdf(y: float, mu: float, sigma: float, xi: float) -> float:
    """log f(y | mu, sigma>0, xi) under GEV parameterization (mu, sigma, xi)."""
    if sigma <= 0.0 or np.isnan(mu):
        return -np.inf
    z = (y - mu) / sigma
    u = 1.0 + xi * z
    if u <= 0.0:
        return -np.inf
    if abs(xi) < 1e-8:  # Gumbel limit
        return -np.log(sigma) - np.exp(-z) - z
    return -np.log(sigma) - (1.0 + 1.0 / xi) * np.log(u) - u ** (-1.0 / xi)

# =============================================================================
# Priors & Config
# =============================================================================

@dataclass
class Priors:
    # NOTE: m_sigma is the prior mean for log(sigma)
    m_sigma: float = 0.0
    s_sigma: float = 10.0
    m_xi: float = 0.0
    s_xi: float = 1.0

    # Process noises for dynamic coords: Q ~ IG(a_q, b_q) (shape, scale)
    # density: f(q) ∝ b^a / Γ(a) * q^{-(a+1)} * exp(-b/q)
    a_q_alpha: float = 1.1
    b_q_alpha: float = 1e-4
    a_q_beta:  float = 1.1
    b_q_beta:  float = 1e-12
    a_q_gamma: float = 1.5
    b_q_gamma: float = 5e-6

    # Deterministic components’ Gaussian priors
    m_level: float = 0.0
    s_level: float = 10.0
    m_slope: float = 0.0
    s_slope: float = 10.0

    # Deterministic season prior (first p-1 means; last implied)
    m_season: Optional[Sequence[float]] = None
    s_season: float = 5.0

@dataclass
class SamplerConfig:
    n_iter: int = 5000
    burn: int = 1000
    thin: int = 5

    # Random-walk proposals (std devs in the proposal space)
    step_logsigma: float = 0.15
    step_xi: float = 0.10
    step_level: float = 0.05
    step_slope: float = 0.02
    step_season: float = 0.05
    step_logQ: float = 0.35  # initial seed for per-coordinate steps

    # PF
    n_particles: int = 300
    trans_eps: float = 1e-12
    random_seed: Optional[int] = 7
    progress: bool = True
    progress_every: int = 0  # 0 => auto (~2% of n_iter)

    # Optional: store a smoothed trajectory at kept draws (costs an extra PF)
    store_states: bool = False

    # ---- Adaptive RW–MH options (Robbins–Monro; windowed) ----
    adapt_steps: bool = False
    adapt_every: int = 25
    adapt_until: str = "burn"     # "burn" or "all"
    adapt_target_1d: float = 0.44
    adapt_eta0: float = 0.05
    adapt_eta_decay: float = 0.75
    step_min: float = 1e-5
    step_max: float = 1.0

# =============================================================================
# Particle Marginal Metropolis–Hastings (PMMH) with APF (structural DGEV)
# =============================================================================

class DGEVPMMH:
    # --------------------------- Construction --------------------------- #
    def __init__(
        self,
        y: np.ndarray,
        period: int,
        level_mode: str = "dynamic",
        trend_mode: str = "dynamic",
        seasonal_mode: str = "dynamic",
        # Initial state priors for dynamic coords
        m0_level: float = 0.0, v0_level: float = 1.0,
        m0_trend: float = 0.0, v0_trend: float = 1.0,
        m0_season: Sequence[float] | None = None,
        v0_season: Sequence[float] | None = None,
        # Priors and config
        priors: Priors = Priors(),
        cfg: SamplerConfig = SamplerConfig(),
        # Deterministic initial values
        level_value_init: float = 0.0,
        slope_value_init: float = 0.0,
        seasonal_vector_init: Optional[Sequence[float]] = None,  # p-1 entries
    ):
        # Data
        self.y = np.asarray(y, float)
        self.T = int(self.y.size)
        self.period = int(period)

        # Modes
        assert level_mode in {"dynamic", "deterministic"}
        assert trend_mode in {"dynamic", "deterministic", "none"}
        assert seasonal_mode in {"dynamic", "deterministic", "none"}
        self.level_mode = level_mode
        self.trend_mode = trend_mode
        self.seasonal_mode = seasonal_mode

        # Priors & config
        self.priors = priors
        self.cfg = cfg
        if cfg.random_seed is not None:
            np.random.seed(cfg.random_seed)

        # ---- Latent state layout
        layout: List[str] = []
        if self.level_mode == "dynamic":
            layout.append("alpha")
        if self.trend_mode == "dynamic":
            layout.append("beta")
        if self.seasonal_mode == "dynamic":
            layout.extend([f"gamma_{k}" for k in range(1, self.period)])
        self._layout = layout
        self.dim = len(layout)

        if self.dim == 0 and (self.seasonal_mode != "deterministic") and (self.level_mode != "deterministic"):
            raise ValueError("At least one contribution to mu_t must exist (dynamic or deterministic).")

        # Indices
        self.idx_alpha = layout.index("alpha") if "alpha" in layout else None
        self.idx_beta  = layout.index("beta")  if "beta"  in layout else None
        if self.seasonal_mode == "dynamic":
            self.idx_gamma_start = layout.index("gamma_1") if "gamma_1" in layout else None
            self.idx_gamma_end   = self.idx_gamma_start + (self.period - 2) if self.idx_gamma_start is not None else None
        else:
            self.idx_gamma_start = None
            self.idx_gamma_end   = None

        # ---- Deterministic parameters
        self.level_value = float(level_value_init)
        self.slope_value = float(slope_value_init)

        if self.seasonal_mode == "deterministic":
            if seasonal_vector_init is not None:
                g_first = np.asarray(seasonal_vector_init, float)
                if g_first.size != self.period - 1:
                    raise ValueError("seasonal_vector_init must have length = period-1.")
            else:
                if self.priors.m_season is not None:
                    g_first = np.asarray(self.priors.m_season, float)
                    if g_first.size != self.period - 1:
                        raise ValueError("priors.m_season must have length = period-1.")
                else:
                    g_first = build_seasonal(self.period)
            g_last = -np.sum(g_first)
            self.season_vec = np.concatenate([g_first, [g_last]]).astype(float)
        else:
            self.season_vec = None

        # ---- Parameter vector θ = [logσ, ξ, det params..., logQ(active)]
        self.logsigma = float(self.priors.m_sigma)
        self.sigma    = float(np.exp(self.logsigma))
        self.xi       = float(self.priors.m_xi)

        # Active process-noise components (only those present)
        self._active_Q_idx: List[int] = []
        if self.idx_alpha is not None:
            self._active_Q_idx.append(self.idx_alpha)
        if self.idx_beta is not None:
            self._active_Q_idx.append(self.idx_beta)
        if self.seasonal_mode == "dynamic" and self.idx_gamma_end is not None:
            # Only the "last" seasonal component carries innovation
            self._active_Q_idx.append(self.idx_gamma_end)

        # initialize Q at prior means (if defined)
        self.Q = np.zeros(self.dim, float)
        if self.idx_alpha is not None:
            self.Q[self.idx_alpha] = self.priors.b_q_alpha / (self.priors.a_q_alpha - 1.0) if self.priors.a_q_alpha > 1.0 else 1.0
        if self.idx_beta is not None:
            self.Q[self.idx_beta]  = self.priors.b_q_beta  / (self.priors.a_q_beta  - 1.0) if self.priors.a_q_beta  > 1.0 else 1.0
        if self.seasonal_mode == "dynamic" and self.idx_gamma_end is not None:
            self.Q[self.idx_gamma_end] = self.priors.b_q_gamma / (self.priors.a_q_gamma - 1.0) if self.priors.a_q_gamma > 1.0 else 1.0
        # work in log-space for active Qs
        self.logQ = np.array([np.log(self.Q[k]) for k in self._active_Q_idx], float) if self._active_Q_idx else np.zeros(0)

        # ---- Initial state prior (x0)
        if self.seasonal_mode == "dynamic":
            m0_season_arr = np.zeros(self.period - 1) if m0_season is None else np.asarray(m0_season, float)
            v0_season_arr = np.ones(self.period - 1)  if v0_season is None else np.asarray(v0_season, float)
            if m0_season_arr.size != self.period - 1 or v0_season_arr.size != self.period - 1:
                raise ValueError("m0_season and v0_season must have length p-1 in dynamic mode.")

        self._m0 = []
        self._v0 = []
        for tag in layout:
            if tag == "alpha":
                self._m0.append(float(m0_level)); self._v0.append(float(v0_level))
            elif tag == "beta":
                self._m0.append(float(m0_trend)); self._v0.append(float(v0_trend))
            else:
                k = int(tag.split("_")[1]) - 1
                self._m0.append(float(m0_season_arr[k]))
                self._v0.append(float(v0_season_arr[k]))
        self._m0 = np.array(self._m0, float) if self.dim > 0 else np.zeros(0)
        self._v0 = np.array(self._v0, float) if self.dim > 0 else np.zeros(0)

        # ---- Storage
        self.keep: Dict[str, np.ndarray] = {}
        self.accepted: int = 0
        self.proposed: int = 0

        # ---- Per-coordinate logQ steps + accept/propose
        self._step_logQ_vec = (
            np.full(len(self._active_Q_idx), float(self.cfg.step_logQ), float)
            if self._active_Q_idx else np.zeros(0, float)
        )
        self.acc_counts = {
            "logsigma": {"acc": 0, "prop": 0},
            "xi":       {"acc": 0, "prop": 0},
            "slope":    {"acc": 0, "prop": 0},
            "season":   {"acc": 0, "prop": 0},
        }
        self.acc_logQ = np.zeros(len(self._active_Q_idx), dtype=int)
        self.prop_logQ = np.zeros(len(self._active_Q_idx), dtype=int)

        # ---- EMAs & adaptation windows
        self.ema_logZ: float = float("nan")
        self.ema_Q_alpha: float = float("nan")
        self._ema_rho: float = 0.02
        self._mh_prev = {k: {"acc":0, "prop":0} for k in self.acc_counts}
        self._mh_prev_logQ = np.zeros((len(self._active_Q_idx), 2), dtype=float)  # [:,0]=acc, [:,1]=prop
        self._adapt_round = 0

        # ---- Truth overlays (optional)
        self.true_sigma: Optional[float] = None
        self.true_xi: Optional[float] = None
        self.true_Q: Optional[np.ndarray] = None
        self.true_mu_t: Optional[np.ndarray] = None

        # ---- last PF diagnostics
        self.last_log_evidence: float = float("nan")
        self.last_pf_diag: Dict[str, float] = {}

    # --------------------- Truth registration (optional) --------------------- #
    def set_truth(self, sigma: Optional[float] = None, xi: Optional[float] = None, Q: Optional[np.ndarray] = None) -> None:
        self.true_sigma = sigma
        self.true_xi = xi
        self.true_Q = None if Q is None else np.asarray(Q, float)

    def set_truth_paths(self, mu: Optional[np.ndarray] = None) -> None:
        self.true_mu_t = None if mu is None else np.asarray(mu, float)

    # ----------------------- Helpers: norm / ESS / EMA ---------------------- #
    @staticmethod
    def _safe_normalize(p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, float)
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        p[p < 0.0] = 0.0
        s = float(np.sum(p))
        if not np.isfinite(s) or s <= 0.0:
            return np.full_like(p, 1.0 / p.size)
        p /= s
        s2 = float(np.sum(p))
        if not np.isclose(s2, 1.0, atol=1e-12):
            p /= s2
        return p

    @staticmethod
    def _ess(w: np.ndarray) -> float:
        s2 = float(np.sum(w * w))
        return (1.0 / s2) if s2 > 0.0 else 0.0

    def _ema(self, prev: float, x: float, rho: float) -> float:
        if not np.isfinite(prev):
            return float(x)
        return (1.0 - rho) * prev + rho * float(x)

    # ----------------------------- Model maps -------------------------------- #
    def _unpack_theta(self, theta: dict) -> None:
        """Write θ dict into object state (for convenience)."""
        self.logsigma = float(theta["logsigma"]); self.sigma = float(np.exp(self.logsigma))
        self.xi = float(theta["xi"])
        # deterministic components
        if self.level_mode == "deterministic" and "level_value" in theta:
            self.level_value = float(theta["level_value"])
        if self.trend_mode == "deterministic" and "slope_value" in theta:
            self.slope_value = float(theta["slope_value"])
        if self.seasonal_mode == "deterministic" and "season_vec" in theta:
            self.season_vec = np.asarray(theta["season_vec"], float)

        # process noises
        if self._active_Q_idx and "logQ" in theta:
            self.logQ = np.asarray(theta["logQ"], float)
            for j, k in enumerate(self._active_Q_idx):
                self.Q[k] = float(np.exp(self.logQ[j]))

    def _pack_theta(self) -> dict:
        """Collect current θ into a dict."""
        theta = dict(
            logsigma=float(self.logsigma),
            xi=float(self.xi),
            logQ=self.logQ.copy() if self._active_Q_idx else np.zeros(0),
        )
        if self.level_mode == "deterministic":
            theta["level_value"] = float(self.level_value)
        if self.trend_mode == "deterministic":
            theta["slope_value"] = float(self.slope_value)
        if self.seasonal_mode == "deterministic":
            theta["season_vec"] = self.season_vec.copy()
        return theta

    # ----------------------------- State model ------------------------------ #
    def _state_mean(self, x_prev: np.ndarray, t: int) -> np.ndarray:
        if self.dim == 0:
            return np.zeros(0, float)
        m = np.zeros_like(x_prev)
        # alpha
        if self.idx_alpha is not None:
            drift = 0.0
            if self.idx_beta is not None:
                drift = x_prev[self.idx_beta]
            elif self.trend_mode == "deterministic":
                drift = self.slope_value
            m[self.idx_alpha] = x_prev[self.idx_alpha] + drift
        # beta (local trend)
        if self.idx_beta is not None:
            m[self.idx_beta] = x_prev[self.idx_beta]
        # dynamic seasonal shift + closure
        if self.seasonal_mode == "dynamic":
            g0, gL = self.idx_gamma_start, self.idx_gamma_end
            if g0 <= gL - 1:
                m[g0:gL] = x_prev[g0 + 1 : gL + 1]
            prev_gamma = x_prev[g0 : gL + 1]
            m[gL] = -np.sum(prev_gamma)
        return m

    def _mu_from_state(self, x_t: np.ndarray, t: int) -> float:
        # alpha contribution
        if self.level_mode == "dynamic":
            base = float(x_t[self.idx_alpha]) if self.idx_alpha is not None else 0.0
        else:
            base = float(self.level_value)
            if self.idx_beta is not None:  # dynamic trend exists
                base += float(x_t[self.idx_beta] * t)
            elif self.trend_mode == "deterministic":
                base += float(self.slope_value * t)
        # seasonal
        if self.seasonal_mode == "dynamic":
            seas = float(x_t[self.idx_gamma_end]) if self.idx_gamma_end is not None else 0.0
        elif self.seasonal_mode == "deterministic":
            seas = float(self.season_vec[t % self.period])
        else:
            seas = 0.0
        return base + seas

    # ------------------- APF likelihood (marginal) -------------------------- #
    def _apf_predictive_loglik(self, y_t: float, x_prev: np.ndarray, t: int) -> float:
        """Cheap proxy m(y_t | x_{t-1}) used by APF: evaluate at E[x_t|x_{t-1}]."""
        if self.dim == 0:
            # purely deterministic mean
            mu_hat = 0.0
            if self.level_mode == "deterministic":
                mu_hat += self.level_value + (self.slope_value * (t-1) if self.trend_mode == "deterministic" else 0.0)
            if self.seasonal_mode == "deterministic":
                mu_hat += self.season_vec[(t-1) % self.period]
        else:
            x_hat = self._state_mean(x_prev, t)
            mu_hat = self._mu_from_state(x_hat, t-1)
        return gev_logpdf(y_t, mu_hat, self.sigma, self.xi)

    @staticmethod
    def _resample_systematic(p: np.ndarray) -> np.ndarray:
        """Systematic resampling: returns ancestor indices."""
        N = p.size
        u0 = np.random.rand() / N
        c = np.cumsum(p)
        a = np.empty(N, dtype=int)
        i = 0
        for m in range(N):
            u = u0 + m / N
            while u > c[i]:
                i += 1
            a[m] = i
        return a

    def _transition_sample(self, x_prev: np.ndarray, t: int) -> np.ndarray:
        if self.dim == 0:
            return np.zeros(0, float)
        mean = self._state_mean(x_prev, t)
        var = np.where(self.Q > 0.0, self.Q, self.cfg.trans_eps)
        return mean + np.random.normal(0.0, np.sqrt(var), size=self.dim)

    def _transition_logpdf(self, x_prev: np.ndarray, x_cur: np.ndarray, t: int) -> float:
        if self.dim == 0:
            return 0.0
        mean = self._state_mean(x_prev, t)
        out = 0.0
        for k in range(self.dim):
            var = self.Q[k] if self.Q[k] > 0.0 else self.cfg.trans_eps
            diff = x_cur[k] - mean[k]
            out += -0.5 * (math.log(2.0 * math.pi * var) + (diff * diff) / var)
        return float(out)

    def _apf_loglik(self) -> Tuple[float, dict]:
        """
        Auxiliary PF to estimate log p(y|θ). Returns (logZ_hat, pf_diag).
        """
        N, T, D = self.cfg.n_particles, self.T, self.dim
        if N <= 0:
            raise ValueError("n_particles must be >= 1")

        # particles, weights, ancestors (only need current&prev slices)
        if D > 0:
            parts_prev = np.random.normal(self._m0, np.sqrt(self._v0), size=(N, D))
        else:
            parts_prev = np.zeros((N, 0), float)

        w_prev = np.full(N, 1.0 / N, float)  # w0 uniform

        logZ = 0.0
        ess_list, maxw_list = [], []

        it = range(1, T + 1)
        if self.cfg.progress:
            it = tqdm(it, total=T, leave=False, desc="Running conditional bootstrap PF...")

        for t in it:
            y_t = self.y[t - 1]

            # Look-ahead scores for ancestors: log w_{t-1} + log m(y_t | x_{t-1})
            look = np.array([self._apf_predictive_loglik(y_t, parts_prev[j, :], t=t) for j in range(N)])
            log_anc = np.log(w_prev + 1e-300) + (look - np.max(look))
            w_anc = self._safe_normalize(np.exp(log_anc))

            # Resample ancestors & propagate
            anc = self._resample_systematic(w_anc)

            if D > 0:
                parts_cur = np.zeros((N, D), float)
                for i in range(N):
                    parts_cur[i, :] = self._transition_sample(parts_prev[anc[i], :], t=t)
            else:
                parts_cur = parts_prev  # dummy

            # Corrected weights: g(y_t|x_t^i) / m(y_t|x_{t-1}^{anc_i})
            lw = np.zeros(N)
            corr = np.zeros(N)
            for i in range(N):
                mu = self._mu_from_state(parts_cur[i, :], t=t-1) if D > 0 else (
                    (self.level_value + (self.slope_value*(t-1) if self.trend_mode=="deterministic" else 0.0))
                    + (self.season_vec[(t-1) % self.period] if self.seasonal_mode=="deterministic" else 0.0)
                )
                lw[i] = gev_logpdf(y_t, mu, self.sigma, self.xi)
                corr[i] = self._apf_predictive_loglik(y_t, parts_prev[anc[i], :], t=t)

            lw_corr = lw - corr
            lw_max = np.max(lw_corr)
            wt = np.exp(lw_corr - lw_max)
            Zt = np.mean(wt) + 1e-300
            logZ += lw_max + math.log(Zt)
            w_cur = wt / (np.sum(wt) + 1e-300)

            # diagnostics
            ess = self._ess(w_cur)
            mxw = float(np.max(w_cur))
            ess_list.append(ess)
            maxw_list.append(mxw)

            if self.cfg.progress and isinstance(it, tqdm):
                it.set_postfix(ESS=f"{ess:4.1f}", MaxW=f"{mxw:0.4f}")

            # step
            parts_prev = parts_cur
            w_prev = w_cur

        pf_diag = {
            "ess_mean": float(np.mean(ess_list)) if ess_list else float("nan"),
            "ess_min": float(np.min(ess_list)) if ess_list else float("nan"),
            "maxw_mean": float(np.mean(maxw_list)) if maxw_list else float("nan"),
            "maxw_max": float(np.max(maxw_list)) if maxw_list else float("nan"),
        }
        return float(logZ), pf_diag

    # ------------------------------- Priors --------------------------------- #
    def _log_prior(self, theta: dict) -> float:
        lp = 0.0
        # logσ ~ N(m_sigma, s_sigma^2)
        lp += -0.5 * ((theta["logsigma"] - self.priors.m_sigma) ** 2) / (self.priors.s_sigma ** 2)
        # xi ~ N(m_xi, s_xi^2)
        lp += -0.5 * ((theta["xi"] - self.priors.m_xi) ** 2) / (self.priors.s_xi ** 2)

        # deterministic Gaussian priors
        if self.level_mode == "deterministic" and "level_value" in theta:
            lp += -0.5 * ((theta["level_value"] - self.priors.m_level) ** 2) / (self.priors.s_level ** 2)
        if self.trend_mode == "deterministic" and "slope_value" in theta:
            lp += -0.5 * ((theta["slope_value"] - self.priors.m_slope) ** 2) / (self.priors.s_slope ** 2)
        if self.seasonal_mode == "deterministic" and "season_vec" in theta:
            m_first = self.priors.m_season
            if m_first is None:
                m_first = np.zeros(self.period - 1, float)
            m_first = np.asarray(m_first, float)
            s = float(self.priors.s_season)
            # first p-1 entries have Gaussian prior, last implied
            lp += -0.5 * np.sum(((theta["season_vec"][:-1] - m_first) / s) ** 2)

        # IG priors on active Q with Jacobian for log-transform: log p(logQ) = log p(Q) + log Q
        if self._active_Q_idx:
            for j, k in enumerate(self._active_Q_idx):
                logQ_j = theta["logQ"][j]
                Q_j = float(np.exp(logQ_j))
                # choose corresponding IG hyperparams
                if k == self.idx_alpha:
                    a, b = self.priors.a_q_alpha, self.priors.b_q_alpha
                elif k == self.idx_beta:
                    a, b = self.priors.a_q_beta,  self.priors.b_q_beta
                else:
                    a, b = self.priors.a_q_gamma, self.priors.b_q_gamma
                if Q_j <= 0.0:
                    return -np.inf
                # up to constants: -(a+1)log Q - b/Q  + log|dQ/dlogQ| = +log Q
                lp += -(a + 1.0) * np.log(Q_j) - (b / Q_j) + np.log(Q_j)
                # constants (a log b - log Γ(a)) omitted
        return float(lp)

    # --------------------- Adaptation helpers (steps) ----------------------- #
    def _get_step(self, key: str) -> float:
        if   key == "logsigma": return self.cfg.step_logsigma
        elif key == "xi":       return self.cfg.step_xi
        elif key == "slope":    return self.cfg.step_slope
        elif key == "season":   return self.cfg.step_season
        else:                   return 0.0

    def _set_step(self, key: str, new_val: float) -> None:
        s = float(np.clip(new_val, self.cfg.step_min, self.cfg.step_max))
        if   key == "logsigma": self.cfg.step_logsigma = s
        elif key == "xi":       self.cfg.step_xi = s
        elif key == "slope":    self.cfg.step_slope = s
        elif key == "season":   self.cfg.step_season = s

    def _adapt_steps(self, it: int) -> None:
        cfg = self.cfg
        if not cfg.adapt_steps:
            return
        in_window = (cfg.adapt_until == "all") or (it < cfg.burn)
        if (it + 1) % max(1, cfg.adapt_every) != 0 or not in_window:
            return

        # learning-rate schedule
        k   = self._adapt_round
        eta = cfg.adapt_eta0 / ((1.0 + k) ** cfg.adapt_eta_decay)
        target = cfg.adapt_target_1d

        changed = []

        # scalar keys
        for key in self.acc_counts:
            acc = self.acc_counts[key]["acc"]; prop = self.acc_counts[key]["prop"]
            acc_w = acc - self._mh_prev[key]["acc"]
            prop_w = prop - self._mh_prev[key]["prop"]
            if prop_w > 0:
                rate = acc_w / prop_w
                s = self._get_step(key)
                s_new = s * np.exp(eta * (rate - target))
                self._set_step(key, s_new)
                changed.append((key, s, self._get_step(key), rate))
                self._mh_prev[key]["acc"] = acc
                self._mh_prev[key]["prop"] = prop

        # per-coordinate logQ
        for j in range(len(self._active_Q_idx)):
            acc = int(self.acc_logQ[j]); prop = int(self.prop_logQ[j])
            acc_w = acc - int(self._mh_prev_logQ[j, 0])
            prop_w = prop - int(self._mh_prev_logQ[j, 1])
            if prop_w > 0:
                rate = acc_w / prop_w
                s = float(self._step_logQ_vec[j])
                s_new = s * np.exp(eta * (rate - target))
                self._step_logQ_vec[j] = float(np.clip(s_new, cfg.step_min, cfg.step_max))
                changed.append((f"logQ[{j}]", s, float(self._step_logQ_vec[j]), rate))
                self._mh_prev_logQ[j, 0] = acc
                self._mh_prev_logQ[j, 1] = prop

        if changed and self.cfg.progress:
            msg = " | ".join([f"{k}: {old:.4g}→{new:.4g} (acc_win={r:.2f})" for (k,old,new,r) in changed])
            print(f"  [adapt] η={eta:.4f} target={target:.2f} :: {msg}")

        self._adapt_round += 1

    # --------------------------------- MCMC --------------------------------- #
    def run(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg

        save_iters = list(range(cfg.burn, cfg.n_iter, cfg.thin))
        n_kept = max(0, len(save_iters))

        # storage
        keep_idx = 0
        self.keep = {
            "sigma": np.zeros(n_kept, float),
            "xi":    np.zeros(n_kept, float),
            "log_evidence": np.zeros(n_kept, float),
        }
        if self._active_Q_idx:
            self.keep["Q"] = np.zeros((n_kept, len(self._active_Q_idx)), float)
        if self.level_mode == "deterministic":
            self.keep["level_value"] = np.zeros(n_kept, float)
        if self.trend_mode == "deterministic":
            self.keep["slope_value"] = np.zeros(n_kept, float)
        if self.seasonal_mode == "deterministic":
            self.keep["season_vector"] = np.zeros((n_kept, self.period), float)
        if cfg.store_states and self.dim > 0:
            self.keep["alpha_t"] = np.zeros((n_kept, self.T), float) if self.idx_alpha is not None else None
            self.keep["beta_t"]  = np.zeros((n_kept, self.T), float) if self.idx_beta  is not None else None
            if self.seasonal_mode == "dynamic":
                self.keep["gamma_t"] = np.zeros((n_kept, self.T), float)

        # initialize θ and compute initial log-likelihood
        theta_cur = self._pack_theta()
        self._unpack_theta(theta_cur)  # sync dependent fields
        if cfg.progress: print("Running conditional bootstrap PF...")
        loglike_cur, pf_diag = self._apf_loglik()
        self.last_log_evidence = float(loglike_cur)
        self.last_pf_diag = pf_diag
        self.ema_logZ = self._ema(self.ema_logZ, self.last_log_evidence, self._ema_rho)
        logprior_cur = self._log_prior(theta_cur)
        logpost_cur = logprior_cur + loglike_cur

        # helper to evaluate proposal and accept/reject
        def _eval_and_accept(theta_prop: dict, block_key: Optional[str]) -> bool:
            nonlocal theta_cur, logpost_cur, logprior_cur, loglike_cur
            self._unpack_theta(theta_prop)
            loglike_p, pf_diag_p = self._apf_loglik()
            logprior_p = self._log_prior(theta_prop)
            logpost_p = logprior_p + loglike_p
            logacc = logpost_p - logpost_cur
            accept = (np.log(np.random.rand()) < min(0.0, logacc))
            if accept:
                theta_cur = theta_prop
                logpost_cur = logpost_p
                logprior_cur = logprior_p
                loglike_cur = loglike_p
                self.last_pf_diag = pf_diag_p
                self.last_log_evidence = float(loglike_cur)
                if block_key is not None and block_key in self.acc_counts:
                    self.acc_counts[block_key]["acc"] += 1
            else:
                self._unpack_theta(theta_cur)
            # EMA after each PF
            self.ema_logZ = self._ema(self.ema_logZ, self.last_log_evidence, self._ema_rho)
            # EMA of Q_alpha if active
            if self.idx_alpha is not None and self._active_Q_idx and (self.idx_alpha in self._active_Q_idx):
                j_alpha = self._active_Q_idx.index(self.idx_alpha)
                self.ema_Q_alpha = self._ema(self.ema_Q_alpha, float(np.exp(self.logQ[j_alpha])), self._ema_rho)
            return accept

        print_every = cfg.progress_every if cfg.progress_every > 0 else max(1, cfg.n_iter // 50)

        for it in range(cfg.n_iter):
            if cfg.progress:
                print(f"Iteration {it + 1}/{cfg.n_iter}")

            # ---- Block A: logQ per-coordinate ----
            if len(self._active_Q_idx):
                if self.cfg.progress:
                    print("Updating Q (per-coordinate)...")
                for j in range(len(self._active_Q_idx)):
                    theta_p = dict(theta_cur)
                    prop = theta_p["logQ"].copy()
                    prop[j] = theta_p["logQ"][j] + np.random.normal(0.0, float(self._step_logQ_vec[j]))
                    theta_p["logQ"] = prop
                    self.prop_logQ[j] += 1
                    accepted = _eval_and_accept(theta_p, block_key=None)  # not counted in acc_counts
                    if accepted:
                        self.acc_logQ[j] += 1

            # ---- Block B: deterministic slope (if any) ----
            if self.trend_mode == "deterministic":
                if self.cfg.progress: print("Updating deterministic slope...")
                th = dict(theta_cur); th["slope_value"] = th["slope_value"] + np.random.normal(0.0, self.cfg.step_slope)
                self.acc_counts["slope"]["prop"] += 1
                _eval_and_accept(th, "slope")

            # ---- Block C: deterministic season (if any) ----
            if self.seasonal_mode == "deterministic":
                if self.cfg.progress: print("Updating deterministic seasonality...")
                th = dict(theta_cur); prop = th["season_vec"].copy()
                prop[:-1] = prop[:-1] + np.random.normal(0.0, self.cfg.step_season, size=self.period - 1)
                prop[-1]  = -np.sum(prop[:-1])
                th["season_vec"] = prop
                self.acc_counts["season"]["prop"] += 1
                _eval_and_accept(th, "season")

            # ---- Block D/E: logσ and ξ ----
            th = dict(theta_cur); th["logsigma"] = th["logsigma"] + np.random.normal(0.0, self.cfg.step_logsigma)
            self.acc_counts["logsigma"]["prop"] += 1
            _eval_and_accept(th, "logsigma")

            th = dict(theta_cur); th["xi"] = th["xi"] + np.random.normal(0.0, self.cfg.step_xi)
            self.acc_counts["xi"]["prop"] += 1
            _eval_and_accept(th, "xi")

            # ---- Adapt step sizes
            self._adapt_steps(it)

            # ---- Pretty status line every k iters
            if cfg.progress and ((it + 1) % print_every == 0 or it == cfg.n_iter - 1):
                def acc_line(key):
                    c = self.acc_counts[key]; a, p = c["acc"], max(1, c["prop"])
                    return f"{a}/{p} ({100.0*a/p:4.1f}%)"
                # logQ compact string
                q_parts = []
                for jj in range(len(self._active_Q_idx)):
                    cur_q = float(np.exp(self.logQ[jj]))
                    rate = (100.0 * self.acc_logQ[jj] / max(1, self.prop_logQ[jj]))
                    q_parts.append(f"Q[{jj}]={cur_q:.3g} (acc {rate:4.1f}% / s {self._step_logQ_vec[jj]:.3g})")
                q_line = " | ".join(q_parts)

                # gamma preview
                gamma_head = ""
                if self.seasonal_mode == "deterministic":
                    gamma_head = f"γ[:3]=[{', '.join([f'{g:.3f}' for g in self.season_vec[:3]])}]"
                elif self.seasonal_mode == "dynamic":
                    gamma_head = "γ[:3]=[dynamic]"

                d = self.last_pf_diag or {}
                pf_part = f" | PF: ESS(mean/min)={d.get('ess_mean', float('nan')):.1f}/{d.get('ess_min', float('nan')):.1f} MaxW(max)={d.get('maxw_max', float('nan')):0.4f}"

                # Q_alpha highlight if present
                qalpha_part = ""
                if self.idx_alpha is not None and self._active_Q_idx and (self.idx_alpha in self._active_Q_idx):
                    j = self._active_Q_idx.index(self.idx_alpha)
                    qalpha_cur = float(np.exp(self.logQ[j]))
                    if np.isfinite(self.ema_Q_alpha):
                        qalpha_part = f" | Q_alpha: cur={qalpha_cur:.6g} ema={self.ema_Q_alpha:.6g}"
                    else:
                        qalpha_part = f" | Q_alpha: cur={qalpha_cur:.6g}"

                print(
                    f"[it {it+1}/{cfg.n_iter}] "
                    f"logZ={self.last_log_evidence:.3f} ema={self.ema_logZ:.3f} "
                    f"σ={np.exp(self.logsigma):.3f} ξ={self.xi:.3f} "
                    f"acc(logσ)={acc_line('logsigma')} acc(ξ)={acc_line('xi')}"
                    f"{(' | acc(slope)=' + acc_line('slope')) if self.trend_mode=='deterministic' else ''}"
                    f"{(' | acc(season)=' + acc_line('season')) if self.seasonal_mode=='deterministic' else ''}"
                    f"{qalpha_part} | {q_line}{pf_part} | {gamma_head}"
                )

            # store kept draws
            if it in save_iters and keep_idx < n_kept:
                self.keep["sigma"][keep_idx] = float(np.exp(self.logsigma))
                self.keep["xi"][keep_idx] = float(self.xi)
                self.keep["log_evidence"][keep_idx] = float(self.last_log_evidence)
                if self._active_Q_idx:
                    self.keep["Q"][keep_idx, :] = np.exp(self.logQ.copy())
                if self.level_mode == "deterministic":
                    self.keep["level_value"][keep_idx] = self.level_value
                if self.trend_mode == "deterministic":
                    self.keep["slope_value"][keep_idx] = self.slope_value
                if self.seasonal_mode == "deterministic":
                    self.keep["season_vector"][keep_idx, :] = self.season_vec.copy()

                if cfg.store_states and self.dim > 0:
                    # optional: one smoothed path using backward sampling
                    path = self._smooth_path()  # (T, D)
                    if self.idx_alpha is not None and self.keep["alpha_t"] is not None:
                        self.keep["alpha_t"][keep_idx, :] = path[:, self.idx_alpha]
                    if self.idx_beta is not None and self.keep["beta_t"] is not None:
                        self.keep["beta_t"][keep_idx, :]  = path[:, self.idx_beta]
                    if self.seasonal_mode == "dynamic" and self.keep.get("gamma_t") is not None:
                        self.keep["gamma_t"][keep_idx, :] = path[:, self.idx_gamma_end]
                keep_idx += 1

        if cfg.progress:
            # per-block accept summary (logQ per-coordinate already shown in lines)
            def acc_report(k):
                c = self.acc_counts[k]; p = max(1, c["prop"])
                return f"{c['acc']}/{p} ({100.0*c['acc']/p:0.1f}%)"
            print("[done] Accept rates:",
                  f"logσ={acc_report('logsigma')}, ξ={acc_report('xi')}, "
                  f"{'slope=' + acc_report('slope') + ', ' if self.trend_mode=='deterministic' else ''}"
                  f"{'season=' + acc_report('season') + ', ' if self.seasonal_mode=='deterministic' else ''}"
                  f"logQ per-coordinate shown above.")

        return self.keep

    # ------------- (Optional) draw one smoothed path at fixed θ ------------- #
    def _smooth_path(self) -> np.ndarray:
        """
        Run a bootstrap/APF and do backward-sampling to draw x_{1:T} | y, θ.
        Returns array (T, D). If D==0, returns (T, 0).
        """
        N, T, D = self.cfg.n_particles, self.T, self.dim
        if D == 0:
            return np.zeros((T, 0), float)

        parts = np.zeros((T + 1, N, D), float)
        a = np.zeros((T + 1, N), int)
        w = np.zeros((T + 1, N), float)

        # initialize from prior
        parts[0] = np.random.normal(self._m0, np.sqrt(self._v0), size=(N, D))
        w[0] = np.full(N, 1.0 / N)

        # forward pass (bootstrap with APF correction)
        for t in range(1, T + 1):
            y_t = self.y[t - 1]
            # APF resampling using look-ahead
            look = np.array([self._apf_predictive_loglik(y_t, parts[t-1, j, :], t=t) for j in range(N)])
            wa = self._safe_normalize(np.exp(np.log(w[t-1] + 1e-300) + look - np.max(look)))
            anc = self._resample_systematic(wa)
            a[t] = anc
            # propagate
            for i in range(N):
                parts[t, i, :] = self._transition_sample(parts[t-1, anc[i], :], t=t)
            # weights correction
            logw = np.zeros(N)
            for i in range(N):
                mu = self._mu_from_state(parts[t, i, :], t=t-1)
                logw[i] = gev_logpdf(y_t, mu, self.sigma, self.xi) - self._apf_predictive_loglik(y_t, parts[t-1, anc[i], :], t=t)
            w[t] = self._safe_normalize(np.exp(logw - np.max(logw)))

        # backward-sampling
        idx = np.zeros(T + 1, dtype=int)
        idx[T] = int(np.random.choice(N, p=w[T]))
        path = np.zeros((T, D), float)
        for t in range(T, 0, -1):
            path[t-1, :] = parts[t, idx[t], :]
            idx[t-1] = a[t, idx[t]]
        return path

    # ------------------------------- Persistence ---------------------------- #
    def save_posterior(self, out_npz_path: str, extra_meta: Optional[dict] = None) -> None:
        _ensure_dir(os.path.dirname(out_npz_path))
        arrays = dict(self.keep)
        arrays["y"] = self.y.copy()

        if self.true_mu_t is not None: arrays["true_mu_t"] = np.asarray(self.true_mu_t, float)
        np.savez_compressed(out_npz_path, **arrays)

        meta = {
            "T": int(self.T),
            "dim": int(self.dim),
            "period": int(self.period),
            "modes": {
                "level_mode": self.level_mode,
                "trend_mode": self.trend_mode,
                "seasonal_mode": self.seasonal_mode,
            },
            "layout": list(self._layout),
            "idx_alpha": self.idx_alpha,
            "idx_beta": self.idx_beta,
            "idx_gamma_start": self.idx_gamma_start,
            "idx_gamma_end": self.idx_gamma_end,
            "cfg": asdict(self.cfg),
            "priors": asdict(self.priors),
            "true_sigma": self.true_sigma,
            "true_xi": self.true_xi,
            "true_Q": (None if self.true_Q is None else np.asarray(self.true_Q, float).tolist()),
            "accept_rate_overall": float(self.accepted / max(1, self.proposed)) if (self.proposed > 0) else None,
            "acc_counts": self.acc_counts,
            "acc_logQ": self.acc_logQ.tolist() if len(self.acc_logQ) else [],
            "prop_logQ": self.prop_logQ.tolist() if len(self.prop_logQ) else [],
            "step_logQ_vec": self._step_logQ_vec.tolist() if len(self._step_logQ_vec) else [],
        }
        if extra_meta:
            meta.update(extra_meta)

        meta_path = out_npz_path.replace(".npz", ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[save] Posterior -> {out_npz_path}")
        print(f"[save] Metadata  -> {meta_path}")

# ------------------------- CLI / Example run & plots ------------------------ #
if __name__ == "__main__":
    import sys, argparse
    import matplotlib.pyplot as plt
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from simulator.extremal_time_series import Extremal_Time_Series
    # plotting optional in your project:
    try:
        from simulator.dgev_plotter import DGEVPlotter  # optional
    except Exception:
        DGEVPlotter = None

    parser = argparse.ArgumentParser(description="DGEV PMMH + APF Sampler")
    # Modes
    parser.add_argument("--level-mode", choices=["dynamic", "deterministic"], default="dynamic")
    parser.add_argument("--trend-mode", choices=["dynamic", "deterministic", "none"], default="none")
    parser.add_argument("--season-mode", choices=["dynamic", "deterministic", "none"], default="none")
    # Basics
    parser.add_argument("--period", type=int, default=4)
    parser.add_argument("--T", type=int, default=100)
    # Initial values (shared)
    parser.add_argument("--level-init", type=float, default=5.0)
    parser.add_argument("--slope-init", type=float, default=0.02)

    parser.add_argument("--true-sigma", type=float, default=2.0)
    parser.add_argument("--true-xi", type=float, default=0.1)
    parser.add_argument("--q-alpha", type=float, default=1e-3)
    parser.add_argument("--q-beta",  type=float, default=1e-9)
    parser.add_argument("--q-gamma", type=float, default=1e-7)

    parser.add_argument("--prior-m-sigma", type=float, default=1.0)
    parser.add_argument("--prior-s-sigma", type=float, default=1.0)
    parser.add_argument("--prior-m-xi", type=float, default=0.0)
    parser.add_argument("--prior-s-xi", type=float, default=0.2)
    parser.add_argument("--prior-aq-alpha", type=float, default=1.1)
    parser.add_argument("--prior-aq-beta", type=float, default=1.1)
    parser.add_argument("--prior-aq-gamma", type=float, default=1.1)
    parser.add_argument("--prior-bq-alpha", type=float, default=1e-3) # mean= b/(a-1)
    parser.add_argument("--prior-bq-beta", type=float, default=1e-3)
    parser.add_argument("--prior-bq-gamma", type=float, default=1e-3)
    parser.add_argument("--prior-m-level", type=float, default=0.0)
    parser.add_argument("--prior-s-level", type=float, default=10.0)
    parser.add_argument("--prior-m-slope", type=float, default=0.0)
    parser.add_argument("--prior-s-slope", type=float, default=10.0)
    parser.add_argument("--prior-m-season", type=str, default=None,
                        help="Comma-separated first (p-1) means for deterministic seasonal prior (e.g. '0,0,0').")
    parser.add_argument("--prior-s-season", type=float, default=5.0)

    # Sampler config
    parser.add_argument("--n-iter", type=int, default=2000)
    parser.add_argument("--burn", type=int, default=200)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--step-logsigma", type=float, default=0.2)
    parser.add_argument("--step-xi", type=float, default=0.2)
    parser.add_argument("--step-level", type=float, default=0.02)
    parser.add_argument("--step-slope", type=float, default=0.0005)
    parser.add_argument("--step-season", type=float, default=0.02)
    parser.add_argument("--step-logQ", type=float, default=0.4)
    parser.add_argument("--particles", type=int, default=400)
    parser.add_argument("--trans-eps", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--progress", default=True)
    parser.add_argument("--progress-every", type=int, default=10,
                        help="print compact summary every k iterations (0=auto)")
    parser.add_argument("--store-states", action="store_true")

    # Adaptive RW–MH CLI
    parser.add_argument("--adapt-steps", action="store_true")
    parser.add_argument("--adapt-every", type=int, default=25)
    parser.add_argument("--adapt-until", choices=["burn","all"], default="burn")
    parser.add_argument("--adapt-eta0", type=float, default=0.05)
    parser.add_argument("--adapt-decay", type=float, default=0.75)
    parser.add_argument("--adapt-target-1d", type=float, default=0.44)
    parser.add_argument("--step-min", type=float, default=1e-5)
    parser.add_argument("--step-max", type=float, default=1.0)

    # Output & plotting
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--show-plots", action="store_true")

    args = parser.parse_args()
    np.random.seed(args.seed)

    sim_level_mode  = args.level_mode
    sim_trend_mode  = args.trend_mode
    sim_season_mode = args.season_mode

    # Seasonal priors for simulator
    m0_season_first = np.zeros(args.period - 1, float)
    v0_season_first = np.full(args.period - 1, 0.5, float)
    if sim_season_mode == "deterministic":
        m0_season_first = build_seasonal(args.period)
    elif sim_season_mode == "none":
        m0_season_first = None
        v0_season_first = None

    # Simulate data
    ts = Extremal_Time_Series(
        parameters=(args.true_sigma, args.true_xi),
        level_mode=sim_level_mode,
        trend_mode=sim_trend_mode,
        seasonal_mode=sim_season_mode,
        period=args.period,
        q_level=args.q_alpha,
        q_trend=args.q_beta,
        q_season=args.q_gamma,
        m0_level=args.level_init, v0_level=0.2,
        m0_trend=(args.slope_init if sim_trend_mode != "none" else 0.0), v0_trend=0.05,
        m0_season=m0_season_first, v0_season=v0_season_first,
        start_date=datetime(1980, 1, 1),
    )

    y = []
    for _ in range(args.T):
        ts.move()
        y.append(ts.measure())
    y = np.asarray(y, float)

    truths = ts.get_truth_paths(as_numpy=False)
    mu_T = np.asarray(truths["mu"][1:1 + args.T], float)

    # Priors & config
    m_season_prior = parse_csv_floats(args.prior_m_season)
    if m_season_prior is not None and len(m_season_prior) != args.period - 1:
        raise ValueError(f"--prior-m-season must have length {args.period - 1} (got {len(m_season_prior)}).")

    priors = Priors(
        m_sigma=float(args.prior_m_sigma), s_sigma=float(args.prior_s_sigma),
        m_xi=float(args.prior_m_xi), s_xi=float(args.prior_s_xi),
        a_q_alpha=float(args.prior_aq_alpha), b_q_alpha=float(args.prior_bq_alpha),
        a_q_beta=float(args.prior_aq_beta), b_q_beta=float(args.prior_bq_beta),
        a_q_gamma=float(args.prior_aq_gamma), b_q_gamma=float(args.prior_bq_gamma),
        m_level=float(args.prior_m_level), s_level=float(args.prior_s_level),
        m_slope=float(args.prior_m_slope), s_slope=float(args.prior_s_slope),
        m_season=m_season_prior, s_season=float(args.prior_s_season),
    )

    cfg = SamplerConfig(
        n_iter=args.n_iter, burn=args.burn, thin=args.thin,
        step_logsigma=args.step_logsigma, step_xi=args.step_xi,
        step_level=args.step_level, step_slope=args.step_slope, step_season=args.step_season,
        step_logQ=args.step_logQ,
        n_particles=args.particles, trans_eps=args.trans_eps,
        random_seed=args.seed, progress=bool(args.progress),
        progress_every=int(args.progress_every),
        store_states=bool(args.store_states),
        adapt_steps=bool(args.adapt_steps),
        adapt_every=int(args.adapt_every),
        adapt_until=str(args.adapt_until),
        adapt_eta0=float(args.adapt_eta0),
        adapt_eta_decay=float(args.adapt_decay),
        adapt_target_1d=float(args.adapt_target_1d),
        step_min=float(args.step_min),
        step_max=float(args.step_max),
    )

    seasonal_init_pminus1 = (
        np.asarray(m_season_prior, float) if (sim_season_mode == "deterministic" and m_season_prior is not None)
        else (build_seasonal(args.period) if sim_season_mode == "deterministic" else None)
    )

    sampler = DGEVPMMH(
        y=y, period=args.period,
        level_mode=sim_level_mode, trend_mode=sim_trend_mode, seasonal_mode=sim_season_mode,
        m0_level=args.level_init, v0_level=0.2,
        m0_trend=(args.slope_init if sim_trend_mode != "none" else 0.0), v0_trend=0.05,
        m0_season=(m0_season_first if sim_season_mode == "dynamic" else None),
        v0_season=(v0_season_first if sim_season_mode == "dynamic" else None),
        priors=priors, cfg=cfg,
        seasonal_vector_init=seasonal_init_pminus1,
    )

    true_Q = []
    if sim_level_mode == "dynamic": true_Q.append(args.q_alpha)
    if sim_trend_mode == "dynamic": true_Q.append(args.q_beta)
    if sim_season_mode == "dynamic": true_Q += [args.q_gamma] + [0.0] * (args.period - 2)
    sampler.set_truth(sigma=args.true_sigma, xi=args.true_xi, Q=(np.asarray(true_Q, float) if len(true_Q) else None))
    sampler.set_truth_paths(mu=mu_T)

    tag = f"{sim_level_mode}-{sim_trend_mode}-{sim_season_mode}"
    out_dir = args.out_dir or os.path.join("results", "simulations", "DGEV",
                                           f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    fig_dir = os.path.join(out_dir, "figures")
    _ensure_dir(out_dir); _ensure_dir(fig_dir)

    t0 = time.time()
    posterior = sampler.run()
    elapsed = time.time() - t0
    print(f"Sampler run time: {elapsed:.2f}s")

    sampler.save_posterior(
        out_npz_path=os.path.join(out_dir, "posterior.npz"),
        extra_meta={"modes": tag, "elapsed_seconds": float(elapsed)},
    )

    # ---- Summaries ----
    print(f"Posterior mean sigma: {np.mean(posterior['sigma']):.3f} (true {args.true_sigma})")
    print(f"Posterior mean xi:    {np.mean(posterior['xi']):.3f} (true {args.true_xi})")

    if "Q" in posterior and posterior["Q"].size > 0:
        qnames = []
        if sampler.idx_alpha is not None: qnames.append("Q_alpha")
        if sampler.idx_beta  is not None: qnames.append("Q_beta")
        if sampler.seasonal_mode == "dynamic": qnames.append("Q_gamma(last)")
        qm = posterior["Q"].mean(axis=0)
        for name, val in zip(qnames, qm):
            print(f"Posterior mean {name}: {val:.6g}")

    if "log_evidence" in posterior and posterior["log_evidence"].size > 0:
        le = posterior["log_evidence"]
        print(f"Mean log p(y|theta): {np.nanmean(le):.3f} | Median: {np.nanmedian(le):.3f} | Best: {np.nanmax(le):.3f}")

    # Optional plotting (unchanged – hook up if you like)
    if not args.no_plots and DGEVPlotter is not None and args.show_plots:
        try:
            plotter = DGEVPlotter()
            # add any quick overviews here
        except Exception as e:
            print("[plot] skipped:", e)
