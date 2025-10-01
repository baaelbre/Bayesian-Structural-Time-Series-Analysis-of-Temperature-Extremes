# %% optimization/dgev_pgbs.py
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

def gev_loglike_sum(y: np.ndarray, mu_vec: np.ndarray, sigma: float, xi: float) -> float:
    """Sum_t log f(y_t | mu_t, sigma, xi)."""
    if sigma <= 0.0 or np.any(np.isnan(mu_vec)):
        return -np.inf
    z = (y - mu_vec) / sigma
    u = 1.0 + xi * z
    if np.any(u <= 0.0):
        return -np.inf
    if abs(xi) < 1e-8:
        return float(np.sum(-np.log(sigma) - np.exp(-z) - z))
    return float(np.sum(-np.log(sigma) - (1.0 + 1.0 / xi) * np.log(u) - u ** (-1.0 / xi)))

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
    a_q_alpha: float = 1.1
    b_q_alpha: float = 1e-4     # E[Q_alpha] ≈ 1e-3
    a_q_beta:  float = 1.1
    b_q_beta:  float = 1e-12    # E[Q_beta]  ≈ 1e-11
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

    # RW–MH step sizes (initial values)
    step_logsigma: float = 0.05
    step_xi: float = 0.05
    step_level: float = 0.05
    step_slope: float = 0.02
    step_season: float = 0.05

    # PMMH step for log Q (shared across components for simplicity)
    step_logQ: float = 0.4

    # PF
    n_particles: int = 200
    trans_eps: float = 1e-12
    random_seed: Optional[int] = 123
    progress: bool = True
    progress_every: int = 0  # 0 => auto (~2% of n_iter)

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
# Particle Gibbs with Ancestor Sampling (PGAS) + APF + PMMH(log Q)
# =============================================================================

class DGEVParticleGibbs:
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
        self.idx_beta = layout.index("beta") if "beta" in layout else None
        if self.seasonal_mode == "dynamic":
            self.idx_gamma_start = layout.index("gamma_1") if "gamma_1" in layout else None
            self.idx_gamma_end = self.idx_gamma_start + (self.period - 2) if self.idx_gamma_start is not None else None
        else:
            self.idx_gamma_start = None
            self.idx_gamma_end = None

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

        # ---- Observation params
        self.logsigma = float(self.priors.m_sigma)   # prior mean on log-scale
        self.sigma    = float(np.exp(self.logsigma))
        self.xi       = float(self.priors.m_xi)

        # ---- Innovation variances Q (for dynamics)
        self.Q = np.zeros(self.dim, float)
        if self.idx_alpha is not None:
            self.Q[self.idx_alpha] = self.priors.b_q_alpha / (self.priors.a_q_alpha - 1.0)
        if self.idx_beta is not None:
            self.Q[self.idx_beta]  = self.priors.b_q_beta  / (self.priors.a_q_beta  - 1.0)
        if self.seasonal_mode == "dynamic":
            self.Q[self.idx_gamma_end] = self.priors.b_q_gamma / (self.priors.a_q_gamma - 1.0)

        # ---- Initial latent path x_{0:T}
        self.x = np.zeros((self.T + 1, self.dim), float)
        if self.seasonal_mode == "dynamic":
            m0_season_arr = np.zeros(self.period - 1) if m0_season is None else np.asarray(m0_season, float)
            v0_season_arr = np.ones(self.period - 1)  if v0_season is None else np.asarray(v0_season, float)
            if m0_season_arr.size != self.period - 1 or v0_season_arr.size != self.period - 1:
                raise ValueError("m0_season and v0_season must have length p-1 in dynamic mode.")

        m0_list, v0_list = [], []
        for tag in layout:
            if tag == "alpha":
                m0_list.append(float(m0_level)); v0_list.append(float(v0_level))
            elif tag == "beta":
                m0_list.append(float(m0_trend)); v0_list.append(float(v0_trend))
            else:
                k = int(tag.split("_")[1]) - 1
                m0_list.append(float(m0_season_arr[k]))
                v0_list.append(float(v0_season_arr[k]))
        if self.dim > 0:
            self.x[0] = np.random.normal(np.array(m0_list), np.sqrt(np.array(v0_list)))
            self._propagate_initial_path(Q_init=np.ones(self.dim) * 1e-6)

        # ---- Storage (filled after knowing n_kept in run())
        self.keep: Dict[str, np.ndarray] = {}

        # ---- MH bookkeeping
        self.accept = {"logsigma": 0, "xi": 0, "level": 0, "slope": 0, "season": 0}
        self.proposals = {"logsigma": 0, "xi": 0, "level": 0, "slope": 0, "season": 0}

        # PMMH acceptance for Q components
        self.accept_Q: Dict[str,int] = {"alpha":0, "beta":0, "gamma_last":0}
        self.proposals_Q: Dict[str,int] = {"alpha":0, "beta":0, "gamma_last":0}

        # ---- adaptation bookkeeping (windowed deltas)
        self._mh_prev_acc = dict(self.accept)
        self._mh_prev_prop = dict(self.proposals)
        self._adapt_round = 0

        # ---- Truth overlays (optional)
        self.true_sigma: Optional[float] = None
        self.true_xi: Optional[float] = None
        self.true_Q: Optional[np.ndarray] = None
        self.true_mu_t: Optional[np.ndarray] = None
        self.true_alpha_t: Optional[np.ndarray] = None
        self.true_beta_t: Optional[np.ndarray] = None
        self.true_gamma_t: Optional[np.ndarray] = None

        # ---- PF diagnostics
        self.last_log_evidence: float = float("nan")
        self.last_pf_diag: Dict[str, float] = {}

    # --------------------- Truth registration (optional) --------------------- #
    def set_truth(self, sigma: Optional[float] = None, xi: Optional[float] = None, Q: Optional[np.ndarray] = None) -> None:
        self.true_sigma = sigma
        self.true_xi = xi
        self.true_Q = None if Q is None else np.asarray(Q, float)

    def set_truth_paths(self, mu: Optional[np.ndarray] = None,
                        alpha: Optional[np.ndarray] = None,
                        beta: Optional[np.ndarray] = None,
                        gamma: Optional[np.ndarray] = None) -> None:
        self.true_mu_t = None if mu is None else np.asarray(mu, float)
        self.true_alpha_t = None if alpha is None else np.asarray(alpha, float)
        self.true_beta_t = None if beta is None else np.asarray(beta, float)
        self.true_gamma_t = None if gamma is None else np.asarray(gamma, float)

    # ----------------------- Helpers: norm / ESS / EMA ----------------------- #
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

    @staticmethod
    def _ema(old: Optional[float], new: float, alpha: float = 0.1) -> float:
        return alpha * new + (1.0 - alpha) * (0.0 if old is None else old)

    def _mh_accept(self, logacc: float) -> bool:
        return (np.log(np.random.rand()) < min(0.0, logacc))

    # ----------------------------- State model ------------------------------ #
    def _state_mean(self, x_prev: np.ndarray, t: int) -> np.ndarray:
        m = np.zeros_like(x_prev)

        # alpha (local level) with drift from beta or deterministic slope
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

        # dynamic seasonal (shift + closure to sum-zero)
        if self.seasonal_mode == "dynamic":
            g0, gL = self.idx_gamma_start, self.idx_gamma_end
            if g0 <= gL - 1:
                m[g0:gL] = x_prev[g0 + 1 : gL + 1]
            prev_gamma = x_prev[g0 : gL + 1]
            m[gL] = -np.sum(prev_gamma)
        return m

    def _alpha_contribution(self, x_t: np.ndarray, t: int) -> float:
        if self.level_mode == "dynamic":
            return float(x_t[self.idx_alpha]) if self.idx_alpha is not None else 0.0
        base = self.level_value
        if self.idx_beta is not None:      # dynamic trend
            return float(base + x_t[self.idx_beta] * t)
        if self.trend_mode == "deterministic":
            return float(base + self.slope_value * t)
        return float(base)

    def _season_contribution(self, x_t: np.ndarray, t: int) -> float:
        if self.seasonal_mode == "dynamic":
            return float(x_t[self.idx_gamma_end]) if self.idx_gamma_end is not None else 0.0
        if self.seasonal_mode == "deterministic":
            return float(self.season_vec[t % self.period])
        return 0.0

    def mu_from_state(self, x_t: np.ndarray, t: int) -> float:
        return self._alpha_contribution(x_t, t) + self._season_contribution(x_t, t)

    def _transition_logpdf(self, x_prev: np.ndarray, x_cur: np.ndarray, t: int) -> float:
        mean = self._state_mean(x_prev, t)
        out = 0.0
        for k in range(self.dim):
            var = self.Q[k] if self.Q[k] > 0.0 else self.cfg.trans_eps
            diff = x_cur[k] - mean[k]
            out += -0.5 * (math.log(2.0 * math.pi * var) + (diff * diff) / var)
        return float(out)

    def _transition_sample(self, x_prev: np.ndarray, t: int) -> np.ndarray:
        mean = self._state_mean(x_prev, t)
        var = np.where(self.Q > 0.0, self.Q, self.cfg.trans_eps)
        return mean + np.random.normal(0.0, np.sqrt(var), size=self.dim)

    def _propagate_initial_path(self, Q_init: np.ndarray) -> None:
        for t in range(1, self.T + 1):
            if self.dim == 0:
                break
            mean = self._state_mean(self.x[t - 1], t)
            self.x[t] = mean + np.random.normal(0.0, np.sqrt(Q_init), size=self.dim)

    # ----------------------- Static parameter updates ----------------------- #
    def _mu_vec_current(self) -> np.ndarray:
        return np.array([self.mu_from_state(self.x[t], t - 1) for t in range(1, self.T + 1)], float)

    # --------- PMMH on log Q: prior, unconditional APF evidence, kernel ----- #
    def _logprior_logQ(self, logQ: float, a: float, b: float) -> float:
        """
        If Q ~ IG(a,b) (density ∝ b^a / Γ(a) * Q^{-(a+1)} exp(-b/Q)),
        and we sample on logQ = log Q, the log prior up to a constant is:
          log p(logQ) = [-(a+1) log Q - b/Q] + log|dQ/dlogQ| = [-(a+1) log Q - b/Q] + log Q
                      = -a * log Q - b * exp(-logQ)
        """
        Q = np.exp(logQ)
        if not np.isfinite(Q) or Q <= 0.0:
            return -np.inf
        return -a * logQ - b / Q

    def _apf_predictive_loglik(self, y_t: float, x_prev: np.ndarray, t: int) -> float:
        """Cheap proxy m(y_t | x_{t-1}) used by APF."""
        if self.dim == 0:
            mu_hat = self._alpha_contribution(np.zeros(0), t-1) + self._season_contribution(np.zeros(0), t-1)
        else:
            x_hat = self._state_mean(x_prev, t)
            mu_hat = self.mu_from_state(x_hat, t-1)
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

    def _apf_log_evidence_unconditional(self) -> Tuple[float, dict]:
        """
        Unconditional APF (no reference path) to get an unbiased estimate of log p(y|theta).
        Used inside PMMH for Q updates.
        """
        N, T, D = self.cfg.n_particles, self.T, self.dim
        if N <= 1:
            raise ValueError("n_particles must be > 1 for APF evidence.")
        # initialize x0 population at current x0 (or prior if you like)
        if D > 0:
            x_prev = np.tile(self.x[0], (N, 1))
        else:
            x_prev = np.zeros((N, 0))
        w_prev = np.full(N, 1.0 / N, float)
        logZ = 0.0
        ess_list, maxw_list = [], []

        for t in range(1, T + 1):
            y_t = self.y[t - 1]
            # look-ahead weights
            look = np.array([self._apf_predictive_loglik(y_t, x_prev[j], t) for j in range(N)], float)
            rescore = np.log(w_prev + 1e-300) + (look - np.max(look))
            resw = self._safe_normalize(np.exp(rescore))

            # resample ancestors
            a_idx = self._resample_systematic(resw)
            x_anc = x_prev[a_idx]

            # propagate
            if D > 0:
                x_cur = np.array([self._transition_sample(x_anc[j], t) for j in range(N)], float)
            else:
                x_cur = np.zeros((N, 0))

            # corrected weights
            lw = np.array(
                [gev_logpdf(y_t, self.mu_from_state(x_cur[j], t-1) if D>0 else 0.0, self.sigma, self.xi)
                 for j in range(N)], float)
            lw_corr = lw - np.array([self._apf_predictive_loglik(y_t, x_anc[j], t) for j in range(N)], float)

            lw_max = np.max(lw_corr)
            w_cur = self._safe_normalize(np.exp(lw_corr - lw_max))
            logZ += lw_max + math.log(np.mean(np.exp(lw_corr - lw_max)) + 1e-300)

            ess_list.append(self._ess(w_cur))
            maxw_list.append(float(np.max(w_cur)))

            # prepare next step
            x_prev = x_cur
            w_prev = w_cur

        pf_diag = {
            "ess_mean": float(np.mean(ess_list)) if ess_list else float("nan"),
            "ess_min": float(np.min(ess_list)) if ess_list else float("nan"),
            "maxw_mean": float(np.mean(maxw_list)) if maxw_list else float("nan"),
            "maxw_max": float(np.max(maxw_list)) if maxw_list else float("nan"),
        }
        return float(logZ), pf_diag

    def pmmh_single_logQ(self, idx: int, a_hyp: float, b_hyp: float, label: str) -> bool:
        """
        PMMH move on log Q[idx]. Uses unconditional APF to estimate marginal likelihood.
        Returns True if accepted.
        """
        step = self.cfg.step_logQ
        logQ_cur = math.log(max(self.Q[idx], 1e-20))
        # current log evidence
        logZ_cur, _ = self._apf_log_evidence_unconditional()
        lp_cur = self._logprior_logQ(logQ_cur, a_hyp, b_hyp)

        # propose
        logQ_prop = logQ_cur + np.random.normal(0.0, step)
        Q_prop = float(np.exp(logQ_prop))
        if not np.isfinite(Q_prop) or Q_prop <= 0.0:
            self.proposals_Q[label] += 1
            return False

        # temporarily set and evaluate evidence
        Q_backup = float(self.Q[idx])
        self.Q[idx] = Q_prop
        logZ_prop, _ = self._apf_log_evidence_unconditional()
        lp_prop = self._logprior_logQ(logQ_prop, a_hyp, b_hyp)

        # MH ratio (proposal symmetric in log-space)
        logacc = (logZ_prop + lp_prop) - (logZ_cur + lp_cur)
        self.proposals_Q[label] += 1
        if self._mh_accept(logacc):
            # keep proposed Q; states will be refreshed by PGAS later in the iteration
            self.accept_Q[label] += 1
            return True
        else:
            # revert
            self.Q[idx] = Q_backup
            return False

    # ---------------- PGAS + APF (conditional SMC with ancestor sampling) --- #
    def _conditional_apf_with_AS(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
        """
        Conditional APF with ancestor sampling (PGAS core).
        Returns (parts, ancestors, weights, logZ, pf_diag).
        parts: (T+1, N, D); ancestors: (T+1, N); weights: (T+1, N).
        """
        N, T, D = self.cfg.n_particles, self.T, self.dim
        parts = np.zeros((T + 1, N, D), float) if D > 0 else np.zeros((T + 1, N, 0), float)
        w = np.zeros((T + 1, N), float)
        a = np.zeros((T + 1, N), int)
        logZ = 0.0

        ess_list, maxw_list = [], []

        if D > 0:
            parts[0, :, :] = self.x[0]

        # ---------- t = 1 ----------
        y0 = self.y[0]
        look = np.zeros(N)
        for j in range(N):
            x_prev = parts[0, j, :] if D > 0 else np.zeros(0)
            look[j] = self._apf_predictive_loglik(y0, x_prev, t=1)
        resw = self._safe_normalize(np.exp(look - np.max(look)))

        anc = self._resample_systematic(resw)
        a[1, :N-1] = anc[:N-1]
        if D > 0:
            for m in range(N - 1):
                parts[1, m, :] = self._transition_sample(parts[0, a[1, m], :], t=1)

            # ancestor sampling for reference particle
            x_ref_1 = self.x[1].copy()
            log_as = np.zeros(N)
            for j in range(N):
                lp_f = self._transition_logpdf(parts[0, j, :], x_ref_1, t=1)
                log_as[j] = np.log(resw[j] + 1e-300) + lp_f
            log_as -= np.max(log_as)
            p_as = self._safe_normalize(np.exp(log_as))
            a[1, N - 1] = np.random.choice(N, p=p_as)
            parts[1, N - 1, :] = x_ref_1
        else:
            a[1, :] = 0  # dummy

        lw = np.zeros(N)
        for i in range(N):
            mu = self.mu_from_state(parts[1, i, :], t=0) if D > 0 else 0.0
            lw[i] = gev_logpdf(y0, mu, self.sigma, self.xi)
        lw_corr = lw.copy()
        for i in range(N):
            j = a[1, i]
            x_prev = parts[0, j, :] if D > 0 else np.zeros(0)
            lw_corr[i] -= self._apf_predictive_loglik(y0, x_prev, t=1)
        lw_max = np.max(lw_corr)
        w[1, :] = self._safe_normalize(np.exp(lw_corr - lw_max))
        logZ += lw_max + math.log(np.mean(np.exp(lw_corr - lw_max)) + 1e-300)
        ess_list.append(self._ess(w[1, :]))
        maxw_list.append(float(np.max(w[1, :])))

        # ---------- t = 2..T ----------
        it = tqdm(range(2, T + 1)) if self.cfg.progress else range(2, T + 1)
        for t in it:
            y_t = self.y[t - 1]
            look = np.zeros(N)
            for j in range(N):
                x_prev = parts[t - 1, j, :] if D > 0 else np.zeros(0)
                look[j] = self._apf_predictive_loglik(y_t, x_prev, t=t)
            prew = w[t - 1, :]
            rescore = np.log(prew + 1e-300) + (look - np.max(look))
            resw = self._safe_normalize(np.exp(rescore))

            anc = self._resample_systematic(resw)
            a[t, :N-1] = anc[:N-1]

            if D > 0:
                for m in range(N - 1):
                    parts[t, m, :] = self._transition_sample(parts[t - 1, a[t, m], :], t=t)

                x_ref_t = self.x[t].copy()
                log_as = np.zeros(N)
                for j in range(N):
                    lp_f = self._transition_logpdf(parts[t - 1, j, :], x_ref_t, t=t)
                    log_as[j] = np.log(w[t - 1, j] + 1e-300) + look[j] + lp_f
                log_as -= np.max(log_as)
                p_as = self._safe_normalize(np.exp(log_as))
                a[t, N - 1] = np.random.choice(N, p=p_as)
                parts[t, N - 1, :] = x_ref_t
            else:
                a[t, :] = a[t - 1, :]

            lw = np.zeros(N)
            for i in range(N):
                mu = self.mu_from_state(parts[t, i, :], t=t-1) if D > 0 else 0.0
                lw[i] = gev_logpdf(y_t, mu, self.sigma, self.xi)
            lw_corr = lw.copy()
            for i in range(N):
                j = a[t, i]
                x_prev = parts[t - 1, j, :] if D > 0 else np.zeros(0)
                lw_corr[i] -= self._apf_predictive_loglik(y_t, x_prev, t=t)

            lw_max = np.max(lw_corr)
            w[t, :] = self._safe_normalize(np.exp(lw_corr - lw_max))
            logZ += lw_max + math.log(np.mean(np.exp(lw_corr - lw_max)) + 1e-300)

            ess_t = self._ess(w[t, :])
            maxw_t = float(np.max(w[t, :]))
            ess_list.append(ess_t)
            maxw_list.append(maxw_t)
            if self.cfg.progress and hasattr(it, "set_postfix"):
                it.set_postfix(ESS=f"{ess_t:6.1f}", MaxW=f"{maxw_t:7.4f}")

        pf_diag = {
            "ess_mean": float(np.mean(ess_list)),
            "ess_min": float(np.min(ess_list)),
            "maxw_mean": float(np.mean(maxw_list)),
            "maxw_max": float(np.max(maxw_list)),
        }
        return parts, a, w, float(logZ), pf_diag

    def update_states_pgas(self) -> None:
        """PGAS sweep using conditional APF with ancestor sampling."""
        parts, a, w, logZ, pf_diag = self._conditional_apf_with_AS()
        N, T = self.cfg.n_particles, self.T
        idx = np.zeros(T + 1, dtype=int)
        idx[T] = np.random.choice(N, p=w[T, :])
        for t in range(T, 0, -1):
            idx[t - 1] = a[t, idx[t]]
        if self.dim > 0:
            for t in range(1, T + 1):
                self.x[t, :] = parts[t, idx[t], :]
        self.last_log_evidence = float(logZ)
        self.last_pf_diag = pf_diag

    # --------------------------- Progress helpers --------------------------- #
    def _fmt_acc(self, key: str) -> str:
        a, p = self.accept[key], self.proposals[key]
        pct = (100.0 * a / p) if p > 0 else 0.0
        return f"{a}/{p} ({pct:4.1f}%)"

    def _fmt_acc_q(self, label: str) -> str:
        a, p = self.accept_Q[label], self.proposals_Q[label]
        pct = (100.0 * a / p) if p > 0 else 0.0
        return f"{a}/{p} ({pct:4.1f}%)"

    def _q_snapshot(self, ema_Q: np.ndarray | None = None) -> str:
        rows = []
        def add(label, idx):
            if idx is None:
                return
            q = float(self.Q[idx]); logq = np.log10(max(q, 1e-20))
            if ema_Q is not None:
                ema = float(ema_Q[idx]); logema = np.log10(max(ema, 1e-20))
                rows.append(f"{label}: cur={logq:6.2f} ema={logema:6.2f} acc={self._fmt_acc_q(label)}")
            else:
                rows.append(f"{label}: cur={logq:6.2f} acc={self._fmt_acc_q(label)}")
        add("alpha", self.idx_alpha)
        add("beta", self.idx_beta)
        if self.seasonal_mode == "dynamic":
            add("gamma_last", self.idx_gamma_end)
        return (" | " + " | ".join(rows)) if rows else ""

    # --------------------------------- MCMC --------------------------------- #
    def run(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg

        save_iters = list(range(cfg.burn, cfg.n_iter, cfg.thin))
        n_kept = max(0, len(save_iters))

        keep_idx = 0
        self.keep = {
            "sigma": np.zeros(n_kept, float),
            "xi":    np.zeros(n_kept, float),
            "mu":    np.zeros((n_kept, self.T), float),
            "log_evidence": np.zeros(n_kept, float),
        }
        if self.dim > 0:
            self.keep["Q"] = np.zeros((n_kept, self.dim), float)
        if self.idx_alpha is not None:
            self.keep["alpha_t"] = np.zeros((n_kept, self.T), float)
        if self.idx_beta is not None:
            self.keep["beta_t"] = np.zeros((n_kept, self.T), float)
        if self.seasonal_mode == "dynamic":
            self.keep["gamma_t"] = np.zeros((n_kept, self.T), float)
        if self.level_mode == "deterministic":
            self.keep["level_value"] = np.zeros(n_kept, float)
        if self.trend_mode == "deterministic":
            self.keep["slope_value"] = np.zeros(n_kept, float)
        if self.seasonal_mode == "deterministic":
            self.keep["season_vector"] = np.zeros((n_kept, self.period), float)

        ema_logZ: Optional[float] = None
        ema_Q = np.zeros(self.dim, float) if self.dim > 0 else None
        print_every = cfg.progress_every if cfg.progress_every > 0 else max(1, cfg.n_iter // 50)

        for it in range(cfg.n_iter):
            if cfg.progress:
                print(f"Iteration {it + 1}/{cfg.n_iter}")

            # 1) PMMH on Q (log-space), then refresh states by PGAS
            if self.dim > 0:
                if cfg.progress: print("  PMMH on log Q ...")
                # alpha
                if self.idx_alpha is not None:
                    self.pmmh_single_logQ(self.idx_alpha, self.priors.a_q_alpha, self.priors.b_q_alpha, "alpha")
                # beta
                if self.idx_beta is not None:
                    self.pmmh_single_logQ(self.idx_beta,  self.priors.a_q_beta,  self.priors.b_q_beta,  "beta")
                # seasonal (last coord)
                if self.seasonal_mode == "dynamic" and self.idx_gamma_end is not None:
                    self.pmmh_single_logQ(self.idx_gamma_end, self.priors.a_q_gamma, self.priors.b_q_gamma, "gamma_last")

            # 2) latent states via PGAS + APF (given new Q)
            if self.dim > 0:
                self.update_states_pgas()
                current_log_ev = float(self.last_log_evidence)
            else:
                mu_vec_now = self._mu_vec_current()
                current_log_ev = float(gev_loglike_sum(self.y, mu_vec_now, self.sigma, self.xi))

            ema_logZ = self._ema(ema_logZ, current_log_ev, alpha=0.1)
            if cfg.progress:
                print(f"  log p(y | theta) [APF + PGAS] = {current_log_ev:.6f}")

            # 3) deterministic structural params
            if self.level_mode == "deterministic":
                if cfg.progress: print("  Updating deterministic level...")
                self.update_level_value()
            if self.trend_mode == "deterministic":
                if cfg.progress: print("  Updating deterministic slope...")
                self.update_slope()
            if self.seasonal_mode == "deterministic":
                if cfg.progress: print("  Updating deterministic seasonality...")
                self.update_season_vec()

            # 4) observation params
            if cfg.progress: print("  Updating log(sigma) / xi ...")
            self.update_logsigma()
            self.update_xi()

            # --- adapt proposal scales if enabled (not wired to Q by default) ---
            self._adapt_steps(it)

            # 5) compact progress line
            if cfg.progress and ((it + 1) % print_every == 0 or it == cfg.n_iter - 1):
                acc_logs = self._fmt_acc('logsigma')
                acc_xi   = self._fmt_acc('xi')
                det_info = ""
                if self.level_mode == "deterministic":
                    det_info += f" | acc(level)={self._fmt_acc('level')} value={self.level_value:.4f}"
                if self.trend_mode == "deterministic":
                    det_info += f" | acc(slope)={self._fmt_acc('slope')} value={self.slope_value:.4f}"
                if self.seasonal_mode == "deterministic":
                    preview = np.array2string(self.season_vec[:min(3, self.period)], precision=3, separator=",")
                    det_info += f" | acc(season)={self._fmt_acc('season')} γ[:3]={preview} ..."
                q_info = self._q_snapshot(ema_Q) if self.dim > 0 else ""
                pf_info = ""
                if self.last_pf_diag:
                    d = self.last_pf_diag
                    pf_info = f" | PF: ESS(mean/min)={d['ess_mean']:.1f}/{d['ess_min']:.1f} MaxW(max)={d['maxw_max']:.4f}"
                print(
                    f"[it {it+1}/{cfg.n_iter}] "
                    f"logZ={current_log_ev:.3f} ema={ema_logZ:.3f} "
                    f"σ={np.exp(self.logsigma):.3f} ξ={self.xi:.3f} "
                    f"acc(logσ)={acc_logs} acc(ξ)={acc_xi}"
                    f"{det_info}{q_info}{pf_info}"
                )

            # 6) store
            if it in save_iters and keep_idx < n_kept:
                mu_vec = self._mu_vec_current()
                self.keep["mu"][keep_idx, :] = mu_vec
                self.keep["sigma"][keep_idx] = float(np.exp(self.logsigma))
                self.keep["xi"][keep_idx] = float(self.xi)
                self.keep["log_evidence"][keep_idx] = current_log_ev
                if self.dim > 0:
                    self.keep["Q"][keep_idx, :] = self.Q
                if self.idx_alpha is not None:
                    self.keep["alpha_t"][keep_idx, :] = self.x[1:self.T + 1, self.idx_alpha]
                if self.idx_beta is not None:
                    self.keep["beta_t"][keep_idx, :] = self.x[1:self.T + 1, self.idx_beta]
                if self.seasonal_mode == "dynamic":
                    self.keep["gamma_t"][keep_idx, :] = self.x[1:self.T + 1, self.idx_gamma_end]
                if self.level_mode == "deterministic":
                    self.keep["level_value"][keep_idx] = self.level_value
                if self.trend_mode == "deterministic":
                    self.keep["slope_value"][keep_idx] = self.slope_value
                if self.seasonal_mode == "deterministic":
                    self.keep["season_vector"][keep_idx, :] = self.season_vec
                keep_idx += 1

        if cfg.progress:
            print(
                f"[{it + 1}/{cfg.n_iter}] "
                f"acc(logsigma)={self._fmt_acc('logsigma')} "
                f"acc(xi)={self._fmt_acc('xi')} "
                f"acc(level)={self._fmt_acc('level')} "
                f"acc(slope)={self._fmt_acc('slope')} "
                f"acc(season)={self._fmt_acc('season')}"
            )
            print(
                f"PMMH Q accept: "
                f"alpha={self._fmt_acc_q('alpha')} "
                f"beta={self._fmt_acc_q('beta')} "
                f"gamma_last={self._fmt_acc_q('gamma_last')}"
            )
        return self.keep

    # ------------------------------- Persistence ---------------------------- #
    def save_posterior(self, out_npz_path: str, extra_meta: Optional[dict] = None) -> None:
        _ensure_dir(os.path.dirname(out_npz_path))

        arrays = dict(self.keep)
        arrays["y"] = self.y.copy()
        arrays["x_last"] = self.x[1 : self.T + 1].copy() if self.dim > 0 else np.zeros((self.T, 0))

        if self.true_mu_t is not None: arrays["true_mu_t"] = np.asarray(self.true_mu_t, float)
        if self.true_alpha_t is not None: arrays["true_alpha_t"] = np.asarray(self.true_alpha_t, float)
        if self.true_beta_t is not None: arrays["true_beta_t"] = np.asarray(self.true_beta_t, float)
        if self.true_gamma_t is not None: arrays["true_gamma_t"] = np.asarray(self.true_gamma_t, float)

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
            "accept": self.accept,
            "proposals": self.proposals,
            "accept_Q": self.accept_Q,
            "proposals_Q": self.proposals_Q,
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
    from simulator.dgev_plotter import DGEVPlotter  # optional

    parser = argparse.ArgumentParser(description="DGEV PGAS + APF + PMMH(log Q)")
    # Modes
    parser.add_argument("--level-mode", choices=["dynamic", "deterministic"], default="dynamic")
    parser.add_argument("--trend-mode", choices=["dynamic", "deterministic", "none"], default="dynamic")
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
    parser.add_argument("--prior-bq-alpha", type=float, default=5e-5)
    parser.add_argument("--prior-bq-beta", type=float, default=5e-5)
    parser.add_argument("--prior-bq-gamma", type=float, default=5e-5)
    parser.add_argument("--prior-m-level", type=float, default=0.0)
    parser.add_argument("--prior-s-level", type=float, default=10.0)
    parser.add_argument("--prior-m-slope", type=float, default=0.0)
    parser.add_argument("--prior-s-slope", type=float, default=10.0)
    parser.add_argument("--prior-m-season", type=str, default=None,
                        help="Comma-separated first (p-1) means for deterministic seasonal prior (e.g. '0,0,0').")
    parser.add_argument("--prior-s-season", type=float, default=5.0)

    # Sampler config
    parser.add_argument("--n-iter", type=int, default=2000)
    parser.add_argument("--burn", type=int, default=100)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--step-logsigma", type=float, default=0.2)
    parser.add_argument("--step-xi", type=float, default=0.2)
    parser.add_argument("--step-level", type=float, default=0.02)
    parser.add_argument("--step-slope", type=float, default=0.0005)
    parser.add_argument("--step-season", type=float, default=0.02)
    parser.add_argument("--step-logQ", type=float, default=0.4)
    parser.add_argument("--particles", type=int, default=250)
    parser.add_argument("--trans-eps", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--progress", default=True)
    parser.add_argument("--progress-every", type=int, default=10,
                        help="print compact summary every k iterations (0=auto)")

    # Adaptive RW–MH CLI (not wired to Q here)
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
    mu_T    = np.asarray(truths["mu"][1:1 + args.T], float)
    alpha_T = np.asarray(truths["alpha"][1:1 + args.T], float) if sim_level_mode == "dynamic" else None
    beta_T  = np.asarray(truths["beta"][1:1 + args.T], float)  if sim_trend_mode == "dynamic" else None
    gamma_T = np.asarray(truths["gamma_last"][1:1 + args.T], float) if sim_season_mode == "dynamic" else None

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

    prior_mean_Q_alpha = priors.b_q_alpha / (priors.a_q_alpha - 1.0) if priors.a_q_alpha > 1.0 else float("inf")
    prior_var_Q_alpha  = (priors.b_q_alpha**2 / ((priors.a_q_alpha - 1.0)**2 * (priors.a_q_alpha - 2.0))
                    if priors.a_q_alpha > 2.0 else float("inf"))
    print(f"[info] Prior on Q_alpha has mean={prior_mean_Q_alpha:.6g} and variance={prior_var_Q_alpha:.6g}.")

    cfg = SamplerConfig(
        n_iter=args.n_iter, burn=args.burn, thin=args.thin,
        step_logsigma=args.step_logsigma, step_xi=args.step_xi,
        step_level=args.step_level, step_slope=args.step_slope, step_season=args.step_season,
        step_logQ=args.step_logQ,
        n_particles=args.particles, trans_eps=args.trans_eps,
        random_seed=args.seed, progress=bool(args.progress),
        progress_every=int(args.progress_every),
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

    sampler = DGEVParticleGibbs(
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
    sampler.set_truth(sigma=args.true_sigma, xi=args.true_xi,
                      Q=(np.asarray(true_Q, float) if len(true_Q) else None))
    sampler.set_truth_paths(mu=mu_T, alpha=alpha_T, beta=beta_T, gamma=gamma_T)

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

    if sampler.true_Q is not None:
        tq = np.asarray(sampler.true_Q, float)
        if sampler.idx_alpha is not None and tq.size > sampler.idx_alpha:
            print(f"True Q_alpha:         {tq[sampler.idx_alpha]:.6g}")
        if sampler.idx_beta is not None and tq.size > sampler.idx_beta:
            print(f"True Q_beta:          {tq[sampler.idx_beta]:.6g}")
        if sampler.seasonal_mode == "dynamic" and sampler.idx_gamma_end is not None and tq.size > sampler.idx_gamma_end:
            print(f"True Q_gamma(last):   {tq[sampler.idx_gamma_end]:.6g}")

    if "Q" in posterior and sampler.dim > 0 and posterior["Q"].size > 0:
        if sampler.idx_alpha is not None:
            print(f"Posterior mean Q_alpha: {np.mean(posterior['Q'][:, sampler.idx_alpha]):.6g}")
        if sampler.idx_beta is not None:
            print(f"Posterior mean Q_beta:  {np.mean(posterior['Q'][:, sampler.idx_beta]):.6g}")
        if sampler.seasonal_mode == "dynamic":
            print(f"Posterior mean Q_gamma: {np.mean(posterior['Q'][:, sampler.idx_gamma_end]):.6g}")

    if "log_evidence" in posterior and posterior["log_evidence"].size > 0:
        le = posterior["log_evidence"]
        print(f"Mean log p(y|theta): {np.nanmean(le):.3f} | Median: {np.nanmedian(le):.3f} | Best: {np.nanmax(le):.3f}")

    if not args.no_plots:
        try:
            plotter = DGEVPlotter()
        except Exception as e:
            print("[plot] skipped:", e)
