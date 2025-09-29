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

def default_season_pminus1(period: int) -> np.ndarray:
    """
    Build a smooth default for the first (p-1) seasonal entries (last is implied
    by the sum-to-zero constraint). We center a cosine over p points and
    return the first p-1 entries.
    """
    g = np.cos(2 * np.pi * np.arange(period) / period)
    g = g - np.mean(g)
    return g[: period - 1].astype(float)

def parse_csv_floats(s: Optional[str]) -> Optional[List[float]]:
    """
    Parse a comma-separated string of floats (e.g. "0,0.2,-0.1") into a list.
    Returns None if s is None or empty.
    """
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
    """
    Log-pdf of GEV(y | mu, sigma, xi) using (mu, sigma>0, xi).
    """
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
    """
    Sum of log-pdfs of GEV(y[t] | mu_vec[t], sigma, xi) for t=0..T-1.
    """
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
    # Observation (NOTE: m_sigma is a prior mean on log-sigma)
    m_sigma: float = 0.0
    s_sigma: float = 10.0
    m_xi: float = 0.0
    s_xi: float = 1.0

    # Innovation variances Q ~ IG(a_q, b_q) for dynamic coords
    # (shape a_q, scale b_q; mean = b_q/(a_q - 1) for a_q > 1)
    a_q: float = 1.5
    b_q: float = 5e-6

    # Deterministic level ~ N(m_level, s_level^2) if level_mode='deterministic'
    m_level: float = 0.0
    s_level: float = 10.0

    # Deterministic slope ~ N(m_slope, s_slope^2) if trend_mode='deterministic'
    m_slope: float = 0.0
    s_slope: float = 10.0

    # Deterministic seasonal prior for the first (p-1) entries (last implied).
    # If provided, len(m_season) must equal period-1 at runtime.
    m_season: Optional[Sequence[float]] = None
    s_season: float = 5.0

@dataclass
class SamplerConfig:
    n_iter: int = 5000
    burn: int = 1000
    thin: int = 5

    # RW–MH steps for static params
    step_logsigma: float = 0.05
    step_xi: float = 0.05
    step_level: float = 0.05
    step_slope: float = 0.02
    step_season: float = 0.05  # jointly for the first (p-1) entries

    # Particle filter (used only if at least one component is dynamic)
    n_particles: int = 200
    trans_eps: float = 1e-8
    random_seed: Optional[int] = 123
    progress: bool = True

# =============================================================================
# Particle Gibbs with Backward Simulation (structural DGEV)
# =============================================================================

class DGEVParticleGibbs:
    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        y: np.ndarray,
        period: int,
        level_mode: str = "dynamic",
        trend_mode: str = "dynamic",
        seasonal_mode: str = "dynamic",
        # Initial state prior for latent coords (means & variances)
        m0_level: float = 0.0,
        v0_level: float = 1.0,
        m0_trend: float = 0.0,
        v0_trend: float = 1.0,
        # For seasonality in the dynamic case: lists of length (period-1)
        m0_season: Sequence[float] | None = None,
        v0_season: Sequence[float] | None = None,
        # Priors and config
        priors: Priors = Priors(),
        cfg: SamplerConfig = SamplerConfig(),
        # Optional initial values for deterministic params
        level_value_init: float = 0.0,
        slope_value_init: float = 0.0,
        # Deterministic seasonal initializer — first (p-1) entries only
        seasonal_vector_init: Optional[Sequence[float]] = None,
    ):
        # data
        self.y = np.asarray(y, float)
        self.T = int(self.y.size)
        self.period = int(period)

        # modes
        assert level_mode in {"dynamic", "deterministic"}
        assert trend_mode in {"dynamic", "deterministic", "none"}
        assert seasonal_mode in {"dynamic", "deterministic", "none"}
        self.level_mode = level_mode
        self.trend_mode = trend_mode
        self.seasonal_mode = seasonal_mode

        # config & priors
        self.priors = priors
        self.cfg = cfg
        if cfg.random_seed is not None:
            np.random.seed(cfg.random_seed)

        # ----- latent state layout: [alpha?][beta?][gamma_1..gamma_{p-1}?] -----
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

        # indices into state vector
        self.idx_alpha = layout.index("alpha") if "alpha" in layout else None
        self.idx_beta = layout.index("beta") if "beta" in layout else None
        if self.seasonal_mode == "dynamic":
            self.idx_gamma_start = layout.index("gamma_1") if "gamma_1" in layout else None
            self.idx_gamma_end = self.idx_gamma_start + (self.period - 2) if self.idx_gamma_start is not None else None
        else:
            self.idx_gamma_start = None
            self.idx_gamma_end = None

        # ----- deterministic parameters (sampled if mode='deterministic') -----
        self.level_value = float(level_value_init)  # intercept
        self.slope_value = float(slope_value_init)  # slope

        # deterministic seasonal vector (full length p; last is implied = -sum(first p-1))
        if self.seasonal_mode == "deterministic":
            if seasonal_vector_init is not None:
                g_first = np.asarray(seasonal_vector_init, float)
                if g_first.size != self.period - 1:
                    raise ValueError("seasonal_vector_init must have length = period-1!")
            else:
                # Use prior mean if available; else a smooth default
                if self.priors.m_season is not None:
                    g_first = np.asarray(self.priors.m_season, float)
                    if g_first.size != self.period - 1:
                        raise ValueError("priors.m_season must have length = period-1!")
                else:
                    g_first = default_season_pminus1(self.period)
            g_last = -np.sum(g_first)
            self.season_vec = np.concatenate([g_first, [g_last]]).astype(float)
        else:
            self.season_vec = None

        # ----- observation parameters (initialized at prior means) -----
        # NOTE: m_sigma is the prior mean of log-sigma.
        self.logsigma = float(self.priors.m_sigma)
        self.sigma    = float(np.exp(self.logsigma))
        self.xi       = float(self.priors.m_xi)

        # ----- innovation variances for latent coords (updated if dynamic) -----
        # Start small but non-zero to reflect tiny expected process noise.
        self.Q = np.zeros(self.dim, float)
        if self.idx_alpha is not None:
            self.Q[self.idx_alpha] = 1e-5
        if self.idx_beta is not None:
            self.Q[self.idx_beta] = 1e-5
        if self.seasonal_mode == "dynamic":
            self.Q[self.idx_gamma_end] = 1e-5  # newest seasonal coord has variance

        # ----- initial state x_{0:T} -----
        self.x = np.zeros((self.T + 1, self.dim), float)

        # season prior vectors (length p-1) for dynamic case
        m0_season_arr = None
        v0_season_arr = None
        if self.seasonal_mode == "dynamic":
            m0_season_arr = np.zeros(self.period - 1, float) if m0_season is None else np.asarray(m0_season, float)
            v0_season_arr = np.ones(self.period - 1, float)  if v0_season is None else np.asarray(v0_season, float)
            if m0_season_arr.size != self.period - 1 or v0_season_arr.size != self.period - 1:
                raise ValueError("m0_season and v0_season must both have length = period-1 in dynamic mode.")

        # set initial means/vars per coordinate in the latent state
        m0_list, v0_list = [], []
        for tag in layout:
            if tag == "alpha":
                m0_list.append(float(m0_level)); v0_list.append(float(v0_level))
            elif tag == "beta":
                m0_list.append(float(m0_trend)); v0_list.append(float(v0_trend))
            else:
                k = int(tag.split("_")[1]) - 1  # gamma_k -> index k-1
                m0_list.append(float(m0_season_arr[k]))
                v0_list.append(float(v0_season_arr[k]))
        if self.dim > 0:
            self.x[0] = np.random.normal(np.array(m0_list), np.sqrt(np.array(v0_list)))
            self._propagate_initial_path(Q_init=np.ones(self.dim) * 1e-6)

        # storage
        n_kept = max(0, (self.cfg.n_iter - self.cfg.burn) // max(1, self.cfg.thin))
        self.keep: Dict[str, np.ndarray] = {
            "sigma": np.zeros(n_kept, float),
            "xi": np.zeros(n_kept, float),
            "mu": np.zeros((n_kept, self.T), float),
            "log_evidence": np.zeros(n_kept, float),  # PF marginal log-likelihood per kept draw
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

        # MH bookkeeping
        self.accept = {"logsigma": 0, "xi": 0, "level": 0, "slope": 0, "season": 0}
        self.proposals = {"logsigma": 0, "xi": 0, "level": 0, "slope": 0, "season": 0}

        # optional truths (for overlays)
        self.true_sigma: Optional[float] = None
        self.true_xi: Optional[float] = None
        self.true_Q: Optional[np.ndarray] = None
        self.true_mu_t: Optional[np.ndarray] = None
        self.true_alpha_t: Optional[np.ndarray] = None
        self.true_beta_t: Optional[np.ndarray] = None
        self.true_gamma_t: Optional[np.ndarray] = None

        # last PF evidence (log p(y | theta) with states marginalized)
        self.last_log_evidence: float = float("nan")

    # ------------------------------------------------------------------ #
    # Truth registration (for plotting/diagnostics)
    # ------------------------------------------------------------------ #
    def set_truth(self, sigma: Optional[float] = None, xi: Optional[float] = None, Q: Optional[np.ndarray] = None) -> None:
        self.true_sigma = sigma
        self.true_xi = xi
        self.true_Q = None if Q is None else np.asarray(Q, float)

    def set_truth_paths(
        self,
        mu: Optional[np.ndarray] = None,
        alpha: Optional[np.ndarray] = None,
        beta: Optional[np.ndarray] = None,
        gamma: Optional[np.ndarray] = None,
    ) -> None:
        self.true_mu_t = None if mu is None else np.asarray(mu, float)
        self.true_alpha_t = None if alpha is None else np.asarray(alpha, float)
        self.true_beta_t = None if beta is None else np.asarray(beta, float)
        self.true_gamma_t = None if gamma is None else np.asarray(gamma, float)

    # ------------------------------------------------------------------ #
    # State-space pieces
    # ------------------------------------------------------------------ #
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

    def _state_mean(self, x_prev: np.ndarray, t: int) -> np.ndarray:
        """
        E[x_t | x_{t-1}] for dynamic components.
        """
        m = np.zeros_like(x_prev)

        # alpha
        if self.idx_alpha is not None:
            drift = 0.0
            if self.idx_beta is not None:
                drift = x_prev[self.idx_beta]
            elif self.trend_mode == "deterministic":
                drift = self.slope_value
            m[self.idx_alpha] = x_prev[self.idx_alpha] + drift

        # beta
        if self.idx_beta is not None:
            m[self.idx_beta] = x_prev[self.idx_beta]

        # dynamic season
        if self.seasonal_mode == "dynamic":
            g0, gL = self.idx_gamma_start, self.idx_gamma_end
            if g0 <= gL - 1:
                m[g0:gL] = x_prev[g0 + 1 : gL + 1]  # shift left
            prev_gamma = x_prev[g0 : gL + 1]
            m[gL] = -np.sum(prev_gamma)
        return m

    def _alpha_contribution(self, x_t: np.ndarray, t: int) -> float:
        """
        Contribution of level+trend to μ_t.
        * dynamic level  -> use alpha_t (trend effect already integrated)
        * deterministic  -> base = level_value + (trend)*t, with trend = beta_t if dynamic, slope_value if deterministic
        """
        if self.level_mode == "dynamic":
            return float(x_t[self.idx_alpha]) if self.idx_alpha is not None else 0.0
        base = self.level_value
        if self.idx_beta is not None:  # dynamic trend
            return float(base + x_t[self.idx_beta] * t)
        elif self.trend_mode == "deterministic":
            return float(base + self.slope_value * t)
        else:
            return float(base)

    def _season_contribution(self, x_t: np.ndarray, t: int) -> float:
        if self.seasonal_mode == "dynamic":
            return float(x_t[self.idx_gamma_end]) if self.idx_gamma_end is not None else 0.0
        elif self.seasonal_mode == "deterministic":
            return float(self.season_vec[t % self.period])
        else:
            return 0.0

    def mu_from_state(self, x_t: np.ndarray, t: int) -> float:
        return self._alpha_contribution(x_t, t) + self._season_contribution(x_t, t)

    def _transition_logpdf(self, x_prev: np.ndarray, x_cur: np.ndarray, t: int) -> float:
        mean = self._state_mean(x_prev, t)
        eps = self.cfg.trans_eps
        out = 0.0
        for k in range(self.dim):
            var = self.Q[k] if self.Q[k] > 0.0 else eps
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

    # ------------------------------------------------------------------ #
    # Parameter updates (static parameters via RW–MH)
    # ------------------------------------------------------------------ #
    def _mu_vec_current(self) -> np.ndarray:
        # y[t-1] ↔ x[t]; t in 1..T; calendar index = t-1
        return np.array([self.mu_from_state(self.x[t], t - 1) for t in range(1, self.T + 1)], float)

    def update_Q(self) -> None:
        """
        Update innovation variances for dynamic components using IG posteriors.
        No-op for deterministic or absent components.
        """
        a0, b0 = self.priors.a_q, self.priors.b_q

        # alpha variance (if dynamic)
        if self.idx_alpha is not None:
            resid = []
            for t in range(1, self.T + 1):
                drift = 0.0
                if self.idx_beta is not None:
                    drift = self.x[t - 1, self.idx_beta]
                elif self.trend_mode == "deterministic":
                    drift = self.slope_value
                mean = self.x[t - 1, self.idx_alpha] + drift
                resid.append(self.x[t, self.idx_alpha] - mean)
            rss = float(np.sum(np.square(resid)))
            a = a0 + 0.5 * self.T
            b = b0 + 0.5 * rss
            self.Q[self.idx_alpha] = 1.0 / np.random.gamma(a, 1.0 / b)

        # beta variance (if dynamic)
        if self.idx_beta is not None:
            resid = self.x[1:, self.idx_beta] - self.x[:-1, self.idx_beta]
            rss = float(np.sum(np.square(resid)))
            a = a0 + 0.5 * self.T
            b = b0 + 0.5 * rss
            self.Q[self.idx_beta] = 1.0 / np.random.gamma(a, 1.0 / b)

        # newest seasonal coord variance (if dynamic)
        if self.seasonal_mode == "dynamic":
            g0, gL = self.idx_gamma_start, self.idx_gamma_end
            resid = []
            for t in range(1, self.T + 1):
                prev_gamma = self.x[t - 1, g0 : gL + 1]
                mean_new = -np.sum(prev_gamma)
                resid.append(self.x[t, gL] - mean_new)
            rss = float(np.sum(np.square(resid)))
            a = a0 + 0.5 * self.T
            b = b0 + 0.5 * rss
            self.Q[gL] = 1.0 / np.random.gamma(a, 1.0 / b)

    def update_logsigma(self) -> None:
        step = self.cfg.step_logsigma
        cur = self.logsigma
        prop = cur + np.random.normal(0.0, step)
        sigma_cur, sigma_prop = float(np.exp(cur)), float(np.exp(prop))

        mu_vec = self._mu_vec_current()
        ll_old = gev_loglike_sum(self.y, mu_vec, sigma_cur, self.xi)
        ll_new = gev_loglike_sum(self.y, mu_vec, sigma_prop, self.xi)
        self.proposals["logsigma"] += 1
        if ll_new == -np.inf:
            return

        lp_old = -0.5 * ((cur - self.priors.m_sigma) ** 2) / (self.priors.s_sigma ** 2)
        lp_new = -0.5 * ((prop - self.priors.m_sigma) ** 2) / (self.priors.s_sigma ** 2)
        logacc = (ll_new + lp_new) - (ll_old + lp_old)
        if np.log(np.random.rand()) < logacc:
            self.logsigma = prop
            self.sigma = sigma_prop
            self.accept["logsigma"] += 1

    def update_xi(self) -> None:
        step = self.cfg.step_xi
        cur = self.xi
        prop = cur + np.random.normal(0.0, step)

        mu_vec = self._mu_vec_current()
        ll_old = gev_loglike_sum(self.y, mu_vec, self.sigma, cur)
        ll_new = gev_loglike_sum(self.y, mu_vec, self.sigma, prop)
        self.proposals["xi"] += 1
        if ll_new == -np.inf:
            return

        lp_old = -0.5 * ((cur - self.priors.m_xi) ** 2) / (self.priors.s_xi ** 2)
        lp_new = -0.5 * ((prop - self.priors.m_xi) ** 2) / (self.priors.s_xi ** 2)
        logacc = (ll_new + lp_new) - (ll_old + lp_old)
        if np.log(np.random.rand()) < logacc:
            self.xi = prop
            self.accept["xi"] += 1

    def update_level_value(self) -> None:
        """
        Sample the deterministic level parameter (if active) via RW–MH.
        Prior: level_value ~ N(m_level, s_level^2).
        """
        if self.level_mode != "deterministic":
            return
        step = self.cfg.step_level
        cur = self.level_value
        prop = cur + np.random.normal(0.0, step)

        old_level = self.level_value
        self.level_value = prop
        mu_vec_prop = self._mu_vec_current()
        self.level_value = old_level
        mu_vec_old = self._mu_vec_current()

        ll_old = gev_loglike_sum(self.y, mu_vec_old, self.sigma, self.xi)
        ll_new = gev_loglike_sum(self.y, mu_vec_prop, self.sigma, self.xi)
        self.proposals["level"] += 1
        if ll_new == -np.inf:
            return

        lp_old = -0.5 * ((cur - self.priors.m_level) ** 2) / (self.priors.s_level ** 2)
        lp_new = -0.5 * ((prop - self.priors.m_level) ** 2) / (self.priors.s_level ** 2)
        logacc = (ll_new + lp_new) - (ll_old + lp_old)
        if np.log(np.random.rand()) < logacc:
            self.level_value = prop
            self.accept["level"] += 1

    def update_slope(self) -> None:
        """
        Sample the deterministic slope parameter (if active) via RW–MH.
        Prior: slope_value ~ N(m_slope, s_slope^2).
        """
        if self.trend_mode != "deterministic":
            return
        step = self.cfg.step_slope
        cur = self.slope_value
        prop = cur + np.random.normal(0.0, step)

        old_beta = self.slope_value
        self.slope_value = prop
        mu_vec_prop = self._mu_vec_current()
        self.slope_value = old_beta
        mu_vec_old = self._mu_vec_current()

        ll_old = gev_loglike_sum(self.y, mu_vec_old, self.sigma, self.xi)
        ll_new = gev_loglike_sum(self.y, mu_vec_prop, self.sigma, self.xi)
        self.proposals["slope"] += 1
        if ll_new == -np.inf:
            return

        lp_old = -0.5 * ((cur - self.priors.m_slope) ** 2) / (self.priors.s_slope ** 2)
        lp_new = -0.5 * ((prop - self.priors.m_slope) ** 2) / (self.priors.s_slope ** 2)
        logacc = (ll_new + lp_new) - (ll_old + lp_old)
        if np.log(np.random.rand()) < logacc:
            self.slope_value = prop
            self.accept["slope"] += 1

    def update_season_vec(self) -> None:
        """
        Sample the deterministic seasonal vector (first p-1 entries) via RW–MH.
        The last entry is always set to enforce sum-to-zero.
        Prior: gamma_first[k] ~ N(m_season[k], s_season^2), independent across k.
        """
        if self.seasonal_mode != "deterministic":
            return
        step = self.cfg.step_season
        v_cur = self.season_vec.copy()

        # propose perturbations to first p-1 entries, keep last to enforce sum-zero
        prop = v_cur.copy()
        prop[:-1] = v_cur[:-1] + np.random.normal(0.0, step, size=self.period - 1)
        prop[-1] = -np.sum(prop[:-1])

        old = self.season_vec
        self.season_vec = prop
        mu_vec_prop = self._mu_vec_current()
        self.season_vec = old
        mu_vec_old = self._mu_vec_current()

        ll_old = gev_loglike_sum(self.y, mu_vec_old, self.sigma, self.xi)
        ll_new = gev_loglike_sum(self.y, mu_vec_prop, self.sigma, self.xi)
        self.proposals["season"] += 1
        if ll_new == -np.inf:
            return

        # Prior mean for first p-1 entries
        m_first = self.priors.m_season
        if m_first is None:
            m_first = np.zeros(self.period - 1, float)
        else:
            m_first = np.asarray(m_first, float)
            if m_first.size != self.period - 1:
                raise ValueError("priors.m_season must have length = period-1!")
        s = float(self.priors.s_season)

        lp_old = -0.5 * np.sum(((v_cur[:-1] - m_first) / s) ** 2)
        lp_new = -0.5 * np.sum(((prop[:-1] - m_first) / s) ** 2)

        logacc = (ll_new + lp_new) - (ll_old + lp_old)
        if np.log(np.random.rand()) < logacc:
            self.season_vec = prop
            self.accept["season"] += 1

    # ------------------------------------------------------------------ #
    # Conditional bootstrap PF + backward simulation
    # ------------------------------------------------------------------ #
    def _conditional_pf(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Conditional bootstrap particle filter that returns:
        - parts: particles over time,
        - w: normalized weights,
        - a: ancestor indices,
        - logZ: marginal log-likelihood log p(y | theta) (states integrated out),
                estimated as sum_t log( (1/N) * sum_m w_t^m ).
        """
        N, T, D = self.cfg.n_particles, self.T, self.dim
        parts = np.zeros((T + 1, N, D), float)
        w = np.zeros((T + 1, N), float)
        a = np.zeros((T + 1, N), int)
        logZ = 0.0  # accumulate PF evidence

        # t=0 : replicate prior state
        if D > 0:
            parts[0, :, :] = self.x[0]

        # ----- t = 1 -----
        for m in range(N - 1):
            if D > 0:
                parts[1, m, :] = self._transition_sample(parts[0, m, :], t=1)
            a[1, m] = m
        if D > 0:
            parts[1, N - 1, :] = self.x[1].copy()
        a[1, N - 1] = N - 1

        # weights for y[0] in log domain
        lw = np.zeros(N, float)
        for m in range(N):
            mu = self.mu_from_state(parts[1, m, :] if D > 0 else np.zeros(0), t=0)
            lw[m] = gev_logpdf(self.y[0], mu, self.sigma, self.xi)

        # evidence increment at t=1: log( mean(exp(lw)) )
        lw_max = np.max(lw)
        log_mean_w = lw_max + math.log(np.mean(np.exp(lw - lw_max)) + 1e-300)
        logZ += log_mean_w

        # normalized weights for resampling
        w[1, :] = self._safe_normalize(np.exp(lw - lw_max))

        if self.cfg.progress:
            print("  Running conditional bootstrap PF...")
        it = tqdm(range(2, T + 1)) if self.cfg.progress else range(2, T + 1)

        for t in it:
            p_res = self._safe_normalize(w[t - 1, :])
            anc = np.random.choice(N, size=N - 1, p=p_res, replace=True)

            for m in range(N - 1):
                if D > 0:
                    x_prev = parts[t - 1, anc[m], :]
                    parts[t, m, :] = self._transition_sample(x_prev, t=t)
                a[t, m] = anc[m]

            # conditional particle remains at reference path
            if D > 0:
                ref_xt = self.x[t]
                logw_prev = np.log(np.clip(w[t - 1, :], 1e-300, None))
                logf = np.array(
                    [self._transition_logpdf(parts[t - 1, j, :], ref_xt, t=t) for j in range(N)],
                    float
                )
                post = self._safe_normalize(np.exp((logw_prev + logf) - np.max(logw_prev + logf)))
                a[t, N - 1] = np.random.choice(N, p=post)
                parts[t, N - 1, :] = ref_xt.copy()
            else:
                a[t, N - 1] = N - 1  # degenerate but consistent

            # importance log-weights for y[t-1]
            y_idx = t - 1
            lw = np.zeros(N, float)
            for m in range(N):
                mu = self.mu_from_state(parts[t, m, :] if D > 0 else np.zeros(0), t=y_idx)
                lw[m] = gev_logpdf(self.y[y_idx], mu, self.sigma, self.xi)

            # evidence increment at time t
            lw_max = np.max(lw)
            log_mean_w = lw_max + math.log(np.mean(np.exp(lw - lw_max)) + 1e-300)
            logZ += log_mean_w

            # normalized weights for resampling
            w[t, :] = self._safe_normalize(np.exp(lw - lw_max))

        return parts, w, a, float(logZ)

    def _backward_simulation(self, parts: np.ndarray, w: np.ndarray) -> np.ndarray:
        N, T, D = self.cfg.n_particles, self.T, self.dim
        if D == 0:
            return self.x  # nothing to back-sample
        idx = np.zeros(T + 1, int)
        idx[T] = np.random.choice(N, p=w[T, :])

        if self.cfg.progress:
            print("  Running backward simulation...")
        it = tqdm(range(T - 1, 0, -1)) if self.cfg.progress else range(T - 1, 0, -1)
        for t in it:
            xtp1 = parts[t + 1, idx[t + 1], :]
            logp = np.array(
                [np.log(w[t, j] + 1e-300) + self._transition_logpdf(parts[t, j, :], xtp1, t=t) for j in range(N)],
                float
            )
            p = self._safe_normalize(np.exp(logp - np.max(logp)))
            idx[t] = np.random.choice(N, p=p)

        x_new = self.x.copy()
        for t in range(1, T + 1):
            x_new[t, :] = parts[t, idx[t], :]
        return x_new

    def update_states_pgbs(self) -> None:
        parts, w, _, logZ = self._conditional_pf()
        self.x = self._backward_simulation(parts, w)
        self.last_log_evidence = float(logZ)

    # ------------------------------------------------------------------ #
    # MCMC driver
    # ------------------------------------------------------------------ #
    def run(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg

        # --- how many draws will actually be saved? ---
        save_iters = list(range(cfg.burn, cfg.n_iter, cfg.thin))
        n_kept = max(0, len(save_iters))

        # re-init storage using n_kept computed above
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

        for it in range(cfg.n_iter):
            if cfg.progress:
                print(f"Iteration {it + 1}/{cfg.n_iter}")

            # 1) latent states (ONLY if at least one component is dynamic)
            if self.dim > 0:
                self.update_states_pgbs()
                current_log_ev = float(self.last_log_evidence)
            else:
                mu_vec_now = self._mu_vec_current()
                current_log_ev = float(gev_loglike_sum(self.y, mu_vec_now, self.sigma, self.xi))

            if cfg.progress:
                print(f"  log p(y | theta) [PF states-marginalized] = {current_log_ev:.6f}")

            # 2) dynamic innovation variances Q
            if self.dim > 0:
                if cfg.progress: print("  Updating Q...")
                self.update_Q()

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

            # 4) observation params (always static)
            if cfg.progress: print("  Updating logsigma / xi ...")
            self.update_logsigma()
            self.update_xi()

            # 5) store
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
                f"acc(logsigma)={self.accept['logsigma']}/{self.proposals['logsigma']} "
                f"acc(xi)={self.accept['xi']}/{self.proposals['xi']} "
                f"acc(level)={self.accept['level']}/{self.proposals['level']} "
                f"acc(slope)={self.accept['slope']}/{self.proposals['slope']} "
                f"acc(season)={self.accept['season']}/{self.proposals['season']}"
            )
        return self.keep

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save_posterior(self, out_npz_path: str, extra_meta: Optional[dict] = None) -> None:
        _ensure_dir(os.path.dirname(out_npz_path))

        arrays = dict(self.keep)
        arrays["y"] = self.y.copy()
        arrays["x_last"] = self.x[1 : self.T + 1].copy() if self.dim > 0 else np.zeros((self.T, 0))

        # optional truths
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
        }
        if extra_meta:
            meta.update(extra_meta)

        meta_path = out_npz_path.replace(".npz", ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[save] Posterior -> {out_npz_path}")
        print(f"[save] Metadata  -> {meta_path}")

# ------------------------- #
# CLI / Example run & plots
# ------------------------- #
if __name__ == "__main__":
    import sys, argparse
    import matplotlib.pyplot as plt
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from simulator.extremal_time_series import Extremal_Time_Series
    from simulator.dgev_plotter import DGEVPlotter  # optional; plots can be done later

    parser = argparse.ArgumentParser(description="DGEV PG-BS Sampler")
    # Modes
    parser.add_argument("--level-mode", choices=["dynamic", "deterministic"], default="dynamic")
    parser.add_argument("--trend-mode", choices=["dynamic", "deterministic", "none"], default="deterministic")
    parser.add_argument("--season-mode", choices=["dynamic", "deterministic", "none"], default="none")
    # Basics
    parser.add_argument("--period", type=int, default=4)
    parser.add_argument("--T", type=int, default=100)
    # Initial values (shared as priors or fixed)
    parser.add_argument("--level-init", type=float, default=5.0)
    parser.add_argument("--slope-init", type=float, default=0.02)
    # Simulation truths (obs + state noise) — typical Q ≈ 1e-5
    parser.add_argument("--true-sigma", type=float, default=2.0)
    parser.add_argument("--true-xi", type=float, default=0.1)
    parser.add_argument("--q-alpha", type=float, default=1e-5)
    parser.add_argument("--q-beta",  type=float, default=1e-5)
    parser.add_argument("--q-gamma", type=float, default=1e-5)
    # Priors (NOTE: prior-m-sigma is for log-sigma); IG prior centered near 1e-5
    parser.add_argument("--prior-m-sigma", type=float, default=2.0)
    parser.add_argument("--prior-s-sigma", type=float, default=1.0)
    parser.add_argument("--prior-m-xi", type=float, default=1.0)
    parser.add_argument("--prior-s-xi", type=float, default=0.2)
    parser.add_argument("--prior-aq", type=float, default=1.5)   # mean b/(a-1) = 5e-6 / 0.5 = 1e-5
    parser.add_argument("--prior-bq", type=float, default=5e-6)  # heavy tail -> prefers small Q but allows larger
    parser.add_argument("--prior-m-level", type=float, default=0.0)
    parser.add_argument("--prior-s-level", type=float, default=10.0)
    parser.add_argument("--prior-m-slope", type=float, default=0.0)
    parser.add_argument("--prior-s-slope", type=float, default=10.0)
    parser.add_argument(
        "--prior-m-season",
        type=str,
        default=None,
        help="Comma-separated first (p-1) means for deterministic seasonal prior (e.g. '0,0,0')."
    )
    parser.add_argument("--prior-s-season", type=float, default=5.0)
    # Sampler config
    parser.add_argument("--n-iter", type=int, default=2000)
    parser.add_argument("--burn", type=int, default=100)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--step-logsigma", type=float, default=0.06)
    parser.add_argument("--step-xi", type=float, default=0.06)
    parser.add_argument("--step-level", type=float, default=0.02)
    parser.add_argument("--step-slope", type=float, default=0.05)
    parser.add_argument("--step-season", type=float, default=0.02)
    parser.add_argument("--particles", type=int, default=250)
    parser.add_argument("--trans-eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--progress", default=True)
    # Output & plotting
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--show-plots", action="store_true")

    args = parser.parse_args()
    np.random.seed(args.seed)

    # --- Modes selected ---
    sim_level_mode  = args.level_mode
    sim_trend_mode  = args.trend_mode
    sim_season_mode = args.season_mode

    # --- Initial seasonal vectors for the SIMULATOR (p-1 entries if season active) ---
    m0_season_first = np.zeros(args.period - 1, float)
    v0_season_first = np.full(args.period - 1, 0.5, float)
    if sim_season_mode == "deterministic":
        # Simulator accepts p-1 for deterministic season as well (last implied)
        m0_season_first = default_season_pminus1(args.period)
    elif sim_season_mode == "none":
        m0_season_first = None
        v0_season_first = None

    # --- Simulate data ---
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
        m0_season=m0_season_first, v0_season=v0_season_first,  # length = period-1 if active
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

    # --- Priors & sampler config ---
    m_season_prior = parse_csv_floats(args.prior_m_season)
    if m_season_prior is not None and len(m_season_prior) != args.period - 1:
        raise ValueError(f"--prior-m-season must have length {args.period - 1} (got {len(m_season_prior)}).")

    priors = Priors(
        # NOTE: m_sigma is on log-scale
        m_sigma=float(args.prior_m_sigma),
        s_sigma=float(args.prior_s_sigma),
        m_xi=float(args.prior_m_xi),
        s_xi=float(args.prior_s_xi),
        a_q=float(args.prior_aq), b_q=float(args.prior_bq),
        m_level=float(args.prior_m_level), s_level=float(args.prior_s_level),
        m_slope=float(args.prior_m_slope), s_slope=float(args.prior_s_slope),
        m_season=m_season_prior, s_season=float(args.prior_s_season),
    )

    # print the mean and variance of the process noise priors
    prior_mean_Q = priors.b_q / (priors.a_q - 1.0) if priors.a_q > 1.0 else float("inf")
    prior_var_Q  = (
        priors.b_q**2 / ((priors.a_q - 1.0)**2 * (priors.a_q - 2.0))
        if priors.a_q > 2.0 else float("inf")
    )
    print(f"[info] Prior on Q has mean={prior_mean_Q:.6g} and variance={prior_var_Q:.6g}.")

    cfg = SamplerConfig(
        n_iter=args.n_iter, burn=args.burn, thin=args.thin,
        step_logsigma=args.step_logsigma, step_xi=args.step_xi,
        step_level=args.step_level, step_slope=args.step_slope,
        step_season=args.step_season,
        n_particles=args.particles, trans_eps=args.trans_eps,
        random_seed=args.seed, progress=args.progress,
    )

    # Initial seasonal vector for deterministic sampler (p-1 entries)
    seasonal_init_pminus1 = (
        np.asarray(m_season_prior, float) if (sim_season_mode == "deterministic" and m_season_prior is not None)
        else (default_season_pminus1(args.period) if sim_season_mode == "deterministic" else None)
    )

    sampler = DGEVParticleGibbs(
        y=y, period=args.period,
        level_mode=sim_level_mode, trend_mode=sim_trend_mode, seasonal_mode=sim_season_mode,
        m0_level=args.level_init, v0_level=0.2,
        m0_trend=(args.slope_init if sim_trend_mode != "none" else 0.0), v0_trend=0.05,
        m0_season=(m0_season_first if sim_season_mode == "dynamic" else None),
        v0_season=(v0_season_first if sim_season_mode == "dynamic" else None),
        priors=priors, cfg=cfg,
        seasonal_vector_init=seasonal_init_pminus1,  # length = period-1
    )

    true_Q = []
    if sim_level_mode == "dynamic": true_Q.append(args.q_alpha)
    if sim_trend_mode == "dynamic": true_Q.append(args.q_beta)
    if sim_season_mode == "dynamic": true_Q += [args.q_gamma] + [0.0] * (args.period - 2)
    sampler.set_truth(sigma=args.true_sigma, xi=args.true_xi,
                      Q=(np.asarray(true_Q, float) if len(true_Q) else None))
    sampler.set_truth_paths(mu=mu_T, alpha=alpha_T, beta=beta_T, gamma=gamma_T)

    tag = f"{sim_level_mode}-{sim_trend_mode}-{sim_season_mode}"
    out_dir = args.out_dir or os.path.join("results", "simulations", "DGEV", f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
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

    # Print true Qs when available
    if sampler.true_Q is not None:
        tq = np.asarray(sampler.true_Q, float)
        if sampler.idx_alpha is not None and tq.size > sampler.idx_alpha:
            print(f"True Q_alpha:         {tq[sampler.idx_alpha]:.6g}")
        if sampler.idx_beta is not None and tq.size > sampler.idx_beta:
            print(f"True Q_beta:          {tq[sampler.idx_beta]:.6g}")
        if sampler.seasonal_mode == "dynamic" and sampler.idx_gamma_end is not None and tq.size > sampler.idx_gamma_end:
            print(f"True Q_gamma(last):   {tq[sampler.idx_gamma_end]:.6g}")

    if "Q" in posterior:
        if sampler.idx_alpha is not None:
            print(f"Posterior mean Q_alpha: {np.mean(posterior['Q'][:, sampler.idx_alpha]):.6g}")
        if sampler.idx_beta is not None:
            print(f"Posterior mean Q_beta:  {np.mean(posterior['Q'][:, sampler.idx_beta]):.6g}")
        if sampler.seasonal_mode == "dynamic":
            print(f"Posterior mean Q_gamma: {np.mean(posterior['Q'][:, sampler.idx_gamma_end]):.6g}")

    if "log_evidence" in posterior and posterior["log_evidence"].size > 0:
        le = posterior["log_evidence"]
        print(f"Mean log p(y|theta): {np.nanmean(le):.3f} | Median: {np.nanmedian(le):.3f} | Best: {np.nanmax(le):.3f}")

    # (Optional) plotting with DGEVPlotter can be added here.
