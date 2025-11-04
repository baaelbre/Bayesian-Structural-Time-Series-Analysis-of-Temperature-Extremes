# %% optimization/dgev_pgas_hc.py
from __future__ import annotations

import os, sys, math, json, time
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, List, Sequence, Any

import numpy as np
from tqdm import tqdm
from datetime import datetime

# =============================================================================
# Small utils
# =============================================================================

def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

def build_seasonal(period: int) -> np.ndarray:
    """Default smooth deterministic seasonal (first p-1 entries; last is closure to sum-zero)."""
    g = np.cos(2 * np.pi * np.arange(period) / period)
    g -= np.mean(g)
    return g[: period - 1].astype(float)

def parse_csv_floats(s: Optional[str]) -> Optional[List[float]]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return [float(tok) for tok in s.split(",")]

def _mad(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    if v.size == 0:
        return 0.0
    m = np.median(v)
    return float(np.median(np.abs(v - m)))

def _robust_sd(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    return _mad(v) / 1.4826 if v.size else 0.0

# =============================================================================
# GEV log-likelihood helpers
# =============================================================================

def gev_logpdf(y: float, mu: float, sigma: float, xi: float) -> float:
    if not np.isfinite(mu) or not np.isfinite(sigma) or not np.isfinite(xi) or sigma <= 0.0:
        return -np.inf
    z = (y - mu) / sigma
    u = 1.0 + xi * z
    if u <= 0.0:
        return -np.inf
    if abs(xi) < 1e-8:  # Gumbel limit
        return -math.log(sigma) - math.exp(-z) - z
    return -math.log(sigma) - (1.0 + 1.0 / xi) * math.log(u) - u ** (-1.0 / xi)

def gev_loglike_sum(y: np.ndarray, mu_vec: np.ndarray, sigma: float, xi: float) -> float:
    if sigma <= 0.0 or mu_vec.size != y.size or np.any(~np.isfinite(mu_vec)):
        return -np.inf
    z = (y - mu_vec) / sigma
    u = 1.0 + xi * z
    if np.any(u <= 0.0):
        return -np.inf
    if abs(xi) < 1e-8:
        return float(np.sum(-np.log(sigma) - np.exp(-z) - z))
    return float(np.sum(-np.log(sigma) - (1.0 + 1.0/xi) * np.log(u) - u ** (-1.0/xi)))

# =============================================================================
# Priors & Config   (Half-Cauchy on process SDs via IG mixtures)
# =============================================================================

@dataclass
class Priors:
    # Observation parameter priors (on logsigma and xi, both Gaussian)
    m_sigma: float = 0.0
    s_sigma: float = 10.0
    m_xi: float = 0.0
    s_xi: float = 1.0

    # Deterministic structural priors (Gaussian)
    m_level: float = 0.0
    s_level: float = 10.0
    m_slope: float = 0.0
    s_slope: float = 10.0
    m_season: Optional[Sequence[float]] = None  # first p-1 means; last implied
    s_season: float = 5.0

    # Half-Cauchy scales A_k for process SDs (s = sqrt(Q))
    hc_scale_alpha: float = 0.5
    hc_scale_beta:  float = 0.5
    hc_scale_gamma: float = 0.5  # applies to the LAST dynamic seasonal coord only

@dataclass
class SamplerConfig:
    n_iter: int = 5000
    burn: int = 1000
    thin: int = 5

    # RW–MH step sizes
    step_logsigma: float = 0.05
    step_xi: float = 0.05
    step_level: float = 0.05
    step_slope: float = 0.02
    step_season: float = 0.05

    # Particle filter
    n_particles: int = 200
    trans_eps: float = 1e-8
    random_seed: Optional[int] = 123
    progress: bool = True
    progress_every: int = 0  # 0 => ~2% of n_iter
    ess_threshold_frac: float = 0.50  # resample when ESS < frac * N

    # Adaptive RW–MH (Robbins–Monro; windowed)
    adapt_steps: bool = True
    adapt_every: int = 25
    adapt_until: str = "burn"         # "burn" or "all"
    adapt_target_1d: float = 0.44
    adapt_eta0: float = 0.05
    adapt_eta_decay: float = 0.75
    step_min: float = 1e-5
    step_max: float = 1.0

# =============================================================================
# DGEV Particle Gibbs with Ancestor Sampling (PGAS) + HC-mixture Q updates
# =============================================================================

class DGEVParticleGibbsHC:
    """
    Structural Dynamic GEV model sampled by PGAS (bootstrap proposal, ESS-triggered resampling).
    Process variances Q use Half-Cauchy priors via IG mixtures (pure Gibbs).
    Everything non-conjugate (σ, ξ, deterministic α/β/γ params) uses RW–MH.
    """

    # ------------------------- Construction ------------------------- #
    def __init__(
        self,
        y: np.ndarray,
        period: int,
        level_mode: str = "dynamic",
        trend_mode: str = "dynamic",
        seasonal_mode: str = "dynamic",
        # x0 priors for dynamic coords (Normal)
        m0_level: float = 0.0, v0_level: float = 1.0,
        m0_trend: float = 0.0, v0_trend: float = 1.0,
        m0_season: Sequence[float] | None = None,   # length p-1 if dynamic
        v0_season: Sequence[float] | None = None,   # length p-1 if dynamic
        # priors & config
        priors: Priors = Priors(),
        cfg: SamplerConfig = SamplerConfig(),
        # deterministic initial values
        level_value_init: float = 0.0,
        slope_value_init: float = 0.0,
        seasonal_vector_init: Optional[Sequence[float]] = None,  # p-1 entries
    ):
        # Data
        self.y = np.asarray(y, float)
        self.T = int(self.y.size)
        self.period = int(period)
        if self.period < 2:
            raise ValueError("period must be >= 2")

        # Modes
        assert level_mode in {"dynamic", "deterministic"}
        assert trend_mode in {"dynamic", "deterministic", "none"}
        assert seasonal_mode in {"dynamic", "deterministic", "none"}
        if trend_mode == "dynamic" and level_mode != "dynamic":
            raise ValueError("dynamic trend requires dynamic level")
        self.level_mode = level_mode
        self.trend_mode = trend_mode
        self.seasonal_mode = seasonal_mode

        # Priors / cfg
        self.priors, self.cfg = priors, cfg
        if cfg.random_seed is not None:
            np.random.seed(cfg.random_seed)

        # Layout (NEWEST-FIRST seasonal state, last coord carries Q_gamma)
        layout: List[str] = []
        if self.level_mode == "dynamic": layout.append("alpha")
        if self.trend_mode == "dynamic": layout.append("beta")
        if self.seasonal_mode == "dynamic":
            layout.extend([f"g{k}" for k in range(1, self.period)])
        self._layout = layout
        self.dim = len(layout)

        self.idx_alpha = layout.index("alpha") if "alpha" in layout else None
        self.idx_beta  = layout.index("beta")  if "beta"  in layout else None
        if self.seasonal_mode == "dynamic":
            self.idx_g_start = layout.index("g1")
            self.idx_g_end   = self.idx_g_start + (self.period - 2)
        else:
            self.idx_g_start = self.idx_g_end = None

        if self.dim == 0 and not (self.level_mode == "deterministic" or self.seasonal_mode == "deterministic"):
            raise ValueError("At least one contribution to μ_t must exist.")

        # Deterministic parameters (outside state)
        self.level_value = float(level_value_init)
        self.slope_value = float(slope_value_init)
        if self.seasonal_mode == "deterministic":
            if seasonal_vector_init is not None:
                g1 = np.asarray(seasonal_vector_init, float)
                if g1.size != self.period - 1:
                    raise ValueError("seasonal_vector_init must have length p-1.")
            else:
                if self.priors.m_season is not None:
                    g1 = np.asarray(self.priors.m_season, float)
                    if g1.size != self.period - 1:
                        raise ValueError("priors.m_season must have length p-1.")
                else:
                    g1 = build_seasonal(self.period)
            self.season_vec = np.r_[g1, -np.sum(g1)]
        else:
            self.season_vec = None

        # Observation parameters
        self.logsigma = float(self.priors.m_sigma)
        self.sigma    = float(np.exp(self.logsigma))
        self.xi       = float(self.priors.m_xi)

        # Process variances Q and auxiliaries (Half-Cauchy via IG mixtures)
        self.Q = np.zeros(self.dim, float)     # variances per coord (alpha, beta, g's-last has noise)
        self.a_aux = {"alpha": 1.0, "beta": 1.0, "gamma": 1.0}  # a_k ~ InvGamma(1/2, 1/A_k^2)

        # Initial latent path x_{0:T}
        self.x = np.zeros((self.T + 1, self.dim), float)
        if self.seasonal_mode == "dynamic":
            m0_s = np.zeros(self.period - 1, float) if m0_season is None else np.asarray(m0_season, float)
            v0_s = np.ones(self.period - 1, float)  if v0_season is None else np.asarray(v0_season, float)
            if m0_s.size != self.period - 1 or v0_s.size != self.period - 1:
                raise ValueError("m0_season and v0_season must be length p-1 in dynamic seasonal mode.")
        m0_list, v0_list = [], []
        for tag in layout:
            if tag == "alpha":
                m0_list.append(float(m0_level));  v0_list.append(float(v0_level))
            elif tag == "beta":
                m0_list.append(float(m0_trend));  v0_list.append(float(v0_trend))
            else:
                k = int(tag[1:]) - 1  # g{k}
                m0_list.append(float(m0_s[k])); v0_list.append(float(v0_s[k]))
        if self.dim > 0:
            self.x[0] = np.random.normal(np.array(m0_list), np.sqrt(np.array(v0_list)))
            self._propagate_initial_path(Q_init=np.ones(self.dim) * 1e-6)

        # Storage
        self.keep: Dict[str, np.ndarray] = {}

        # MH bookkeeping (no MH for process sds now)
        self.accept = {"logsigma": 0, "xi": 0, "level": 0, "slope": 0, "season": 0}
        self.proposals = {"logsigma": 0, "xi": 0, "level": 0, "slope": 0, "season": 0}
        self._mh_prev_acc = dict(self.accept)
        self._mh_prev_prop = dict(self.proposals)
        self._adapt_round = 0

        # Truth overlays (optional)
        self.true_sigma: Optional[float] = None
        self.true_xi: Optional[float] = None
        self.true_Q: Optional[np.ndarray] = None
        self.true_mu_t: Optional[np.ndarray] = None
        self.true_alpha_t: Optional[np.ndarray] = None
        self.true_beta_t: Optional[np.ndarray] = None
        self.true_gamma_t: Optional[np.ndarray] = None

        # PF diagnostics
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

    # ----------------------------- State model ------------------------------ #
    def _state_mean(self, x_prev: np.ndarray, t: int) -> np.ndarray:
        m = np.zeros_like(x_prev)
        # alpha with drift
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
        # seasonal (shift & closure)
        if self.seasonal_mode == "dynamic":
            g0, gL = self.idx_g_start, self.idx_g_end
            if g0 <= gL - 1:
                m[g0:gL] = x_prev[g0 + 1 : gL + 1]
            prev_gamma = x_prev[g0 : gL + 1]
            m[gL] = -np.sum(prev_gamma)
        return m

    def _alpha_contribution(self, x_t: np.ndarray, t: int) -> float:
        if self.level_mode == "dynamic":
            return float(x_t[self.idx_alpha]) if self.idx_alpha is not None else 0.0
        base = self.level_value
        if self.idx_beta is not None:
            return float(base + x_t[self.idx_beta] * t)
        if self.trend_mode == "deterministic":
            return float(base + self.slope_value * t)
        return float(base)

    def _season_contribution(self, x_t: np.ndarray, t: int) -> float:
        if self.seasonal_mode == "dynamic":
            return float(x_t[self.idx_g_end]) if self.idx_g_end is not None else 0.0
        if self.seasonal_mode == "deterministic":
            return float(self.season_vec[t % self.period])
        return 0.0

    def mu_from_state(self, x_t: np.ndarray, t: int) -> float:
        return self._alpha_contribution(x_t, t) + self._season_contribution(x_t, t)

    def _transition_logpdf(self, x_prev: np.ndarray, x_cur: np.ndarray, t: int) -> float:
        mean = self._state_mean(x_prev, t)
        eps = self.cfg.trans_eps
        out = 0.0
        for k in range(self.dim):
            var = self.Q[k] if self.Q[k] > 0 else eps
            diff = x_cur[k] - mean[k]
            out += -0.5 * (math.log(2.0 * math.pi * var) + (diff * diff) / var)
        return float(out)

    def _transition_sample(self, x_prev: np.ndarray, t: int) -> np.ndarray:
        mean = self._state_mean(x_prev, t)
        var = np.where(self.Q > 0.0, self.Q, self.cfg.trans_eps)
        return mean + np.random.normal(0.0, np.sqrt(var), size=self.dim)

    def _propagate_initial_path(self, Q_init: np.ndarray) -> None:
        for t in range(1, self.T + 1):
            if self.dim == 0: break
            mean = self._state_mean(self.x[t - 1], t)
            self.x[t] = mean + np.random.normal(0.0, np.sqrt(Q_init), size=self.dim)

    # ------------------ Innovation sums-of-squares for Gibbs ------------------ #
    def _innovation_ss_alpha(self) -> Tuple[float, int]:
        if self.idx_alpha is None: return 0.0, 0
        ss = 0.0
        for t in range(1, self.T + 1):
            drift = 0.0
            if self.idx_beta is not None:
                drift = self.x[t - 1, self.idx_beta]
            elif self.trend_mode == "deterministic":
                drift = self.slope_value
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
            prev = self.x[t - 1, gs : ge + 1]
            mean_new_first = -float(np.sum(prev))
            ss += (self.x[t, ge] - mean_new_first) ** 2
        return float(ss), self.T

    # ----------------------- Conditional SMC (PGAS) ----------------------- #
    @staticmethod
    def _safe_normalize(p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, float)
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        p[p < 0.0] = 0.0
        s = float(np.sum(p))
        if not np.isfinite(s) or s <= 0.0:
            return np.full_like(p, 1.0 / max(1, p.size))
        p /= s
        s2 = float(np.sum(p))
        if not np.isclose(s2, 1.0, atol=1e-12):
            p /= s2
        return p

    @staticmethod
    def _ess(w: np.ndarray) -> float:
        s2 = float(np.sum(w * w))
        return (1.0 / s2) if s2 > 0 else 0.0

    @staticmethod
    def _ema(old: Optional[float], new: float, alpha: float = 0.1) -> float:
        return alpha * new + (1.0 - alpha) * (0.0 if old is None else old)

    def _conditional_pgas(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
        N, T, D = self.cfg.n_particles, self.T, self.dim
        parts = np.zeros((T + 1, N, D), float) if D > 0 else np.zeros((T + 1, N, 0))
        w = np.zeros((T + 1, N), float)
        a = np.zeros((T + 1, N), int)
        logZ = 0.0

        ess_list: List[float] = []
        maxw_list: List[float] = []
        resample_count = 0

        if D > 0:
            parts[0, :, :] = self.x[0]

        # t = 1
        for n in range(N - 1):
            if D > 0:
                parts[1, n, :] = self._transition_sample(parts[0, n, :], t=1)
            a[1, n] = n
        if D > 0:
            parts[1, N - 1, :] = self.x[1].copy()

        lw = np.zeros(N, float)
        for n in range(N):
            mu = self.mu_from_state(parts[1, n, :] if D > 0 else np.zeros(0), t=0)
            lw[n] = gev_logpdf(self.y[0], mu, self.sigma, self.xi)

        # AS for reference at t=1
        if D > 0:
            logf = np.array([self._transition_logpdf(parts[0, j, :], parts[1, N - 1, :], t=1) for j in range(N)], float)
            post = self._safe_normalize(np.exp(logf - np.max(logf)))
            a[1, N - 1] = np.random.choice(N, p=post)
        else:
            a[1, N - 1] = N - 1

        lw_max = np.max(lw)
        logZ += lw_max + math.log(np.mean(np.exp(lw - lw_max)) + 1e-300)
        w[1, :] = self._safe_normalize(np.exp(lw - lw_max))
        ess_list.append(self._ess(w[1, :]))
        maxw_list.append(float(np.max(w[1, :])))

        if self.cfg.progress:
            eta = self.cfg.ess_threshold_frac
            print(f"  Running conditional PGAS (bootstrap, ESS-triggered; η={eta:.2f}, N={N})")

        # t = 2..T
        it = tqdm(range(2, T + 1)) if self.cfg.progress else range(2, T + 1)
        for t in it:
            res_p = self._safe_normalize(w[t - 1, :])
            ess_prev = self._ess(res_p)
            thresh = self.cfg.ess_threshold_frac * N
            do_resample = (ess_prev < thresh)

            if do_resample:
                resample_count += 1
                anc = np.random.choice(N, size=N - 1, p=res_p, replace=True)
            else:
                anc = np.arange(N - 1, dtype=int)

            for n in range(N - 1):
                a[t, n] = anc[n]
                if D > 0:
                    prev = parts[t - 1, a[t, n], :]
                    parts[t, n, :] = self._transition_sample(prev, t=t)

            if D > 0:
                x_ref_t = self.x[t].copy()
                parts[t, N - 1, :] = x_ref_t
                logw_prev = np.log(np.clip(w[t - 1, :], 1e-300, None))
                logf = np.array([self._transition_logpdf(parts[t - 1, j, :], x_ref_t, t=t) for j in range(N)], float)
                post = self._safe_normalize(np.exp((logw_prev + logf) - np.max(logw_prev + logf)))
                a[t, N - 1] = np.random.choice(N, p=post)
            else:
                a[t, N - 1] = N - 1

            # weights
            y_idx = t - 1
            lw = np.zeros(N, float)
            for n in range(N):
                mu = self.mu_from_state(parts[t, n, :] if D > 0 else np.zeros(0), t=y_idx)
                lw[n] = gev_logpdf(self.y[y_idx], mu, self.sigma, self.xi)

            lw_eff = lw if do_resample else (lw + np.log(np.clip(w[t - 1, :], 1e-300, None)))
            lw_eff_max = np.max(lw_eff)
            logZ += lw_eff_max + math.log(np.mean(np.exp(lw_eff - lw_eff_max)) + 1e-300)
            w[t, :] = self._safe_normalize(np.exp(lw_eff - lw_eff_max))

            ess_t = self._ess(w[t, :])
            maxw_t = float(np.max(w[t, :]))
            ess_list.append(ess_t)
            maxw_list.append(maxw_t)
            if self.cfg.progress and hasattr(it, "set_postfix_str"):
                decision = "resamp" if do_resample else "skip"
                it.set_postfix_str(f"ESS_prev={ess_prev:6.1f} -> {decision} | ESS={ess_t:6.1f} MaxW={maxw_t:7.4f}")

        steps_considered = max(1, T - 1)
        pf_diag = {
            "ess_mean": float(np.mean(ess_list)),
            "ess_min": float(np.min(ess_list)),
            "maxw_mean": float(np.mean(maxw_list)),
            "maxw_max": float(np.max(maxw_list)),
            "resample_count": int(resample_count),
            "resample_rate": float(resample_count / steps_considered),
        }
        return parts, w, a, float(logZ), pf_diag

    def _trace_single_trajectory(self, parts: np.ndarray, a: np.ndarray, w: np.ndarray) -> np.ndarray:
        N, T, D = self.cfg.n_particles, self.T, self.dim
        if D == 0:
            return self.x
        idx = np.zeros(T + 1, dtype=int)
        idx[T] = np.random.choice(N, p=w[T, :])
        for t in range(T, 1, -1):
            idx[t - 1] = a[t, idx[t]]
        x_new = self.x.copy()
        x_new[0, :] = parts[0, 0, :] if D > 0 else self.x[0, :]
        for t in range(1, T + 1):
            x_new[t, :] = parts[t, idx[t], :]
        return x_new

    def update_states_pgas(self) -> None:
        parts, w, a, logZ, pf_diag = self._conditional_pgas()
        self.x = self._trace_single_trajectory(parts, a, w)
        self.last_log_evidence = float(logZ)
        self.last_pf_diag = pf_diag

    # ----------------------- Half-Cauchy mixture updates for Q ----------------------- #
    @staticmethod
    def _sample_invgamma(shape: float, scale: float) -> float:
        # InvGamma(shape, scale): 1/X ~ Gamma(shape, 1/scale)
        return 1.0 / np.random.gamma(shape, 1.0 / scale)

    def _gibbs_Q_block(self, label: str, SS: float, T_eff: int, A_scale: float, idx: Optional[int]) -> None:
        if idx is None or T_eff <= 0:
            return
        # Prior: Q | a ~ InvGamma(1/2, 1/a),  a ~ InvGamma(1/2, 1/A^2)
        a_key = "alpha" if label == "alpha" else ("beta" if label == "beta" else "gamma")
        a_cur = float(self.a_aux[a_key])
        # Q | a, x
        shape_Q = 0.5 * T_eff + 0.5
        scale_Q = 0.5 * SS + 1.0 / max(a_cur, 1e-300)
        Q_draw = self._sample_invgamma(shape_Q, scale_Q)
        Q_draw = max(Q_draw, 0.0)
        self.Q[idx] = Q_draw
        # a | Q
        shape_a = 1.0
        scale_a = (1.0 / (A_scale * A_scale)) + (1.0 / max(Q_draw, 1e-300))
        self.a_aux[a_key] = self._sample_invgamma(shape_a, scale_a)

    def update_process_Q_halfcauchy(self) -> None:
        # α
        if self.idx_alpha is not None:
            SS, T_eff = self._innovation_ss_alpha()
            self._gibbs_Q_block("alpha", SS, T_eff, float(self.priors.hc_scale_alpha), self.idx_alpha)
        # β
        if self.idx_beta is not None:
            SS, T_eff = self._innovation_ss_beta()
            self._gibbs_Q_block("beta", SS, T_eff, float(self.priors.hc_scale_beta), self.idx_beta)
        # γ (last only)
        if self.seasonal_mode == "dynamic":
            SS, T_eff = self._innovation_ss_gamma()
            self._gibbs_Q_block("gamma", SS, T_eff, float(self.priors.hc_scale_gamma), self.idx_g_end)

    # ----------------------- Observation & deterministic MH ------------------ #
    def _get_step(self, key: str) -> float:
        c = self.cfg
        return {
            "logsigma": c.step_logsigma, "xi": c.step_xi,
            "level": c.step_level, "slope": c.step_slope, "season": c.step_season
        }[key]

    def _set_step(self, key: str, val: float) -> None:
        v = float(np.clip(val, self.cfg.step_min, self.cfg.step_max))
        if   key == "logsigma": self.cfg.step_logsigma = v
        elif key == "xi":       self.cfg.step_xi = v
        elif key == "level":    self.cfg.step_level = v
        elif key == "slope":    self.cfg.step_slope = v
        elif key == "season":   self.cfg.step_season = v
        else: raise KeyError(key)

    def _mh_accept(self, logacc: float) -> bool:
        return (np.log(np.random.rand()) < min(0.0, logacc))

    def _mu_vec_current(self) -> np.ndarray:
        return np.array([self.mu_from_state(self.x[t], t - 1) for t in range(1, self.T + 1)], float)

    def update_logsigma(self) -> None:
        step = self._get_step("logsigma")
        cur = self.logsigma
        prop = cur + np.random.normal(0.0, step)
        sig_cur, sig_prop = float(np.exp(cur)), float(np.exp(prop))
        mu_vec = self._mu_vec_current()
        ll_old = gev_loglike_sum(self.y, mu_vec, sig_cur, self.xi)
        ll_new = gev_loglike_sum(self.y, mu_vec, sig_prop, self.xi)
        self.proposals["logsigma"] += 1
        if ll_new == -np.inf:
            return
        lp_old = -0.5 * ((cur - self.priors.m_sigma) ** 2) / (self.priors.s_sigma ** 2)
        lp_new = -0.5 * ((prop - self.priors.m_sigma) ** 2) / (self.priors.s_sigma ** 2)
        if self._mh_accept((ll_new + lp_new) - (ll_old + lp_old)):
            self.logsigma = prop
            self.sigma = sig_prop
            self.accept["logsigma"] += 1

    def update_xi(self) -> None:
        step = self._get_step("xi")
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
        if self._mh_accept((ll_new + lp_new) - (ll_old + lp_old)):
            self.xi = prop
            self.accept["xi"] += 1

    def update_level_value(self) -> None:
        if self.level_mode != "deterministic": return
        step = self._get_step("level")
        cur = self.level_value
        prop = cur + np.random.normal(0.0, step)

        old = self.level_value
        self.level_value = prop
        mu_vec_prop = self._mu_vec_current()
        self.level_value = old
        mu_vec_old = self._mu_vec_current()

        ll_old = gev_loglike_sum(self.y, mu_vec_old, self.sigma, self.xi)
        ll_new = gev_loglike_sum(self.y, mu_vec_prop, self.sigma, self.xi)
        self.proposals["level"] += 1
        if ll_new == -np.inf:
            return
        lp_old = -0.5 * ((cur - self.priors.m_level) ** 2) / (self.priors.s_level ** 2)
        lp_new = -0.5 * ((prop - self.priors.m_level) ** 2) / (self.priors.s_level ** 2)
        if self._mh_accept((ll_new + lp_new) - (ll_old + lp_old)):
            self.level_value = prop
            self.accept["level"] += 1

    def _alpha_transition_loglike_given_slope(self, slope: float) -> float:
        if self.idx_alpha is None:
            return 0.0
        var = self.Q[self.idx_alpha] if self.Q[self.idx_alpha] > 0.0 else self.cfg.trans_eps
        inv_var = 1.0 / var
        cst = -0.5 * np.log(2.0 * np.pi * var)
        ll = 0.0
        for t in range(1, self.T + 1):
            drift = (self.x[t - 1, self.idx_beta] if self.idx_beta is not None else slope)
            mean = self.x[t - 1, self.idx_alpha] + drift
            diff = self.x[t, self.idx_alpha] - mean
            ll += cst - 0.5 * diff * diff * inv_var
        return float(ll)

    def update_slope(self) -> None:
        if self.trend_mode != "deterministic": return
        step = self._get_step("slope")
        cur = self.slope_value
        prop = cur + np.random.normal(0.0, step)

        if self.level_mode == "deterministic":
            mu_vec_old = self._mu_vec_current()
            self.slope_value = prop
            mu_vec_new = self._mu_vec_current()
            self.slope_value = cur
            ll_obs_old = gev_loglike_sum(self.y, mu_vec_old, self.sigma, self.xi)
            ll_obs_new = gev_loglike_sum(self.y, mu_vec_new, self.sigma, self.xi)
            if ll_obs_new == -np.inf:
                self.proposals["slope"] += 1
                return
        else:
            ll_obs_old = 0.0; ll_obs_new = 0.0

        if self.idx_alpha is not None:
            ll_tr_old = self._alpha_transition_loglike_given_slope(cur)
            ll_tr_new = self._alpha_transition_loglike_given_slope(prop)
        else:
            ll_tr_old = ll_tr_new = 0.0

        lp_old = -0.5 * ((cur - self.priors.m_slope) ** 2) / (self.priors.s_slope ** 2)
        lp_new = -0.5 * ((prop - self.priors.m_slope) ** 2) / (self.priors.s_slope ** 2)

        self.proposals["slope"] += 1
        if self._mh_accept((ll_obs_new + ll_tr_new + lp_new) - (ll_obs_old + ll_tr_old + lp_old)):
            self.slope_value = prop
            self.accept["slope"] += 1

    def update_season_vec(self) -> None:
        if self.seasonal_mode != "deterministic": return
        step = self._get_step("season")
        v_cur = self.season_vec.copy()
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

        m_first = self.priors.m_season
        if m_first is None:
            m_first = np.zeros(self.period - 1, float)
        else:
            m_first = np.asarray(m_first, float)
            if m_first.size != self.period - 1:
                raise ValueError("priors.m_season must have length p-1.")
        s = float(self.priors.s_season)

        lp_old = -0.5 * np.sum(((v_cur[:-1] - m_first) / s) ** 2)
        lp_new = -0.5 * np.sum(((prop[:-1] - m_first) / s) ** 2)
        if self._mh_accept((ll_new + lp_new) - (ll_old + lp_old)):
            self.season_vec = prop
            self.accept["season"] += 1

    # ------------------- Adaptive step-size for MH ------------------ #
    def _adapt_steps(self, it: int) -> None:
        cfg = self.cfg
        if not cfg.adapt_steps:
            return
        in_window = (cfg.adapt_until == "all") or (it < cfg.burn)
        if (it + 1) % max(1, cfg.adapt_every) != 0 or (not in_window):
            return
        k = self._adapt_round
        eta = cfg.adapt_eta0 / ((1.0 + k) ** cfg.adapt_eta_decay)
        keys: List[str] = ["logsigma", "xi"]
        if self.level_mode == "deterministic":   keys.append("level")
        if self.trend_mode == "deterministic":   keys.append("slope")
        if self.seasonal_mode == "deterministic": keys.append("season")
        target = cfg.adapt_target_1d
        for key in keys:
            acc_now = self.accept[key]; prop_now = self.proposals[key]
            acc_win = acc_now - self._mh_prev_acc[key]
            prop_win = prop_now - self._mh_prev_prop[key]
            if prop_win <= 0: continue
            rate = acc_win / max(1, prop_win)
            s = self._get_step(key)
            s_new = s * np.exp(eta * (rate - target))
            self._set_step(key, s_new)
            self._mh_prev_acc[key] = acc_now
            self._mh_prev_prop[key] = prop_now
        self._adapt_round += 1

    # ----------------------------- Progress ----------------------------- #
    def _acc_pct(self, key: str) -> str:
        a, p = self.accept.get(key, 0), self.proposals.get(key, 0)
        return f"{100.0 * a / p:0.1f}%" if p > 0 else "0.0%"

    def _format_Q_slot(self, label: str, idx: Optional[int]) -> str:
        if idx is None:
            return f"Q_{label}=/"
        return f"Q_{label}={self.Q[idx]:.3e}"

    def _format_det_season(self) -> str:
        if self.season_vec is None:
            return "γ=[]"
        v = self.season_vec
        if v.size <= 4:
            inside = ", ".join(f"{z:.3f}" for z in v)
        else:
            inside = ", ".join(f"{z:.3f}" for z in v[:4]) + ", …"
        return f"γ=[{inside}]"

    def _progress_line(self, it: int, ema_logZ: Optional[float]) -> str:
        obs = f"| σ={np.exp(self.logsigma):.3f} ({self._acc_pct('logsigma')}) ξ={self.xi:.3f} ({self._acc_pct('xi')})"
        q_alpha = self._format_Q_slot("α", self.idx_alpha) if self.idx_alpha is not None else "Q_α=/"
        q_beta  = self._format_Q_slot("β", self.idx_beta)  if self.idx_beta  is not None else "Q_β=/"
        q_gamma = self._format_Q_slot("γ", self.idx_g_end) if self.seasonal_mode == "dynamic" else "Q_γ=/"
        det = []
        if self.level_mode == "deterministic": det.append(f"α={self.level_value:.4f} ({self._acc_pct('level')})")
        if self.trend_mode == "deterministic": det.append(f"β={self.slope_value:.4f} ({self._acc_pct('slope')})")
        if self.seasonal_mode == "deterministic": det.append(self._format_det_season() + f" ({self._acc_pct('season')})")
        det_block = ("| " + " ".join(det)) if det else ""
        pf = ""
        if self.last_pf_diag:
            d = self.last_pf_diag
            pf = (f"| PF: ESS(mean/min)={d['ess_mean']:.1f}/{d['ess_min']:.1f} "
                  f"MaxW(max)={d['maxw_max']:.4f} RR={100.0*d.get('resample_rate',0.0):.1f}%")
        head = (f"[it {it+1}/{self.cfg.n_iter}] logZ={self.last_log_evidence:.3f} "
                f"ema={float(ema_logZ) if ema_logZ is not None else float('nan'):.3f}")
        return f"{head} | {obs.split('|')[1].strip()} | {q_alpha}, {q_beta}, {q_gamma} {det_block} {pf}".rstrip()

    # --------------------------------- MCMC --------------------------------- #
    def run(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        save_iters = list(range(cfg.burn, cfg.n_iter, cfg.thin))
        n_kept = len(save_iters)
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
        print_every = cfg.progress_every if cfg.progress_every > 0 else max(1, cfg.n_iter // 50)

        for it in range(cfg.n_iter):
            if cfg.progress:
                print(f"Iteration {it + 1}/{cfg.n_iter}")

            # 1) latent states (PGAS)
            if self.dim > 0:
                self.update_states_pgas()
                current_log_ev = float(self.last_log_evidence)
            else:
                mu_now = self._mu_vec_current()
                current_log_ev = float(gev_loglike_sum(self.y, mu_now, self.sigma, self.xi))

            ema_logZ = self._ema(ema_logZ, current_log_ev, alpha=0.1)

            # 2) process Q via Half-Cauchy IG mixtures (Gibbs)
            if self.dim > 0:
                self.update_process_Q_halfcauchy()

            # 3) deterministic parameters (MH)
            if self.level_mode == "deterministic":   self.update_level_value()
            if self.trend_mode == "deterministic":   self.update_slope()
            if self.seasonal_mode == "deterministic": self.update_season_vec()

            # 4) observation params (MH)
            self.update_logsigma()
            self.update_xi()

            # 5) adapt MH steps
            self._adapt_steps(it)

            # 6) progress
            if cfg.progress and ((it + 1) % print_every == 0 or it == cfg.n_iter - 1):
                print(self._progress_line(it, ema_logZ))

            # 7) store
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
                    self.keep["gamma_t"][keep_idx, :] = self.x[1:self.T + 1, self.idx_g_end]
                if self.level_mode == "deterministic":
                    self.keep["level_value"][keep_idx] = self.level_value
                if self.trend_mode == "deterministic":
                    self.keep["slope_value"][keep_idx] = self.slope_value
                if self.seasonal_mode == "deterministic":
                    self.keep["season_vector"][keep_idx, :] = self.season_vec
                keep_idx += 1

        if cfg.progress:
            print(self._progress_line(cfg.n_iter - 1, ema_logZ))
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
            "idx_g_start": self.idx_g_start,
            "idx_g_end": self.idx_g_end,
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

# ------------------------- CLI / Example run ------------------------ #
if __name__ == "__main__":
    import argparse
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from simulator.extremal_time_series import Extremal_Time_Series

    p = argparse.ArgumentParser(description="DGEV PGAS with Half-Cauchy (IG-mixture) process noise priors")

    # Modes
    p.add_argument("--level-mode", choices=["dynamic", "deterministic"], default="dynamic")
    p.add_argument("--trend-mode", choices=["dynamic", "deterministic", "none"], default="dynamic")
    p.add_argument("--season-mode", choices=["dynamic", "deterministic", "none"], default="dynamic")

    # Basics
    p.add_argument("--period", type=int, default=4)
    p.add_argument("--T", type=int, default=500)

    # Initial values (shared)
    p.add_argument("--level-init", type=float, default=5.0)
    p.add_argument("--slope-init", type=float, default=0.02)

    # Truth for simulator
    p.add_argument("--true-sigma", type=float, default=2.0)
    p.add_argument("--true-xi", type=float, default=0.1)
    p.add_argument("--q-alpha", type=float, default=1e-1)
    p.add_argument("--q-beta",  type=float, default=1e-5)
    p.add_argument("--q-gamma", type=float, default=1e-7)

    # Observation priors
    p.add_argument("--prior-m-sigma", type=float, default=1.0)
    p.add_argument("--prior-s-sigma", type=float, default=1.0)
    p.add_argument("--prior-m-xi", type=float, default=0.0)
    p.add_argument("--prior-s-xi", type=float, default=0.2)

    # Deterministic structural priors
    p.add_argument("--prior-m-level", type=float, default=0.0)
    p.add_argument("--prior-s-level", type=float, default=10.0)
    p.add_argument("--prior-m-slope", type=float, default=0.0)
    p.add_argument("--prior-s-slope", type=float, default=10.0)
    p.add_argument("--prior-m-season", type=str, default=None, help="Comma-separated first (p−1) means (deterministic seasonal).")
    p.add_argument("--prior-s-season", type=float, default=5.0)

    # Half-Cauchy scales for process SDs
    p.add_argument("--hc-scale-alpha", type=float, default=0.5)
    p.add_argument("--hc-scale-beta",  type=float, default=0.5)
    p.add_argument("--hc-scale-gamma", type=float, default=0.5)

    # Sampler config
    p.add_argument("--n-iter", type=int, default=4000)
    p.add_argument("--burn", type=int, default=1000)
    p.add_argument("--thin", type=int, default=1)
    p.add_argument("--step-logsigma", type=float, default=0.2)
    p.add_argument("--step-xi", type=float, default=0.2)
    p.add_argument("--step-level", type=float, default=0.02)
    p.add_argument("--step-slope", type=float, default=0.02)
    p.add_argument("--step-season", type=float, default=0.02)

    p.add_argument("--particles", type=int, default=250)
    p.add_argument("--trans-eps", type=float, default=1e-8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--progress", default=True)
    p.add_argument("--progress-every", type=int, default=10)

    # ESS trigger
    p.add_argument("--ess-frac", type=float, default=0.5)

    # Adaptive RW–MH
    p.add_argument("--adapt-steps", default=True)
    p.add_argument("--adapt-every", type=int, default=25)
    p.add_argument("--adapt-until", choices=["burn", "all"], default="burn")
    p.add_argument("--adapt-eta0", type=float, default=0.2)
    p.add_argument("--adapt-decay", type=float, default=0.75)
    p.add_argument("--adapt-target-1d", type=float, default=0.44)
    p.add_argument("--step-min", type=float, default=1e-5)
    p.add_argument("--step-max", type=float, default=1.0)

    # Output
    p.add_argument("--out-dir", type=str, default=None)

    args = p.parse_args()
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
        q_level=args.q_alpha, q_trend=args.q_beta, q_season=args.q_gamma,
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

    truths: Dict[str, Any] = ts.get_truth_paths(as_numpy=False)
    # Robust seasonal key to avoid KeyError ('gamma_last' vs 'gamma')
    gamma_key = "gamma_last" if "gamma_last" in truths else ("gamma" if "gamma" in truths else None)

    mu_T    = np.asarray(truths["mu"][1:1 + args.T], float)
    alpha_T = np.asarray(truths["alpha"][1:1 + args.T], float) if sim_level_mode == "dynamic" else None
    beta_T  = np.asarray(truths["beta"][1:1 + args.T], float)  if sim_trend_mode == "dynamic" else None
    gamma_T = (np.asarray(truths[gamma_key][1:1 + args.T], float) if (sim_season_mode == "dynamic" and gamma_key is not None) else None)

    pri_m_season = parse_csv_floats(args.prior_m_season)
    if pri_m_season is not None and len(pri_m_season) != args.period - 1:
        raise ValueError(f"--prior-m-season must have length {args.period - 1}.")

    priors = Priors(
        m_sigma=float(args.prior_m_sigma), s_sigma=float(args.prior_s_sigma),
        m_xi=float(args.prior_m_xi), s_xi=float(args.prior_s_xi),
        m_level=float(args.prior_m_level), s_level=float(args.prior_s_level),
        m_slope=float(args.prior_m_slope), s_slope=float(args.prior_s_slope),
        m_season=pri_m_season, s_season=float(args.prior_s_season),
        hc_scale_alpha=float(args.hc_scale_alpha),
        hc_scale_beta=float(args.hc_scale_beta),
        hc_scale_gamma=float(args.hc_scale_gamma),
    )

    cfg = SamplerConfig(
        n_iter=args.n_iter, burn=args.burn, thin=args.thin,
        step_logsigma=args.step_logsigma, step_xi=args.step_xi,
        step_level=args.step_level, step_slope=args.step_slope, step_season=args.step_season,
        n_particles=args.particles, trans_eps=args.trans_eps,
        random_seed=args.seed, progress=bool(args.progress),
        progress_every=int(args.progress_every),
        ess_threshold_frac=float(args.ess_frac),
        adapt_steps=bool(args.adapt_steps),
        adapt_every=int(args.adapt_every),
        adapt_until=str(args.adapt_until),
        adapt_eta0=float(args.adapt_eta0),
        adapt_eta_decay=float(args.adapt_decay),
        adapt_target_1d=float(args.adapt_target_1d),
        step_min=float(args.step_min), step_max=float(args.step_max),
    )

    seasonal_init_pminus1 = (
        np.asarray(pri_m_season, float) if (sim_season_mode == "deterministic" and pri_m_season is not None)
        else (build_seasonal(args.period) if sim_season_mode == "deterministic" else None)
    )

    sampler = DGEVParticleGibbsHC(
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
    _ensure_dir(out_dir)

    t0 = time.time()
    posterior = sampler.run()
    elapsed = time.time() - t0
    print(f"Sampler run time: {elapsed:.2f}s")

    sampler.save_posterior(
        out_npz_path=os.path.join(out_dir, "posterior.npz"),
        extra_meta={"modes": tag, "elapsed_seconds": float(elapsed)},
    )

    print(f"Posterior mean sigma: {np.mean(posterior['sigma']):.3f} (true {args.true_sigma})")
    print(f"Posterior mean xi:    {np.mean(posterior['xi']):.3f} (true {args.true_xi})")

    if sampler.true_Q is not None:
        tq = np.asarray(sampler.true_Q, float)
        if sampler.idx_alpha is not None and tq.size > sampler.idx_alpha:
            print(f"True Q_alpha:       {tq[sampler.idx_alpha]:.6g}")
        if sampler.idx_beta is not None and tq.size > sampler.idx_beta:
            print(f"True Q_beta:        {tq[sampler.idx_beta]:.6g}")
        if sampler.seasonal_mode == "dynamic" and sampler.idx_g_end is not None and tq.size > sampler.idx_g_end:
            print(f"True Q_gamma(last): {tq[sampler.idx_g_end]:.6g}")

    if "Q" in posterior and sampler.dim > 0 and posterior["Q"].size > 0:
        if sampler.idx_alpha is not None:
            print(f"Post mean Q_alpha:  {np.mean(posterior['Q'][:, sampler.idx_alpha]):.6g}")
        if sampler.idx_beta is not None:
            print(f"Post mean Q_beta:   {np.mean(posterior['Q'][:, sampler.idx_beta]):.6g}")
        if sampler.seasonal_mode == "dynamic":
            print(f"Post mean Q_gamma:  {np.mean(posterior['Q'][:, sampler.idx_g_end]):.6g}")

    if "log_evidence" in posterior and posterior["log_evidence"].size > 0:
        le = posterior["log_evidence"]
        print(f"Mean log p(y|θ): {np.nanmean(le):.3f} | Median: {np.nanmedian(le):.3f} | Best: {np.nanmax(le):.3f}")
