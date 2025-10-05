# %% optimization/dlmgibbs.py
from __future__ import annotations

import os, math, json, time
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, Dict, List, Sequence, Callable

import numpy as np
from numpy.linalg import inv
from datetime import datetime

# =============================================================================
# Utilities
# =============================================================================

def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

def _mad(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    med = np.median(v)
    return float(np.median(np.abs(v - med)))

def _robust_sd(v: np.ndarray) -> float:
    return _mad(np.asarray(v, float)) / 1.4826 if v.size else 0.0

# =============================================================================
# Priors & Config (Normal on logsigma & deterministic params; PC priors on sds)
# with Gamma hyperpriors on the PC rates lambda
# =============================================================================

@dataclass
class PCPrior:
    """
    PC prior for innovation sd s>0: p(s | lambda) = lambda * exp(-lambda * s),
    with optional Gamma(a_lambda, b_lambda) hyperprior (shape–rate) on lambda.
    If lambda_s is None, we sample lambda ~ Gamma; initialization can be
    auto-calibrated from data using frac*robust_scale and alpha_prob via
        lambda_init = -log(alpha_prob) / (frac * scale)
    """
    # Fixed lambda (if provided). If None -> lambda is random with Gamma hyperprior.
    lambda_s: Optional[float] = None
    # Hyperprior for lambda (shape–rate). Only used if lambda_s is None.
    a_lambda: float = 1.0
    b_lambda: float = 1.0
    # Calibration helpers for initial lambda value when lambda_s is None
    frac: float = 0.10
    alpha_prob: float = 0.05

@dataclass
class Priors:
    # Observation: Normal prior on log σ
    m_sigma: float = 0.0
    s_sigma: float = 1.0

    # Deterministic parameters (Normal)
    m_level: float = 0.0
    s_level: float = 10.0
    m_slope: float = 0.0
    s_slope: float = 10.0

    # Deterministic seasonal prior (first p-1 entries; last implied)
    m_season: Optional[Sequence[float]] = None
    s_season: float = 5.0

    # PC priors (with hyperpriors) for state-noise sds s = sqrt(q)
    pc_alpha: PCPrior = field(default_factory=PCPrior)
    pc_beta:  PCPrior = field(default_factory=PCPrior)
    pc_gamma: PCPrior = field(default_factory=PCPrior)

@dataclass
class SamplerConfig:
    n_iter: int = 5000
    burn: int = 1000
    thin: int = 5

    random_seed: Optional[int] = 123
    progress: bool = True
    progress_every: int = 0  # 0 => auto (~2% of n_iter)

    # RW–MH step sizes (initial)
    step_logsigma: float = 0.05
    step_level: float = 0.05
    step_slope: float = 0.02
    step_season: float = 0.05

    # steps for log-sd proposals (PC priors) where s = exp(z)
    step_log_s_alpha: float = 0.10
    step_log_s_beta:  float = 0.10
    step_log_s_gamma: float = 0.10

    # ---- Adaptive RW–MH options (Robbins–Monro; windowed) ----
    adapt_steps: bool = True
    adapt_every: int = 25
    adapt_until: str = "burn"         # "burn" or "all"
    adapt_target_1d: float = 0.44
    adapt_eta0: float = 0.05
    adapt_eta_decay: float = 0.75
    step_min: float = 1e-5
    step_max: float = 1.0

# =============================================================================
# DLM with FFBS + RW–MH (logsigma, deterministic params, process sds)
# and Gamma hyperpriors on PC rates lambda
# =============================================================================

class DLMGibbs:
    """
    Gaussian structural DLM:

        y_t = mu_t + eps_t,  eps_t ~ N(0, sigma^2)

    Components:
      level_mode    ∈ {"dynamic","deterministic"}
      trend_mode    ∈ {"dynamic","deterministic","none"}
      seasonal_mode ∈ {"dynamic","deterministic","none"}

    Dynamic latent layout (if present): [alpha][beta][g1..g_{p-1}]
      seasonal dynamics: shift + closure g_{p-1,t+1} = -sum(g_{1..p-1,t}) + N(0,q_gamma)

    FFBS samples latent x_{0:T}; RW–MH updates:
      - logsigma (Normal prior)
      - deterministic level/slope/season (Normal priors)
      - process sds s_alpha, s_beta, s_gamma with PC priors on s (exp-rate lambda)
      - NEW: lambda_alpha, lambda_beta, lambda_gamma ~ Gamma(a_lambda,b_lambda) (Gibbs)
    """

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
        m0_season: Optional[Sequence[float]] = None,
        v0_season: Optional[Sequence[float]] = None,
        # Initial values (deterministic)
        level_value_init: float = 0.0,
        slope_value_init: float = 0.0,
        seasonal_vector_init: Optional[Sequence[float]] = None,  # first p-1 entries
        # Initial variances (used to start MH chains)
        sigma2_init: float = 1.0,
        q_alpha_init: float = 1e-3,
        q_beta_init:  float = 1e-9,
        q_gamma_init: float = 1e-7,
        # Priors & config
        priors: Priors = Priors(),
        cfg: SamplerConfig = SamplerConfig(),
    ):
        # Data
        self.y = np.asarray(y, float)
        self.T = int(self.y.size)
        self.period = int(period)
        assert self.period >= 2

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

        # ---- State layout
        layout: List[str] = []
        if self.level_mode == "dynamic": layout.append("alpha")
        if self.trend_mode == "dynamic": layout.append("beta")
        if self.seasonal_mode == "dynamic": layout.extend([f"g{k}" for k in range(1, self.period)])
        self._layout = layout
        self.dim = len(layout)

        self.idx_alpha = layout.index("alpha") if "alpha" in layout else None
        self.idx_beta  = layout.index("beta")  if "beta"  in layout else None
        if self.seasonal_mode == "dynamic":
            self.idx_g_start = layout.index("g1") if "g1" in layout else None
            self.idx_g_end   = self.idx_g_start + (self.period - 2) if self.idx_g_start is not None else None
        else:
            self.idx_g_start = None
            self.idx_g_end = None

        # ---- Deterministic components
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
                    g_first = np.zeros(self.period - 1, float)
            g_last = -np.sum(g_first)
            self.season_vec = np.concatenate([g_first, [g_last]]).astype(float)
        else:
            self.season_vec = None

        # ---- Observation variance and process sds (store both q and s)
        self.logsigma = 0.5 * math.log(max(1e-12, sigma2_init))
        self.sigma2   = float(np.exp(2.0 * self.logsigma))

        self.s_alpha = math.sqrt(max(1e-18, q_alpha_init)) if self.idx_alpha is not None else 0.0
        self.s_beta  = math.sqrt(max(1e-18, q_beta_init))  if self.idx_beta  is not None else 0.0
        self.s_gamma = math.sqrt(max(1e-18, q_gamma_init)) if self.seasonal_mode == "dynamic" else 0.0

        # ---- Initialize PC lambdas (fixed or random-with-hyperprior)
        self.lambda_alpha, self.lambda_beta, self.lambda_gamma = self._init_pc_lambdas()

        # ---- Initial latent path x_{0:T}
        self.x = np.zeros((self.T + 1, self.dim), float)
        if self.dim > 0:
            m0_list, v0_list = [], []
            if self.idx_alpha is not None:
                m0_list.append(float(m0_level)); v0_list.append(float(v0_level))
            if self.idx_beta is not None:
                m0_list.append(float(m0_trend)); v0_list.append(float(v0_trend))
            if self.seasonal_mode == "dynamic":
                m0_season = np.zeros(self.period - 1) if m0_season is None else np.asarray(m0_season, float)
                v0_season = np.ones(self.period - 1)  if v0_season is None else np.asarray(v0_season, float)
                if m0_season.size != self.period - 1 or v0_season.size != self.period - 1:
                    raise ValueError("m0_season and v0_season must have length p-1 in dynamic mode.")
                for k in range(self.period - 1):
                    m0_list.append(float(m0_season[k])); v0_list.append(float(v0_season[k]))
            self.m0 = np.array(m0_list, float)
            self.C0 = np.diag(np.maximum(1e-10, np.array(v0_list, float)))
            self.x[0] = np.random.multivariate_normal(self.m0, self.C0)
            self._propagate_initial_path(Q_init=np.ones(self.dim) * 1e-6)
        else:
            self.m0 = np.array([], float)
            self.C0 = np.zeros((0, 0), float)

        # ---- Storage (filled after knowing n_kept in run())
        self.keep: Dict[str, np.ndarray] = {}

        # ---- MH bookkeeping
        self.accept = {
            "logsigma": 0, "level": 0, "slope": 0, "season": 0,
            "log_s_alpha": 0, "log_s_beta": 0, "log_s_gamma": 0
        }
        self.proposals = dict(self.accept)

        # ---- adaptation bookkeeping (windowed deltas)
        self._mh_prev_acc = dict(self.accept)
        self._mh_prev_prop = dict(self.proposals)
        self._adapt_round = 0

        # ---- Truth overlays (optional)
        self.true_sigma: Optional[float] = None
        self.true_Q: Optional[np.ndarray] = None
        self.true_mu_t: Optional[np.ndarray] = None
        self.true_alpha_t: Optional[np.ndarray] = None
        self.true_beta_t: Optional[np.ndarray] = None
        self.true_gamma_t: Optional[np.ndarray] = None

    # --------------------- Truth registration (optional) --------------------- #
    def set_truth(self, sigma: Optional[float] = None, Q: Optional[np.ndarray] = None) -> None:
        self.true_sigma = sigma
        self.true_Q = None if Q is None else np.asarray(Q, float)

    def set_truth_paths(self, mu: Optional[np.ndarray] = None,
                        alpha: Optional[np.ndarray] = None,
                        beta: Optional[np.ndarray] = None,
                        gamma: Optional[np.ndarray] = None) -> None:
        self.true_mu_t = None if mu is None else np.asarray(mu, float)
        self.true_alpha_t = None if alpha is None else np.asarray(alpha, float)
        self.true_beta_t = None if beta is None else np.asarray(beta, float)
        self.true_gamma_t = None if gamma is None else np.asarray(gamma, float)

    # --------------------- Initialize / auto-calibrate lambdas ---------------- #
    def _init_pc_lambdas(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        If user provides pc_*.lambda_s, we fix it.
        Otherwise, we initialize lambda via data-calibration:
            lambda_init = -log(alpha_prob) / (frac * scale)
        and then sample lambda each iteration using Gamma hyperprior.
        """
        y = self.y
        sd1 = _robust_sd(np.diff(y)) if y.size >= 2 else 0.0      # alpha
        sd2 = _robust_sd(np.diff(y, n=2)) if y.size >= 3 else 0.0 # beta
        sdg = sd1                                                 # gamma proxy

        def _cal(pc: PCPrior, sd: float) -> float:
            u = max(1e-12, pc.frac * max(1e-12, sd))
            return float(-math.log(max(1e-12, pc.alpha_prob)) / u)

        la = (float(self.priors.pc_alpha.lambda_s) if (self.priors.pc_alpha.lambda_s is not None)
              else _cal(self.priors.pc_alpha, sd1) if self.idx_alpha is not None else None)
        lb = (float(self.priors.pc_beta.lambda_s) if (self.priors.pc_beta.lambda_s is not None)
              else _cal(self.priors.pc_beta,  sd2) if self.idx_beta  is not None else None)
        lg = (float(self.priors.pc_gamma.lambda_s) if (self.priors.pc_gamma.lambda_s is not None)
              else _cal(self.priors.pc_gamma, sdg) if self.seasonal_mode == "dynamic" else None)

        if self.cfg.progress:
            la_s = f"{la:.4g}" if la is not None else "n/a"
            lb_s = f"{lb:.4g}" if lb is not None else "n/a"
            lg_s = f"{lg:.4g}" if lg is not None else "n/a"
            print(f"[init] PC lambda init: α={la_s}, β={lb_s}, γ={lg_s} (fixed if provided)")

        return la, lb, lg

    # ----------------------------- Helpers ---------------------------------- #
    def _H_t(self) -> np.ndarray:
        if self.dim == 0: return np.zeros((1, 0))
        h = np.zeros(self.dim, float)
        if self.idx_alpha is not None: h[self.idx_alpha] = 1.0
        if self.seasonal_mode == "dynamic": h[self.idx_g_end] = 1.0
        return h.reshape(1, -1)

    def _G_t(self) -> np.ndarray:
        if self.dim == 0: return np.zeros((0, 0))
        G = np.eye(self.dim, dtype=float)
        if (self.idx_alpha is not None) and (self.idx_beta is not None):
            G[self.idx_alpha, self.idx_beta] = 1.0
        if self.seasonal_mode == "dynamic":
            gs, ge = self.idx_g_start, self.idx_g_end
            for k in range(ge - gs):
                G[gs + k, gs + k] = 0.0
                G[gs + k, gs + k + 1] = 1.0
            G[ge, gs:ge + 1] = 0.0
        return G

    def _u_t(self, x_prev: np.ndarray) -> np.ndarray:
        if self.dim == 0: return np.zeros(0)
        u = np.zeros(self.dim, float)
        if (self.idx_alpha is not None) and (self.idx_beta is None) and (self.trend_mode == "deterministic"):
            u[self.idx_alpha] = self.slope_value
        if self.seasonal_mode == "dynamic":
            gs, ge = self.idx_g_start, self.idx_g_end
            prev = x_prev[gs:ge + 1]
            u[ge] = -float(np.sum(prev))
        return u

    def _Q_mat(self) -> np.ndarray:
        if self.dim == 0: return np.zeros((0, 0))
        Q = np.zeros((self.dim, self.dim), float)
        if self.idx_alpha is not None and self.s_alpha > 0.0: Q[self.idx_alpha, self.idx_alpha] = self.s_alpha ** 2
        if self.idx_beta  is not None and self.s_beta  > 0.0: Q[self.idx_beta,  self.idx_beta]  = self.s_beta  ** 2
        if self.seasonal_mode == "dynamic" and self.s_gamma > 0.0: Q[self.idx_g_end, self.idx_g_end] = self.s_gamma ** 2
        return Q

    def _deterministic_mu_t(self, t: int) -> float:
        out = 0.0
        if self.level_mode == "deterministic":
            out += self.level_value
            if self.trend_mode == "deterministic": out += self.slope_value * t
        if self.seasonal_mode == "deterministic":
            out += float(self.season_vec[t % self.period])
        return out

    def _mu_vec_current(self) -> np.ndarray:
        H = self._H_t()
        mu = np.zeros(self.T, float)
        for t in range(1, self.T + 1):
            mu_dyn = float(H @ self.x[t]) if self.dim > 0 else 0.0
            mu[t - 1] = self._deterministic_mu_t(t - 1) + mu_dyn
        return mu

    def _gaussian_loglike(self, mu_vec: np.ndarray) -> float:
        e = self.y - mu_vec
        s2 = self.sigma2
        return float(-0.5 * self.T * (math.log(2.0 * math.pi * s2)) - 0.5 * np.sum(e * e) / s2)

    # ------------------------- FFBS: sample x_{0:T} ------------------------- #
    def _ffbs(self) -> np.ndarray:
        if self.dim == 0:
            return self.x.copy()

        H = self._H_t()
        G = self._G_t()
        Q = self._Q_mat()
        R = float(self.sigma2)

        # Forward filter
        a = np.zeros((self.T + 1, self.dim))
        Rm = np.zeros((self.T + 1, self.dim, self.dim))
        m = np.zeros((self.T + 1, self.dim))
        C = np.zeros((self.T + 1, self.dim, self.dim))
        m[0] = self.m0
        C[0] = self.C0

        for t in range(1, self.T + 1):
            u = self._u_t(m[t - 1])
            a[t] = G @ m[t - 1] + u
            Rm[t] = G @ C[t - 1] @ G.T + Q

            det_mu = self._deterministic_mu_t(t - 1)
            y_t = self.y[t - 1] - det_mu

            S = float(H @ Rm[t] @ H.T + R)
            K = (Rm[t] @ H.T) / S
            v = y_t - float(H @ a[t])

            m[t] = a[t] + (K.flatten() * v)
            C[t] = Rm[t] - K @ (H @ Rm[t])

        # Backward sampling (Carter–Kohn)
        x = np.zeros_like(self.x)
        x[self.T] = np.random.multivariate_normal(m[self.T], C[self.T])
        for t in range(self.T - 1, -1, -1):
            u = self._u_t(m[t])                       # a_{t+1} mean shift
            J = C[t] @ G.T @ np.linalg.inv(Rm[t + 1]) # smoother gain
            mean = m[t] + J @ (x[t + 1] - (G @ m[t] + u))
            cov  = C[t] - J @ Rm[t + 1] @ J.T         # conditional covariance
            cov = 0.5 * (cov + cov.T)
            eig = np.linalg.eigvalsh(cov)
            if eig.min() <= 0: cov += (1e-10 - eig.min()) * np.eye(cov.shape[0])
            x[t] = np.random.multivariate_normal(mean, cov)

        return x

    def _propagate_initial_path(self, Q_init: np.ndarray) -> None:
        if self.dim == 0: return
        G = self._G_t()
        for t in range(1, self.T + 1):
            u = self._u_t(self.x[t - 1])
            self.x[t] = (G @ self.x[t - 1] + u +
                         np.random.normal(0.0, np.sqrt(Q_init), size=self.dim))

    # ----------------------- MH updates (logsigma, sds, det) ---------------- #
    def _mh_accept(self, logacc: float) -> bool:
        return (np.log(np.random.rand()) < min(0.0, logacc))

    def _get_step(self, key: str) -> float:
        c = self.cfg
        return {
            "logsigma": c.step_logsigma,
            "level": c.step_level,
            "slope": c.step_slope,
            "season": c.step_season,
            "log_s_alpha": c.step_log_s_alpha,
            "log_s_beta": c.step_log_s_beta,
            "log_s_gamma": c.step_log_s_gamma,
        }[key]

    def _set_step(self, key: str, val: float) -> None:
        v = float(np.clip(val, self.cfg.step_min, self.cfg.step_max))
        if   key == "logsigma":     self.cfg.step_logsigma = v
        elif key == "level":        self.cfg.step_level = v
        elif key == "slope":        self.cfg.step_slope = v
        elif key == "season":       self.cfg.step_season = v
        elif key == "log_s_alpha":  self.cfg.step_log_s_alpha = v
        elif key == "log_s_beta":   self.cfg.step_log_s_beta = v
        elif key == "log_s_gamma":  self.cfg.step_log_s_gamma = v
        else: raise KeyError(key)

    # ---- logsigma ~ N(m_sigma, s_sigma^2) with RW–MH on logsigma
    def update_logsigma(self) -> None:
        step = self._get_step("logsigma")
        cur = self.logsigma
        prop = cur + np.random.normal(0.0, step)

        sigma2_prop = float(np.exp(2.0 * prop))

        mu_vec = self._mu_vec_current()
        e = self.y - mu_vec
        ll_old = self._gaussian_loglike(mu_vec)                 # with current self.sigma2
        ll_new = float(-0.5 * self.T * (math.log(2.0 * math.pi * sigma2_prop))
                       - 0.5 * np.sum(e * e) / sigma2_prop)

        self.proposals["logsigma"] += 1
        lp_old = -0.5 * ((cur  - self.priors.m_sigma) ** 2) / (self.priors.s_sigma ** 2)
        lp_new = -0.5 * ((prop - self.priors.m_sigma) ** 2) / (self.priors.s_sigma ** 2)

        logacc = (ll_new + lp_new) - (ll_old + lp_old)
        if self._mh_accept(logacc):
            self.logsigma = prop
            self.sigma2 = sigma2_prop
            self.accept["logsigma"] += 1

    # ---- PC prior MH for a single sd s = exp(z), given lambda
    def _update_log_sd_single(self, key: str, get_s: Callable[[], float],
                              set_s: Callable[[float], None], lambda_s: Optional[float]) -> None:
        if lambda_s is None:  # if the component doesn't exist or lambda isn't defined, skip
            return
        step = self._get_step(key)
        s_cur = float(get_s())
        z_cur = math.log(max(1e-18, s_cur))
        z_prop = z_cur + np.random.normal(0.0, step)
        s_prop = math.exp(z_prop)

        # Transition residual sums for each coordinate
        if key == "log_s_alpha" and self.idx_alpha is not None:
            ss = 0.0
            for t in range(1, self.T + 1):
                drift = 0.0
                if self.idx_beta is not None:
                    drift = self.x[t - 1, self.idx_beta]
                elif self.trend_mode == "deterministic":
                    drift = self.slope_value
                mean = self.x[t - 1, self.idx_alpha] + drift
                ss += (self.x[t, self.idx_alpha] - mean) ** 2
            T_eff = self.T
        elif key == "log_s_beta" and self.idx_beta is not None:
            diffs = self.x[1:, self.idx_beta] - self.x[:-1, self.idx_beta]
            ss = float(np.sum(diffs * diffs)); T_eff = self.T
        elif key == "log_s_gamma" and self.seasonal_mode == "dynamic":
            gs, ge = self.idx_g_start, self.idx_g_end
            ss = 0.0
            for t in range(1, self.T + 1):
                prev = self.x[t - 1, gs:ge + 1]
                mean_new = -float(np.sum(prev))
                ss += (self.x[t, ge] - mean_new) ** 2
            T_eff = self.T
        else:
            return

        # Gaussian innovations with sd s: log-lik ∝ -T log s - ss/(2 s^2)
        ll_cur  = -(T_eff * z_cur)  - (ss * math.exp(-2.0 * z_cur)  / 2.0)
        ll_prop = -(T_eff * z_prop) - (ss * math.exp(-2.0 * z_prop) / 2.0)

        # PC prior term on s with Jacobian, given lambda
        lp_cur  = -lambda_s * math.exp(z_cur)  + z_cur
        lp_prop = -lambda_s * math.exp(z_prop) + z_prop

        self.proposals[key] += 1
        logacc = (ll_prop + lp_prop) - (ll_cur + lp_cur)
        if self._mh_accept(logacc):
            set_s(s_prop)
            self.accept[key] += 1

    def update_process_sds(self) -> None:
        # Use current lambdas (fixed or the last Gibbs draw)
        if self.idx_alpha is not None and (self.lambda_alpha is not None):
            self._update_log_sd_single("log_s_alpha",
                                       lambda: self.s_alpha,
                                       lambda v: setattr(self, "s_alpha", v),
                                       self.lambda_alpha)
        if self.idx_beta is not None and (self.lambda_beta is not None):
            self._update_log_sd_single("log_s_beta",
                                       lambda: self.s_beta,
                                       lambda v: setattr(self, "s_beta", v),
                                       self.lambda_beta)
        if self.seasonal_mode == "dynamic" and (self.lambda_gamma is not None):
            self._update_log_sd_single("log_s_gamma",
                                       lambda: self.s_gamma,
                                       lambda v: setattr(self, "s_gamma", v),
                                       self.lambda_gamma)

    # ---- Gibbs updates for lambda given s (Gamma hyperprior)
    #     lambda | s  ~ Gamma(a_lambda + 1, b_lambda + s)  (shape–rate)
    def _gibbs_lambda_single(self, comp: str) -> None:
        if comp == "alpha":
            pc = self.priors.pc_alpha
            if pc.lambda_s is not None or self.idx_alpha is None:
                return
            a_post = pc.a_lambda + 1.0
            b_post = pc.b_lambda + max(0.0, float(self.s_alpha))
            self.lambda_alpha = float(np.random.gamma(shape=a_post, scale=1.0 / b_post))
        elif comp == "beta":
            pc = self.priors.pc_beta
            if pc.lambda_s is not None or self.idx_beta is None:
                return
            a_post = pc.a_lambda + 1.0
            b_post = pc.b_lambda + max(0.0, float(self.s_beta))
            self.lambda_beta = float(np.random.gamma(shape=a_post, scale=1.0 / b_post))
        elif comp == "gamma":
            pc = self.priors.pc_gamma
            if pc.lambda_s is not None or self.seasonal_mode != "dynamic":
                return
            a_post = pc.a_lambda + 1.0
            b_post = pc.b_lambda + max(0.0, float(self.s_gamma))
            self.lambda_gamma = float(np.random.gamma(shape=a_post, scale=1.0 / b_post))

    def update_pc_lambdas(self) -> None:
        self._gibbs_lambda_single("alpha")
        self._gibbs_lambda_single("beta")
        self._gibbs_lambda_single("gamma")

    # ---- Deterministic structural parameter MH updates (Normal priors) ---- #
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

        ll_old = self._gaussian_loglike(mu_vec_old)
        ll_new = self._gaussian_loglike(mu_vec_prop)

        lp_old = -0.5 * ((cur - self.priors.m_level) ** 2) / (self.priors.s_level ** 2)
        lp_new = -0.5 * ((prop - self.priors.m_level) ** 2) / (self.priors.s_level ** 2)

        self.proposals["level"] += 1
        logacc = (ll_new + lp_new) - (ll_old + lp_old)
        if self._mh_accept(logacc):
            self.level_value = prop
            self.accept["level"] += 1

    def update_slope_value(self) -> None:
        if self.trend_mode != "deterministic": return
        step = self._get_step("slope")
        cur = self.slope_value
        prop = cur + np.random.normal(0.0, step)

        old = self.slope_value
        self.slope_value = prop
        mu_vec_prop = self._mu_vec_current()
        self.slope_value = old
        mu_vec_old = self._mu_vec_current()

        ll_old = self._gaussian_loglike(mu_vec_old)
        ll_new = self._gaussian_loglike(mu_vec_prop)

        lp_old = -0.5 * ((cur - self.priors.m_slope) ** 2) / (self.priors.s_slope ** 2)
        lp_new = -0.5 * ((prop - self.priors.m_slope) ** 2) / (self.priors.s_slope ** 2)

        self.proposals["slope"] += 1
        logacc = (ll_new + lp_new) - (ll_old + lp_old)
        if self._mh_accept(logacc):
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

        ll_old = self._gaussian_loglike(mu_vec_old)
        ll_new = self._gaussian_loglike(mu_vec_prop)

        m_first = self.priors.m_season
        if m_first is None:
            m_first = np.zeros(self.period - 1, float)
        else:
            m_first = np.asarray(m_first, float)
            if m_first.size != self.period - 1:
                raise ValueError("priors.m_season must have length = period-1.")
        s = float(self.priors.s_season)

        lp_old = -0.5 * np.sum(((v_cur[:-1] - m_first) / s) ** 2)
        lp_new = -0.5 * np.sum(((prop[:-1] - m_first) / s) ** 2)

        self.proposals["season"] += 1
        logacc = (ll_new + lp_new) - (ll_old + lp_old)
        if self._mh_accept(logacc):
            self.season_vec = prop
            self.accept["season"] += 1

    # ------------------- Adaptive step-size (like DGEV) ------------------- #
    def _fmt_acc(self, key: str) -> str:
        a, p = self.accept[key], self.proposals[key]
        pct = (100.0 * a / p) if p > 0 else 0.0
        return f"{a}/{p} ({pct:4.1f}%)"

    def _q_snapshot(self) -> str:
        rows = []
        def add(label, val):
            if val is None: return
            q = float(val ** 2)
            logq = np.log10(max(q, 1e-20))
            rows.append(f"{label}: log10(Q)={logq:6.2f}")
        if self.idx_alpha is not None: add("Q_alpha", self.s_alpha)
        if self.idx_beta  is not None: add("Q_beta",  self.s_beta)
        if self.seasonal_mode == "dynamic": add("Q_gamma(last)", self.s_gamma)
        lam_info = []
        if self.lambda_alpha is not None: lam_info.append(f"λ_α={self.lambda_alpha:.3g}")
        if self.lambda_beta  is not None: lam_info.append(f"λ_β={self.lambda_beta:.3g}")
        if self.lambda_gamma is not None: lam_info.append(f"λ_γ={self.lambda_gamma:.3g}")
        out = (" | " + " | ".join(rows)) if rows else ""
        if lam_info: out += " | " + " ".join(lam_info)
        return out

    def _adapt_steps(self, it: int) -> None:
        cfg = self.cfg
        if not cfg.adapt_steps:
            return
        in_window = (cfg.adapt_until == "all") or (it < cfg.burn)
        if (it + 1) % max(1, cfg.adapt_every) != 0 or (not in_window):
            return
        k = self._adapt_round
        eta = cfg.adapt_eta0 / ((1.0 + k) ** cfg.adapt_eta_decay)
        keys: List[str] = ["logsigma"]
        if self.level_mode == "deterministic": keys.append("level")
        if self.trend_mode == "deterministic": keys.append("slope")
        if self.seasonal_mode == "deterministic": keys.append("season")
        if self.idx_alpha is not None: keys.append("log_s_alpha")
        if self.idx_beta  is not None: keys.append("log_s_beta")
        if self.seasonal_mode == "dynamic": keys.append("log_s_gamma")
        target = cfg.adapt_target_1d
        changed = []
        for key in keys:
            acc_now = self.accept[key]; prop_now = self.proposals[key]
            acc_win = acc_now - self._mh_prev_acc[key]
            prop_win = prop_now - self._mh_prev_prop[key]
            if prop_win <= 0: continue
            rate = acc_win / max(1, prop_win)
            s = self._get_step(key)
            s_new = s * np.exp(eta * (rate - target))
            self._set_step(key, s_new)
            changed.append((key, s, self._get_step(key), rate))
            self._mh_prev_acc[key] = acc_now
            self._mh_prev_prop[key] = prop_now
        if changed and self.cfg.progress:
            msg = " | ".join([f"{k}: {old:.4g}→{new:.4g} (acc_win={r:.2f})"
                              for (k, old, new, r) in changed])
            print(f"  [adapt] η={eta:.4f} target={target:.2f} :: {msg}")
        self._adapt_round += 1

    # --------------------------------- MCMC --------------------------------- #
    def run(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg

        save_iters = list(range(cfg.burn, cfg.n_iter, cfg.thin))
        n_kept = max(0, len(save_iters))

        keep_idx = 0
        self.keep = {
            "sigma": np.zeros(n_kept, float),
            "mu":    np.zeros((n_kept, self.T), float),
        }
        if self.dim > 0:
            self.keep["Q"]  = np.zeros((n_kept, self.dim), float)
            self.keep["sd"] = np.zeros((n_kept, self.dim), float)
            self.keep["x"]  = np.zeros((n_kept, self.T, self.dim), float)
        # store lambdas (if present for the component)
        if self.idx_alpha is not None: self.keep["lambda_alpha"] = np.zeros(n_kept, float)
        if self.idx_beta  is not None: self.keep["lambda_beta"]  = np.zeros(n_kept, float)
        if self.seasonal_mode == "dynamic": self.keep["lambda_gamma"] = np.zeros(n_kept, float)

        if self.level_mode == "deterministic":
            self.keep["level_value"] = np.zeros(n_kept, float)
        if self.trend_mode == "deterministic":
            self.keep["slope_value"] = np.zeros(n_kept, float)
        if self.seasonal_mode == "deterministic":
            self.keep["season_vector"] = np.zeros((n_kept, self.period), float)

        print_every = cfg.progress_every if cfg.progress_every > 0 else max(1, cfg.n_iter // 50)

        for it in range(cfg.n_iter):
            # 1) Sample dynamic states via FFBS
            if self.dim > 0:
                self.x = self._ffbs()

            # 2) Update process sds (PC priors, RW–MH) given current lambdas
            if self.dim > 0:
                self.update_process_sds()

            # 3) Update PC lambdas (Gibbs) if not fixed
            self.update_pc_lambdas()

            # 4) Update deterministic structural params (RW–MH with Normal priors)
            if self.level_mode == "deterministic":  self.update_level_value()
            if self.trend_mode == "deterministic":  self.update_slope_value()
            if self.seasonal_mode == "deterministic": self.update_season_vec()

            # 5) Update observation variance via logsigma MH (Normal prior)
            self.update_logsigma()

            # --- adapt proposal scales if enabled ---
            self._adapt_steps(it)

            # Progress line
            if cfg.progress and ((it + 1) % print_every == 0 or it == cfg.n_iter - 1):
                det_info = ""
                if self.level_mode == "deterministic":
                    det_info += f" | acc(level)={self._fmt_acc('level')} value={self.level_value:.4f}"
                if self.trend_mode == "deterministic":
                    det_info += f" | acc(slope)={self._fmt_acc('slope')} value={self.slope_value:.4f}"
                if self.seasonal_mode == "deterministic":
                    prev = np.array2string(self.season_vec[:min(3, self.period)], precision=3, separator=",")
                    det_info += f" | acc(season)={self._fmt_acc('season')} γ[:3]={prev} ..."
                acc_sa = self._fmt_acc('log_s_alpha') if self.idx_alpha is not None else "n/a"
                acc_sb = self._fmt_acc('log_s_beta')  if self.idx_beta  is not None else "n/a"
                acc_sg = self._fmt_acc('log_s_gamma') if self.seasonal_mode == "dynamic" else "n/a"
                pc_info = f" | acc(log s_α/β/γ)={acc_sa}/{acc_sb}/{acc_sg}"
                q_info = self._q_snapshot()
                print(f"[it {it+1}/{cfg.n_iter}] σ={math.exp(self.logsigma):.3f} {pc_info}{det_info}{q_info}")

            # Save
            if it in save_iters and keep_idx < n_kept:
                mu_vec = self._mu_vec_current()
                self.keep["mu"][keep_idx, :] = mu_vec
                self.keep["sigma"][keep_idx] = float(np.exp(self.logsigma))
                if self.dim > 0:
                    Q_now = self._Q_mat()
                    self.keep["Q"][keep_idx, :]  = np.diag(Q_now)
                    self.keep["sd"][keep_idx, :] = np.sqrt(np.maximum(0.0, np.diag(Q_now)))
                    self.keep["x"][keep_idx, :, :] = self.x[1:self.T + 1, :]
                if "lambda_alpha" in self.keep: self.keep["lambda_alpha"][keep_idx] = float(self.lambda_alpha or 0.0)
                if "lambda_beta"  in self.keep: self.keep["lambda_beta"][keep_idx]  = float(self.lambda_beta  or 0.0)
                if "lambda_gamma" in self.keep: self.keep["lambda_gamma"][keep_idx] = float(self.lambda_gamma or 0.0)
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
                f"acc(level)={self._fmt_acc('level')} "
                f"acc(slope)={self._fmt_acc('slope')} "
                f"acc(season)={self._fmt_acc('season')} "
                f"acc(log s_α)={self._fmt_acc('log_s_alpha')} "
                f"acc(log s_β)={self._fmt_acc('log_s_beta')} "
                f"acc(log s_γ)={self._fmt_acc('log_s_gamma')}"
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
            "idx_g_start": self.idx_g_start,
            "idx_g_end": self.idx_g_end,
            "cfg": asdict(self.cfg),
            "priors": asdict(self.priors),
            "pc_lambdas_init": {
                "alpha_init": self.lambda_alpha,
                "beta_init":  self.lambda_beta,
                "gamma_init": self.lambda_gamma,
            },
            "true_sigma": self.true_sigma,
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


# ------------------------- CLI / Example run ------------------------------- #
if __name__ == "__main__":
    import sys, argparse
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    # Reuse your simulator for ground truth
    from simulator.mean_time_series import Mean_Time_Series  # adapt path if needed

    parser = argparse.ArgumentParser(description="Gaussian DLM (FFBS) with RW–MH and PC priors + Gamma hyperpriors on lambda (DLMGibbs)")
    # Modes
    parser.add_argument("--level-mode", choices=["dynamic", "deterministic"], default="dynamic")
    parser.add_argument("--trend-mode", choices=["dynamic", "deterministic", "none"], default="none")
    parser.add_argument("--season-mode", choices=["dynamic", "deterministic", "none"], default="none")
    # Basics
    parser.add_argument("--period", type=int, default=12)
    parser.add_argument("--T", type=int, default=200)
    # Initial values / truth for simulator
    parser.add_argument("--level-init", type=float, default=5.0)
    parser.add_argument("--slope-init", type=float, default=0.02)

    parser.add_argument("--true-sigma", type=float, default=2.0)
    parser.add_argument("--q-alpha", type=float, default=0.05)
    parser.add_argument("--q-beta",  type=float, default=0.02)
    parser.add_argument("--q-gamma", type=float, default=0.15)

    # Priors: Normal on logsigma & deterministic params
    parser.add_argument("--prior-m-sigma", type=float, default=1.0)
    parser.add_argument("--prior-s-sigma", type=float, default=1.0)
    parser.add_argument("--prior-m-level", type=float, default=0.0)
    parser.add_argument("--prior-s-level", type=float, default=10.0)
    parser.add_argument("--prior-m-slope", type=float, default=0.0)
    parser.add_argument("--prior-s-slope", type=float, default=10.0)
    parser.add_argument("--prior-m-season", type=str, default=None,
                        help="Comma-separated first (p-1) means for deterministic seasonal prior (e.g. '0,0,0').")
    parser.add_argument("--prior-s-season", type=float, default=5.0)

    # PC prior calibration (data-dependent defaults; you can override)
    parser.add_argument("--pc-frac-alpha", type=float, default=0.10)
    parser.add_argument("--pc-frac-beta",  type=float, default=0.10)
    parser.add_argument("--pc-frac-gamma", type=float, default=0.10)
    parser.add_argument("--pc-alpha-prob", type=float, default=0.5,
                        help="Tail prob α in P(s>u)=α for all coords.")

    # Optional fixed lambdas (if set, hyperprior sampling disabled for that coord)
    parser.add_argument("--pc-lambda-alpha", type=float, default=None)
    parser.add_argument("--pc-lambda-beta",  type=float, default=None)
    parser.add_argument("--pc-lambda-gamma", type=float, default=None)

    # Hyperpriors on lambda (shape–rate). Used only if the corresponding fixed lambda is None.
    parser.add_argument("--pc-a-lambda-alpha", type=float, default=1.0)
    parser.add_argument("--pc-b-lambda-alpha", type=float, default=1.0)
    parser.add_argument("--pc-a-lambda-beta",  type=float, default=1.0)
    parser.add_argument("--pc-b-lambda-beta",  type=float, default=1.0)
    parser.add_argument("--pc-a-lambda-gamma", type=float, default=1.0)
    parser.add_argument("--pc-b-lambda-gamma", type=float, default=1.0)

    # Sampler config
    parser.add_argument("--n-iter", type=int, default=2000)
    parser.add_argument("--burn", type=int, default=200)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--progress", default=True)
    parser.add_argument("--progress-every", type=int, default=10)

    # RW–MH steps
    parser.add_argument("--step-logsigma", type=float, default=0.2)
    parser.add_argument("--step-level", type=float, default=0.02)
    parser.add_argument("--step-slope", type=float, default=0.01)
    parser.add_argument("--step-season", type=float, default=0.02)
    parser.add_argument("--step-log-s-alpha", type=float, default=0.25)
    parser.add_argument("--step-log-s-beta",  type=float, default=0.10)
    parser.add_argument("--step-log-s-gamma", type=float, default=0.10)

    # Adaptive RW–MH CLI
    parser.add_argument("--adapt-steps", default=True)
    parser.add_argument("--adapt-every", type=int, default=25)
    parser.add_argument("--adapt-until", choices=["burn","all"], default="all")
    parser.add_argument("--adapt-eta0", type=float, default=0.1)
    parser.add_argument("--adapt-decay", type=float, default=0.75)
    parser.add_argument("--adapt-target-1d", type=float, default=0.44)
    parser.add_argument("--step-min", type=float, default=1e-5)
    parser.add_argument("--step-max", type=float, default=1.0)

    # Output
    parser.add_argument("--out-dir", type=str, default=None)

    args = parser.parse_args()
    np.random.seed(args.seed)

    # Build seasonal priors for simulator (length p-1)
    p = int(args.period)
    m0_season_dyn = [0.0] * (p - 1)
    v0_season_dyn = [0.5] * (p - 1)
    m0_season_det = [0.0] * (p - 1)  # can be overwritten via --prior-m-season
    v0_season_det = [0.0] * (p - 1)
    m0_season_none = [0.0] * (p - 1)
    v0_season_none = [1.0] * (p - 1)

    sim_level_mode  = args.level_mode
    sim_trend_mode  = args.trend_mode
    sim_season_mode = args.season_mode

    if sim_season_mode == "deterministic":
        m0_season = m0_season_det
        v0_season = v0_season_det
        q_season  = args.q_gamma  # ignored
    elif sim_season_mode == "dynamic":
        m0_season = m0_season_dyn
        v0_season = v0_season_dyn
        q_season  = args.q_gamma
    else:
        m0_season = m0_season_none
        v0_season = v0_season_none
        q_season  = args.q_gamma  # ignored

    # Simulate data
    mts = Mean_Time_Series(
        sigma=args.true_sigma,
        level_mode=sim_level_mode,
        trend_mode=sim_trend_mode,
        seasonal_mode=sim_season_mode,
        period=p,
        q_level=args.q_alpha,
        q_trend=args.q_beta,
        q_season=q_season,
        m0_level=args.level_init, v0_level=0.25,
        m0_trend=(args.slope_init if sim_trend_mode != "none" else 0.0), v0_trend=0.05,
        m0_season=m0_season, v0_season=v0_season,
        start_date=datetime(2000, 1, 1),
    )

    y = []
    for _ in range(args.T):
        mts.move(); y.append(mts.measure())
    y = np.asarray(y, float)

    truths = mts.get_truth_paths(as_numpy=True)
    mu_T    = truths["mu"][1:1 + args.T]
    alpha_T = truths["alpha"][1:1 + args.T] if sim_level_mode == "dynamic" else None
    beta_T  = truths["beta"][1:1 + args.T]  if sim_trend_mode == "dynamic" else None
    gamma_T = truths["gamma_last"][1:1 + args.T] if sim_season_mode == "dynamic" else None

    # Priors
    m_season_prior = None
    if args.prior_m_season is not None:
        toks = [t.strip() for t in args.prior_m_season.split(",") if t.strip() != ""]
        m_season_prior = np.array([float(z) for z in toks], float)
        if m_season_prior.size != p - 1:
            raise ValueError(f"--prior-m-season must have length {p-1} (got {m_season_prior.size}).")

    priors = Priors(
        m_sigma=float(args.prior_m_sigma), s_sigma=float(args.prior_s_sigma),
        m_level=float(args.prior_m_level), s_level=float(args.prior_s_level),
        m_slope=float(args.prior_m_slope), s_slope=float(args.prior_s_slope),
        m_season=m_season_prior, s_season=float(args.prior_s_season),
        pc_alpha=PCPrior(
            lambda_s=(None if args.pc_lambda_alpha is None else float(args.pc_lambda_alpha)),
            a_lambda=float(args.pc_a_lambda_alpha), b_lambda=float(args.pc_b_lambda_alpha),
            frac=float(args.pc_frac_alpha), alpha_prob=float(args.pc_alpha_prob)
        ),
        pc_beta=PCPrior(
            lambda_s=(None if args.pc_lambda_beta is None else float(args.pc_lambda_beta)),
            a_lambda=float(args.pc_a_lambda_beta), b_lambda=float(args.pc_b_lambda_beta),
            frac=float(args.pc_frac_beta), alpha_prob=float(args.pc_alpha_prob)
        ),
        pc_gamma=PCPrior(
            lambda_s=(None if args.pc_lambda_gamma is None else float(args.pc_lambda_gamma)),
            a_lambda=float(args.pc_a_lambda_gamma), b_lambda=float(args.pc_b_lambda_gamma),
            frac=float(args.pc_frac_gamma), alpha_prob=float(args.pc_alpha_prob)
        ),
    )

    cfg = SamplerConfig(
        n_iter=args.n_iter, burn=args.burn, thin=args.thin,
        random_seed=args.seed, progress=bool(args.progress),
        progress_every=int(args.progress_every),
        step_logsigma=float(args.step_logsigma),
        step_level=float(args.step_level),
        step_slope=float(args.step_slope),
        step_season=float(args.step_season),
        step_log_s_alpha=float(args.step_log_s_alpha),
        step_log_s_beta=float(args.step_log_s_beta),
        step_log_s_gamma=float(args.step_log_s_gamma),
        adapt_steps=bool(args.adapt_steps),
        adapt_every=int(args.adapt_every),
        adapt_until=str(args.adapt_until),
        adapt_eta0=float(args.adapt_eta0),
        adapt_eta_decay=float(args.adapt_decay),
        adapt_target_1d=float(args.adapt_target_1d),
        step_min=float(args.step_min),
        step_max=float(args.step_max),
    )

    # Initial deterministic seasonal vector if needed
    seasonal_init_pminus1 = (
        m_season_prior if (sim_season_mode == "deterministic" and m_season_prior is not None)
        else (np.zeros(p - 1, float) if sim_season_mode == "deterministic" else None)
    )

    sampler = DLMGibbs(
        y=y, period=p,
        level_mode=sim_level_mode, trend_mode=sim_trend_mode, seasonal_mode=sim_season_mode,
        m0_level=args.level_init, v0_level=0.25,
        m0_trend=(args.slope_init if sim_trend_mode != "none" else 0.0), v0_trend=0.05,
        m0_season=(m0_season if sim_season_mode == "dynamic" else None),
        v0_season=(v0_season if sim_season_mode == "dynamic" else None),
        level_value_init=args.level_init,
        slope_value_init=(args.slope_init if sim_trend_mode == "deterministic" else 0.0),
        seasonal_vector_init=seasonal_init_pminus1,
        sigma2_init=args.true_sigma**2,
        q_alpha_init=args.q_alpha, q_beta_init=args.q_beta, q_gamma_init=args.q_gamma,
        priors=priors, cfg=cfg,
    )

    true_Q = []
    if sim_level_mode == "dynamic": true_Q.append(args.q_alpha)
    if sim_trend_mode == "dynamic": true_Q.append(args.q_beta)
    if sim_season_mode == "dynamic": true_Q += [args.q_gamma] + [0.0] * (p - 2)
    sampler.set_truth(sigma=args.true_sigma, Q=(np.asarray(true_Q, float) if len(true_Q) else None))
    sampler.set_truth_paths(mu=mu_T, alpha=alpha_T, beta=beta_T, gamma=gamma_T)

    tag = f"{sim_level_mode}-{sim_trend_mode}-{sim_season_mode}"
    out_dir = args.out_dir or os.path.join("results", "simulations", "DLM",
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

    # ---- Summaries ----
    print(f"Posterior mean sigma: {np.mean(posterior['sigma']):.3f} (true {args.true_sigma})")
    if "Q" in posterior and sampler.dim > 0 and posterior["Q"].size > 0:
        if sampler.idx_alpha is not None:
            print(f"Posterior mean Q_alpha: {np.mean(posterior['Q'][:, sampler.idx_alpha]):.6g}")
        if sampler.idx_beta is not None:
            print(f"Posterior mean Q_beta:  {np.mean(posterior['Q'][:, sampler.idx_beta]):.6g}")
        if sampler.seasonal_mode == "dynamic":
            print(f"Posterior mean Q_gamma: {np.mean(posterior['Q'][:, sampler.idx_g_end]):.6g}")
    if "lambda_alpha" in posterior:
        print(f"Posterior mean λ_alpha: {np.mean(posterior['lambda_alpha']):.6g}")
    if "lambda_beta" in posterior:
        print(f"Posterior mean λ_beta:  {np.mean(posterior['lambda_beta']):.6g}")
    if "lambda_gamma" in posterior:
        print(f"Posterior mean λ_gamma: {np.mean(posterior['lambda_gamma']):.6g}")
