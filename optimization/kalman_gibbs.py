# %% DLM_Gibbs.py
from __future__ import annotations

import os
import math
import time
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Tuple, List

import numpy as np


# =========================
# Priors & configuration
# =========================
@dataclass
class DLM_Priors:
    """
    Conjugate priors:
      - R (obs variance) ~ IG(a_sigma_y, b_sigma_y)
      - Q_• (process variances for *dynamic* coords only) ~ IG(a_Q_•, b_Q_•)
      - theta (static deterministic coefficients) ~ N(m_theta_scale, s_theta_scale^2 I)
    IG parameterization: variance ~ IG(a, b) with p(v) ∝ v^{-(a+1)} exp(-b/v).
    """
    # Observation variance R = sigma_y^2
    a_sigma_y: float = 2.5
    b_sigma_y: float = 1.0

    # Process variances (only for dynamic blocks)
    a_Q_alpha: float = 2.5  # level innovation variance
    b_Q_alpha: float = 0.1
    a_Q_beta:  float = 2.5  # trend innovation variance
    b_Q_beta:  float = 0.1
    a_Q_gamma: float = 2.5  # newest seasonal coord variance
    b_Q_gamma: float = 0.1

    # Deterministic coefficients theta ~ N(m, s^2 I)
    m_theta_scale: float = 0.0
    s_theta_scale: float = 10.0


@dataclass
class DLM_Config:
    n_iter: int = 4000
    burn: int = 1000
    thin: int = 2
    random_seed: Optional[int] = 123
    progress: bool = True
    trans_eps: float = 1e-10  # jitter for KF/RTS stability (Q=0 coords)


# ==================================
# Core DLM with conjugate Gibbs + FFBS
# ==================================
class DLM_Gibbs:
    """
    Structural Gaussian DLM with switches:

      level_mode    in {"dynamic", "deterministic"}          (no 'none' for level)
      trend_mode    in {"dynamic", "deterministic", "none"}
      seasonal_mode in {"dynamic", "deterministic", "none"}  (period >= 2)

    Observation:
      y_t = H_t x_t  +  X_t theta  +  eps_t,  eps_t ~ N(0, R)
      where:
        - x_t collects the *dynamic* blocks only (latent state),
        - theta collects all *deterministic* blocks (static parameters),
        - H_t and X_t depend on chosen modes.

    Components:
      μ_t = (level + trend contribution) + seasonal contribution

    State dynamics (only for dynamic blocks):
      alpha_{t+1} = alpha_t + (beta_t if beta dynamic else 0) + w_alpha
      beta_{t+1}  = beta_t + w_beta
      gamma_{1..p-2,t+1} = gamma_{2..p-1,t}
      gamma_{p-1,t+1}    = -sum(gamma_{1..p-1,t}) + w_gamma

    Deterministic components are handled only via theta (static):
      - intercept (level deterministic)
      - slope (trend deterministic)
      - seasonal (first p-1 dummies; last implied to enforce sum-to-zero)
    """

    # ------------------------- #
    # Construction
    # ------------------------- #
    def __init__(
        self,
        y: np.ndarray,
        period: int,
        level_mode: str = "dynamic",
        trend_mode: str = "dynamic",
        seasonal_mode: str = "dynamic",
        priors: DLM_Priors = DLM_Priors(),
        cfg: DLM_Config = DLM_Config(),
    ):
        self.y = np.asarray(y, float)
        self.T = int(self.y.size)
        self.period = int(period)
        assert self.period >= 2, "period must be >= 2"

        assert level_mode in {"dynamic", "deterministic"}
        assert trend_mode in {"dynamic", "deterministic", "none"}
        assert seasonal_mode in {"dynamic", "deterministic", "none"}

        self.level_mode = level_mode
        self.trend_mode = trend_mode
        self.seasonal_mode = seasonal_mode

        self.priors = priors
        self.cfg = cfg
        if cfg.random_seed is not None:
            np.random.seed(cfg.random_seed)

        # ----- latent state layout: ONLY dynamic blocks -----
        # tags: ['alpha' (if dynamic)] + ['beta' (if dynamic)] + ['gamma1'..'gamma_{p-1}' (if dynamic)]
        self.tags: List[str] = []
        if self.level_mode == "dynamic":
            self.tags.append("alpha")
        if self.trend_mode == "dynamic":
            self.tags.append("beta")
        if self.seasonal_mode == "dynamic":
            self.tags.extend([f"gamma{k}" for k in range(1, self.period)])
        self.dim = len(self.tags)

        # indices
        self.i_alpha = self.tags.index("alpha") if "alpha" in self.tags else None
        self.i_beta  = self.tags.index("beta")  if "beta"  in self.tags else None
        if self.seasonal_mode == "dynamic":
            self.i_g0 = self.tags.index("gamma1")
            self.i_gL = self.i_g0 + (self.period - 2)
        else:
            self.i_g0 = None
            self.i_gL = None

        # ----- static theta layout: ONLY deterministic blocks -----
        self.theta_names: List[str] = []
        if self.level_mode == "deterministic":
            self.theta_names.append("intercept")
        if self.trend_mode == "deterministic":
            self.theta_names.append("slope")
        if self.seasonal_mode == "deterministic":
            # first (p-1) seasonal indicators; last is implied by sum-to-zero
            self.theta_names += [f"season{k}" for k in range(1, self.period)]

        self.p_theta = len(self.theta_names)
        self.theta = np.zeros(self.p_theta, float)

        # initial state path (T+1; x[0] diffuse prior)
        self.x = np.zeros((self.T + 1, self.dim), float)

        # variances (R, diag(Q))
        self.R = 1.0
        self.Qdiag = np.zeros(self.dim)
        if self.i_alpha is not None:
            self.Qdiag[self.i_alpha] = 1e-3
        if self.i_beta is not None:
            self.Qdiag[self.i_beta] = 1e-3
        if self.i_gL is not None:
            self.Qdiag[self.i_gL] = 1e-3

        # storage
        kept = max(0, (cfg.n_iter - cfg.burn) // max(1, cfg.thin))
        self.keep: Dict[str, np.ndarray] = {
            "mu": np.zeros((kept, self.T), float),
            "sigma_y": np.zeros(kept, float),
        }
        if self.dim > 0:
            self.keep["x"] = np.zeros((kept, self.T, self.dim), float)
            self.keep["Q"] = np.zeros((kept, self.dim), float)
        if self.p_theta > 0:
            self.keep["theta"] = np.zeros((kept, self.p_theta), float)

    # ------------------------------------------------
    # System matrices for dynamic state: F, Q, H_t
    # ------------------------------------------------
    def _build_F_Q_Ht(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        F (D×D): transition for dynamic blocks
        Q (D×D): diag of process variances (zeros for non-innovating coords except eps for stability)
        Ht (T×D): time-varying obs matrix for dynamic blocks:
                  - alpha contributes with coefficient 1
                  - if alpha is deterministic and beta dynamic -> beta contributes as t
                  - seasonal dynamic contributes via newest coord (gamma_{p-1})
        """
        D = self.dim
        F = np.eye(D)
        Q = np.diag(self.Qdiag.copy())
        Ht = np.zeros((self.T, D))

        # alpha block (if dynamic)
        if self.i_alpha is not None:
            F[self.i_alpha, self.i_alpha] = 1.0
            if self.i_beta is not None:
                F[self.i_alpha, self.i_beta] = 1.0  # drift from beta
            Ht[:, self.i_alpha] = 1.0

        # beta block (if dynamic)
        if self.i_beta is not None:
            F[self.i_beta, self.i_beta] = 1.0
            # If level is deterministic but trend is dynamic, beta_t contributes as slope * t
            if self.i_alpha is None and self.level_mode == "deterministic":
                Ht[:, self.i_beta] = np.arange(self.T, dtype=float)

        # seasonal dynamic (if any)
        if self.seasonal_mode == "dynamic":
            # shift gamma1..gamma_{p-2} <- gamma2..gamma_{p-1}
            for k in range(self.i_g0, self.i_gL):
                F[k, k] = 0.0
                F[k, k + 1] = 1.0
            # newest coord rule: gamma_{p-1,t+1} = -sum(gamma_{1..p-1,t}) + w
            F[self.i_gL, self.i_g0:self.i_gL + 1] = -1.0
            # newest coord contributes to observation
            Ht[:, self.i_gL] = 1.0

        return F, Q, Ht

    # ------------------------------
    # Design matrix X for theta
    # ------------------------------
    def _design_matrix(self) -> np.ndarray:
        """
        Build X (T×p_theta) for deterministic components only.
        - intercept (level deterministic)
        - slope * t (trend deterministic)
        - seasonal dummies for 1..p-1 (last implied); time index uses 0..T-1
        """
        cols = []
        if self.level_mode == "deterministic":
            cols.append(np.ones(self.T))
        if self.trend_mode == "deterministic":
            cols.append(np.arange(self.T, dtype=float))
        if self.seasonal_mode == "deterministic":
            p = self.period
            tmod = np.arange(self.T) % p
            for k in range(p - 1):  # drop last to enforce sum-to-zero
                cols.append((tmod == k).astype(float))
        if len(cols) == 0:
            return np.zeros((self.T, 0))
        return np.column_stack(cols)

    # -----------------------------------------
    # Kalman filter & RTS smoother (moments)
    # -----------------------------------------
    def _kalman_filter(self, F, Q_eff, Ht, R, offset_mu: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], float]:
        T, D = self.T, self.dim
        m = np.zeros((T + 1, D), float)
        C = [np.eye(D) * 1e6 for _ in range(T + 1)]  # diffuse prior
        loglik = 0.0

        for t in range(1, T + 1):
            # predict
            a = F @ m[t - 1]
            Rpred = F @ C[t - 1] @ F.T + Q_eff
            # forecast
            yhat = float(Ht[t - 1].dot(a)) + offset_mu[t - 1]
            S = float(Ht[t - 1].reshape(1, -1) @ Rpred @ Ht[t - 1].reshape(-1, 1)) + R
            v = self.y[t - 1] - yhat
            # update
            K = (Rpred @ Ht[t - 1].reshape(-1, 1)).reshape(-1) / S
            m[t] = a + K * v
            C[t] = Rpred - np.outer(K, K) * S
            loglik += -0.5 * (math.log(2 * math.pi * S) + (v * v) / S)

        return m, C, float(loglik)

    def _rts_smoother(self, F, Q_eff, m, C) -> Tuple[np.ndarray, List[np.ndarray]]:
        T, D = self.T, self.dim
        ms = m.copy()
        Cs = [Ci.copy() for Ci in C]
        for t in range(T - 1, -1, -1):
            Rpred = F @ C[t] @ F.T + Q_eff
            J = C[t] @ F.T @ np.linalg.pinv(Rpred)
            ms[t] = m[t] + J @ (ms[t + 1] - F @ m[t])
            Cs[t] = C[t] + J @ (Cs[t + 1] - Rpred) @ J.T
        return ms, Cs

    # --------------------
    # FFBS state sampling
    # --------------------
    def _ffbs(self, F, Q, Ht, R, offset_mu) -> np.ndarray:
        if self.dim == 0:
            return np.zeros_like(self.x)

        # numerical stabilizer for deterministic rows
        Q_eff = Q.copy()
        eps = self.cfg.trans_eps
        for d in range(self.dim):
            if Q_eff[d, d] <= 0.0:
                Q_eff[d, d] = eps

        m, C, _ = self._kalman_filter(F, Q_eff, Ht, R, offset_mu)
        ms, Cs = self._rts_smoother(F, Q_eff, m, C)

        T, D = self.T, self.dim
        x = np.zeros((T + 1, D), float)
        # sample x_T
        x[T] = np.random.multivariate_normal(ms[T], Cs[T] + 1e-12 * np.eye(D))
        for t in range(T - 1, -1, -1):
            Rpred = F @ C[t] @ F.T + Q_eff
            J = C[t] @ F.T @ np.linalg.pinv(Rpred)
            mean = ms[t] + J @ (x[t + 1] - F @ m[t])
            cov = Cs[t] - J @ Rpred @ J.T
            x[t] = np.random.multivariate_normal(mean, cov + 1e-12 * np.eye(D))
        return x

    # ------------------------------
    # Theta (static deterministic) step
    # ------------------------------
    def _offset_from_theta(self) -> np.ndarray:
        if self.p_theta == 0:
            return np.zeros(self.T)
        X = self._design_matrix()
        return X @ self.theta

    def _sample_theta(self, R_var: float, x_path: np.ndarray, Ht: np.ndarray) -> None:
        """
        theta | y, x, R : Gaussian regression with prior N(m0, s^2 I).
        We regress (y - H_t x_t) on X.
        """
        if self.p_theta == 0:
            return
        X = self._design_matrix()

        # dynamic contribution H_t x_t to subtract
        Hx = np.zeros(self.T)
        if self.dim > 0:
            for t in range(self.T):
                Hx[t] = float(Ht[t].dot(x_path[t + 1]))
        r = self.y - Hx

        s2 = float(self.priors.s_theta_scale ** 2)
        S0_inv = (1.0 / s2) * np.eye(self.p_theta)
        m0 = np.full(self.p_theta, float(self.priors.m_theta_scale))

        XtX = (X.T @ X) / R_var
        XtR = (X.T @ r) / R_var
        Sn_inv = S0_inv + XtX
        Sn = np.linalg.pinv(Sn_inv)
        mn = Sn @ (S0_inv @ m0 + XtR)
        self.theta = np.random.multivariate_normal(mn, Sn)

    # ---------------------------------
    # Conjugate variance updates
    # ---------------------------------
    def _update_R_IG(self, x_path: np.ndarray, Ht: np.ndarray) -> float:
        """R | y, x, theta ~ IG(a*, b*)."""
        dyn = np.zeros(self.T)
        if self.dim > 0:
            for t in range(self.T):
                dyn[t] = float(Ht[t].dot(x_path[t + 1]))
        resid = self.y - (dyn + self._offset_from_theta())
        rss = float(np.sum(resid * resid))
        a = self.priors.a_sigma_y + 0.5 * self.T
        b = self.priors.b_sigma_y + 0.5 * rss
        return float(1.0 / np.random.gamma(a, 1.0 / b))

    def _update_Q_IG(self, x_path: np.ndarray) -> np.ndarray:
        """Q diag | x ~ IG(a*, b*) per *dynamic* coordinate only."""
        Qdiag = np.zeros(self.dim)

        # alpha innovations
        if self.i_alpha is not None:
            inc = []
            for t in range(1, self.T + 1):
                drift = x_path[t - 1, self.i_beta] if self.i_beta is not None else 0.0
                inc.append(x_path[t, self.i_alpha] - (x_path[t - 1, self.i_alpha] + drift))
            inc = np.asarray(inc)
            rss = float(np.sum(inc * inc))
            a = self.priors.a_Q_alpha + 0.5 * inc.size
            b = self.priors.b_Q_alpha + 0.5 * rss
            Qdiag[self.i_alpha] = 1.0 / np.random.gamma(a, 1.0 / b)

        # beta innovations
        if self.i_beta is not None:
            inc = x_path[1:, self.i_beta] - x_path[:-1, self.i_beta]
            rss = float(np.sum(inc * inc))
            a = self.priors.a_Q_beta + 0.5 * inc.size
            b = self.priors.b_Q_beta + 0.5 * rss
            Qdiag[self.i_beta] = 1.0 / np.random.gamma(a, 1.0 / b)

        # seasonal newest coord innovations
        if self.seasonal_mode == "dynamic":
            inc = []
            for t in range(1, self.T + 1):
                prev = x_path[t - 1, self.i_g0:self.i_gL + 1]
                mean_new = -float(np.sum(prev))
                inc.append(x_path[t, self.i_gL] - mean_new)
            inc = np.asarray(inc)
            rss = float(np.sum(inc * inc))
            a = self.priors.a_Q_gamma + 0.5 * inc.size
            b = self.priors.b_Q_gamma + 0.5 * rss
            Qdiag[self.i_gL] = 1.0 / np.random.gamma(a, 1.0 / b)

        return Qdiag

    # -------------------------
    # Main sampler
    # -------------------------
    def run(self) -> Dict[str, np.ndarray]:
        kept = max(0, (self.cfg.n_iter - self.cfg.burn) // max(1, self.cfg.thin))
        keep_i = 0
        last_progress = -1

        for it in range(self.cfg.n_iter):
            # 1) States via FFBS given current (R, Q, theta)
            F, Q, Ht = self._build_F_Q_Ht()
            offset = self._offset_from_theta()
            self.x = self._ffbs(F, Q, Ht, self.R, offset)

            # 2) theta | y,x,R (Gaussian)
            self._sample_theta(R_var=self.R, x_path=self.x, Ht=Ht)

            # 3) R | y,x,theta (Inverse-Gamma)
            self.R = self._update_R_IG(self.x, Ht)

            # 4) Q | x (Inverse-Gamma per dynamic block)
            if self.dim > 0:
                self.Qdiag = self._update_Q_IG(self.x)
            else:
                self.Qdiag = np.zeros(0)

            # 5) Save
            if it >= self.cfg.burn and ((it - self.cfg.burn) % self.cfg.thin == 0):
                if "x" in self.keep:
                    self.keep["x"][keep_i, :, :] = self.x[1:self.T + 1]
                    self.keep["Q"][keep_i, :] = self.Qdiag
                self.keep["sigma_y"][keep_i] = math.sqrt(self.R)
                if self.p_theta > 0:
                    self.keep["theta"][keep_i, :] = self.theta
                # reconstruct mu_t for this draw
                mu = np.zeros(self.T)
                if self.dim > 0:
                    for t in range(self.T):
                        mu[t] = float(Ht[t].dot(self.x[t + 1]))
                mu += self._offset_from_theta()
                self.keep["mu"][keep_i, :] = mu
                keep_i += 1

            if self.cfg.progress:
                pct = int(100 * (it + 1) / self.cfg.n_iter)
                if pct >= last_progress + 10 or it == self.cfg.n_iter - 1:
                    print(f"[{it+1}/{self.cfg.n_iter}] kept={keep_i}")
                    last_progress = pct

        return self.keep


# ------------------------------------------------------------
# CLI / Example run using simulator.mean_time_series
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from simulator.mean_time_series import Mean_Time_Series  # simulator

    parser = argparse.ArgumentParser(
        description="Gaussian DLM (FFBS+Gibbs): dynamic components latent, deterministic components as static parameters."
    )

    # Modes (match Mean_Time_Series & sampler)
    parser.add_argument("--level-mode",  choices=["dynamic", "deterministic"], default="dynamic")
    parser.add_argument("--trend-mode",  choices=["dynamic", "deterministic", "none"], default="deterministic")
    parser.add_argument("--season-mode", choices=["dynamic", "deterministic", "none"], default="deterministic")

    # Data & simulation controls
    parser.add_argument("--period", type=int, default=12)
    parser.add_argument("--T", type=int, default=240)
    parser.add_argument("--seed", type=int, default=7)

    # Truths for simulator (obs sd and process variances)
    parser.add_argument("--true-sigma-y", type=float, default=1.5)
    parser.add_argument("--q-alpha", type=float, default=0.01)
    parser.add_argument("--q-beta",  type=float, default=0.005)
    parser.add_argument("--q-gamma", type=float, default=0.02)

    # Simulator priors / fixed values
    parser.add_argument("--m0-level",  type=float, default=5.0)
    parser.add_argument("--v0-level",  type=float, default=0.25)
    parser.add_argument("--m0-trend",  type=float, default=0.01)
    parser.add_argument("--v0-trend",  type=float, default=0.05)

    # Conjugate priors (Inverse-Gamma on variances)
    parser.add_argument("--a-sigma-y", type=float, default=2.5)
    parser.add_argument("--b-sigma-y", type=float, default=1.0)
    parser.add_argument("--a-Q-alpha", type=float, default=2.5)
    parser.add_argument("--b-Q-alpha", type=float, default=0.1)
    parser.add_argument("--a-Q-beta",  type=float, default=2.5)
    parser.add_argument("--b-Q-beta",  type=float, default=0.1)
    parser.add_argument("--a-Q-gamma", type=float, default=2.5)
    parser.add_argument("--b-Q-gamma", type=float, default=0.1)

    # Sampler config
    parser.add_argument("--n-iter", type=int, default=2000)
    parser.add_argument("--burn",   type=int, default=500)
    parser.add_argument("--thin",   type=int, default=2)
    parser.add_argument("--progress", default=True)

    # Output & plotting
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # --- Build seasonal priors for simulator (length p-1 where needed) ---
    p = args.period
    seas_vec_full = [np.cos(2 * np.pi * k / p) + 0.25 * np.cos(4 * np.pi * k / p) for k in range(p)]
    m0_season_det = seas_vec_full[: p - 1]
    v0_season_det = [0.0] * (p - 1)
    m0_season_dyn = [0.0] * (p - 1)
    v0_season_dyn = [0.5] * (p - 1)
    m0_season_none = [0.0] * (p - 1)
    v0_season_none = [1.0] * (p - 1)

    if args.season_mode == "deterministic":
        m0_season = m0_season_det
        v0_season = v0_season_det
        q_season  = args.q_gamma  # ignored by deterministic simulator
    elif args.season_mode == "dynamic":
        m0_season = m0_season_dyn
        v0_season = v0_season_dyn
        q_season  = args.q_gamma
    else:  # none
        m0_season = m0_season_none
        v0_season = v0_season_none
        q_season  = args.q_gamma  # ignored by 'none'

    # Trend variance used only if trend is dynamic
    q_trend = args.q_beta if args.trend_mode == "dynamic" else 0.0
    q_level = args.q_alpha if args.level_mode == "dynamic" else 0.0

    # m0_trend=0 if trend="none" to respect simulator semantics
    m0_trend = (args.m0_trend if args.trend_mode != "none" else 0.0)

    # --- Simulate with Mean_Time_Series ---
    from datetime import datetime as _dt
    sim = Mean_Time_Series(
        sigma=args.true_sigma_y,
        level_mode=args.level_mode,
        trend_mode=args.trend_mode,
        seasonal_mode=args.season_mode,
        period=args.period,
        q_level=q_level,
        q_trend=q_trend,
        q_season=q_season,
        m0_level=args.m0_level, v0_level=args.v0_level,
        m0_trend=m0_trend,       v0_trend=args.v0_trend,
        m0_season=m0_season,     v0_season=v0_season,
        start_date=_dt(2000, 1, 1),
    )

    y = []
    for _ in range(args.T):
        sim.move()
        y.append(sim.measure())
    y = np.asarray(y, float)

    truth = sim.get_truth_paths(as_numpy=True)
    mu_true = truth["mu"][1:1 + args.T]
    dates   = truth["index"][:args.T]

    # --- Priors & config ---
    pri = DLM_Priors(
        a_sigma_y=args.a_sigma_y, b_sigma_y=args.b_sigma_y,
        a_Q_alpha=args.a_Q_alpha, b_Q_alpha=args.b_Q_alpha,
        a_Q_beta=args.a_Q_beta,   b_Q_beta=args.b_Q_beta,
        a_Q_gamma=args.a_Q_gamma, b_Q_gamma=args.b_Q_gamma,
        m_theta_scale=0.0, s_theta_scale=10.0
    )
    cfg = DLM_Config(
        n_iter=args.n_iter, burn=args.burn, thin=args.thin,
        random_seed=args.seed, progress=bool(args.progress)
    )

    # --- Fit with fully conjugate DLM Gibbs ---
    t0 = time.time()
    mdl = DLM_Gibbs(
        y=y, period=args.period,
        level_mode=args.level_mode, trend_mode=args.trend_mode, seasonal_mode=args.season_mode,
        priors=pri, cfg=cfg
    )
    posterior = mdl.run()
    elapsed = time.time() - t0
    print(f"Run time: {elapsed:.2f}s")

    # --- Summaries ---
    print(f"Posterior mean sigma_y: {np.mean(posterior['sigma_y']):.3f}  (true {args.true_sigma_y})")
    if "Q" in posterior:
        Qmean = np.mean(posterior["Q"], axis=0)
        print("Posterior mean Q diag:", Qmean)

    # --- Optional quick plot ---
    if not args.no_plots:
        mu_hat = np.mean(posterior["mu"], axis=0)
        lo = np.percentile(posterior["mu"], 5, axis=0)
        hi = np.percentile(posterior["mu"], 95, axis=0)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.plot(dates, y, label="y", linewidth=1.0)
        plt.plot(dates, mu_hat, "--", label="E[mu|y]", linewidth=1.0)
        plt.fill_between(dates, lo, hi, alpha=0.2, label="90% band")
        if mu_true is not None and len(mu_true) == args.T:
            plt.plot(dates, mu_true, ":", label="mu (true)", linewidth=1.0)
        plt.title(f"DLM fit | level={args.level_mode}, trend={args.trend_mode}, season={args.season_mode}")
        plt.legend()
        plt.tight_layout()
        quick_fig = plt.gcf()
    else:
        quick_fig = None

    # --- Save (unified layout) ---
    tag = f"{args.level_mode}-{args.trend_mode}-{args.season_mode}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = args.out_dir or "results"
    run_dir = os.path.join(base_dir, f"{tag}_{timestamp}")
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Save arrays
    np.savez_compressed(os.path.join(run_dir, "posterior.npz"), **posterior)

    # Meta with flags/indices (so plotters can infer)
    meta = {
        "T": int(args.T),
        "period": int(args.period),
        "modes": {
            "level_mode": args.level_mode,
            "trend_mode": args.trend_mode,
            "seasonal_mode": args.season_mode,
        },
        # What is in the latent state vs static theta
        "latent_state_components": [*mdl.tags],            # dynamic only
        "static_theta_components": mdl.theta_names,        # deterministic only
        "idx_alpha": mdl.i_alpha,
        "idx_beta": mdl.i_beta,
        "idx_gamma_last": (mdl.i_gL if args.season_mode == "dynamic" else None),
        "seed": int(args.seed),
        "elapsed_seconds": float(elapsed),
    }
    with open(os.path.join(run_dir, "posterior.meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Save quick figure if made
    if quick_fig is not None:
        png_path = os.path.join(fig_dir, "quick_overview.png")
        quick_fig.savefig(png_path, dpi=200, bbox_inches="tight")
        print(f"[save] Figure -> {png_path}")

    print(f"[save] Posterior -> {os.path.join(run_dir,'posterior.npz')}")
    print(f"[save] Metadata  -> {os.path.join(run_dir,'posterior.meta.json')}")
    print(f"[info] Figures will be saved to: {fig_dir}")
