# %% dlm_gibbs_conjugate.py
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List


# =========================
# Priors & configuration
# =========================
@dataclass
class DLM_Priors:
    """
    Fully conjugate priors (Inverse-Gamma on variances; Gaussian on deterministic coefficients).
    IG(shape=a, scale=b): variance ~ IG(a,b) i.e. p(v) ∝ v^{-(a+1)} exp(-b/v).
    """
    # Observation variance R = sigma_y^2
    a_sigma_y: float = 2.5
    b_sigma_y: float = 1.0

    # Process variances (per dynamic block)
    a_Q_alpha: float = 2.5  # level innovation variance
    b_Q_alpha: float = 0.1
    a_Q_beta:  float = 2.5  # trend innovation variance
    b_Q_beta:  float = 0.1
    a_Q_gamma: float = 2.5  # newest seasonal coord variance
    b_Q_gamma: float = 0.1

    # Deterministic coefficients theta ~ N(m0, S0)
    # theta collects [intercept, slope, season_1..season_{p-1}] when active
    m_theta_scale: float = 0.0            # prior mean for all entries (isotropic)
    s_theta_scale: float = 10.0           # prior sd  (S0 = s^2 I)


@dataclass
class DLM_Config:
    n_iter: int = 4000
    burn: int = 1000
    thin: int = 2
    random_seed: Optional[int] = 123
    progress: bool = True


# ==================================
# Core DLM with conjugate Gibbs + FFBS
# ==================================
class DLM_Gibbs_Conjugate:
    """
    Linear-Gaussian DLM with switches:
      level_mode in {"dynamic", "deterministic"}       (no 'none' for level)
      trend_mode in {"dynamic", "deterministic", "none"}
      seasonal_mode in {"dynamic", "deterministic", "none"} with period >= 2

    Observation: y_t = mu_t + eps_t, eps_t ~ N(0, R)
      mu_t = (alpha+trend contribution) + seasonal contribution

    Dynamics (when active):
      alpha_{t+1} = alpha_t + beta_t + w_alpha             (if beta dynamic)
      alpha_{t+1} = alpha_t + w_alpha                      (if beta not dynamic)
      beta_{t+1}  = beta_t + w_beta
      g_{t+1,1..p-2} = g_{t,2..p-1};  g_{t+1,p-1} = -sum(g_{t,1..p-1}) + w_gamma

    Conjugate pieces:
      - theta | (y, x, R): Gaussian
      - R | (y, x, theta): Inverse-Gamma
      - Q_{•} | x: Inverse-Gamma based on innovation residual sums of squares
    """

    # ------------------------- #
    # Construction
    # ------------------------- #
    def __init__(self,
                 y: np.ndarray,
                 period: int,
                 level_mode: str = "dynamic",
                 trend_mode: str = "dynamic",
                 seasonal_mode: str = "dynamic",
                 priors: DLM_Priors = DLM_Priors(),
                 cfg: DLM_Config = DLM_Config()):
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

        # ----- build state layout -----
        self.tags: List[str] = []
        if self.level_mode == "dynamic":
            self.tags.append("alpha")
        if self.trend_mode == "dynamic":
            self.tags.append("beta")
        if self.seasonal_mode == "dynamic":
            self.tags.extend([f"g{k}" for k in range(1, self.period)])
        self.dim = len(self.tags)

        # state indices
        self.i_alpha = self.tags.index("alpha") if "alpha" in self.tags else None
        self.i_beta  = self.tags.index("beta")  if "beta"  in self.tags else None
        if self.seasonal_mode == "dynamic":
            self.i_g0 = self.tags.index("g1")
            self.i_gL = self.i_g0 + (self.period - 2)
        else:
            self.i_g0 = None
            self.i_gL = None

        # Deterministic coefficients theta: count & names
        p_theta = 0
        self.theta_names = []
        if self.level_mode == "deterministic":
            p_theta += 1; self.theta_names.append("intercept")
        if self.trend_mode == "deterministic":
            p_theta += 1; self.theta_names.append("slope")
        if self.seasonal_mode == "deterministic":
            p_theta += (self.period - 1)
            self.theta_names += [f"s{k}" for k in range(1, self.period)]
        self.p_theta = p_theta
        self.theta = np.zeros(p_theta, float)

        # initial state path (T+1; x[0] prior)
        self.x = np.zeros((self.T + 1, self.dim), float)

        # variances (start moderate)
        self.R = 1.0         # observation variance
        self.Qdiag = np.zeros(self.dim)  # process variances per active coord

        # storage
        kept = max(0, (cfg.n_iter - cfg.burn) // max(1, cfg.thin))
        self.keep: Dict[str, np.ndarray] = {
            "mu": np.zeros((kept, self.T), float),
            "sigma_y": np.zeros(kept, float),  # store std
        }
        if self.dim > 0:
            self.keep["x"] = np.zeros((kept, self.T, self.dim), float)
            self.keep["Q"] = np.zeros((kept, self.dim), float)
        if self.p_theta > 0:
            self.keep["theta"] = np.zeros((kept, self.p_theta), float)

    # ------------------------------------------------
    # System matrices: F, Q (diag), H (1×D)
    # ------------------------------------------------
    def _build_F_Q_H(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        D = self.dim
        F = np.eye(D)
        Q = np.diag(self.Qdiag.copy())
        H = np.zeros(D)

        # alpha block
        if self.i_alpha is not None:
            F[self.i_alpha, self.i_alpha] = 1.0
            if self.i_beta is not None:  # dynamic trend adds drift inside F
                F[self.i_alpha, self.i_beta] = 1.0

        # beta block
        if self.i_beta is not None:
            F[self.i_beta, self.i_beta] = 1.0

        # seasonal dynamic
        if self.seasonal_mode == "dynamic":
            # shift g1..g_{p-2} <- g2..g_{p-1}
            for k in range(self.i_g0, self.i_gL):
                F[k, k + 1] = 1.0
                F[k, k] = 0.0
            # newest coord = -sum(prev) + noise
            F[self.i_gL, self.i_g0:self.i_gL + 1] = -1.0
            H[self.i_gL] = 1.0

        # observation picks alpha if dynamic
        if self.i_alpha is not None:
            H[self.i_alpha] = 1.0

        return F, Q, H.reshape(1, -1)

    # -----------------------------------------
    # Kalman filter & RTS smoother (moments)
    # -----------------------------------------
    def _kalman_filter(self, F, Q, H, R, offset_mu: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], float]:
        """
        Returns filtered means m[t], covs C[t], and log-likelihood (optional).
        Diffuse-ish prior used for C[0].
        """
        T, D = self.T, self.dim
        m = np.zeros((T + 1, D), float)
        C = [np.eye(D) * 1e6 for _ in range(T + 1)]  # diffuse
        loglik = 0.0

        for t in range(1, T + 1):
            # predict
            a = F @ m[t - 1]
            Rpred = F @ C[t - 1] @ F.T + Q

            # forecast
            yhat = (H @ a.reshape(-1, 1)).item() + offset_mu[t - 1]
            S = (H @ Rpred @ H.T).item() + R
            v = self.y[t - 1] - yhat

            # update
            K = (Rpred @ H.T).reshape(-1) / S
            m[t] = a + K * v
            C[t] = Rpred - np.outer(K, K) * S

            loglik += -0.5 * (math.log(2 * math.pi * S) + (v * v) / S)

        return m, C, float(loglik)

    def _rts_smoother(self, F, Q, m, C) -> Tuple[np.ndarray, List[np.ndarray]]:
        T, D = self.T, self.dim
        ms = m.copy()
        Cs = [Ci.copy() for Ci in C]
        for t in range(T - 1, -1, -1):
            Rpred = F @ C[t] @ F.T + Q
            J = C[t] @ F.T @ np.linalg.pinv(Rpred)
            ms[t] = m[t] + J @ (ms[t + 1] - F @ m[t])
            Cs[t] = C[t] + J @ (Cs[t + 1] - Rpred) @ J.T
        return ms, Cs

    # --------------------
    # FFBS state sampling
    # --------------------
    def _ffbs(self, F, Q, H, R, offset_mu) -> np.ndarray:
        if self.dim == 0:
            return np.zeros_like(self.x)
        m, C, _ = self._kalman_filter(F, Q, H, R, offset_mu)
        ms, Cs = self._rts_smoother(F, Q, m, C)

        T, D = self.T, self.dim
        x = np.zeros((T + 1, D), float)
        # sample x_T
        x[T] = np.random.multivariate_normal(ms[T], Cs[T] + 1e-12 * np.eye(D))
        for t in range(T - 1, -1, -1):
            Rpred = F @ C[t] @ F.T + Q
            J = C[t] @ F.T @ np.linalg.pinv(Rpred)
            mean = ms[t] + J @ (x[t + 1] - F @ m[t])
            cov = Cs[t] - J @ Rpred @ J.T
            x[t] = np.random.multivariate_normal(mean, cov + 1e-12 * np.eye(D))
        return x

    # ------------------------------
    # Deterministic mean (theta) step
    # ------------------------------
    def _design_matrix(self) -> np.ndarray:
        """Build X for deterministic components (if any)."""
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

    def _offset_from_theta(self) -> np.ndarray:
        if self.p_theta == 0:
            return np.zeros(self.T)
        X = self._design_matrix()
        return X @ self.theta

    def _sample_theta(self, R_var: float, x_path: np.ndarray) -> None:
        """theta | y, x, R : Gaussian regression with prior N(m0, S0)."""
        if self.p_theta == 0:
            return
        X = self._design_matrix()
        # dynamic part Hx
        H_dyn = np.zeros(self.T)
        if self.dim > 0:
            for t in range(self.T):
                v = 0.0
                if self.i_alpha is not None:
                    v += x_path[t + 1, self.i_alpha]
                if self.seasonal_mode == "dynamic":
                    v += x_path[t + 1, self.i_gL]
                H_dyn[t] = v
        r = self.y - H_dyn

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
    def _update_R_IG(self, x_path: np.ndarray) -> float:
        """R | y, x, theta ~ IG(a*, b*)."""
        mu_dyn = np.zeros(self.T)
        if self.dim > 0:
            for t in range(self.T):
                val = 0.0
                if self.i_alpha is not None:
                    val += x_path[t + 1, self.i_alpha]
                if self.seasonal_mode == "dynamic":
                    val += x_path[t + 1, self.i_gL]
                mu_dyn[t] = val
        resid = self.y - (mu_dyn + self._offset_from_theta())
        rss = float(np.sum(resid * resid))
        a = self.priors.a_sigma_y + 0.5 * self.T
        b = self.priors.b_sigma_y + 0.5 * rss
        return float(1.0 / np.random.gamma(a, 1.0 / b))

    def _update_Q_IG(self, x_path: np.ndarray) -> np.ndarray:
        """Diagonal Q variances | x ~ IG(a*, b*) per dynamic coordinate."""
        Qdiag = np.zeros(self.dim)
        # alpha innovations
        if self.i_alpha is not None:
            inc = []
            for t in range(1, self.T + 1):
                drift = 0.0
                if self.i_beta is not None:
                    drift = x_path[t - 1, self.i_beta]
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

        for it in range(self.cfg.n_iter):
            # 1) States via FFBS given current (R, Q, theta)
            F, Q, H = self._build_F_Q_H()
            offset = self._offset_from_theta()
            self.x = self._ffbs(F, Q, H, self.R, offset)

            # 2) theta | y,x,R (Gaussian)
            self._sample_theta(R_var=self.R, x_path=self.x)

            # 3) R | y,x,theta (Inverse-Gamma)
            self.R = self._update_R_IG(self.x)

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
                # reconstruct mu_t path for this draw
                mu = np.zeros(self.T)
                if self.dim > 0:
                    for t in range(self.T):
                        val = 0.0
                        if self.i_alpha is not None:
                            val += self.x[t + 1, self.i_alpha]
                        if self.seasonal_mode == "dynamic":
                            val += self.x[t + 1, self.i_gL]
                        mu[t] = val
                mu += self._offset_from_theta()
                self.keep["mu"][keep_i, :] = mu
                keep_i += 1

            if self.cfg.progress and (it % max(1, self.cfg.n_iter // 10) == 0 or it == self.cfg.n_iter - 1):
                print(f"[{it+1}/{self.cfg.n_iter}] kept={keep_i}")

        return self.keep


# -------------------------
# Minimal example
# -------------------------
# ------------------------------------------------------------
# CLI / Example run using simulator.mean_time_series
# ------------------------------------------------------------
if __name__ == "__main__":
    import os, argparse, json, time
    import numpy as np
    import matplotlib.pyplot as plt
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from simulator.mean_time_series import Mean_Time_Series  # <- your simulator
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Conjugate Gaussian DLM (FFBS+Gibbs) demo using Mean_Time_Series data")

    # Modes (match Mean_Time_Series & sampler)
    parser.add_argument("--level-mode",  choices=["dynamic", "deterministic"], default="deterministic")
    parser.add_argument("--trend-mode",  choices=["dynamic", "deterministic", "none"], default="deterministic")
    parser.add_argument("--season-mode", choices=["dynamic", "deterministic", "none"], default="deterministic")

    # Data & simulation controls
    parser.add_argument("--period", type=int, default=12)
    parser.add_argument("--T", type=int, default=240)
    parser.add_argument("--seed", type=int, default=7)

    # Truths for simulator (obs sd and process variances)
    parser.add_argument("--true-sigma-y", type=float, default=1.5)  # observation std
    parser.add_argument("--q-alpha", type=float, default=0.01)      # level variance
    parser.add_argument("--q-beta",  type=float, default=0.005)     # trend variance
    parser.add_argument("--q-gamma", type=float, default=0.02)      # newest seasonal coord variance

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

    # Output options
    parser.add_argument("--out-dir", type=str, default="results_dlm")
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
        q_season  = args.q_gamma  # ignored by deterministic simulator, harmless
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
        start_date=datetime(2000, 1, 1),
    )

    y = []
    for _ in range(args.T):
        sim.move()
        y.append(sim.measure())
    y = np.asarray(y, float)

    truth = sim.get_truth_paths(as_numpy=True)
    mu_true    = truth["mu"][1:1 + args.T]
    alpha_true = truth["alpha"][1:1 + args.T] if args.level_mode == "dynamic" else None
    beta_true  = truth["beta"][1:1 + args.T]  if args.trend_mode == "dynamic" else None
    gamma_true = truth["gamma_last"][1:1 + args.T] if args.season_mode == "dynamic" else None
    dates      = truth["index"][:args.T]

    # --- Set up conjugate priors and config for the sampler ---
    pri = DLM_Priors(
        a_sigma_y=args.a_sigma_y, b_sigma_y=args.b_sigma_y,
        a_Q_alpha=args.a_Q_alpha, b_Q_alpha=args.b_Q_alpha,
        a_Q_beta=args.a_Q_beta,   b_Q_beta=args.b_Q_beta,
        a_Q_gamma=args.a_Q_gamma, b_Q_gamma=args.b_Q_gamma,
        m_theta_scale=0.0, s_theta_scale=10.0
    )
    cfg = DLM_Config(n_iter=args.n_iter, burn=args.burn, thin=args.thin,
                     random_seed=args.seed, progress=args.progress)

    # --- Fit with fully conjugate DLM Gibbs ---
    t0 = time.time()
    mdl = DLM_Gibbs_Conjugate(
        y=y, period=args.period,
        level_mode=args.level_mode, trend_mode=args.trend_mode, seasonal_mode=args.season_mode,
        priors=pri, cfg=cfg
    )
    posterior = mdl.run()
    elapsed = time.time() - t0
    print(f"Run time: {elapsed:.2f}s")

    # --- Quick summaries ---
    print(f"Posterior mean sigma_y: {np.mean(posterior['sigma_y']):.3f}  (true {args.true_sigma_y})")
    if "Q" in posterior:
        Qmean = np.mean(posterior["Q"], axis=0)
        print("Posterior mean Q diag:", Qmean)

    # --- Optional quick plot ---
    if not args.no_plots:
        mu_hat = np.mean(posterior["mu"], axis=0)
        lo = np.percentile(posterior["mu"], 5, axis=0)
        hi = np.percentile(posterior["mu"], 95, axis=0)

        plt.figure(figsize=(12,4))
        plt.plot(dates, y, label="y", linewidth=1.0)
        plt.plot(dates, mu_hat, "--", label="E[mu|y]", linewidth=1.0)
        plt.fill_between(dates, lo, hi, alpha=0.2, label="90% band")
        if mu_true is not None:
            plt.plot(dates, mu_true, ":",
                     label="mu (true)", linewidth=1.0)
        plt.title(f"DLM fit | level={args.level_mode}, trend={args.trend_mode}, season={args.season_mode}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # --- Save (optional) ---
    out_dir = args.out_dir
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, "posterior.npz"), **posterior)
        meta = {
            "T": int(args.T),
            "period": int(args.period),
            "modes": dict(level=args.level_mode, trend=args.trend_mode, season=args.season_mode),
            "seed": int(args.seed),
            "elapsed_seconds": float(elapsed),
        }
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[save] posterior -> {os.path.join(out_dir,'posterior.npz')}")
        print(f"[save] meta      -> {os.path.join(out_dir,'meta.json')}")

