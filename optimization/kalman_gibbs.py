# %% DLM_Gibbs.py
from __future__ import annotations

import os, math, time, json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Tuple, List

import numpy as np

# =========================
# Priors & configuration
# =========================
@dataclass
class DLM_Priors:
    """
    Conjugate priors.
    - Variances use Inverse-Gamma: v ~ IG(a, b) with pdf ∝ v^{-(a+1)} exp(-b/v).
    - Static parameters use Gaussian priors.
    """
    # Observation variance R = sigma_y^2
    a_sigma_y: float = 2.5
    b_sigma_y: float = 1.0

    # Process variances (per dynamic block)
    a_Q_alpha: float = 2.5
    b_Q_alpha: float = 0.1
    a_Q_beta:  float = 2.5
    b_Q_beta:  float = 0.1
    a_Q_gamma: float = 2.5
    b_Q_gamma: float = 0.1

    # Static parameters priors (Gaussian)
    # Used for: intercept, slope (either obs or transition), seasonal dummies
    m_theta: float = 0.0
    s_theta: float = 10.0


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
class DLM_Gibbs:
    """
    Linear-Gaussian DLM with switches:
      level_mode in {"dynamic", "deterministic"}       (no 'none' for level)
      trend_mode in {"dynamic", "deterministic", "none"}
      seasonal_mode in {"dynamic", "deterministic", "none"}

    Observation: y_t = mu_t + eps_t, eps_t ~ N(0, R)
      mu_t = (level contribution) + (seasonal contribution)

    Dynamics (when active):
      alpha_{t+1} = alpha_t + [beta_t if dynamic] + [slope if trend deterministic and level dynamic] + w_alpha
      beta_{t+1}  = beta_t + w_beta
      seasonal (p-1 dims): shift left; newest coord = -sum(prev p-1) + w_gamma

    Static parameters:
      - Observation static params (theta_obs): intercept (if level deterministic),
        seasonal dummies (if seasonal deterministic), slope (ONLY if level deterministic).
      - Transition static params: slope (ONLY if level dynamic & trend deterministic).

    Conjugate pieces:
      - theta_obs | (y, x, R): Gaussian regression
      - slope_transition | (x, Q_alpha): Gaussian (from level increments)
      - R | (y, x, theta_obs): Inverse-Gamma
      - Q_{•} | x: Inverse-Gamma from innovation sums of squares
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
        # data & modes
        self.y = np.asarray(y, float)
        self.T = int(self.y.size)
        self.period = int(period)
        assert self.period >= 2, "period must be >= 2"

        assert level_mode in {"dynamic", "deterministic"}
        assert trend_mode in {"dynamic", "deterministic", "none"}
        assert seasonal_mode in {"dynamic", "deterministic", "none"}

        # NOTE: "trend dynamic" without a dynamic level is not representable
        # with the current state layout (you'd need an accumulating level).
        if level_mode == "deterministic" and trend_mode == "dynamic":
            raise ValueError("trend_mode='dynamic' requires level_mode='dynamic'.")

        self.level_mode = level_mode
        self.trend_mode = trend_mode
        self.seasonal_mode = seasonal_mode

        self.priors = priors
        self.cfg = cfg
        if cfg.random_seed is not None:
            np.random.seed(cfg.random_seed)

        # ----- latent state layout: ONLY dynamic components -----
        tags: List[str] = []
        if self.level_mode == "dynamic":
            tags.append("alpha")
        if self.trend_mode == "dynamic":
            tags.append("beta")
        if self.seasonal_mode == "dynamic":
            tags.extend([f"gamma{k}" for k in range(1, self.period)])  # p-1 coords
        self.tags = tags
        self.dim = len(tags)

        # indices
        self.i_alpha = tags.index("alpha") if "alpha" in tags else None
        self.i_beta  = tags.index("beta")  if "beta"  in tags else None
        if self.seasonal_mode == "dynamic":
            self.i_g0 = tags.index("gamma1")
            self.i_gL = self.i_g0 + (self.period - 2)
        else:
            self.i_g0 = None
            self.i_gL = None

        # ----- static params split -----
        # Observation static parameters (theta_obs):
        #   - intercept if level deterministic
        #   - slope if (level deterministic AND trend deterministic)
        #   - seasonal dummies (first p-1) if seasonal deterministic
        self._slope_in_transition = (self.level_mode == "dynamic" and self.trend_mode == "deterministic")
        obs_cols: List[str] = []
        if self.level_mode == "deterministic":
            obs_cols.append("intercept")
            if self.trend_mode == "deterministic":
                obs_cols.append("slope")  # in observation ONLY when level is deterministic
        if self.seasonal_mode == "deterministic":
            obs_cols += [f"season{k}" for k in range(1, self.period)]  # last implied by sum-to-zero
        self.obs_param_names = obs_cols
        self.p_obs = len(obs_cols)
        self.theta_obs = np.zeros(self.p_obs, float)

        # Transition static parameter: slope (only if level dynamic & trend deterministic)
        self.slope_tr = 0.0  # acts in alpha transition; ignored otherwise

        # ----- state path (T+1; x[0] prior-diffuse) -----
        self.x = np.zeros((self.T + 1, self.dim), float)

        # variances
        self.R = 1.0
        self.Qdiag = np.zeros(self.dim)

        # storage (kept draws after burn/thin)
        kept = max(0, (cfg.n_iter - cfg.burn) // max(1, cfg.thin))
        self.keep: Dict[str, np.ndarray] = {
            "mu": np.zeros((kept, self.T), float),
            "sigma_y": np.zeros(kept, float),  # store std
        }
        if self.dim > 0:
            self.keep["x"] = np.zeros((kept, self.T, self.dim), float)
            self.keep["Q"] = np.zeros((kept, self.dim), float)
        if self.p_obs > 0:
            self.keep["theta_obs"] = np.zeros((kept, self.p_obs), float)
        if self._slope_in_transition:
            self.keep["slope_transition"] = np.zeros(kept, float)

    # ------------------------------------------------
    # Design matrices & controls
    # ------------------------------------------------
    def _design_matrix_obs(self) -> np.ndarray:
        """Build X_obs for *observation* static params only."""
        if self.p_obs == 0:
            return np.zeros((self.T, 0))
        cols = []
        for name in self.obs_param_names:
            if name == "intercept":
                cols.append(np.ones(self.T))
            elif name == "slope":
                cols.append(np.arange(self.T, dtype=float))  # only when level is deterministic
            elif name.startswith("season"):
                k = int(name.replace("season", ""))
                tmod = (np.arange(self.T) % self.period)
                cols.append((tmod == k).astype(float))  # last level implied by sum-to-zero
            else:
                raise RuntimeError(f"Unknown obs param column: {name}")
        return np.column_stack(cols)

    def _offset_from_obs_params(self) -> np.ndarray:
        if self.p_obs == 0:
            return np.zeros(self.T)
        X = self._design_matrix_obs()
        return X @ self.theta_obs

    def _control_seq(self) -> np.ndarray:
        """
        Control sequence b[t] added in the *transition*:
          x_t = F x_{t-1} + b_{t-1} + w_{t-1}
        Only used to feed deterministic slope into alpha when level is dynamic
        and trend is deterministic.
        Returns (T, D) array where row t is b_t for transition from t->t+1.
        """
        if self.dim == 0:
            return np.zeros((self.T, 0))
        b = np.zeros((self.T, self.dim), float)
        if self._slope_in_transition and self.i_alpha is not None:
            b[:, self.i_alpha] = self.slope_tr  # constant drift each step
        return b

    # ------------------------------------------------
    # System matrices: F, Q (diag), H (time-constant here)
    # ------------------------------------------------
    def _build_F_Q_H(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        D = self.dim
        F = np.eye(D)
        Q = np.diag(self.Qdiag.copy())
        H = np.zeros(D)

        # alpha
        if self.i_alpha is not None:
            F[self.i_alpha, self.i_alpha] = 1.0
            H[self.i_alpha] = 1.0  # alpha contributes directly to y_t
            if self.i_beta is not None:
                F[self.i_alpha, self.i_beta] = 1.0  # drift by dynamic beta

        # beta
        if self.i_beta is not None:
            F[self.i_beta, self.i_beta] = 1.0

        # seasonal dynamic (p-1 dims)
        if self.seasonal_mode == "dynamic":
            # shift gamma1..gamma_{p-2} <- gamma2..gamma_{p-1}
            for k in range(self.i_g0, self.i_gL):
                F[k, k + 1] = 1.0
                F[k, k] = 0.0
            # newest coord = -sum(prev p-1) + noise
            F[self.i_gL, self.i_g0:self.i_gL + 1] = -1.0
            H[self.i_gL] = 1.0  # last coord enters observation

        return F, Q, H.reshape(1, -1)

    # -----------------------------------------
    # Kalman filter & RTS smoother (with controls)
    # -----------------------------------------
    def _kalman_filter(self, F, Q, H, R, offset_mu: np.ndarray, b_seq: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Time-invariant H but with a per-step transition control b_seq[t-1].
        Returns filtered means m[t], covs C[t] for t=0..T.
        Diffuse prior for C[0].
        """
        T, D = self.T, self.dim
        m = np.zeros((T + 1, D), float)
        C = [np.eye(D) * 1e6 for _ in range(T + 1)]  # diffuse

        for t in range(1, T + 1):
            # predict
            a = F @ m[t - 1]
            if D > 0:
                a = a + b_seq[t - 1]  # control in transition
            Rpred = F @ C[t - 1] @ F.T + Q

            # forecast & update
            yhat = (H @ a.reshape(-1, 1)).item() + offset_mu[t - 1]
            S = (H @ Rpred @ H.T).item() + R
            v = self.y[t - 1] - yhat

            K = (Rpred @ H.T).reshape(-1) / S
            m[t] = a + K * v
            C[t] = Rpred - np.outer(K, K) * S

        return m, C

    def _rts_smoother(self, F, Q, m, C, b_seq: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        T, D = self.T, self.dim
        ms = m.copy()
        Cs = [Ci.copy() for Ci in C]
        for t in range(T - 1, -1, -1):
            Rpred = F @ C[t] @ F.T + Q
            J = C[t] @ F.T @ np.linalg.pinv(Rpred)
            ms[t] = m[t] + J @ (ms[t + 1] - (F @ m[t] + (b_seq[t] if D > 0 else 0.0)))
            Cs[t] = C[t] + J @ (Cs[t + 1] - Rpred) @ J.T
        return ms, Cs

    # --------------------
    # FFBS state sampling
    # --------------------
    def _ffbs(self, F, Q, H, R, offset_mu, b_seq) -> np.ndarray:
        if self.dim == 0:
            return np.zeros_like(self.x)
        m, C = self._kalman_filter(F, Q, H, R, offset_mu, b_seq)
        ms, Cs = self._rts_smoother(F, Q, m, C, b_seq)

        T, D = self.T, self.dim
        x = np.zeros((T + 1, D), float)
        # sample x_T
        x[T] = np.random.multivariate_normal(ms[T], Cs[T] + 1e-12 * np.eye(D))
        for t in range(T - 1, -1, -1):
            Rpred = F @ C[t] @ F.T + Q
            J = C[t] @ F.T @ np.linalg.pinv(Rpred)
            mean = ms[t] + J @ (x[t + 1] - (F @ m[t] + (b_seq[t] if D > 0 else 0.0)))
            cov = Cs[t] - J @ Rpred @ J.T
            x[t] = np.random.multivariate_normal(mean, cov + 1e-12 * np.eye(D))
        return x

    # ------------------------------
    # Static parameter updates
    # ------------------------------
    def _sample_theta_obs(self, R_var: float, x_path: np.ndarray, H: np.ndarray) -> None:
        """Gaussian regression for observation static params (theta_obs)."""
        if self.p_obs == 0:
            return
        X = self._design_matrix_obs()
        # subtract dynamic contribution H x_t
        Hx = np.zeros(self.T)
        if self.dim > 0:
            for t in range(self.T):
                Hx[t] = float(H @ x_path[t + 1])
        r = self.y - Hx

        s2 = float(self.priors.s_theta ** 2)
        S0_inv = (1.0 / s2) * np.eye(self.p_obs)
        m0 = np.full(self.p_obs, float(self.priors.m_theta))

        XtX = (X.T @ X) / R_var
        XtR = (X.T @ r) / R_var
        Sn_inv = S0_inv + XtX
        Sn = np.linalg.pinv(Sn_inv)
        mn = Sn @ (S0_inv @ m0 + XtR)
        self.theta_obs = np.random.multivariate_normal(mn, Sn)

    def _sample_slope_transition(self, x_path: np.ndarray) -> None:
        """
        Conjugate Gaussian update for the slope that appears *in the alpha transition*.
        Only used if (level dynamic & trend deterministic).
        Model for increments:  d_t = alpha_t - alpha_{t-1} - [beta_{t-1} if dynamic]  ~  N(slope, Q_alpha)
        Prior: slope ~ N(m_theta, s_theta^2)
        """
        if not self._slope_in_transition or self.i_alpha is None:
            return
        # collect increments
        d = []
        for t in range(1, self.T + 1):
            base = x_path[t, self.i_alpha] - x_path[t - 1, self.i_alpha]
            if self.i_beta is not None:
                base -= x_path[t - 1, self.i_beta]
            d.append(base)
        d = np.asarray(d, float)

        var = float(self.Qdiag[self.i_alpha]) if self.Qdiag[self.i_alpha] > 0 else 1e-8
        n = d.size
        s0 = float(self.priors.s_theta)
        m0 = float(self.priors.m_theta)

        Sn_inv = 1.0 / (s0 * s0) + n / var
        Sn = 1.0 / Sn_inv
        mn = Sn * (m0 / (s0 * s0) + np.sum(d) / var)
        self.slope_tr = float(np.random.normal(mn, math.sqrt(Sn)))

    # ---------------------------------
    # Conjugate variance updates
    # ---------------------------------
    def _update_R_IG(self, x_path: np.ndarray, H: np.ndarray) -> float:
        """R | y, x, theta_obs ~ IG(a*, b*)."""
        dyn = np.zeros(self.T)
        if self.dim > 0:
            for t in range(self.T):
                dyn[t] = float(H @ x_path[t + 1])
        resid = self.y - (dyn + self._offset_from_obs_params())
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
                    drift += x_path[t - 1, self.i_beta]
                if self._slope_in_transition:
                    drift += self.slope_tr
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
            # 1) States via FFBS given current (R, Q, static params)
            F, Q, H = self._build_F_Q_H()
            b_seq = self._control_seq()
            offset = self._offset_from_obs_params()
            self.x = self._ffbs(F, Q, H, self.R, offset, b_seq)

            # 2a) Observation static params (Gaussian regression)
            self._sample_theta_obs(R_var=self.R, x_path=self.x, H=H)

            # 2b) Transition static param (slope drift), if applicable
            if self._slope_in_transition and self.i_alpha is not None:
                self._sample_slope_transition(self.x)

            # 3) R | y,x,theta_obs
            self.R = self._update_R_IG(self.x, H)

            # 4) Q | x
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
                if self.p_obs > 0:
                    self.keep["theta_obs"][keep_i, :] = self.theta_obs
                if self._slope_in_transition:
                    self.keep["slope_transition"][keep_i] = self.slope_tr
                # reconstruct mu_t for this draw
                mu = np.zeros(self.T)
                if self.dim > 0:
                    for t in range(self.T):
                        mu[t] = float(H @ self.x[t + 1])
                mu += self._offset_from_obs_params()
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
    from simulator.mean_time_series import Mean_Time_Series  # <- simulator

    parser = argparse.ArgumentParser(
        description="Conjugate Gaussian DLM (FFBS+Gibbs) with deterministic-vs-dynamic split"
    )

    # Modes
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

    # Priors
    parser.add_argument("--a-sigma-y", type=float, default=2.5)
    parser.add_argument("--b-sigma-y", type=float, default=1.0)
    parser.add_argument("--a-Q-alpha", type=float, default=2.5)
    parser.add_argument("--b-Q-alpha", type=float, default=0.1)
    parser.add_argument("--a-Q-beta",  type=float, default=2.5)
    parser.add_argument("--b-Q-beta",  type=float, default=0.1)
    parser.add_argument("--a-Q-gamma", type=float, default=2.5)
    parser.add_argument("--b-Q-gamma", type=float, default=0.1)
    parser.add_argument("--m-theta", type=float, default=0.0)
    parser.add_argument("--s-theta", type=float, default=10.0)

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
        m0_season = m0_season_det; v0_season = v0_season_det; q_season = args.q_gamma
    elif args.season_mode == "dynamic":
        m0_season = m0_season_dyn; v0_season = v0_season_dyn; q_season = args.q_gamma
    else:
        m0_season = m0_season_none; v0_season = v0_season_none; q_season = args.q_gamma

    q_trend = args.q_beta if args.trend_mode == "dynamic" else 0.0
    q_level = args.q_alpha if args.level_mode == "dynamic" else 0.0
    m0_trend = (args.m0_trend if args.trend_mode != "none" else 0.0)

    # --- Simulate with Mean_Time_Series ---
    from datetime import datetime as _dt
    from simulator.mean_time_series import Mean_Time_Series
    sim = Mean_Time_Series(
        sigma=args.true_sigma_y,
        level_mode=args.level_mode,
        trend_mode=args.trend_mode,
        seasonal_mode=args.season_mode,
        period=args.period,
        q_level=q_level, q_trend=q_trend, q_season=q_season,
        m0_level=args.m0_level, v0_level=args.v0_level,
        m0_trend=m0_trend,       v0_trend=args.v0_trend,
        m0_season=m0_season,     v0_season=v0_season,
        start_date=_dt(2000, 1, 1),
    )

    y = np.array([sim.move() or sim.measure() for _ in range(args.T)], float)

    truth = sim.get_truth_paths(as_numpy=True)
    mu_true = truth["mu"][1:1 + args.T]
    dates   = truth["index"][:args.T]

    pri = DLM_Priors(
        a_sigma_y=args.a_sigma_y, b_sigma_y=args.b_sigma_y,
        a_Q_alpha=args.a_Q_alpha, b_Q_alpha=args.b_Q_alpha,
        a_Q_beta=args.a_Q_beta,   b_Q_beta=args.b_Q_beta,
        a_Q_gamma=args.a_Q_gamma, b_Q_gamma=args.b_Q_gamma,
        m_theta=args.m_theta,     s_theta=args.s_theta
    )
    cfg = DLM_Config(
        n_iter=args.n_iter, burn=args.burn, thin=args.thin,
        random_seed=args.seed, progress=bool(args.progress)
    )

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
        print("Posterior mean Q diag:", np.mean(posterior["Q"], axis=0))
    if "theta_obs" in posterior:
        print("Posterior mean theta_obs:", np.mean(posterior["theta_obs"], axis=0))
    if "slope_transition" in posterior:
        print("Posterior mean slope (transition):", float(np.mean(posterior["slope_transition"])))

    # --- Plot ---
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

    # --- Save ---
    tag = f"{args.level_mode}-{args.trend_mode}-{args.season_mode}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = args.out_dir or "results"
    run_dir = os.path.join(base_dir, f"{tag}_{timestamp}")
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    np.savez_compressed(os.path.join(run_dir, "posterior.npz"), **posterior)

    meta = {
        "T": int(args.T),
        "period": int(args.period),
        "modes": {
            "level_mode": args.level_mode,
            "trend_mode": args.trend_mode,
            "seasonal_mode": args.season_mode,
        },
        "latent_state_layout": list(mdl.tags),
        "obs_static_params": mdl.obs_param_names,
        "has_transition_slope": bool(mdl._slope_in_transition),
        "idx_alpha": mdl.i_alpha,
        "idx_beta": mdl.i_beta,
        "idx_gamma_end": (mdl.i_gL if args.season_mode == "dynamic" else None),
        "seed": int(args.seed),
        "elapsed_seconds": float(elapsed),
    }
    with open(os.path.join(run_dir, "posterior.meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if quick_fig is not None:
        png_path = os.path.join(fig_dir, "quick_overview.png")
        quick_fig.savefig(png_path, dpi=200, bbox_inches="tight")
        print(f"[save] Figure -> {png_path}")

    print(f"[save] Posterior -> {os.path.join(run_dir,'posterior.npz')}")
    print(f"[save] Metadata  -> {os.path.join(run_dir,'posterior.meta.json')}")
    print(f"[info] Figures will be saved to: {fig_dir}")
