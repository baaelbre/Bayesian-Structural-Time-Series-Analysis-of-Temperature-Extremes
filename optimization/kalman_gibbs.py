# %% DLM_Gibbs.py
from __future__ import annotations

import os, math, time, json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Tuple, List

import numpy as np
from tqdm import tqdm

# =============================================================================
# Utilities
# =============================================================================

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# =============================================================================
# Priors & configuration
# =============================================================================

@dataclass
class DLM_Priors:
    """
    Conjugate priors for a linear-Gaussian DLM.

    Variances use Inverse-Gamma IG(a,b):  p(v) ∝ v^{-(a+1)} exp(-b/v)
    Static parameters use Gaussian priors N(m_theta, s_theta^2) independently.
    """
    # Observation variance R = sigma_y^2
    a_sigma: float = 2.5
    b_sigma: float = 1.0

    # Process variances (per dynamic block)
    a_Q_alpha: float = 2.5
    b_Q_alpha: float = 0.1
    a_Q_beta:  float = 2.5
    b_Q_beta:  float = 0.1
    a_Q_gamma: float = 2.5
    b_Q_gamma: float = 0.1

    # Static parameters (Gaussian) — only used if deterministic pieces exist
    m_theta: float = 0.0
    s_theta: float = 10.0


@dataclass
class DLM_Config:
    n_iter: int = 4000
    burn: int = 1000
    thin: int = 2
    random_seed: Optional[int] = 123
    progress: bool = True
    progress_every: int = 0   # 0 => auto (~2% of n_iter)

# =============================================================================
# DLM with conjugate Gibbs + FFBS (Kalman + RTS)
# =============================================================================

class DLM_Gibbs:
    """
    Linear-Gaussian DLM with deterministic/dynamic switches:

      level_mode   ∈ {"dynamic", "deterministic"}         (no 'none' for level)
      trend_mode   ∈ {"dynamic", "deterministic", "none"}
      seasonal_mode∈ {"dynamic", "deterministic", "none"}

    Observation:
      y_t = mu_t + eps_t,     eps_t ~ N(0, R)
      mu_t = (level contribution) + (seasonal contribution) + (deterministic regressors)

    Dynamics (when active):
      alpha_{t+1} = alpha_t + [beta_{t} if dynamic] + w_alpha
      beta_{t+1}  = beta_t + w_beta
      seasonal (p-1 dims): shift left; newest coord = -sum(prev p-1) + w_gamma

    Conjugate Gibbs (no MH):
      - deterministic obs params (intercept, β_det in obs, seasonal dummies) ~ Gaussian
      - R | (y, x, det params)                                               ~ IG
      - Q_{•} | x                                                            ~ IG per dynamic coordinate
    """

    # --------------------------- Construction --------------------------- #
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
        # Data & modes
        self.y = np.asarray(y, float)
        self.T = int(self.y.size)
        self.period = int(period)
        assert self.period >= 2, "period must be >= 2"

        assert level_mode in {"dynamic", "deterministic"}
        assert trend_mode in {"dynamic", "deterministic", "none"}
        assert seasonal_mode in {"dynamic", "deterministic", "none"}

        # dynamic trend requires a dynamic level
        if level_mode == "deterministic" and trend_mode == "dynamic":
            raise ValueError("trend_mode='dynamic' requires level_mode='dynamic'.")

        self.level_mode = level_mode
        self.trend_mode = trend_mode
        self.seasonal_mode = seasonal_mode

        self.priors = priors
        self.cfg = cfg
        if cfg.random_seed is not None:
            np.random.seed(cfg.random_seed)

        # ----- latent state layout: ONLY dynamic components ----- #
        tags: List[str] = []
        if self.level_mode == "dynamic":
            tags.append("alpha")
        if self.trend_mode == "dynamic":
            tags.append("beta")
        if self.seasonal_mode == "dynamic":
            tags.extend([f"gamma_{k}" for k in range(1, self.period)])  # p-1 coords
        self.tags = tags
        self.dim = len(tags)

        # indices
        self.i_alpha = tags.index("alpha") if "alpha" in tags else None
        self.i_beta  = tags.index("beta")  if "beta"  in tags else None
        if self.seasonal_mode == "dynamic":
            self.i_g0 = tags.index("gamma_1")
            self.i_gL = self.i_g0 + (self.period - 2)
        else:
            self.i_g0 = None
            self.i_gL = None

        # ----- deterministic / static parameters split ----- #
        # Observation X_obs θ_obs contains:
        #   - intercept if level deterministic
        #   - slope (t) whenever trend is deterministic (regardless of level mode)
        #   - seasonal dummies (first p-1) if seasonal deterministic (last implied)
        self._slope_in_obs = (self.trend_mode == "deterministic")
        self._slope_in_transition = False  # <-- IMPORTANT: no drift in alpha transition

        obs_cols: List[str] = []
        if self.level_mode == "deterministic":
            obs_cols.append("intercept")
        if self._slope_in_obs:
            obs_cols.append("slope_obs")
        if self.seasonal_mode == "deterministic":
            obs_cols += [f"season{k}" for k in range(1, self.period)]  # last implied by sum-to-zero
        self.obs_param_names = obs_cols
        self.p_obs = len(obs_cols)
        self.theta_obs = np.zeros(self.p_obs, float)

        # ----- initial state path & variances ----- #
        self.x = np.zeros((self.T + 1, self.dim), float)  # 0..T

        # observation & process variances
        self.R = 1.0
        self.Qdiag = np.full(self.dim, 1e-2) if self.dim > 0 else np.zeros(0)

        # ---- Truth overlays (optional) ---- #
        self.true_sigma_y: Optional[float] = None
        self.true_Q: Optional[np.ndarray] = None
        self.true_mu_t: Optional[np.ndarray] = None

        # ---- rolling diagnostics ---- #
        self.last_log_evidence: float = float("nan")
        self._ema_logZ: Optional[float] = None

        # ---- storage ---- #
        save_iters = list(range(self.cfg.burn, self.cfg.n_iter, max(1, self.cfg.thin)))
        self._save_iters = save_iters
        n_kept = max(0, len(save_iters))
        self.keep: Dict[str, np.ndarray] = {
            "mu":       np.zeros((n_kept, self.T), float),
            "sigma_y":  np.zeros(n_kept, float),     # store std
            "log_evidence": np.zeros(n_kept, float),
        }
        if self.dim > 0:
            self.keep["x"] = np.zeros((n_kept, self.T, self.dim), float)
            self.keep["Q"] = np.zeros((n_kept, self.dim), float)
        # deterministic pieces stored separately
        if self.level_mode == "deterministic":
            self.keep["intercept"] = np.zeros(n_kept, float)
        if self._slope_in_obs:
            self.keep["slope_obs"] = np.zeros(n_kept, float)
        if self.seasonal_mode == "deterministic":
            self.keep["season_pminus1"] = np.zeros((n_kept, self.period - 1), float)
            self.keep["season_full"]    = np.zeros((n_kept, self.period), float)  # with implied last

    # ------------------------ Truth registration (optional) ------------------------ #
    def set_truth(self, sigma_y: Optional[float] = None, Q: Optional[np.ndarray] = None) -> None:
        self.true_sigma_y = sigma_y
        self.true_Q = None if Q is None else np.asarray(Q, float)

    def set_truth_paths(self, mu: Optional[np.ndarray] = None) -> None:
        self.true_mu_t = None if mu is None else np.asarray(mu, float)

    # ------------------------------- Design matrices ------------------------------ #
    def _design_matrix_obs(self) -> np.ndarray:
        """Build X_obs for observation static params θ_obs."""
        if self.p_obs == 0:
            return np.zeros((self.T, 0))
        cols = []
        t = np.arange(self.T, dtype=float)
        for name in self.obs_param_names:
            if name == "intercept":
                cols.append(np.ones(self.T))
            elif name == "slope_obs":
                cols.append(t)
            elif name.startswith("season"):
                k = int(name.replace("season", ""))
                cols.append(((t % self.period) == k).astype(float))  # last implied
            else:
                raise RuntimeError(f"Unknown obs param column: {name}")
        return np.column_stack(cols)

    def _offset_from_obs_params(self) -> np.ndarray:
        if self.p_obs == 0:
            return np.zeros(self.T)
        return self._design_matrix_obs() @ self.theta_obs

    def _control_seq(self) -> np.ndarray:
        """No deterministic drift in transitions."""
        if self.dim == 0:
            return np.zeros((self.T, 0))
        return np.zeros((self.T, self.dim), float)

    # ----------------------------- System matrices ----------------------------- #
    def _build_F_Q_H(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        D = self.dim
        F = np.eye(D)
        Q = np.diag(np.clip(self.Qdiag.copy(), 1e-12, None))
        H = np.zeros(D)

        # alpha
        if self.i_alpha is not None:
            H[self.i_alpha] = 1.0
            if self.i_beta is not None:
                F[self.i_alpha, self.i_beta] = 1.0  # drift by beta

        # beta
        if self.i_beta is not None:
            F[self.i_beta, self.i_beta] = 1.0

        # seasonal dynamic (p-1 dims)
        if self.seasonal_mode == "dynamic":
            # shift gamma1..gamma_{p-2} <- gamma2..gamma_{p-1}
            for k in range(self.i_g0, self.i_gL):
                F[k, k] = 0.0
                F[k, k + 1] = 1.0
            # newest coord = -sum(prev p-1) + noise
            F[self.i_gL, self.i_g0:self.i_gL + 1] = -1.0
            H[self.i_gL] = 1.0  # last coord enters observation

        return F, Q, H.reshape(1, -1)  # H as (1, D)

    # ----------------------- Kalman filter & RTS smoother ---------------------- #
    def _kalman_innovations_and_filter(
        self, F, Q, H, R, offset_mu: np.ndarray, b_seq: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray], float]:
        """
        Returns filtered means m[t], covs C[t] for t=0..T and innovation log-likelihood.
        Diffuse prior: m[0]=0, C[0]=1e6 I.
        """
        T, D = self.T, self.dim
        m = np.zeros((T + 1, D), float)
        C = [np.eye(D) * 1e6 for _ in range(T + 1)]
        loglik = 0.0

        for t in range(1, T + 1):
            # Predict
            a = F @ m[t - 1]
            if D > 0:
                a = a + b_seq[t - 1]
            Rpred = F @ C[t - 1] @ F.T + Q

            # Forecast & update
            yhat = (H @ a.reshape(-1, 1)).item() + offset_mu[t - 1]
            S = (H @ Rpred @ H.T).item() + R
            v = self.y[t - 1] - yhat

            # Innovations log-likelihood
            S_safe = max(S, 1e-300)
            loglik += -0.5 * (math.log(2.0 * math.pi) + math.log(S_safe) + (v * v) / S_safe)

            # Kalman update
            if D > 0:
                K = (Rpred @ H.T).reshape(-1) / max(S, 1e-12)
                m[t] = a + K * v
                C[t] = Rpred - np.outer(K, K) * S_safe
            else:
                m[t] = np.zeros(0)
                C[t] = np.zeros((0, 0))

        return m, C, float(loglik)

    def _rts_smoother(self, F, Q, m, C, b_seq: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        T, D = self.T, self.dim
        ms = m.copy()
        Cs = [Ci.copy() for Ci in C]
        for t in range(T - 1, -1, -1):
            if D == 0:
                continue
            Rpred = F @ C[t] @ F.T + Q
            J = C[t] @ F.T @ np.linalg.pinv(Rpred)
            ms[t] = m[t] + J @ (ms[t + 1] - (F @ m[t] + b_seq[t]))
            Cs[t] = C[t] + J @ (Cs[t + 1] - Rpred) @ J.T
        return ms, Cs

    # ----------------------------- FFBS state sampling ---------------------------- #
    def _ffbs(self, F, Q, H, R, offset_mu, b_seq) -> Tuple[np.ndarray, float]:
        if self.dim == 0:
            # no dynamic state: only deterministic parameters
            resid = self.y - offset_mu
            R_safe = max(R, 1e-12)
            loglik = -0.5 * np.sum(np.log(2 * np.pi * R_safe) + (resid ** 2) / R_safe)
            return np.zeros((self.T + 1, 0)), float(loglik)

        m, C, loglik = self._kalman_innovations_and_filter(F, Q, H, R, offset_mu, b_seq)
        ms, Cs = self._rts_smoother(F, Q, m, C, b_seq)

        T, D = self.T, self.dim
        x = np.zeros((T + 1, D), float)

        # sample x_T
        covT = Cs[T] + 1e-12 * np.eye(D)
        x[T] = np.random.multivariate_normal(ms[T], covT)

        for t in range(T - 1, -1, -1):
            Rpred = F @ C[t] @ F.T + Q
            J = C[t] @ F.T @ np.linalg.pinv(Rpred)
            mean = ms[t] + J @ (x[t + 1] - (F @ m[t] + b_seq[t]))
            cov = Cs[t] - J @ Rpred @ J.T
            cov = (cov + cov.T) * 0.5
            cov += 1e-12 * np.eye(D)
            x[t] = np.random.multivariate_normal(mean, cov)

        return x, float(loglik)

    # ------------------------------ Static parameter updates ------------------------------ #
    def _sample_theta_obs(self, R_var: float, x_path: np.ndarray, H: np.ndarray) -> None:
        if self.p_obs == 0:
            return
        X = self._design_matrix_obs()

        # subtract dynamic contribution H x_t
        Hx = np.zeros(self.T)
        if self.dim > 0:
            Hx = (H @ x_path[1:].T).ravel()
        r = self.y - Hx

        s2 = float(self.priors.s_theta ** 2)
        S0_inv = (1.0 / s2) * np.eye(self.p_obs)
        m0 = np.full(self.p_obs, float(self.priors.m_theta))

        R_safe = max(R_var, 1e-12)
        XtX = (X.T @ X) / R_safe
        XtR = (X.T @ r) / R_safe
        Sn_inv = S0_inv + XtX
        Sn = np.linalg.pinv(Sn_inv)
        mn = Sn @ (S0_inv @ m0 + XtR)
        Sn = Sn + 1e-12 * np.eye(self.p_obs)
        self.theta_obs = np.random.multivariate_normal(mn, Sn)

    # --------------------------------- Conjugate variance updates --------------------------------- #
    def _update_R_IG(self, x_path: np.ndarray, H: np.ndarray) -> float:
        dyn = np.zeros(self.T)
        if self.dim > 0:
            dyn = (H @ x_path[1:].T).ravel()
        resid = self.y - (dyn + self._offset_from_obs_params())
        rss = float(resid @ resid)
        a = self.priors.a_sigma + 0.5 * self.T
        b = self.priors.b_sigma + 0.5 * rss
        return float(1.0 / np.random.gamma(a, 1.0 / max(b, 1e-12)))

    def _update_Q_IG(self, x_path: np.ndarray) -> np.ndarray:
        Qdiag = np.zeros(self.dim)

        # alpha innovations
        if self.i_alpha is not None:
            inc = []
            for t in range(1, self.T + 1):
                drift = 0.0
                if self.i_beta is not None:  # dynamic trend
                    drift += x_path[t - 1, self.i_beta]
                inc.append(x_path[t, self.i_alpha] - (x_path[t - 1, self.i_alpha] + drift))
            inc = np.asarray(inc)
            rss = float(inc @ inc)
            a = self.priors.a_Q_alpha + 0.5 * inc.size
            b = self.priors.b_Q_alpha + 0.5 * rss
            Qdiag[self.i_alpha] = 1.0 / np.random.gamma(a, 1.0 / max(b, 1e-12))

        # beta innovations
        if self.i_beta is not None:
            inc = x_path[1:, self.i_beta] - x_path[:-1, self.i_beta]
            rss = float(inc @ inc)
            a = self.priors.a_Q_beta + 0.5 * inc.size
            b = self.priors.b_Q_beta + 0.5 * rss
            Qdiag[self.i_beta] = 1.0 / np.random.gamma(a, 1.0 / max(b, 1e-12))

        # seasonal newest coord innovations
        if self.seasonal_mode == "dynamic":
            inc = []
            for t in range(1, self.T + 1):
                prev = x_path[t - 1, self.i_g0:self.i_gL + 1]
                mean_new = -float(np.sum(prev))
                inc.append(x_path[t, self.i_gL] - mean_new)
            inc = np.asarray(inc)
            rss = float(inc @ inc)
            a = self.priors.a_Q_gamma + 0.5 * inc.size
            b = self.priors.b_Q_gamma + 0.5 * rss
            Qdiag[self.i_gL] = 1.0 / np.random.gamma(a, 1.0 / max(b, 1e-12))

        return Qdiag

    # --------------------------------- Pretty progress helpers --------------------------------- #
    def _q_snapshot(self, ema_Q: np.ndarray | None = None) -> str:
        """Readable Q subset in log10, matching DGEV style."""
        if self.dim == 0:
            return ""
        rows = []
        def add(label, idx):
            if idx is None: return
            q = float(self.Qdiag[idx])
            logq = np.log10(max(q, 1e-20))
            if ema_Q is not None:
                ema = float(ema_Q[idx])
                logema = np.log10(max(ema, 1e-20))
                rows.append(f"{label}[log10]: cur={logq:6.2f} ema={logema:6.2f}")
            else:
                rows.append(f"{label}[log10]: cur={logq:6.2f}")
        add("Q_alpha", self.i_alpha)
        add("Q_beta",  self.i_beta)
        if self.seasonal_mode == "dynamic":
            add("Q_gamma(last)", self.i_gL)
        return " | " + " | ".join(rows)

    def _theta_snapshot_named(self) -> str:
        if self.p_obs == 0:
            return ""
        out = []
        pos = 0
        if self.level_mode == "deterministic":
            out.append(f"intercept≈{self.theta_obs[pos]:.4f}"); pos += 1
        if self._slope_in_obs:
            out.append(f"slope_obs≈{self.theta_obs[pos]:.6f}"); pos += 1
        if self.seasonal_mode == "deterministic":
            seas = self.theta_obs[pos:pos + (self.period - 1)]
            out.append("γ_det[:3]≈" + np.array2string(seas[:min(3, seas.size)], precision=3, separator=","))
        return " | " + " | ".join(out) if out else ""

    # --------------------------------- Main sampler --------------------------------- #
    def run(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        save_iters = self._save_iters
        n_kept = self.keep["sigma_y"].size
        keep_idx = 0

        print_every = cfg.progress_every if cfg.progress_every > 0 else max(1, cfg.n_iter // 50)
        ema_Q = np.zeros(self.dim, float) if self.dim > 0 else None

        loop = tqdm(range(cfg.n_iter), disable=not cfg.progress, leave=False)
        for it in loop:
            # 1) States via FFBS given current (R, Q, static params)
            F, Q, H = self._build_F_Q_H()
            b_seq = self._control_seq()
            offset = self._offset_from_obs_params()
            self.x, current_log_ev = self._ffbs(F, Q, H, self.R, offset, b_seq)

            # logZ EMA
            self.last_log_evidence = float(current_log_ev)
            self._ema_logZ = self.last_log_evidence if (self._ema_logZ is None) else (0.9 * self._ema_logZ + 0.1 * self.last_log_evidence)

            # 2) deterministic obs params
            self._sample_theta_obs(R_var=self.R, x_path=self.x, H=H)

            # 3) R | y,x,θ_obs
            self.R = self._update_R_IG(self.x, H)

            # 4) Q | x
            if self.dim > 0:
                self.Qdiag = self._update_Q_IG(self.x)
                if ema_Q is not None:
                    ema_Q = 0.9 * ema_Q + 0.1 * self.Qdiag

            # 5) Compact progress line
            if ((it + 1) % print_every == 0) or (it + 1 == cfg.n_iter):
                msg = (
                    f"[it {it+1}/{cfg.n_iter}] "
                    f"logZ={self.last_log_evidence:.3f} ema={self._ema_logZ:.3f} "
                    f"σ={math.sqrt(max(self.R,0.0)):.3f}"
                    f"{self._q_snapshot(ema_Q)}"
                    f"{self._theta_snapshot_named()}"
                )
                if cfg.progress:
                    loop.set_postfix_str(msg)

            # 6) Save draws
            if it in save_iters and keep_idx < n_kept:
                # reconstruct mu_t for this draw
                if self.dim > 0:
                    mu = (H @ self.x[1:].T).ravel()
                else:
                    mu = np.zeros(self.T)
                mu += self._offset_from_obs_params()

                self.keep["mu"][keep_idx, :] = mu
                self.keep["sigma_y"][keep_idx] = math.sqrt(max(self.R, 0.0))
                self.keep["log_evidence"][keep_idx] = self.last_log_evidence

                if self.dim > 0:
                    self.keep["x"][keep_idx, :, :] = self.x[1:self.T + 1]
                    self.keep["Q"][keep_idx, :] = self.Qdiag

                # deterministic pieces saved by name
                pos = 0
                if self.level_mode == "deterministic":
                    self.keep["intercept"][keep_idx] = self.theta_obs[pos]; pos += 1
                if self._slope_in_obs:
                    self.keep["slope_obs"][keep_idx] = self.theta_obs[pos]; pos += 1
                if self.seasonal_mode == "deterministic":
                    seas = self.theta_obs[pos:pos + (self.period - 1)]
                    self.keep["season_pminus1"][keep_idx, :] = seas
                    full = np.concatenate([seas, [-float(np.sum(seas))]])
                    self.keep["season_full"][keep_idx, :] = full

                keep_idx += 1

        if self.cfg.progress:
            tqdm.write(f"[{it + 1}/{cfg.n_iter}] kept={keep_idx} | final σ={math.sqrt(max(self.R,0.0)):.3f}")

        return self.keep

    # --------------------------------- Persistence --------------------------------- #
    def save_posterior(self, out_npz_path: str, extra_meta: Optional[dict] = None) -> None:
        _ensure_dir(os.path.dirname(out_npz_path))

        arrays = dict(self.keep)
        arrays["y"] = self.y.copy()
        arrays["x_last"] = self.x[1 : self.T + 1].copy() if self.dim > 0 else np.zeros((self.T, 0))
        if self.true_mu_t is not None:
            arrays["true_mu_t"] = np.asarray(self.true_mu_t, float)

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
            "latent_state_layout": list(self.tags),
            "idx_alpha": self.i_alpha,
            "idx_beta": self.i_beta,
            "idx_gamma_start": self.i_g0,
            "idx_gamma_end": self.i_gL,
            "cfg": asdict(self.cfg),
            "priors": asdict(self.priors),
            "true_sigma_y": self.true_sigma_y,
            "true_Q": (None if self.true_Q is None else np.asarray(self.true_Q, float).tolist()),
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
    import argparse
    import matplotlib.pyplot as plt
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from simulator.mean_time_series import Mean_Time_Series  # <- your Gaussian simulator

    parser = argparse.ArgumentParser(description="Gaussian DLM (FFBS+Gibbs)")

    # Modes
    parser.add_argument("--level-mode",  choices=["dynamic", "deterministic"], default="dynamic")
    parser.add_argument("--trend-mode",  choices=["dynamic", "deterministic", "none"], default="deterministic")
    parser.add_argument("--season-mode", choices=["dynamic", "deterministic", "none"], default="none")

    # Data & simulation controls
    parser.add_argument("--period", type=int, default=12)
    parser.add_argument("--T", type=int, default=240)
    parser.add_argument("--seed", type=int, default=7)

    # Truths for simulator (obs sd and process variances)
    parser.add_argument("--true-sigma", type=float, default=1.5)
    parser.add_argument("--q-alpha", type=float, default=0.01)
    parser.add_argument("--q-beta",  type=float, default=0.005)
    parser.add_argument("--q-gamma", type=float, default=0.02)

    # Simulator priors / fixed values
    parser.add_argument("--m0-level",  type=float, default=5.0)
    parser.add_argument("--v0-level",  type=float, default=0.25)
    parser.add_argument("--m0-trend",  type=float, default=0.01)
    parser.add_argument("--v0-trend",  type=float, default=0.05)

    # Priors (choose a_sigma/b_sigma so mean(b/(a-1)) roughly matches expected σ^2)
    parser.add_argument("--a-sigma", type=float, default=2.1)
    parser.add_argument("--b-sigma", type=float, default=3.0)
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
    parser.add_argument("--progress-every", type=int, default=10)

    # Output & plotting
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--show-plots", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # --- Simulate Gaussian series ---
    sim = Mean_Time_Series(
        sigma=args.true_sigma,
        level_mode=args.level_mode,
        trend_mode=args.trend_mode,
        seasonal_mode=args.season_mode,
        period=args.period,
        q_level=(args.q_alpha if args.level_mode == "dynamic" else 0.0),
        q_trend=(args.q_beta  if args.trend_mode == "dynamic" else 0.0),
        q_season=(args.q_gamma if args.season_mode == "dynamic" else 0.0),
        m0_level=args.m0_level, v0_level=args.v0_level,
        m0_trend=(args.m0_trend if args.trend_mode != "none" else 0.0), v0_trend=args.v0_trend,
        m0_season=[0.0]*(args.period-1), v0_season=[0.5]*(args.period-1),
        start_date=datetime(2000, 1, 1),
    )

    y = np.array([sim.move() or sim.measure() for _ in range(args.T)], float)
    truth = sim.get_truth_paths(as_numpy=True)
    mu_true = truth["mu"][1:1 + args.T]
    dates   = truth["index"][:args.T]

    # Priors & config
    pri = DLM_Priors(
        a_sigma=args.a_sigma, b_sigma=args.b_sigma,
        a_Q_alpha=args.a_Q_alpha, b_Q_alpha=args.b_Q_alpha,
        a_Q_beta=args.a_Q_beta,   b_Q_beta=args.b_Q_beta,
        a_Q_gamma=args.a_Q_gamma, b_Q_gamma=args.b_Q_gamma,
        m_theta=args.m_theta,     s_theta=args.s_theta,
    )
    cfg = DLM_Config(
        n_iter=args.n_iter, burn=args.burn, thin=args.thin,
        random_seed=args.seed, progress=bool(args.progress),
        progress_every=int(args.progress_every),
    )

    # --- Run sampler ---
    t0 = time.time()
    mdl = DLM_Gibbs(
        y=y, period=args.period,
        level_mode=args.level_mode, trend_mode=args.trend_mode, seasonal_mode=args.season_mode,
        priors=pri, cfg=cfg
    )
    mdl.set_truth(sigma_y=args.true_sigma)
    mdl.set_truth_paths(mu=mu_true)

    posterior = mdl.run()
    elapsed = time.time() - t0
    print(f"Run time: {elapsed:.2f}s")

    # --- Summaries (DGEV-style) ---
    print(f"Posterior mean sigma_y: {np.mean(posterior['sigma_y']):.3f}  (true {args.true_sigma})")

    if "Q" in posterior and posterior["Q"].size > 0:
        means = np.mean(posterior["Q"], axis=0)
        log10_means = np.log10(np.clip(means, 1e-20, None))
        print("Posterior mean Q diag (log10):", log10_means)

    # deterministic bits
    if "intercept" in posterior:
        print(f"Posterior mean intercept: {float(np.mean(posterior['intercept'])):.4f}")
    if "slope_obs" in posterior:
        print(f"Posterior mean slope_obs: {float(np.mean(posterior['slope_obs'])):.6f}")
    if "season_full" in posterior:
        sf = posterior["season_full"]
        print("Posterior mean seasonal (full cycle):", np.mean(sf, axis=0))

    if "log_evidence" in posterior:
        le = posterior["log_evidence"]
        print(f"Mean log p(y|θ): {np.nanmean(le):.3f} | Median: {np.nanmedian(le):.3f} | Best: {np.nanmax(le):.3f}")

    # --- Quick plot ---
    quick_fig = None
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

    # --- Save posterior & metadata ---
    tag = f"{args.level_mode}-{args.trend_mode}-{args.season_mode}"
    out_dir = args.out_dir or os.path.join("results", "simulations", "DLM",
                                           f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    fig_dir = os.path.join(out_dir, "figures")
    _ensure_dir(out_dir); _ensure_dir(fig_dir)

    mdl.save_posterior(
        out_npz_path=os.path.join(out_dir, "posterior.npz"),
        extra_meta={"modes": tag, "elapsed_seconds": float(elapsed)},
    )

    if quick_fig is not None:
        png_path = os.path.join(fig_dir, "quick_overview.png")
        quick_fig.savefig(png_path, dpi=200, bbox_inches="tight")
        print(f"[save] Figure -> {png_path}")

    print(f"[info] Figures will be saved to: {fig_dir}")
