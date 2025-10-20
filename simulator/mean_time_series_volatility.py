# mean_time_series.py
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt


def _season_F(period: int) -> np.ndarray:
    """
    Companion matrix for a sum-to-zero seasonal (period p => state size p-1),
    newest-first convention. First row is -1's, lower subdiagonal is identity.
    """
    p = int(period)
    if p < 2:
        raise ValueError("period must be >= 2")
    m = p - 1
    if m == 0:
        return np.zeros((0, 0))
    F = np.zeros((m, m), float)
    F[0, :] = -1.0
    if m > 1:
        F[1:, :-1] = np.eye(m - 1)
    return F


class Mean_Time_Series:
    """
    Structural Gaussian simulator with independent switches for BOTH
      - the mean (μ) block, and
      - the log-scale (η = ln σ) block.

    Modes
    -----
    μ block:
      level_mode       in {"dynamic","deterministic"}
      trend_mode       in {"dynamic","deterministic","none"}
      seasonal_mode    in {"dynamic","deterministic","none"}

    η block:
      level_mode_sigma    in {"dynamic","deterministic"}
      trend_mode_sigma    in {"dynamic","deterministic","none"}
      seasonal_mode_sigma in {"dynamic","deterministic","none"}

    Observation
    -----------
      y_t ~ Normal( μ_t , exp(η_t)^2 )

    Decomposition
    -------------
      μ_t  = (level + linear trend contribution) + (seasonal contribution)
      η_t  = (level + linear trend contribution) + (seasonal contribution)

    Seasonal state uses "newest-first" storage for both blocks.
    """

    # ------------------------------ init ------------------------------
    def __init__(self,
                 # ---------- μ block modes ----------
                 level_mode: str = "dynamic",
                 trend_mode: str = "dynamic",
                 seasonal_mode: str = "dynamic",

                 # ---------- η block modes ----------
                 level_mode_sigma: str = "deterministic",
                 trend_mode_sigma: str = "none",
                 seasonal_mode_sigma: str = "none",

                 # shared seasonal period
                 period: int = 12,

                 # ---------- μ innovations ----------
                 q_level: float = 0.05,
                 q_trend: float = 0.01,
                 q_season: float = 0.10,

                 # ---------- η innovations ----------
                 q_level_sigma: float = 0.00,
                 q_trend_sigma: float = 0.00,
                 q_season_sigma: float = 0.00,

                 # ---------- μ priors / fixed ----------
                 m0_level: float = 0.0,
                 v0_level: float = 1.0,
                 m0_trend: float = 0.0,
                 v0_trend: float = 1.0,
                 m0_season: list[float] | np.ndarray | None = None,  # length period-1, oldest→newest
                 v0_season: list[float] | np.ndarray | None = None,  # length period-1

                 # ---------- η priors / fixed ----------
                 m0_level_sigma: float | None = None,   # default to 0 => σ=1
                 v0_level_sigma: float | None = None,   # default to 0 (deterministic unless dynamic + q>0)
                 m0_trend_sigma: float = 0.0,
                 v0_trend_sigma: float = 1.0,
                 m0_season_sigma: list[float] | np.ndarray | None = None,
                 v0_season_sigma: list[float] | np.ndarray | None = None,

                 # time
                 start_date: datetime | None = None):

        # ---- validate modes (μ) ----
        assert level_mode in {"dynamic", "deterministic"}
        assert trend_mode in {"dynamic", "deterministic", "none"}
        assert seasonal_mode in {"dynamic", "deterministic", "none"}

        # ---- validate modes (η) ----
        assert level_mode_sigma in {"dynamic", "deterministic"}
        assert trend_mode_sigma in {"dynamic", "deterministic", "none"}
        assert seasonal_mode_sigma in {"dynamic", "deterministic", "none"}

        # basic
        self.period = int(period)
        if self.period < 2:
            raise ValueError("period must be >= 2.")

        # =========== μ (mean) PARAMETERS ===========
        self.q_level = float(q_level)
        self.q_trend = float(q_trend)
        self.q_season = float(q_season)

        self.v0_level = float(v0_level)
        self.v0_trend = float(v0_trend)
        self.m0_level = float(m0_level)
        self.m0_trend = float(m0_trend)

        if m0_season is None:
            m0_season = [0.0] * (self.period - 1)
        if v0_season is None:
            v0_season = [1.0] * (self.period - 1)
        m0_season = np.asarray(m0_season, float).reshape(-1)
        v0_season = np.asarray(v0_season, float).reshape(-1)
        if m0_season.size != self.period - 1:
            raise ValueError(f"m0_season must have length period-1 = {self.period-1}.")
        if v0_season.size != self.period - 1:
            raise ValueError(f"v0_season must have length period-1 = {self.period-1}.")
        if np.any(v0_season < 0.0) or self.v0_level < 0.0 or self.v0_trend < 0.0:
            raise ValueError("All μ prior variances must be >= 0.")
        # flip to newest-first storage
        self.m0_season = m0_season[::-1].copy()
        self.v0_season = v0_season[::-1].copy()

        # =========== η (log σ) PARAMETERS ===========
        # Defaults: neutral scale σ=1 (η=0), deterministic unless you choose dynamic with q>0
        if m0_level_sigma is None:
            m0_level_sigma = 0.0
        if v0_level_sigma is None:
            v0_level_sigma = 0.0

        self.q_level_sigma = float(q_level_sigma)
        self.q_trend_sigma = float(q_trend_sigma)
        self.q_season_sigma = float(q_season_sigma)

        self.v0_level_sigma = float(v0_level_sigma)
        self.v0_trend_sigma = float(v0_trend_sigma)

        self.m0_level_sigma = float(m0_level_sigma)
        self.m0_trend_sigma = float(m0_trend_sigma)

        if m0_season_sigma is None:
            m0_season_sigma = [0.0] * (self.period - 1)
        if v0_season_sigma is None:
            v_default_sig = 0.0 if seasonal_mode_sigma == "deterministic" else 0.5
            v0_season_sigma = [v_default_sig] * (self.period - 1)

        m0_season_sigma = np.asarray(m0_season_sigma, float).reshape(-1)
        v0_season_sigma = np.asarray(v0_season_sigma, float).reshape(-1)
        if m0_season_sigma.size != self.period - 1:
            raise ValueError(f"m0_season_sigma must have length period-1 = {self.period-1}.")
        if v0_season_sigma.size != self.period - 1:
            raise ValueError(f"v0_season_sigma must have length period-1 = {self.period-1}.")
        if np.any(v0_season_sigma < 0.0) or self.v0_level_sigma < 0.0 or self.v0_trend_sigma < 0.0:
            raise ValueError("All σ prior variances must be >= 0.")
        self.m0_season_sigma = m0_season_sigma[::-1].copy()
        self.v0_season_sigma = v0_season_sigma[::-1].copy()

        # ---------- auto-overrides (μ) ----------
        if level_mode == "dynamic" and (self.q_level == 0.0 or self.v0_level == 0.0):
            level_mode = "deterministic"
        if trend_mode == "dynamic" and (self.q_trend == 0.0 or self.v0_trend == 0.0):
            trend_mode = "none" if np.isclose(self.m0_trend, 0.0) else "deterministic"
        if seasonal_mode == "dynamic" and (self.q_season == 0.0 or np.allclose(self.v0_season, 0.0)):
            seasonal_mode = "none" if np.allclose(self.m0_season, 0.0) else "deterministic"

        # ---------- auto-overrides (η) ----------
        if level_mode_sigma == "dynamic" and (self.q_level_sigma == 0.0 or self.v0_level_sigma == 0.0):
            level_mode_sigma = "deterministic"
        if trend_mode_sigma == "dynamic" and (self.q_trend_sigma == 0.0 or self.v0_trend_sigma == 0.0):
            trend_mode_sigma = "none" if np.isclose(self.m0_trend_sigma, 0.0) else "deterministic"
        if seasonal_mode_sigma == "dynamic" and (self.q_season_sigma == 0.0 or np.allclose(self.v0_season_sigma, 0.0)):
            seasonal_mode_sigma = "none" if np.allclose(self.m0_season_sigma, 0.0) else "deterministic"

        # finalize modes
        self.level_mode_mu = level_mode
        self.trend_mode_mu = trend_mode
        self.seasonal_mode_mu = seasonal_mode
        self.level_mode_sigma = level_mode_sigma
        self.trend_mode_sigma = trend_mode_sigma
        self.seasonal_mode_sigma = seasonal_mode_sigma

        # ---------- deterministic proxies from priors ----------
        # μ
        self.fixed_level_mu = (self.m0_level if self.level_mode_mu == "deterministic" else None)
        if self.trend_mode_mu == "deterministic":
            self.fixed_trend_mu = float(self.m0_trend)
        elif self.trend_mode_mu == "none":
            self.fixed_trend_mu = 0.0
        else:
            self.fixed_trend_mu = None
        if self.seasonal_mode_mu == "deterministic":
            last = -float(np.sum(self.m0_season[::-1]))  # sum-zero
            self.fixed_season_mu = list(self.m0_season[::-1].astype(float)) + [last]
        elif self.seasonal_mode_mu == "none":
            self.fixed_season_mu = [0.0] * self.period
        else:
            self.fixed_season_mu = None

        # η
        self.fixed_level_sigma = (self.m0_level_sigma if self.level_mode_sigma == "deterministic" else None)
        if self.trend_mode_sigma == "deterministic":
            self.fixed_trend_sigma = float(self.m0_trend_sigma)
        elif self.trend_mode_sigma == "none":
            self.fixed_trend_sigma = 0.0
        else:
            self.fixed_trend_sigma = None
        if self.seasonal_mode_sigma == "deterministic":
            last = -float(np.sum(self.m0_season_sigma[::-1]))
            self.fixed_season_sigma = list(self.m0_season_sigma[::-1].astype(float)) + [last]
        elif self.seasonal_mode_sigma == "none":
            self.fixed_season_sigma = [0.0] * self.period
        else:
            self.fixed_season_sigma = None

        # ---------- state layouts ----------
        self._state_layout_mu: list[str] = []
        if self.level_mode_mu == "dynamic": self._state_layout_mu.append("alpha_mu")
        if self.trend_mode_mu == "dynamic": self._state_layout_mu.append("beta_mu")
        if self.seasonal_mode_mu == "dynamic":
            self._state_layout_mu.extend([f"gmu{k}" for k in range(1, self.period)])

        self._state_layout_sigma: list[str] = []
        if self.level_mode_sigma == "dynamic": self._state_layout_sigma.append("alpha_sig")
        if self.trend_mode_sigma == "dynamic": self._state_layout_sigma.append("beta_sig")
        if self.seasonal_mode_sigma == "dynamic":
            self._state_layout_sigma.extend([f"gsig{k}" for k in range(1, self.period)])

        self.n_latent_mu = len(self._state_layout_mu)
        self.n_latent_sigma = len(self._state_layout_sigma)

        # ---------- initialize latent state vectors ----------
        self.x_mu = np.array([], float)
        self.x_sig = np.array([], float)

        if self.n_latent_mu > 0:
            m0_vec, v0_vec = [], []
            for tag in self._state_layout_mu:
                if tag == "alpha_mu":
                    m0_vec.append(self.m0_level); v0_vec.append(self.v0_level)
                elif tag == "beta_mu":
                    m0_vec.append(self.m0_trend); v0_vec.append(self.v0_trend)
                else:
                    k = int(tag[3:])  # gmu{k}
                    m0_vec.append(self.m0_season[k - 1]); v0_vec.append(self.v0_season[k - 1])
            self.x_mu = np.random.normal(m0_vec, np.sqrt(v0_vec))

        if self.n_latent_sigma > 0:
            m0_vec, v0_vec = [], []
            for tag in self._state_layout_sigma:
                if tag == "alpha_sig":
                    m0_vec.append(self.m0_level_sigma); v0_vec.append(self.v0_level_sigma)
                elif tag == "beta_sig":
                    m0_vec.append(self.m0_trend_sigma); v0_vec.append(self.v0_trend_sigma)
                else:
                    k = int(tag[4:])  # gsig{k}
                    m0_vec.append(self.m0_season_sigma[k - 1]); v0_vec.append(self.v0_season_sigma[k - 1])
            self.x_sig = np.random.normal(m0_vec, np.sqrt(v0_vec))

        # ---------- time bookkeeping & outputs ----------
        self.t = 0
        self.current_date = start_date if start_date else datetime.now()

        self.index: list[datetime] = []
        self.all_measurements: list[float] = []
        # paths
        self.alpha_mu_path: list[float] = []
        self.beta_mu_path: list[float] = []
        self.gamma_mu_path: list[float] = []
        self.mu_path: list[float] = []

        self.alpha_sig_path: list[float] = []
        self.beta_sig_path: list[float] = []
        self.gamma_sig_path: list[float] = []
        self.eta_path: list[float] = []
        self.sigma_path: list[float] = []

        # initial record
        self._record_truth()

    # ------------------------- matrix builders -------------------------
    def F_mu(self) -> np.ndarray:
        """Transition for μ dynamic coordinates in the order of _state_layout_mu."""
        n = self.n_latent_mu
        if n == 0:
            return np.zeros((0, 0))
        F = np.eye(n)
        if "alpha_mu" in self._state_layout_mu:
            i = self._state_layout_mu.index("alpha_mu")
            if "beta_mu" in self._state_layout_mu:
                j = self._state_layout_mu.index("beta_mu")
                F[i, i] = 1.0; F[i, j] = 1.0
                F[j, i] = 0.0; F[j, j] = 1.0
        if self.seasonal_mode_mu == "dynamic":
            start = 0
            if "alpha_mu" in self._state_layout_mu: start += 1
            if "beta_mu"  in self._state_layout_mu: start += 1
            F[start:, start:] = _season_F(self.period)
        return F

    def A_mu(self) -> np.ndarray:
        """Observation design mapping μ-state -> contribution to μ_t."""
        if self.n_latent_mu == 0:
            return np.zeros((1, 0))
        A = np.zeros((1, self.n_latent_mu))
        if "alpha_mu" in self._state_layout_mu:
            A[0, self._state_layout_mu.index("alpha_mu")] = 1.0
        if self.seasonal_mode_mu == "dynamic":
            start = 0
            if "alpha_mu" in self._state_layout_mu: start += 1
            if "beta_mu"  in self._state_layout_mu: start += 1
            if (self.n_latent_mu - start) > 0:
                A[0, start] = 1.0
        return A

    def F_sigma(self) -> np.ndarray:
        """Transition for η dynamic coordinates in the order of _state_layout_sigma."""
        n = self.n_latent_sigma
        if n == 0:
            return np.zeros((0, 0))
        F = np.eye(n)
        if "alpha_sig" in self._state_layout_sigma:
            i = self._state_layout_sigma.index("alpha_sig")
            if "beta_sig" in self._state_layout_sigma:
                j = self._state_layout_sigma.index("beta_sig")
                F[i, i] = 1.0; F[i, j] = 1.0
                F[j, i] = 0.0; F[j, j] = 1.0
        if self.seasonal_mode_sigma == "dynamic":
            start = 0
            if "alpha_sig" in self._state_layout_sigma: start += 1
            if "beta_sig"  in self._state_layout_sigma: start += 1
            F[start:, start:] = _season_F(self.period)
        return F

    def A_sigma(self) -> np.ndarray:
        """Observation design mapping η-state -> contribution to η_t."""
        if self.n_latent_sigma == 0:
            return np.zeros((1, 0))
        A = np.zeros((1, self.n_latent_sigma))
        if "alpha_sig" in self._state_layout_sigma:
            A[0, self._state_layout_sigma.index("alpha_sig")] = 1.0
        if self.seasonal_mode_sigma == "dynamic":
            start = 0
            if "alpha_sig" in self._state_layout_sigma: start += 1
            if "beta_sig"  in self._state_layout_sigma: start += 1
            if (self.n_latent_sigma - start) > 0:
                A[0, start] = 1.0
        return A

    # ------------------ contributions & recording ------------------
    def _get_alpha(self, block: str) -> float | None:
        layout = self._state_layout_mu if block == "mu" else self._state_layout_sigma
        x = self.x_mu if block == "mu" else self.x_sig
        key = "alpha_mu" if block == "mu" else "alpha_sig"
        if key in layout:
            return float(x[layout.index(key)])
        return None

    def _get_beta(self, block: str) -> float | None:
        layout = self._state_layout_mu if block == "mu" else self._state_layout_sigma
        x = self.x_mu if block == "mu" else self.x_sig
        key = "beta_mu" if block == "mu" else "beta_sig"
        if key in layout:
            return float(x[layout.index(key)])
        return None

    def _get_season_vec(self, block: str) -> list[float]:
        if block == "mu":
            if self.seasonal_mode_mu != "dynamic": return []
            layout = self._state_layout_mu; x = self.x_mu
        else:
            if self.seasonal_mode_sigma != "dynamic": return []
            layout = self._state_layout_sigma; x = self.x_sig
        start = 0
        if ("alpha_mu" if block == "mu" else "alpha_sig") in layout: start += 1
        if ("beta_mu"  if block == "mu" else "beta_sig")  in layout: start += 1
        return list(x[start:])

    def _alpha_contribution_mu(self) -> float:
        t = self.t
        if self.level_mode_mu == "dynamic":
            a = self._get_alpha("mu")
            return 0.0 if a is None else float(a)
        base = float(self.fixed_level_mu)
        if self.trend_mode_mu == "dynamic":
            b = self._get_beta("mu")
            return base + (0.0 if b is None else float(b)) * t
        elif self.trend_mode_mu == "deterministic":
            return base + float(self.fixed_trend_mu) * t
        else:
            return base

    def _alpha_contribution_sigma(self) -> float:
        t = self.t
        if self.level_mode_sigma == "dynamic":
            a = self._get_alpha("sigma")
            return 0.0 if a is None else float(a)
        base = float(self.fixed_level_sigma)
        if self.trend_mode_sigma == "dynamic":
            b = self._get_beta("sigma")
            return base + (0.0 if b is None else float(b)) * t
        elif self.trend_mode_sigma == "deterministic":
            return base + float(self.fixed_trend_sigma) * t
        else:
            return base

    def _seasonal_contribution(self, block: str) -> float:
        if block == "mu":
            if self.seasonal_mode_mu == "dynamic":
                g = self._get_season_vec("mu"); return float(g[0]) if g else 0.0
            elif self.seasonal_mode_mu == "deterministic":
                return float(self.fixed_season_mu[self.t % self.period])
            else:
                return 0.0
        else:
            if self.seasonal_mode_sigma == "dynamic":
                g = self._get_season_vec("sigma"); return float(g[0]) if g else 0.0
            elif self.seasonal_mode_sigma == "deterministic":
                return float(self.fixed_season_sigma[self.t % self.period])
            else:
                return 0.0

    def _record_truth(self) -> None:
        # μ block
        alpha_mu_c = self._alpha_contribution_mu()
        beta_mu_v  = (self._get_beta("mu") if self.trend_mode_mu == "dynamic"
                      else (self.fixed_trend_mu if self.trend_mode_mu == "deterministic" else 0.0))
        gamma_mu_c = self._seasonal_contribution("mu")
        mu_t       = float(alpha_mu_c + gamma_mu_c)

        # η block
        alpha_sig_c = self._alpha_contribution_sigma()
        beta_sig_v  = (self._get_beta("sigma") if self.trend_mode_sigma == "dynamic"
                       else (self.fixed_trend_sigma if self.trend_mode_sigma == "deterministic" else 0.0))
        gamma_sig_c = self._seasonal_contribution("sigma")
        eta_t       = float(alpha_sig_c + gamma_sig_c)
        sigma_t     = float(np.exp(eta_t))

        # store
        self.alpha_mu_path.append(alpha_mu_c)
        self.beta_mu_path.append(beta_mu_v)
        self.gamma_mu_path.append(gamma_mu_c)
        self.mu_path.append(mu_t)

        self.alpha_sig_path.append(alpha_sig_c)
        self.beta_sig_path.append(beta_sig_v)
        self.gamma_sig_path.append(gamma_sig_c)
        self.eta_path.append(eta_t)
        self.sigma_path.append(sigma_t)

    # ----------------- one-step evolution -----------------
    def _evolve_block(self, block: str) -> None:
        """Evolve either 'mu' or 'sigma' one step."""
        if block == "mu":
            layout = self._state_layout_mu; x = self.x_mu
            q_level, q_trend, q_season = self.q_level, self.q_trend, self.q_season
            trend_mode, seasonal_mode, fixed_trend = self.trend_mode_mu, self.seasonal_mode_mu, self.fixed_trend_mu
        else:
            layout = self._state_layout_sigma; x = self.x_sig
            q_level, q_trend, q_season = self.q_level_sigma, self.q_trend_sigma, self.q_season_sigma
            trend_mode, seasonal_mode, fixed_trend = self.trend_mode_sigma, self.seasonal_mode_sigma, self.fixed_trend_sigma

        if len(layout) == 0:
            return

        new_x = np.array(x, copy=True)

        # level
        key_alpha = "alpha_mu" if block == "mu" else "alpha_sig"
        if key_alpha in layout:
            i = layout.index(key_alpha)
            drift = 0.0
            if trend_mode == "dynamic":
                b = self._get_beta(block); drift = 0.0 if b is None else float(b)
            elif trend_mode == "deterministic":
                drift = fixed_trend
            new_x[i] = x[i] + drift + np.random.normal(0.0, np.sqrt(q_level))

        # trend
        key_beta = "beta_mu" if block == "mu" else "beta_sig"
        if key_beta in layout:
            i = layout.index(key_beta)
            new_x[i] = x[i] + np.random.normal(0.0, np.sqrt(q_trend))

        # seasonal
        if seasonal_mode == "dynamic":
            start = 0
            if key_alpha in layout: start += 1
            if key_beta  in layout: start += 1
            g_prev = list(x[start:])
            mean_new = -float(np.sum(g_prev))
            g_new_first = np.random.normal(mean_new, np.sqrt(q_season))
            g_new_vec = [g_new_first] + g_prev[:-1] if len(g_prev) > 0 else [g_new_first]
            new_x[start:] = np.array(g_new_vec)

        if block == "mu":
            self.x_mu = new_x
        else:
            self.x_sig = new_x

    def move(self) -> None:
        """Advance t -> t+1, evolving both μ and η latent states (independently)."""
        self._evolve_block("mu")
        self._evolve_block("sigma")
        self.t += 1
        self._record_truth()

    # ----------------- measurement -----------------
    def measure(self) -> float:
        """Draw y_t ~ Normal( μ_t , exp(η_t)^2 ) using last recorded paths."""
        mu = self.mu_path[-1]
        sigma = self.sigma_path[-1]
        y = norm.rvs(loc=mu, scale=sigma)
        self.all_measurements.append(y)
        self.index.append(self._advance_and_get_time())
        return y

    # ----------------- calendar stepping -----------------
    def _advance_and_get_time(self) -> datetime:
        dt = self.current_date
        if self.period == 12:
            self.current_date += relativedelta(months=+1)
        elif self.period == 4:
            self.current_date += relativedelta(months=+3)
        else:
            self.current_date += relativedelta(years=+1)
        return dt

    # ----------------- public getters -----------------
    def get_truth_paths(self, as_numpy: bool = True) -> dict[str, list | np.ndarray]:
        to_arr = np.asarray if as_numpy else (lambda x: list(x))
        return {
            # μ block
            "alpha_mu_t": to_arr(self.alpha_mu_path),
            "beta_mu_t":  to_arr(self.beta_mu_path),
            "gamma_mu_t": to_arr(self.gamma_mu_path),
            "mu_t":       to_arr(self.mu_path),
            # η block
            "alpha_sig_t": to_arr(self.alpha_sig_path),
            "beta_sig_t":  to_arr(self.beta_sig_path),
            "gamma_sig_t": to_arr(self.gamma_sig_path),
            "eta_t":       to_arr(self.eta_path),
            "sigma_t":     to_arr(self.sigma_path),
            # time index
            "index": list(self.index),
        }


# ------------------------------------------------------------
# CLI: generate a single time series with full control
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import pandas as pd

    def _csv_floats(s: str | None) -> list[float] | None:
        if s is None or s.strip() == "":
            return None
        return [float(x) for x in s.split(",")]

    def _maybe_parse(s: str | None, period_minus_one: int, default_val: float) -> list[float]:
        out = _csv_floats(s)
        return out if out is not None else [default_val] * period_minus_one

    def _parse_date(s: str | None):
        if not s:
            return None
        parts = [int(p) for p in s.split("-")]
        if len(parts) == 1:  return datetime(parts[0], 1, 1)
        if len(parts) == 2:  return datetime(parts[0], parts[1], 1)
        if len(parts) == 3:  return datetime(parts[0], parts[1], parts[2])
        raise ValueError("start-date must be YYYY, YYYY-MM, or YYYY-MM-DD")

    p = argparse.ArgumentParser(
        description="Simulate a structural Gaussian time series with parallel models for mean and log-sigma."
    )

    # Core simulation controls
    p.add_argument("--T", type=int, default=200, help="Number of observations to generate.")
    p.add_argument("--period", type=int, default=4, help="Seasonal period (>=2). 12=monthly, 4=quarterly, else yearly steps.")
    p.add_argument("--start-date", type=str, default="2000-01-01", help="Start date (YYYY[-MM[-DD]]).")

    # μ modes
    p.add_argument("--level-mode", choices=["dynamic", "deterministic"], default="deterministic")
    p.add_argument("--trend-mode", choices=["dynamic", "deterministic", "none"], default="none")
    p.add_argument("--seasonal-mode", choices=["dynamic", "deterministic", "none"], default="none")

    # η modes (ln σ)
    p.add_argument("--level-mode-sigma", choices=["dynamic", "deterministic"], default="deterministic")
    p.add_argument("--trend-mode-sigma", choices=["dynamic", "deterministic", "none"], default="deterministic")
    p.add_argument("--seasonal-mode-sigma", choices=["dynamic", "deterministic", "none"], default="deterministic")

    # μ innovations
    p.add_argument("--q-level", type=float, default=0.05)
    p.add_argument("--q-trend", type=float, default=0.002)
    p.add_argument("--q-season", type=float, default=0.15)

    # η innovations (ln σ)
    p.add_argument("--q-level-sigma", type=float, default=0.05)
    p.add_argument("--q-trend-sigma", type=float, default=0.01)
    p.add_argument("--q-season-sigma", type=float, default=0.10)

    # μ priors / fixed
    p.add_argument("--m0-level", type=float, default=5.0)
    p.add_argument("--v0-level", type=float, default=0.25)
    p.add_argument("--m0-trend", type=float, default=0.015)
    p.add_argument("--v0-trend", type=float, default=0.05)
    p.add_argument("--m0-season", type=str, default="", help="Comma-separated length (period-1): (γ_{-(p-2)},...,γ_{-1},γ_0).")
    p.add_argument("--v0-season", type=str, default="", help="Comma-separated length (period-1) variances.")

    # η priors / fixed (for ln σ)
    p.add_argument("--m0-level-sigma", type=float, default=0.0, help="Prior mean of η level (0 => σ=1).")
    p.add_argument("--v0-level-sigma", type=float, default=0.0, help="Prior var of η level (0 => deterministic).")
    p.add_argument("--m0-trend-sigma", type=float, default=0.01)
    p.add_argument("--v0-trend-sigma", type=float, default=0.05)
    p.add_argument("--m0-season-sigma", type=str, default="", help="Comma-separated length (period-1) (oldest→newest).")
    p.add_argument("--v0-season-sigma", type=str, default="", help="Comma-separated length (period-1) variances.")

    # I/O & misc
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--plot", default=True, help="Show matplotlib figures.")
    p.add_argument("--save-csv", type=str, default="", help="Path to save CSV with all series.")
    p.add_argument("--print-summary", default=True, help="Print a small summary table at the end.")

    args = p.parse_args()
    np.random.seed(args.seed)

    p_minus_1 = max(0, args.period - 1)
    m0_season = _maybe_parse(args.m0_season, p_minus_1, 5.0)
    v0_season = _maybe_parse(args.v0_season, p_minus_1, 0.5 if args.seasonal_mode != "deterministic" else 0.0)

    m0_season_sigma = _maybe_parse(args.m0_season_sigma, p_minus_1, 0.0)
    v0_season_sigma = _maybe_parse(args.v0_season_sigma, p_minus_1, 0.5 if args.seasonal_mode_sigma != "deterministic" else 0.0)

    start_date = _parse_date(args.start_date)

    mts = Mean_Time_Series(
        # μ block
        level_mode=args.level_mode,
        trend_mode=args.trend_mode,
        seasonal_mode=args.seasonal_mode,
        # η block
        level_mode_sigma=args.level_mode_sigma,
        trend_mode_sigma=args.trend_mode_sigma,
        seasonal_mode_sigma=args.seasonal_mode_sigma,
        # period
        period=args.period,
        # μ innovations
        q_level=args.q_level,
        q_trend=args.q_trend,
        q_season=args.q_season,
        # η innovations
        q_level_sigma=args.q_level_sigma,
        q_trend_sigma=args.q_trend_sigma,
        q_season_sigma=args.q_season_sigma,
        # μ priors
        m0_level=args.m0_level,
        v0_level=args.v0_level,
        m0_trend=(0.0 if args.trend_mode == "none" else args.m0_trend),
        v0_trend=args.v0_trend,
        m0_season=m0_season,
        v0_season=v0_season,
        # η priors
        m0_level_sigma=args.m0_level_sigma,
        v0_level_sigma=args.v0_level_sigma,
        m0_trend_sigma=(0.0 if args.trend_mode_sigma == "none" else args.m0_trend_sigma),
        v0_trend_sigma=args.v0_trend_sigma,
        m0_season_sigma=m0_season_sigma,
        v0_season_sigma=v0_season_sigma,
        # time
        start_date=start_date,
    )

    # Simulate
    y = []
    for _ in range(args.T):
        mts.move()
        y.append(mts.measure())

    truths = mts.get_truth_paths(as_numpy=True)
    y_arr     = np.asarray(y, float)
    dates_T   = truths["index"][:args.T]

    mu_arr    = truths["mu_t"][1:1 + args.T]
    alpha_mu  = truths["alpha_mu_t"][1:1 + args.T]
    beta_mu   = truths["beta_mu_t"][1:1 + args.T]
    gamma_mu  = truths["gamma_mu_t"][1:1 + args.T]

    eta_arr   = truths["eta_t"][1:1 + args.T]
    sigma_arr = truths["sigma_t"][1:1 + args.T]
    alpha_sig = truths["alpha_sig_t"][1:1 + args.T]
    beta_sig  = truths["beta_sig_t"][1:1 + args.T]
    gamma_sig = truths["gamma_sig_t"][1:1 + args.T]

    # Pack into DataFrame
    import pandas as pd
    df = pd.DataFrame({
        "date": dates_T,
        "y_t": y_arr,
        # μ block
        "mu_t": mu_arr,
        "alpha_mu_t": alpha_mu,
        "beta_mu_t": beta_mu,
        "gamma_mu_t": gamma_mu,
        # η block
        "eta_t": eta_arr,
        "sigma_t": sigma_arr,
        "alpha_sig_t": alpha_sig,
        "beta_sig_t": beta_sig,
        "gamma_sig_t": gamma_sig,
    })

    # Save?
    if args.save_csv:
        df.to_csv(args.save_csv, index=False)
        print(f"Saved {len(df)} rows to {args.save_csv}")

    # Summary?
    if args.print_summary:
        with np.printoptions(suppress=True, precision=4):
            print("\n--- Summary ---")
            print(f"[μ]  level={args.level_mode}, trend={args.trend_mode}, season={args.seasonal_mode}")
            print(f"[η]  level={args.level_mode_sigma}, trend={args.trend_mode_sigma}, season={args.seasonal_mode_sigma}")
            print(f"[μ]  q_level={args.q_level}, q_trend={args.q_trend}, q_season={args.q_season}")
            print(f"[η]  q_level={args.q_level_sigma}, q_trend={args.q_trend_sigma}, q_season={args.q_season_sigma}")
            print(f"period={args.period}, start={dates_T[0]}, end={dates_T[-1]}")
            print(f"y mean={y_arr.mean():.3f}, sd={y_arr.std(ddof=1):.3f}")
            print(f"avg sigma={sigma_arr.mean():.3f}, sd sigma={sigma_arr.std(ddof=1):.3f}")

    # Plots?
    if args.plot:
        # Figure 1: y and μ
        plt.figure(figsize=(10, 4))
        plt.plot(dates_T, y_arr, label=r"$y_t$", linewidth=1.0)
        plt.plot(dates_T, mu_arr, "--", label=r"$\mu_t$", linewidth=1.0)
        ttl = (f"[μ] level={args.level_mode}, trend={args.trend_mode}, season={args.seasonal_mode} | "
               f"[η] level={args.level_mode_sigma}, trend={args.trend_mode_sigma}, season={args.seasonal_mode_sigma}")
        plt.title(ttl)
        plt.grid(True); plt.legend(); plt.tight_layout()

        # Figure 2: truth paths (μ)
        fig2, ax2 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        ax2[0].plot(dates_T, alpha_mu, linewidth=1.0); ax2[0].set_ylabel(r"$\alpha^{(\mu)}_t$"); ax2[0].grid(True)
        ax2[1].plot(dates_T, beta_mu,  linewidth=1.0); ax2[1].set_ylabel(r"$\beta^{(\mu)}_t$");  ax2[1].grid(True)
        ax2[2].plot(dates_T, gamma_mu, linewidth=1.0); ax2[2].set_ylabel(r"$\gamma^{(\mu)}_t$"); ax2[2].set_xlabel("time"); ax2[2].grid(True)
        fig2.suptitle("Truth paths — mean block")
        fig2.tight_layout()

        # Figure 3: truth paths (η)
        fig3, ax3 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        ax3[0].plot(dates_T, alpha_sig, linewidth=1.0); ax3[0].set_ylabel(r"$\alpha^{(\eta)}_t$"); ax3[0].grid(True)
        ax3[1].plot(dates_T, beta_sig,  linewidth=1.0); ax3[1].set_ylabel(r"$\beta^{(\eta)}_t$");  ax3[1].grid(True)
        ax3[2].plot(dates_T, gamma_sig, linewidth=1.0); ax3[2].set_ylabel(r"$\gamma^{(\eta)}_t$"); ax3[2].set_xlabel("time"); ax3[2].grid(True)
        fig3.suptitle("Truth paths — log-sigma block")
        fig3.tight_layout()

        # Figure 4: σ and η
        fig4, ax4 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax4[0].plot(dates_T, sigma_arr, linewidth=1.0); ax4[0].set_ylabel(r"$\sigma_t$"); ax4[0].grid(True)
        ax4[1].plot(dates_T, eta_arr,   linewidth=1.0); ax4[1].set_ylabel(r"$\eta_t=\ln\sigma_t$"); ax4[1].set_xlabel("time"); ax4[1].grid(True)
        fig4.suptitle("Scale paths")
        fig4.tight_layout()

        plt.show(block=True)
