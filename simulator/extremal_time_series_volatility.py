# extremal_time_series_volatility.py
from __future__ import annotations

import os
from typing import List, Dict, Any, Optional
import numpy as np
from scipy.stats import genextreme
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

class Extremal_Time_Series:
    """
    Structural DGEV simulator with  switches for:
      - Location block (mu):     level/trend/season
      - Log-scale block (eta):   level/trend/season
    Shape (xi) is fixed.

    Observation:
      y_t ~ GEV( loc = mu_t, scale = exp(eta_t), shape = xi )
      NOTE: SciPy's genextreme uses c = -xi.

    Seasonality:
      - Latent seasonal vector has length (period-1): [g1, g2, ..., g_{p-1}] (g1 is observed).
      - Deterministic season builds a length-p vector with sum-to-zero closure.
      - Dynamic season update:
          g1_t  ~ N( -sum(g_{t-1}), q_season ),
          gk_t   = g_{k-1,t-1} for k=2..(p-1)
    """

    # ------------------------------ init ------------------------------
    def __init__(
        self,
        # ----- fixed shape -----
        xi: float = 0.1,

        # ----- MU block modes -----
        level_mode_mu: str = "dynamic",
        trend_mode_mu: str = "dynamic",
        seasonal_mode_mu: str = "dynamic",

        # ----- ETA (log-sigma) block modes -----
        level_mode_sigma: str = "deterministic", # this is the baseline
        trend_mode_sigma: str = "none",
        seasonal_mode_sigma: str = "none",

        # shared seasonal period
        period: int = 12,

        # ----- MU innovations -----
        q_level_mu: float = 0.05,
        q_trend_mu: float = 0.01,
        q_season_mu: float = 0.10,

        # ----- ETA innovations -----
        q_level_sigma: float = 0.01,
        q_trend_sigma: float = 0.01,
        q_season_sigma: float = 0.01,

        # ----- MU priors -----
        m0_level_mu: float = 0.0,
        v0_level_mu: float = 1.0,
        m0_trend_mu: float = 0.0,
        v0_trend_mu: float = 1.0,
        m0_season_mu: Optional[list[float]] = None,  # length p-1, oldest→newest
        v0_season_mu: Optional[list[float]] = None,

        # ----- ETA priors -----
        m0_level_sigma: float = 0.0,  # 0 => sigma=1 baseline
        v0_level_sigma: float = 0.0,  # 0 => deterministic unless dynamic + q>0
        m0_trend_sigma: float = 0.0,
        v0_trend_sigma: float = 1.0,
        m0_season_sigma: Optional[list[float]] = None,  # length p-1, oldest→newest
        v0_season_sigma: Optional[list[float]] = None,

        # time
        start_date: Optional[datetime] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        # ---- validate modes ----
        for m in (level_mode_mu, level_mode_sigma):
            if m not in {"dynamic", "deterministic"}:
                raise ValueError("level_mode must be 'dynamic' or 'deterministic'.")
        for m in (trend_mode_mu, trend_mode_sigma):
            if m not in {"dynamic", "deterministic", "none"}:
                raise ValueError("trend_mode must be 'dynamic', 'deterministic', or 'none'.")
        for m in (seasonal_mode_mu, seasonal_mode_sigma):
            if m not in {"dynamic", "deterministic", "none"}:
                raise ValueError("seasonal_mode must be 'dynamic', 'deterministic', or 'none'.")
        if period < 2:
            raise ValueError("period must be >= 2.")

        self.rng = rng if rng is not None else np.random.default_rng()
        self.period = int(period)
        self.xi = float(xi)  # fixed shape

        # =========== MU (location) PARAMETERS ===========
        self.q_level_mu = float(q_level_mu)
        self.q_trend_mu = float(q_trend_mu)
        self.q_season_mu = float(q_season_mu)

        self.v0_level_mu = float(v0_level_mu)
        self.v0_trend_mu = float(v0_trend_mu)
        self.m0_level_mu = float(m0_level_mu)
        self.m0_trend_mu = float(m0_trend_mu)

        if m0_season_mu is None:
            m0_season_mu = [0.0] * (self.period - 1)
        if v0_season_mu is None:
            v0_season_mu = [1.0] * (self.period - 1)
        m0_season_mu = np.asarray(m0_season_mu, float).reshape(-1)
        v0_season_mu = np.asarray(v0_season_mu, float).reshape(-1)
        if m0_season_mu.size != self.period - 1 or v0_season_mu.size != self.period - 1:
            raise ValueError("mu seasonal priors must have length period-1.")
        if np.any(v0_season_mu < 0.0) or self.v0_level_mu < 0.0 or self.v0_trend_mu < 0.0:
            raise ValueError("All mu prior variances must be >= 0.")
        # flip to newest-first storage
        self.m0_season_mu = m0_season_mu[::-1].copy()
        self.v0_season_mu = v0_season_mu[::-1].copy()

        # =========== ETA (log sigma) PARAMETERS ===========
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
            v0_season_sigma = [0.0 if seasonal_mode_sigma == "deterministic" else 0.5] * (self.period - 1)
        m0_season_sigma = np.asarray(m0_season_sigma, float).reshape(-1)
        v0_season_sigma = np.asarray(v0_season_sigma, float).reshape(-1)
        if m0_season_sigma.size != self.period - 1 or v0_season_sigma.size != self.period - 1:
            raise ValueError("sigma seasonal priors must have length period-1.")
        if np.any(v0_season_sigma < 0.0) or self.v0_level_sigma < 0.0 or self.v0_trend_sigma < 0.0:
            raise ValueError("All sigma prior variances must be >= 0.")
        # flip to newest-first
        self.m0_season_sigma = m0_season_sigma[::-1].copy()
        self.v0_season_sigma = v0_season_sigma[::-1].copy()

        # ---------- auto-overrides (mu) ----------
        if level_mode_mu == "dynamic" and (self.q_level_mu == 0.0 or self.v0_level_mu == 0.0):
            level_mode_mu = "deterministic"
        if trend_mode_mu == "dynamic" and (self.q_trend_mu == 0.0 or self.v0_trend_mu == 0.0):
            trend_mode_mu = "none" if np.isclose(self.m0_trend_mu, 0.0) else "deterministic"
        if seasonal_mode_mu == "dynamic" and (self.q_season_mu == 0.0 or np.allclose(self.v0_season_mu, 0.0)):
            seasonal_mode_mu = "none" if np.allclose(self.m0_season_mu, 0.0) else "deterministic"

        # ---------- auto-overrides (sigma) ----------
        if level_mode_sigma == "dynamic" and (self.q_level_sigma == 0.0 or self.v0_level_sigma == 0.0):
            level_mode_sigma = "deterministic"
        if trend_mode_sigma == "dynamic" and (self.q_trend_sigma == 0.0 or self.v0_trend_sigma == 0.0):
            trend_mode_sigma = "none" if np.isclose(self.m0_trend_sigma, 0.0) else "deterministic"
        if seasonal_mode_sigma == "dynamic" and (self.q_season_sigma == 0.0 or np.allclose(self.v0_season_sigma, 0.0)):
            seasonal_mode_sigma = "none" if np.allclose(self.m0_season_sigma, 0.0) else "deterministic"

        # finalize modes
        self.level_mode_mu = level_mode_mu
        self.trend_mode_mu = trend_mode_mu
        self.seasonal_mode_mu = seasonal_mode_mu

        self.level_mode_sigma = level_mode_sigma
        self.trend_mode_sigma = trend_mode_sigma
        self.seasonal_mode_sigma = seasonal_mode_sigma

        # ---------- deterministic proxies from priors ----------
        # mu
        self.fixed_level_mu = (self.m0_level_mu if self.level_mode_mu == "deterministic" else None)
        if self.trend_mode_mu == "deterministic":
            self.fixed_trend_mu = float(self.m0_trend_mu)
        elif self.trend_mode_mu == "none":
            self.fixed_trend_mu = 0.0
        else:
            self.fixed_trend_mu = None
        if self.seasonal_mode_mu == "deterministic":
            last = -float(np.sum(self.m0_season_mu[::-1]))  # use input order for sum-zero
            self.fixed_season_mu = list(self.m0_season_mu[::-1].astype(float)) + [last]
        elif self.seasonal_mode_mu == "none":
            self.fixed_season_mu = [0.0] * self.period
        else:
            self.fixed_season_mu = None

        # sigma (eta)
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

        # ---------- initialize latent state vectors ----------
        self.x_mu = np.array([], float)
        self.x_sig = np.array([], float)

        if len(self._state_layout_mu) > 0:
            m0_vec, v0_vec = [], []
            for tag in self._state_layout_mu:
                if tag == "alpha_mu":
                    m0_vec.append(self.m0_level_mu); v0_vec.append(self.v0_level_mu)
                elif tag == "beta_mu":
                    m0_vec.append(self.m0_trend_mu); v0_vec.append(self.v0_trend_mu)
                else:
                    k = int(tag[3:])
                    m0_vec.append(self.m0_season_mu[k - 1]); v0_vec.append(self.v0_season_mu[k - 1])
            self.x_mu = self.rng.normal(np.array(m0_vec), np.sqrt(np.array(v0_vec)))

        if len(self._state_layout_sigma) > 0:
            m0_vec, v0_vec = [], []
            for tag in self._state_layout_sigma:
                if tag == "alpha_sig":
                    m0_vec.append(self.m0_level_sigma); v0_vec.append(self.v0_level_sigma)
                elif tag == "beta_sig":
                    m0_vec.append(self.m0_trend_sigma); v0_vec.append(self.v0_trend_sigma)
                else:
                    k = int(tag[4:])
                    m0_vec.append(self.m0_season_sigma[k - 1]); v0_vec.append(self.v0_season_sigma[k - 1])
            self.x_sig = self.rng.normal(np.array(m0_vec), np.sqrt(np.array(v0_vec)))

        # ---------- time bookkeeping & outputs ----------
        self.t = 0
        self.current_date = start_date if start_date else datetime.now()

        self.index: List[datetime] = []
        self.y_path: List[float] = []

        # mu paths
        self.alpha_mu_path: List[float] = []
        self.beta_mu_path: List[float] = []
        self.gamma_mu_path: List[float] = []
        self.mu_path: List[float] = []

        # sigma (eta) paths
        self.alpha_sig_path: List[float] = []
        self.beta_sig_path: List[float] = []
        self.gamma_sig_path: List[float] = []
        self.eta_path: List[float] = []
        self.sigma_path: List[float] = []

        # initial record
        self._record_truth()

    # ---------------- helpers ----------------
    def _get_alpha(self, block: str) -> Optional[float]:
        layout = self._state_layout_mu if block == "mu" else self._state_layout_sigma
        x = self.x_mu if block == "mu" else self.x_sig
        key = "alpha_mu" if block == "mu" else "alpha_sig"
        if key in layout:
            return float(x[layout.index(key)])
        return None

    def _get_beta(self, block: str) -> Optional[float]:
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
            a = self._get_alpha("mu"); return 0.0 if a is None else float(a)
        base = float(self.fixed_level_mu)
        if self.trend_mode_mu == "dynamic":
            b = self._get_beta("mu"); return base + (0.0 if b is None else float(b)) * t
        elif self.trend_mode_mu == "deterministic":
            return base + float(self.fixed_trend_mu) * t
        else:
            return base

    def _alpha_contribution_sigma(self) -> float:
        t = self.t
        if self.level_mode_sigma == "dynamic":
            a = self._get_alpha("sigma"); return 0.0 if a is None else float(a)
        base = float(self.fixed_level_sigma)
        if self.trend_mode_sigma == "dynamic":
            b = self._get_beta("sigma"); return base + (0.0 if b is None else float(b)) * t
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
        # mu
        alpha_mu_c = self._alpha_contribution_mu()
        beta_mu_v  = (self._get_beta("mu") if self.trend_mode_mu == "dynamic"
                      else (self.fixed_trend_mu if self.trend_mode_mu == "deterministic" else 0.0))
        gamma_mu_c = self._seasonal_contribution("mu")
        mu_t       = float(alpha_mu_c + gamma_mu_c)

        # sigma (eta)
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

    # ---------------- evolution ----------------
    def _evolve_block(self, block: str) -> None:
        if block == "mu":
            layout, x = self._state_layout_mu, self.x_mu
            q_level, q_trend, q_season = self.q_level_mu, self.q_trend_mu, self.q_season_mu
            trend_mode, seasonal_mode, fixed_trend = self.trend_mode_mu, self.seasonal_mode_mu, self.fixed_trend_mu
            key_alpha, key_beta = "alpha_mu", "beta_mu"
        else:
            layout, x = self._state_layout_sigma, self.x_sig
            q_level, q_trend, q_season = self.q_level_sigma, self.q_trend_sigma, self.q_season_sigma
            trend_mode, seasonal_mode, fixed_trend = self.trend_mode_sigma, self.seasonal_mode_sigma, self.fixed_trend_sigma
            key_alpha, key_beta = "alpha_sig", "beta_sig"

        if len(layout) == 0:
            return

        new_x = np.array(x, copy=True)

        # level
        if key_alpha in layout:
            i = layout.index(key_alpha)
            drift = 0.0
            if trend_mode == "dynamic":
                b = self._get_beta("mu" if block == "mu" else "sigma")
                drift = 0.0 if b is None else float(b)
            elif trend_mode == "deterministic":
                drift = fixed_trend
            new_x[i] = x[i] + drift + self.rng.normal(0.0, np.sqrt(q_level))

        # trend
        if key_beta in layout:
            i = layout.index(key_beta)
            new_x[i] = x[i] + self.rng.normal(0.0, np.sqrt(q_trend))

        # seasonal
        if seasonal_mode == "dynamic":
            start = 0
            if key_alpha in layout: start += 1
            if key_beta  in layout: start += 1
            g_prev = list(x[start:])
            mean_new = -float(np.sum(g_prev))
            g_new_first = self.rng.normal(mean_new, np.sqrt(q_season))
            g_new_vec = [g_new_first] + (g_prev[:-1] if len(g_prev) > 0 else [])
            new_x[start:] = np.array(g_new_vec)

        if block == "mu":
            self.x_mu = new_x
        else:
            self.x_sig = new_x

    def move(self) -> None:
        self._evolve_block("mu")
        self._evolve_block("sigma")
        self.t += 1
        self._record_truth()

    # ---------------- measurement ----------------
    def measure(self) -> float:
        """Draw y_t ~ GEV(mu_t, sigma_t, xi). SciPy uses c=-xi."""
        mu = self.mu_path[-1]
        sigma = self.sigma_path[-1]
        y = genextreme.rvs(-self.xi, loc=mu, scale=sigma, random_state=self.rng)
        self.y_path.append(float(y))
        self.index.append(self._advance_and_get_time())
        return float(y)

    # ---------------- calendar stepping ----------------
    def _advance_and_get_time(self) -> datetime:
        dt = self.current_date
        if self.period == 12:
            self.current_date += relativedelta(months=+1)
        elif self.period == 4:
            self.current_date += relativedelta(months=+3)
        else:
            self.current_date += relativedelta(years=+1)
        return dt

    # ---------------- public getters ----------------
    def get_truth_paths(self, as_numpy: bool = True) -> Dict[str, Any]:
        to_arr = np.asarray if as_numpy else (lambda x: list(x))
        return {
            # mu block
            "alpha_mu": to_arr(self.alpha_mu_path),
            "beta_mu":  to_arr(self.beta_mu_path),
            "gamma_mu": to_arr(self.gamma_mu_path),
            "mu":       to_arr(self.mu_path),
            # sigma (eta) block
            "alpha_sig": to_arr(self.alpha_sig_path),
            "beta_sig":  to_arr(self.beta_sig_path),
            "gamma_sig": to_arr(self.gamma_sig_path),
            "eta":       to_arr(self.eta_path),
            "sigma":     to_arr(self.sigma_path),
            # index
            "index": list(self.index),
        }


# ---------------------------------------------------------
# Generate and save ALL 18 x 18 combinations of modes
# ---------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd

    rng = np.random.default_rng(123)
    outdir = os.path.join("results", "DGEV")
    os.makedirs(outdir, exist_ok=True)

    # Simulation horizon and calendar
    T = 240
    period = 12
    xi_fixed = 0.1

    # Convenience seasonal templates (length = period)
    seas_full = [np.cos(2 * np.pi * k / period) + 0.25 * np.cos(4 * np.pi * k / period) for k in range(period)]
    # Use first p-1 as (oldest->newest) for deterministic prior; we'll flip to newest-first internally
    m0_season_mu_det = seas_full[: period - 1]
    v0_season_mu_det = [0.0] * (period - 1)
    m0_season_mu_dyn = [0.0] * (period - 1)
    v0_season_mu_dyn = [0.5] * (period - 1)
    m0_season_mu_none = [0.0] * (period - 1)
    v0_season_mu_none = [1.0] * (period - 1)

    # For sigma (eta)
    m0_season_sig_det = [0.2 * v for v in seas_full[: period - 1]]
    v0_season_sig_det = [0.0] * (period - 1)
    m0_season_sig_dyn = [0.0] * (period - 1)
    v0_season_sig_dyn = [0.5] * (period - 1)
    m0_season_sig_none = [0.0] * (period - 1)
    v0_season_sig_none = [1.0] * (period - 1)

    level_grid    = ["dynamic", "deterministic"]         # 2
    trend_grid    = ["dynamic", "deterministic", "none"] # 3
    seasonal_grid = ["dynamic", "deterministic", "none"] # 3

    run_id = 0
    for lev_mu in level_grid:
        for tr_mu in trend_grid:
            for seas_mu in seasonal_grid:

                # MU priors/modes
                if seas_mu == "deterministic":
                    m0_season_mu = m0_season_mu_det; v0_season_mu = v0_season_mu_det; q_season_mu = 0.15
                elif seas_mu == "dynamic":
                    m0_season_mu = m0_season_mu_dyn; v0_season_mu = v0_season_mu_dyn; q_season_mu = 0.15
                else:
                    m0_season_mu = m0_season_mu_none; v0_season_mu = v0_season_mu_none; q_season_mu = 0.15

                for lev_sig in level_grid:
                    for tr_sig in trend_grid:
                        for seas_sig in seasonal_grid:
                            run_id += 1

                            # SIGMA priors/modes (for eta)
                            if seas_sig == "deterministic":
                                m0_season_sigma = m0_season_sig_det; v0_season_sigma = v0_season_sig_det; q_season_sigma = 0.10
                            elif seas_sig == "dynamic":
                                m0_season_sigma = m0_season_sig_dyn; v0_season_sigma = v0_season_sig_dyn; q_season_sigma = 0.10
                            else:
                                m0_season_sigma = m0_season_sig_none; v0_season_sigma = v0_season_sig_none; q_season_sigma = 0.10

                            ets = Extremal_Time_Series(
                                xi=xi_fixed,
                                # MU block
                                level_mode_mu=lev_mu,
                                trend_mode_mu=tr_mu,
                                seasonal_mode_mu=seas_mu,
                                # SIGMA block
                                level_mode_sigma=lev_sig,
                                trend_mode_sigma=tr_sig,
                                seasonal_mode_sigma=seas_sig,
                                # period
                                period=period,
                                # MU innovations
                                q_level_mu=0.05,
                                q_trend_mu=0.02,
                                q_season_mu=q_season_mu,
                                # SIGMA innovations (eta)
                                q_level_sigma=0.05,
                                q_trend_sigma=0.01,
                                q_season_sigma=q_season_sigma,
                                # MU priors
                                m0_level_mu=5.5,
                                v0_level_mu=0.25,
                                m0_trend_mu=(0.012 if tr_mu != "none" else 0.0),
                                v0_trend_mu=0.05,
                                m0_season_mu=m0_season_mu,
                                v0_season_mu=v0_season_mu,
                                # SIGMA priors (eta)
                                m0_level_sigma=0.0,          # eta baseline (sigma=1); override by modes/trend below
                                v0_level_sigma=(0.0 if lev_sig == "deterministic" else 0.25),
                                m0_trend_sigma=(0.004 if tr_sig != "none" else 0.0),
                                v0_trend_sigma=0.05,
                                m0_season_sigma=m0_season_sigma,
                                v0_season_sigma=v0_season_sigma,
                                # time
                                start_date=datetime(2000, 1, 1),
                                rng=rng,
                            )

                            # Simulate
                            y_vals = []
                            for _ in range(T):
                                ets.move()
                                y_vals.append(ets.measure())

                            truths = ets.get_truth_paths(as_numpy=True)
                            dates_T    = truths["index"][:T]
                            mu_T       = truths["mu"][1:1 + T]
                            alpha_mu_T = truths["alpha_mu"][1:1 + T]
                            beta_mu_T  = truths["beta_mu"][1:1 + T]
                            gamma_mu_T = truths["gamma_mu"][1:1 + T]

                            eta_T      = truths["eta"][1:1 + T]
                            sigma_T    = truths["sigma"][1:1 + T]
                            alpha_s_T  = truths["alpha_sig"][1:1 + T]
                            beta_s_T   = truths["beta_sig"][1:1 + T]
                            gamma_s_T  = truths["gamma_sig"][1:1 + T]

                            y_T        = np.asarray(y_vals, float)

                            tag = (
                                f"mu(L={lev_mu},T={tr_mu},S={seas_mu})__"
                                f"sig(L={lev_sig},T={tr_sig},S={seas_sig})__id{run_id:03d}"
                            )
                            csv_path = os.path.join(outdir, f"{tag}.csv")
                            png_main = os.path.join(outdir, f"{tag}_main.png")
                            png_comp_mu = os.path.join(outdir, f"{tag}_mu_components.png")
                            png_comp_sig = os.path.join(outdir, f"{tag}_sigma_components.png")

                            df = pd.DataFrame({
                                "date": [d.isoformat() for d in dates_T],
                                "y": y_T,
                                # mu block
                                "mu": mu_T,
                                "alpha_mu": alpha_mu_T,
                                "beta_mu": beta_mu_T,
                                "gamma_mu": gamma_mu_T,
                                # sigma (eta) block
                                "sigma": sigma_T,
                                "eta": eta_T,
                                "alpha_sig": alpha_s_T,
                                "beta_sig": beta_s_T,
                                "gamma_sig": gamma_s_T,
                            })
                            df.to_csv(csv_path, index=False)

                            # Main plot: y vs mu, and sigma
                            plt.figure(figsize=(11, 4))
                            plt.plot(y_T, label="y_t", linewidth=1.0, alpha=0.8)
                            plt.plot(mu_T, "--", label="mu_t", linewidth=1.0)
                            plt.title(f"GEV series — {tag} (xi={xi_fixed})")
                            plt.xlabel("t"); plt.ylabel("value"); plt.grid(True); plt.legend()
                            plt.tight_layout(); plt.savefig(png_main, dpi=150); plt.close()

                            # Components (mu)
                            fig, ax = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
                            ax[0].plot(alpha_mu_T); ax[0].set_ylabel("alpha_mu"); ax[0].grid(True)
                            ax[1].plot(beta_mu_T);  ax[1].set_ylabel("beta_mu");  ax[1].grid(True)
                            ax[2].plot(gamma_mu_T); ax[2].set_ylabel("gamma_mu"); ax[2].set_xlabel("t"); ax[2].grid(True)
                            fig.suptitle("Location block components")
                            fig.tight_layout(); fig.savefig(png_comp_mu, dpi=150); plt.close(fig)

                            # Components (sigma / eta)
                            fig2, ax2 = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
                            ax2[0].plot(alpha_s_T); ax2[0].set_ylabel("alpha_sig"); ax2[0].grid(True)
                            ax2[1].plot(beta_s_T);  ax2[1].set_ylabel("beta_sig");  ax2[1].grid(True)
                            ax2[2].plot(gamma_s_T); ax2[2].set_ylabel("gamma_sig"); ax2[2].set_xlabel("t"); ax2[2].grid(True)
                            fig2.suptitle("Log-scale block components (eta)")
                            fig2.tight_layout(); fig2.savefig(png_comp_sig, dpi=150); plt.close(fig2)

    print(f"Saved all simulated DGEV series to: {outdir}")
