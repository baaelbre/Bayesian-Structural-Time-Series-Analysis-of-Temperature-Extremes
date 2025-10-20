import numpy as np
from typing import List, Dict, Any, Optional, Union
from scipy.stats import genextreme
from datetime import datetime
from dateutil.relativedelta import relativedelta


ArrayLike = Union[List[float], np.ndarray]


class Extremal_Time_Series:
    """
    Structural DGEV simulator with independent switches for level, slope, and seasonality.

    Modes
    -----
      level_mode      ∈ {"dynamic","deterministic"}        (level cannot be 'none')
      trend_mode      ∈ {"dynamic","deterministic","none"}
      seasonal_mode   ∈ {"dynamic","deterministic","none"}

    Observation
    -----------
      y_t ~ GEV(mu_t, σ, ξ), using SciPy's genextreme with shape c = -ξ.

    Location decomposition
    ----------------------
      mu_t = (level + linear trend contribution) + (seasonal contribution)

    Parameter semantics
    -------------------
    m0 / v0:
      - dynamic: m0_* is the prior mean of the initial latent state; v0_* its variance
      - deterministic: m0_* is the fixed value (vector for season)
      - none (trend/season only): behaves as m0_* = 0 (vector of zeros for season)

    Seasonality (STRICT; NEWEST-FIRST)
    ----------------------------------
      - Seasonal latent state has length (period-1) with NEWEST-FIRST ordering:
            g_t = [g1_t, g2_t, ..., g_{p-1,t}],  where g1_t is the coord used in the observation.
      - deterministic: full seasonal vector is [m0_season, -sum(m0_season)] (sum-to-zero).
      - none: m0_season should be zeros; implied last is 0 too.
      - dynamic:
            * State length is (period-1).
            * Transition is the “sum-to-zero with newest-first closure”:
                g1_t  ~ N( -sum(g_{t-1}), q_season )
                gk_t  = g_{k-1,t-1}   for k=2..(p-1)
            * Observation loads g1_t.

    Auto-overrides when a requested dynamic component cannot evolve
    ---------------------------------------------------------------
      If q_* == 0 or v0_* == 0:
        * if m0_* == 0  -> mode := "none"  (except level: becomes 'deterministic')
        * else          -> mode := "deterministic"
    """

    def __init__(
        self,
        parameters: tuple[float, float] = (1.0, 0.1),
        level_mode: str = "dynamic",
        trend_mode: str = "dynamic",
        seasonal_mode: str = "dynamic",
        period: int = 12,
        # latent process innovation variances
        q_level: float = 0.05,
        q_trend: float = 0.01,
        q_season: float = 0.10,
        # priors / fixed values
        m0_level: float = 0.0,
        v0_level: float = 1.0,
        m0_trend: float = 0.0,
        v0_trend: float = 1.0,
        # SEASONAL PRIOR MUST BE LENGTH (period-1)
        m0_season: Optional[ArrayLike] = None,   # list/array of length period-1 (NEWEST-FIRST)
        v0_season: Optional[ArrayLike] = None,   # list/array of length period-1
        start_date: Optional[datetime] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        # ---------------- Basic checks ----------------
        if level_mode not in {"dynamic", "deterministic"}:
            raise ValueError("level_mode must be 'dynamic' or 'deterministic'.")
        if trend_mode not in {"dynamic", "deterministic", "none"}:
            raise ValueError("trend_mode must be 'dynamic', 'deterministic', or 'none'.")
        if seasonal_mode not in {"dynamic", "deterministic", "none"}:
            raise ValueError("seasonal_mode must be 'dynamic', 'deterministic', or 'none'.")
        if period < 2:
            raise ValueError("period must be >= 2.")

        self.rng = rng if rng is not None else np.random.default_rng()
        self.period = int(period)

        sigma, xi = parameters
        self.parameters = (float(sigma), float(xi))

        # ------------- store q/v0/m0 -------------
        self.q_level = float(q_level)
        self.q_trend = float(q_trend)
        self.q_season = float(q_season)

        self.v0_level = float(v0_level)
        self.v0_trend = float(v0_trend)

        self.m0_level = float(m0_level)
        self.m0_trend = float(m0_trend)

        # Seasonal defaults (length = period-1)
        if m0_season is None:
            m0_season = [0.0] * (self.period - 1)
        if v0_season is None:
            v0_season = [1.0] * (self.period - 1)
        self.m0_season = np.asarray(m0_season, dtype=float).reshape(-1)
        self.v0_season = np.asarray(v0_season, dtype=float).reshape(-1)

        if self.m0_season.size != self.period - 1:
            raise ValueError(f"m0_season must have length period-1 = {self.period-1}.")
        if self.v0_season.size != self.period - 1:
            raise ValueError(f"v0_season must have length period-1 = {self.period-1}.")
        if np.any(self.v0_season < 0.0) or self.v0_level < 0.0 or self.v0_trend < 0.0:
            raise ValueError("All prior variances must be >= 0.")

        # ------------- auto-overrides for dynamic components -------------
        # Level: cannot be 'none'
        if level_mode == "dynamic" and (self.q_level == 0.0 or self.v0_level == 0.0):
            level_mode = "deterministic"

        # Trend
        if trend_mode == "dynamic" and (self.q_trend == 0.0 or self.v0_trend == 0.0):
            trend_mode = "none" if np.isclose(self.m0_trend, 0.0) else "deterministic"

        # Season
        if seasonal_mode == "dynamic" and (self.q_season == 0.0 or np.allclose(self.v0_season, 0.0)):
            seasonal_mode = "none" if np.allclose(self.m0_season, 0.0) else "deterministic"

        # Finalize modes
        self.level_mode = level_mode
        self.trend_mode = trend_mode
        self.seasonal_mode = seasonal_mode

        # ------------- deterministic proxies from m0 -------------
        self.fixed_level: Optional[float] = self.m0_level if self.level_mode == "deterministic" else None
        if self.trend_mode == "deterministic":
            self.fixed_trend = float(self.m0_trend)
        elif self.trend_mode == "none":
            self.fixed_trend = 0.0
        else:
            self.fixed_trend = None  # dynamic trend

        # For season deterministic: build full vector of length period with sum-zero
        if self.seasonal_mode == "deterministic":
            last = -float(np.sum(self.m0_season))
            self.fixed_season = list(self.m0_season.astype(float)) + [last]
        elif self.seasonal_mode == "none":
            self.fixed_season = [0.0] * self.period
        else:
            self.fixed_season = None  # dynamic

        # ------------- latent layout [alpha][beta][g1..g_{p-1}] -------------
        self._state_layout: List[str] = []
        if self.level_mode == "dynamic":
            self._state_layout.append("alpha")
        if self.trend_mode == "dynamic":
            self._state_layout.append("beta")
        if self.seasonal_mode == "dynamic":
            # NEWEST-FIRST seasonal coordinates
            self._state_layout.extend([f"g{k}" for k in range(1, self.period)])
        self.n_latent = len(self._state_layout)

        # ------------- initialize latent state -------------
        if self.n_latent > 0:
            m0_vec, v0_vec = [], []
            for tag in self._state_layout:
                if tag == "alpha":
                    m0_vec.append(self.m0_level); v0_vec.append(self.v0_level)
                elif tag == "beta":
                    m0_vec.append(self.m0_trend); v0_vec.append(self.v0_trend)
                else:
                    k = int(tag[1:])  # k in 1..(p-1)
                    m0_vec.append(self.m0_season[k - 1])
                    v0_vec.append(self.v0_season[k - 1])
            self.x_t = self.rng.normal(loc=np.array(m0_vec), scale=np.sqrt(np.array(v0_vec)))
        else:
            self.x_t = np.array([])

        # ------------- time bookkeeping & outputs -------------
        self.t = 0
        self.current_date = start_date if start_date else datetime.now()

        self.index: List[datetime] = []
        self.all_measurements: List[float] = []
        self.mu_path: List[float] = []
        self.alpha_path: List[float] = []
        self.beta_path: List[float] = []
        self.gamma_path: List[float] = []   # observed seasonal coord (g1; NEWEST-FIRST)

        # initial record
        self._record_truth()

    # ---------------- accessors ----------------
    def _get_alpha_state(self) -> Optional[float]:
        if "alpha" in self._state_layout:
            return float(self.x_t[self._state_layout.index("alpha")])
        return None

    def _get_beta_state(self) -> Optional[float]:
        if "beta" in self._state_layout:
            return float(self.x_t[self._state_layout.index("beta")])
        return None

    def _get_season_vec(self) -> List[float]:
        """Return the (period-1) seasonal latent vector (NEWEST-FIRST) if dynamic; else []."""
        if self.seasonal_mode != "dynamic":
            return []
        start = 0
        if "alpha" in self._state_layout: start += 1
        if "beta"  in self._state_layout: start += 1
        return list(self.x_t[start:])  # [g1, g2, ..., g_{p-1}] NEWEST-FIRST

    # ---------------- one-step evolution ----------------
    def move(self) -> None:
        """
        Advance t -> t+1.
          alpha_{t+1} = alpha_t + drift + N(0, q_level)
          beta_{t+1}  = beta_t  + N(0, q_trend)
          season (NEWEST-FIRST):
              g1(t)  ~ N( -sum(g(t-1)), q_season )
              gk(t)  = g_{k-1}(t-1),  k=2..(p-1)
        """
        if self.n_latent == 0:
            self.t += 1
            self._record_truth()
            return

        new_x = np.array(self.x_t, copy=True)

        # alpha
        if "alpha" in self._state_layout:
            drift = 0.0
            if self.trend_mode == "dynamic":
                drift = self._get_beta_state() or 0.0
            elif self.trend_mode == "deterministic":
                drift = float(self.fixed_trend)
            i = self._state_layout.index("alpha")
            new_x[i] = self.x_t[i] + drift + self.rng.normal(0.0, np.sqrt(self.q_level))

        # beta
        if "beta" in self._state_layout:
            i = self._state_layout.index("beta")
            new_x[i] = self.x_t[i] + self.rng.normal(0.0, np.sqrt(self.q_trend))

        # season (period-1 vector, NEWEST-FIRST)
        if self.seasonal_mode == "dynamic":
            start = 0
            if "alpha" in self._state_layout: start += 1
            if "beta"  in self._state_layout: start += 1

            g_prev = list(self.x_t[start:])  # [g1, g2, ..., g_{p-1}] at t-1
            mean_new_first = -float(np.sum(g_prev))
            g1_new = self.rng.normal(mean_new_first, np.sqrt(self.q_season))
            g_new_vec = [g1_new] + (g_prev[:-1] if len(g_prev) > 0 else [])
            new_x[start:] = np.array(g_new_vec)

        self.x_t = new_x
        self.t += 1
        self._record_truth()

    # ---------------- measurement ----------------
    def measure(self) -> float:
        """Draw y_t ~ GEV(mu_t, sigma, xi) using last recorded mu_t."""
        sigma, xi = self.parameters
        mu = self.mu_path[-1]
        # SciPy genextreme uses c = -xi
        y = genextreme.rvs(-xi, loc=mu, scale=sigma, random_state=self.rng)
        self.all_measurements.append(float(y))
        self.index.append(self._advance_and_get_time())
        return float(y)

    # ---------------- contributions & recording ----------------
    def _alpha_contribution(self) -> float:
        """Level + linear trend contribution to mu_t."""
        t = self.t
        if self.level_mode == "dynamic":
            a = self._get_alpha_state()
            return 0.0 if a is None else float(a)

        base = float(self.fixed_level)
        if self.trend_mode == "dynamic":
            b = self._get_beta_state()
            return base + (0.0 if b is None else float(b)) * t
        elif self.trend_mode == "deterministic":
            return base + float(self.fixed_trend) * t
        else:
            return base

    def _beta_value_for_path(self) -> float:
        if self.trend_mode == "dynamic":
            b = self._get_beta_state()
            return 0.0 if b is None else float(b)
        elif self.trend_mode == "deterministic":
            return float(self.fixed_trend)
        else:
            return 0.0

    def _seasonal_contribution(self) -> float:
        if self.seasonal_mode == "dynamic":
            g_vec = self._get_season_vec()   # NEWEST-FIRST
            return float(g_vec[0]) if len(g_vec) else 0.0  # use g1_t
        elif self.seasonal_mode == "deterministic":
            return float(self.fixed_season[self.t % self.period])
        else:
            return 0.0

    def _record_truth(self) -> None:
        alpha_c = self._alpha_contribution()
        beta_v  = self._beta_value_for_path()
        gamma_c = self._seasonal_contribution()   # observed seasonal coord (g1_t if dynamic)
        mu_t    = float(alpha_c + gamma_c)

        self.alpha_path.append(alpha_c)
        self.beta_path.append(beta_v)
        self.gamma_path.append(gamma_c)  # observed coord
        self.mu_path.append(mu_t)

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
        """
        Returns:
            alpha:   path of alpha contribution to mu (float array)
            beta:    path of beta value (float array)
            gamma:   path of observed seasonal coord (g1_t if dynamic; full cycle value if deterministic)
            mu:      path of mu_t
            index:   list of datetimes (timestamps for each measurement)
        Note: For backward compatibility with older code, 'gamma' here is the
              coord that actually enters the observation.
        """
        to_arr = np.asarray if as_numpy else (lambda x: list(x))
        return {
            "alpha": to_arr(self.alpha_path),
            "beta":  to_arr(self.beta_path),
            "gamma": to_arr(self.gamma_path),
            "mu":    to_arr(self.mu_path),
            "index": list(self.index),
        }


# -------------------------------
# Generate and save all 18 combos
# -------------------------------
if __name__ == "__main__":
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    # Reproducibility
    rng = np.random.default_rng(123)

    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    except NameError:
        base_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    outdir = os.path.join(base_dir, "simulated_extremal_series")
    os.makedirs(outdir, exist_ok=True)

    T = 240
    SIGMA_XI = (8.0, 0.1)
    period = 12

    # Full seasonal shape for convenience (length = period),
    # but pass only the first (period-1) entries to m0_season (NEWEST-FIRST).
    seas_vec_full = [
        np.cos(2 * np.pi * k / period) + 0.25 * np.cos(4 * np.pi * k / period)
        for k in range(period)
    ]
    # Interpret first p-1 entries as NEWEST-FIRST initial coords (g1..g_{p-1} at t=0).
    m0_season_det = seas_vec_full[: period - 1]
    v0_season_det = [0.0] * (period - 1)  # deterministic → zero variance
    m0_season_dyn = [0.0] * (period - 1)  # dynamic prior mean
    v0_season_dyn = [0.5] * (period - 1)  # dynamic prior variance
    m0_season_none = [0.0] * (period - 1)  # none → zeros still required
    v0_season_none = [1.0] * (period - 1)  # arbitrary (unused), keep valid

    level_grid    = ["dynamic", "deterministic"]               # 2
    trend_grid    = ["dynamic", "deterministic", "none"]       # 3
    seasonal_grid = ["dynamic", "deterministic", "none"]       # 3  -> 18

    run_id = 0
    for lev in level_grid:
        for tr in trend_grid:
            for seas in seasonal_grid:
                run_id += 1

                # Seasonal priors per mode (length = period-1)
                if seas == "deterministic":
                    m0_season = m0_season_det
                    v0_season = v0_season_det
                    q_season  = 0.15   # ignored by deterministic mode
                elif seas == "dynamic":
                    m0_season = m0_season_dyn
                    v0_season = v0_season_dyn
                    q_season  = 0.15
                else:  # seas == "none"
                    m0_season = m0_season_none
                    v0_season = v0_season_none
                    q_season  = 0.15   # ignored by 'none'

                kwargs = dict(
                    parameters=SIGMA_XI,
                    level_mode=lev,
                    trend_mode=tr,
                    seasonal_mode=seas,
                    period=period,
                    # Stochastic variances (only matter when component is latent)
                    q_level=0.05,
                    q_trend=0.02,
                    q_season=q_season,
                    # Priors / fixed values via m0_*
                    m0_level=5.5,                     # prior mean for alpha_0 or fixed level
                    v0_level=0.25,
                    m0_trend=(0.012 if tr != "none" else 0.0),  # prior mean or fixed trend; 0 if none
                    v0_trend=0.05,
                    m0_season=m0_season,
                    v0_season=v0_season,
                    start_date=datetime(2000, 1, 1),
                    rng=rng,
                )

                ets = Extremal_Time_Series(**kwargs)

                # Simulate exactly T observations
                y = []
                for _ in range(T):
                    ets.move()
                    y.append(ets.measure())

                truths = ets.get_truth_paths(as_numpy=False)
                # Shift by one to align with measurements (t indexes after first move/measure)
                mu_T    = np.array(truths["mu"][1:1 + T], dtype=float)
                alpha_T = np.array(truths["alpha"][1:1 + T], dtype=float) if 'alpha' in truths else np.zeros(T)
                beta_T  = np.array(truths["beta"][1:1 + T], dtype=float)  if 'beta'  in truths else np.zeros(T)
                gamma_T = np.array(truths["gamma"][1:1 + T], dtype=float)
                dates_T = truths["index"][:T]
                y_T     = np.array(y, dtype=float)

                tag = f"case_{run_id:02d}_level-{lev}_trend-{tr}_season-{seas}"
                csv_path = os.path.join(outdir, f"{tag}.csv")
                png_path = os.path.join(outdir, f"{tag}.png")

                df = pd.DataFrame({
                    "date": [d.isoformat() for d in dates_T],
                    "y": y_T,
                    "mu": mu_T,
                    "alpha": alpha_T,
                    "beta": beta_T,
                    "gamma": gamma_T,   # observed seasonal coord (g1_t if dynamic)
                })
                df.to_csv(csv_path, index=False)

                # Quick plot: y vs mu
                import matplotlib.pyplot as plt
                plt.figure(figsize=(11, 3.8))
                plt.plot(y_T, label="y_t", alpha=0.75)
                plt.plot(mu_T, "--", label="mu_t")
                plt.title(f"{tag}")
                plt.xlabel("t")
                plt.ylabel("value")
                plt.legend()
                plt.tight_layout()
                plt.savefig(png_path, dpi=150)
                plt.close()

                # Component plots
                png_comp_path = os.path.join(outdir, f"{tag}_components.png")
                fig, ax = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
                ax[0].plot(alpha_T); ax[0].set_title("alpha (intercept + trend contribution)")
                ax[1].plot(beta_T);  ax[1].set_title("beta (slope)")
                ax[2].plot(gamma_T); ax[2].set_title("gamma (seasonal contribution; observed coord)")
                for a in ax: a.grid(True)
                plt.tight_layout()
                plt.savefig(png_comp_path, dpi=150)
                plt.close()

    print(f"Saved 18 simulated series (CSV + PNG + components) to: {outdir}")
