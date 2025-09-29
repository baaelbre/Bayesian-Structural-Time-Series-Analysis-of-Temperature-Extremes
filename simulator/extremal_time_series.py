import numpy as np
from scipy.stats import genextreme
from datetime import datetime
from dateutil.relativedelta import relativedelta


class Extremal_Time_Series:
    """
    Structural DGEV simulator with independent switches for level, slope, and seasonality.

    Modes:
      level_mode      in {"dynamic","deterministic"}  (level cannot be 'none')
      trend_mode      in {"dynamic","deterministic","none"}
      seasonal_mode   in {"dynamic","deterministic","none"}

    Observation:
      y_t ~ GEV(mu_t, sigma, xi), with SciPy genextreme using shape c = -xi.

    Location decomposition:
      mu_t = (level + linear trend contribution) + (seasonal contribution)

    m0 / v0 semantics:
      - dynamic: m0_* is the prior mean of the initial latent state, v0_* its variance
      - deterministic: m0_* is the fixed value (vector for season)
      - none (trend/season only): behaves as m0_* = 0 (vector of zeros for season)

    Seasonality (STRICT):
      - m0_season and v0_season MUST be lists (or 1D arrays) of length (period-1).
      - deterministic: full seasonal vector is [m0_season, -sum(m0_season)] (sum-to-zero).
      - none: m0_season should be all zeros; implied last is 0 too.
      - dynamic: the (period-1) latent coords have priors m0_season/v0_season; each step
        the new last coord is drawn with mean = -sum(previous coords), var = q_season.

    Auto-overrides when a requested dynamic component cannot evolve:
      - If q_* == 0 or v0_* == 0:
          * if m0_* == 0  -> mode := "none"  (except level: becomes 'deterministic')
          * else          -> mode := "deterministic"
    """

    def __init__(self,
                 parameters=(1.0, 0.1),
                 level_mode="dynamic",
                 trend_mode="dynamic",
                 seasonal_mode="dynamic",
                 period=12,
                 # latent process innovation variances
                 q_level=0.05,
                 q_trend=0.01,
                 q_season=0.10,
                 # priors / fixed values
                 m0_level=0.0,
                 v0_level=1.0,
                 m0_trend=0.0,
                 v0_trend=1.0,
                 # SEASONAL PRIOR MUST BE LENGTH (period-1)
                 m0_season=None,   # list/array of length period-1
                 v0_season=None,   # list/array of length period-1
                 start_date=None):

        # ---------- basic checks on modes ----------
        assert level_mode in {"dynamic", "deterministic"}
        assert trend_mode in {"dynamic", "deterministic", "none"}
        assert seasonal_mode in {"dynamic", "deterministic", "none"}

        self.period = int(period)
        if self.period < 2:
            raise ValueError("period must be >= 2.")
        sigma, xi = parameters
        self.parameters = (float(sigma), float(xi))

        # ---------- store q/v0/m0 ----------
        self.q_level = float(q_level)
        self.q_trend = float(q_trend)
        self.q_season = float(q_season)

        self.v0_level = float(v0_level)
        self.v0_trend = float(v0_trend)

        self.m0_level = float(m0_level)
        self.m0_trend = float(m0_trend)

        # Season defaults -> lists length (period-1)
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

        # ---------- auto-overrides for dynamic components ----------
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

        # ---------- deterministic proxies from m0 ----------
        self.fixed_level = (self.m0_level if self.level_mode == "deterministic" else None)

        if self.trend_mode == "deterministic":
            self.fixed_trend = float(self.m0_trend)
        elif self.trend_mode == "none":
            self.fixed_trend = 0.0
        else:
            self.fixed_trend = None  # dynamic

        # For season deterministic: build full vector of length period with sum-zero
        if self.seasonal_mode == "deterministic":
            last = -float(np.sum(self.m0_season))
            self.fixed_season = list(self.m0_season.astype(float)) + [last]
        elif self.seasonal_mode == "none":
            # not used in mu, but keep a consistent implied vector (all zeros)
            self.fixed_season = [0.0] * self.period
        else:
            self.fixed_season = None  # dynamic

        # ---------- build latent layout [alpha][beta][g1..g_{p-1}] ----------
        self._state_layout = []
        if self.level_mode == "dynamic":
            self._state_layout.append("alpha")
        if self.trend_mode == "dynamic":
            self._state_layout.append("beta")
        if self.seasonal_mode == "dynamic":
            self._state_layout.extend([f"g{k}" for k in range(1, self.period)])

        self.n_latent = len(self._state_layout)

        # ---------- initialize latent state ----------
        if self.n_latent > 0:
            m0_vec, v0_vec = [], []
            for tag in self._state_layout:
                if tag == "alpha":
                    m0_vec.append(self.m0_level); v0_vec.append(self.v0_level)
                elif tag == "beta":
                    m0_vec.append(self.m0_trend); v0_vec.append(self.v0_trend)
                else:
                    # g1..g_{p-1}: use the provided lists m0_season / v0_season
                    k = int(tag[1:])  # k in 1..(p-1)
                    m0_vec.append(self.m0_season[k - 1])
                    v0_vec.append(self.v0_season[k - 1])
            self.x_t = np.random.normal(loc=np.array(m0_vec), scale=np.sqrt(np.array(v0_vec)))
        else:
            self.x_t = np.array([])

        # ---------- time bookkeeping & outputs ----------
        self.t = 0
        self.current_date = start_date if start_date else datetime.now()

        self.index = []
        self.all_measurements = []
        self.mu_path = []
        self.alpha_path = []
        self.beta_path = []
        self.gamma_last_path = []

        # initial record
        self._record_truth()

    # ----------------- accessors -----------------
    def _get_alpha_state(self):
        if "alpha" in self._state_layout:
            return float(self.x_t[self._state_layout.index("alpha")])
        return None

    def _get_beta_state(self):
        if "beta" in self._state_layout:
            return float(self.x_t[self._state_layout.index("beta")])
        return None

    def _get_season_vec(self):
        """Return the (period-1) seasonal latent vector if dynamic; else []."""
        if self.seasonal_mode != "dynamic":
            return []
        start = 0
        if "alpha" in self._state_layout: start += 1
        if "beta"  in self._state_layout: start += 1
        return list(self.x_t[start:])

    # ----------------- one-step evolution -----------------
    def move(self):
        """
        Advance t -> t+1.
          alpha_{t+1} = alpha_t + drift + N(0, q_level)
          beta_{t+1}  = beta_t  + N(0, q_trend)
          season: shift g[1..p-1], draw new last with mean = -sum(prev), var = q_season
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
                drift = self._get_beta_state()
            elif self.trend_mode == "deterministic":
                drift = self.fixed_trend
            i = self._state_layout.index("alpha")
            new_x[i] = self.x_t[i] + drift + np.random.normal(0.0, np.sqrt(self.q_level))

        # beta
        if "beta" in self._state_layout:
            i = self._state_layout.index("beta")
            new_x[i] = self.x_t[i] + np.random.normal(0.0, np.sqrt(self.q_trend))

        # season (period-1 vector)
        if self.seasonal_mode == "dynamic":
            start = 0
            if "alpha" in self._state_layout: start += 1
            if "beta"  in self._state_layout: start += 1
            g_prev = list(self.x_t[start:])  # length = period-1
            mean_new = -float(np.sum(g_prev))
            g_new_last = np.random.normal(mean_new, np.sqrt(self.q_season))
            g_new_vec = (g_prev[1:] + [g_new_last]) if len(g_prev) > 0 else [g_new_last]
            new_x[start:] = np.array(g_new_vec)

        self.x_t = new_x
        self.t += 1
        self._record_truth()

    # ----------------- measurement -----------------
    def measure(self):
        """Draw y_t ~ GEV(mu_t, sigma, xi) using last recorded mu_t."""
        sigma, xi = self.parameters
        mu = self.mu_path[-1]
        y = genextreme.rvs(-xi, loc=mu, scale=sigma)
        self.all_measurements.append(y)
        self.index.append(self._advance_and_get_time())
        return y

    # ----------------- contributions & recording -----------------
    def _alpha_contribution(self):
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

    def _beta_value_for_path(self):
        if self.trend_mode == "dynamic":
            b = self._get_beta_state()
            return 0.0 if b is None else float(b)
        elif self.trend_mode == "deterministic":
            return float(self.fixed_trend)
        else:
            return 0.0

    def _seasonal_contribution(self):
        if self.seasonal_mode == "dynamic":
            g_vec = self._get_season_vec()
            return float(g_vec[-1]) if len(g_vec) else 0.0
        elif self.seasonal_mode == "deterministic":
            return float(self.fixed_season[self.t % self.period])
        else:
            return 0.0

    def _record_truth(self):
        alpha_c = self._alpha_contribution()
        beta_v  = self._beta_value_for_path()
        gamma_c = self._seasonal_contribution()
        mu_t    = float(alpha_c + gamma_c)

        self.alpha_path.append(alpha_c)
        self.beta_path.append(beta_v)
        self.gamma_last_path.append(gamma_c)
        self.mu_path.append(mu_t)

    # ----------------- calendar stepping -----------------
    def _advance_and_get_time(self):
        dt = self.current_date
        if self.period == 12:
            self.current_date += relativedelta(months=+1)
        elif self.period == 4:
            self.current_date += relativedelta(months=+3)
        else:
            self.current_date += relativedelta(years=+1)
        return dt

    # ----------------- public getters -----------------
    def get_truth_paths(self, as_numpy=True):
        to_arr = np.asarray if as_numpy else (lambda x: list(x))
        return {
            "alpha": to_arr(self.alpha_path),
            "beta": to_arr(self.beta_path),
            "gamma_last": to_arr(self.gamma_last_path),
            "mu": to_arr(self.mu_path),
            "index": list(self.index),
        }

# -------------------------------
# Generate and save all 18 combos
# -------------------------------
if __name__ == "__main__":
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    np.random.seed(123)

    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    except NameError:
        base_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    outdir = os.path.join(base_dir, "simulated_extremal_series")
    os.makedirs(outdir, exist_ok=True)

    T = 240
    SIGMA_XI = (8.0, 0.1)
    period = 12

    # Full seasonal *shape* for convenience (length = period),
    # but we will pass only the first period-1 entries to m0_season.
    seas_vec_full = [
        np.cos(2 * np.pi * k / period) + 0.25 * np.cos(4 * np.pi * k / period)
        for k in range(period)
    ]
    m0_season_det = seas_vec_full[: period - 1]        # length = period-1
    v0_season_det = [0.0] * (period - 1)               # deterministic → zero variance
    m0_season_dyn = [0.0] * (period - 1)               # dynamic prior mean
    v0_season_dyn = [0.5] * (period - 1)               # dynamic prior variance
    m0_season_none = [0.0] * (period - 1)              # none → zeros still required
    v0_season_none = [1.0] * (period - 1)              # arbitrary (unused), keep valid

    level_grid    = ["dynamic", "deterministic"]               # 2
    trend_grid    = ["dynamic", "deterministic", "none"]       # 3
    seasonal_grid = ["dynamic", "deterministic", "none"]       # 3  -> 18 in total

    run_id = 0
    for lev in level_grid:
        for tr in trend_grid:
            for seas in seasonal_grid:
                run_id += 1

                # Seasonal priors per mode (must be length period-1)
                if seas == "deterministic":
                    m0_season = m0_season_det
                    v0_season = v0_season_det
                    q_season  = 0.15   # harmless; ignored by deterministic mode
                elif seas == "dynamic":
                    m0_season = m0_season_dyn
                    v0_season = v0_season_dyn
                    q_season  = 0.15
                else:  # seas == "none"
                    m0_season = m0_season_none
                    v0_season = v0_season_none
                    q_season  = 0.15   # harmless; ignored by 'none'

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
                    m0_trend=(0.012 if tr != "none" else 0.0),   # prior mean or fixed trend; 0 if none
                    v0_trend=0.05,
                    # NEW: enforce length period-1 for season
                    m0_season=m0_season,
                    v0_season=v0_season,
                    start_date=datetime(2000, 1, 1),
                )

                ets = Extremal_Time_Series(**kwargs)

                # Simulate exactly T observations
                y = []
                for _ in range(T):
                    ets.move()
                    y.append(ets.measure())

                truths = ets.get_truth_paths(as_numpy=False)
                mu_T    = np.array(truths["mu"][1:1 + T], dtype=float)
                alpha_T = np.array(truths["alpha"][1:1 + T], dtype=float) if 'alpha' in truths else np.array([0.0]*T)
                beta_T  = np.array(truths["beta"][1:1 + T], dtype=float) if 'beta' in truths  else np.array([0.0]*T)
                gamma_T = np.array(truths["gamma_last"][1:1 + T], dtype=float)
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
                    "gamma": gamma_T,
                })
                df.to_csv(csv_path, index=False)

                # Quick plot: y vs mu
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
                ax[2].plot(gamma_T); ax[2].set_title("gamma (seasonal contribution)")
                for a in ax: a.grid(True)
                plt.tight_layout()
                plt.savefig(png_comp_path, dpi=150)
                plt.close()

    print(f"Saved 18 simulated series (CSV + PNG + components) to: {outdir}")
