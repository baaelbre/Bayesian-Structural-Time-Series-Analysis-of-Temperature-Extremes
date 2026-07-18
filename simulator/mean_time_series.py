import numpy as np
from scipy.stats import norm
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt


class Mean_Time_Series:
    """
    Structural Gaussian STM simulator with independent switches for level, trend, and seasonality.

    Modes
    -----
      level_mode      in {"dynamic","deterministic"}         (level cannot be 'none')
      trend_mode      in {"dynamic","deterministic","none"}
      seasonal_mode   in {"dynamic","deterministic","none"}

    Observation
    -----------
      y_t ~ Normal(mu_t, sigma^2)

    Location decomposition
    ----------------------
      mu_t = (level + linear trend contribution) + (seasonal contribution)

    m0 / v0 semantics
    -----------------
      - dynamic: m0_* is prior mean of initial latent state coordinate(s), v0_* the prior variance
      - deterministic: m0_* is the fixed value (vector for season); v0_* ignored
      - none (trend/season only): behaves as m0_* = 0 (vector of zeros for season)

    Seasonal convention (LaTeX-consistent)
    --------------------------------------
      State stores the (period-1) seasonal coordinates in *newest-first* order:
        [γ_t, γ_{t-1}, ..., γ_{t-(p-2)}].

      Evolution (for dynamic seasonality):
        γ_t = -∑_{j=1}^{p-1} γ_{t-j} + ε_{γ,t},  ε_{γ,t} ~ N(0, q_season)
        Then the vector shifts right: [γ_t, γ_{t-1}, ..., γ_{t-(p-2)}].

      Observation uses the *first* seasonal coord (the current γ_t).

      INPUT ordering for m0_season is *oldest→newest*:
        m0_season = (γ_{-(p-2)}, ..., γ_{-1}, γ_0).
      This is flipped internally to newest-first storage.
    """

    def __init__(self,
                 sigma=1.0,
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
                 m0_season=None,   # list/array length (period-1): (γ_{-(p-2)},...,γ_{-1},γ_0)
                 v0_season=None,   # list/array length (period-1) matching m0_season
                 start_date=None):

        # ---------- basic checks on modes ----------
        assert level_mode in {"dynamic", "deterministic"}
        assert trend_mode in {"dynamic", "deterministic", "none"}
        assert seasonal_mode in {"dynamic", "deterministic", "none"}

        self.period = int(period)
        if self.period < 2:
            raise ValueError("period must be >= 2.")
        self.sigma = float(sigma)

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

        m0_season = np.asarray(m0_season, dtype=float).reshape(-1)
        v0_season = np.asarray(v0_season, dtype=float).reshape(-1)

        if m0_season.size != self.period - 1:
            raise ValueError(f"m0_season must have length period-1 = {self.period-1}.")
        if v0_season.size != self.period - 1:
            raise ValueError(f"v0_season must have length period-1 = {self.period-1}.")
        if np.any(v0_season < 0.0) or self.v0_level < 0.0 or self.v0_trend < 0.0:
            raise ValueError("All prior variances must be >= 0.")

        # Flip to newest-first internal storage: [γ_0, γ_{-1}, ..., γ_{-(p-2)}]
        self.m0_season = m0_season[::-1].copy()
        self.v0_season = v0_season[::-1].copy()

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
            # Use input order (oldest->newest) for printing; full vector (period) must sum to zero
            last = -float(np.sum(m0_season))
            self.fixed_season = list(m0_season.astype(float)) + [last]
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
                    # g1..g_{p-1} (newest-first already)
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
        self.gamma_path = []   # current γ_t each step

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
        """Return the (period-1) seasonal latent vector (newest-first) if dynamic; else []."""
        if self.seasonal_mode != "dynamic":
            return []
        start = 0
        if "alpha" in self._state_layout: start += 1
        if "beta"  in self._state_layout: start += 1
        return list(self.x_t[start:])

    # ----------------- contributions & recording -----------------
    def _alpha_contribution(self):
        """
        Level + (linear trend contribution) at time t.
        If level is dynamic, its state already includes the current level (no t-multiplication).
        If level is deterministic, we add trend * t here (dynamic or deterministic, if present).
        """
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
            g_vec = self._get_season_vec()  # newest-first
            return float(g_vec[0]) if len(g_vec) else 0.0   # observe FIRST coord = γ_t
        elif self.seasonal_mode == "deterministic":
            return float(self.fixed_season[self.t % self.period])
        else:
            return 0.0

    def _record_truth(self):
        alpha_c = self._alpha_contribution()
        beta_v  = self._beta_value_for_path()
        gamma_c = self._seasonal_contribution()  # γ_t (current)
        mu_t    = float(alpha_c + gamma_c)

        self.alpha_path.append(alpha_c)
        self.beta_path.append(beta_v)
        self.gamma_path.append(gamma_c)
        self.mu_path.append(mu_t)

    # ----------------- one-step evolution -----------------
    def move(self):
        """
        Advance t -> t+1.
          alpha_{t+1} = alpha_t + drift + N(0, q_level)
          beta_{t+1}  = beta_t  + N(0, q_trend)
          season (dynamic): draw γ_t = -sum(past) + ε, then shift right to keep newest-first.
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

        # season (period-1 vector, newest-first)
        if self.seasonal_mode == "dynamic":
            start = 0
            if "alpha" in self._state_layout: start += 1
            if "beta"  in self._state_layout: start += 1
            g_prev = list(self.x_t[start:])  # [γ_{t-1}, γ_{t-2}, ...] before update it's currently [γ_0, γ_{-1}, ...]
            mean_new = -float(np.sum(g_prev))
            g_new_first = np.random.normal(mean_new, np.sqrt(self.q_season))  # this is γ_t
            g_new_vec = [g_new_first] + g_prev[:-1] if len(g_prev) > 0 else [g_new_first]
            new_x[start:] = np.array(g_new_vec)

        self.x_t = new_x
        self.t += 1
        self._record_truth()

    # ----------------- measurement -----------------
    def measure(self):
        """Draw y_t ~ Normal(mu_t, sigma^2) using last recorded mu_t."""
        mu = self.mu_path[-1]
        y = norm.rvs(loc=mu, scale=self.sigma)
        self.all_measurements.append(y)
        self.index.append(self._advance_and_get_time())
        return y

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
            "alpha_t": to_arr(self.alpha_path),
            "beta_t": to_arr(self.beta_path),
            "gamma_t": to_arr(self.gamma_path),  # current seasonal effect
            "mu_t": to_arr(self.mu_path),
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

    def _parse_date(s: str | None):
        if not s:
            return None
        # Accept YYYY, YYYY-MM, or YYYY-MM-DD
        parts = [int(p) for p in s.split("-")]
        if len(parts) == 1:
            return datetime(parts[0], 1, 1)
        if len(parts) == 2:
            return datetime(parts[0], parts[1], 1)
        if len(parts) == 3:
            return datetime(parts[0], parts[1], parts[2])
        raise ValueError("start-date must be YYYY, YYYY-MM, or YYYY-MM-DD")

    p = argparse.ArgumentParser(
        description="Simulate a structural Gaussian time series (level/trend/season with independent modes)."
    )

    # Core simulation controls
    p.add_argument("--T", type=int, default=200, help="Number of observations to generate.")
    p.add_argument("--period", type=int, default=12, help="Seasonal period (>=2). 12=monthly, 4=quarterly, else yearly steps.")
    p.add_argument("--sigma", type=float, default=2.0, help="Observation noise SD.")
    p.add_argument("--start-date", type=str, default="2000-01-01", help="Start date (YYYY[-MM[-DD]]).")

    # Modes
    p.add_argument("--level-mode", choices=["dynamic", "deterministic"], default="dynamic")
    p.add_argument("--trend-mode", choices=["dynamic", "deterministic", "none"], default="dynamic")
    p.add_argument("--seasonal-mode", choices=["dynamic", "deterministic", "none"], default="dynamic")

    # Innovations (q_*) — only used for dynamic components
    p.add_argument("--q-level", type=float, default=0.05)
    p.add_argument("--q-trend", type=float, default=0.002)
    p.add_argument("--q-season", type=float, default=0.15)

    # Priors / fixed values
    p.add_argument("--m0-level", type=float, default=5.0)
    p.add_argument("--v0-level", type=float, default=0.25)
    p.add_argument("--m0-trend", type=float, default=0.015)
    p.add_argument("--v0-trend", type=float, default=0.05)
    p.add_argument("--m0-season", type=str, default="", help="Comma-separated length (period-1): (γ_{-(p-2)},...,γ_{-1},γ_0).")
    p.add_argument("--v0-season", type=str, default="", help="Comma-separated length (period-1) variances.")

    # I/O & misc
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--plot", default=True, help="Show matplotlib figures.")
    p.add_argument("--save-csv", type=str, default="", help="Path to save CSV with date,y,mu,alpha,beta,gamma.")
    p.add_argument("--print-summary", default=True, help="Print a small summary table at the end.")

    args = p.parse_args()
    np.random.seed(args.seed)

    # Build seasonal vectors (period-1) if provided or default
    def _maybe_parse(s, default_val):
        out = _csv_floats(s)
        return out if out is not None else [default_val] * (args.period - 1)

    m0_season = _maybe_parse(args.m0_season, 5.0)
    # sensible defaults: variance 0 for deterministic season, else 0.5
    v_default = 0.0 if args.seasonal_mode == "deterministic" else 0.5
    v0_season = _maybe_parse(args.v0_season, v_default)

    start_date = _parse_date(args.start_date)

    mts = Mean_Time_Series(
        sigma=args.sigma,
        level_mode=args.level_mode,
        trend_mode=args.trend_mode,
        seasonal_mode=args.seasonal_mode,
        period=args.period,
        q_level=args.q_level,
        q_trend=args.q_trend,
        q_season=args.q_season,
        m0_level=args.m0_level,
        v0_level=args.v0_level,
        m0_trend=(0.0 if args.trend_mode == "none" else args.m0_trend),
        v0_trend=args.v0_trend,
        m0_season=m0_season,
        v0_season=v0_season,
        start_date=start_date,
    )

    # Simulate
    y = []
    for _ in range(args.T):
        mts.move()
        y.append(mts.measure())

    truths = mts.get_truth_paths(as_numpy=True)
    y_arr     = np.asarray(y, float)
    mu_arr    = truths["mu_t"][1:1 + args.T]
    alpha_arr = truths["alpha_t"][1:1 + args.T]
    beta_arr  = truths["beta_t"][1:1 + args.T]
    gamma_arr = truths["gamma_t"][1:1 + args.T]
    dates_T   = truths["index"][:args.T]

    # Pack into DataFrame
    import pandas as pd
    df = pd.DataFrame({
        "date": dates_T,
        "y_t": y_arr,
        "mu_t": mu_arr,
        "alpha_t": alpha_arr,
        "beta_t": beta_arr,
        "gamma_t": gamma_arr,
    })

    # Save?
    if args.save_csv:
        out = args.save_csv
        df.to_csv(out, index=False)
        print(f"Saved {len(df)} rows to {out}")

    # Summary?
    if args.print_summary:
        with np.printoptions(suppress=True, precision=4):
            print("\n--- Summary ---")
            print(f"level_mode={args.level_mode}, trend_mode={args.trend_mode}, seasonal_mode={args.seasonal_mode}")
            print(f"sigma={args.sigma}, q_level={args.q_level}, q_trend={args.q_trend}, q_season={args.q_season}")
            print(f"m0_level={args.m0_level}, v0_level={args.v0_level}, "
                  f"m0_trend={(0.0 if args.trend_mode=='none' else args.m0_trend)}, v0_trend={args.v0_trend}")
            print(f"m0_season (oldest→newest) = {m0_season}")
            print(f"v0_season (oldest→newest) = {v0_season}")
            print(f"period={args.period}, start={dates_T[0]}, end={dates_T[-1]}")
            print(f"y mean={y_arr.mean():.3f}, sd={y_arr.std(ddof=1):.3f}")

    # Plots?
    if args.plot:
        # Figure 1: y and mu
        plt.figure(figsize=(10, 4))
        plt.plot(dates_T, y_arr, label=r"$y_t$", linewidth=1.0)
        plt.plot(dates_T, mu_arr, "--", label=r"$\mu_t$", linewidth=1.0)
        ttl = f"level={args.level_mode}, trend={args.trend_mode}, season={args.seasonal_mode}"
        plt.title(ttl)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Figure 2: truth paths alpha, beta, gamma
        fig2, ax2 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        ax2[0].plot(dates_T, alpha_arr, linewidth=1.0)
        ax2[0].set_ylabel(r"$\alpha_t$")
        ax2[0].grid(True)

        ax2[1].plot(dates_T, beta_arr, linewidth=1.0)
        ax2[1].set_ylabel(r"$\beta_t$")
        ax2[1].grid(True)

        ax2[2].plot(dates_T, gamma_arr, linewidth=1.0)
        ax2[2].set_ylabel(r"$\gamma_t$")
        ax2[2].set_xlabel("time")
        ax2[2].grid(True)

        fig2.suptitle("Truth paths")
        fig2.tight_layout()

        plt.show()
