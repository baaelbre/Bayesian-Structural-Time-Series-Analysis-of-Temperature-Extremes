# %% triple_time_series_structural.py
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Optional, Dict, Tuple

from scipy.stats import genextreme, norm
from scipy.optimize import minimize


def _advance_date(d: datetime, freq: str) -> datetime:
    if freq == "M":
        return d + relativedelta(months=+1)
    if freq == "Q":
        return d + relativedelta(months=+3)
    if freq == "Y":
        return d + relativedelta(years=+1)
    raise ValueError("freq must be one of {'M','Q','Y'}")


@dataclass
class TripleTimeSeries:
    """
    Structural Non-Dynamic Model:

      m_t = alpha + beta * t
      bulk:   \bar S_t ~ N(m_t, sigma_GA^2)

      mu_up_t   = m_t + (delta0_up + delta1_up * t)
      mu_down_t = m_t + (delta0_down + delta1_down * t)

      upper:  S_up_t      ~ GEV(mu_up_t,   sigma_up,   xi_up)           (maxima)
      lower:  -S_down_t   ~ GEV(mu_down_t, sigma_down, xi_down)  (sign-flip minima)

    SciPy's genextreme uses shape `c = -xi` for GEV maxima.
    """

    # --- time & calendar ---
    T: int = 180
    t0: float = 0.0
    dt: float = 1.0
    start_date: datetime = datetime(1990, 1, 1)
    freq: str = "M"

    # --- parameters (truth for simulation) ---
    alpha: float = 0.0
    beta: float = 0.02
    sigma_GA: float = 1.3

    sigma_up: float = 2.4
    xi_up: float = -0.25
    delta0_up: float = 6.0
    delta1_up: float = 0.015

    sigma_down: float = 2.0
    xi_down: float = -0.07
    delta0_down: float = -3.0
    delta1_down: float = 0.02

    # --- RNG ---
    seed: Optional[int] = 123

    # --- internal ---
    rng: np.random.Generator = field(init=False, repr=False)
    df_: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------
    def simulate(self) -> pd.DataFrame:
        """Simulate Gaussian bulk + two GEV tails according to the model."""
        T = int(self.T)
        t = self.t0 + self.dt * np.arange(T, dtype=float)

        # Linear predictors
        m_t = self.alpha + self.beta * t
        mu_up_t = m_t + (self.delta0_up + self.delta1_up * t)
        mu_dn_t = m_t + (self.delta0_down + self.delta1_down * t)

        # Observations
        bulk = self.rng.normal(loc=m_t, scale=self.sigma_GA, size=T)

        c_up = -self.xi_up
        upper = genextreme.rvs(c_up, loc=mu_up_t, scale=self.sigma_up,
                               random_state=self.rng, size=T)

        c_dn = -self.xi_down
        Z = genextreme.rvs(c_dn, loc=mu_dn_t, scale=self.sigma_down,
                           random_state=self.rng, size=T)
        lower = -Z

        # Build date index
        dates, d = [], self.start_date
        for _ in range(T):
            dates.append(d)
            d = _advance_date(d, self.freq)

        df = pd.DataFrame({
            "t": t,
            "m_t": m_t,
            "mu_up_t": mu_up_t,
            "mu_down_t": mu_dn_t,
            "bulk_obs": bulk,
            "upper_obs": upper,
            "lower_obs": lower
        }, index=pd.Index(dates, name="date"))

        self.df_ = df
        return df

    # -------------------------------------------------------------------------
    # Joint log-likelihood
    # -------------------------------------------------------------------------
    @staticmethod
    def _gev_logpdf(y: np.ndarray, mu: np.ndarray, sigma: float, xi: float) -> np.ndarray:
        """
        Log-pdf of GEV maxima (SciPy parameterization: c = -xi).
        y ~ GEV(mu, sigma, xi)  <=>  genextreme(c=-xi, loc=mu, scale=sigma)
        """
        if sigma <= 0:
            return np.full_like(y, -np.inf, dtype=float)
        c = -xi
        return genextreme.logpdf(y, c, loc=mu, scale=sigma)

    def loglik(self, theta: Dict[str, float]) -> float:
        """
        Joint log-likelihood:
          L = Π_t  N(bulk_t | m_t, sigma_GA^2)
                    * GEV(upper_t | mu_up_t, sigma_up, xi_up)
                    * GEV(-lower_t | mu_down_t, sigma_down, xi_down)
        where
          m_t = alpha + beta t
          mu_up_t   = m_t + delta0_up   + delta1_up   t
          mu_down_t = m_t + delta0_down + delta1_down t
        """
        if self.df_ is None:
            raise RuntimeError("simulate() first (or set df_ externally).")

        df = self.df_
        t = df["t"].to_numpy()

        # unpack theta
        alpha = theta["alpha"]; beta = theta["beta"]; sigma_GA = theta["sigma_GA"]
        delta0_up = theta["delta0_up"]; delta1_up = theta["delta1_up"]
        sigma_up = theta["sigma_up"]; xi_up = theta["xi_up"]
        delta0_dn = theta["delta0_dn"]; delta1_dn = theta["delta1_dn"]
        sigma_down = theta["sigma_down"]; xi_down = theta["xi_down"]

        # linear predictors
        m_t = alpha + beta * t
        mu_up_t = m_t + (delta0_up + delta1_up * t)
        mu_dn_t = m_t + (delta0_dn + delta1_dn * t)

        # components
        ll_bulk = norm.logpdf(df["bulk_obs"].values, loc=m_t, scale=sigma_GA).sum()
        ll_up = self._gev_logpdf(df["upper_obs"].values, mu_up_t, sigma_up, xi_up).sum()
        ll_dn = self._gev_logpdf(-df["lower_obs"].values, mu_dn_t, sigma_down, xi_down).sum()

        return float(ll_bulk + ll_up + ll_dn)

    # -------------------------------------------------------------------------
    # MLE convenience (optional)
    # -------------------------------------------------------------------------
    def fit_mle(
        self,
        init: Optional[Dict[str, float]] = None,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        options: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Maximize the joint log-likelihood over θ (Gaussian + two GEVs).
        Returns MLE dict.

        For stability:
          - enforce sigma_* > 0 via bounds
          - xi can be free, but watch for support violations in practice
        """
        if self.df_ is None:
            raise RuntimeError("simulate() first.")

        # default init at (near) truth
        init = init or {
            "alpha": self.alpha, "beta": self.beta, "sigma_GA": max(self.sigma_GA, 1e-2),
            "delta0_up": self.delta0_up, "delta1_up": self.delta1_up,
            "sigma_up": max(self.sigma_up, 1e-2), "xi_up": self.xi_up,
            "delta0_dn": self.delta0_down, "delta1_dn": self.delta1_down,
            "sigma_down": max(self.sigma_down, 1e-2), "xi_down": self.xi_down
        }

        # bounds
        bnds = bounds or {
            "alpha": (-np.inf, np.inf), "beta": (-np.inf, np.inf),
            "sigma_GA": (1e-6, np.inf),
            "delta0_up": (-np.inf, np.inf), "delta1_up": (-np.inf, np.inf),
            "sigma_up": (1e-6, np.inf), "xi_up": (-1.0, 1.0),  # adjust if desired
            "delta0_dn": (-np.inf, np.inf), "delta1_dn": (-np.inf, np.inf),
            "sigma_down": (1e-6, np.inf), "xi_down": (-1.0, 1.0)
        }

        keys = list(init.keys())

        def pack(th: Dict[str, float]) -> np.ndarray:
            return np.array([th[k] for k in keys], dtype=float)

        def unpack(x: np.ndarray) -> Dict[str, float]:
            return {k: float(v) for k, v in zip(keys, x)}

        def nll(x: np.ndarray) -> float:
            th = unpack(x)
            # NOTE: genextreme already checks support; if invalid, returns -inf → nll=+inf
            ll = self.loglik(th)
            return -ll

        x0 = pack(init)
        bounds_seq = [bnds[k] for k in keys]

        res = minimize(nll, x0=x0, method="L-BFGS-B", bounds=bounds_seq, options=options)
        mle = unpack(res.x)

        mle["success"] = bool(res.success)
        mle["message"] = res.message
        mle["fun"] = float(res.fun)      # minimized negative log-lik
        mle["loglik"] = -float(res.fun)  # maximized log-lik

        return mle

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    def plot_all(self):
        """Plot the three observed series and their latent linear predictors."""
        if self.df_ is None:
            raise RuntimeError("simulate() first.")
        df = self.df_

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index, df["bulk_obs"],  label="Bulk (Gaussian)", alpha=0.8)
        ax.plot(df.index, df["upper_obs"], label="Upper (GEV maxima)", alpha=0.7)
        ax.plot(df.index, -df["lower_obs"], label="Lower (GEV minima)", alpha=0.7)

        ax.plot(df.index, df["m_t"],       lw=2, label=r"$m_t=\alpha+\beta t$")
        ax.plot(df.index, df["mu_up_t"],   lw=2, label=r"$\mu^{(\uparrow)}_t$")
        ax.plot(df.index, df["mu_down_t"], lw=2, label=r"$\mu^{(\downarrow)}_t$")

        ax.set_title("Gaussian bulk + two GEV tails (structural non-dynamic)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend(ncol=2)
        plt.tight_layout()
        plt.show()

    def plot_residuals(self):
        """Quick residual diagnostic: obs minus latent location."""
        if self.df_ is None:
            raise RuntimeError("simulate() first.")
        df = self.df_.copy()
        res_bulk = df["bulk_obs"] - df["m_t"]
        res_up = df["upper_obs"] - df["mu_up_t"]
        res_dn = (-df["lower_obs"]) - df["mu_down_t"]  # compare on -S for GEV

        fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        axes[0].plot(df.index, res_bulk); axes[0].axhline(0, color="k", lw=1)
        axes[0].set_title("Residuals: bulk - m_t")
        axes[0].grid(True)

        axes[1].plot(df.index, res_up); axes[1].axhline(0, color="k", lw=1)
        axes[1].set_title("Residuals: upper - mu_up_t")
        axes[1].grid(True)

        axes[2].plot(df.index, res_dn); axes[2].axhline(0, color="k", lw=1)
        axes[2].set_title("Residuals: (-lower) - mu_down_t  (GEV scale)")
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()


# ----------------------------- #
#%% Example usage
# ----------------------------- #
if __name__ == "__main__":
    sim = TripleTimeSeries(
        T=144,                        # 12 years monthly
        alpha=0.0, beta=0.025,        # bulk trend
        sigma_GA=0.9,
        sigma_up=1.3, xi_up=-0.25, delta0_up=8,  delta1_up=0.02,
        sigma_down=1.1, xi_down=-0.07, delta0_down=-8, delta1_down=0.02,
        start_date=datetime(2000, 1, 1), freq="M", seed=7
    )
    df = sim.simulate()
    print(df.head())

    # Plots
    sim.plot_all()


    #%% joint MLE (on the same data)

    alpha_init = 0.5; beta_init = 0.01; sigma_GA_init = 1.0
    delta0_up_init = 8; delta1_up_init = 0.02; sigma_up_init = 1.3; xi_up_init = -0.25
    delta0_dn_init = -8; delta1_dn_init = 0.02; sigma_down_init = 1.1; xi_down_init = -0.07

    mle = sim.fit_mle(init={
            "alpha": alpha_init, "beta": beta_init, "sigma_GA": max(sigma_GA_init, 1e-2),
            "delta0_up": delta0_up_init, "delta1_up": delta1_up_init,
            "sigma_up": max(sigma_up_init, 1e-2), "xi_up": xi_up_init,
            "delta0_dn": delta0_dn_init, "delta1_dn": delta1_dn_init,
            "sigma_down": max(sigma_down_init, 1e-2), "xi_down": xi_down_init})
    
    print("\nMLE summary:")
    for k, v in mle.items():
        if isinstance(v, float):
            print(f"  {k:>12s}: {v: .6f}")
        else:
            print(f"  {k:>12s}: {v}")

# %%
