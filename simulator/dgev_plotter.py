# %% simulator/dgev_plotter.py
import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


class DGEVPlotter:
    """
    Standalone plotting helper for DGEVParticleGibbs results.

    Handles both dynamic and deterministic components.

    Expected sampler-like attributes:
      keep, y, T,
      include_level, include_trend, include_seasonality,
      idx_alpha, idx_beta, idx_gamma_end,
      true_sigma, true_xi, true_Q, true_mu_t, true_alpha_t, true_beta_t, true_gamma_t,
      accept, proposals
    """

    def __init__(self, sampler_like, level: float = 0.90):
        self.s = sampler_like
        self.level = float(level)
        if not (0.0 < self.level < 1.0):
            raise ValueError("`level` must be in (0,1).")
        self.lo_q = (1.0 - self.level) / 2.0
        self.hi_q = 1.0 - self.lo_q
        self.band_label = f"{int(round(self.level * 100))}% band"

    # ---------- small MCMC utilities ----------
    @staticmethod
    def _acf(x, max_lag=40):
        x = np.asarray(x, float)
        n = x.size
        if n == 0:
            return np.array([np.nan])
        if n == 1:
            return np.array([1.0])
        x = x - np.mean(x)
        denom = float(np.dot(x, x)) + 1e-300
        L = int(min(max_lag, n - 1))
        ac = np.empty(L + 1, dtype=float)
        for k in range(L + 1):
            ac[k] = float(np.dot(x[:n - k], x[k:])) / denom
        return ac

    @staticmethod
    def _ess(x, max_lag=100):
        ac = DGEVPlotter._acf(x, max_lag=max_lag)
        if not np.all(np.isfinite(ac)) or ac.size <= 1:
            return float(len(x))
        s = 0.0
        for k in range(1, ac.size):
            if ac[k] <= 0:
                break
            s += 2.0 * ac[k]
        n = len(x)
        return float(n) / (1.0 + s)

    @staticmethod
    def _geweke_z(x, first_frac=0.1, last_frac=0.5):
        x = np.asarray(x, float)
        n = x.size
        if n < 4:
            return np.nan
        a = max(1, int(np.floor(first_frac * n)))
        b = max(1, int(np.floor(last_frac * n)))
        xa = x[:a]
        xb = x[n - b:]
        if xa.size < 2 or xb.size < 2:
            return np.nan
        ma, mb = np.mean(xa), np.mean(xb)
        va = np.var(xa, ddof=1) / xa.size
        vb = np.var(xb, ddof=1) / xb.size
        denom = np.sqrt(va + vb) + 1e-300
        return (ma - mb) / denom

    # ---------- helpers for deterministic components ----------
    def _tile_static_to_T(self, values_1d: np.ndarray) -> np.ndarray:
        v = np.asarray(values_1d, float)[:, None]  # (n_keep, 1)
        return v * np.ones((1, self.s.T), dtype=float)  # (n_keep, T)

    def _tile_season_to_T(self, season_draws_2d: np.ndarray) -> np.ndarray:
        V = np.asarray(season_draws_2d, float)  # (n_keep, p)
        if V.ndim != 2:
            raise ValueError("season_vector must be (n_keep, period).")
        _, P = V.shape
        reps = int(np.ceil(self.s.T / P))
        return np.tile(V, reps)[:, : self.s.T]  # (n_keep, T)

    # ---------- summaries for time-varying draws ----------
    def _summarize_paths(self, draws_2d: np.ndarray, center: str = "median", map_bins: int = 50):
        X = np.asarray(draws_2d, float)
        lo = np.quantile(X, self.lo_q, axis=0)
        hi = np.quantile(X, self.hi_q, axis=0)

        map_line = None
        if center == "median":
            ctr = np.median(X, axis=0)
        elif center == "mean":
            ctr = np.mean(X, axis=0)
        elif center == "map":
            T = X.shape[1]
            map_vals = np.empty(T, dtype=float)
            for t in range(T):
                col = X[:, t]
                col = col[np.isfinite(col)]
                if col.size == 0:
                    map_vals[t] = np.nan
                    continue
                hist, edges = np.histogram(col, bins=map_bins, density=True)
                j = int(np.argmax(hist))
                map_vals[t] = 0.5 * (edges[j] + edges[j + 1])
            ctr = map_vals
            map_line = map_vals
        else:
            raise ValueError("center must be one of {'median','mean','map'}")

        return ctr, lo, hi, map_line

    def _plot_component_figure(
        self,
        series_draws: np.ndarray,
        title: str,
        ylabel: str,
        true_series: Optional[np.ndarray],
        save_dir: Optional[str],
        fname_prefix: str,
        center: str = "median",
        map_bins: int = 50,
        show: bool = True,
    ):
        ctr, lo, hi, map_line = self._summarize_paths(series_draws, center=center, map_bins=map_bins)
        fig, ax = plt.subplots(figsize=(12, 3.2))
        t = np.arange(self.s.T)
        ax.plot(ctr, lw=1.8, label=f"{ylabel} {center}")
        ax.fill_between(t, lo, hi, alpha=0.25, label=f"{int(round(self.level*100))}% band")
        if map_line is not None and center != "map":
            ax.plot(map_line, lw=1.0, ls=":", label=f"{ylabel} MAP (fast)")
        if true_series is not None and len(true_series) == self.s.T and np.all(np.isfinite(true_series)):
            ax.plot(true_series, lw=1.4, ls="--", label=f"true {ylabel}")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, self.s.T - 1)
        ax.legend(loc="best")
        plt.tight_layout()
        if save_dir:
            _ensure_dir(save_dir)
            out_path = os.path.join(save_dir, f"{fname_prefix}.png")
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            print(f"[save] Figure -> {out_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    # ---------- figures ----------
    def plot_diagnostics(self, save_dir: Optional[str] = None, fname_prefix: str = "diagnostics", show: bool = True):
        kept = self.s.keep
        fig, axs = plt.subplots(3, 3, figsize=(13, 10))
        axs = axs.ravel()

        # traces
        axs[0].plot(kept["sigma"], lw=1)
        axs[0].set_title(r"trace: $\sigma$")
        if getattr(self.s, "true_sigma", None) is not None:
            axs[0].axhline(self.s.true_sigma, ls="--", lw=1.2, label=r"true $\sigma$")
            axs[0].legend()

        axs[1].plot(kept["xi"], lw=1)
        axs[1].set_title(r"trace: $\xi$")
        if getattr(self.s, "true_xi", None) is not None:
            axs[1].axhline(self.s.true_xi, ls="--", lw=1.2, label=r"true $\xi$")
            axs[1].legend()

        # innovation variance traces for dynamic states only
        if getattr(self.s, "include_level", False) and "Q" in kept and self.s.idx_alpha is not None:
            axs[2].plot(kept["Q"][:, self.s.idx_alpha], lw=1)
            axs[2].set_title(r"trace: $Q_\alpha$")
            if getattr(self.s, "true_Q", None) is not None:
                axs[2].axhline(self.s.true_Q[self.s.idx_alpha], ls="--", lw=1.2, label=r"true $Q_\alpha$")
                axs[2].legend()
        else:
            axs[2].axis("off")

        # posteriors (marginals)
        axs[3].hist(kept["sigma"], bins=30, density=True)
        axs[3].set_title(r"posterior: $\sigma$")
        if getattr(self.s, "true_sigma", None) is not None:
            axs[3].axvline(self.s.true_sigma, ls="--", lw=1.5, label=r"true $\sigma$")
            axs[3].legend()

        axs[4].hist(kept["xi"], bins=30, density=True)
        axs[4].set_title(r"posterior: $\xi$")
        if getattr(self.s, "true_xi", None) is not None:
            axs[4].axvline(self.s.true_xi, ls="--", lw=1.5, label=r"true $\xi$")
            axs[4].legend()

        if getattr(self.s, "include_trend", False) and "Q" in kept and self.s.idx_beta is not None:
            axs[5].hist(kept["Q"][:, self.s.idx_beta], bins=30, density=True)
            axs[5].set_title(r"posterior: $Q_\beta$")
            if getattr(self.s, "true_Q", None) is not None:
                axs[5].axvline(self.s.true_Q[self.s.idx_beta], ls="--", lw=1.5, label=r"true $Q_\beta$")
                axs[5].legend()
        else:
            axs[5].axis("off")

        if getattr(self.s, "include_seasonality", False) and "Q" in kept and self.s.idx_gamma_end is not None:
            axs[6].hist(kept["Q"][:, self.s.idx_gamma_end], bins=30, density=True)
            axs[6].set_title(r"posterior: $Q_\gamma$")
            if getattr(self.s, "true_Q", None) is not None:
                axs[6].axvline(self.s.true_Q[self.s.idx_gamma_end], ls="--", lw=1.5, label=r"true $Q_\gamma$")
                axs[6].legend()
        else:
            axs[6].axis("off")

        # acceptance summary
        def rate(a, p):
            return 0.0 if (p is None or p == 0) else a / p

        acc = getattr(self.s, "accept", {"logsigma": 0, "xi": 0})
        props = getattr(self.s, "proposals", {"logsigma": 1, "xi": 1})
        txt = (
            f"accept(logsigma) = {rate(acc.get('logsigma', 0), props.get('logsigma', 0)):.2f}\n"
            f"accept(xi) = {rate(acc.get('xi', 0), props.get('xi', 0)):.2f}\n"
            f"(states via PG-BSi: acceptance = 1)"
        )
        axs[7].axis("off")
        axs[7].text(0.05, 0.6, txt, fontsize=12)

        # posterior μ vs y
        mu_med = np.median(kept["mu"], axis=0)
        mu_lo = np.quantile(kept["mu"], self.lo_q, axis=0)
        mu_hi = np.quantile(kept["mu"], self.hi_q, axis=0)
        axs[8].plot(self.s.y, label=r"$y_t$", lw=1, alpha=0.6)
        axs[8].plot(mu_med, label=r"$\mu_t$ median", lw=1.5)
        axs[8].fill_between(np.arange(self.s.T), mu_lo, mu_hi, alpha=0.2, label=self.band_label)
        if getattr(self.s, "true_mu_t", None) is not None and len(self.s.true_mu_t) == self.s.T:
            axs[8].plot(self.s.true_mu_t, lw=1.5, ls="--", label=r"true $\mu_t$")
        axs[8].set_title(r"Filtered $\mu_t$ vs observations $y_t$")
        axs[8].legend()

        plt.tight_layout()
        if save_dir is not None:
            _ensure_dir(save_dir)
            out_path = os.path.join(save_dir, f"{fname_prefix}.png")
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            print(f"[save] Figure -> {out_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_mcmc_diagnostics_extra(
        self, max_lag=40, save_dir: Optional[str] = None, fname_prefix: str = "mcmc_extra", show: bool = True
    ):
        s = self.s.keep["sigma"]
        x = self.s.keep["xi"]
        fig, axs = plt.subplots(2, 3, figsize=(14, 8))

        # sigma
        axs[0, 0].plot(np.cumsum(s) / np.arange(1, len(s) + 1))
        axs[0, 0].set_title(r"$\sigma$ running mean")
        ac_s = self._acf(s, max_lag=max_lag)
        axs[0, 1].stem(range(len(ac_s)), ac_s)
        axs[0, 1].set_title(r"$\sigma$ ACF")
        ess_s = self._ess(s, max_lag=5 * max_lag)
        gz_s = self._geweke_z(s)
        axs[0, 2].hist(s, bins=30, density=True)
        axs[0, 2].set_title(r"$\sigma$ hist" + f"(ESS≈{ess_s:.0f}, Geweke z≈{gz_s:.2f})")
        if getattr(self.s, "true_sigma", None) is not None:
            axs[0, 2].axvline(self.s.true_sigma, ls="--", lw=1.2, label=r"true $\sigma$")
            axs[0, 2].legend()

        # xi
        axs[1, 0].plot(np.cumsum(x) / np.arange(1, len(x) + 1))
        axs[1, 0].set_title(r"$\xi$ running mean")
        ac_x = self._acf(x, max_lag=max_lag)
        axs[1, 1].stem(range(len(ac_x)), ac_x)
        axs[1, 1].set_title(r"$\xi$ ACF")
        ess_x = self._ess(x, max_lag=5 * max_lag)
        gz_x = self._geweke_z(x)
        axs[1, 2].hist(x, bins=30, density=True)
        axs[1, 2].set_title(r"$\xi$ hist " + f"(ESS≈{ess_x:.0f}, Geweke z≈{gz_x:.2f})")
        if getattr(self.s, "true_xi", None) is not None:
            axs[1, 2].axvline(self.s.true_xi, ls="--", lw=1.2, label=r"true $\xi$")
            axs[1, 2].legend()

        plt.tight_layout()
        if save_dir is not None:
            _ensure_dir(save_dir)
            out_path = os.path.join(save_dir, f"{fname_prefix}.png")
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_states_and_observations(
        self, save_dir: Optional[str] = None, fname_prefix: str = "states", show: bool = True
    ):
        """
        Combined figure: data vs μ_t band (+ true μ), plus α_t, β_t, γ_t
        with credible bands and true overlays when provided.
        """
        mu_med = np.median(self.s.keep["mu"], axis=0)
        mu_lo = np.quantile(self.s.keep["mu"], self.lo_q, axis=0)
        mu_hi = np.quantile(self.s.keep["mu"], self.hi_q, axis=0)

        n_rows = (
            1
            + (1 if getattr(self.s, "include_level", False) else 0)
            + (1 if getattr(self.s, "include_trend", False) else 0)
            + (1 if getattr(self.s, "include_seasonality", False) else 0)
        )
        fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3.0 * n_rows), sharex=True)
        if n_rows == 1:
            axes = [axes]

        # (1) y vs μ bands (+ true μ)
        ax = axes[0]
        ax.plot(self.s.y, label=r"$y_t$", lw=1, alpha=0.7)
        ax.plot(mu_med, label=r"$\mu_t$ median", lw=1.6)
        ax.fill_between(np.arange(self.s.T), mu_lo, mu_hi, alpha=0.25, label=rf"$\mu_t$ {self.band_label}")
        if getattr(self.s, "true_mu_t", None) is not None and len(self.s.true_mu_t) == self.s.T:
            ax.plot(self.s.true_mu_t, lw=1.4, ls="--", label=r"true $\mu_t$")
        ax.set_title("Data and posterior $\mu_t$")
        ax.legend(loc="best")

        row = 1
        # (2) α_t
        if getattr(self.s, "include_level", False) and "alpha_t" in self.s.keep:
            at = self.s.keep["alpha_t"]
            a_med = np.median(at, axis=0)
            a_lo = np.quantile(at, self.lo_q, axis=0)
            a_hi = np.quantile(at, self.hi_q, axis=0)
            ax = axes[row]
            ax.plot(a_med, lw=1.6, label=r"$\alpha_t$ median")
            ax.fill_between(np.arange(self.s.T), a_lo, a_hi, alpha=0.25, label=self.band_label)
            if getattr(self.s, "true_alpha_t", None) is not None and len(self.s.true_alpha_t) == self.s.T:
                ax.plot(self.s.true_alpha_t, lw=1.4, ls="--", label=r"true $\alpha_t$")
            ax.set_title(r"Level component $\alpha_t$")
            ax.legend(loc="best")
            row += 1

        # (3) β
        if getattr(self.s, "include_trend", False):
            if "beta_t" in self.s.keep:
                bt = self.s.keep["beta_t"]
                b_med = np.median(bt, axis=0)
                b_lo = np.quantile(bt, self.lo_q, axis=0)
                b_hi = np.quantile(bt, self.hi_q, axis=0)
                ax = axes[row]
                ax.plot(b_med, lw=1.6, label=r"$\beta_t$ median")
                ax.fill_between(np.arange(self.s.T), b_lo, b_hi, alpha=0.25, label=self.band_label)
                if getattr(self.s, "true_beta_t", None) is not None and len(self.s.true_beta_t) == self.s.T:
                    ax.plot(self.s.true_beta_t, lw=1.4, ls="--", label=r"true $\beta_t$")
                ax.set_title(r"Trend component $\beta_t$")
                ax.legend(loc="best")
                row += 1
            elif ("slope_value" in self.s.keep) or ("beta_value" in self.s.keep):
                vals = self.s.keep["slope_value"] if "slope_value" in self.s.keep else self.s.keep["beta_value"]
                bv = self._tile_static_to_T(np.asarray(vals, float))
                b_med = np.median(bv, axis=0)
                b_lo = np.quantile(bv, self.lo_q, axis=0)
                b_hi = np.quantile(bv, self.hi_q, axis=0)
                ax = axes[row]
                ax.plot(b_med, lw=1.6, label=r"$\beta$ (det.) median")
                ax.fill_between(np.arange(self.s.T), b_lo, b_hi, alpha=0.25, label=self.band_label)
                ax.set_title(r"Deterministic slope $\beta$")
                ax.legend(loc="best")
                row += 1

        # (4) γ
        if getattr(self.s, "include_seasonality", False):
            if "gamma_t" in self.s.keep:
                gt = self.s.keep["gamma_t"]
                g_med = np.median(gt, axis=0)
                g_lo = np.quantile(gt, self.lo_q, axis=0)
                g_hi = np.quantile(gt, self.hi_q, axis=0)
                ax = axes[row]
                ax.plot(g_med, lw=1.6, label=r"$\gamma_t$ median")
                ax.fill_between(np.arange(self.s.T), g_lo, g_hi, alpha=0.25, label=self.band_label)
                if getattr(self.s, "true_gamma_t", None) is not None and len(self.s.true_gamma_t) == self.s.T:
                    ax.plot(self.s.true_gamma_t, lw=1.4, ls="--", label=r"true $\gamma_t$")
                ax.set_title(r"Seasonal component $\gamma_t$")
                ax.legend(loc="best")
            elif "season_vector" in self.s.keep:
                tiled = self._tile_season_to_T(self.s.keep["season_vector"])
                t_med = np.median(tiled, axis=0)
                t_lo = np.quantile(tiled, self.lo_q, axis=0)
                t_hi = np.quantile(tiled, self.hi_q, axis=0)
                ax = axes[row]
                ax.plot(t_med, lw=1.6, label=r"season (det.) median")
                ax.fill_between(np.arange(self.s.T), t_lo, t_hi, alpha=0.25, label=self.band_label)
                ax.set_title(r"Deterministic seasonality")
                ax.legend(loc="best")

        plt.tight_layout()
        if save_dir is not None:
            _ensure_dir(save_dir)
            out_path = os.path.join(save_dir, f"{fname_prefix}.png")
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            print(f"[save] Figure -> {out_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_components_separately(
        self,
        save_dir: Optional[str] = None,
        fname_prefix: str = "components",
        center: str = "median",
        map_bins: int = 50,
        show: bool = True,
    ):
        # μ_t
        self._plot_component_figure(
            self.s.keep["mu"],
            title=r"Posterior $\mu_t$",
            ylabel=r"$\mu_t$",
            true_series=getattr(self.s, "true_mu_t", None),
            save_dir=save_dir,
            fname_prefix=f"{fname_prefix}_mu",
            center=center,
            map_bins=map_bins,
            show=show,
        )

        # α_t
        if getattr(self.s, "include_level", False) and "alpha_t" in self.s.keep and self.s.keep["alpha_t"].size > 0:
            self._plot_component_figure(
                self.s.keep["alpha_t"],
                title=r"Posterior level $\alpha_t$",
                ylabel=r"$\alpha_t$",
                true_series=getattr(self.s, "true_alpha_t", None),
                save_dir=save_dir,
                fname_prefix=f"{fname_prefix}_alpha",
                center=center,
                map_bins=map_bins,
                show=show,
            )

        # β
        if getattr(self.s, "include_trend", False):
            if "beta_t" in self.s.keep and self.s.keep["beta_t"].size > 0:
                self._plot_component_figure(
                    self.s.keep["beta_t"],
                    title=r"Posterior trend $\beta_t$",
                    ylabel=r"$\beta_t$",
                    true_series=getattr(self.s, "true_beta_t", None),
                    save_dir=save_dir,
                    fname_prefix=f"{fname_prefix}_beta",
                    center=center,
                    map_bins=map_bins,
                    show=show,
                )
            elif ("slope_value" in self.s.keep) or ("beta_value" in self.s.keep):
                vals = self.s.keep["slope_value"] if "slope_value" in self.s.keep else self.s.keep["beta_value"]
                bv = self._tile_static_to_T(np.asarray(vals, float))
                self._plot_component_figure(
                    bv,
                    title=r"Posterior (deterministic) slope $\beta$",
                    ylabel=r"$\beta$",
                    true_series=None,
                    save_dir=save_dir,
                    fname_prefix=f"{fname_prefix}_beta_det",
                    center=center,
                    map_bins=map_bins,
                    show=show,
                )

        # γ
        if getattr(self.s, "include_seasonality", False):
            if "gamma_t" in self.s.keep and self.s.keep["gamma_t"].size > 0:
                self._plot_component_figure(
                    self.s.keep["gamma_t"],
                    title=r"Posterior season (last coord) $\gamma_t$",
                    ylabel=r"$\gamma_t$",
                    true_series=getattr(self.s, "true_gamma_t", None),
                    save_dir=save_dir,
                    fname_prefix=f"{fname_prefix}_gamma",
                    center=center,
                    map_bins=map_bins,
                    show=show,
                )
            elif "season_vector" in self.s.keep:
                tiled = self._tile_season_to_T(self.s.keep["season_vector"])
                self._plot_component_figure(
                    tiled,
                    title=r"Posterior (deterministic) seasonality",
                    ylabel=r"$\gamma_t$",
                    true_series=None,
                    save_dir=save_dir,
                    fname_prefix=f"{fname_prefix}_season_det",
                    center=center,
                    map_bins=map_bins,
                    show=show,
                )

    def plot_quick_hist_panel(
        self, posterior: dict, save_dir: Optional[str] = None, fname_prefix: str = "quick_hist", show: bool = True
    ):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].hist(posterior["sigma"], bins=40, density=True)
        axes[0].set_title(r"$\sigma | y$")
        axes[1].hist(posterior["xi"], bins=40, density=True)
        axes[1].set_title(r"$\xi | y$")
        if getattr(self.s, "include_level", False) and "Q" in posterior and self.s.idx_alpha is not None:
            axes[2].hist(posterior["Q"][:, self.s.idx_alpha], bins=40, density=True)
            axes[2].set_title(r"$Q_\alpha | y$")
        else:
            axes[2].axis("off")
        plt.tight_layout()
        if save_dir is not None:
            _ensure_dir(save_dir)
            out_path = os.path.join(save_dir, f"{fname_prefix}.png")
            plt.savefig(out_path, dpi=200, bbox_inches="tight")
            print(f"[save] Figure -> {out_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)


# ----------------------------- #
# CLI: Load a posterior and plot
# ----------------------------- #
if __name__ == "__main__":
    import sys
    import argparse

    # Allow importing from project root
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    # Use your simplified posterior helpers
    from optimization.posterior_bundle import load_posterior, find_latest_run

    parser = argparse.ArgumentParser(description="Plot DGEV posterior diagnostics from a saved run.")
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Path to a run directory (containing posterior.npz) or to a posterior.npz file. If provided, search is skipped.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="results/simulations/DGEV",
        help="Root folder to search when --run is not given (e.g., 'results', 'uccle/TX', or 'uccle/TN').",
    )
    parser.add_argument("--level", type=float, default=0.90, help="Credible interval level for bands.")
    parser.add_argument("--center", type=str, default="median", choices=["median", "mean", "map"],
                        help="Center line for per-component plots.")
    parser.add_argument("--show", default=True, help="Also display figures interactively (in addition to saving).")
    parser.add_argument("--skip-states", default=False, help="Skip the stacked states/observations panel.")
    parser.add_argument("--skip-separate", default=False, help="Skip separate component figures.")
    parser.add_argument("--map-bins", type=int, default=60, help="Bins for fast per-time MAP estimation.")

    args = parser.parse_args()

    # ---- Resolve target run (simple) ----
    if args.run is not None:
        target = args.run
        print(f"[info] Using explicit run: {target}")
    else:
        print(f"[info] Searching latest run under: {args.root}")
        target = find_latest_run(root=args.root)

    if not target:
        print("[error] No runs found. Provide --run or ensure results exist in the given --root.")
        sys.exit(1)

    print(f"[info] Using run: {target}")

    # Load posterior bundle
    bundle = load_posterior(target)
    draws = bundle.draws
    meta = bundle.meta

    # Build a minimal 'sampler-like' object expected by DGEVPlotter
    class Sampler:
        pass

    s = Sampler()
    s.keep = draws

    # Infer T
    if "T" in meta:
        s.T = int(meta["T"])
    elif "mu" in draws:
        s.T = int(draws["mu"].shape[1])
    else:
        s.T = int(draws["sigma"].shape[0])

    # y may not have been saved; try to use it if present, otherwise NaNs
    y_draw = draws.get("y", None)
    if y_draw is not None and len(y_draw) == s.T:
        s.y = y_draw
    else:
        s.y = np.full(s.T, np.nan, dtype=float)

    # Component presence: infer from draws if meta flags are missing
    s.include_level = bool(meta.get("include_level", ("alpha_t" in draws)))
    s.include_trend = bool(meta.get("include_trend", (("beta_t" in draws) or ("beta_value" in draws) or ("slope_value" in draws))))
    s.include_seasonality = bool(meta.get("include_seasonality", (("gamma_t" in draws) or ("season_vector" in draws))))

    # Indices
    s.idx_alpha = meta.get("idx_alpha", None)
    s.idx_beta = meta.get("idx_beta", None)
    s.idx_gamma_end = meta.get("idx_gamma_end", None)

    # Truth overlays
    s.true_sigma = meta.get("true_sigma", None)
    s.true_xi = meta.get("true_xi", None)
    _true_Q = meta.get("true_Q", None)
    s.true_Q = (np.array(_true_Q) if _true_Q is not None else None)
    s.true_mu_t = draws.get("true_mu_t", None)
    s.true_alpha_t = draws.get("true_alpha_t", None)
    s.true_beta_t = draws.get("true_beta_t", None)
    s.true_gamma_t = draws.get("true_gamma_t", None)

    # MCMC bookkeeping (for acceptance text panel)
    s.accept = meta.get("accept")
    s.iterations = meta.get("proposals")

    # Save directory (figures saved back into the run directory)
    base_dir = os.path.dirname(bundle.npz_path)
    save_dir = os.path.join(base_dir, "figures")
    _ensure_dir(save_dir)
    print(f"[info] Saving figures to: {save_dir}")

    # Make plots
    plotter = DGEVPlotter(s, level=float(args.level))
    plotter.plot_diagnostics(save_dir=save_dir, fname_prefix="diagnostics", show=bool(args.show))
    plotter.plot_mcmc_diagnostics_extra(40, save_dir=save_dir, fname_prefix="mcmc_extra", show=bool(args.show))
    if ("mu" in draws) and (not bool(args.skip_states)):
        plotter.plot_states_and_observations(save_dir=save_dir, fname_prefix="states", show=bool(args.show))
    else:
        print("[warn] Skipping states/observations plot (no 'mu' in draws or --skip-states set).")

    if not bool(args.skip_separate):
        plotter.plot_components_separately(
            save_dir=save_dir,
            fname_prefix="components",
            center=str(args.center),
            map_bins=int(args.map_bins),
            show=bool(args.show),
        )

    plotter.plot_quick_hist_panel(draws, save_dir=save_dir, fname_prefix="quick_hist", show=bool(args.show))
    print("[done] Plots generated.")
