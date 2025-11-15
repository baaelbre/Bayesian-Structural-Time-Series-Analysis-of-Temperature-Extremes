# %% simulator/dgev_plotter.py
import os
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


class DGEVPlotter:
    """
    Standalone plotting helper for **harmonic structural DGEV** results.

    The expected sampler-like object `s` should have:
      - s.keep: dict of posterior arrays, e.g.
          'sigma'  : (N,)
          'xi'     : (N,)
          'mu'     : (N, T)
          optional dynamic states:
            'alpha_t' : (N, T)   # level
            'beta_t'  : (N, T)   # trend
            'gamma_t' : (N, T)   # seasonal contribution (loaded harmonic state)
          optional process variances:
            'Q_alpha' : (N,)
            'Q_beta'  : (N,)
            'Q_gamma' : (N,)
          optional deterministic parameters (if some blocks are deterministic):
            'm0_alpha' : (N,)
            'm0_beta'  : (N,)
            'm0_cos'   : (N, K)
            'm0_sin'   : (N, K)
            'm0_nyq'   : (N,)
          optional overlays / truth:
            'true_mu_t', 'true_alpha_t', 'true_beta_t', 'true_gamma_t'
            meta fields: 'true_sigma', 'true_xi'
      - s.T: length of time series
      - s.y: data (T,) or NaNs
      - s.include_level, s.include_trend, s.include_seasonality: booleans
      - s.accept, s.proposals: dicts of MH/RJ acceptance counts (optional)
    """

    def __init__(self, sampler_like, level: float = 0.90):
        self.s = sampler_like
        self.level = float(level)
        if not (0.0 < self.level < 1.0):
            raise ValueError("`level` must be in (0, 1).")
        self.lo_q = (1.0 - self.level) / 2.0
        self.hi_q = 1.0 - self.lo_q
        self.band_label = f"{int(round(self.level * 100))}% band"

    # ------------------------------------------------------------------
    # Small MCMC utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _acf(x: np.ndarray, max_lag: int = 40) -> np.ndarray:
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
            ac[k] = float(np.dot(x[: n - k], x[k:])) / denom
        return ac

    @staticmethod
    def _ess(x: np.ndarray, max_lag: int = 100) -> float:
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
    def _geweke_z(x: np.ndarray, first_frac: float = 0.1, last_frac: float = 0.5) -> float:
        x = np.asarray(x, float)
        n = x.size
        if n < 4:
            return np.nan
        a = max(1, int(np.floor(first_frac * n)))
        b = max(1, int(np.floor(last_frac * n)))
        xa = x[:a]
        xb = x[n - b :]
        if xa.size < 2 or xb.size < 2:
            return np.nan
        ma, mb = np.mean(xa), np.mean(xb)
        va = np.var(xa, ddof=1) / xa.size
        vb = np.var(xb, ddof=1) / xb.size
        denom = np.sqrt(va + vb) + 1e-300
        return (ma - mb) / denom

    # ------------------------------------------------------------------
    # Summaries for time-varying paths
    # ------------------------------------------------------------------
    def _summarize_paths(
        self,
        draws_2d: np.ndarray,
        center: str = "median",
        map_bins: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        draws_2d: (N, T)
        Returns: (center_line, lo_band, hi_band, fast_MAP_line or None)
        """
        X = np.asarray(draws_2d, float)
        lo = np.quantile(X, self.lo_q, axis=0)
        hi = np.quantile(X, self.hi_q, axis=0)

        map_line: Optional[np.ndarray] = None
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
    ) -> None:
        ctr, lo, hi, map_line = self._summarize_paths(series_draws, center=center, map_bins=map_bins)
        fig, ax = plt.subplots(figsize=(12, 3.2))
        t = np.arange(self.s.T)
        ax.plot(t, ctr, lw=1.8, label=f"{ylabel} {center}")
        ax.fill_between(t, lo, hi, alpha=0.25, label=self.band_label)
        if map_line is not None and center != "map":
            ax.plot(t, map_line, lw=1.0, ls=":", label=f"{ylabel} MAP (fast)")
        if true_series is not None and len(true_series) == self.s.T and np.all(np.isfinite(true_series)):
            ax.plot(t, true_series, lw=1.4, ls="--", label=f"true {ylabel}")
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

    # ------------------------------------------------------------------
    # Acceptance helpers (for MH steps like logsigma, xi)
    # ------------------------------------------------------------------
    def _accept_pct(self, key: str) -> Optional[float]:
        acc = getattr(self.s, "accept", {}) or {}
        props = getattr(self.s, "proposals", {}) or {}
        a = acc.get(key, None)
        p = props.get(key, None)
        if a is None or p is None or p == 0:
            return None
        return 100.0 * float(a) / float(p)

    def _fmt_accept(self, key: str) -> str:
        pct = self._accept_pct(key)
        return "{:.1f}%".format(pct) if pct is not None else "n/a"

    # ------------------------------------------------------------------
    # Main diagnostics panel: traces + marginals + μ_t vs y
    # ------------------------------------------------------------------
    def plot_diagnostics(
        self,
        save_dir: Optional[str] = None,
        fname_prefix: str = "diagnostics",
        show: bool = True,
    ) -> None:
        kept = self.s.keep
        fig, axs = plt.subplots(3, 3, figsize=(13, 10))
        axs = axs.ravel()

        # trace: sigma
        if "sigma" in kept:
            axs[0].plot(kept["sigma"], lw=1)
            axs[0].set_title(r"trace: $\sigma$")
            if getattr(self.s, "true_sigma", None) is not None:
                axs[0].axhline(self.s.true_sigma, ls="--", lw=1.2, label=r"true $\sigma$")
                axs[0].legend()
        else:
            axs[0].axis("off")

        # trace: xi
        if "xi" in kept:
            axs[1].plot(kept["xi"], lw=1)
            axs[1].set_title(r"trace: $\xi$")
            if getattr(self.s, "true_xi", None) is not None:
                axs[1].axhline(self.s.true_xi, ls="--", lw=1.2, label=r"true $\xi$")
                axs[1].legend()
        else:
            axs[1].axis("off")

        # trace: process noise (pick one to show, prioritize Q_alpha)
        if "Q_alpha" in kept:
            axs[2].plot(kept["Q_alpha"], lw=1)
            axs[2].set_title(r"trace: $Q_\alpha$")
        elif "Q_beta" in kept:
            axs[2].plot(kept["Q_beta"], lw=1)
            axs[2].set_title(r"trace: $Q_\beta$")
        elif "Q_gamma" in kept:
            axs[2].plot(kept["Q_gamma"], lw=1)
            axs[2].set_title(r"trace: $Q_\gamma$")
        else:
            axs[2].axis("off")

        # posterior marginals: sigma
        if "sigma" in kept:
            axs[3].hist(kept["sigma"], bins=30, density=True)
            axs[3].set_title(r"posterior: $\sigma$")
            if getattr(self.s, "true_sigma", None) is not None:
                axs[3].axvline(self.s.true_sigma, ls="--", lw=1.5, label=r"true $\sigma$")
                axs[3].legend()
        else:
            axs[3].axis("off")

        # posterior marginals: xi
        if "xi" in kept:
            axs[4].hist(kept["xi"], bins=30, density=True)
            axs[4].set_title(r"posterior: $\xi$")
            if getattr(self.s, "true_xi", None) is not None:
                axs[4].axvline(self.s.true_xi, ls="--", lw=1.5, label=r"true $\xi$")
                axs[4].legend()
        else:
            axs[4].axis("off")

        # posterior: some Q
        if "Q_alpha" in kept:
            q = kept["Q_alpha"]
            axs[5].hist(q, bins=30, density=True)
            axs[5].set_title(r"posterior: $Q_\alpha$")
        elif "Q_beta" in kept:
            q = kept["Q_beta"]
            axs[5].hist(q, bins=30, density=True)
            axs[5].set_title(r"posterior: $Q_\beta$")
        elif "Q_gamma" in kept:
            q = kept["Q_gamma"]
            axs[5].hist(q, bins=30, density=True)
            axs[5].set_title(r"posterior: $Q_\gamma$")
        else:
            axs[5].axis("off")

        # acceptance summary
        axs[6].axis("off")
        lines = []
        if "sigma" in kept:
            lines.append(f"accept(logsigma) = {self._fmt_accept('logsigma')}")
        if "xi" in kept:
            lines.append(f"accept(xi) = {self._fmt_accept('xi')}")
        if "states" in getattr(self.s, "accept", {}):
            lines.append(f"accept(states/PGAS) ≈ {self._fmt_accept('states')}")
        if not lines:
            lines.append("no MH/RJ acceptance info available")
        axs[6].text(0.05, 0.95, "\n".join(lines), fontsize=12, va="top")

        # μ_t vs y
        if "mu" in kept:
            mu = kept["mu"]
            mu_med = np.median(mu, axis=0)
            mu_lo = np.quantile(mu, self.lo_q, axis=0)
            mu_hi = np.quantile(mu, self.hi_q, axis=0)
            t = np.arange(self.s.T)
            axs[7].plot(t, self.s.y, label=r"$y_t$", lw=1, alpha=0.6)
            axs[7].plot(t, mu_med, label=r"$\mu_t$ median", lw=1.5)
            axs[7].fill_between(t, mu_lo, mu_hi, alpha=0.25, label=self.band_label)
            true_mu = getattr(self.s, "true_mu_t", None)
            if isinstance(true_mu, np.ndarray) and true_mu.size == self.s.T:
                axs[7].plot(t, true_mu, lw=1.4, ls="--", label=r"true $\mu_t$")
            axs[7].set_title(r"Filtered location $\mu_t$ vs observations $y_t$")
            axs[7].legend()
        else:
            axs[7].axis("off")

        # free panel (optional extra: xi vs sigma scatter if present)
        if "sigma" in kept and "xi" in kept:
            axs[8].scatter(kept["sigma"], kept["xi"], s=4, alpha=0.4)
            axs[8].set_xlabel(r"$\sigma$")
            axs[8].set_ylabel(r"$\xi$")
            axs[8].set_title(r"Joint draws $(\sigma,\xi)$")
        else:
            axs[8].axis("off")

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

    # ------------------------------------------------------------------
    # Extra MCMC diagnostics for sigma, xi
    # ------------------------------------------------------------------
    def plot_mcmc_diagnostics_extra(
        self,
        max_lag: int = 40,
        save_dir: Optional[str] = None,
        fname_prefix: str = "mcmc_extra",
        show: bool = True,
    ) -> None:
        kept = self.s.keep
        s = kept.get("sigma", None)
        x = kept.get("xi", None)

        if s is None and x is None:
            print("[info] No sigma/xi in posterior; skipping extra MCMC diagnostics.")
            return

        fig, axs = plt.subplots(2, 3, figsize=(14, 8))

        # sigma row
        if s is not None:
            s = np.asarray(s, float)
            axs[0, 0].plot(np.cumsum(s) / np.arange(1, len(s) + 1))
            axs[0, 0].set_title(r"$\sigma$ running mean")

            ac_s = self._acf(s, max_lag=max_lag)
            axs[0, 1].stem(range(len(ac_s)), ac_s)
            axs[0, 1].set_title(r"$\sigma$ ACF")

            ess_s = self._ess(s, max_lag=5 * max_lag)
            gz_s = self._geweke_z(s)
            axs[0, 2].hist(s, bins=30, density=True)
            ttl = r"$\sigma$ hist" + f" (ESS≈{ess_s:.0f}, Geweke z≈{gz_s:.2f})"
            axs[0, 2].set_title(ttl)
            if getattr(self.s, "true_sigma", None) is not None:
                axs[0, 2].axvline(self.s.true_sigma, ls="--", lw=1.2, label=r"true $\sigma$")
                axs[0, 2].legend()
        else:
            for j in range(3):
                axs[0, j].axis("off")

        # xi row
        if x is not None:
            x = np.asarray(x, float)
            axs[1, 0].plot(np.cumsum(x) / np.arange(1, len(x) + 1))
            axs[1, 0].set_title(r"$\xi$ running mean")

            ac_x = self._acf(x, max_lag=max_lag)
            axs[1, 1].stem(range(len(ac_x)), ac_x)
            axs[1, 1].set_title(r"$\xi$ ACF")

            ess_x = self._ess(x, max_lag=5 * max_lag)
            gz_x = self._geweke_z(x)
            axs[1, 2].hist(x, bins=30, density=True)
            ttl = r"$\xi$ hist" + f" (ESS≈{ess_x:.0f}, Geweke z≈{gz_x:.2f})"
            axs[1, 2].set_title(ttl)
            if getattr(self.s, "true_xi", None) is not None:
                axs[1, 2].axvline(self.s.true_xi, ls="--", lw=1.2, label=r"true $\xi$")
                axs[1, 2].legend()
        else:
            for j in range(3):
                axs[1, j].axis("off")

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

    # ------------------------------------------------------------------
    # Process-noise diagnostics: Q_alpha, Q_beta, Q_gamma
    # ------------------------------------------------------------------
    def plot_process_noise_diagnostics(
        self,
        max_lag: int = 40,
        save_dir: Optional[str] = None,
        fname_prefix: str = "q_diagnostics",
        show: bool = True,
    ) -> None:
        kept = self.s.keep
        entries: List[Tuple[str, str]] = []  # (key, latex_label)

        if "Q_alpha" in kept:
            entries.append(("Q_alpha", r"$Q_\alpha$"))
        if "Q_beta" in kept:
            entries.append(("Q_beta", r"$Q_\beta$"))
        if "Q_gamma" in kept:
            entries.append(("Q_gamma", r"$Q_\gamma$"))

        if not entries:
            print("[info] No Q_alpha/Q_beta/Q_gamma in posterior; skipping Q diagnostics.")
            return

        R = len(entries)
        fig, axs = plt.subplots(R, 3, figsize=(14, 3.2 * R), squeeze=False)

        for i, (key, label) in enumerate(entries):
            q = np.asarray(kept[key], float)

            # Posterior
            axs[i, 0].hist(q, bins=40, density=True)
            axs[i, 0].set_title(fr"{label} posterior")

            # Running mean
            run = np.cumsum(q) / np.arange(1, q.size + 1)
            axs[i, 1].plot(run)
            axs[i, 1].set_title(fr"{label} running mean")

            # ACF
            ac = self._acf(q, max_lag=max_lag)
            axs[i, 2].stem(range(len(ac)), ac)
            axs[i, 2].set_title(fr"{label} ACF")

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

    # ------------------------------------------------------------------
    # Deterministic parameter diagnostics (m0_alpha, m0_beta, m0_cos/sin/nyq)
    # ------------------------------------------------------------------
    def plot_deterministic_param_diagnostics(
        self,
        max_lag: int = 40,
        season_k: int = 6,
        save_dir: Optional[str] = None,
        fname_prefix: str = "deterministic_diagnostics",
        show: bool = True,
    ) -> None:
        kept = self.s.keep
        entries_scalar: List[Tuple[str, np.ndarray, str]] = []
        entries_matrix: List[Tuple[str, np.ndarray, str]] = []

        if "m0_alpha" in kept:
            entries_scalar.append(("m0_alpha", np.asarray(kept["m0_alpha"], float), r"$m_{0,\alpha}$"))
        if "m0_beta" in kept:
            entries_scalar.append(("m0_beta", np.asarray(kept["m0_beta"], float), r"$m_{0,\beta}$"))

        if "m0_cos" in kept:
            entries_matrix.append(("m0_cos", np.asarray(kept["m0_cos"], float), r"$m_{0,\cos,k}$"))
        if "m0_sin" in kept:
            entries_matrix.append(("m0_sin", np.asarray(kept["m0_sin"], float), r"$m_{0,\sin,k}$"))
        if "m0_nyq" in kept:
            entries_scalar.append(("m0_nyq", np.asarray(kept["m0_nyq"], float), r"$m_{0,\text{nyq}}$"))

        if not entries_scalar and not entries_matrix:
            print("[info] No deterministic hyperparameters (m0_*) found; skipping deterministic diagnostics.")
            return

        # Count rows: one per scalar + up to season_k per matrix entry
        R = len(entries_scalar)
        for _, mat, _ in entries_matrix:
            K = mat.shape[1]
            R += min(season_k, K)

        fig, axs = plt.subplots(R, 4, figsize=(16, 3.0 * R), squeeze=False)

        row = 0
        # Scalars
        for key, series, label in entries_scalar:
            series = np.asarray(series, float)
            axs[row, 0].plot(series, lw=1.0)
            axs[row, 0].set_title(f"{label} trace")

            run = np.cumsum(series) / np.arange(1, series.size + 1)
            axs[row, 1].plot(run, lw=1.0)
            axs[row, 1].set_title(f"{label} running mean")

            ac = self._acf(series, max_lag=max_lag)
            axs[row, 2].stem(range(len(ac)), ac)
            axs[row, 2].set_title(f"{label} ACF")

            axs[row, 3].hist(series, bins=40, density=True)
            axs[row, 3].set_title(f"{label} posterior")

            row += 1

        # Matrix entries (cos/sin per harmonic)
        for key, mat, label in entries_matrix:
            mat = np.asarray(mat, float)  # (N, K)
            K = mat.shape[1]
            kmax = min(season_k, K)
            for k in range(kmax):
                series = mat[:, k]
                lbl = f"{label}, k={k+1}"

                axs[row, 0].plot(series, lw=1.0)
                axs[row, 0].set_title(f"{lbl} trace")

                run = np.cumsum(series) / np.arange(1, series.size + 1)
                axs[row, 1].plot(run, lw=1.0)
                axs[row, 1].set_title(f"{lbl} running mean")

                ac = self._acf(series, max_lag=max_lag)
                axs[row, 2].stem(range(len(ac)), ac)
                axs[row, 2].set_title(f"{lbl} ACF")

                axs[row, 3].hist(series, bins=40, density=True)
                axs[row, 3].set_title(f"{lbl} posterior")

                row += 1

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

    # ------------------------------------------------------------------
    # Stacked states + observations panel (μ_t, α_t, β_t, γ_t)
    # ------------------------------------------------------------------
    def plot_states_and_observations(
        self,
        save_dir: Optional[str] = None,
        fname_prefix: str = "states",
        show: bool = True,
    ) -> None:
        kept = self.s.keep
        if "mu" not in kept:
            print("[warn] No 'mu' in draws; skipping states/observations plot.")
            return

        mu = kept["mu"]
        mu_med = np.median(mu, axis=0)
        mu_lo = np.quantile(mu, self.lo_q, axis=0)
        mu_hi = np.quantile(mu, self.hi_q, axis=0)

        n_rows = 1
        if getattr(self.s, "include_level", False) and "alpha_t" in kept:
            n_rows += 1
        if getattr(self.s, "include_trend", False) and (
            "beta_t" in kept or "beta_value" in kept
        ):
            n_rows += 1
        if getattr(self.s, "include_seasonality", False) and (
            "gamma_t" in kept
        ):
            n_rows += 1

        fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3.0 * n_rows), sharex=True)
        if n_rows == 1:
            axes = [axes]
        t = np.arange(self.s.T)

        # (1) y vs μ
        ax = axes[0]
        ax.plot(t, self.s.y, label=r"$y_t$", lw=1, alpha=0.7)
        ax.plot(t, mu_med, label=r"$\mu_t$ median", lw=1.6)
        ax.fill_between(t, mu_lo, mu_hi, alpha=0.25, label=self.band_label)
        true_mu = getattr(self.s, "true_mu_t", None)
        if isinstance(true_mu, np.ndarray) and true_mu.size == self.s.T:
            ax.plot(t, true_mu, lw=1.4, ls="--", label=r"true $\mu_t$")
        ax.set_title(r"Data and posterior location $\mu_t$")
        ax.legend(loc="best")

        row = 1

        # α_t
        if getattr(self.s, "include_level", False) and "alpha_t" in kept:
            at = kept["alpha_t"]
            a_med = np.median(at, axis=0)
            a_lo = np.quantile(at, self.lo_q, axis=0)
            a_hi = np.quantile(at, self.hi_q, axis=0)
            ax = axes[row]
            ax.plot(t, a_med, lw=1.6, label=r"$\alpha_t$ median")
            ax.fill_between(t, a_lo, a_hi, alpha=0.25, label=self.band_label)
            true_alpha = getattr(self.s, "true_alpha_t", None)
            if isinstance(true_alpha, np.ndarray) and true_alpha.size == self.s.T:
                ax.plot(t, true_alpha, lw=1.4, ls="--", label=r"true $\alpha_t$")
            ax.set_title(r"Level component $\alpha_t$")
            ax.legend(loc="best")
            row += 1

        # β_t or deterministic β
        if getattr(self.s, "include_trend", False):
            if "beta_t" in kept:
                bt = kept["beta_t"]
                b_med = np.median(bt, axis=0)
                b_lo = np.quantile(bt, self.lo_q, axis=0)
                b_hi = np.quantile(bt, self.hi_q, axis=0)
                ax = axes[row]
                ax.plot(t, b_med, lw=1.6, label=r"$\beta_t$ median")
                ax.fill_between(t, b_lo, b_hi, alpha=0.25, label=self.band_label)
                true_beta = getattr(self.s, "true_beta_t", None)
                if isinstance(true_beta, np.ndarray) and true_beta.size == self.s.T:
                    ax.plot(t, true_beta, lw=1.4, ls="--", label=r"true $\beta_t$")
                ax.set_title(r"Trend component $\beta_t$")
                ax.legend(loc="best")
                row += 1
            elif "beta_value" in kept:
                vals = np.asarray(kept["beta_value"], float)
                bv = vals[:, None] * np.ones((1, self.s.T))
                b_med = np.median(bv, axis=0)
                b_lo = np.quantile(bv, self.lo_q, axis=0)
                b_hi = np.quantile(bv, self.hi_q, axis=0)
                ax = axes[row]
                ax.plot(t, b_med, lw=1.6, label=r"$\beta$ (det.) median")
                ax.fill_between(t, b_lo, b_hi, alpha=0.25, label=self.band_label)
                ax.set_title(r"Deterministic slope $\beta$")
                ax.legend(loc="best")
                row += 1

        # γ_t
        if getattr(self.s, "include_seasonality", False) and "gamma_t" in kept:
            gt = kept["gamma_t"]
            g_med = np.median(gt, axis=0)
            g_lo = np.quantile(gt, self.lo_q, axis=0)
            g_hi = np.quantile(gt, self.hi_q, axis=0)
            ax = axes[row]
            ax.plot(t, g_med, lw=1.6, label=r"$\gamma_t$ median")
            ax.fill_between(t, g_lo, g_hi, alpha=0.25, label=self.band_label)
            true_gamma = getattr(self.s, "true_gamma_t", None)
            if isinstance(true_gamma, np.ndarray) and true_gamma.size == self.s.T:
                ax.plot(t, true_gamma, lw=1.4, ls="--", label=r"true $\gamma_t$")
            ax.set_title(r"Seasonal component $\gamma_t$")
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

    # ------------------------------------------------------------------
    # Per-component figures (μ, α, β, γ)
    # ------------------------------------------------------------------
    def plot_components_separately(
        self,
        save_dir: Optional[str] = None,
        fname_prefix: str = "components",
        center: str = "median",
        map_bins: int = 50,
        show: bool = True,
    ) -> None:
        kept = self.s.keep

        # μ_t
        if "mu" in kept:
            self._plot_component_figure(
                kept["mu"],
                title=r"Posterior location $\mu_t$",
                ylabel=r"$\mu_t$",
                true_series=getattr(self.s, "true_mu_t", None),
                save_dir=save_dir,
                fname_prefix=f"{fname_prefix}_mu",
                center=center,
                map_bins=map_bins,
                show=show,
            )

        # α_t
        if getattr(self.s, "include_level", False) and "alpha_t" in kept:
            self._plot_component_figure(
                kept["alpha_t"],
                title=r"Posterior level $\alpha_t$",
                ylabel=r"$\alpha_t$",
                true_series=getattr(self.s, "true_alpha_t", None),
                save_dir=save_dir,
                fname_prefix=f"{fname_prefix}_alpha",
                center=center,
                map_bins=map_bins,
                show=show,
            )

        # β_t or deterministic β
        if getattr(self.s, "include_trend", False):
            if "beta_t" in kept:
                self._plot_component_figure(
                    kept["beta_t"],
                    title=r"Posterior trend $\beta_t$",
                    ylabel=r"$\beta_t$",
                    true_series=getattr(self.s, "true_beta_t", None),
                    save_dir=save_dir,
                    fname_prefix=f"{fname_prefix}_beta",
                    center=center,
                    map_bins=map_bins,
                    show=show,
                )
            elif "beta_value" in kept:
                vals = np.asarray(kept["beta_value"], float)
                bv = vals[:, None] * np.ones((1, self.s.T))
                self._plot_component_figure(
                    bv,
                    title=r"Posterior deterministic slope $\beta$",
                    ylabel=r"$\beta$",
                    true_series=None,
                    save_dir=save_dir,
                    fname_prefix=f"{fname_prefix}_beta_det",
                    center=center,
                    map_bins=map_bins,
                    show=show,
                )

        # γ_t
        if getattr(self.s, "include_seasonality", False) and "gamma_t" in kept:
            self._plot_component_figure(
                kept["gamma_t"],
                title=r"Posterior seasonal contribution $\gamma_t$",
                ylabel=r"$\gamma_t$",
                true_series=getattr(self.s, "true_gamma_t", None),
                save_dir=save_dir,
                fname_prefix=f"{fname_prefix}_gamma",
                center=center,
                map_bins=map_bins,
                show=show,
            )

    # ------------------------------------------------------------------
    # Quick little hist panel (σ, ξ, maybe Q_alpha)
    # ------------------------------------------------------------------
    def plot_quick_hist_panel(
        self,
        posterior: Dict[str, np.ndarray],
        save_dir: Optional[str] = None,
        fname_prefix: str = "quick_hist",
        show: bool = True,
    ) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        if "sigma" in posterior:
            axes[0].hist(posterior["sigma"], bins=40, density=True)
            axes[0].set_title(r"$\sigma | y$")
        else:
            axes[0].axis("off")

        if "xi" in posterior:
            axes[1].hist(posterior["xi"], bins=40, density=True)
            axes[1].set_title(r"$\xi | y$")
        else:
            axes[1].axis("off")

        if "Q_alpha" in posterior:
            axes[2].hist(posterior["Q_alpha"], bins=40, density=True)
            axes[2].set_title(r"$Q_\alpha | y$")
        elif "Q_beta" in posterior:
            axes[2].hist(posterior["Q_beta"], bins=40, density=True)
            axes[2].set_title(r"$Q_\beta | y$")
        elif "Q_gamma" in posterior:
            axes[2].hist(posterior["Q_gamma"], bins=40, density=True)
            axes[2].set_title(r"$Q_\gamma | y$")
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
# CLI: load posterior and plot
# ----------------------------- #
if __name__ == "__main__":
    import sys
    import argparse

    # Allow importing from project root
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from optimization.posterior_bundle import load_posterior, find_latest_run

    parser = argparse.ArgumentParser(description="Plot harmonic structural DGEV posterior diagnostics.")
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Path to a run directory (containing posterior.npz) or to a posterior.npz file. "
             "If omitted, the script searches for the latest run under --root.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="results/simulations/DGEV_harm",
        help="Root folder to search when --run is not given.",
    )
    parser.add_argument("--level", type=float, default=0.90, help="Credible interval level for bands.")
    parser.add_argument(
        "--center",
        type=str,
        default="median",
        choices=["median", "mean", "map"],
        help="Center line for per-component plots.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively in addition to saving.",
    )
    parser.add_argument(
        "--skip-states",
        action="store_true",
        help="Skip stacked states/observations panel.",
    )
    parser.add_argument(
        "--skip-separate",
        action="store_true",
        help="Skip separate component figures.",
    )
    parser.add_argument("--map-bins", type=int, default=60, help="Bins for fast per-time MAP estimation.")
    parser.add_argument("--max-lag", type=int, default=40, help="Max lag for ACF plots.")
    parser.add_argument("--season-k", type=int, default=6, help="Number of harmonic coefficients to show in m0 diagnostics.")
    parser.add_argument(
        "--run-default",
        type=str,
        default=None,
        help="Optional default run path to use if search fails.",
    )

    args = parser.parse_args()

    # Resolve run
    target = args.run
    if not target:
        print(f"[info] Searching latest run under: {args.root}")
        target = find_latest_run(root=args.root)
        if not target and args.run_default:
            target = args.run_default

    if not target:
        print("[error] No runs found. Provide --run or ensure results exist under --root.")
        sys.exit(1)

    print(f"[info] Using run: {target}")

    bundle = load_posterior(target)
    draws = bundle.draws
    meta = bundle.meta

    # Build sampler-like object
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
        # Fallback
        s.T = int(draws.get("y", np.array([np.nan])).shape[0])

    # y: use if saved, else NaNs
    y_draw = draws.get("y", None)
    if y_draw is not None and len(y_draw) == s.T:
        s.y = np.asarray(y_draw, float)
    else:
        s.y = np.full(s.T, np.nan, dtype=float)

    # Component presence flags
    s.include_level = bool(meta.get("include_level", "alpha_t" in draws))
    s.include_trend = bool(meta.get("include_trend", ("beta_t" in draws) or ("beta_value" in draws)))
    s.include_seasonality = bool(meta.get("include_seasonality", "gamma_t" in draws))

    # Truth overlays
    s.true_sigma = meta.get("true_sigma", None)
    s.true_xi = meta.get("true_xi", None)
    s.true_mu_t = draws.get("true_mu_t", None)
    s.true_alpha_t = draws.get("true_alpha_t", None)
    s.true_beta_t = draws.get("true_beta_t", None)
    s.true_gamma_t = draws.get("true_gamma_t", None)

    # MCMC bookkeeping
    s.accept = meta.get("accept", {})
    s.proposals = meta.get("proposals", {})

    # Save directory
    base_dir = os.path.dirname(bundle.npz_path)
    save_dir = os.path.join(base_dir, "figures")
    _ensure_dir(save_dir)
    print(f"[info] Saving figures to: {save_dir}")

    plotter = DGEVPlotter(s, level=float(args.level))

    # Make plots
    plotter.plot_diagnostics(save_dir=save_dir, fname_prefix="diagnostics", show=args.show)
    plotter.plot_mcmc_diagnostics_extra(
        max_lag=args.max_lag,
        save_dir=save_dir,
        fname_prefix="mcmc_extra",
        show=args.show,
    )
    plotter.plot_process_noise_diagnostics(
        max_lag=args.max_lag,
        save_dir=save_dir,
        fname_prefix="q_diagnostics",
        show=args.show,
    )
    plotter.plot_deterministic_param_diagnostics(
        max_lag=args.max_lag,
        season_k=args.season_k,
        save_dir=save_dir,
        fname_prefix="deterministic_diagnostics",
        show=args.show,
    )

    if ("mu" in draws) and (not args.skip_states):
        plotter.plot_states_and_observations(
            save_dir=save_dir,
            fname_prefix="states",
            show=args.show,
        )
    else:
        print("[info] Skipping states/observations panel (no 'mu' in draws or --skip-states).")

    if not args.skip_separate:
        plotter.plot_components_separately(
            save_dir=save_dir,
            fname_prefix="components",
            center=str(args.center),
            map_bins=int(args.map_bins),
            show=args.show,
        )

    plotter.plot_quick_hist_panel(
        draws,
        save_dir=save_dir,
        fname_prefix="quick_hist",
        show=args.show,
    )
    print("[done] Plots generated.")
