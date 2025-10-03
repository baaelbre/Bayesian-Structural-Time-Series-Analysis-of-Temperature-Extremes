# %% simulator/dlm_plotter.py
import os
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import json


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# =========================
# Small MCMC utilities
# =========================
def _acf(x, max_lag=40):
    x = np.asarray(x, float)
    n = x.size
    if n <= 1:
        return np.array([1.0 if n == 1 else np.nan])
    x = x - np.mean(x)
    denom = float(np.dot(x, x)) + 1e-300
    L = int(min(max_lag, n - 1))
    ac = np.empty(L + 1, dtype=float)
    for k in range(L + 1):
        ac[k] = float(np.dot(x[: n - k], x[k:])) / denom
    return ac


def _ess(x, max_lag=100):
    ac = _acf(x, max_lag=max_lag)
    if not np.all(np.isfinite(ac)) or ac.size <= 1:
        return float(len(x))
    s = 0.0
    for k in range(1, ac.size):
        if ac[k] <= 0:
            break
        s += 2.0 * ac[k]
    n = len(x)
    return float(n) / (1.0 + s)


def _geweke_z(x, first_frac=0.1, last_frac=0.5):
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


# =========================
# Plotter (aligned to conjugate DLM Gibbs outputs)
# =========================
class DLMPlotter:
    """
    Plotting helper for the conjugate Gaussian DLM Gibbs sampler.

    Expected bundle:
      draws: saved arrays; keys include:
             - "mu" (n_keep, T), "sigma2" (n_keep,)
             - "q_alpha","q_beta","q_gamma" (each optional, n_keep,)
             - "x" (n_keep, T, dim) optional time paths (packed as in sampler) OR
               separate "alpha_t","beta_t","gamma_t" (n_keep, T)
             - deterministic: "level_value","slope_value","season_vector" (n_keep,*)
             - "y", optional truths: "true_mu_t","true_alpha_t","true_beta_t","true_gamma_t"
      meta:  dict with at least {"T","period", and either "modes" string "lev-trend-seas"
             or "modes": {"level_mode","trend_mode","seasonal_mode"}}.
    """

    def __init__(self, draws: Dict[str, np.ndarray], meta: Dict[str, Any], level: float = 0.90):
        self.draws = draws
        self.meta = meta
        self.level = float(level)
        if not (0 < self.level < 1):
            raise ValueError("level must be in (0,1).")

        # Core dims
        self.T = int(meta.get("T") or draws["mu"].shape[1])
        self.period = int(meta.get("period", 12))

        # Modes: robust to string tag or dict
        modes = meta.get("modes", None)
        if isinstance(modes, str):
            toks = modes.split("-")
            if len(toks) == 3:
                self.level_mode, self.trend_mode, self.season_mode = toks
            else:
                # fallback to dict keys if malformed
                mm = meta.get("modes", {})
                self.level_mode = mm.get("level_mode", "dynamic")
                self.trend_mode = mm.get("trend_mode", "none")
                self.season_mode = mm.get("seasonal_mode", "none")
        elif isinstance(modes, dict):
            self.level_mode = modes.get("level_mode", "dynamic")
            self.trend_mode = modes.get("trend_mode", "none")
            self.season_mode = modes.get("seasonal_mode", "none")
        else:
            # ultimate fallback
            self.level_mode, self.trend_mode, self.season_mode = "dynamic", "none", "none"

        # Optional truth overlays
        self.true_mu    = draws.get("true_mu_t")
        self.true_alpha = draws.get("true_alpha_t")
        self.true_beta  = draws.get("true_beta_t")
        self.true_gamma = draws.get("true_gamma_t")

        # Optional data
        self.y = draws.get("y", None)

        # Quantiles for bands
        self.lo_q = (1.0 - self.level) / 2.0
        self.hi_q = 1.0 - self.lo_q
        self.band_label = f"{int(round(self.level * 100))}% band"

        # Dynamic-state indexing if a packed 'x' is present
        self.has_x = ("x" in draws) and (draws["x"].ndim == 3)
        self.idx_alpha = self.idx_beta = self.idx_gamma_end = None
        if self.has_x:
            dim = draws["x"].shape[2]
            i_alpha = 0 if self.level_mode == "dynamic" else None
            i_beta = None
            if self.trend_mode == "dynamic":
                i_beta = (1 if i_alpha is not None else 0)
            i_g0 = i_gL = None
            if self.season_mode == "dynamic":
                start = (1 if i_alpha is not None else 0) + (1 if i_beta is not None else 0)
                if (self.period - 1) > 0:
                    i_g0 = start
                    i_gL = start + (self.period - 2)
                if dim != start + (self.period - 1):
                    i_g0 = i_gL = None
            self.idx_alpha = i_alpha
            self.idx_beta = i_beta
            self.idx_gamma_end = i_gL

    # ---------- helpers ----------
    def _summarize_paths(self, draws_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.asarray(draws_2d, float)  # (n_keep, T)
        ctr = np.median(X, axis=0)
        lo = np.quantile(X, self.lo_q, axis=0)
        hi = np.quantile(X, self.hi_q, axis=0)
        return ctr, lo, hi

    def _tile_static_to_T(self, values_1d: np.ndarray) -> np.ndarray:
        v = np.asarray(values_1d, float)[:, None]
        return v * np.ones((1, self.T), dtype=float)

    def _season_ts_from_draws(self) -> Optional[np.ndarray]:
        """Build deterministic seasonal time series (n_keep, T) from per-draw 'season_vector' (n_keep, p)."""
        if self.season_mode != "deterministic":
            return None
        Seas = self.draws.get("season_vector", None)
        if Seas is None:
            return None
        Seas = np.asarray(Seas, float)  # (n_keep, p)
        n_keep, p = Seas.shape
        reps = int(np.ceil(self.T / p))
        tiled = np.tile(Seas, reps)[:, : self.T]  # repeat entire pattern
        return tiled

    def _get_alpha_draws(self) -> Optional[np.ndarray]:
        if self.level_mode != "dynamic":
            return None
        if self.has_x and self.idx_alpha is not None:
            return self.draws["x"][:, :, self.idx_alpha]
        return self.draws.get("alpha_t", None)

    def _get_beta_draws(self) -> Optional[np.ndarray]:
        if self.trend_mode != "dynamic":
            return None
        if self.has_x and self.idx_beta is not None:
            return self.draws["x"][:, :, self.idx_beta]
        return self.draws.get("beta_t", None)

    def _get_gamma_draws(self) -> Optional[np.ndarray]:
        if self.season_mode != "dynamic":
            return None
        if self.has_x and self.idx_gamma_end is not None:
            return self.draws["x"][:, :, self.idx_gamma_end]
        return self.draws.get("gamma_t", None)

    def _get_qs(self) -> List[Tuple[str, np.ndarray, str]]:
        """Collect any available process-variance posteriors."""
        out: List[Tuple[str, np.ndarray, str]] = []
        if ("q_alpha" in self.draws) and self.level_mode == "dynamic":
            out.append(("q_alpha", np.asarray(self.draws["q_alpha"], float), r"$q_\alpha$"))
        if ("q_beta"  in self.draws) and self.trend_mode == "dynamic":
            out.append(("q_beta",  np.asarray(self.draws["q_beta"],  float), r"$q_\beta$"))
        if ("q_gamma" in self.draws) and self.season_mode == "dynamic":
            out.append(("q_gamma", np.asarray(self.draws["q_gamma"], float), r"$q_\gamma$"))
        return out

    # ---------- figures ----------
    def plot_diagnostics(self, save_dir: Optional[str] = None, fname_prefix: str = "diagnostics", show: bool = True):
        """
        Panels:
          [0] trace σ
          [1] trace for available q_* (overlaid)
          [2] μ vs y with band (and true μ, if provided)
          [3] posterior σ
          [4] posterior q_* (overlaid, with legend)
          [5] running mean of σ
        """
        n_keep = self.draws["mu"].shape[0]
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))
        axs = axs.ravel()

        # σ traces/hist
        s2 = np.asarray(self.draws.get("sigma2", np.array([])))
        if s2.size > 0:
            sig = np.sqrt(np.clip(s2, 0, None))
            axs[0].plot(sig, lw=1)
            axs[0].set_title(r"trace: $\sigma$")
            axs[3].hist(sig, bins=30, density=True)
            axs[3].set_title(r"posterior: $\sigma$")
            ess_s = _ess(sig, max_lag=200)
            gz_s = _geweke_z(sig)
            axs[3].set_xlabel(f"ESS≈{ess_s:.0f}, Geweke z≈{gz_s:.2f}")
        else:
            axs[0].axis("off"); axs[3].axis("off")

        # q_* traces/hists (overlaid)
        qs = self._get_qs()
        if qs:
            axs[1].set_title("trace: process variances")
            for key, arr, lab in qs:
                axs[1].plot(arr, lw=1, alpha=0.9, label=lab)
            axs[1].legend()

            axs[4].set_title("posterior: process variances")
            for key, arr, lab in qs:
                axs[4].hist(arr[np.isfinite(arr)], bins=30, density=True, alpha=0.65, label=lab)
            axs[4].legend()
        else:
            axs[1].axis("off"); axs[4].axis("off")

        # Posterior μ vs y
        mu = self.draws["mu"]  # (n_keep, T)
        mu_ctr, mu_lo, mu_hi = self._summarize_paths(mu)
        axs[2].plot(mu_ctr, lw=1.5, label=r"$\mu_t$ median")
        axs[2].fill_between(np.arange(self.T), mu_lo, mu_hi, alpha=0.25, label=self.band_label)
        if self.y is not None and len(self.y) == self.T:
            axs[2].plot(self.y, lw=1, alpha=0.6, label=r"$y_t$")
        if (self.true_mu is not None) and len(self.true_mu) == self.T:
            axs[2].plot(self.true_mu, ls="--", lw=1.2, label=r"true $\mu_t$")
        axs[2].set_title(r"$\mu_t$ vs data")
        axs[2].legend()

        # Running mean of σ
        if s2.size > 0:
            axs[5].plot(np.cumsum(sig) / np.arange(1, sig.size + 1))
            axs[5].set_title(r"$\sigma$ running mean")
        else:
            axs[5].axis("off")

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

    def plot_states_and_observations(self, save_dir: Optional[str] = None, fname_prefix: str = "states", show: bool = True):
        rows = 1  # mu vs y
        if self.level_mode == "dynamic" and (self._get_alpha_draws() is not None): rows += 1
        if self.trend_mode == "dynamic" and (self._get_beta_draws()  is not None): rows += 1
        if self.season_mode == "dynamic" and (self._get_gamma_draws() is not None): rows += 1
        if self.trend_mode == "deterministic" and ("slope_value" in self.draws): rows += 1
        if self.season_mode == "deterministic" and ("season_vector" in self.draws): rows += 1

        fig, axes = plt.subplots(rows, 1, figsize=(12, 3.0 * rows), sharex=True)
        if rows == 1:
            axes = [axes]
        r = 0

        # μ vs y
        mu = self.draws["mu"]
        mu_ctr, mu_lo, mu_hi = self._summarize_paths(mu)
        ax = axes[r]
        if self.y is not None and len(self.y) == self.T:
            ax.plot(self.y, lw=1, alpha=0.7, label=r"$y_t$")
        ax.plot(mu_ctr, lw=1.6, label=r"$\mu_t$ median")
        ax.fill_between(np.arange(self.T), mu_lo, mu_hi, alpha=0.25, label=self.band_label)
        if (self.true_mu is not None) and len(self.true_mu) == self.T:
            ax.plot(self.true_mu, lw=1.2, ls="--", label=r"true $\mu_t$")
        ax.set_title("Data and posterior $\mu_t$")
        ax.legend()
        r += 1

        # α_t (dynamic)
        a_draws = self._get_alpha_draws()
        if a_draws is not None:
            a_ctr, a_lo, a_hi = self._summarize_paths(a_draws)
            ax = axes[r]
            ax.plot(a_ctr, lw=1.6, label=r"$\alpha_t$ median")
            ax.fill_between(np.arange(self.T), a_lo, a_hi, alpha=0.25, label=self.band_label)
            if (self.true_alpha is not None) and len(self.true_alpha) == self.T:
                ax.plot(self.true_alpha, lw=1.2, ls="--", label=r"true $\alpha_t$")
            ax.set_title(r"Level component $\alpha_t$")
            ax.legend(); r += 1

        # β_t (dynamic) or slope (deterministic)
        b_draws = self._get_beta_draws()
        if b_draws is not None:
            b_ctr, b_lo, b_hi = self._summarize_paths(b_draws)
            ax = axes[r]
            ax.plot(b_ctr, lw=1.6, label=r"$\beta_t$ median")
            ax.fill_between(np.arange(self.T), b_lo, b_hi, alpha=0.25, label=self.band_label)
            if (self.true_beta is not None) and len(self.true_beta) == self.T:
                ax.plot(self.true_beta, lw=1.2, ls="--", label=r"true $\beta_t$")
            ax.set_title(r"Trend component $\beta_t$")
            ax.legend(); r += 1
        elif self.trend_mode == "deterministic" and ("slope_value" in self.draws):
            slope_draws = np.asarray(self.draws["slope_value"], float)  # (n_keep,)
            slope_ts = self._tile_static_to_T(slope_draws)
            s_ctr, s_lo, s_hi = self._summarize_paths(slope_ts)
            ax = axes[r]
            ax.plot(s_ctr, lw=1.6, label=r"slope (det.) median")
            ax.fill_between(np.arange(self.T), s_lo, s_hi, alpha=0.25, label=self.band_label)
            ax.set_title("Deterministic slope")
            ax.legend(); r += 1

        # γ_t (dynamic) or deterministic seasonal profile
        g_draws = self._get_gamma_draws()
        if g_draws is not None:
            g_ctr, g_lo, g_hi = self._summarize_paths(g_draws)
            ax = axes[r]
            ax.plot(g_ctr, lw=1.6, label=r"$\gamma_t$ median")
            ax.fill_between(np.arange(self.T), g_lo, g_hi, alpha=0.25, label=self.band_label)
            if (self.true_gamma is not None) and len(self.true_gamma) == self.T:
                ax.plot(self.true_gamma, lw=1.2, ls="--", label=r"true $\gamma_t$")
            ax.set_title("Seasonal last-coordinate $\gamma_t$")
            ax.legend()
        elif self.season_mode == "deterministic" and ("season_vector" in self.draws):
            seas_ts = self._season_ts_from_draws()
            if seas_ts is not None:
                s_ctr, s_lo, s_hi = self._summarize_paths(seas_ts)
                ax = axes[r]
                ax.plot(s_ctr, lw=1.6, label="season (det.) median")
                ax.fill_between(np.arange(self.T), s_lo, s_hi, alpha=0.25, label=self.band_label)
                ax.set_title("Deterministic seasonality")
                ax.legend()

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

    def plot_components_separately(self, save_dir: Optional[str] = None, fname_prefix: str = "components", show: bool = True):
        # μ_t
        mu = self.draws["mu"]
        mu_ctr, mu_lo, mu_hi = self._summarize_paths(mu)
        fig, ax = plt.subplots(figsize=(12, 3.2))
        ax.plot(mu_ctr, lw=1.8, label=r"$\mu_t$ median")
        ax.fill_between(np.arange(self.T), mu_lo, mu_hi, alpha=0.25, label=self.band_label)
        if (self.true_mu is not None) and len(self.true_mu) == self.T:
            ax.plot(self.true_mu, lw=1.2, ls="--", label=r"true $\mu_t$")
        ax.set_title(r"Posterior $\mu_t$")
        ax.legend()
        plt.tight_layout()
        if save_dir:
            _ensure_dir(save_dir)
            out_path = os.path.join(save_dir, f"{fname_prefix}_mu.png")
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            print(f"[save] Figure -> {out_path}")
        if show: plt.show()
        else: plt.close(fig)

        # α_t
        a_draws = self._get_alpha_draws()
        if a_draws is not None:
            a_ctr, a_lo, a_hi = self._summarize_paths(a_draws)
            fig, ax = plt.subplots(figsize=(12, 3.2))
            ax.plot(a_ctr, lw=1.8, label=r"$\alpha_t$ median")
            ax.fill_between(np.arange(self.T), a_lo, a_hi, alpha=0.25, label=self.band_label)
            if (self.true_alpha is not None) and len(self.true_alpha) == self.T:
                ax.plot(self.true_alpha, lw=1.2, ls="--", label=r"true $\alpha_t$")
            ax.set_title(r"Posterior level $\alpha_t$")
            ax.legend()
            plt.tight_layout()
            if save_dir:
                out_path = os.path.join(save_dir, f"{fname_prefix}_alpha.png")
                fig.savefig(out_path, dpi=200, bbox_inches="tight")
                print(f"[save] Figure -> {out_path}")
            if show: plt.show()
            else: plt.close(fig)

        # β
        b_draws = self._get_beta_draws()
        if b_draws is not None:
            b_ctr, b_lo, b_hi = self._summarize_paths(b_draws)
            fig, ax = plt.subplots(figsize=(12, 3.2))
            ax.plot(b_ctr, lw=1.8, label=r"$\beta_t$ median")
            ax.fill_between(np.arange(self.T), b_lo, b_hi, alpha=0.25, label=self.band_label)
            if (self.true_beta is not None) and len(self.true_beta) == self.T:
                ax.plot(self.true_beta, lw=1.2, ls="--", label=r"true $\beta_t$")
            ax.set_title(r"Posterior trend $\beta_t$")
            ax.legend()
            plt.tight_layout()
            if save_dir:
                out_path = os.path.join(save_dir, f"{fname_prefix}_beta.png")
                fig.savefig(out_path, dpi=200, bbox_inches="tight")
                print(f"[save] Figure -> {out_path}")
            if show: plt.show()
            else: plt.close(fig)
        elif self.trend_mode == "deterministic" and ("slope_value" in self.draws):
            slope_draws = np.asarray(self.draws["slope_value"], float)
            slope_ts = self._tile_static_to_T(slope_draws)
            s_ctr, s_lo, s_hi = self._summarize_paths(slope_ts)
            fig, ax = plt.subplots(figsize=(12, 3.2))
            ax.plot(s_ctr, lw=1.8, label=r"$\beta$ (det.) median")
            ax.fill_between(np.arange(self.T), s_lo, s_hi, alpha=0.25, label=self.band_label)
            ax.set_title(r"Deterministic slope")
            ax.legend()
            plt.tight_layout()
            if save_dir:
                out_path = os.path.join(save_dir, f"{fname_prefix}_beta_det.png")
                fig.savefig(out_path, dpi=200, bbox_inches="tight")
                print(f"[save] Figure -> {out_path}")
            if show: plt.show()
            else: plt.close(fig)

        # γ
        g_draws = self._get_gamma_draws()
        if g_draws is not None:
            g_ctr, g_lo, g_hi = self._summarize_paths(g_draws)
            fig, ax = plt.subplots(figsize=(12, 3.2))
            ax.plot(g_ctr, lw=1.8, label=r"$\gamma_t$ median")
            ax.fill_between(np.arange(self.T), g_lo, g_hi, alpha=0.25, label=self.band_label)
            if (self.true_gamma is not None) and len(self.true_gamma) == self.T:
                ax.plot(self.true_gamma, lw=1.2, ls="--", label=r"true $\gamma_t$")
            ax.set_title(r"Posterior seasonal last coord $\gamma_t$")
            ax.legend()
            plt.tight_layout()
            if save_dir:
                out_path = os.path.join(save_dir, f"{fname_prefix}_gamma.png")
                fig.savefig(out_path, dpi=200, bbox_inches="tight")
                print(f"[save] Figure -> {out_path}")
            if show: plt.show()
            else: plt.close(fig)
        elif self.season_mode == "deterministic" and ("season_vector" in self.draws):
            seas_ts = self._season_ts_from_draws()
            if seas_ts is not None:
                s_ctr, s_lo, s_hi = self._summarize_paths(seas_ts)
                fig, ax = plt.subplots(figsize=(12, 3.2))
                ax.plot(s_ctr, lw=1.8, label="season (det.) median")
                ax.fill_between(np.arange(self.T), s_lo, s_hi, alpha=0.25, label=self.band_label)
                ax.set_title("Deterministic seasonality")
                ax.legend()
                plt.tight_layout()
                if save_dir:
                    out_path = os.path.join(save_dir, f"{fname_prefix}_season_det.png")
                    fig.savefig(out_path, dpi=200, bbox_inches="tight")
                    print(f"[save] Figure -> {out_path}")
                if show: plt.show()
                else: plt.close(fig)

    def plot_quick_hist_panel(self, save_dir: Optional[str] = None, fname_prefix: str = "quick_hist", show: bool = True):
        """
        Fast 1x3 histogram panel: σ, q_alpha (if present), q_beta (else q_gamma if present).
        """
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        # sigma
        s2 = self.draws.get("sigma2", None)
        if s2 is not None and np.size(s2) > 0:
            axes[0].hist(np.sqrt(np.clip(s2, 0, None)), bins=40, density=True)
        axes[0].set_title(r"$\sigma \mid y$")

        # choose two q's to show if available
        qs = self._get_qs()
        if qs:
            # first q
            axes[1].hist(qs[0][1][np.isfinite(qs[0][1])], bins=40, density=True)
            axes[1].set_title(fr"{qs[0][2]} | y")
            # second q if any
            if len(qs) > 1:
                axes[2].hist(qs[1][1][np.isfinite(qs[1][1])], bins=40, density=True)
                axes[2].set_title(fr"{qs[1][2]} | y")
            else:
                axes[2].axis("off")
        else:
            axes[1].axis("off"); axes[2].axis("off")

        plt.tight_layout()
        if save_dir:
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
    import argparse
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from optimization.posterior_bundle import load_posterior, find_latest_run

    parser = argparse.ArgumentParser(description="Plot DLM Gibbs (conjugate Gaussian) posterior diagnostics.")
    parser.add_argument("--run", type=str, default=None,
                        help="Path to a run directory or directly to posterior.npz. If omitted, we search under --root.")
    parser.add_argument("--root", type=str, default="results/simulations/DLM",
                        help="Search root (e.g. 'results/simulations/DLM').")
    parser.add_argument("--level", type=float, default=0.90, help="Credible interval level for bands.")
    parser.add_argument("--show", default=True, help="Show figures interactively.")
    parser.add_argument("--skip-states", action="store_true", help="Skip stacked states panel.")
    parser.add_argument("--skip-separate", action="store_true", help="Skip separate component figures.")
    args = parser.parse_args()

    # Resolve run
    if args.run:
        target = args.run
        print(f"[info] Using explicit run: {target}")
    else:
        print(f"[info] Searching latest run under: {args.root}")
        target = find_latest_run(root=args.root)
        if not target:
            raise FileNotFoundError(f"No 'posterior.npz' found under '{args.root}'. Pass --run explicitly.")
    bundle = load_posterior(target)
    draws, meta = bundle.draws, bundle.meta

    run_dir = os.path.dirname(bundle.npz_path)
    save_dir = os.path.join(run_dir, "figures")
    _ensure_dir(save_dir)
    print(f"[info] Using run dir: {run_dir}")
    print(f"[info] Saving figures to: {save_dir}")

    plotter = DLMPlotter(draws=draws, meta=meta, level=float(args.level))
    plotter.plot_diagnostics(save_dir=save_dir, fname_prefix="diagnostics", show=bool(args.show))
    if not args.skip_states:
        plotter.plot_states_and_observations(save_dir=save_dir, fname_prefix="states", show=bool(args.show))
    if not args.skip_separate:
        plotter.plot_components_separately(save_dir=save_dir, fname_prefix="components", show=bool(args.show))
    plotter.plot_quick_hist_panel(save_dir=save_dir, fname_prefix="quick_hist", show=bool(args.show))
    print("[done] Plots generated.")
