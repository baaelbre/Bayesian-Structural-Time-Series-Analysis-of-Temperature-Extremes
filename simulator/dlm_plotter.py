# %% simulator/dlm_plotter.py
import os
from typing import Optional, Tuple, Dict, Any
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
# Plotter
# =========================
class DLMPlotter:
    """
    Standalone plotting helper for the conjugate Gaussian DLM Gibbs sampler.

    Expected 'bundle' or 'sampler-like' fields:
      draws: dict with keys like {"mu", "sigma_y", "x", "Q", "theta", ...}
      meta:  dict with at least {"T", "period", "modes": {"level","trend","season"}}
      y:     optional (vector length T) for overlays

    Dynamic states layout (when present in 'x'):
      tags = [alpha?][beta?][g1..g_{p-1}?] in that order, exactly like the sampler.
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
        modes = meta.get("modes", {})
        self.level_mode = modes.get("level", "dynamic")
        self.trend_mode = modes.get("trend", "dynamic")
        self.season_mode = modes.get("season", "dynamic")

        # Optional truth overlays
        self.true_mu = draws.get("true_mu_t")
        self.true_alpha = draws.get("true_alpha_t")
        self.true_beta = draws.get("true_beta_t")
        self.true_gamma = draws.get("true_gamma_t")

        # Optional data
        self.y = draws.get("y", None)

        # Quantiles for bands
        self.lo_q = (1.0 - self.level) / 2.0
        self.hi_q = 1.0 - self.lo_q
        self.band_label = f"{int(round(self.level * 100))}% band"

        # Figure out indices into x for α,β,γ (if x present)
        self.has_x = ("x" in draws) and (draws["x"].ndim == 3)
        if self.has_x:
            # draws["x"] is (n_keep, T, dim)
            dim = draws["x"].shape[2]
            i_alpha = 0 if self.level_mode == "dynamic" else None
            i_beta = None
            if self.trend_mode == "dynamic":
                i_beta = (1 if i_alpha is not None else 0)
            i_g0 = i_gL = None
            if self.season_mode == "dynamic":
                # g block starts after alpha/beta
                start = (1 if i_alpha is not None else 0) + (1 if i_beta is not None else 0)
                if (self.period - 1) > 0:
                    i_g0 = start
                    i_gL = start + (self.period - 2)  # last coord index
                # Basic sanity
                if dim != start + (self.period - 1):
                    # If mismatch, just null out season indices
                    i_g0 = i_gL = None
            self.idx_alpha = i_alpha
            self.idx_beta = i_beta
            self.idx_gamma_end = i_gL
        else:
            self.idx_alpha = self.idx_beta = self.idx_gamma_end = None

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

    def _tile_season_to_T_from_theta(self, theta_draws: np.ndarray) -> np.ndarray:
        """
        theta collects [intercept?, slope?, season_1..season_{p-1}?] depending on modes.
        Return a (n_keep, T) time series representing the seasonal contribution.
        """
        if ("theta" not in self.draws) or (self.season_mode != "deterministic"):
            return np.zeros((self.draws["mu"].shape[0], self.T))
        Theta = np.asarray(theta_draws, float)  # (n_keep, p_theta)
        n_keep, p_theta = Theta.shape

        # build index where season entries start inside theta
        offset = 0
        if self.level_mode == "deterministic":
            offset += 1
        if self.trend_mode == "deterministic":
            offset += 1
        num_seas = self.period - 1
        if num_seas <= 0:
            return np.zeros((n_keep, self.T))
        seas = Theta[:, offset : offset + num_seas]  # (n_keep, p-1)

        # expand to full seasonal vector with last = -sum(first p-1), then tile across time
        last = -np.sum(seas, axis=1, keepdims=True)  # (n_keep, 1)
        full = np.concatenate([seas, last], axis=1)  # (n_keep, p)
        reps = int(np.ceil(self.T / self.period))
        tiled = np.tile(full, (1, reps))[:, : self.T]  # (n_keep, T)
        # align the dummy index per time step
        # We need to select the appropriate seasonal entry by t mod p:
        idx = np.arange(self.T) % self.period
        out = np.take_along_axis(tiled.reshape(n_keep, reps, self.period).reshape(n_keep, -1), idx[None, :], axis=1)
        return out

    # ---------- figures ----------
    def plot_diagnostics(self, save_dir: Optional[str] = None, fname_prefix: str = "diagnostics", show: bool = True):
        n_keep = self.draws["mu"].shape[0]
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))
        axs = axs.ravel()

        # trace & hist for sigma_y
        sig = np.asarray(self.draws.get("sigma_y", np.array([])))
        if sig.size > 0:
            axs[0].plot(sig, lw=1)
            axs[0].set_title(r"trace: $\sigma_y$")
            axs[3].hist(sig, bins=30, density=True)
            axs[3].set_title(r"posterior: $\sigma_y$")
            ess_s = _ess(sig, max_lag=200)
            gz_s = _geweke_z(sig)
            axs[3].set_xlabel(f"ESS≈{ess_s:.0f}, Geweke z≈{gz_s:.2f}")
        else:
            axs[0].axis("off"); axs[3].axis("off")

        # Q diag traces/hists if present
        Q = self.draws.get("Q", None)
        if Q is not None and Q.ndim == 2 and Q.shape[0] == n_keep:
            # Show up to 2 Q entries (alpha and gamma) commonly used
            axs[1].set_title("trace: selected Q entries")
            for j in range(min(Q.shape[1], 2)):
                axs[1].plot(Q[:, j], lw=1, alpha=0.8, label=f"Q[{j}]")
            axs[1].legend()

            axs[4].set_title("posterior: selected Q entries")
            for j in range(min(Q.shape[1], 2)):
                axs[4].hist(Q[:, j], bins=30, density=True, alpha=0.6, label=f"Q[{j}]")
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
            axs[2].plot(self.true_mu, ls="--", lw=1.2, label="true $\mu_t$")
        axs[2].set_title(r"$\mu_t$ vs data")
        axs[2].legend()

        # Running mean of sigma_y (if present) & simple ACF
        if sig.size > 0:
            axs[5].plot(np.cumsum(sig) / np.arange(1, sig.size + 1))
            axs[5].set_title(r"$\sigma_y$ running mean")
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
        if self.level_mode == "dynamic" and self.idx_alpha is not None: rows += 1
        if self.trend_mode == "dynamic" and self.idx_beta is not None: rows += 1
        if self.season_mode == "dynamic" and self.idx_gamma_end is not None: rows += 1
        if self.trend_mode == "deterministic" and ("theta" in self.draws): rows += 1
        if self.season_mode == "deterministic" and ("theta" in self.draws): rows += 1

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
            ax.plot(self.true_mu, lw=1.2, ls="--", label="true $\mu_t$")
        ax.set_title("Data and posterior $\mu_t$")
        ax.legend()
        r += 1

        # α_t (dynamic)
        if self.level_mode == "dynamic" and self.idx_alpha is not None and self.has_x:
            at = self.draws["x"][:, :, self.idx_alpha]
            a_ctr, a_lo, a_hi = self._summarize_paths(at)
            ax = axes[r]
            ax.plot(a_ctr, lw=1.6, label=r"$\alpha_t$ median")
            ax.fill_between(np.arange(self.T), a_lo, a_hi, alpha=0.25, label=self.band_label)
            if (self.true_alpha is not None) and len(self.true_alpha) == self.T:
                ax.plot(self.true_alpha, lw=1.2, ls="--", label=r"true $\alpha_t$")
            ax.set_title(r"Level component $\alpha_t$")
            ax.legend()
            r += 1

        # β_t (dynamic) or slope (deterministic)
        if self.trend_mode == "dynamic" and self.idx_beta is not None and self.has_x:
            bt = self.draws["x"][:, :, self.idx_beta]
            b_ctr, b_lo, b_hi = self._summarize_paths(bt)
            ax = axes[r]
            ax.plot(b_ctr, lw=1.6, label=r"$\beta_t$ median")
            ax.fill_between(np.arange(self.T), b_lo, b_hi, alpha=0.25, label=self.band_label)
            if (self.true_beta is not None) and len(self.true_beta) == self.T:
                ax.plot(self.true_beta, lw=1.2, ls="--", label="true $\beta_t$")
            ax.set_title("Trend component $\beta_t$")
            ax.legend()
            r += 1
        elif self.trend_mode == "deterministic" and ("theta" in self.draws):
            # slope is a scalar per draw; tile to T
            Theta = self.draws["theta"]
            offset = (1 if self.level_mode == "deterministic" else 0)
            slope_draws = Theta[:, offset] if Theta.shape[1] > offset else np.zeros(Theta.shape[0])
            slope_ts = self._tile_static_to_T(slope_draws)
            s_ctr, s_lo, s_hi = self._summarize_paths(slope_ts)
            ax = axes[r]
            ax.plot(s_ctr, lw=1.6, label=r"slope $\beta$ (det.) median")
            ax.fill_between(np.arange(self.T), s_lo, s_hi, alpha=0.25, label=self.band_label)
            ax.set_title("Deterministic slope $\beta$")
            ax.legend()
            r += 1

        # γ_t (dynamic) or deterministic seasonal profile
        if self.season_mode == "dynamic" and self.idx_gamma_end is not None and self.has_x:
            gt = self.draws["x"][:, :, self.idx_gamma_end]
            g_ctr, g_lo, g_hi = self._summarize_paths(gt)
            ax = axes[r]
            ax.plot(g_ctr, lw=1.6, label=r"$\gamma_t$ median")
            ax.fill_between(np.arange(self.T), g_lo, g_hi, alpha=0.25, label=self.band_label)
            if (self.true_gamma is not None) and len(self.true_gamma) == self.T:
                ax.plot(self.true_gamma, lw=1.2, ls="--", label="true $\gamma_t$")
            ax.set_title("Seasonal last-coordinate $\gamma_t$")
            ax.legend()
        elif self.season_mode == "deterministic" and ("theta" in self.draws):
            seas_ts = self._tile_season_to_T_from_theta(self.draws["theta"])
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
            ax.plot(self.true_mu, lw=1.2, ls="--", label="true $\mu_t$")
        ax.set_title(r"Posterior $\mu_t$")
        ax.legend()
        plt.tight_layout()
        if save_dir:
            _ensure_dir(save_dir)
            out_path = os.path.join(save_dir, f"{fname_prefix}_mu.png")
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            print(f"[save] Figure -> {out_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)

        # α_t
        if self.level_mode == "dynamic" and self.idx_alpha is not None and self.has_x:
            at = self.draws["x"][:, :, self.idx_alpha]
            a_ctr, a_lo, a_hi = self._summarize_paths(at)
            fig, ax = plt.subplots(figsize=(12, 3.2))
            ax.plot(a_ctr, lw=1.8, label=r"$\alpha_t$ median")
            ax.fill_between(np.arange(self.T), a_lo, a_hi, alpha=0.25, label=self.band_label)
            if (self.true_alpha is not None) and len(self.true_alpha) == self.T:
                ax.plot(self.true_alpha, lw=1.2, ls="--", label="true $\alpha_t$")
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
        if self.trend_mode == "dynamic" and self.idx_beta is not None and self.has_x:
            bt = self.draws["x"][:, :, self.idx_beta]
            b_ctr, b_lo, b_hi = self._summarize_paths(bt)
            fig, ax = plt.subplots(figsize=(12, 3.2))
            ax.plot(b_ctr, lw=1.8, label=r"$\beta_t$ median")
            ax.fill_between(np.arange(self.T), b_lo, b_hi, alpha=0.25, label=self.band_label)
            if (self.true_beta is not None) and len(self.true_beta) == self.T:
                ax.plot(self.true_beta, lw=1.2, ls="--", label="true $\beta_t$")
            ax.set_title(r"Posterior trend $\beta_t$")
            ax.legend()
            plt.tight_layout()
            if save_dir:
                out_path = os.path.join(save_dir, f"{fname_prefix}_beta.png")
                fig.savefig(out_path, dpi=200, bbox_inches="tight")
                print(f"[save] Figure -> {out_path}")
            if show: plt.show()
            else: plt.close(fig)
        elif self.trend_mode == "deterministic" and ("theta" in self.draws):
            Theta = self.draws["theta"]
            offset = (1 if self.level_mode == "deterministic" else 0)
            slope_draws = Theta[:, offset] if Theta.shape[1] > offset else np.zeros(Theta.shape[0])
            slope_ts = self._tile_static_to_T(slope_draws)
            s_ctr, s_lo, s_hi = self._summarize_paths(slope_ts)
            fig, ax = plt.subplots(figsize=(12, 3.2))
            ax.plot(s_ctr, lw=1.8, label=r"$\beta$ (det.) median")
            ax.fill_between(np.arange(self.T), s_lo, s_hi, alpha=0.25, label=self.band_label)
            ax.set_title(r"Deterministic slope $\beta$")
            ax.legend()
            plt.tight_layout()
            if save_dir:
                out_path = os.path.join(save_dir, f"{fname_prefix}_beta_det.png")
                fig.savefig(out_path, dpi=200, bbox_inches="tight")
                print(f"[save] Figure -> {out_path}")
            if show: plt.show()
            else: plt.close(fig)

        # γ
        if self.season_mode == "dynamic" and self.idx_gamma_end is not None and self.has_x:
            gt = self.draws["x"][:, :, self.idx_gamma_end]
            g_ctr, g_lo, g_hi = self._summarize_paths(gt)
            fig, ax = plt.subplots(figsize=(12, 3.2))
            ax.plot(g_ctr, lw=1.8, label=r"$\gamma_t$ median")
            ax.fill_between(np.arange(self.T), g_lo, g_hi, alpha=0.25, label=self.band_label)
            if (self.true_gamma is not None) and len(self.true_gamma) == self.T:
                ax.plot(self.true_gamma, lw=1.2, ls="--", label="true $\gamma_t$")
            ax.set_title(r"Posterior seasonal last coord $\gamma_t$")
            ax.legend()
            plt.tight_layout()
            if save_dir:
                out_path = os.path.join(save_dir, f"{fname_prefix}_gamma.png")
                fig.savefig(out_path, dpi=200, bbox_inches="tight")
                print(f"[save] Figure -> {out_path}")
            if show: plt.show()
            else: plt.close(fig)
        elif self.season_mode == "deterministic" and ("theta" in self.draws):
            seas_ts = self._tile_season_to_T_from_theta(self.draws["theta"])
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
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        # sigma_y
        sig = self.draws.get("sigma_y", None)
        if sig is not None:
            axes[0].hist(sig, bins=40, density=True)
        axes[0].set_title(r"$\sigma_y | y$")
        # One or two Q entries if present
        Q = self.draws.get("Q", None)
        if Q is not None and Q.ndim == 2:
            axes[1].hist(Q[:, 0], bins=40, density=True)
            axes[1].set_title(r"$Q[0] | y$")
            if Q.shape[1] > 1:
                axes[2].hist(Q[:, 1], bins=40, density=True)
                axes[2].set_title(r"$Q[1] | y$")
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
# ----------------------------- #
# CLI: Load a posterior and plot (via optimization/posterior_io)
# ----------------------------- #
if __name__ == "__main__":
    import argparse
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from optimization.posterior_bundle import load_posterior, find_latest_run

    parser = argparse.ArgumentParser(description="Plot DLM Gibbs (conjugate Gaussian) posterior diagnostics.")
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Path to a run directory (containing posterior.npz & posterior.meta.json) "
             "or directly to posterior.npz. If omitted, we'll search under --root."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="results/simulations/DLM",
        help="Search root (choose one of: 'results/simulations/DLM', 'uccle/TXm', 'uccle/TNm')."
    )
    parser.add_argument("--level", type=float, default=0.90, help="Credible interval level for bands.")
    parser.add_argument("--show", action="store_true", help="Show figures interactively.")
    parser.add_argument("--skip-states", action="store_true", help="Skip stacked states panel.")
    parser.add_argument("--skip-separate", action="store_true", help="Skip separate component figures.")
    args = parser.parse_args()

    # Resolve the target run using your IO helpers
    if args.run:
        target = args.run
        print(f"[info] Using explicit run: {target}")
    else:
        print(f"[info] Searching latest run under: {args.root}")
        target = find_latest_run(root=args.root)
        if not target:
            raise FileNotFoundError(
                f"No 'posterior.npz' found under '{args.root}'. "
                f"Pass --run explicitly or pick a valid --root (results_dlm | uccle/TXm | uccle/TNm)."
            )

    # Load posterior bundle (draws + meta + paths)
    bundle = load_posterior(target)
    draws, meta = bundle.draws, bundle.meta

    # Where to save figures (same folder as the posterior)
    run_dir = os.path.dirname(bundle.npz_path)
    save_dir = os.path.join(run_dir, "figures")
    _ensure_dir(save_dir)

    print(f"[info] Using run dir: {run_dir}")
    print(f"[info] Saving figures to: {save_dir}")

    # Make plots
    plotter = DLMPlotter(draws=draws, meta=meta, level=float(args.level))
    plotter.plot_diagnostics(save_dir=save_dir, fname_prefix="diagnostics", show=bool(args.show))
    if not args.skip_states:
        plotter.plot_states_and_observations(save_dir=save_dir, fname_prefix="states", show=bool(args.show))
    if not args.skip_separate:
        plotter.plot_components_separately(save_dir=save_dir, fname_prefix="components", show=bool(args.show))
    plotter.plot_quick_hist_panel(save_dir=save_dir, fname_prefix="quick_hist", show=bool(args.show))
    print("[done] Plots generated.")

