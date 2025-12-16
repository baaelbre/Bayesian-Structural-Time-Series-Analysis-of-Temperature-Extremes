# %% simulator/dlm_plotter.py
from __future__ import annotations

import os
import math
import sys
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------
# I/O helpers: posterior loader
# ---------------------------------------------------------------------
try:
    from optimization.posterior_bundle import load_posterior, find_latest_run
except Exception as e:
    raise ImportError(
        "Could not import optimization.posterior_bundle.load_posterior/find_latest_run.\n"
        "Make sure optimization/posterior_bundle.py is on PYTHONPATH."
    ) from e


# =============================================================================
# Small utils
# =============================================================================
def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _acf(x: np.ndarray, max_lag: int = 200) -> np.ndarray:
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


def _ess(x: np.ndarray, max_lag: int = 200) -> float:
    ac = _acf(x, max_lag=max_lag)
    if not np.all(np.isfinite(ac)) or ac.size <= 1:
        return float(len(x))
    s = 0.0
    for k in range(1, ac.size):
        if ac[k] <= 0:
            break
        s += 2.0 * ac[k]
    n = len(x)
    return float(n) / max(1e-12, (1.0 + s))


def _geweke_z(x: np.ndarray, first_frac: float = 0.1, last_frac: float = 0.5) -> float:
    x = np.asarray(x, float)
    n = x.size
    if n < 8:
        return np.nan
    a = max(2, int(np.floor(first_frac * n)))
    b = max(2, int(np.floor(last_frac * n)))
    xa, xb = x[:a], x[n - b :]
    ma, mb = float(np.mean(xa)), float(np.mean(xb))
    va = float(np.var(xa, ddof=1)) / max(1, xa.size)
    vb = float(np.var(xb, ddof=1)) / max(1, xb.size)
    denom = math.sqrt(max(1e-300, va + vb))
    return (ma - mb) / denom


def _qtiles(arr_2d: np.ndarray, level: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = (1 - level) / 2.0
    hi = 1.0 - lo
    return (
        np.quantile(arr_2d, 0.5, axis=0),
        np.quantile(arr_2d, lo, axis=0),
        np.quantile(arr_2d, hi, axis=0),
    )


def _maybe(draws: Dict[str, Any], key: str) -> Optional[np.ndarray]:
    v = draws.get(key, None)
    return None if v is None else np.asarray(v)


# =============================================================================
# Plotter
# =============================================================================
class DLMPlotter:
    """
    Plotter for posterior bundles from the Gaussian DLM sampler.

    Expected core keys in draws:
      - mu: (S, T)
      - y: (T,) optional
      - x: (S, T, dim) centred state draws (optional)

    Common scalars:
      - sigma or sigma2
      - alpha0, beta0 (and optionally gamma0)
      - s_alpha, s_beta, s_gamma
      - Q_alpha, Q_beta, Q_gamma or Q (S, K)

    Notes:
      - No correlation plots in this version.
      - No trace/hist/ACF for lambda2 or tau_* in this version.
      - Separate component plots only (no combined states.png).
      - Slope rescaling is applied to the plotted values ONLY, and is never shown in any label/title.
    """

    def __init__(self, draws: Dict[str, np.ndarray], meta: Dict[str, Any], level: float = 0.90):
        self.draws = draws
        self.meta = meta
        self.level = float(level)
        if not (0 < self.level < 1):
            raise ValueError("level must be in (0,1)")

        if "mu" not in draws:
            raise ValueError("draws must contain 'mu' of shape (S, T).")

        self.mu = np.asarray(draws["mu"], float)
        if self.mu.ndim != 2:
            raise ValueError("'mu' must be a 2D array (S, T).")

        self.S, self.T = self.mu.shape
        self.period = int(meta.get("period", 12))

        # optional data & truths
        self.y = _maybe(draws, "y")
        self.true_mu = _maybe(draws, "true_mu_t")
        self.true_alpha = _maybe(draws, "true_alpha_t")
        self.true_beta = _maybe(draws, "true_beta_t")
        self.true_gamma = _maybe(draws, "true_gamma_t")

        # sigma
        self.sigma = None
        if "sigma" in draws and np.asarray(draws["sigma"]).shape[0] == self.S:
            self.sigma = np.asarray(draws["sigma"], float)
        elif "sigma2" in draws and np.asarray(draws["sigma2"]).shape[0] == self.S:
            self.sigma = np.sqrt(np.clip(np.asarray(draws["sigma2"], float), 0, None))

        # ---- scalar & vector params (per draw) ----
        self.scalar_params: Dict[str, np.ndarray] = {}
        self.vector_params: Dict[str, np.ndarray] = {}

        for k, v in draws.items():
            if k in {"y", "mu", "x", "true_mu_t", "true_alpha_t", "true_beta_t", "true_gamma_t"}:
                continue
            arr = np.asarray(v)
            if arr.ndim == 1 and arr.shape[0] == self.S:
                self.scalar_params[k] = arr.astype(float)
            elif arr.ndim == 2 and arr.shape[0] == self.S and arr.shape[1] != self.T:
                self.vector_params[k] = arr.astype(float)

        # baselines (alpha0 exists in your runs)
        self.alpha0 = np.asarray(draws["alpha0"], float) if "alpha0" in draws else _maybe(draws, "m0_alpha")
        self.beta0 = np.asarray(draws["beta0"], float) if "beta0" in draws else _maybe(draws, "m0_beta")
        self.gamma0 = np.asarray(draws["gamma0"], float) if "gamma0" in draws else _maybe(draws, "m0_gamma")

        # signed SDs
        self.s_alpha = self.scalar_params.get("s_alpha", None)
        self.s_beta = self.scalar_params.get("s_beta", None)
        self.s_gamma = self.scalar_params.get("s_gamma", None)

        # Q matrix
        self.Q = None
        self.Q_names: List[str] = []
        if "Q" in draws:
            Qmat = np.asarray(draws["Q"], float)
            if Qmat.ndim == 2 and Qmat.shape[0] == self.S:
                self.Q = Qmat
                layout = meta.get("layout")
                if isinstance(layout, (list, tuple)) and len(layout) == Qmat.shape[1]:
                    self.Q_names = [rf"$Q_{{{nm}}}$" for nm in layout]
                else:
                    self.Q_names = [rf"$Q_{{{j}}}$" for j in range(Qmat.shape[1])]
        else:
            cols = []
            names = []
            for nm, lab in (("Q_alpha", r"$Q_\alpha$"), ("Q_beta", r"$Q_\beta$"), ("Q_gamma", r"$Q_\gamma$")):
                if nm in self.scalar_params:
                    cols.append(self.scalar_params[nm].reshape(self.S, 1))
                    names.append(lab)
            if cols:
                self.Q = np.concatenate(cols, axis=1)
                self.Q_names = names

        # ---- state layout indexing ----
        self.has_x = ("x" in draws) and (np.asarray(draws["x"]).ndim == 3)
        self.idx_alpha = None
        self.idx_beta = None
        self.idx_g0 = None

        if self.has_x:
            x = np.asarray(draws["x"])
            dim = x.shape[2]
            layout = meta.get("layout")
            layout_list = list(layout) if isinstance(layout, (list, tuple)) else None

            if layout_list:
                if "alpha" in layout_list:
                    self.idx_alpha = layout_list.index("alpha")
                if "beta" in layout_list:
                    self.idx_beta = layout_list.index("beta")
                g_indices = [i for i, nm in enumerate(layout_list) if str(nm).startswith("g")]
                if g_indices:
                    self.idx_g0 = g_indices[0]
            else:
                # heuristic fallback: alpha, beta, then seasonal block
                self.idx_alpha = 0 if dim >= 1 else None
                self.idx_beta = 1 if dim >= 2 else None
                i = 2
                if self.period > 1 and dim >= i + (self.period - 1):
                    self.idx_g0 = i

        self.band_label_default = rf"{int(round(self.level * 100))}\% band"

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _summarize_ribbon(self, arr_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _qtiles(np.asarray(arr_2d, float), self.level)

    def _trace_hist_acf_panel(
        self,
        series: np.ndarray,
        name: str,
        *,
        title_trace: Optional[str] = None,
        title_hist: Optional[str] = None,
        title_acf: Optional[str] = None,
        xlabel_trace: str = "iteration",
        xlabel_acf: str = "lag",
        max_lag: int = 200,
        save_dir: Optional[str] = None,
        fname: Optional[str] = None,
        show: bool = True,
    ) -> None:
        s = np.asarray(series, float).ravel()
        ac = _acf(s, max_lag=max_lag)
        ess = _ess(s, max_lag=max_lag)
        gz = _geweke_z(s)

        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

        axs[0].plot(s, lw=1)
        axs[0].set_title(title_trace or rf"trace: {name}")
        axs[0].set_xlabel(xlabel_trace)

        axs[1].hist(s, bins=40, density=True)
        axs[1].set_title(title_hist or rf"hist: {name}")

        axs[2].bar(np.arange(ac.size), ac, width=0.9)
        axs[2].set_xlim(-0.5, ac.size - 0.5)
        axs[2].set_title(title_acf or rf"ACF: {name} (ESS$\approx${ess:.0f}, z$\approx${gz:.2f})")
        axs[2].set_xlabel(xlabel_acf)

        plt.tight_layout()
        if save_dir and fname:
            _ensure_dir(save_dir)
            path = os.path.join(save_dir, fname)
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"[save] {path}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def _component_draws(self, which: str) -> Optional[np.ndarray]:
        if not self.has_x:
            return None
        x = np.asarray(self.draws["x"])
        if which == "alpha" and self.idx_alpha is not None:
            return x[:, :, self.idx_alpha]
        if which == "beta" and self.idx_beta is not None:
            return x[:, :, self.idx_beta]
        if which in ("gamma", "seasonal") and self.idx_g0 is not None:
            return x[:, :, self.idx_g0]
        return None

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    def figure_overview(
        self,
        *,
        save_dir: Optional[str] = None,
        fname: str = "overview.png",
        show: bool = True,
        color: str = "C0",
        band_alpha: float = 0.25,
        band_label: Optional[str] = None,
        # ---- text ----
        title_mu: str = r"Posterior $\mu_t$",
        title_sigma_trace: str = r"trace: $\sigma$",
        title_sigma_hist: Optional[str] = None,
        title_Q: str = r"Process variances (log$_{10}$ scale)",
        title_baselines: str = r"Baselines",
        title_rmse: str = r"running RMSE($\mu$) vs truth",
        xlabel_time: str = r"$t$",
        ylabel_mu: str = r"$\mu_t$",
        # ---- axis tweaks ----
        ylims_mu: Optional[Tuple[float, float]] = None,
        yscale_mu: Optional[str] = None,
    ) -> None:
        band_label = band_label or self.band_label_default

        fig, axs = plt.subplots(2, 3, figsize=(13, 8))
        axs = axs.ravel()

        # [0] mu ribbon
        t = np.arange(self.T)
        ctr, lo, hi = self._summarize_ribbon(self.mu)
        axs[0].plot(t, ctr, lw=1.6, color=color, label=r"median")
        axs[0].fill_between(t, lo, hi, alpha=band_alpha, color=color, label=band_label)
        if self.y is not None and len(self.y) == self.T:
            axs[0].plot(t, self.y, lw=1.0, alpha=0.6, label=r"$y_t$")
        if self.true_mu is not None and len(self.true_mu) == self.T:
            axs[0].plot(t, self.true_mu, lw=1.2, ls="--", color="k", alpha=0.8, label=r"truth")
        axs[0].set_title(title_mu)
        axs[0].set_xlabel(xlabel_time)
        axs[0].set_ylabel(ylabel_mu)
        if yscale_mu is not None:
            axs[0].set_yscale(yscale_mu)
        if ylims_mu is not None:
            axs[0].set_ylim(*ylims_mu)
        axs[0].legend(loc="upper left")

        # [1],[2] sigma
        if self.sigma is not None:
            axs[1].plot(self.sigma, lw=1)
            axs[1].set_title(title_sigma_trace)
            axs[1].set_xlabel(r"kept draw")

            axs[2].hist(self.sigma, bins=40, density=True)
            if title_sigma_hist is None:
                es = _ess(self.sigma)
                gz = _geweke_z(self.sigma)
                axs[2].set_title(rf"posterior: $\sigma$  (ESS$\approx${es:.0f}, z$\approx${gz:.2f})")
            else:
                axs[2].set_title(title_sigma_hist)
        else:
            axs[1].axis("off")
            axs[2].axis("off")

        # [3] log10(Q)
        if self.Q is not None and self.Q.size:
            Q = np.asarray(self.Q, float)
            logQ = np.log10(np.clip(Q, 1e-20, None))
            ax = axs[3]
            labels = self.Q_names if self.Q_names else [rf"$Q_{{{j}}}$" for j in range(logQ.shape[1])]
            for j in range(logQ.shape[1]):
                ax.hist(logQ[:, j], bins=40, density=True, alpha=0.55, label=rf"$\log_{{10}}$ {labels[j]}")
            ax.set_title(title_Q)
            ax.legend()
        else:
            axs[3].axis("off")

        # [4] baselines
        any_baseline = (self.alpha0 is not None) or (self.beta0 is not None) or (self.gamma0 is not None)
        if any_baseline:
            ax = axs[4]
            if self.alpha0 is not None:
                ax.hist(np.asarray(self.alpha0).ravel(), bins=40, density=True, alpha=0.6, label=r"$\alpha_0$")
            if self.beta0 is not None:
                ax.hist(np.asarray(self.beta0).ravel(), bins=40, density=True, alpha=0.6, label=r"$\beta_0$")
            if self.gamma0 is not None and np.asarray(self.gamma0).ndim == 2 and np.asarray(self.gamma0).shape[1] > 0:
                ax.hist(np.asarray(self.gamma0)[:, 0], bins=40, density=True, alpha=0.6, label=r"$\gamma_0[0]$")
            ax.set_title(title_baselines)
            ax.legend()
        else:
            axs[4].axis("off")

        # [5] running RMSE if truth available
        if self.true_mu is not None and len(self.true_mu) == self.T:
            err = np.mean((self.mu - self.true_mu.reshape(1, -1)) ** 2, axis=1) ** 0.5
            running = np.cumsum(err) / np.arange(1, err.size + 1)
            axs[5].plot(running, lw=1.2)
            axs[5].set_title(title_rmse)
            axs[5].set_xlabel(r"kept draw")
        else:
            axs[5].axis("off")

        plt.tight_layout()
        if save_dir:
            _ensure_dir(save_dir)
            out = os.path.join(save_dir, fname)
            fig.savefig(out, dpi=200, bbox_inches="tight")
            print(f"[save] {out}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def figure_trace_acf_core(
        self,
        *,
        save_dir: Optional[str] = None,
        show: bool = True,
        max_lag: int = 200,
        # names shown in plot titles (make them LaTeX if you want)
        name_sigma: str = r"$\sigma$",
        name_s_alpha: str = r"$s_\alpha$",
        name_s_beta: str = r"$s_\beta$",
        name_s_gamma: str = r"$s_\gamma$",
        plot_other_scalars: bool = True,
    ) -> None:
        """
        Trace+hist+ACF for:
          - sigma
          - s_alpha/s_beta/s_gamma (if present)
          - log10(Q[...]) for Q-columns not represented by s_*
          - optionally: other scalar params (EXCLUDING lambda2 and tau_*)
        """

        if self.sigma is not None:
            self._trace_hist_acf_panel(
                self.sigma,
                name_sigma,
                max_lag=max_lag,
                save_dir=save_dir,
                fname="trace_hist_acf_sigma.png",
                show=show,
            )

        if self.s_alpha is not None:
            self._trace_hist_acf_panel(
                self.s_alpha,
                name_s_alpha,
                max_lag=max_lag,
                save_dir=save_dir,
                fname="trace_hist_acf_s_alpha.png",
                show=show,
            )
        if self.s_beta is not None:
            self._trace_hist_acf_panel(
                self.s_beta,
                name_s_beta,
                max_lag=max_lag,
                save_dir=save_dir,
                fname="trace_hist_acf_s_beta.png",
                show=show,
            )
        if self.s_gamma is not None:
            self._trace_hist_acf_panel(
                self.s_gamma,
                name_s_gamma,
                max_lag=max_lag,
                save_dir=save_dir,
                fname="trace_hist_acf_s_gamma.png",
                show=show,
            )

        if self.Q is not None and self.Q.size:
            Q = np.asarray(self.Q, float)
            labels = self.Q_names if self.Q_names else [rf"$Q_{{{j}}}$" for j in range(Q.shape[1])]
            for j in range(Q.shape[1]):
                nm = labels[j]
                # if you have the scalar Q_alpha/Q_beta/Q_gamma path, these may appear; handle both label styles
                if (nm in (r"$Q_\alpha$", "Q_alpha")) and self.s_alpha is not None:
                    continue
                if (nm in (r"$Q_\beta$", "Q_beta")) and self.s_beta is not None:
                    continue
                if (nm in (r"$Q_\gamma$", "Q_gamma")) and self.s_gamma is not None:
                    continue
                self._trace_hist_acf_panel(
                    np.log10(np.clip(Q[:, j], 1e-20, None)),
                    rf"$\log_{{10}}({nm})$",
                    max_lag=max_lag,
                    save_dir=save_dir,
                    fname=f"trace_hist_acf_log10Q_{j}.png",
                    show=show,
                )

        if plot_other_scalars:
            skip = {
                "sigma", "sigma2",
                "s_alpha", "s_beta", "s_gamma",
                "Q_alpha", "Q_beta", "Q_gamma",
                "lambda2", "tau_alpha", "tau_beta", "tau_gamma",
            }
            for k, arr in sorted(self.scalar_params.items()):
                if k in skip:
                    continue
                if k.startswith("tau_"):
                    continue
                # let user override names via cli if desired
                self._trace_hist_acf_panel(
                    arr,
                    k,
                    max_lag=max_lag,
                    save_dir=save_dir,
                    fname=f"trace_hist_acf_{k}.png",
                    show=show,
                )

    def figure_states_separate(
        self,
        *,
        save_dir: Optional[str] = None,
        fname_prefix: str = "state",
        show: bool = True,
        color: str = "C0",
        band_alpha: float = 0.25,
        band_label: Optional[str] = None,
        xlabel_time: str = r"$t$",
        # ---- titles/labels (LaTeX-ready strings) ----
        title_level: str = r"Level $\alpha_t$",
        ylabel_level: str = r"$\alpha_t$",
        title_slope: str = r"Slope $\beta_t$",
        ylabel_slope: str = r"$\beta_t$",
        title_seasonality: str = r"Seasonality $\gamma_t$ (contribution)",
        ylabel_seasonality: str = r"$\gamma_t$",
        # ---- scaling / axes ----
        slope_scale: float = 1.0,
        ylims: Optional[Dict[str, Tuple[float, float]]] = None,
        yscales: Optional[Dict[str, str]] = None,
        zero_line_slope: bool = True,
        zero_line_seasonality: bool = True,
    ) -> None:
        """
        Writes up to three files:
          - <prefix>_level.png
          - <prefix>_slope.png
          - <prefix>_seasonality.png

        IMPORTANT:
          slope_scale rescales the plotted values ONLY.
          It is never reflected in any title/label (by design).
        """
        band_label = band_label or self.band_label_default

        if not self.has_x:
            print("[states] no centred state draws 'x' found; skipping.")
            return

        t = np.arange(self.T)
        ylims = ylims or {}
        yscales = yscales or {}

        def _plot_component(
            arr2d: np.ndarray,
            *,
            out_name: str,
            title: str,
            ylabel: str,
            truth: Optional[np.ndarray],
            zero_line: bool,
            ylim: Optional[Tuple[float, float]],
            yscale: Optional[str],
        ) -> None:
            ctr, lo, hi = self._summarize_ribbon(arr2d)
            fig, ax = plt.subplots(1, 1, figsize=(12, 3.4))
            ax.plot(t, ctr, lw=1.6, color=color, label=r"median")
            ax.fill_between(t, lo, hi, alpha=band_alpha, color=color, label=band_label)

            if truth is not None and len(truth) == self.T:
                ax.plot(t, truth, lw=1.2, ls="--", color="k", alpha=0.8, label=r"truth")

            if zero_line:
                ax.axhline(0.0, lw=0.8, color="k", alpha=0.25)

            ax.set_title(title)
            ax.set_xlabel(xlabel_time)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")

            if yscale is not None:
                ax.set_yscale(yscale)
            if ylim is not None:
                ax.set_ylim(*ylim)

            plt.tight_layout()
            if save_dir:
                _ensure_dir(save_dir)
                out = os.path.join(save_dir, out_name)
                fig.savefig(out, dpi=200, bbox_inches="tight")
                print(f"[save] {out}")
            if show:
                plt.show()
            else:
                plt.close(fig)

        # level
        A = self._component_draws("alpha")
        if A is not None:
            _plot_component(
                A,
                out_name=f"{fname_prefix}_level.png",
                title=title_level,
                ylabel=ylabel_level,
                truth=self.true_alpha,
                zero_line=False,
                ylim=ylims.get("level"),
                yscale=yscales.get("level"),
            )

        # slope (scaled, but no mention in labels/titles)
        B = self._component_draws("beta")
        if B is not None:
            sc = float(slope_scale)
            Bp = B * sc
            tb = self.true_beta * sc if (self.true_beta is not None and len(self.true_beta) == self.T) else None

            _plot_component(
                Bp,
                out_name=f"{fname_prefix}_slope.png",
                title=title_slope,
                ylabel=ylabel_slope,
                truth=tb,
                zero_line=zero_line_slope,
                ylim=ylims.get("slope"),
                yscale=yscales.get("slope"),
            )

        # seasonality
        G = self._component_draws("gamma")
        if G is not None:
            _plot_component(
                G,
                out_name=f"{fname_prefix}_seasonality.png",
                title=title_seasonality,
                ylabel=ylabel_seasonality,
                truth=self.true_gamma,
                zero_line=zero_line_seasonality,
                ylim=ylims.get("seasonality"),
                yscale=yscales.get("seasonality"),
            )

    def quick_report(
        self,
        *,
        save_dir: Optional[str] = None,
        fname: str = "quick_report.png",
        show: bool = True,
        color: str = "C0",
        band_alpha: float = 0.25,
        band_label: Optional[str] = None,
        title_mu: str = r"$\mu_t$",
        title_sigma: str = r"$\sigma \mid y$",
        title_scale: str = r"process scale",
        xlabel_time: str = r"$t$",
        ylabel_mu: str = r"$\mu_t$",
        ylabel_scale: Optional[str] = None,
    ) -> None:
        band_label = band_label or self.band_label_default

        ctr, lo, hi = self._summarize_ribbon(self.mu)
        t = np.arange(self.T)

        fig, axs = plt.subplots(1, 3, figsize=(14, 4))

        axs[0].plot(t, ctr, lw=1.6, color=color, label=r"median")
        axs[0].fill_between(t, lo, hi, alpha=band_alpha, color=color, label=band_label)
        if self.true_mu is not None and len(self.true_mu) == self.T:
            axs[0].plot(t, self.true_mu, lw=1.2, ls="--", color="k", alpha=0.8, label=r"truth")
        axs[0].set_title(title_mu)
        axs[0].set_xlabel(xlabel_time)
        axs[0].set_ylabel(ylabel_mu)
        axs[0].legend()

        if self.sigma is not None:
            axs[1].hist(self.sigma, bins=40, density=True)
            axs[1].set_title(title_sigma)
        else:
            axs[1].axis("off")

        if self.s_alpha is not None:
            axs[2].hist(self.s_alpha, bins=40, density=True)
            axs[2].set_title(title_scale)
            if ylabel_scale is not None:
                axs[2].set_xlabel(ylabel_scale)
        elif self.Q is not None and self.Q.size:
            logQ = np.log10(np.clip(self.Q[:, 0], 1e-20, None))
            axs[2].hist(logQ, bins=40, density=True)
            axs[2].set_title(title_scale)
            if ylabel_scale is not None:
                axs[2].set_xlabel(ylabel_scale)
        else:
            axs[2].axis("off")

        plt.tight_layout()
        if save_dir:
            _ensure_dir(save_dir)
            out = os.path.join(save_dir, fname)
            fig.savefig(out, dpi=200, bbox_inches="tight")
            print(f"[save] {out}")
        if show:
            plt.show()
        else:
            plt.close(fig)


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    import argparse
    import ast

    def _parse_value(raw: str):
        s = raw.strip()
        low = s.lower()
        if low in ("none", "null"):
            return None
        if low in ("true", "false"):
            return low == "true"
        try:
            return ast.literal_eval(s)
        except Exception:
            return s

    def _set_nested(d: dict, key: str, value):
        parts = [p for p in key.split(".") if p]
        cur = d
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = value

    def _parse_kv_list(items: List[str]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for it in items:
            if "=" not in it:
                raise ValueError(f"Expected K=V, got: {it!r}")
            k, v = it.split("=", 1)
            k = k.strip()
            val = _parse_value(v)
            if "." in k:
                _set_nested(out, k, val)
            else:
                out[k] = val
        return out

    parser = argparse.ArgumentParser(
        description=(
            "DLM plotter for Gaussian structural models (non-centred state bundles).\n"
            "Produces overview, scalar trace/hist/ACF (excluding lambda2/tau_*), "
            "separate state component plots, and a quick report.\n"
            "Use --<section>-kw K=V (repeatable) to override ANY kwargs.\n"
            "Nested dicts: use dot notation, e.g. ylims.slope=(-1,1)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--target", type=str, default=None,
                        help="Path to a run directory or directly to posterior.npz. If omitted, searches under --root.")
    parser.add_argument("--root", type=str, default="results/simulations/DLM",
                        help="Search root when --target is omitted.")
    parser.add_argument("--level", type=float, default=0.90, help="Credible band level.")
    parser.add_argument("--show", action="store_true", default=False, help="Show figures interactively.")
    parser.add_argument("--out", type=str, default=None, help="Directory to save figures. Default: <run>/figures")

    parser.add_argument("--skip-overview", action="store_true", help="Skip overview figure.")
    parser.add_argument("--skip-traceacf", action="store_true", help="Skip trace+hist+ACF panels.")
    parser.add_argument("--skip-states", action="store_true", help="Skip separate state plots.")
    parser.add_argument("--skip-quick", action="store_true", help="Skip quick report.")

    parser.add_argument("--overview-kw", action="append", default=[], metavar="K=V",
                        help="Override kwargs for plotter.figure_overview(...). Repeatable. Supports nested keys via dots.")
    parser.add_argument("--traceacf-kw", action="append", default=[], metavar="K=V",
                        help="Override kwargs for plotter.figure_trace_acf_core(...). Repeatable. Supports nested keys via dots.")
    parser.add_argument("--states-kw", action="append", default=[], metavar="K=V",
                        help="Override kwargs for plotter.figure_states_separate(...). Repeatable. Supports nested keys via dots.")
    parser.add_argument("--quick-kw", action="append", default=[], metavar="K=V",
                        help="Override kwargs for plotter.quick_report(...). Repeatable. Supports nested keys via dots.")

    args = parser.parse_args()

    # Resolve run path
    run_path = args.target
    if run_path is None:
        print(f"[info] --target not provided; searching for the latest posterior under --root={args.root!r} ...")
        run_path = find_latest_run(root=args.root)
        if run_path is None:
            print(f"[error] No 'posterior.npz' found under {args.root!r}. Provide --target or change --root.")
            sys.exit(1)
        print(f"[info] Using latest run: {run_path}")

    # Load bundle
    bundle = load_posterior(run_path)
    draws, meta, npz_path = bundle.draws, bundle.meta, bundle.npz_path

    # Output dir
    out_dir = args.out or os.path.join(os.path.dirname(npz_path), "figures")
    _ensure_dir(out_dir)
    print(f"[info] saving figures to: {out_dir}")

    # Plotter
    plotter = DLMPlotter(draws=draws, meta=meta, level=float(args.level))

    # Parse kw overrides
    overview_kw = _parse_kv_list(args.overview_kw)
    traceacf_kw = _parse_kv_list(args.traceacf_kw)
    states_kw = _parse_kv_list(args.states_kw)
    quick_kw = _parse_kv_list(args.quick_kw)

    # Call plots
    if not args.skip_overview:
        plotter.figure_overview(save_dir=out_dir, show=args.show, **overview_kw)

    if not args.skip_traceacf:
        plotter.figure_trace_acf_core(save_dir=out_dir, show=args.show, **traceacf_kw)

    if not args.skip_states:
        plotter.figure_states_separate(save_dir=out_dir, show=args.show, **states_kw)

    if not args.skip_quick:
        plotter.quick_report(save_dir=out_dir, show=args.show, **quick_kw)

    print("[done] plots written.")
