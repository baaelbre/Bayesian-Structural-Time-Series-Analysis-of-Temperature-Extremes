# %% simulator/dlm_plotter.py
from __future__ import annotations

import os
import sys
import math
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


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def _robust_sd_from_mad(mad: float) -> float:
    # For Normal: MAD ≈ 0.6745 * sd  => sd ≈ MAD/0.6745
    if not np.isfinite(mad) or mad <= 0:
        return 0.0
    return float(mad) / 0.6745


def _acf(x: np.ndarray, max_lag: int = 200) -> np.ndarray:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
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
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return float(max(1, x.size))

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
    x = x[np.isfinite(x)]
    n = x.size
    if n < 8:
        return float("nan")
    a = max(2, int(np.floor(first_frac * n)))
    b = max(2, int(np.floor(last_frac * n)))
    xa, xb = x[:a], x[n - b :]
    ma, mb = float(np.mean(xa)), float(np.mean(xb))
    va = float(np.var(xa, ddof=1)) / max(1, xa.size)
    vb = float(np.var(xb, ddof=1)) / max(1, xb.size)
    denom = math.sqrt(max(1e-300, va + vb))
    return (ma - mb) / denom


def _maybe(draws: Dict[str, Any], key: str) -> Optional[np.ndarray]:
    v = draws.get(key, None)
    return None if v is None else np.asarray(v)


def _normalize_center(center: str) -> str:
    c = str(center).strip().lower()
    if c in ("median", "q50", "q0.5", "quantile", "quantile50"):
        return "median"
    if c in ("mean", "avg", "average", "expectation"):
        return "mean"
    raise ValueError("center must be one of {'median','mean'} (aliases allowed: q50/avg/...).")


def _center_label(center: str) -> str:
    c = _normalize_center(center)
    return "mean" if c == "mean" else "median"


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

    New (trace/hist handling):
      - You can zoom traces/hists (trace_ylim, hist_xlim)
      - Or handle outliers for plotting/diagnostics:
          plot_policy: "none" | "clip" | "drop"
          diag_policy: "raw" | "clipped" | "clean"
          clip_q=(qlo,qhi)   quantile bounds
          clip_nmad=k        bounds = median ± k * MAD/0.6745
          max_abs=a          bounds = [-a, +a]

    New (ribbon center):
      - center: "median" (default) or "mean"
        Used for the central line only; the band remains quantile-based.
    """

    def __init__(self, draws: Dict[str, np.ndarray], meta: Dict[str, Any], level: float = 0.90):
        self.draws = draws
        self.meta = meta
        self.level = float(level)
        if not (0.0 < self.level < 1.0):
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
        self.sigma: Optional[np.ndarray] = None
        if "sigma" in draws and np.asarray(draws["sigma"]).shape[0] == self.S:
            self.sigma = np.asarray(draws["sigma"], float)
        elif "sigma2" in draws and np.asarray(draws["sigma2"]).shape[0] == self.S:
            self.sigma = np.sqrt(np.clip(np.asarray(draws["sigma2"], float), 0.0, None))

        # scalar & vector params
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

        # baselines
        self.alpha0 = np.asarray(draws["alpha0"], float) if "alpha0" in draws else _maybe(draws, "m0_alpha")
        self.beta0 = np.asarray(draws["beta0"], float) if "beta0" in draws else _maybe(draws, "m0_beta")
        self.gamma0 = np.asarray(draws["gamma0"], float) if "gamma0" in draws else _maybe(draws, "m0_gamma")

        # signed SDs
        self.s_alpha = self.scalar_params.get("s_alpha", None)
        self.s_beta = self.scalar_params.get("s_beta", None)
        self.s_gamma = self.scalar_params.get("s_gamma", None)

        # Q matrix
        self.Q: Optional[np.ndarray] = None
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
            cols, names = [], []
            for nm, lab in (("Q_alpha", r"$Q_\alpha$"), ("Q_beta", r"$Q_\beta$"), ("Q_gamma", r"$Q_\gamma$")):
                if nm in self.scalar_params:
                    cols.append(self.scalar_params[nm].reshape(self.S, 1))
                    names.append(lab)
            if cols:
                self.Q = np.concatenate(cols, axis=1)
                self.Q_names = names

        # state layout indexing
        self.has_x = ("x" in draws) and (np.asarray(draws["x"]).ndim == 3)
        self.idx_alpha: Optional[int] = None
        self.idx_beta: Optional[int] = None
        self.idx_g0: Optional[int] = None

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
                self.idx_alpha = 0 if dim >= 1 else None
                self.idx_beta = 1 if dim >= 2 else None
                i = 2
                if self.period > 1 and dim >= i + (self.period - 1):
                    self.idx_g0 = i

        self.band_label_default = rf"{int(round(self.level * 100))}% band"

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _summarize_ribbon(
        self,
        arr_2d: np.ndarray,
        *,
        center: str = "median",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (center_line, lo, hi) where lo/hi are quantiles at the chosen level,
        and center_line is either posterior median or posterior mean.
        """
        arr_2d = np.asarray(arr_2d, float)
        c = _normalize_center(center)

        lo_q = (1.0 - self.level) / 2.0
        hi_q = 1.0 - lo_q

        lo = np.quantile(arr_2d, lo_q, axis=0)
        hi = np.quantile(arr_2d, hi_q, axis=0)

        if c == "mean":
            ctr = np.mean(arr_2d, axis=0)
        else:
            ctr = np.quantile(arr_2d, 0.5, axis=0)

        return ctr, lo, hi

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
    # Trace/Hist/ACF panel (unchanged)
    # ------------------------------------------------------------------
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
        trace_ylim: Optional[Tuple[float, float]] = None,
        hist_xlim: Optional[Tuple[float, float]] = None,
        plot_policy: str = "none",    # "none" | "clip" | "drop"
        diag_policy: str = "clean",   # "raw" | "clipped" | "clean"
        drop_nonfinite: bool = True,
        clip_q: Optional[Tuple[float, float]] = None,
        clip_nmad: Optional[float] = None,
        max_abs: Optional[float] = None,
        hist_bins: int = 40,
        auto_zoom_if_clipped: bool = True,
    ) -> None:
        s_raw = np.asarray(series, float).ravel()
        finite_mask = np.isfinite(s_raw)
        n_nonfinite = int(np.sum(~finite_mask))
        s_finite = s_raw[finite_mask] if drop_nonfinite else s_raw.copy()

        if s_finite.size == 0:
            fig, axs = plt.subplots(1, 3, figsize=(15, 4))
            for ax in axs:
                ax.axis("off")
            fig.suptitle(rf"{name}: no finite samples")
            plt.tight_layout()
            if save_dir and fname:
                _ensure_dir(save_dir)
                out = os.path.join(save_dir, fname)
                fig.savefig(out, dpi=200, bbox_inches="tight")
                print(f"[save] {out}")
            if show:
                plt.show()
            else:
                plt.close(fig)
            return

        plot_policy = str(plot_policy).lower()
        diag_policy = str(diag_policy).lower()
        if plot_policy not in ("none", "clip", "drop"):
            raise ValueError("plot_policy must be one of: 'none', 'clip', 'drop'")
        if diag_policy not in ("raw", "clipped", "clean"):
            raise ValueError("diag_policy must be one of: 'raw', 'clipped', 'clean'")

        lo_bound, hi_bound = -np.inf, np.inf

        if max_abs is not None:
            a = float(abs(max_abs))
            lo_bound = max(lo_bound, -a)
            hi_bound = min(hi_bound, +a)

        if clip_q is not None:
            ql, qh = float(clip_q[0]), float(clip_q[1])
            ql = max(0.0, min(1.0, ql))
            qh = max(0.0, min(1.0, qh))
            if qh <= ql:
                raise ValueError(f"clip_q must satisfy q_high > q_low, got {clip_q}")
            qlo = float(np.quantile(s_finite, ql))
            qhi = float(np.quantile(s_finite, qh))
            lo_bound = max(lo_bound, qlo)
            hi_bound = min(hi_bound, qhi)

        if clip_nmad is not None:
            k = float(clip_nmad)
            med = float(np.median(s_finite))
            mad = _mad(s_finite)
            rsd = _robust_sd_from_mad(mad)
            if rsd > 0:
                lo_bound = max(lo_bound, med - k * rsd)
                hi_bound = min(hi_bound, med + k * rsd)

        use_bounds = np.isfinite(lo_bound) or np.isfinite(hi_bound)
        if not use_bounds:
            lo_bound, hi_bound = -np.inf, np.inf

        if use_bounds:
            out_mask_finite = (s_finite < lo_bound) | (s_finite > hi_bound)
        else:
            out_mask_finite = np.zeros_like(s_finite, dtype=bool)
        n_out = int(np.sum(out_mask_finite))

        s_plot = s_raw.copy()
        s_plot[~finite_mask] = np.nan

        if use_bounds and plot_policy == "clip":
            s_plot = np.clip(s_plot, lo_bound, hi_bound)
        elif use_bounds and plot_policy == "drop":
            out_mask_raw = np.zeros_like(s_raw, dtype=bool)
            out_mask_raw[finite_mask] = out_mask_finite
            s_plot[out_mask_raw] = np.nan

        s_hist = s_finite.copy()
        if use_bounds and plot_policy == "clip":
            s_hist = np.clip(s_hist, lo_bound, hi_bound)
        elif use_bounds and plot_policy == "drop":
            s_hist = s_hist[~out_mask_finite]

        if diag_policy == "raw":
            s_diag = s_finite.copy()
        elif diag_policy == "clipped":
            s_diag = np.clip(s_finite, lo_bound, hi_bound) if use_bounds else s_finite.copy()
        else:
            s_diag = s_finite[~out_mask_finite] if use_bounds else s_finite.copy()

        ac = _acf(s_diag, max_lag=max_lag)
        ess = _ess(s_diag, max_lag=max_lag)
        gz = _geweke_z(s_diag)

        extra = []
        if drop_nonfinite and n_nonfinite > 0:
            extra.append(f"nonfinite={n_nonfinite}")
        if use_bounds and n_out > 0:
            extra.append(f"outliers={n_out}")
        extra_txt = f" ({', '.join(extra)})" if extra else ""

        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

        axs[0].plot(s_plot, lw=1)
        axs[0].set_title(title_trace or rf"trace: {name}{extra_txt}")
        axs[0].set_xlabel(xlabel_trace)
        if trace_ylim is not None:
            axs[0].set_ylim(*trace_ylim)
        elif auto_zoom_if_clipped and use_bounds and plot_policy in ("clip", "drop") and np.isfinite(lo_bound) and np.isfinite(hi_bound):
            axs[0].set_ylim(lo_bound, hi_bound)

        hist_range = None
        if hist_xlim is not None:
            hist_range = (float(hist_xlim[0]), float(hist_xlim[1]))
        elif auto_zoom_if_clipped and use_bounds and plot_policy in ("clip", "drop") and np.isfinite(lo_bound) and np.isfinite(hi_bound):
            hist_range = (float(lo_bound), float(hi_bound))

        axs[1].hist(s_hist, bins=int(hist_bins), density=True, range=hist_range)
        axs[1].set_title(title_hist or rf"hist: {name}{extra_txt}")
        if hist_xlim is not None:
            axs[1].set_xlim(*hist_xlim)

        axs[2].bar(np.arange(ac.size), ac, width=0.9)
        axs[2].set_xlim(-0.5, ac.size - 0.5)
        axs[2].set_title(title_acf or rf"ACF: {name} (ESS$\approx${ess:.0f}, z$\approx${gz:.2f})")
        axs[2].set_xlabel(xlabel_acf)

        plt.tight_layout()
        if save_dir and fname:
            _ensure_dir(save_dir)
            out = os.path.join(save_dir, fname)
            fig.savefig(out, dpi=200, bbox_inches="tight")
            print(f"[save] {out}")
        if show:
            plt.show()
        else:
            plt.close(fig)

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
        center: str = "median",
        title_mu: str = r"Posterior $\mu_t$",
        title_sigma_trace: str = r"trace: $\sigma$",
        title_sigma_hist: Optional[str] = None,
        title_Q: str = r"Process variances (log$_{10}$ scale)",
        title_baselines: str = r"Baselines",
        title_rmse: str = r"running RMSE($\mu$) vs truth",
        xlabel_time: str = r"$t$",
        ylabel_mu: str = r"$\mu_t$",
        ylims_mu: Optional[Tuple[float, float]] = None,
        yscale_mu: Optional[str] = None,
    ) -> None:
        band_label = band_label or self.band_label_default
        c_lab = _center_label(center)

        fig, axs = plt.subplots(2, 3, figsize=(13, 8))
        axs = axs.ravel()

        t = np.arange(self.T)
        ctr, lo, hi = self._summarize_ribbon(self.mu, center=center)
        axs[0].plot(t, ctr, lw=1.6, color=color, label=c_lab)
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
        name_sigma: str = r"$\sigma$",
        name_s_alpha: str = r"$s_\alpha$",
        name_s_beta: str = r"$s_\beta$",
        name_s_gamma: str = r"$s_\gamma$",
        plot_other_scalars: bool = True,
        trace_ylim: Optional[Tuple[float, float]] = None,
        hist_xlim: Optional[Tuple[float, float]] = None,
        plot_policy: str = "none",
        diag_policy: str = "clean",
        drop_nonfinite: bool = True,
        clip_q: Optional[Tuple[float, float]] = None,
        clip_nmad: Optional[float] = None,
        max_abs: Optional[float] = None,
        hist_bins: int = 40,
        auto_zoom_if_clipped: bool = True,
    ) -> None:
        def _panel(series: np.ndarray, nm: str, fname: str) -> None:
            self._trace_hist_acf_panel(
                series,
                nm,
                max_lag=max_lag,
                save_dir=save_dir,
                fname=fname,
                show=show,
                trace_ylim=trace_ylim,
                hist_xlim=hist_xlim,
                plot_policy=plot_policy,
                diag_policy=diag_policy,
                drop_nonfinite=drop_nonfinite,
                clip_q=clip_q,
                clip_nmad=clip_nmad,
                max_abs=max_abs,
                hist_bins=hist_bins,
                auto_zoom_if_clipped=auto_zoom_if_clipped,
            )

        if self.sigma is not None:
            _panel(self.sigma, name_sigma, "trace_hist_acf_sigma.png")
        if self.s_alpha is not None:
            _panel(self.s_alpha, name_s_alpha, "trace_hist_acf_s_alpha.png")
        if self.s_beta is not None:
            _panel(self.s_beta, name_s_beta, "trace_hist_acf_s_beta.png")
        if self.s_gamma is not None:
            _panel(self.s_gamma, name_s_gamma, "trace_hist_acf_s_gamma.png")

        if self.Q is not None and self.Q.size:
            Q = np.asarray(self.Q, float)
            labels = self.Q_names if self.Q_names else [rf"$Q_{{{j}}}$" for j in range(Q.shape[1])]
            for j in range(Q.shape[1]):
                nm = labels[j]
                if (nm in (r"$Q_\alpha$", "Q_alpha")) and self.s_alpha is not None:
                    continue
                if (nm in (r"$Q_\beta$", "Q_beta")) and self.s_beta is not None:
                    continue
                if (nm in (r"$Q_\gamma$", "Q_gamma")) and self.s_gamma is not None:
                    continue
                series = np.log10(np.clip(Q[:, j], 1e-20, None))
                _panel(series, rf"$\log_{{10}}({nm})$", f"trace_hist_acf_log10Q_{j}.png")

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
                _panel(arr, k, f"trace_hist_acf_{k}.png")

    def figure_states_separate(
        self,
        *,
        save_dir: Optional[str] = None,
        fname_prefix: str = "state",
        show: bool = True,
        color: str = "C0",
        band_alpha: float = 0.25,
        band_label: Optional[str] = None,
        center: str = "median",
        xlabel_time: str = r"$t$",
        title_level: str = r"Level $\alpha_t$",
        ylabel_level: str = r"$\alpha_t$",
        title_slope: str = r"Slope $\beta_t$",
        ylabel_slope: str = r"$\beta_t$",
        title_seasonality: str = r"Seasonality $\gamma_t$ (contribution)",
        ylabel_seasonality: str = r"$\gamma_t$",
        slope_scale: float = 1.0,
        ylims: Optional[Dict[str, Tuple[float, float]]] = None,
        yscales: Optional[Dict[str, str]] = None,
        zero_line_slope: bool = True,
        zero_line_seasonality: bool = True,
    ) -> None:
        band_label = band_label or self.band_label_default
        c_lab = _center_label(center)

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
            ctr, lo, hi = self._summarize_ribbon(arr2d, center=center)

            fig, ax = plt.subplots(1, 1, figsize=(12, 3.4))
            ax.plot(t, ctr, lw=1.6, color=color, label=c_lab)
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
        center: str = "median",
        title_mu: str = r"$\mu_t$",
        title_sigma: str = r"$\sigma \mid y$",
        title_scale: str = r"process scale",
        xlabel_time: str = r"$t$",
        ylabel_mu: str = r"$\mu_t$",
        ylabel_scale: Optional[str] = None,
    ) -> None:
        band_label = band_label or self.band_label_default
        c_lab = _center_label(center)

        ctr, lo, hi = self._summarize_ribbon(self.mu, center=center)
        t = np.arange(self.T)

        fig, axs = plt.subplots(1, 3, figsize=(14, 4))

        axs[0].plot(t, ctr, lw=1.6, color=color, label=c_lab)
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
            "Produces overview, scalar trace/hist/ACF, separate state component plots, and a quick report.\n"
            "Use --<section>-kw K=V (repeatable) to override kwargs.\n"
            "Nested dicts: use dot notation, e.g. ylims.slope=(-1,1).\n\n"
            "Ribbon center (use via --overview-kw/--states-kw/--quick-kw):\n"
            "  center='median' (default) or center='mean'\n"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--target", type=str, default=None,
                        help="Path to a run directory or directly to posterior.npz. If omitted, searches under --root.")
    parser.add_argument("--root", type=str, default="results/simulations/DLM", help="Search root if --target is omitted.")
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
    parser.add_argument("--states-kw", action="append", default=["slope_scale=1", "center=mean"], metavar="K=V",
                        help="Override kwargs for plotter.figure_states_separate(...). Repeatable. Supports nested keys via dots.")
    parser.add_argument("--quick-kw", action="append", default=[], metavar="K=V",
                        help="Override kwargs for plotter.quick_report(...). Repeatable. Supports nested keys via dots.")

    args = parser.parse_args()

    run_path = args.target
    if run_path is None:
        print(f"[info] --target not provided; searching for the latest posterior under --root={args.root!r} ...")
        run_path = find_latest_run(root=args.root)
        if run_path is None:
            print(f"[error] No 'posterior.npz' found under {args.root!r}. Provide --target or change --root.")
            sys.exit(1)
        print(f"[info] Using latest run: {run_path}")

    bundle = load_posterior(run_path)
    draws, meta, npz_path = bundle.draws, bundle.meta, bundle.npz_path

    out_dir = args.out or os.path.join(os.path.dirname(npz_path), "figures")
    _ensure_dir(out_dir)
    print(f"[info] saving figures to: {out_dir}")

    plotter = DLMPlotter(draws=draws, meta=meta, level=float(args.level))

    overview_kw = _parse_kv_list(args.overview_kw)
    traceacf_kw = _parse_kv_list(args.traceacf_kw)
    states_kw = _parse_kv_list(args.states_kw)
    quick_kw = _parse_kv_list(args.quick_kw)

    if not args.skip_overview:
        plotter.figure_overview(save_dir=out_dir, show=args.show, **overview_kw)
    if not args.skip_traceacf:
        plotter.figure_trace_acf_core(save_dir=out_dir, show=args.show, **traceacf_kw)
    if not args.skip_states:
        plotter.figure_states_separate(save_dir=out_dir, show=args.show, **states_kw)
    if not args.skip_quick:
        plotter.quick_report(save_dir=out_dir, show=args.show, **quick_kw)

    print("[done] plots written.")
