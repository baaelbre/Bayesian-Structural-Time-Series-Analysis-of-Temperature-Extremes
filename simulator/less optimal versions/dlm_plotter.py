# %% simulator/dlm_plotter.py
from __future__ import annotations
import os, json, math
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))



# ---------------------------------------------------------------------
# I/O helpers: use your optimization/posterior_io loader
# ---------------------------------------------------------------------
try:
    from optimization.posterior_bundle import load_posterior, find_latest_run
except Exception as e:
    raise ImportError(
        "Could not import optimization.posterior_io.load_posterior/find_latest_run.\n"
        "Make sure optimization/posterior_io.py is on PYTHONPATH."
    ) from e

# -----------------------------
# Small utils
# -----------------------------
def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

def _safe(x, default=None):
    return default if x is None else x

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
    # positive-sequence truncation
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

def _qtiles(arr: np.ndarray, level: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = (1 - level) / 2.0
    hi = 1.0 - lo
    return (np.quantile(arr, 0.5, axis=0),
            np.quantile(arr, lo, axis=0),
            np.quantile(arr, hi, axis=0))

def _maybe(a: Dict[str, Any], k: str):
    return a[k] if (k in a and a[k] is not None) else None


# -----------------------------
# Plotter
# -----------------------------
class DLMPlotter:
    """
    Plotter for the Kalman-Gibbs sampler with PC priors + Gamma hyperpriors + (log) slice/RW updates.

    Expected keys in 'draws' (npz):
      - mu: (S, T)
      - sigma: (S,)   or legacy 'sigma2': (S,)
      - Q: (S, dim)   (process variances)
      - sd: (S, dim)  (optional)
      - x: (S, T, dim)  (optional; dynamic state draws)
      - lambda_alpha/beta/gamma: (S,) (optional)
      - y: (T,)  (data)
      - true_*: optional truth overlays

    'meta' should include:
      - T, period, modes (dict or "lev-trend-season" string), and optionally idx_*.
    """

    def __init__(self, draws: Dict[str, np.ndarray], meta: Dict[str, Any], level: float = 0.90):
        self.draws = draws
        self.meta = meta
        self.level = float(level)
        if not (0 < self.level < 1):
            raise ValueError("level must be in (0,1)")

        # Core dims
        self.T = int(meta.get("T") or draws["mu"].shape[1])
        self.period = int(meta.get("period", 12))

        # Modes (robust)
        modes = meta.get("modes")
        if isinstance(modes, str):
            toks = modes.split("-")
            self.level_mode  = toks[0] if len(toks) > 0 else "dynamic"
            self.trend_mode  = toks[1] if len(toks) > 1 else "none"
            self.season_mode = toks[2] if len(toks) > 2 else "none"
        else:
            mm = modes if isinstance(modes, dict) else {}
            self.level_mode  = mm.get("level_mode", "dynamic")
            self.trend_mode  = mm.get("trend_mode", "none")
            self.season_mode = mm.get("seasonal_mode", "none")

        # Optional data & truths
        self.y           = _maybe(draws, "y")
        self.true_mu     = _maybe(draws, "true_mu_t")
        self.true_alpha  = _maybe(draws, "true_alpha_t")
        self.true_beta   = _maybe(draws, "true_beta_t")
        self.true_gamma  = _maybe(draws, "true_gamma_t")

        # State indexing (try meta first; fall back to layout logic)
        self.has_x = ("x" in draws) and draws["x"].ndim == 3
        self.idx_alpha = meta.get("idx_alpha")
        self.idx_beta  = meta.get("idx_beta")
        self.idx_g_end = meta.get("idx_g_end")
        if self.has_x and any(v is None for v in [self.idx_alpha, self.idx_beta, self.idx_g_end]):
            # infer from modes & period
            dim = draws["x"].shape[2]
            i_alpha = 0 if self.level_mode == "dynamic" else None
            i_beta = None
            if self.trend_mode == "dynamic":
                i_beta = (1 if i_alpha is not None else 0)
            i_g_end = None
            if self.season_mode == "dynamic":
                start = (1 if i_alpha is not None else 0) + (1 if i_beta is not None else 0)
                if (self.period - 1) > 0 and dim >= start + (self.period - 1):
                    i_g_end = start + (self.period - 2)
            self.idx_alpha = _safe(self.idx_alpha, i_alpha)
            self.idx_beta  = _safe(self.idx_beta,  i_beta)
            self.idx_g_end = _safe(self.idx_g_end, i_g_end)

        # sigma (supports both 'sigma' and legacy 'sigma2')
        if "sigma" in draws:
            self.sigma = np.asarray(draws["sigma"], float)
        elif "sigma2" in draws:
            self.sigma = np.sqrt(np.clip(np.asarray(draws["sigma2"], float), 0, None))
        else:
            self.sigma = None

        # process variances & sds
        self.Q  = _maybe(draws, "Q")
        self.sd = _maybe(draws, "sd")

        # hyper lambdas (optional)
        self.lam_a = _maybe(draws, "lambda_alpha")
        self.lam_b = _maybe(draws, "lambda_beta")
        self.lam_g = _maybe(draws, "lambda_gamma")

        # quantiles for ribbons
        self.lo_q = (1 - self.level) / 2.0
        self.hi_q = 1.0 - self.lo_q
        self.band_label = f"{int(round(self.level * 100))}% band"

    # ---------- helpers ----------
    def _summarize_ribbon(self, arr_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _qtiles(np.asarray(arr_2d, float), self.level)

    def _trace_acf_panel(self, series: np.ndarray, name: str,
                         max_lag: int = 200,
                         save_dir: Optional[str] = None,
                         fname: Optional[str] = None,
                         show: bool = True):
        s = np.asarray(series, float).ravel()
        ac = _acf(s, max_lag=max_lag)
        ess = _ess(s, max_lag=max_lag)
        gz  = _geweke_z(s)

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].plot(s, lw=1)
        axs[0].set_title(f"trace: {name}")
        axs[0].set_xlabel("iteration")

        axs[1].bar(np.arange(ac.size), ac, width=0.9)
        axs[1].set_xlim(-0.5, ac.size - 0.5)
        axs[1].set_title(f"ACF: {name} (ESS≈{ess:.0f}, Geweke z≈{gz:.2f})")
        axs[1].set_xlabel("lag")

        plt.tight_layout()
        if save_dir and fname:
            _ensure_dir(save_dir)
            path = os.path.join(save_dir, fname)
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"[save] {path}")
        if show: plt.show()
        else: plt.close(fig)

    def _component_draws(self, key: str) -> Optional[np.ndarray]:
        if not self.has_x: return None
        if key == "alpha" and self.idx_alpha is not None:
            return self.draws["x"][:, :, self.idx_alpha]
        if key == "beta" and self.idx_beta is not None:
            return self.draws["x"][:, :, self.idx_beta]
        if key == "gamma" and self.idx_g_end is not None:
            return self.draws["x"][:, :, self.idx_g_end]
        return None

    # ---------- figures ----------
    def figure_overview(self, save_dir: Optional[str] = None,
                        fname_prefix: str = "overview",
                        show: bool = True):
        """
        Overview (2x3):
          [0] μ_t ribbon (with y and true μ)
          [1] σ trace
          [2] σ histogram
          [3] log10(Q) histograms (available coords)
          [4] λ_* histograms (if present)
          [5] μ_t running mean error vs true μ (if truth available)
        """
        mu = np.asarray(self.draws["mu"], float)  # (S,T)
        m_ctr, m_lo, m_hi = self._summarize_ribbon(mu)

        fig, axs = plt.subplots(2, 3, figsize=(13, 8))
        axs = axs.ravel()

        # [0] μ ribbon
        t = np.arange(self.T)
        axs[0].plot(m_ctr, lw=1.6, label="μ median")
        axs[0].fill_between(t, m_lo, m_hi, alpha=0.25, label=self.band_label)
        if self.y is not None and len(self.y) == self.T:
            axs[0].plot(self.y, lw=1.0, alpha=0.6, label="y")
        if self.true_mu is not None and len(self.true_mu) == self.T:
            axs[0].plot(self.true_mu, lw=1.2, ls="--", label="true μ")
        axs[0].set_title("Posterior μ_t")
        axs[0].legend(loc="upper left")

        # [1],[2] σ trace + hist
        if self.sigma is not None:
            axs[1].plot(self.sigma, lw=1)
            axs[1].set_title("trace: σ")

            axs[2].hist(self.sigma, bins=40, density=True)
            es = _ess(self.sigma)
            gz = _geweke_z(self.sigma)
            axs[2].set_title(f"posterior: σ  (ESS≈{es:.0f}, z≈{gz:.2f})")
        else:
            axs[1].axis("off"); axs[2].axis("off")

        # [3] log10(Q) histograms (alpha/beta/gamma if present)
        if self.Q is not None and self.Q.size:
            Q = np.asarray(self.Q, float)
            logQ = np.log10(np.clip(Q, 1e-20, None))
            ax = axs[3]
            for j in range(min(Q.shape[1], 3)):  # show up to first 3 coords
                ax.hist(logQ[:, j], bins=40, density=True, alpha=0.55, label=f"log10 Q[{j}]")
            ax.set_title("Process variance (log10 scale)")
            ax.legend()
        else:
            axs[3].axis("off")

        # [4] lambda hyperpriors
        has_any_lambda = any(v is not None for v in [self.lam_a, self.lam_b, self.lam_g])
        if has_any_lambda:
            ax = axs[4]
            if self.lam_a is not None: ax.hist(self.lam_a, bins=40, density=True, alpha=0.6, label="λ_α")
            if self.lam_b is not None: ax.hist(self.lam_b, bins=40, density=True, alpha=0.6, label="λ_β")
            if self.lam_g is not None: ax.hist(self.lam_g, bins=40, density=True, alpha=0.6, label="λ_γ")
            ax.set_title("PC rate posteriors λ")
            ax.legend()
        else:
            axs[4].axis("off")

        # [5] μ running RMSE vs truth (if available)
        if self.true_mu is not None and len(self.true_mu) == self.T:
            err = np.mean((mu - self.true_mu.reshape(1, -1))**2, axis=1)**0.5  # draw-wise RMSE
            running = np.cumsum(err) / np.arange(1, err.size + 1)
            axs[5].plot(running, lw=1.2)
            axs[5].set_title("running RMSE(μ) vs truth")
            axs[5].set_xlabel("kept draw")
        else:
            axs[5].axis("off")

        plt.tight_layout()
        if save_dir:
            _ensure_dir(save_dir)
            out = os.path.join(save_dir, f"{fname_prefix}.png")
            fig.savefig(out, dpi=200, bbox_inches="tight")
            print(f"[save] {out}")
        if show: plt.show()
        else: plt.close(fig)

    def figure_trace_acf_core(self, save_dir: Optional[str] = None, show: bool = True):
        """
        Make trace+ACF panels for the most informative scalars:
          - σ
          - log10 Q for α, β, γ (if present)
          - λ_α, λ_β, λ_γ (if present)
        """
        if self.sigma is not None:
            self._trace_acf_panel(self.sigma, "σ",
                                  save_dir=save_dir, fname="trace_acf_sigma.png", show=show)

        if self.Q is not None and self.Q.size:
            Q = np.asarray(self.Q, float)
            names = ["α", "β", "γ(last)"]
            for j in range(min(Q.shape[1], 3)):
                label = f"log10 Q[{names[j] if j < len(names) else j}]"
                self._trace_acf_panel(np.log10(np.clip(Q[:, j], 1e-20, None)),
                                      label,
                                      save_dir=save_dir, fname=f"trace_acf_log10Q_{j}.png", show=show)

        if self.lam_a is not None:
            self._trace_acf_panel(self.lam_a, "λ_α", save_dir=save_dir, fname="trace_acf_lambda_alpha.png", show=show)
        if self.lam_b is not None:
            self._trace_acf_panel(self.lam_b, "λ_β", save_dir=save_dir, fname="trace_acf_lambda_beta.png", show=show)
        if self.lam_g is not None:
            self._trace_acf_panel(self.lam_g, "λ_γ", save_dir=save_dir, fname="trace_acf_lambda_gamma.png", show=show)

    def figure_states(self, save_dir: Optional[str] = None,
                      fname_prefix: str = "states",
                      show: bool = True):
        """
        Stacked ribbons for μ, α (if dynamic), β (if dynamic), γ_last (if seasonal dynamic).
        """
        rows = 1
        rows += 1 if self._component_draws("alpha") is not None else 0
        rows += 1 if self._component_draws("beta")  is not None else 0
        rows += 1 if self._component_draws("gamma") is not None else 0

        fig, axes = plt.subplots(rows, 1, figsize=(12, 3.0 * rows), sharex=True)
        if rows == 1: axes = [axes]
        r = 0
        t = np.arange(self.T)

        # μ
        mu = np.asarray(self.draws["mu"], float)
        ctr, lo, hi = self._summarize_ribbon(mu)
        ax = axes[r]
        if self.y is not None and len(self.y) == self.T:
            ax.plot(self.y, lw=1.0, alpha=0.6, label="y")
        ax.plot(ctr, lw=1.6, label="μ median")
        ax.fill_between(t, lo, hi, alpha=0.25, label=self.band_label)
        if self.true_mu is not None and len(self.true_mu) == self.T:
            ax.plot(self.true_mu, lw=1.2, ls="--", label="true μ")
        ax.set_title("Posterior μ_t")
        ax.legend(); r += 1

        # α
        A = self._component_draws("alpha")
        if A is not None:
            c, lo, hi = self._summarize_ribbon(A)
            ax = axes[r]; ax.plot(c, lw=1.6, label="α median")
            ax.fill_between(t, lo, hi, alpha=0.25, label=self.band_label)
            if self.true_alpha is not None and len(self.true_alpha) == self.T:
                ax.plot(self.true_alpha, lw=1.2, ls="--", label="true α")
            ax.set_title("Level α_t"); ax.legend(); r += 1

        # β
        B = self._component_draws("beta")
        if B is not None:
            c, lo, hi = self._summarize_ribbon(B)
            ax = axes[r]; ax.plot(c, lw=1.6, label="β median")
            ax.fill_between(t, lo, hi, alpha=0.25, label=self.band_label)
            if self.true_beta is not None and len(self.true_beta) == self.T:
                ax.plot(self.true_beta, lw=1.2, ls="--", label="true β")
            ax.set_title("Trend β_t"); ax.legend(); r += 1

        # γ(last)
        G = self._component_draws("gamma")
        if G is not None:
            c, lo, hi = self._summarize_ribbon(G)
            ax = axes[r]; ax.plot(c, lw=1.6, label="γ(last) median")
            ax.fill_between(t, lo, hi, alpha=0.25, label=self.band_label)
            if self.true_gamma is not None and len(self.true_gamma) == self.T:
                ax.plot(self.true_gamma, lw=1.2, ls="--", label="true γ")
            ax.set_title("Seasonal last coordinate γ_t"); ax.legend()

        plt.tight_layout()
        if save_dir:
            _ensure_dir(save_dir)
            out = os.path.join(save_dir, f"{fname_prefix}.png")
            fig.savefig(out, dpi=200, bbox_inches="tight")
            print(f"[save] {out}")
        if show: plt.show()
        else: plt.close(fig)

    def quick_report(self, save_dir: Optional[str] = None,
                     fname_prefix: str = "quick_report",
                     show: bool = True):
        """
        1x3 compact panel: μ ribbon, σ posterior, log10(Q_alpha) posterior (if available).
        """
        mu = np.asarray(self.draws["mu"], float)
        ctr, lo, hi = self._summarize_ribbon(mu)
        t = np.arange(self.T)

        fig, axs = plt.subplots(1, 3, figsize=(14, 4))

        axs[0].plot(ctr, lw=1.6, label="μ median")
        axs[0].fill_between(t, lo, hi, alpha=0.25, label=self.band_label)
        if self.true_mu is not None and len(self.true_mu) == self.T:
            axs[0].plot(self.true_mu, lw=1.2, ls="--", label="true μ")
        axs[0].set_title("μ_t"); axs[0].legend()

        if self.sigma is not None:
            axs[1].hist(self.sigma, bins=40, density=True)
            axs[1].set_title("σ | y")
        else:
            axs[1].axis("off")

        if self.Q is not None and self.Q.size:
            logQ = np.log10(np.clip(self.Q[:, 0], 1e-20, None))  # α coord if present
            axs[2].hist(logQ, bins=40, density=True)
            axs[2].set_title("log10 Q[α] | y")
        else:
            axs[2].axis("off")

        plt.tight_layout()
        if save_dir:
            _ensure_dir(save_dir)
            out = os.path.join(save_dir, f"{fname_prefix}.png")
            fig.savefig(out, dpi=200, bbox_inches="tight")
            print(f"[save] {out}")
        if show: plt.show()
        else: plt.close(fig)


# -----------------------------
# CLI: load and plot via posterior_io
# -----------------------------
if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(
        description="DLM plotter for Kalman-Gibbs with PC + hyper + slice; uses optimization/posterior_io.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", type=str, default=None,
                        help="Path to a run directory or directly to posterior.npz. If omitted, searches under --root.")
    parser.add_argument("--root", type=str, default="results/simulations/DLM",
                        help="Search root when --target is omitted.")
    parser.add_argument("--level", type=float, default=0.90, help="Credible band level.")
    parser.add_argument("--show", action="store_true", default=False, help="Show figures interactively.")
    parser.add_argument("--out", type=str, default=None, help="Directory to save figures. Default: <run>/figures")
    parser.add_argument("--skip-traceacf", action="store_true", help="Skip trace+ACF panels.")
    parser.add_argument("--skip-states",   action="store_true", help="Skip state ribbons.")
    parser.add_argument("--skip-overview", action="store_true", help="Skip overview figure.")
    parser.add_argument("--skip-quick",    action="store_true", help="Skip quick 1x3 panel.")
    args = parser.parse_args()

    # Resolve run path using your helper
    run_path = args.target
    if run_path is None:
        print(f"[info] --target not provided; searching for the latest posterior under --root={args.root!r} ...")
        run_path = find_latest_run(root=args.root)
        if run_path is None:
            print(f"[error] No 'posterior.npz' found under {args.root!r}. Provide --target or change --root.")
            sys.exit(1)
        print(f"[info] Using latest run: {run_path}")

    # Load bundle
    bundle = load_posterior(run_path)  # returns PosteriorBundle(draws, meta, npz_path, meta_path)
    draws, meta, npz_path = bundle.draws, bundle.meta, bundle.npz_path

    # Save dir
    out_dir = args.out or os.path.join(os.path.dirname(npz_path), "figures")
    _ensure_dir(out_dir)
    print(f"[info] saving figures to: {out_dir}")

    # Plot
    plotter = DLMPlotter(draws=draws, meta=meta, level=float(args.level))

    if not args.skip_overview:
        plotter.figure_overview(save_dir=out_dir, fname_prefix="overview", show=args.show)

    if not args.skip_traceacf:
        plotter.figure_trace_acf_core(save_dir=out_dir, show=args.show)

    if not args.skip_states:
        plotter.figure_states(save_dir=out_dir, fname_prefix="states", show=args.show)

    if not args.skip_quick:
        plotter.quick_report(save_dir=out_dir, fname_prefix="quick_report", show=args.show)

    print("[done] plots written.")
