# %% simulator/dlm_plotter.py
from __future__ import annotations
import os, json, math
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import sys

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
    return (
        np.quantile(arr, 0.5, axis=0),
        np.quantile(arr, lo, axis=0),
        np.quantile(arr, hi, axis=0),
    )


def _maybe(a: Dict[str, Any], k: str):
    return a[k] if (k in a and a[k] is not None) else None


# -----------------------------
# Plotter
# -----------------------------
class DLMPlotter:
    """
    Generic plotter for posterior bundles from Gaussian DLM samplers
    (including the FS-prior non-centred version, double-gamma, and older variants).

    It introspects the contents of 'draws' and tries to:

      - Plot μ_t ribbons (with y and true μ if present).
      - Plot trace + hist + ACF for all scalar parameters (shape (S,)).
      - Plot log10(Q) histograms for process variances, whether stored in:
          * a matrix 'Q' (S, dim), or
          * separate 'Q_alpha', 'Q_beta', 'Q_gamma' arrays.
      - Plot trace + hist + ACF for *signed* process SDs s_alpha, s_beta, s_gamma
        if present (FS style, ±√Q).
      - Plot state ribbons (μ, α, β, last seasonal γ) from 'x' if available.
      - Plot correlation scatter matrices for signed SDs, m0_*, and joint m0_*–s_*.

    Expected keys (flexible; many are optional):

      Core:
        - mu: (S, T)
        - y: (T,)
        - x: (S, T, dim)   (optional; centred state draws)

      Parameters (FS/double-gamma version):
        - sigma: (S,)
        - s_alpha, s_beta, s_gamma: (S,)  signed process SDs
        - Q_alpha, Q_beta, Q_gamma: (S,)
        - m0_alpha, m0_beta: (S,)
        - m0_gamma: (S, K) with K = period-1
        - P0_alpha, P0_beta, P0_gamma: (S,)

      Optional legacy:
        - sigma2: (S,)
        - Q: (S, dim)
        - lambda_alpha/beta/gamma: (S,)

      Truth overlays:
        - true_mu_t, true_alpha_t, true_beta_t, true_gamma_t: (T,)
    """

    def __init__(self, draws: Dict[str, np.ndarray], meta: Dict[str, Any], level: float = 0.90):
        self.draws = draws
        self.meta = meta
        self.level = float(level)
        if not (0 < self.level < 1):
            raise ValueError("level must be in (0,1)")

        # Core: mu
        if "mu" not in draws:
            raise ValueError("draws must contain 'mu' (S, T).")
        self.mu = np.asarray(draws["mu"], float)
        self.S, self.T = self.mu.shape

        # period
        self.period = int(meta.get("period", 12))

        # Modes (robust)
        modes = meta.get("modes")
        if isinstance(modes, str):
            toks = modes.split("-")
            self.level_mode = toks[0] if len(toks) > 0 else "dynamic"
            self.trend_mode = toks[1] if len(toks) > 1 else "none"
            self.season_mode = toks[2] if len(toks) > 2 else "none"
        else:
            mm = modes if isinstance(modes, dict) else {}
            self.level_mode = mm.get("level_mode", "dynamic")
            self.trend_mode = mm.get("trend_mode", "none")
            self.season_mode = mm.get("seasonal_mode", "none")

        # Optional data & truths
        self.y = _maybe(draws, "y")
        self.true_mu = _maybe(draws, "true_mu_t")
        self.true_alpha = _maybe(draws, "true_alpha_t")
        self.true_beta = _maybe(draws, "true_beta_t")
        self.true_gamma = _maybe(draws, "true_gamma_t")

        # layout and state indexing
        self.has_x = ("x" in draws) and draws["x"].ndim == 3
        self.layout = meta.get("layout")
        if self.layout is not None:
            self.layout = list(self.layout)
        self.idx_alpha = None
        self.idx_beta = None
        self.idx_g_end = None

        if self.has_x:
            dim = draws["x"].shape[2]
            if self.layout:
                if "alpha" in self.layout:
                    self.idx_alpha = self.layout.index("alpha")
                if "beta" in self.layout:
                    self.idx_beta = self.layout.index("beta")
                g_indices = [i for i, nm in enumerate(self.layout) if nm.startswith("g")]
                if g_indices:
                    self.idx_g_end = g_indices[-1]
            else:
                # fallback: old layout heuristic
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
                self.idx_beta = _safe(self.idx_beta, i_beta)
                self.idx_g_end = _safe(self.idx_g_end, i_g_end)

        # sigma (supports both 'sigma' and legacy 'sigma2')
        self.sigma = None
        if "sigma" in draws and np.asarray(draws["sigma"]).shape[0] == self.S:
            self.sigma = np.asarray(draws["sigma"], float)
        elif "sigma2" in draws and np.asarray(draws["sigma2"]).shape[0] == self.S:
            self.sigma = np.sqrt(np.clip(np.asarray(draws["sigma2"], float), 0, None))

        # lambdas (legacy PC priors)
        self.lam_a = _maybe(draws, "lambda_alpha")
        self.lam_b = _maybe(draws, "lambda_beta")
        self.lam_g = _maybe(draws, "lambda_gamma")

        # Categorise all posterior arrays into scalar / vector (per draw)
        self.scalar_params: Dict[str, np.ndarray] = {}
        self.vector_params: Dict[str, np.ndarray] = {}

        for k, v in draws.items():
            arr = np.asarray(v)
            # Skip observation paths, state paths and truths
            if k in ["y", "mu", "x", "true_mu_t", "true_alpha_t", "true_beta_t", "true_gamma_t"]:
                continue
            # scalar parameter: shape (S,)
            if arr.ndim == 1 and arr.shape[0] == self.S:
                self.scalar_params[k] = arr.astype(float)
            # vector parameter: shape (S, K) with K != T (to avoid double-count 'mu')
            elif arr.ndim == 2 and arr.shape[0] == self.S and arr.shape[1] != self.T:
                self.vector_params[k] = arr.astype(float)

        # Convenience aliases for FS-style / double-gamma outputs (if present)
        self.Q_alpha = _maybe(self.scalar_params, "Q_alpha")
        self.Q_beta = _maybe(self.scalar_params, "Q_beta")
        self.Q_gamma = _maybe(self.scalar_params, "Q_gamma")
        self.m0_alpha = _maybe(self.scalar_params, "m0_alpha")
        self.m0_beta = _maybe(self.scalar_params, "m0_beta")
        self.P0_alpha = _maybe(self.scalar_params, "P0_alpha")
        self.P0_beta = _maybe(self.scalar_params, "P0_beta")
        self.P0_gamma = _maybe(self.scalar_params, "P0_gamma")
        self.m0_gamma = _maybe(self.vector_params, "m0_gamma")

        # Signed process SDs (Fruhwirth-Schnatter / double-gamma style, s_k with Q_k = s_k^2)
        self.s_alpha = _maybe(self.scalar_params, "s_alpha")
        self.s_beta = _maybe(self.scalar_params, "s_beta")
        self.s_gamma = _maybe(self.scalar_params, "s_gamma")

        # Double-gamma scales (if present)
        self.xi_alpha = _maybe(self.scalar_params, "xi_alpha")
        self.xi_beta = _maybe(self.scalar_params, "xi_beta")
        self.xi_gamma = _maybe(self.scalar_params, "xi_gamma")
        self.tau = _maybe(self.scalar_params, "tau")

        # Build unified process variance matrix Q (S, K_Q) if available
        self.Q = None
        self.Q_names: List[str] = []

        if "Q" in draws:
            Qmat = np.asarray(draws["Q"], float)
            if Qmat.ndim == 2 and Qmat.shape[0] == self.S:
                self.Q = Qmat
                if self.layout and len(self.layout) == Qmat.shape[1]:
                    self.Q_names = [f"Q[{nm}]" for nm in self.layout]
                else:
                    self.Q_names = [f"Q[{j}]" for j in range(Qmat.shape[1])]
        else:
            cols = []
            names = []
            if self.Q_alpha is not None:
                cols.append(self.Q_alpha.reshape(self.S, 1))
                names.append("Q_alpha")
            if self.Q_beta is not None:
                cols.append(self.Q_beta.reshape(self.S, 1))
                names.append("Q_beta")
            if self.Q_gamma is not None:
                cols.append(self.Q_gamma.reshape(self.S, 1))
                names.append("Q_gamma")
            if cols:
                self.Q = np.concatenate(cols, axis=1)
                self.Q_names = names

        # quantiles for ribbons
        self.lo_q = (1 - self.level) / 2.0
        self.hi_q = 1.0 - self.lo_q
        self.band_label = f"{int(round(self.level * 100))}% band"

    # ---------- helpers ----------
    def _summarize_ribbon(self, arr_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _qtiles(np.asarray(arr_2d, float), self.level)

    def _trace_hist_acf_panel(
        self,
        series: np.ndarray,
        name: str,
        max_lag: int = 200,
        save_dir: Optional[str] = None,
        fname: Optional[str] = None,
        show: bool = True,
    ):
        """
        1x3 panel: trace, histogram, ACF with ESS + Geweke z.
        """
        s = np.asarray(series, float).ravel()
        ac = _acf(s, max_lag=max_lag)
        ess = _ess(s, max_lag=max_lag)
        gz = _geweke_z(s)

        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        # trace
        axs[0].plot(s, lw=1)
        axs[0].set_title(f"trace: {name}")
        axs[0].set_xlabel("iteration")

        # hist
        axs[1].hist(s, bins=40, density=True)
        axs[1].set_title(f"hist: {name}")

        # ACF
        axs[2].bar(np.arange(ac.size), ac, width=0.9)
        axs[2].set_xlim(-0.5, ac.size - 0.5)
        axs[2].set_title(f"ACF: {name} (ESS≈{ess:.0f}, z≈{gz:.2f})")
        axs[2].set_xlabel("lag")

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

    def _component_draws(self, key: str) -> Optional[np.ndarray]:
        if not self.has_x:
            return None
        if key == "alpha" and self.idx_alpha is not None:
            return self.draws["x"][:, :, self.idx_alpha]
        if key == "beta" and self.idx_beta is not None:
            return self.draws["x"][:, :, self.idx_beta]
        if key == "gamma" and self.idx_g_end is not None:
            return self.draws["x"][:, :, self.idx_g_end]
        return None

    def _scatter_matrix(
        self,
        data: np.ndarray,
        labels: List[str],
        title: str,
        fname: str,
        save_dir: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Simple scatter-matrix:
          - diagonal: histogram
          - off-diagonal: scatter with Pearson ρ annotated
        """
        data = np.asarray(data, float)
        n = data.shape[1]
        if n < 2:
            print(f"[corr] not enough variables for {title}, need at least 2.")
            return

        fig, axes = plt.subplots(n, n, figsize=(3.0 * n, 3.0 * n))
        for i in range(n):
            for j in range(n):
                ax = axes[i, j]
                x = data[:, j]
                y = data[:, i]

                if i == j:
                    ax.hist(x, bins=40, density=True)
                    ax.set_ylabel(labels[i])
                else:
                    ax.scatter(x, y, s=4, alpha=0.4)
                    # Pearson correlation (guard against NaNs / constants)
                    if np.std(x) > 1e-12 and np.std(y) > 1e-12:
                        r = float(np.corrcoef(x, y)[0, 1])
                    else:
                        r = float("nan")
                    ax.text(
                        0.05,
                        0.9,
                        f"ρ={r:.2f}" if np.isfinite(r) else "ρ=NA",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=8,
                    )

                # tidy ticks
                if i < n - 1:
                    ax.set_xticklabels([])
                if j > 0:
                    ax.set_yticklabels([])

        fig.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_dir:
            _ensure_dir(save_dir)
            path = os.path.join(save_dir, fname)
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"[save] {path}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    # ---------- figures ----------
    def figure_overview(
        self,
        save_dir: Optional[str] = None,
        fname_prefix: str = "overview",
        show: bool = True,
    ):
        """
        Overview (2x3):

          [0] μ_t ribbon (with y and true μ)
          [1] σ trace
          [2] σ histogram
          [3] log10(Q) histograms (available coords)
          [4] m0_*, m0_gamma[0] histograms (if present)
          [5] μ_t running RMSE vs true μ (if truth available)
        """
        mu = self.mu
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
            axs[1].set_xlabel("kept draw")

            axs[2].hist(self.sigma, bins=40, density=True)
            es = _ess(self.sigma)
            gz = _geweke_z(self.sigma)
            axs[2].set_title(f"posterior: σ  (ESS≈{es:.0f}, z≈{gz:.2f})")
        else:
            axs[1].axis("off")
            axs[2].axis("off")

        # [3] log10(Q) histograms (up to first few coords)
        if self.Q is not None and self.Q.size:
            Q = np.asarray(self.Q, float)
            logQ = np.log10(np.clip(Q, 1e-20, None))
            ax = axs[3]
            n_cols = logQ.shape[1]
            labels = self.Q_names if self.Q_names else [f"Q[{j}]" for j in range(n_cols)]
            for j in range(n_cols):
                ax.hist(logQ[:, j], bins=40, density=True, alpha=0.55, label=f"log10 {labels[j]}")
            ax.set_title("Process variances (log10 scale)")
            ax.legend()
        else:
            axs[3].axis("off")

        # [4] Baselines m0_* (if present)
        any_m0 = any(
            v is not None for v in [self.m0_alpha, self.m0_beta, self.m0_gamma]
        )
        if any_m0:
            ax = axs[4]
            if self.m0_alpha is not None:
                ax.hist(self.m0_alpha, bins=40, density=True, alpha=0.6, label="m0_alpha")
            if self.m0_beta is not None:
                ax.hist(self.m0_beta, bins=40, density=True, alpha=0.6, label="m0_beta")
            if self.m0_gamma is not None and self.m0_gamma.shape[1] > 0:
                ax.hist(self.m0_gamma[:, 0], bins=40, density=True, alpha=0.6, label="m0_gamma[0]")
            ax.set_title("Baselines m0_*")
            ax.legend()
        else:
            axs[4].axis("off")

        # [5] μ running RMSE vs truth (if available)
        if self.true_mu is not None and len(self.true_mu) == self.T:
            err = np.mean((mu - self.true_mu.reshape(1, -1)) ** 2, axis=1) ** 0.5  # draw-wise RMSE
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
        if show:
            plt.show()
        else:
            plt.close(fig)

    def figure_trace_acf_core(self, save_dir: Optional[str] = None, show: bool = True):
        """
        Trace + hist + ACF panels for key scalars and ALL remaining scalar parameters:

          - σ
          - signed process SDs s_alpha, s_beta, s_gamma (if present)
            (Fruhwirth-Schnatter style / double-gamma, ±√Q)
          - log10 Q for remaining Q-coordinates (for which no signed s_* exist)
          - λ_α, λ_β, λ_γ (if present)
          - any other scalar posterior array (shape (S,))
        """
        # σ
        if self.sigma is not None:
            self._trace_hist_acf_panel(
                self.sigma,
                "σ",
                save_dir=save_dir,
                fname="trace_hist_acf_sigma.png",
                show=show,
            )

        # Signed process SDs: s_alpha, s_beta, s_gamma
        if self.s_alpha is not None:
            self._trace_hist_acf_panel(
                self.s_alpha,
                "s_alpha (±√Q_alpha)",
                save_dir=save_dir,
                fname="trace_hist_acf_s_alpha.png",
                show=show,
            )
        if self.s_beta is not None:
            self._trace_hist_acf_panel(
                self.s_beta,
                "s_beta (±√Q_beta)",
                save_dir=save_dir,
                fname="trace_hist_acf_s_beta.png",
                show=show,
            )
        if self.s_gamma is not None:
            self._trace_hist_acf_panel(
                self.s_gamma,
                "s_gamma (±√Q_gamma)",
                save_dir=save_dir,
                fname="trace_hist_acf_s_gamma.png",
                show=show,
            )

        # Q's (log10 scale) only for those coordinates that do NOT have a signed s_*
        if self.Q is not None and self.Q.size:
            Q = np.asarray(self.Q, float)
            labels = self.Q_names if self.Q_names else [f"Q[{j}]" for j in range(Q.shape[1])]
            for j in range(Q.shape[1]):
                name_j = labels[j]
                # skip FS-coords if we have signed SDs for them
                if name_j == "Q_alpha" and self.s_alpha is not None:
                    continue
                if name_j == "Q_beta" and self.s_beta is not None:
                    continue
                if name_j == "Q_gamma" and self.s_gamma is not None:
                    continue
                self._trace_hist_acf_panel(
                    np.log10(np.clip(Q[:, j], 1e-20, None)),
                    f"log10 {name_j}",
                    save_dir=save_dir,
                    fname=f"trace_hist_acf_log10Q_{j}.png",
                    show=show,
                )

        # lambdas (legacy PC priors)
        if self.lam_a is not None:
            self._trace_hist_acf_panel(
                self.lam_a,
                "λ_α",
                save_dir=save_dir,
                fname="trace_hist_acf_lambda_alpha.png",
                show=show,
            )
        if self.lam_b is not None:
            self._trace_hist_acf_panel(
                self.lam_b,
                "λ_β",
                save_dir=save_dir,
                fname="trace_hist_acf_lambda_beta.png",
                show=show,
            )
        if self.lam_g is not None:
            self._trace_hist_acf_panel(
                self.lam_g,
                "λ_γ",
                save_dir=save_dir,
                fname="trace_hist_acf_lambda_gamma.png",
                show=show,
            )

        # All other scalar parameters
        skip_keys = {
            "sigma",
            "sigma2",
            "Q_alpha",
            "Q_beta",
            "Q_gamma",
            "s_alpha",
            "s_beta",
            "s_gamma",
            "lambda_alpha",
            "lambda_beta",
            "lambda_gamma",
            "xi_alpha",
            "xi_beta",
            "xi_gamma",
            "tau",
        }
        for key, arr in sorted(self.scalar_params.items()):
            if key in skip_keys:
                continue
            self._trace_hist_acf_panel(
                arr,
                key,
                save_dir=save_dir,
                fname=f"trace_hist_acf_{key}.png",
                show=show,
            )

    def figure_states(
        self,
        save_dir: Optional[str] = None,
        fname_prefix: str = "states",
        show: bool = True,
    ):
        """
        Stacked ribbons for:
          - μ
          - α (if dynamic and in x)
          - β (if dynamic and in x)
          - γ_last (if seasonal dynamic and in x)
        """
        rows = 1
        rows += 1 if self._component_draws("alpha") is not None else 0
        rows += 1 if self._component_draws("beta") is not None else 0
        rows += 1 if self._component_draws("gamma") is not None else 0

        fig, axes = plt.subplots(rows, 1, figsize=(12, 3.0 * rows), sharex=True)
        if rows == 1:
            axes = [axes]
        r = 0
        t = np.arange(self.T)

        # μ
        mu = self.mu
        ctr, lo, hi = self._summarize_ribbon(mu)
        ax = axes[r]
        if self.y is not None and len(self.y) == self.T:
            ax.plot(self.y, lw=1.0, alpha=0.6, label="y")
        ax.plot(ctr, lw=1.6, label="μ median")
        ax.fill_between(t, lo, hi, alpha=0.25, label=self.band_label)
        if self.true_mu is not None and len(self.true_mu) == self.T:
            ax.plot(self.true_mu, lw=1.2, ls="--", label="true μ")
        ax.set_title("Posterior μ_t")
        ax.legend()
        r += 1

        # α
        A = self._component_draws("alpha")
        if A is not None:
            c, lo, hi = self._summarize_ribbon(A)
            ax = axes[r]
            ax.plot(c, lw=1.6, label="α median")
            ax.fill_between(t, lo, hi, alpha=0.25, label=self.band_label)
            if self.true_alpha is not None and len(self.true_alpha) == self.T:
                ax.plot(self.true_alpha, lw=1.2, ls="--", label="true α")
            ax.set_title("Level α_t")
            ax.legend()
            r += 1

        # β
        B = self._component_draws("beta")
        if B is not None:
            c, lo, hi = self._summarize_ribbon(B)
            ax = axes[r]
            ax.plot(c, lw=1.6, label="β median")
            ax.fill_between(t, lo, hi, alpha=0.25, label=self.band_label)
            if self.true_beta is not None and len(self.true_beta) == self.T:
                ax.plot(self.true_beta, lw=1.2, ls="--", label="true β")
            ax.set_title("Trend β_t")
            ax.legend()
            r += 1

        # γ(last)
        G = self._component_draws("gamma")
        if G is not None:
            c, lo, hi = self._summarize_ribbon(G)
            ax = axes[r]
            ax.plot(c, lw=1.6, label="γ(last) median")
            ax.fill_between(t, lo, hi, alpha=0.25, label=self.band_label)
            if self.true_gamma is not None and len(self.true_gamma) == self.T:
                ax.plot(self.true_gamma, lw=1.2, ls="--", label="true γ")
            ax.set_title("Seasonal last coordinate γ_t")
            ax.legend()

        plt.tight_layout()
        if save_dir:
            _ensure_dir(save_dir)
            out = os.path.join(save_dir, f"{fname_prefix}.png")
            fig.savefig(out, dpi=200, bbox_inches="tight")
            print(f"[save] {out}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def figure_correlations(
        self,
        save_dir: Optional[str] = None,
        show: bool = True,
        max_vars: int = 6,
    ) -> None:
        """
        Correlation plots (scatter matrices) for:

          1) Signed process SDs: s_alpha, s_beta, s_gamma (if >= 2 exist).
          2) Baselines m0_*: m0_alpha, m0_beta, m0_gamma[j] (if >= 2 exist).
          3) Joint m0_* and signed SDs (if >= 2 total).

        max_vars limits the dimensionality of each scatter matrix
        (if there are more variables, the first max_vars are used).
        """
        # --------- 1) Signed process SDs only ---------
        cols_s: List[np.ndarray] = []
        labels_s: List[str] = []
        if self.s_alpha is not None:
            cols_s.append(self.s_alpha)
            labels_s.append("s_alpha")
        if self.s_beta is not None:
            cols_s.append(self.s_beta)
            labels_s.append("s_beta")
        if self.s_gamma is not None:
            cols_s.append(self.s_gamma)
            labels_s.append("s_gamma")

        if len(cols_s) >= 2:
            data_s = np.column_stack(cols_s)
            if data_s.shape[1] > max_vars:
                print(f"[corr] s_*: limiting to first {max_vars} variables.")
                data_s = data_s[:, :max_vars]
                labels_s = labels_s[:max_vars]
            self._scatter_matrix(
                data_s,
                labels_s,
                title="Correlation: signed process SDs",
                fname="corr_s_scatter_matrix.png",
                save_dir=save_dir,
                show=show,
            )
        else:
            print("[corr] fewer than 2 signed process SDs; skipping s_* correlation plot.")

        # --------- 2) m0_* only ---------
        cols_m0: List[np.ndarray] = []
        labels_m0: List[str] = []
        if self.m0_alpha is not None:
            cols_m0.append(self.m0_alpha)
            labels_m0.append("m0_alpha")
        if self.m0_beta is not None:
            cols_m0.append(self.m0_beta)
            labels_m0.append("m0_beta")
        if self.m0_gamma is not None:
            K = self.m0_gamma.shape[1]
            for k in range(K):
                cols_m0.append(self.m0_gamma[:, k])
                labels_m0.append(f"m0_gamma[{k}]")

        if len(cols_m0) >= 2:
            data_m0 = np.column_stack(cols_m0)
            if data_m0.shape[1] > max_vars:
                print(f"[corr] m0_*: limiting to first {max_vars} variables.")
                data_m0 = data_m0[:, :max_vars]
                labels_m0 = labels_m0[:max_vars]
            self._scatter_matrix(
                data_m0,
                labels_m0,
                title="Correlation: baselines m0_*",
                fname="corr_m0_scatter_matrix.png",
                save_dir=save_dir,
                show=show,
            )
        else:
            print("[corr] fewer than 2 m0_* variables; skipping m0 correlation plot.")

        # --------- 3) Joint m0_* and signed SDs ---------
        cols_joint: List[np.ndarray] = []
        labels_joint: List[str] = []

        # m0_* first
        if self.m0_alpha is not None:
            cols_joint.append(self.m0_alpha)
            labels_joint.append("m0_alpha")
        if self.m0_beta is not None:
            cols_joint.append(self.m0_beta)
            labels_joint.append("m0_beta")
        if self.m0_gamma is not None:
            K = self.m0_gamma.shape[1]
            for k in range(K):
                cols_joint.append(self.m0_gamma[:, k])
                labels_joint.append(f"m0_gamma[{k}]")

        # then signed SDs
        if self.s_alpha is not None:
            cols_joint.append(self.s_alpha)
            labels_joint.append("s_alpha")
        if self.s_beta is not None:
            cols_joint.append(self.s_beta)
            labels_joint.append("s_beta")
        if self.s_gamma is not None:
            cols_joint.append(self.s_gamma)
            labels_joint.append("s_gamma")

        if len(cols_joint) >= 2:
            data_joint = np.column_stack(cols_joint)
            if data_joint.shape[1] > max_vars:
                print(f"[corr] m0_* + s_*: limiting to first {max_vars} variables.")
                data_joint = data_joint[:, :max_vars]
                labels_joint = labels_joint[:max_vars]
            self._scatter_matrix(
                data_joint,
                labels_joint,
                title="Correlation: m0_* and signed process SDs",
                fname="corr_m0_s_scatter_matrix.png",
                save_dir=save_dir,
                show=show,
            )
        else:
            print("[corr] fewer than 2 total variables; skipping joint m0_*–s_* correlation plot.")

    def quick_report(
        self,
        save_dir: Optional[str] = None,
        fname_prefix: str = "quick_report",
        show: bool = True,
    ):
        """
        1x3 compact panel: μ ribbon, σ posterior, and process scale:

          - If s_alpha is available: hist of s_alpha (±√Q_alpha)
          - Else: hist of log10(Q_alpha) or of first Q-column.
        """
        mu = self.mu
        ctr, lo, hi = self._summarize_ribbon(mu)
        t = np.arange(self.T)

        fig, axs = plt.subplots(1, 3, figsize=(14, 4))

        # μ
        axs[0].plot(ctr, lw=1.6, label="μ median")
        axs[0].fill_between(t, lo, hi, alpha=0.25, label=self.band_label)
        if self.true_mu is not None and len(self.true_mu) == self.T:
            axs[0].plot(self.true_mu, lw=1.2, ls="--", label="true μ")
        axs[0].set_title("μ_t")
        axs[0].legend()

        # σ
        if self.sigma is not None:
            axs[1].hist(self.sigma, bins=40, density=True)
            axs[1].set_title("σ | y")
        else:
            axs[1].axis("off")

        # Process scale: prefer s_alpha if available, otherwise log10(Q_alpha / Q[0])
        if self.s_alpha is not None:
            axs[2].hist(self.s_alpha, bins=40, density=True)
            axs[2].set_title("s_alpha (±√Q_alpha) | y")
        elif self.Q is not None and self.Q.size:
            logQ = np.log10(np.clip(self.Q[:, 0], 1e-20, None))
            label = self.Q_names[0] if self.Q_names else "Q[0]"
            axs[2].hist(logQ, bins=40, density=True)
            axs[2].set_title(f"log10 {label} | y")
        else:
            axs[2].axis("off")

        plt.tight_layout()
        if save_dir:
            _ensure_dir(save_dir)
            out = os.path.join(save_dir, f"{fname_prefix}.png")
            fig.savefig(out, dpi=200, bbox_inches="tight")
            print(f"[save] {out}")
        if show:
            plt.show()
        else:
            plt.close(fig)


# -----------------------------
# CLI: load and plot via posterior_bundle
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "DLM plotter for Gaussian structural models (FS, double-gamma, or legacy).\n"
            "Loads a posterior bundle via optimization/posterior_bundle "
            "and produces overview, scalars (trace+hist+ACF), correlation, and state plots."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Path to a run directory or directly to posterior.npz. "
             "If omitted, searches under --root.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="results/simulations/DLM",
        help="Search root when --target is omitted.",
    )
    parser.add_argument("--level", type=float, default=0.90, help="Credible band level.")
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Show figures interactively.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Directory to save figures. Default: <run>/figures",
    )
    parser.add_argument(
        "--skip-traceacf",
        action="store_true",
        help="Skip trace+hist+ACF panels for scalar parameters.",
    )
    parser.add_argument(
        "--skip-states",
        action="store_true",
        help="Skip state ribbons.",
    )
    parser.add_argument(
        "--skip-overview",
        action="store_true",
        help="Skip overview figure.",
    )
    parser.add_argument(
        "--skip-quick",
        action="store_true",
        help="Skip quick 1x3 panel.",
    )
    parser.add_argument(
        "--skip-corr",
        action="store_true",
        help="Skip correlation scatter-matrix plots (m0_*, s_* and joint).",
    )
    args = parser.parse_args()

    # Resolve run path using helper
    run_path = args.target
    if run_path is None:
        print(
            f"[info] --target not provided; searching for the latest posterior under --root={args.root!r} ..."
        )
        run_path = find_latest_run(root=args.root)
        if run_path is None:
            print(f"[error] No 'posterior.npz' found under {args.root!r}. Provide --target or change --root.")
            sys.exit(1)
        print(f"[info] Using latest run: {run_path}")

    # Load bundle
    bundle = load_posterior(run_path)  # PosteriorBundle(draws, meta, npz_path, meta_path)
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

    if not args.skip_corr:
        plotter.figure_correlations(save_dir=out_dir, show=args.show)

    if not args.skip_quick:
        plotter.quick_report(save_dir=out_dir, fname_prefix="quick_report", show=args.show)

    print("[done] plots written.")
