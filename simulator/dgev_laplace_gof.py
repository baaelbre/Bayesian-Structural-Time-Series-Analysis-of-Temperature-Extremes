# %% simulator/dgev_laplace_gof.py
from __future__ import annotations
"""
DGEV (Laplace-NCP) Goodness-of-Fit
=================================

Mirrors simulator/dlm_gof.py but for the *dynamic GEV* model fitted via the Laplace-NCP
approximate Gibbs sampler (optimization/dgev_laplace.py).

Diagnostics implemented (as in your manuscript "Model diagnostics"):

1) PITs (probability integral transforms)
   For the DGEV observation equation,
       Y_t | (mu_t, sigma, xi) ~ GEV(mu_t, sigma, xi)
   so for posterior draws (mu_t^{(m)}, sigma^{(m)}, xi^{(m)}),
       u_t^{(m)} = F_GEV(y_t ; mu_t^{(m)}, sigma^{(m)}, xi^{(m)}).

   Outputs:
   - PIT histogram with posterior median + credible band (over posterior draws)
   - PIT PP-plot (empirical PIT quantiles vs Uniform(0,1)) with posterior band

   Styling (per requirement):
   - PIT histogram: NO title, NO legend
   - PIT PP plot:   NO title, NO legend

2) Posterior predictive assessment (Gelman-style; Gaetan & Grigoletto-style discrepancy)
   Discrepancy: KS distance of PITs to Uniform(0,1), K(·).

   For each posterior draw m:
       K_obs^{(m)} = KS( {u_t^{(m)}} , U(0,1) )
       Generate PITs under the fitted model:
           u_rep^{(m)} ~ iid Uniform(0,1) of length T_eff
       K_rep^{(m)} = KS( u_rep^{(m)}, U(0,1) )

   Output:
   - scatter plot of (K_obs^{(m)}, K_rep^{(m)}) with 45° line

   Styling (per requirement):
   - KS scatter: NO title, NO legend, axis labels "K_obs" and "K_rep"

3) JSON summary
   Writes a compact JSON file with key GOF summaries (p_B and KS summaries).
   Optionally include full ks arrays with --json-full (can be large).

Robust posterior discovery mirrors simulator/dlm_plotter.py:
- optimization.posterior_bundle.load_posterior/find_latest_run
- fallback to newest posterior*.npz under --root

Expected posterior bundle content
--------------------------------
Required:
  - draws['mu']    : (S, T) posterior draws of mu_t  (location trajectory on MODEL scale)
  - draws['y']     : (T,)   observed series on MODEL scale

Needed for PIT:
  - draws['sigma'] : (S,) posterior draws of observation scale (>0)
  - draws['xi']    : (S,) posterior draws of shape
    (Optionally accept draws['sigma2'] and take sqrt.)

Notes
-----
- This GOF works on the MODEL scale stored in posterior.npz. If you modeled minima via
  z_t=-y_t, then draws['y'] should already be that transformed series, and the PITs are
  for the fitted model.

CLI
---
python -m simulator.dgev_laplace_gof --target <run_or_npz> --out <dir> --show
python -m simulator.dgev_laplace_gof --root results/... (auto-picks latest)
"""

import os
import sys
import math
import json
import argparse
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Make optimization/ visible
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Posterior loader
try:
    from optimization.posterior_bundle import load_posterior, find_latest_run  # type: ignore
except Exception as e:
    raise ImportError(
        "Could not import optimization.posterior_bundle.load_posterior/find_latest_run.\n"
        "Make sure optimization/posterior_bundle.py is on PYTHONPATH."
    ) from e

# Utilities (same module used by plotter)
try:
    from simulator.utils import _ensure_dir, find_latest_posterior_npz  # type: ignore
except Exception as e:
    raise ImportError(
        "Could not import simulator.utils.\n"
        "Make sure simulator/utils.py is on PYTHONPATH."
    ) from e


# =============================================================================
# Helpers (module-level so wrappers can reuse)
# =============================================================================
def _ks_to_uniform(u: np.ndarray) -> float:
    """One-sample Kolmogorov–Smirnov distance to U(0,1)."""
    v = np.asarray(u, float).ravel()
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n == 0:
        return float("nan")
    v = np.sort(np.clip(v, 0.0, 1.0))
    i = np.arange(1, n + 1, dtype=float)
    d_plus = float(np.max(i / n - v))
    d_minus = float(np.max(v - (i - 1.0) / n))
    return float(max(d_plus, d_minus))


def _summarize(x: np.ndarray, qs: Tuple[float, float] = (0.05, 0.95)) -> Dict[str, float]:
    v = np.asarray(x, float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {
            "n": 0.0,
            "mean": float("nan"),
            "sd": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "median": float("nan"),
            f"q{int(qs[0]*100):02d}": float("nan"),
            f"q{int(qs[1]*100):02d}": float("nan"),
        }
    ql, qh = float(qs[0]), float(qs[1])
    return {
        "n": float(v.size),
        "mean": float(np.mean(v)),
        "sd": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "median": float(np.quantile(v, 0.5)),
        f"q{int(ql*100):02d}": float(np.quantile(v, ql)),
        f"q{int(qh*100):02d}": float(np.quantile(v, qh)),
    }


def _draw_subsample_index(S: int, *, max_draws: Optional[int], rng: np.random.Generator) -> Optional[np.ndarray]:
    """Shared subsample index to keep draw alignment across arrays."""
    if max_draws is None or int(max_draws) <= 0 or int(S) <= int(max_draws):
        return None
    idx = rng.choice(int(S), size=int(max_draws), replace=False)
    idx.sort()
    return idx


def _gev_cdf(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """
    Vectorized GEV CDF F(y; mu, sigma, xi) with draw-wise (row-wise) Gumbel handling.

    Shapes:
      y     : (T_eff,) or (1,T_eff)
      mu    : (S,T_eff)
      sigma : (S,1) or (S,)
      xi    : (S,1) or (S,)

    Returns:
      F : (S,T_eff) in [0,1]
    """
    y = np.asarray(y, float)
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    xi = np.asarray(xi, float)

    # Ensure sigma is (S,1)
    if sigma.ndim == 1:
        sigma = sigma[:, None]
    sigma = np.clip(sigma, 1e-12, None)

    # Ensure xi is (S,) for row-wise decisions
    xi_vec = xi.reshape(-1)

    z = (y - mu) / sigma  # (S,T_eff) via broadcasting
    S, T = z.shape

    eps = 1e-6
    is_gumbel = (np.abs(xi_vec) < eps)  # (S,)

    F = np.empty((S, T), dtype=float)

    # --- Gumbel draws (xi ~ 0): F = exp(-exp(-z))
    if np.any(is_gumbel):
        zg = z[is_gumbel, :]
        t = np.exp(np.clip(-zg, -700.0, 700.0))
        F[is_gumbel, :] = np.exp(-t)

    # --- Non-Gumbel draws
    if np.any(~is_gumbel):
        zn = z[~is_gumbel, :]                          # (S_n,T)
        xin = xi_vec[~is_gumbel][:, None]              # (S_n,1)

        t = 1.0 + xin * zn                              # (S_n,T)
        valid = t > 0.0

        # Outside support:
        #   xi > 0 and t<=0  => y below lower bound => F=0
        #   xi < 0 and t<=0  => y above upper bound => F=1
        Finv = np.where(xin > 0.0, 0.0, 1.0)            # (S_n,1) broadcast
        Fn = np.broadcast_to(Finv, t.shape).astype(float)

        # For valid region: F = exp(-(t)^(-1/xi))
        tv = np.where(valid, t, 1.0)                    # safe filler where invalid
        logpow = (-1.0 / xin) * np.log(tv)              # (S_n,T)
        pow_ = np.exp(np.clip(logpow, -700.0, 700.0))   # (S_n,T)
        Fn = np.where(valid, np.exp(-pow_), Fn)

        F[~is_gumbel, :] = Fn

    return np.clip(F, 0.0, 1.0)

# =============================================================================
# Core computations
# =============================================================================
def compute_pit_draws(
    *, y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, xi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute posterior PIT draws u^{(m)}_t = F_GEV(y_t; mu^{(m)}_t, sigma^{(m)}, xi^{(m)}).

    Returns:
      u    : (S, T_eff)
      mask : (T,) boolean mask of finite y (T_eff = sum(mask))
    """
    y = np.asarray(y, float).ravel()
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float).ravel()
    xi = np.asarray(xi, float).ravel()

    if mu.ndim != 2:
        raise ValueError("mu must be (S,T)")
    S, T = mu.shape
    if y.size != T:
        raise ValueError(f"y has length {y.size} but mu has T={T}")
    if sigma.size != S:
        raise ValueError(f"sigma has length {sigma.size} but mu has S={S}")
    if xi.size != S:
        raise ValueError(f"xi has length {xi.size} but mu has S={S}")

    mask = np.isfinite(y)
    if not np.any(mask):
        raise ValueError("y has no finite values; cannot compute PITs")

    y_eff = y[mask][None, :]  # (1, T_eff)
    mu_eff = mu[:, mask]      # (S, T_eff)
    sig_eff = np.clip(sigma, 1e-12, None)[:, None]  # (S,1)
    xi_eff = xi[:, None]      # (S,1)

    u = _gev_cdf(y_eff, mu_eff, sig_eff, xi_eff)
    return np.clip(u, 0.0, 1.0), mask


# =============================================================================
# Class interface (mirrors DLMGoodnessOfFit)
# =============================================================================
@dataclass(slots=True)
class GOFConfig:
    level: float = 0.90
    bins: int = 20
    seed: int = 123
    max_draws: Optional[int] = None
    burn: int = 0
    thin: int = 1
    skip_pit: bool = False
    skip_ppc: bool = False
    show: bool = False
    json_name: str = "gof_results.json"
    json_full: bool = False
    color: Optional[str] = None


class DGEVLaplaceGoodnessOfFit:
    """
    Reusable goodness-of-fit runner for DGEV Laplace-NCP posterior bundles.

    Typical usage:
        gof = DGEVLaplaceGoodnessOfFit(level=0.9, bins=20, color=None)
        out = gof.run_from_target(target=None, root="results/...", out_dir=None)
    """

    def __init__(self, *, level: float = 0.90, bins: int = 20, color: Optional[str] = None) -> None:
        self.level = float(level)
        self.bins = int(bins)
        self.color = color

    # --------------------
    # discovery / loading
    # --------------------
    @staticmethod
    def resolve_bundle(target: Optional[str], root: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], str]:
        run_path = target
        if run_path is None:
            print(f"[info] --target not provided; searching for the latest posterior under --root={root!r} ...")
            run_path = find_latest_run(root=root)
            if run_path is None:
                npz_path = find_latest_posterior_npz(root)
                if npz_path is None:
                    raise FileNotFoundError(f"No posterior runs found under {root!r}. Provide --target or change --root.")
                run_path = npz_path
                print(f"[info] find_latest_run found nothing; using latest npz: {run_path}")
            else:
                print(f"[info] Using latest run: {run_path}")

        bundle = load_posterior(run_path)
        return bundle.draws, bundle.meta, bundle.npz_path

    @staticmethod
    def apply_burn_thin(
        draws: Dict[str, Any],
        meta: Dict[str, Any],
        *,
        burn: int = 0,
        thin: int = 1,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Post-hoc burn/thin applied to all draw arrays aligned on axis 0."""
        if "mu" not in draws:
            print("[warn] 'mu' not in draws; skipping burn/thin.")
            return draws, meta

        burn = int(burn or 0)
        thin = int(thin or 1)
        if burn < 0:
            raise ValueError(f"--burn must be >= 0, got {burn}")
        if thin < 1:
            raise ValueError(f"--thin must be >= 1, got {thin}")

        mu_arr = np.asarray(draws["mu"])
        if mu_arr.ndim < 2:
            print("[warn] 'mu' does not look like (n_samp, T); skipping burn/thin.")
            return draws, meta

        n_samp = int(mu_arr.shape[0])
        if burn >= n_samp:
            raise ValueError(f"--burn={burn} ≥ number of saved samples ({n_samp}).")

        idx = slice(burn, None, thin)
        n_used = int(math.ceil((n_samp - burn) / thin))
        print(f"[info] post-processing chains: raw n={n_samp}, burn={burn}, thin={thin} → used n={n_used}")

        out = dict(draws)
        for k, v in list(out.items()):
            if not isinstance(v, np.ndarray):
                continue
            arr = np.asarray(v)
            if arr.ndim >= 1 and arr.shape[0] == n_samp:
                out[k] = arr[idx, ...]

        meta2 = dict(meta)
        postproc = dict(meta2.get("postproc", {}))
        postproc.update(
            {
                "extra_burn": int(burn),
                "thin": int(thin),
                "n_samples_raw": int(n_samp),
                "n_samples_used": int(np.asarray(out["mu"]).shape[0]),
            }
        )
        meta2["postproc"] = postproc
        return out, meta2

    @staticmethod
    def extract_core_arrays(draws: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract (mu_all, y, sigma_all, xi_all) from bundle draws."""
        if "mu" not in draws:
            raise KeyError("draws must contain 'mu' of shape (S,T).")
        if "y" not in draws:
            raise KeyError("draws must contain 'y' of shape (T,).")

        mu_all = np.asarray(draws["mu"], float)
        y = np.asarray(draws["y"], float).ravel()
        if mu_all.ndim != 2:
            raise ValueError("draws['mu'] must be 2D (S,T).")
        S_all, T = mu_all.shape
        if y.size != T:
            raise ValueError(f"draws['y'] has length {y.size} but draws['mu'] has T={T}.")

        sigma_all: Optional[np.ndarray] = None
        if "sigma" in draws:
            s = np.asarray(draws["sigma"], float).ravel()
            if s.ndim == 1 and s.shape[0] == S_all:
                sigma_all = s
        if sigma_all is None and "sigma2" in draws:
            s2 = np.asarray(draws["sigma2"], float).ravel()
            if s2.ndim == 1 and s2.shape[0] == S_all:
                sigma_all = np.sqrt(np.clip(s2, 0.0, None))
        if sigma_all is None:
            raise KeyError("Need draws['sigma'] (or draws['sigma2']) aligned with draws['mu'].")

        if "xi" not in draws:
            raise KeyError("draws must contain 'xi' of shape (S,).")
        xi_all = np.asarray(draws["xi"], float).ravel()
        if xi_all.ndim != 1 or xi_all.shape[0] != S_all:
            raise ValueError("draws['xi'] must be (S,) aligned with draws['mu'].")

        return mu_all, y, sigma_all, xi_all

    @staticmethod
    def subsample_aligned(
        mu_all: np.ndarray,
        sigma_all: np.ndarray,
        xi_all: np.ndarray,
        *,
        max_draws: Optional[int],
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Aligned subsample along axis 0. Returns (mu, sigma, xi, idx_used)."""
        S_all = int(mu_all.shape[0])
        idx = _draw_subsample_index(S_all, max_draws=max_draws, rng=rng)
        if idx is None:
            return mu_all, sigma_all, xi_all, None
        return mu_all[idx, :], sigma_all[idx], xi_all[idx], idx

    # --------------------
    # plotting
    # --------------------
    def figure_pit_hist(
        self,
        u: np.ndarray,
        *,
        save_dir: Optional[str] = None,
        fname: str = "gof_pit_hist.png",
        show: bool = True,
    ) -> Dict[str, Any]:
        """PIT histogram (no title/legend). Returns small numeric summary."""
        U = np.asarray(u, float)
        if U.ndim != 2:
            raise ValueError("u must be (S,T_eff)")
        S, T = U.shape

        lev = float(self.level)
        if not (0.0 < lev < 1.0):
            raise ValueError("level must be in (0,1)")

        B = int(self.bins)
        if B < 5:
            raise ValueError("bins must be >= 5")

        idx = np.minimum((U * B).astype(int), B - 1)  # (S,T)
        dens = np.zeros((S, B), float)
        for m in range(S):
            c = np.bincount(idx[m], minlength=B).astype(float)
            dens[m] = c * (B / float(T))  # density

        lo_q = (1.0 - lev) / 2.0
        hi_q = 1.0 - lo_q

        dens_med = np.quantile(dens, 0.5, axis=0)
        dens_lo = np.quantile(dens, lo_q, axis=0)
        dens_hi = np.quantile(dens, hi_q, axis=0)

        edges = np.linspace(0.0, 1.0, B + 1)
        mids = 0.5 * (edges[:-1] + edges[1:])

        fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.2))
        if self.color:
            ax.fill_between(mids, dens_lo, dens_hi, alpha=0.25, step="mid", color=self.color)
            ax.step(mids, dens_med, where="mid", lw=2.0, color=self.color)
        else:
            ax.fill_between(mids, dens_lo, dens_hi, alpha=0.25, step="mid")
            ax.step(mids, dens_med, where="mid", lw=2.0)

        ax.axhline(1.0, lw=1.2, ls="--", color="k", alpha=0.7)

        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("PIT value")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.25)

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

        max_abs_dev = float(np.max(np.abs(dens_med - 1.0))) if dens_med.size else float("nan")
        return {"bins": int(B), "level": float(lev), "max_abs_dev_from_uniform_density_median": float(max_abs_dev)}

    def figure_pit_pp(
        self,
        u: np.ndarray,
        *,
        save_dir: Optional[str] = None,
        fname: str = "gof_pit_pp.png",
        show: bool = True,
    ) -> Dict[str, Any]:
        """PIT PP plot (no title/legend). Returns small numeric summary."""
        U = np.asarray(u, float)
        if U.ndim != 2:
            raise ValueError("u must be (S,T_eff)")
        S, T = U.shape

        lev = float(self.level)
        if not (0.0 < lev < 1.0):
            raise ValueError("level must be in (0,1)")

        lo_q = (1.0 - lev) / 2.0
        hi_q = 1.0 - lo_q

        U_sorted = np.sort(U, axis=1)  # (S,T)
        q_med = np.quantile(U_sorted, 0.5, axis=0)
        q_lo = np.quantile(U_sorted, lo_q, axis=0)
        q_hi = np.quantile(U_sorted, hi_q, axis=0)

        p = (np.arange(1, T + 1, dtype=float) - 0.5) / float(T)

        fig, ax = plt.subplots(1, 1, figsize=(6.6, 6.0))
        if self.color:
            ax.fill_between(p, q_lo, q_hi, alpha=0.25, color=self.color)
            ax.plot(p, q_med, lw=2.0, color=self.color)
        else:
            ax.fill_between(p, q_lo, q_hi, alpha=0.25)
            ax.plot(p, q_med, lw=2.0)

        ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color="k", alpha=0.7)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Uniform quantiles")
        ax.set_ylabel("Empirical PIT quantiles")
        ax.grid(True, alpha=0.25)

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

        max_abs_dev = float(np.max(np.abs(q_med - p))) if q_med.size else float("nan")
        return {"level": float(lev), "max_abs_dev_from_45deg_median": float(max_abs_dev)}

    def posterior_predictive_ks_scatter(
        self,
        *,
        u_obs: np.ndarray,
        rng: np.random.Generator,
        save_dir: Optional[str] = None,
        fname: str = "gof_ppc_ks_scatter.png",
        show: bool = True,
    ) -> Dict[str, Any]:
        """
        PPC KS scatter:
          K_obs^{(m)} = KS(u_obs^{(m)}, U(0,1))
          K_rep^{(m)} = KS(U_rep, U(0,1)) with U_rep ~ iid Uniform(0,1)

        Styling: NO title/legend, x="K_obs", y="K_rep".
        """
        U = np.asarray(u_obs, float)
        if U.ndim != 2:
            raise ValueError("u_obs must be (S,T_eff)")
        S, T_eff = U.shape

        ks_obs = np.zeros(S, float)
        ks_rep = np.zeros(S, float)

        for m in range(S):
            ks_obs[m] = _ks_to_uniform(U[m])
            u_rep = rng.random(T_eff)
            ks_rep[m] = _ks_to_uniform(u_rep)

        finite = np.isfinite(ks_obs) & np.isfinite(ks_rep)
        p_B = float(np.mean(ks_rep[finite] > ks_obs[finite])) if np.any(finite) else float("nan")

        fig, ax = plt.subplots(1, 1, figsize=(6.6, 6.0))
        if self.color:
            ax.scatter(ks_obs, ks_rep, s=14, alpha=0.6, color=self.color)
        else:
            ax.scatter(ks_obs, ks_rep, s=14, alpha=0.6)

        if np.any(finite):
            mn = float(min(np.min(ks_obs[finite]), np.min(ks_rep[finite])))
            mx = float(max(np.max(ks_obs[finite]), np.max(ks_rep[finite])))
            pad = 0.02 * (mx - mn + 1e-12)
            ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad], ls="--", color="k", alpha=0.7)

        ax.set_xlabel("K_obs")
        ax.set_ylabel("K_rep")
        ax.grid(True, alpha=0.25)

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

        return {"ks_obs": ks_obs, "ks_rep": ks_rep, "p_B": float(p_B)}

    # --------------------
    # end-to-end runner
    # --------------------
    def run_from_draws(
        self,
        *,
        draws: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
        npz_path: Optional[str] = None,
        out_dir: Optional[str] = None,
        cfg: Optional[GOFConfig] = None,
    ) -> Dict[str, Any]:
        cfg = cfg or GOFConfig(level=self.level, bins=self.bins, color=self.color)
        self.level = float(cfg.level)
        self.bins = int(cfg.bins)
        if cfg.color is not None:
            self.color = cfg.color

        rng = np.random.default_rng(int(cfg.seed))

        mu_all, y, sigma_all, xi_all = self.extract_core_arrays(draws)

        # optional subsample (aligned)
        mu, sigma, xi, idx_used = self.subsample_aligned(
            mu_all, sigma_all, xi_all, max_draws=cfg.max_draws, rng=rng
        )
        if idx_used is not None:
            print(f"[info] subsampled draws: S={mu.shape[0]} (from {mu_all.shape[0]})")

        # PIT draws
        u_obs, mask = compute_pit_draws(y=y, mu=mu, sigma=sigma, xi=xi)
        S, T_eff = u_obs.shape
        n_missing = int(np.sum(~mask))
        print(f"[info] PIT draws computed: S={S}, T_eff={T_eff} (dropped {n_missing} non-finite y)")

        # output directory
        if out_dir is None:
            if npz_path is None:
                raise ValueError("out_dir is None and npz_path is None; provide at least one.")
            out_dir = os.path.join(os.path.dirname(str(npz_path)), "gof")
        _ensure_dir(out_dir)
        print(f"[info] saving GOF figures to: {out_dir}")

        results: Dict[str, Any] = {
            "npz_path": (str(npz_path) if npz_path is not None else None),
            "out_dir": str(out_dir),
            "seed": int(cfg.seed),
            "S_total": int(mu_all.shape[0]),
            "S_used": int(S),
            "T": int(y.size),
            "T_eff": int(T_eff),
            "n_nonfinite_y": int(n_missing),
            "level": float(self.level),
            "bins": int(self.bins),
            "color": (str(self.color) if self.color is not None else None),
        }
        if meta is not None:
            results["meta"] = dict(meta)

        # PIT plots
        if cfg.skip_pit:
            results["pit"] = {"skipped": True}
        else:
            pit_hist_summary = self.figure_pit_hist(u_obs, save_dir=out_dir, show=bool(cfg.show))
            pit_pp_summary = self.figure_pit_pp(u_obs, save_dir=out_dir, show=bool(cfg.show))
            results["pit"] = {"skipped": False, "hist": pit_hist_summary, "pp": pit_pp_summary}

        # PPC
        if cfg.skip_ppc:
            results["ppc"] = {"skipped": True}
        else:
            ppc = self.posterior_predictive_ks_scatter(u_obs=u_obs, rng=rng, save_dir=out_dir, show=bool(cfg.show))
            ks_obs = np.asarray(ppc["ks_obs"], float)
            ks_rep = np.asarray(ppc["ks_rep"], float)
            results["ppc"] = {
                "skipped": False,
                "p_B": float(ppc["p_B"]),
                "ks_obs_summary": _summarize(ks_obs),
                "ks_rep_summary": _summarize(ks_rep),
            }
            if bool(cfg.json_full):
                results["ppc"]["ks_obs"] = ks_obs.tolist()
                results["ppc"]["ks_rep"] = ks_rep.tolist()

            print(f"[ppc] p_B = {results['ppc']['p_B']:.6f}")
            print(
                f"[ppc] mean(K_obs)={results['ppc']['ks_obs_summary']['mean']:.6f}, "
                f"mean(K_rep)={results['ppc']['ks_rep_summary']['mean']:.6f}"
            )

        # JSON
        json_path = os.path.join(out_dir, str(cfg.json_name))
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[save] JSON summary -> {json_path}")

        return results

    def run_from_target(
        self,
        *,
        target: Optional[str],
        root: str,
        out_dir: Optional[str] = None,
        cfg: Optional[GOFConfig] = None,
        meta_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cfg = cfg or GOFConfig(level=self.level, bins=self.bins, color=self.color)
        draws, meta, npz_path = self.resolve_bundle(target, root)

        # post-hoc burn/thin (general)
        if int(cfg.burn) > 0 or int(cfg.thin) > 1:
            draws, meta = self.apply_burn_thin(draws, meta, burn=int(cfg.burn), thin=int(cfg.thin))

        if meta_override is not None:
            meta2 = dict(meta)
            meta2.update(dict(meta_override))
            meta = meta2

        return self.run_from_draws(draws=draws, meta=meta, npz_path=npz_path, out_dir=out_dir, cfg=cfg)


# =============================================================================
# CLI
# =============================================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Goodness-of-fit diagnostics for DGEV Laplace-NCP posterior bundles (PIT + posterior predictive KS).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--target", type=str, default=None, help="Run directory or posterior.npz path.")
    p.add_argument("--root", type=str, default="results/simulations/DGEV_NCP_LASSO", help="Search root if --target omitted.")
    p.add_argument("--out", type=str, default=None, help="Directory to save figures (default: <run>/gof).")
    p.add_argument("--show", action="store_true", default=False, help="Show figures interactively.")

    p.add_argument("--level", type=float, default=0.90, help="Credible band level for PIT plots.")
    p.add_argument("--bins", type=int, default=20, help="Number of bins for PIT histogram.")

    p.add_argument("--burn", type=int, default=0, help="Extra burn-in draws (post-hoc).")
    p.add_argument("--thin", type=int, default=1, help="Extra thinning factor (post-hoc).")

    p.add_argument("--max-draws", type=int, default=None, help="Subsample posterior draws for speed (None = all).")
    p.add_argument("--seed", type=int, default=123, help="RNG seed (subsampling + PPC simulation).")

    p.add_argument("--skip-pit", action="store_true", help="Skip PIT histogram and PP plot.")
    p.add_argument("--skip-ppc", action="store_true", help="Skip posterior predictive KS scatter.")

    p.add_argument("--json-name", type=str, default="gof_results.json", help="Filename for JSON summary.")
    p.add_argument("--json-full", action="store_true", help="Also store ks_obs/ks_rep arrays in JSON (large).")

    p.add_argument("--color", type=str, default=None, help="Optional matplotlib color for PIT/KS points.")

    return p


def main() -> None:
    args = build_argparser().parse_args()

    cfg = GOFConfig(
        level=float(args.level),
        bins=int(args.bins),
        seed=int(args.seed),
        max_draws=(None if args.max_draws is None else int(args.max_draws)),
        burn=int(args.burn),
        thin=int(args.thin),
        skip_pit=bool(args.skip_pit),
        skip_ppc=bool(args.skip_ppc),
        show=bool(args.show),
        json_name=str(args.json_name),
        json_full=bool(args.json_full),
        color=(None if args.color in (None, "", "None", "none") else str(args.color)),
    )

    runner = DGEVLaplaceGoodnessOfFit(level=cfg.level, bins=cfg.bins, color=cfg.color)
    runner.run_from_target(target=args.target, root=str(args.root), out_dir=args.out, cfg=cfg)

    print("[done] GOF diagnostics written.")


if __name__ == "__main__":
    main()
